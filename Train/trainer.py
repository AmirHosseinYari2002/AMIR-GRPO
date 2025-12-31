from __future__ import annotations

import torch
import torch.nn.functional as F
from contextlib import nullcontext
from typing import List, Tuple, Optional
from trl.trainer.grpo_trainer import GRPOTrainer
from trl.trainer.dpo_trainer import DPOTrainer as _TRL_DPOTrainer
from torch.cuda.amp import autocast


class AMIR_GRPO_Trainer(GRPOTrainer):
    """GRPO trainer with a DPO-style contrastive regularizer.

    This class extends :class:`GRPOTrainer` by computing contrastive pairs of
    generations within each group and adding a DPO-style loss term. The
    relative contribution of this term is controlled by ``lambda_reg`` and
    adjusted dynamically based on the ratio between GRPO and DPO losses.

    Parameters
    ----------
    *args:
        Positional arguments forwarded to :class:`GRPOTrainer`.
    lambda_reg:
        Initial weight for the DPO-style regularization term. This value is
        dynamically updated during training to keep the regularization effect
        within a target range.
    reward_margin:
        Minimum reward difference between two generations in the same group
        for them to form a (chosen, rejected) pair.
    pair_mode:
        Strategy used to form pairs inside a group. Either ``"all"`` (all
        valid pairs respecting ranking order) or ``"topk"`` (pair everything
        against the top-scoring generation).
    max_pairs_per_group:
        Optional cap on the number of pairs per group. If set, only the pairs
        with the largest reward differences are kept.
    beta_dpo:
        Temperature parameter used in the DPO loss. Higher values sharpen the
        contrast between chosen and rejected generations.
    ref_free:
        If ``True``, use an implicit reference policy (zero log-probs). If
        ``False``, a separate reference model is used when available.
    dpo_chunk_size:
        Microbatch size used when computing sequence log-probabilities for the
        DPO term, to reduce peak memory usage.
    **kwargs:
        Additional keyword arguments forwarded to :class:`GRPOTrainer`.

    Attributes
    ----------
    lambda_reg:
        Current weight of the DPO-style regularization term.
    _dpo_cache:
        Per-step cache containing rewards and tokenized inputs required to
        compute the DPO loss in :meth:`compute_loss`.
    """

    def __init__(
        self,
        *args,
        lambda_reg: float = 0.0,
        reward_margin: float = 0.0,
        pair_mode: str = "all",  # "all" | "topk"
        max_pairs_per_group: Optional[int] = None,
        beta_dpo: float = 1.0,
        ref_free: bool = False,
        dpo_chunk_size: int = 2,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)
        self.lambda_reg = float(lambda_reg)
        self.reward_margin = float(reward_margin)
        self.pair_mode = pair_mode
        self.max_pairs_per_group = max_pairs_per_group
        self.beta_dpo = float(beta_dpo)
        self.ref_free = bool(ref_free)
        self.dpo_chunk_size = int(dpo_chunk_size)
        self._dpo_cache = None

        # TRL flag set by PEFT casting (match GRPO behavior)
        if not hasattr(self, "_peft_has_been_casted_to_bf16"):
            self._peft_has_been_casted_to_bf16 = False

    # -----------------------------------------------------------------------
    # Internal utilities
    # -----------------------------------------------------------------------

    def _amp_ctx(self):
        """Return a context manager for AMP, matching TRL's bf16 policy.

        Autocasting is only enabled if PEFT weights have been cast to
        bfloat16.

        Returns
        -------
        ctx:
            A context manager that either enables autocast on the accelerator
            device or acts as a no-op.
        """
        return (
            autocast(self.accelerator.device.type)
            if self._peft_has_been_casted_to_bf16
            else nullcontext()
        )

    def _build_contrastive_pairs(
        self, rewards: torch.Tensor, group_size: int
    ) -> List[Tuple[int, int]]:
        """Build (chosen, rejected) index pairs within each group.

        Rewards are assumed to be arranged in blocks of size ``group_size``,
        one block per prompt. Inside each group, generations are sorted by
        reward in descending order and turned into contrastive pairs according
        to the configured ``pair_mode``.

        Parameters
        ----------
        rewards:
            1D tensor of shape ``[B * G]`` containing per-completion rewards,
            where ``B`` is the number of prompts and ``G`` is the group size.
        group_size:
            Number of generations per prompt (``G``).

        Returns
        -------
        pairs:
            A list of ``(i, j)`` index pairs (global indices into ``rewards``)
            such that the reward difference ``rewards[i] - rewards[j]`` is
            greater than ``reward_margin``, and both indices belong to the
            same group.
        """
        if rewards.numel() == 0 or group_size <= 1:
            return []

        B = rewards.numel() // group_size
        pairs: List[Tuple[int, int]] = []

        for b in range(B):
            start = b * group_size
            r = rewards[start : start + group_size]
            order = torch.argsort(r, descending=True)

            # We'll first collect candidates as (diff, global_i, global_j)
            cand: List[Tuple[float, int, int]] = []

            if self.pair_mode == "topk":
                # Always pair everything with the top element, but track diffs
                top_local = order[0].item()
                top_val = r[top_local]

                for o in order[1:]:
                    j_local = o.item()
                    diff = (top_val - r[j_local]).item()
                    if diff > self.reward_margin:
                        cand.append((diff, start + top_local, start + j_local))
            else:
                # "all" pairs within group, respecting ranking order in `order`
                for ii in range(group_size):
                    i_local = order[ii].item()
                    for jj in range(ii + 1, group_size):
                        j_local = order[jj].item()
                        diff = (r[i_local] - r[j_local]).item()
                        if diff > self.reward_margin:
                            cand.append((diff, start + i_local, start + j_local))

            # If max_pairs_per_group is set, keep those with largest diff
            if (
                self.max_pairs_per_group is not None
                and len(cand) > self.max_pairs_per_group
            ):
                cand.sort(key=lambda x: x[0], reverse=True)  # sort by diff desc
                cand = cand[: self.max_pairs_per_group]

            # Drop diffs, keep only index pairs
            pairs.extend([(gi, gj) for _, gi, gj in cand])

        return pairs

    @staticmethod
    def _make_labels(
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        prompt_lens: torch.Tensor,
        label_pad: int = -100,
    ) -> torch.Tensor:
        """Create DPO-style labels that mask prompt and padding tokens.

        Prompt tokens (before ``prompt_lens``) and positions where
        ``attention_mask == 0`` are set to ``label_pad``. Completion tokens
        retain their original IDs.

        Parameters
        ----------
        input_ids:
            Tensor of shape ``[N, T]`` with token IDs for prompt + completion.
        attention_mask:
            Tensor of shape ``[N, T]`` with 1 for valid tokens and 0 for pad.
        prompt_lens:
            1D tensor of shape ``[N]`` with the prompt length (in tokens) for
            each sequence.
        label_pad:
            Special label value used to mask out tokens (default: ``-100``).

        Returns
        -------
        labels:
            Tensor of shape ``[N, T]`` where prompt and pad tokens are set to
            ``label_pad`` and completion tokens contain their original IDs.
        """
        N, T = input_ids.shape
        labels = input_ids.clone()
        arange = torch.arange(T, device=input_ids.device).unsqueeze(0)
        is_prompt = arange < prompt_lens.unsqueeze(1)
        labels[is_prompt] = label_pad
        labels[attention_mask == 0] = label_pad
        return labels

    @staticmethod
    def _fallback_get_batch_logps(
        logits: torch.Tensor, labels: torch.Tensor
    ) -> torch.Tensor:
        """Compute summed token log-probabilities for each sequence.

        This serves as a local fallback for TRL's
        ``DPOTrainer._get_batch_logps``. It sums log-probabilities over all
        positions where ``labels != -100`` (no length normalization).

        Parameters
        ----------
        logits:
            Tensor of shape ``[N, T, V]`` with token logits, where ``V`` is
            the vocabulary size.
        labels:
            Tensor of shape ``[N, T]`` with target token IDs and ``-100`` at
            ignored positions.

        Returns
        -------
        logps:
            1D tensor of shape ``[N]`` with summed token log-probabilities for
            each sequence.
        """
        # logits: [N,T,V]; labels: [N,T] with -100 masked
        n, t, v = logits.shape
        # shift to next-token prediction like standard LM loss
        logits = logits[:, :-1, :]  # [N,T-1,V]
        labels = labels[:, 1:]  # [N,T-1]
        logp = logits.log_softmax(-1)
        # gather per-token logp
        gather_mask = labels.ne(-100)
        safe_labels = labels.masked_fill(~gather_mask, 0)
        tok_logps = logp.gather(-1, safe_labels.unsqueeze(-1)).squeeze(-1)
        tok_logps = tok_logps * gather_mask.to(tok_logps.dtype)
        return tok_logps.sum(dim=1)

    def _get_dpo_logps(
        self, logits: torch.Tensor, labels: torch.Tensor
    ) -> torch.Tensor:
        """Wrapper to obtain per-sequence log-prob sums for DPO.

        Parameters
        ----------
        logits:
            Tensor of shape ``[N, T, V]`` with token logits.
        labels:
            Tensor of shape ``[N, T]`` with target labels and ``-100`` at
            ignored positions.

        Returns
        -------
        logps:
            1D tensor of shape ``[N]`` containing the sum of token
            log-probabilities over all non-masked positions.
        """
        return self._fallback_get_batch_logps(logits, labels)

    def _compute_seq_logps_microbatch(
        self,
        model,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        labels: torch.Tensor,
        chunk_size: int,
        requires_grad: bool = True,
    ) -> torch.Tensor:
        """Compute length-normalized sequence log-probabilities in chunks.

        This helper runs the model forward in small microbatches to reduce
        peak memory usage. For each sequence, we sum token log-probabilities
        over completion tokens (where ``labels != -100``) and divide by the
        completion length.

        Parameters
        ----------
        model:
            The policy (or reference) model used to compute logits.
        input_ids:
            Tensor of shape ``[N, T]`` with token IDs for prompt + completion.
        attention_mask:
            Tensor of shape ``[N, T]`` with 1 for valid tokens and 0 for pad.
        labels:
            Tensor of shape ``[N, T]`` with labels and ``-100`` for ignored
            positions.
        chunk_size:
            Number of sequences per microbatch for the forward pass.
        requires_grad:
            If ``True``, gradients are tracked; otherwise, the forward pass
            is wrapped in ``torch.no_grad``.

        Returns
        -------
        seq_logps:
            1D tensor of shape ``[N]`` containing length-normalized sequence
            log-probabilities.
        """
        device = input_ids.device
        N = input_ids.size(0)
        seq_logps = torch.empty(N, device=device, dtype=torch.float32)

        # choose grad / no-grad context
        outer_ctx = nullcontext if requires_grad else torch.no_grad

        with outer_ctx():
            for start in range(0, N, chunk_size):
                end = min(start + chunk_size, N)

                ids_chunk = input_ids[start:end]
                mask_chunk = attention_mask[start:end]
                labels_chunk = labels[start:end]

                with self.accelerator.autocast():
                    out_chunk = model(input_ids=ids_chunk, attention_mask=mask_chunk)

                logps_chunk = self._get_dpo_logps(
                    out_chunk.logits, labels_chunk
                )  # [B_chunk]
                comp_lens_chunk = (
                    (labels_chunk != -100).sum(dim=1).clamp_min(1)
                )  # [B_chunk]
                logps_chunk = logps_chunk / comp_lens_chunk  # length-normalized

                seq_logps[start:end] = logps_chunk

                # free chunk-local tensors
                del out_chunk, logps_chunk, comp_lens_chunk
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()

        return seq_logps

    # -----------------------------------------------------------------------
    # GRPO hook: cache extra fields for DPO-style regularization
    # -----------------------------------------------------------------------

    def _generate_and_score_completions(self, inputs):
        """Run GRPO generation & scoring, then cache minimal DPO context.

        This method extends the parent implementation by caching:

        - Per-completion rewards (for pair mining).
        - Concatenated ``input_ids`` and ``attention_mask`` for
          prompt + completion.
        - Prompt lengths (one per completion).
        - Group size (number of generations per prompt).

        Parameters
        ----------
        inputs:
            Batch inputs as expected by :class:`GRPOTrainer`.

        Returns
        -------
        output:
            The dictionary returned by
            ``super()._generate_and_score_completions``, unchanged.
        """
        base_out = super()._generate_and_score_completions(inputs)

        device = self.accelerator.device

        prompt_ids = base_out["prompt_ids"]  # [N, P]
        prompt_mask = base_out["prompt_mask"]  # [N, P]
        completion_ids = base_out["completion_ids"]  # [N, C]
        completion_mask = base_out["completion_mask"]  # [N, C]

        prompt_completion_ids = torch.cat([prompt_ids, completion_ids], dim=1)  # [N, T]
        attention_mask = torch.cat([prompt_mask, completion_mask], dim=1).to(
            device
        )  # [N, T]
        prompt_lens = prompt_mask.sum(dim=1)  # [N]

        # Recompute rewards (mirrors parent reward flow)
        prompt_ids_cpu = prompt_ids.detach().cpu()
        completion_ids_cpu = completion_ids.detach().cpu()
        prompts_text = self.processing_class.batch_decode(
            prompt_ids_cpu, skip_special_tokens=True
        )
        completions_text = self.processing_class.batch_decode(
            completion_ids_cpu, skip_special_tokens=True
        )
        prompts = [x["prompt"] for x in inputs]

        # Conversational case: attach generated text to last assistant message.
        if isinstance(prompts[0], list):
            completions = []
            for prompt, completion in zip(prompts, completions_text, strict=True):
                last_msg = (
                    prompt[-1]
                    if (isinstance(prompt, list) and len(prompt) > 0)
                    else None
                )
                bootstrap = ""
                if last_msg and last_msg.get("role") == "assistant":
                    bootstrap = last_msg.get("content", "")
                    if isinstance(bootstrap, list):
                        if len(bootstrap) == 1 and bootstrap[0].get("type") == "text":
                            bootstrap = bootstrap[0].get("text", "")
                        else:
                            bootstrap = "".join(
                                part.get("text", "")
                                for part in bootstrap
                                if part.get("type") == "text"
                            )
                completions.append(
                    [{"role": "assistant", "content": bootstrap + completion}]
                )
        else:
            completions = completions_text

        rewards_per_func = self._calculate_rewards(
            inputs,
            prompts,
            completions,
            [ids.tolist() for ids in completion_ids],
        )
        rewards = (
            rewards_per_func * self.reward_weights.to(device).unsqueeze(0)
        ).nansum(
            dim=1
        )  # [N]

        # Cache minimal DPO context on CPU (no gradients).
        self._dpo_cache = {
            "rewards": rewards.detach().cpu(),
            "prompt_completion_ids": prompt_completion_ids.detach().cpu(),
            "attention_mask": attention_mask.detach().cpu(),
            "prompt_lens": prompt_lens.detach().cpu(),
            "group_size": int(self.num_generations),
        }

        # free temps
        del (
            rewards,
            rewards_per_func,
            prompts_text,
            completions_text,
            prompts,
            completions,
        )
        del prompt_completion_ids, attention_mask, prompt_lens
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        return base_out

    # -----------------------------------------------------------------------
    # GRPO hook: add DPO-style loss on top of GRPO loss
    # -----------------------------------------------------------------------

    def compute_loss(self, model, inputs, return_outputs=False, **kwargs):
        """Compute GRPO loss and add a DPO-style regularization term.

        The method calls the parent :meth:`compute_loss` to obtain the GRPO
        loss, then, if DPO-style regularization is enabled and cached context
        is available, mines contrastive pairs and computes a DPO-style loss.
        The final loss is:

            ``mixed_loss = grpo_loss + lambda_reg * dpo_loss``

        Parameters
        ----------
        model:
            The policy model being trained.
        inputs:
            Batch inputs as provided by the trainer.
        return_outputs:
            Kept for API compatibility; this implementation always returns
            a scalar loss tensor.
        **kwargs:
            Additional keyword arguments forwarded to the parent
            :meth:`compute_loss`.

        Returns
        -------
        loss:
            A scalar loss tensor combining GRPO and DPO-style terms.
        """
        # Vanilla GRPO loss (scalar)
        grpo_loss = super().compute_loss(model, inputs, return_outputs=False, **kwargs)

        # If regularization is disabled or cache missing → GRPO only.
        if self.lambda_reg == 0.0 or self._dpo_cache is None:
            return grpo_loss

        ctx = self._dpo_cache
        dev = model.device
        rewards: torch.Tensor = ctx["rewards"].to(dev, non_blocking=True)
        input_ids_all: torch.Tensor = ctx["prompt_completion_ids"].to(
            dev, non_blocking=True
        )
        attention_mask_all: torch.Tensor = ctx["attention_mask"].to(
            dev, non_blocking=True
        )
        prompt_lens_all: torch.Tensor = ctx["prompt_lens"].to(dev, non_blocking=True)
        G: int = int(ctx["group_size"])

        # Mine contrastive pairs
        pairs = self._build_contrastive_pairs(rewards, G)
        if not pairs:
            # no pairs → just GRPO
            self._dpo_cache = None
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            return grpo_loss

        # build index tensors
        i_idx = torch.tensor([i for (i, _) in pairs], device=dev, dtype=torch.long)
        j_idx = torch.tensor([j for (_, j) in pairs], device=dev, dtype=torch.long)

        # Slice chosen / rejected sequences.
        ids_chosen = input_ids_all[i_idx]
        ids_reject = input_ids_all[j_idx]
        mask_chosen = attention_mask_all[i_idx]
        mask_reject = attention_mask_all[j_idx]
        prompt_lens = prompt_lens_all[i_idx]

        # Labels: mask prompts & pads; keep completions.
        labels_chosen = self._make_labels(ids_chosen, mask_chosen, prompt_lens)
        labels_reject = self._make_labels(ids_reject, mask_reject, prompt_lens)

        # Stack (chosen, rejected).
        input_ids = torch.cat([ids_chosen, ids_reject], dim=0)  # [2K, T]
        attention_mask = torch.cat([mask_chosen, mask_reject], dim=0)  # [2K, T]
        labels = torch.cat([labels_chosen, labels_reject], dim=0)  # [2K, T]

        K = ids_chosen.size(0)

        # Policy log-probs via microbatched forward
        seq_logps = self._compute_seq_logps_microbatch(
            model,
            input_ids=input_ids,
            attention_mask=attention_mask,
            labels=labels,
            chunk_size=self.dpo_chunk_size,
            requires_grad=True,
        )  # [2K]

        chosen_logps = seq_logps[:K]
        reject_logps = seq_logps[K:]

        # Optional reference model (if not ref-free).
        if not self.ref_free and getattr(self, "ref_model", None) is not None:
            ref_dev = next(self.ref_model.parameters()).device
            input_ids_ref = input_ids.to(ref_dev, non_blocking=True)
            attention_mask_ref = attention_mask.to(ref_dev, non_blocking=True)
            labels_ref = labels.to(ref_dev, non_blocking=True)

            ref_seq_logps = self._compute_seq_logps_microbatch(
                self.ref_model,
                input_ids=input_ids_ref,
                attention_mask=attention_mask_ref,
                labels=labels_ref,
                chunk_size=self.dpo_chunk_size,
                requires_grad=False,
            )  # [2K] on ref_dev

            if ref_dev != dev:
                ref_seq_logps = ref_seq_logps.to(dev, non_blocking=True)

            ref_chosen_logps = ref_seq_logps[:K]
            ref_reject_logps = ref_seq_logps[K:]
        else:
            # implicit reference: zeros
            ref_chosen_logps = torch.zeros_like(chosen_logps)
            ref_reject_logps = torch.zeros_like(reject_logps)

        # DPO-style loss (standard sigmoid form)
        beta = self.beta_dpo
        logits = beta * (
            (chosen_logps - ref_chosen_logps) - (reject_logps - ref_reject_logps)
        )
        dpo_loss = -F.logsigmoid(logits).mean()

        # Dynamic lambda_reg tuning (ratio-based)
        grpo = float(grpo_loss.detach())
        dpo_val = float(dpo_loss.detach())
        r = (self.lambda_reg * dpo_val) / max(1e-9, grpo)

        target_low, target_high = 0.30, 0.70
        up, down = 1.25, 0.8
        lam_min, lam_max = 0.005, 0.05

        new_lambda = self.lambda_reg
        if r < target_low:
            new_lambda = min(lam_max, self.lambda_reg * up)
        elif r > target_high:
            new_lambda = max(lam_min, self.lambda_reg * down)

        self.lambda_reg = new_lambda

        # Optional metrics logging.
        if hasattr(self, "_metrics"):
            mode = "train" if self.model.training else "eval"
            self._metrics[mode]["loss/dpo"].append(dpo_loss.detach().item())
            self._metrics[mode]["pairs/num_pairs"].append(len(pairs))
            gaps = (rewards[i_idx] - rewards[j_idx]).detach()
            self._metrics[mode]["pairs/mean_gap"].append(gaps.mean().item())

        print(
            f"grpo_loss: {grpo_loss}  -  self.lambda_reg * dpo_loss: {self.lambda_reg * dpo_loss}"
        )
        mixed = grpo_loss + self.lambda_reg * dpo_loss

        # free VRAM
        del rewards, input_ids_all, attention_mask_all, prompt_lens_all
        del (
            ids_chosen,
            ids_reject,
            mask_chosen,
            mask_reject,
            labels_chosen,
            labels_reject,
        )
        del (
            input_ids,
            attention_mask,
            labels,
            seq_logps,
            chosen_logps,
            reject_logps,
            logits,
            dpo_loss,
        )
        if "ref_seq_logps" in locals():
            del ref_seq_logps, ref_chosen_logps, ref_reject_logps
        self._dpo_cache = None
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        return mixed
