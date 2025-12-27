from __future__ import annotations

import torch
import torch.nn.functional as F
from contextlib import nullcontext
from typing import List, Tuple, Optional
from trl.trainer.grpo_trainer import GRPOTrainer
from trl.trainer.dpo_trainer import DPOTrainer as _TRL_DPOTrainer
from torch.cuda.amp import autocast


class GRPOWithDPOTrainer(GRPOTrainer):
    """
    GRPO + DPO: add a DPO-style contrastive term on top of the vanilla GRPO loss.

    Flow:
      1) override _generate_and_score_completions to cache minimal extra fields
         for DPO (rewards, prompt+completion input_ids/attention_mask, prompt_lens, group_size).
      2) override compute_loss to get GRPO loss via super(), then build (chosen,rejected)
         pairs from reward-ranked generations and compute a DPO loss, and finally mix both losses.

    Returns the same scalar loss as GRPOTrainer.compute_loss (no signature changes).
    """

    def __init__(
        self,
        *args,
        lambda_pair: float = 0.0,
        pair_threshold: float = 0.0,
        pair_mining: str = "all",           # "all" | "topk"
        max_pairs_per_group: Optional[int] = None,
        beta_dpo: float = 1.0,
        implicit_ref: bool = False,
        dpo_chunk_size: int = 2,           # NEW: microbatch size for DPO forward
        **kwargs,
    ):
        super().__init__(*args, **kwargs)
        self.lambda_pair = float(lambda_pair)
        self.pair_threshold = float(pair_threshold)
        self.pair_mining = pair_mining
        self.max_pairs_per_group = max_pairs_per_group
        self.beta_dpo = float(beta_dpo)
        self.implicit_ref = bool(implicit_ref)
        self.dpo_chunk_size = int(dpo_chunk_size)
        self._dpo_cache = None  # set per step

        # TRL flag set by PEFT casting (match GRPO behavior)
        if not hasattr(self, "_peft_has_been_casted_to_bf16"):
            self._peft_has_been_casted_to_bf16 = False

    # ----------------------- utils -----------------------

    def _amp_ctx(self):
        # Match TRL: only autocast if PEFT was casted to bf16
        return (
            autocast(self.accelerator.device.type)
            if self._peft_has_been_casted_to_bf16
            else nullcontext()
        )

    def _build_contrastive_pairs(self, rewards: torch.Tensor, group_size: int) -> List[Tuple[int, int]]:
        """
        rewards: [B*G] tensor aligned with completions
        group_size: G
        returns list of (i, j) global indices such that r_i - r_j > threshold and i ranks above j within each group.
        If max_pairs_per_group is set, keep the pairs with the largest reward differences within each group.
        """
        if rewards.numel() == 0 or group_size <= 1:
            return []

        B = rewards.numel() // group_size
        pairs: List[Tuple[int, int]] = []

        for b in range(B):
            start = b * group_size
            r = rewards[start:start + group_size]
            order = torch.argsort(r, descending=True)

            # We'll first collect candidates as (diff, global_i, global_j)
            cand: List[Tuple[float, int, int]] = []

            if self.pair_mining == "topk":
                # Always pair everything with the top element, but track diffs
                top_local = order[0].item()
                top_val = r[top_local]

                for o in order[1:]:
                    j_local = o.item()
                    diff = (top_val - r[j_local]).item()
                    if diff > self.pair_threshold:
                        cand.append((diff, start + top_local, start + j_local))
            else:
                # "all" pairs within group, respecting ranking order in `order`
                for ii in range(group_size):
                    i_local = order[ii].item()
                    for jj in range(ii + 1, group_size):
                        j_local = order[jj].item()
                        diff = (r[i_local] - r[j_local]).item()
                        if diff > self.pair_threshold:
                            cand.append((diff, start + i_local, start + j_local))

            # If max_pairs_per_group is set, keep those with largest diff
            if self.max_pairs_per_group is not None and len(cand) > self.max_pairs_per_group:
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
        """
        Create DPO-style labels: mask all prompt tokens and padding, keep completion tokens.
        Shapes: [N, T]; prompt_lens [N]
        """
        N, T = input_ids.shape
        labels = input_ids.clone()
        arange = torch.arange(T, device=input_ids.device).unsqueeze(0)
        is_prompt = arange < prompt_lens.unsqueeze(1)
        labels[is_prompt] = label_pad
        labels[attention_mask == 0] = label_pad
        return labels

    @staticmethod
    def _fallback_get_batch_logps(logits: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
        """
        Fallback for TRL's DPOTrainer._get_batch_logps:
        Returns a [N] tensor with SUM of token log-probs over positions where labels != -100.
        """
        # logits: [N,T,V]; labels: [N,T] with -100 masked
        n, t, v = logits.shape
        # shift to next-token prediction like standard LM loss
        logits = logits[:, :-1, :]               # [N,T-1,V]
        labels = labels[:, 1:]                   # [N,T-1]
        logp = logits.log_softmax(-1)
        # gather per-token logp
        gather_mask = labels.ne(-100)
        safe_labels = labels.masked_fill(~gather_mask, 0)
        tok_logps = logp.gather(-1, safe_labels.unsqueeze(-1)).squeeze(-1)
        tok_logps = tok_logps * gather_mask.to(tok_logps.dtype)
        return tok_logps.sum(dim=1)              # sum over tokens (NOT length-avg)

    def _get_dpo_logps(self, logits: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
        """
        Returns [N] log-prob sums over non-masked tokens.
        Uses a local implementation for stability across TRL versions.
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
        """
        Run model forward in small chunks and return length-normalized
        sequence log-probs [N].

        This reduces peak memory compared to one big forward.
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

                logps_chunk = self._get_dpo_logps(out_chunk.logits, labels_chunk)  # [B_chunk]
                comp_lens_chunk = (labels_chunk != -100).sum(dim=1).clamp_min(1)   # [B_chunk]
                logps_chunk = logps_chunk / comp_lens_chunk                        # length-normalized

                seq_logps[start:end] = logps_chunk

                # free chunk-local tensors
                del out_chunk, logps_chunk, comp_lens_chunk
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()

        return seq_logps

    # ----------------------- GRPO hook: cache extra fields -----------------------

    def _generate_and_score_completions(self, inputs):
        """
        Run the parent pipeline and additionally cache minimal DPO inputs we need:
          - per-completion rewards (for pair mining)
          - concatenated input_ids and attention_mask (prompt + completion)
          - prompt lengths (per item)
          - group size (num_generations)
        """
        base_out = super()._generate_and_score_completions(inputs)

        device = self.accelerator.device

        prompt_ids = base_out["prompt_ids"]           # [N, P]
        prompt_mask = base_out["prompt_mask"]         # [N, P]
        completion_ids = base_out["completion_ids"]   # [N, C]
        completion_mask = base_out["completion_mask"] # [N, C]

        prompt_completion_ids = torch.cat([prompt_ids, completion_ids], dim=1)          # [N, T]
        attention_mask = torch.cat([prompt_mask, completion_mask], dim=1).to(device)    # [N, T]
        prompt_lens = prompt_mask.sum(dim=1)                                            # [N]

        # ---- recompute rewards (mirrors parent reward flow) ----
        prompt_ids_cpu = prompt_ids.detach().cpu()
        completion_ids_cpu = completion_ids.detach().cpu()
        prompts_text = self.processing_class.batch_decode(prompt_ids_cpu, skip_special_tokens=True)
        completions_text = self.processing_class.batch_decode(completion_ids_cpu, skip_special_tokens=True)
        prompts = [x["prompt"] for x in inputs]

        # conversational case
        if isinstance(prompts[0], list):
            completions = []
            for prompt, completion in zip(prompts, completions_text, strict=True):
                last_msg = prompt[-1] if (isinstance(prompt, list) and len(prompt) > 0) else None
                bootstrap = ""
                if last_msg and last_msg.get("role") == "assistant":
                    bootstrap = last_msg.get("content", "")
                    if isinstance(bootstrap, list):
                        if len(bootstrap) == 1 and bootstrap[0].get("type") == "text":
                            bootstrap = bootstrap[0].get("text", "")
                        else:
                            bootstrap = "".join(
                                part.get("text", "") for part in bootstrap if part.get("type") == "text"
                            )
                completions.append([{"role": "assistant", "content": bootstrap + completion}])
        else:
            completions = completions_text

        rewards_per_func = self._calculate_rewards(
            inputs,
            prompts,
            completions,
            [ids.tolist() for ids in completion_ids],
        )
        rewards = (rewards_per_func * self.reward_weights.to(device).unsqueeze(0)).nansum(dim=1)  # [N]

        # ---- cache minimal DPO context on CPU (no grads) ----
        self._dpo_cache = {
            "rewards": rewards.detach().cpu(),
            "prompt_completion_ids": prompt_completion_ids.detach().cpu(),
            "attention_mask": attention_mask.detach().cpu(),
            "prompt_lens": prompt_lens.detach().cpu(),
            "group_size": int(self.num_generations),
        }

        # free temps
        del rewards, rewards_per_func, prompts_text, completions_text, prompts, completions
        del prompt_completion_ids, attention_mask, prompt_lens
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        return base_out

    # ----------------------- GRPO hook: add DPO loss -----------------------

    def compute_loss(self, model, inputs, return_outputs=False, **kwargs):
        # Vanilla GRPO loss (scalar)
        grpo_loss = super().compute_loss(model, inputs, return_outputs=False, **kwargs)

        # If DPO disabled or cache missing, return GRPO only
        if self.lambda_pair == 0.0 or self._dpo_cache is None:
            return grpo_loss

        ctx = self._dpo_cache
        dev = model.device
        rewards: torch.Tensor = ctx["rewards"].to(dev, non_blocking=True)
        input_ids_all: torch.Tensor = ctx["prompt_completion_ids"].to(dev, non_blocking=True)
        attention_mask_all: torch.Tensor = ctx["attention_mask"].to(dev, non_blocking=True)
        prompt_lens_all: torch.Tensor = ctx["prompt_lens"].to(dev, non_blocking=True)
        G: int = int(ctx["group_size"])

        # ---- mine pairs ----
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

        # slice chosen / rejected (same prompts; different completions)
        ids_chosen = input_ids_all[i_idx]
        ids_reject = input_ids_all[j_idx]
        mask_chosen = attention_mask_all[i_idx]
        mask_reject = attention_mask_all[j_idx]
        prompt_lens = prompt_lens_all[i_idx]  # per pair, prompt length matches the chosen sample's prompt

        # labels: mask prompts & pads; keep completions
        labels_chosen = self._make_labels(ids_chosen, mask_chosen, prompt_lens)
        labels_reject = self._make_labels(ids_reject, mask_reject, prompt_lens)

        # ---- stack (chosen,rejected) ----
        input_ids = torch.cat([ids_chosen, ids_reject], dim=0)           # [2K, T]
        attention_mask = torch.cat([mask_chosen, mask_reject], dim=0)    # [2K, T]
        labels = torch.cat([labels_chosen, labels_reject], dim=0)        # [2K, T]

        K = ids_chosen.size(0)

        # ---- policy log-probs via microbatched forward ----
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

        # optional reference (also microbatched, but no grad)
        if not self.implicit_ref and getattr(self, "ref_model", None) is not None:
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

        # ---- DPO loss (standard sigmoid form) ----
        beta = self.beta_dpo
        logits = beta * ((chosen_logps - ref_chosen_logps) - (reject_logps - ref_reject_logps))
        dpo_loss = -F.logsigmoid(logits).mean()

        # ------- dynamic lambda_pair tuning (unchanged logic) -------
        grpo = float(grpo_loss.detach())
        dpo_val = float(dpo_loss.detach())
        r = (self.lambda_pair * dpo_val) / max(1e-9, grpo)

        target_low, target_high = 0.30, 0.70
        up, down = 1.25, 0.8
        lam_min, lam_max = 0.005, 0.05

        new_lambda = self.lambda_pair
        if r < target_low:
            new_lambda = min(lam_max, self.lambda_pair * up)
        elif r > target_high:
            new_lambda = max(lam_min, self.lambda_pair * down)

        self.lambda_pair = new_lambda

        # metrics (optional)
        if hasattr(self, "_metrics"):
            mode = "train" if self.model.training else "eval"
            self._metrics[mode]["loss/dpo"].append(dpo_loss.detach().item())
            self._metrics[mode]["pairs/num_pairs"].append(len(pairs))
            gaps = (rewards[i_idx] - rewards[j_idx]).detach()
            self._metrics[mode]["pairs/mean_gap"].append(gaps.mean().item())

        print(f"grpo_loss: {grpo_loss}  -  self.lambda_pair * dpo_loss: {self.lambda_pair * dpo_loss}")
        mixed = grpo_loss + self.lambda_pair * dpo_loss

        # free VRAM
        del rewards, input_ids_all, attention_mask_all, prompt_lens_all
        del ids_chosen, ids_reject, mask_chosen, mask_reject, labels_chosen, labels_reject
        del input_ids, attention_mask, labels, seq_logps, chosen_logps, reject_logps, logits, dpo_loss
        if "ref_seq_logps" in locals():
            del ref_seq_logps, ref_chosen_logps, ref_reject_logps
        self._dpo_cache = None
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        return mixed