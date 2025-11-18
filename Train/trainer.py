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
         pairs from reward-ranked generations and compute a DPO loss using TRL DPO utilities
         (or a local fallback), and finally mix both losses.

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
        **kwargs,
    ):
        super().__init__(*args, **kwargs)
        self.lambda_pair = float(lambda_pair)
        self.pair_threshold = float(pair_threshold)
        self.pair_mining = pair_mining
        self.max_pairs_per_group = max_pairs_per_group
        self.beta_dpo = float(beta_dpo)
        self.implicit_ref = bool(implicit_ref)
        self._dpo_cache = None  # set per step

        # TRL flag set by PEFT casting (match GRPO behavior)
        if not hasattr(self, "_peft_has_been_casted_to_bf16"):
            self._peft_has_been_casted_to_bf16 = False

    # ----------------------- utils -----------------------

    def _amp_ctx(self):
        # Match TRL: only autocast if PEFT was casted to bf16
        return autocast(self.accelerator.device.type) if self._peft_has_been_casted_to_bf16 else nullcontext()

    def _build_contrastive_pairs(self, rewards: torch.Tensor, group_size: int) -> List[Tuple[int, int]]:
        """
        rewards: [B*G] tensor aligned with completions
        group_size: G
        returns list of (i, j) global indices such that r_i - r_j > threshold and i ranks above j within each group.
        """
        if rewards.numel() == 0 or group_size <= 1:
            return []
        B = rewards.numel() // group_size
        pairs: List[Tuple[int, int]] = []
        for b in range(B):
            start = b * group_size
            r = rewards[start:start + group_size]
            order = torch.argsort(r, descending=True)
            if self.pair_mining == "topk":
                top = order[0].item()
                cand: List[Tuple[int, int]] = []
                for o in order[1:]:
                    o_i = o.item()
                    if (r[top] - r[o_i]) > self.pair_threshold:
                        cand.append((start + top, start + o_i))
                if self.max_pairs_per_group is not None:
                    cand = cand[: self.max_pairs_per_group]
                pairs.extend(cand)
            else:
                # "all"
                cand = []
                for ii in range(group_size):
                    for jj in range(ii + 1, group_size):
                        i = order[ii].item()
                        j = order[jj].item()
                        if (r[i] - r[j]) > self.pair_threshold:
                            cand.append((start + i, start + j))
                if self.max_pairs_per_group is not None:
                    cand = cand[: self.max_pairs_per_group]
                pairs.extend(cand)
        return pairs

    @staticmethod
    def _make_labels(input_ids: torch.Tensor, attention_mask: torch.Tensor, prompt_lens: torch.Tensor, label_pad: int = -100) -> torch.Tensor:
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
        Use TRL's private/public helper if available; otherwise fallback.
        """
        if _TRL_DPOTrainer is not None:
            get_fn = getattr(_TRL_DPOTrainer, "_get_batch_logps", None) or getattr(_TRL_DPOTrainer, "get_batch_logps", None)
            if get_fn is not None:
                # TRL returns (seq_logps, token_logps?) in some versions; handle both
                out = get_fn(logits, labels)
                if isinstance(out, (tuple, list)):
                    return out[0]
                return out
        # Fallback
        return self._fallback_get_batch_logps(logits, labels)

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

        # ---- one forward pass over stacked (chosen,rejected) ----
        input_ids = torch.cat([ids_chosen, ids_reject], dim=0)           # [2K, T]
        attention_mask = torch.cat([mask_chosen, mask_reject], dim=0)    # [2K, T]
        labels = torch.cat([labels_chosen, labels_reject], dim=0)        # [2K, T]

        with self._amp_ctx():
            out = model(input_ids=input_ids, attention_mask=attention_mask)

        # log-prob sums over completion tokens (policy)
        seq_logps = self._get_dpo_logps(out.logits, labels)              # [2K]
        K = ids_chosen.size(0)
        chosen_logps = seq_logps[:K]
        reject_logps = seq_logps[K:]

        # optional reference
        if not self.implicit_ref and getattr(self, "ref_model", None) is not None:
            with torch.no_grad():
                ref_out = self.ref_model(input_ids=input_ids, attention_mask=attention_mask)
                ref_seq_logps = self._get_dpo_logps(ref_out.logits, labels)
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
        del input_ids, attention_mask, labels, out, seq_logps, chosen_logps, reject_logps, logits, dpo_loss
        if 'ref_seq_logps' in locals():
            del ref_seq_logps, ref_chosen_logps, ref_reject_logps
        self._dpo_cache = None
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        return mixed