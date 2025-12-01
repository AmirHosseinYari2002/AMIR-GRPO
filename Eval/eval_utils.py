from __future__ import annotations

import math
from typing import List, Optional, Sequence, Tuple, Union, Dict

import torch
from tqdm import tqdm
from transformers import AutoTokenizer, AutoModelForCausalLM, PreTrainedTokenizerBase, PreTrainedModel
from peft import PeftModel

from Data.math_grader import answer_tag_reward_fn_for_orz


# ------------------------------
# Device / dtype helpers
# ------------------------------

def _get_device(preferred: Optional[torch.device] = None) -> torch.device:
    """
    Return a usable torch.device.
    """
    if preferred is not None:
        return preferred
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def _choose_dtype() -> torch.dtype:
    """
    Prefer bfloat16 on capable GPUs, else float16 on CUDA, else float32 on CPU.
    """
    if torch.cuda.is_available():
        # bf16 if the hardware supports it; otherwise fp16
        if getattr(torch.cuda, "is_bf16_supported", lambda: False)():
            return torch.bfloat16
        return torch.float16
    return torch.float32

# ------------------------------
# Codex-style pass@k helper
# ------------------------------

def _estimate_pass_at_k(
    correct_counts: List[int],
    n_samples: int,
    ks: Sequence[int],
) -> Dict[int, float]:
    """
    Codex-style pass@k estimator.

    correct_counts[i] = number of correct samples for problem i
    n_samples         = total samples per problem (assumed constant)
    ks                = iterable of k values (e.g. [1, 5, 10])

    Returns: dict mapping k -> pass@k
    """
    pass_at_k: Dict[int, float] = {}

    num_problems = len(correct_counts)
    if num_problems == 0:
        return {k: 0.0 for k in ks}

    for k in ks:
        if k > n_samples:
            # Not defined (can't choose k unique programs out of < k samples)
            pass_at_k[k] = 0.0
            continue

        # Average over problems:
        # pass@k = (1 / |D|) * sum_i [ 1 - C(n - c_i, k) / C(n, k) ]
        total = 0.0
        denom = math.comb(n_samples, k)
        for c in correct_counts:
            if c == 0:
                # No correct samples for this problem → contributes 0
                continue
            if c >= n_samples:
                # All samples correct → contributes 1
                total += 1.0
                continue

            num = math.comb(n_samples - c, k)
            total += 1.0 - (num / denom)

        pass_at_k[k] = total / num_problems

    return pass_at_k


# ------------------------------
# Model / tokenizer loading
# ------------------------------

def load_model_and_tokenizer(
    directory_path: str,
    hf_token: str,
    *,
    device: Optional[torch.device] = None,
    load_in_4bit: bool = False,
    dtype: Optional[torch.dtype] = None,
) -> Tuple[PreTrainedModel, PreTrainedTokenizerBase]:
    """
    Load a (possibly PEFT-adapted) CausalLM and tokenizer from `directory_path`.
    """
    device = _get_device(device)
    dtype = dtype or _choose_dtype()

    model = AutoModelForCausalLM.from_pretrained(
        directory_path,
        load_in_4bit=load_in_4bit,
        dtype=dtype, 
        device_map=device,
        token=hf_token,
    )

    # Attach adapters
    model = PeftModel.from_pretrained(model, directory_path, token=hf_token)
    tokenizer = AutoTokenizer.from_pretrained(directory_path, token=hf_token)

    # If use_cache exists, leave it True for eval/generation.
    if hasattr(model.config, "use_cache"):
        model.config.use_cache = True

    model.eval()
    return model, tokenizer


# ------------------------------
# Generation
# ------------------------------

def _apply_chat_template_if_available(
    tokenizer: PreTrainedTokenizerBase,
    prompt: Union[str, Sequence[dict]]
) -> str:
    """
    If the tokenizer provides a chat template and `prompt` is a list of messages,
    apply it; otherwise return `prompt` as a plain string.
    """
    if hasattr(tokenizer, "apply_chat_template") and not isinstance(prompt, str):
        # Expecting a list of chat messages (OpenAI-style dicts)
        return tokenizer.apply_chat_template(prompt, tokenize=False, add_generation_prompt=True)
    elif hasattr(tokenizer, "apply_chat_template") and isinstance(prompt, str):
        # Some tokenizers can still accept raw strings; keep your original behavior
        return tokenizer.apply_chat_template(prompt, tokenize=False, add_generation_prompt=True)
    else:
        return str(prompt)


def generate_batch(
    prompts: List[Union[str, Sequence[dict]]],
    tokenizer: PreTrainedTokenizerBase,
    model: PreTrainedModel,
    *,
    device: Optional[torch.device] = None,
    max_new_tokens: int = 1024,
    do_sample: bool = True,
    temperature: float = 0.6,
    top_p: float = 1.0,
    num_return_sequences: int = 2,
) -> List[str]:
    """
    Generate num_return_sequences per prompt.
    This function aggressively frees CUDA memory after use.
    """
    device = _get_device(device)

    # Apply chat template on CPU
    input_texts = [_apply_chat_template_if_available(tokenizer, p) for p in prompts]

    # Tokenize on CPU
    enc = tokenizer(
        input_texts,
        return_tensors="pt",
        padding=True,
        truncation=True,
    )

    # Move to GPU only here
    enc = {k: v.to(device) for k, v in enc.items()}

    with torch.inference_mode():
        outputs = model.generate(
            **enc,
            max_new_tokens=max_new_tokens,
            do_sample=do_sample,
            temperature=temperature,
            top_p=top_p,
            num_return_sequences=num_return_sequences,
        )

    # 🔹 Free encoder on GPU
    del enc

    # 🔹 Move outputs back to CPU ASAP
    outputs = outputs.to("cpu")

    # 🔹 Free GPU memory
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    # Decode on CPU
    decoded = tokenizer.batch_decode(outputs, skip_special_tokens=True)

    # 🔹 Drop output tensor now that we're done with it
    del outputs

    return decoded


# ------------------------------
# Batched evaluation
# ------------------------------

def evaluate_model_batched(
    model: PreTrainedModel,
    tokenizer: PreTrainedTokenizerBase,
    dataset,
    *,
    batch_size: int = 1,
    max_samples: Optional[int] = None,
    device: Optional[torch.device] = None,
    progress: bool = True,
    # Total samples per problem (n in the Codex formula)
    num_samples: int = 8,
    # k values for which to compute pass@k
    ks: Sequence[int] = (1, 2, 4),
) -> Dict[int, float]:
    """
    Evaluate on a test dataset using Codex-style pass@k.

    For each problem:
      - Generate `num_samples` completions.
      - Count how many are correct (c_i).
      - After the whole dataset is processed, compute pass@k
        for each k in `ks` using the Codex estimator.

    Returns:
      A dict mapping each k in `ks` to its estimated pass@k.
    """

    device = _get_device(device)
    if next(model.parameters()).device.type != device.type:
        model.to(device)
    model.eval()

    buf_prompts: List[Union[str, Sequence[dict]]] = []
    buf_gts: List[Union[str, float, int, list]] = []

    # Stores number of correct samples per problem
    per_problem_correct_counts: List[int] = []

    iterator = enumerate(dataset)
    if progress:
        try:
            n = len(dataset)
        except TypeError:
            n = None
        iterator = enumerate(tqdm(dataset, total=n))

    for i, sample in iterator:
        if max_samples is not None and i >= max_samples:
            break

        buf_prompts.append(sample["prompt"])
        buf_gts.append(sample["answer"])

        flush = (
            len(buf_prompts) == batch_size
            or (max_samples is not None and i + 1 == max_samples)
            or (len(buf_prompts) and (i + 1 == len(dataset) if hasattr(dataset, "__len__") else False))
        )
        if not flush:
            continue

        # Generate N samples per prompt in this batch
        outputs = generate_batch(
            prompts=buf_prompts,
            tokenizer=tokenizer,
            model=model,
            device=device,
            num_return_sequences=num_samples,
            do_sample=True,
            temperature=0.6,
        )

        num_prompts = len(buf_prompts)

        # For each prompt in the batch, count how many of its samples are correct
        for k_idx in range(num_prompts):
            gt = buf_gts[k_idx]

            start_index = k_idx * num_samples
            end_index = start_index + num_samples
            ensemble_outputs = outputs[start_index:end_index]

            correct_count = 0
            for pred_text in ensemble_outputs:
                _, reward = answer_tag_reward_fn_for_orz(pred_text, gt, fast=False)
                if reward == 1.0:
                    correct_count += 1

            per_problem_correct_counts.append(correct_count)

        # 🔹 Free CPU-side big lists for this batch
        buf_prompts.clear()
        buf_gts.clear()
        del outputs

        # 🔹 Clear GPU cache between batches
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    # Compute Codex-style pass@k
    pass_at_k = _estimate_pass_at_k(
        correct_counts=per_problem_correct_counts,
        n_samples=num_samples,
        ks=ks,
    )

    # Pretty print
    num_problems = len(per_problem_correct_counts)
    print(f"Evaluated on {num_problems} problems with {num_samples} samples each.")
    for k in ks:
        print(f"pass@{k}: {pass_at_k[k] * 100:.2f}%")

    return pass_at_k