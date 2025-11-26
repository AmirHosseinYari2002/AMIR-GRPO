from __future__ import annotations

import math
from typing import List, Optional, Sequence, Tuple, Union

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

    # If you're on CUDA, letting HF/accelerate shard automatically is usually safest.
    # On CPU, we keep defaults.
    device_map = "auto" if device.type == "cuda" else None

    model = AutoModelForCausalLM.from_pretrained(
        directory_path,
        load_in_4bit=load_in_4bit,
        dtype=dtype, 
        device_map=device_map,
        token=hf_token,
    )

    # Attach adapters
    model = PeftModel.from_pretrained(model, directory_path, token=hf_token)
    tokenizer = AutoTokenizer.from_pretrained(directory_path, token=hf_token)

    # If use_cache exists, leave it True for eval/generation.
    if hasattr(model.config, "use_cache"):
        model.config.use_cache = True

    # If we didn't device_map=auto, ensure model is on the chosen device.
    if device_map is None:
        model.to(device)

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
    do_sample: bool = False,
    temperature: float = 1.0,
    top_p: float = 1.0,
) -> List[str]:
    """
    Deterministic (greedy) by default.
    """
    device = _get_device(device)

    input_texts = [_apply_chat_template_if_available(tokenizer, p) for p in prompts]

    enc = tokenizer(
        input_texts,
        return_tensors="pt",
        padding=True,
        truncation=True,
    ).to(device)

    with torch.no_grad():
        outputs = model.generate(
            **enc,
            max_new_tokens=max_new_tokens,
            do_sample=do_sample,
            temperature=temperature,
            top_p=top_p,
        )

    # Return full decoded sequences (prompt + completion) to keep behavior consistent
    decoded = tokenizer.batch_decode(outputs, skip_special_tokens=True)
    return decoded


# ------------------------------
# Batched evaluation
# ------------------------------

def evaluate_model_batched(
    model: PreTrainedModel,
    tokenizer: PreTrainedTokenizerBase,
    dataset,
    *,
    batch_size: int = 8,
    max_samples: Optional[int] = None,
    device: Optional[torch.device] = None,
    progress: bool = True,
) -> Tuple[float, int, int]:
    """
    Evaluate on a test dataset using answer_tag_reward_fn_for_orz.
    """
    device = _get_device(device)
    if next(model.parameters()).device.type != device.type:
        model.to(device)
    model.eval()

    correct = 0
    total = 0
    buf_prompts: List[Union[str, Sequence[dict]]] = []
    buf_gts: List[Union[str, float, int, list]] = []

    iterator = enumerate(dataset)
    if progress:
        try:
            n = len(dataset)
        except TypeError:
            n = None  # unknown length
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

        # Generate a batch of responses
        outputs = generate_batch(
            prompts=buf_prompts,
            tokenizer=tokenizer,
            model=model,
            device=device,
        )

        # Grade each response with the provided reward fn
        for pred_text, gt in zip(outputs, buf_gts):
            _, reward = answer_tag_reward_fn_for_orz(pred_text, gt, fast=False)
            is_correct = (reward == 1.0)

            if is_correct:
                correct += 1
            total += 1

            if total % 50 == 0:
                acc_now = correct / total * 100
                print(f"[Step {total}] Current Accuracy: {acc_now:.2f}% ({correct}/{total})")

        buf_prompts.clear()
        buf_gts.clear()

    acc = correct / total * 100 if total else 0.0
    print(f"✅ Accuracy: {acc:.2f}% ({correct}/{total})")
    return acc, correct, total
