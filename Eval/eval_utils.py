from __future__ import annotations

import math
from typing import List, Optional, Sequence, Tuple, Union, Dict

import torch
from tqdm import tqdm
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    PreTrainedTokenizerBase,
    PreTrainedModel,
)
from peft import PeftModel

from Data.math_grader import answer_tag_reward_fn_for_orz


# ------------------------------
# Device / dtype helpers
# ------------------------------


def _get_device(preferred: Optional[torch.device] = None) -> torch.device:
    """
    Get a usable torch device.

    Parameters
    ----------
    preferred : torch.device or None, optional
        If provided, return this device as-is. If None, select CUDA if available,
        otherwise CPU.

    Returns
    -------
    torch.device
        The selected device.
    """
    if preferred is not None:
        return preferred
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def _choose_dtype() -> torch.dtype:
    """
    Choose a sensible default dtype for inference.

    The preference order is:
    1) bfloat16 on CUDA when bf16 is supported
    2) float16 on CUDA otherwise
    3) float32 on CPU

    Returns
    -------
    torch.dtype
        The chosen dtype.
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
    Estimate pass@k using the unbiased estimator.

    Parameters
    ----------
    correct_counts : list of int
        `correct_counts[i]` is the number of correct samples for problem `i`.
    n_samples : int
        Total number of generated samples per problem (assumed constant).
    ks : sequence of int
        Values of `k` for which to compute pass@k.

    Returns
    -------
    dict of int to float
        Mapping from k -> estimated pass@k (in [0, 1]).
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
    Load a (possibly PEFT-adapted) CausalLM and tokenizer from a directory.

    This function:
    1) Loads a base causal language model from `directory_path`.
    2) Attaches PEFT adapters from the same `directory_path`.
    3) Loads the tokenizer from `directory_path`.
    4) Sets `model.eval()`.

    Parameters
    ----------
    directory_path : str
        Local path or Hugging Face repo ID containing the model + (optionally)
        adapter weights.
    hf_token : str
        Hugging Face access token for private repositories, gated models, etc.
    device : torch.device or None, optional
        Preferred device for inference. If None, selects CUDA if available else CPU.
    load_in_4bit : bool, optional
        Whether to load the base model in 4-bit quantized mode (bitsandbytes).
    dtype : torch.dtype or None, optional
        Torch dtype to use when loading weights. If None, a default is selected via
        `_choose_dtype()`.

    Returns
    -------
    model : transformers.PreTrainedModel
        The loaded model with PEFT adapters attached.
    tokenizer : transformers.PreTrainedTokenizerBase
        The corresponding tokenizer.
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
    tokenizer: PreTrainedTokenizerBase, prompt: Union[str, Sequence[dict]]
) -> str:
    """
    Convert a prompt into a single text string suitable for tokenization.

    If the tokenizer provides `apply_chat_template` and the prompt is a list of
    chat messages, apply the chat template to produce the full formatted text.

    Parameters
    ----------
    tokenizer : transformers.PreTrainedTokenizerBase
        Tokenizer used for formatting and tokenization.
    prompt : str or sequence of dict
        Either a plain string prompt or a chat-style message list.

    Returns
    -------
    str
        The formatted input text.
    """
    if hasattr(tokenizer, "apply_chat_template") and not isinstance(prompt, str):
        # Expecting a list of chat messages (OpenAI-style dicts)
        return tokenizer.apply_chat_template(
            prompt, tokenize=False, add_generation_prompt=True
        )
    elif hasattr(tokenizer, "apply_chat_template") and isinstance(prompt, str):
        # Some tokenizers can still accept raw strings; keep your original behavior
        return tokenizer.apply_chat_template(
            prompt, tokenize=False, add_generation_prompt=True
        )
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
    Generate completions for a batch of prompts.

    Parameters
    ----------
    prompts : list of (str or sequence of dict)
        Input prompts. Each prompt can be either a raw string or a chat message list.
    tokenizer : transformers.PreTrainedTokenizerBase
        Tokenizer used for tokenization and decoding.
    model : transformers.PreTrainedModel
        Causal LM used for generation.
    device : torch.device or None, optional
        Device to run generation on. If None, selects CUDA if available else CPU.
    max_new_tokens : int, optional
        Maximum number of new tokens to generate per sequence.
    do_sample : bool, optional
        Whether to sample (stochastic decoding). If False, uses greedy decoding.
    temperature : float, optional
        Sampling temperature (only relevant if `do_sample=True`).
    top_p : float, optional
        Nucleus sampling probability threshold (only relevant if `do_sample=True`).
    num_return_sequences : int, optional
        Number of sequences to return *per prompt*. The output list will have
        length `len(prompts) * num_return_sequences`.

    Returns
    -------
    list of str
        Decoded generated texts, ordered by prompt then by sequence index.
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

    # Free encoder on GPU
    del enc

    # Move outputs back to CPU ASAP
    outputs = outputs.to("cpu")

    # Free GPU memory
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    # Decode on CPU
    decoded = tokenizer.batch_decode(outputs, skip_special_tokens=True)

    # Drop output tensor now that we're done with it
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
    Evaluate a model on a dataset using pass@k.

    For each problem in `dataset`:
      1) Generate `num_samples` completions.
      2) Evaluate each completion using `answer_tag_reward_fn_for_orz`.
      3) Record how many completions are correct for that problem (`c_i`).

    After processing the dataset, estimate pass@k for each k in `ks` using the estimator.

    Parameters
    ----------
    model : transformers.PreTrainedModel
        Model to evaluate.
    tokenizer : transformers.PreTrainedTokenizerBase
        Tokenizer corresponding to the model.
    dataset : sequence of mapping
        Dataset where each item provides:
          - item["prompt"]: str or chat message list
          - item["answer"]: ground-truth answer used by the reward function
    batch_size : int, optional
        Number of problems to process per generation batch.
    max_samples : int or None, optional
        If provided, stop after this many dataset items.
    device : torch.device or None, optional
        Device for evaluation. If None, selects CUDA if available else CPU.
    progress : bool, optional
        Whether to show a tqdm progress bar.
    num_samples : int, optional
        Number of generated samples per problem (n in the Codex formula).
    ks : sequence of int, optional
        Values of k for pass@k estimation.

    Returns
    -------
    dict of int to float
        Mapping from each k in `ks` to estimated pass@k (in [0, 1]).
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
            or (
                len(buf_prompts)
                and (i + 1 == len(dataset) if hasattr(dataset, "__len__") else False)
            )
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

        # Free CPU-side big lists for this batch
        buf_prompts.clear()
        buf_gts.clear()
        del outputs

        # Clear GPU cache between batches
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
