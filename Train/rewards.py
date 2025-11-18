from __future__ import annotations

import math
import operator
import re
from typing import List, Optional, Sequence

from Data.data import extract_numeric_answer, extract_xml_answer

# Public API
__all__ = [
    "correctness_reward_func",
    "int_reward_func",
    "strict_format_reward_func",
    "strict_format_reward_func_with_calib",
    "xmlcount_reward_func",
    "expression_correctness_reward_func",
    "extract_confidence",
    "brier_score",
]


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

# Support LaTeX boxed answers, e.g., \boxed{42} or \boxed(42)
_BOXED_PATTERNS = (
    re.compile(r"\\boxed\{\s*([^}]*)\s*\}"),
    re.compile(r"\\boxed\(\s*([^)]*)\s*\)"),
)

# XML format patterns
_STRICT_XML_PATTERN = re.compile(
    r"^<think>\s*[\s\S]*?\s*</think>\s*<answer>\s*[\s\S]*?\s*</answer>\s*$",
    re.DOTALL | re.MULTILINE,
)

_STRICT_XML_WITH_CALIB_PATTERN = re.compile(
    r"^"
    r"<think>\s*[\s\S]*?\s*</think>\s*"
    r"<answer>\s*[\s\S]*?\s*</answer>\s*"
    r"<analysis>\s*[\s\S]*?\s*</analysis>\s*"
    r"<confidence>\s*(?:0(?:\.\d+)?|1(?:\.0+)?)\s*</confidence>\s*$",
    re.DOTALL | re.MULTILINE,
)

# Expression like:  "12.5 + 3.5 = 16.0"  (single binary op)
_EXPR_CLAIM_PATTERN = re.compile(
    r"(\d+(?:\.\d+)?\s*[+\-*/]\s*\d+(?:\.\d+)?)\s*=\s*(\d+(?:\.\d+)?)"
)

_CONFIDENCE_PATTERN = re.compile(r"<confidence>\s*([0-9]*\.?[0-9]+)\s*</confidence>")


def _parse_float_maybe(text: str) -> Optional[float]:
    """Parse a float from text if possible, handling commas and whitespace."""
    try:
        return float(text.replace(",", "").strip())
    except Exception:
        return None


def _safe_compute(expr: str) -> Optional[float]:
    """Safely compute simple expressions of the form ``a OP b``.

    Supported operators: ``+``, ``-``, ``*``, ``/``.
    Returns ``None`` on failure or division by zero.
    """
    # Normalize spaces
    expr = expr.strip()
    # Find operator (first occurrence among the supported set)
    for op_char, op_fn in ("+", operator.add), ("-", operator.sub), ("*", operator.mul), ("/", operator.truediv):
        if op_char in expr:
            left, right = expr.split(op_char, 1)
            a = _parse_float_maybe(left)
            b = _parse_float_maybe(right)
            if a is None or b is None:
                return None
            try:
                if op_char == "/" and b == 0:
                    return None
                return op_fn(a, b)
            except Exception:
                return None
    return None


# ---------------------------------------------------------------------------
# Reward functions
# ---------------------------------------------------------------------------

def correctness_reward_func(
    completions: Sequence[Sequence[dict]],
    answer: Sequence[object],
    tol: float = 1e-3,
    **kwargs,
) -> List[float]:
    """Binary correctness reward (2.0 correct / 0.0 incorrect).

    Strategy:
    1) Extract text from completions.
    2) Try to capture a LaTeX ``\\boxed{...}`` answer if present; else use raw text.
    3) Convert to numeric via project helper ``extract_numeric_answer``.
    4) Compare with ground truth numerically within tolerance; otherwise fall
       back to case-insensitive string equality.
    """
    responses = [completion[0]["content"] for completion in completions]
    numeric_preds = [extract_numeric_answer(p) for p in responses]

    rewards: List[float] = []
    for pred, gt_str in zip(numeric_preds, answer):
        # Ground-truth normalization
        gt_f = _parse_float_maybe(str(gt_str))

        is_correct = False
        if pred is not None and gt_f is not None:
            is_correct = math.isclose(pred, gt_f, rel_tol=tol, abs_tol=tol)
        else:
            is_correct = str(pred).strip().lower() == str(gt_str).strip().lower()

        rewards.append(2.0 if is_correct else 0.0)

    return rewards


def int_reward_func(completions: Sequence[Sequence[dict]], **kwargs) -> List[float]:
    """Small shaping reward (+0.2) if an integer-like answer is present.

    Uses the XML extractor then numeric parser; returns 0.0 if missing.
    """
    responses = [completion[0]["content"] for completion in completions]
    extracted_responses = [extract_xml_answer(r) for r in responses]
    numeric_preds = [extract_numeric_answer(p) for p in extracted_responses]
    return [0.2 if r is not None else 0.0 for r in numeric_preds]


def strict_format_reward_func(completions: Sequence[Sequence[dict]], **kwargs) -> List[float]:
    """Reward (+0.5) if completion matches ``<think>...</think><answer>...</answer>``."""
    responses = [completion[0]["content"] for completion in completions]
    return [0.5 if _STRICT_XML_PATTERN.match(r.strip()) else 0.0 for r in responses]


def strict_format_reward_func_with_calib(
    completions: Sequence[Sequence[dict]], **kwargs
) -> List[float]:
    """Reward (+0.5) for matching the full structure:

    ``<think>...</think><answer>...</answer><analysis>...</analysis><confidence>p</confidence>``

    where ``p`` is a float in ``[0, 1]``.
    """
    responses = [completion[0]["content"] for completion in completions]
    return [0.5 if _STRICT_XML_WITH_CALIB_PATTERN.match(r.strip()) else 0.0 for r in responses]


def _xml_tag_count_score(text: str) -> float:
    """Heuristic: reward presence of each required tag once; penalize trailing junk.

    +0.025 for each of the six tags (opening/closing for think, answer, analysis,
    confidence). If text continues after ``</confidence>``, apply a small penalty
    proportional to trailing length.
    """
    score = 0.0
    # Tag counts
    if text.count("<think>") == 1:
        score += 0.025
    if text.count("</think>") == 1:
        score += 0.025
    if text.count("<answer>") == 1:
        score += 0.025
    if text.count("</answer>") == 1:
        score += 0.025
    if text.count("<analysis>") == 1:
        score += 0.025
    if text.count("</analysis>") == 1:
        score += 0.025
    if text.count("<confidence>") == 1:
        score += 0.025
    if text.count("</confidence>") == 1:
        score += 0.025
        trailing = text.split("</confidence>")[-1].strip()
        if trailing:
            score -= min(len(trailing) * 0.001, 0.1)
    return score


def xmlcount_reward_func(completions: Sequence[Sequence[dict]], **kwargs) -> List[float]:
    """Shaping reward based on presence of required XML tags."""
    contents = [completion[0]["content"] for completion in completions]
    return [_xml_tag_count_score(c) for c in contents]


def expression_correctness_reward_func(
    completions: Sequence[Sequence[dict]], **kwargs
) -> List[float]:
    """Reward +0.1 per expression where computed value matches the claimed result.

    Recognizes claims like: ``"48 / 2 = 24"`` or ``"12.5 + 3.5 = 16.0"``.
    Only *single* binary operators are supported.
    """
    rewards: List[float] = []
    for completion in completions:
        text = completion[0]["content"]
        matches = _EXPR_CLAIM_PATTERN.findall(text)
        reward = 0.0
        for expr, claimed in matches:
            computed = _safe_compute(expr)
            try:
                claimed_val = float(claimed)
            except Exception:
                claimed_val = None
            if computed is not None and claimed_val is not None:
                if abs(computed - claimed_val) < 1e-6:
                    reward += 0.1
        rewards.append(reward)
    return rewards


def extract_confidence(text: str) -> float:
    """Extract ``<confidence>p</confidence>`` where ``p`` in [0, 1]. Defaults to 0.5."""
    m = _CONFIDENCE_PATTERN.search(text)
    if not m:
        return 0.5
    try:
        conf = float(m.group(1))
    except Exception:
        return 0.5
    return max(0.0, min(conf, 1.0))


def brier_score(
    completions: Sequence[Sequence[dict]],
    answer: Sequence[object],
    **kwargs,
) -> List[float]:
    """Brier-style score in [0, 1], aligned with correctness and stated confidence.

    We first compute binary correctness (2.0 correct, 0.0 incorrect) and map it
    to ``a in {0, 1}``. For each completion with stated confidence ``p``, we
    return ``1 - (p - a)^2``—higher is better, peaking when the confidence
    matches correctness.
    """
    correctness_rewards = correctness_reward_func(completions, answer, **kwargs)
    responses = [completion[0]["content"] for completion in completions]

    out: List[float] = []
    for resp, corr_reward in zip(responses, correctness_rewards):
        p = extract_confidence(resp)
        a = 1.0 if corr_reward == 2.0 else 0.0
        out.append(1.0 - (p - a) ** 2)
    return out
