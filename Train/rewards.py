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
# Patterns and helpers
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
    """Parse a float from text if possible, handling commas and whitespace.

    Parameters
    ----------
    text:
        Input string that may contain a float representation.

    Returns
    -------
    value:
        Parsed float value if successful, otherwise ``None``.
    """
    try:
        return float(text.replace(",", "").strip())
    except Exception:
        return None


def _safe_compute(expr: str) -> Optional[float]:
    """Safely compute simple expressions of the form ``a OP b``.

    Supported operators are ``+``, ``-``, ``*``, and ``/``. Any parsing error
    or division by zero will cause this function to return ``None``.

    Parameters
    ----------
    expr:
        A string containing a simple binary arithmetic expression.

    Returns
    -------
    value:
        The computed float value if successful, otherwise ``None``.
    """
    # Normalize spaces
    expr = expr.strip()
    # Find operator (first occurrence among the supported set)
    for op_char, op_fn in (
        ("+", operator.add),
        ("-", operator.sub),
        ("*", operator.mul),
        ("/", operator.truediv),
    ):
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


def _xml_tag_count_score(text: str) -> float:
    """Compute a shaping score based on presence of required XML tags.

    This heuristic rewards correct usage of the four XML tag pairs
    `<think>`, `<answer>`, `<analysis>`, and `<confidence>`. Each correctly
    used opening/closing tag pair contributes a small positive amount. If
    there is trailing content after the final `</confidence>` tag, a small
    penalty proportional to the trailing length is applied.

    Parameters
    ----------
    text:
        The full model completion to inspect.

    Returns
    -------
    score:
        A scalar score reflecting the XML tag usage quality.
    """
    score = 0.0
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


# ---------------------------------------------------------------------------
# Reward functions (public API)
# ---------------------------------------------------------------------------


def correctness_reward_func(
    completions: Sequence[Sequence[dict]],
    answer: Sequence[object],
    tol: float = 1e-3,
    **kwargs,
) -> List[float]:
    """Binary correctness reward (2.0 correct / 0.0 incorrect).

    This reward treats each completion as either correct or incorrect. It
    extracts a numeric prediction when possible, compares it against the
    ground-truth answer with a tolerance, and falls back to string-based
    comparison when numeric parsing fails.

    Parameters
    ----------
    completions:
        Batched model completions, each a sequence of message dictionaries.
        Only the first message's ``"content"`` field is used.
    answer:
        Sequence of ground-truth answers (numeric or string-like) aligned
        with the completions.
    tol:
        Absolute and relative tolerance used for numeric comparisons.
    **kwargs:
        Additional keyword arguments accepted for API compatibility but
        ignored by this function.

    Returns
    -------
    rewards:
        A list of scalar rewards (2.0 for correct, 0.0 for incorrect),
        one per completion.
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
    """Shaping reward for the presence of an integer-like answer.

    This function extracts the answer from an XML-formatted completion,
    parses it numerically, and returns a small positive reward if a numeric
    value is successfully obtained.

    Parameters
    ----------
    completions:
        Batched model completions, each a sequence of message dictionaries.
        Only the first message's ``"content"`` field is used.
    **kwargs:
        Additional keyword arguments accepted for API compatibility but
        ignored by this function.

    Returns
    -------
    rewards:
        A list of scalar rewards (+0.2 if an integer-like value is present,
        0.0 otherwise), one per completion.
    """
    responses = [completion[0]["content"] for completion in completions]
    extracted_responses = [extract_xml_answer(r) for r in responses]
    numeric_preds = [extract_numeric_answer(p) for p in extracted_responses]
    return [0.2 if r is not None else 0.0 for r in numeric_preds]


def strict_format_reward_func(
    completions: Sequence[Sequence[dict]], **kwargs
) -> List[float]:
    """Reward strict `<think><answer>` XML formatting.

    A completion is rewarded if it exactly matches the pattern:

        ``<think>...</think><answer>...</answer>``

    with no extra content before or after these tags.

    Parameters
    ----------
    completions:
        Batched model completions, each a sequence of message dictionaries.
        Only the first message's ``"content"`` field is used.
    **kwargs:
        Additional keyword arguments accepted for API compatibility but
        ignored by this function.

    Returns
    -------
    rewards:
        A list of scalar rewards (+0.5 if the strict format is matched,
        0.0 otherwise), one per completion.
    """
    responses = [completion[0]["content"] for completion in completions]
    return [0.5 if _STRICT_XML_PATTERN.match(r.strip()) else 0.0 for r in responses]


def strict_format_reward_func_with_calib(
    completions: Sequence[Sequence[dict]], **kwargs
) -> List[float]:
    """Reward strict calibrated XML formatting.

    A completion is rewarded if it exactly matches the pattern:

        ``<think>...</think><answer>...</answer><analysis>...</analysis><confidence>p</confidence>``

    where ``p`` is a float in the interval ``[0, 1]``.

    Parameters
    ----------
    completions:
        Batched model completions, each a sequence of message dictionaries.
        Only the first message's ``"content"`` field is used.
    **kwargs:
        Additional keyword arguments accepted for API compatibility but
        ignored by this function.

    Returns
    -------
    rewards:
        A list of scalar rewards (+0.5 if the strict calibrated format is
        matched, 0.0 otherwise), one per completion.
    """
    responses = [completion[0]["content"] for completion in completions]
    return [
        0.5 if _STRICT_XML_WITH_CALIB_PATTERN.match(r.strip()) else 0.0
        for r in responses
    ]


def xmlcount_reward_func(
    completions: Sequence[Sequence[dict]], **kwargs
) -> List[float]:
    """Shaping reward based on usage of required XML tags.

    This function applies a heuristic shaping score that encourages correct
    usage of `<think>`, `<answer>`, `<analysis>`, and `<confidence>` tags,
    and gently penalizes trailing content after the final closing tag.

    Parameters
    ----------
    completions:
        Batched model completions, each a sequence of message dictionaries.
        Only the first message's ``"content"`` field is used.
    **kwargs:
        Additional keyword arguments accepted for API compatibility but
        ignored by this function.

    Returns
    -------
    rewards:
        A list of scalar shaping rewards, one per completion.
    """
    contents = [completion[0]["content"] for completion in completions]
    return [_xml_tag_count_score(c) for c in contents]


def expression_correctness_reward_func(
    completions: Sequence[Sequence[dict]], **kwargs
) -> List[float]:
    """Reward correctness of simple arithmetic claims in text.

    The function scans each completion for patterns of the form
    ``"a OP b = c"`` where a single binary operator (`+`, `-`, `*`, `/`)
    is used. Each correctly evaluated expression contributes a small
    positive reward.

    Parameters
    ----------
    completions:
        Batched model completions, each a sequence of message dictionaries.
        Only the first message's ``"content"`` field is used.
    **kwargs:
        Additional keyword arguments accepted for API compatibility but
        ignored by this function.

    Returns
    -------
    rewards:
        A list of scalar rewards (sum of +0.1 per correct expression),
        one per completion.
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
    """Extract a normalized confidence value from XML.

    The function looks for a tag of the form
    ``<confidence>p</confidence>`` where ``p`` is a real number. If the
    tag or value is missing or invalid, a default of 0.5 is used. Values
    are clipped to the interval ``[0, 1]``.

    Parameters
    ----------
    text:
        The completion text containing a possible confidence tag.

    Returns
    -------
    confidence:
        A float value in the range ``[0, 1]`` representing the extracted
        or default confidence.
    """
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
    """Compute a Brier-style score based on correctness and confidence.

    For each completion, this function first computes a binary correctness
    label via :func:`correctness_reward_func`, mapping the reward to
    ``a in {0, 1}``. It then extracts a stated confidence ``p`` from the
    completion and returns:

        ``1 - (p - a) ** 2``

    Higher values indicate better calibrated confidence relative to the
    actual correctness.

    Parameters
    ----------
    completions:
        Batched model completions, each a sequence of message dictionaries.
        Only the first message's ``"content"`` field is used.
    answer:
        Sequence of ground-truth answers aligned with the completions.
    **kwargs:
        Additional keyword arguments forwarded to
        :func:`correctness_reward_func`.

    Returns
    -------
    scores:
        A list of Brier-style scores in the range ``[0, 1]``, one per
        completion.
    """
    correctness_rewards = correctness_reward_func(completions, answer, **kwargs)
    responses = [completion[0]["content"] for completion in completions]

    out: List[float] = []
    for resp, corr_reward in zip(responses, correctness_rewards):
        p = extract_confidence(resp)
        a = 1.0 if corr_reward == 2.0 else 0.0
        out.append(1.0 - (p - a) ** 2)
    return out
