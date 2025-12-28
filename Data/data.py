from __future__ import annotations

from typing import Optional, Any, Dict
import re

from datasets import Dataset, load_dataset

from config import parse_args

__all__ = [
    "build_system_prompt",
    "extract_numeric_answer",
    "extract_xml_answer",
    "extract_hash_answer",
    "get_gsm8k_questions",
]


# ---------------------------------------------------------------------------
# Prompt templates
# ---------------------------------------------------------------------------

_BASE_TEMPLATE = """A conversation between User and Assistant. The user asks a question, and the Assistant solves it. 
The Assistant first thinks about the reasoning process in the mind, provides the user with the final answer.
The reasoning process and answer are enclosed within <think> </think> and <answer> </answer> tags, respectively. Please reason step by step, and put your final answer within \\boxed{{}}.
The final format that must be followed is:
<think> reasoning process here </think>
<answer> final answer here </answer>
"""

_CALIBRATED_TEMPLATE = """A conversation between User and Assistant. The user asks a question, and the Assistant solves it. 
The Assistant first thinks about the reasoning process in the mind, provides the user with the final answer, 
then analyzes its confidence about the solution and provides the user with its confidence level. 
The confidence level is a number between 0 and 1 (inclusive) enclosed within <confidence> </confidence> tags. 
The final answer is enclosed between <answer> </answer> tags. 
The analysis about confidence and uncertainty is enclosed within <analysis> </analysis> tags. 
The Assistant should reason about its confidence in the solution and its uncertainty in the solution within these tags. 
The final format that must be followed is:
<think> reasoning process here </think>
<answer> \\boxed{{final answer here}} </answer>
<analysis> analysis about confidence and uncertainty here </analysis>
<confidence> confidence level here (number between 0 and 1) </confidence>
"""


def build_system_prompt(calibration: bool = False) -> str:
    """Return the appropriate system prompt template.

    Parameters
    ----------
    calibration:
        If ``True``, return the calibrated prompt template that includes
        ``<analysis>`` and ``<confidence>`` sections. Otherwise, return the
        base template with only ``<think>`` and ``<answer>`` fields.

    Returns
    -------
    prompt:
        The formatted system prompt string to be used in the chat messages.
    """

    return _CALIBRATED_TEMPLATE if calibration else _BASE_TEMPLATE


# ---------------------------------------------------------------------------
# Numeric & text extractors
# ---------------------------------------------------------------------------

_NUMBER_RE = re.compile(
    r"""
    (?<![A-Za-z_])                # don't start in the middle of a word
    [-+−]?                        # optional ASCII or Unicode minus/plus
    (?:                           # number body
        (?:\d{1,3}(?:,\d{3})+|\d+)  # integers with/without thousands
        (?:\.\d+)?                  # optional decimal
        (?:[eE][+-]?\d+)?           # optional scientific exponent
    )
    %?                            # optional percent sign
    (?![A-Za-z_])                 # don't end in the middle of a word
    """,
    re.VERBOSE,
)

# Matches \boxed{...} anywhere, including inside \( ... \), \[ ... \], or $$ ... $$
_BOXED_RE = re.compile(r"\\boxed\s*\{\s*([^{}]+?)\s*\}", re.DOTALL)

# XML <answer>...</answer> (multiline, greedy minimal)
_XML_ANSWER_RE = re.compile(r"<answer>\s*([\s\S]*?)\s*</answer>", re.IGNORECASE)


def _clean_to_float(s: str) -> Optional[float]:
    """Normalize common LaTeX/Unicode artifacts and parse as float.

    This helper:
    - Replaces LaTeX percent and thousand separators.
    - Converts Unicode minus to ASCII minus.
    - Strips surrounding dollar signs and whitespace.
    - Removes trailing percent signs, if any.

    Parameters
    ----------
    s:
        Raw string containing a numeric value, potentially with LaTeX or
        formatting artifacts.

    Returns
    -------
    value:
        Parsed float value if successful, otherwise ``None``.
    """
    s = (
        s.replace(r"\%", "%")
        .replace("−", "-")  # Unicode minus to ASCII
        .replace("$", "")
        .replace(r"\,", "")  # LaTeX thousand separator
        .replace(",", "")  # regular thousand separator
        .strip()
    )
    if s.endswith("%"):
        s = s[:-1]
    try:
        return float(s)
    except ValueError:
        return None


def extract_numeric_answer(text: str) -> Optional[float]:
    """Extract a numeric answer from model output.

    Strategy
    --------
    1. Prefer the **last** ``\\boxed{...}`` region and take the first number
       inside it.
    2. If no boxed answer is present, fall back to the **last** number
       appearing anywhere in the text.

    Parameters
    ----------
    text:
        Full model output as a string.

    Returns
    -------
    value:
        Parsed float value if a numeric answer can be extracted, otherwise
        ``None``.
    """
    boxed_matches = list(_BOXED_RE.finditer(text))
    if boxed_matches:
        inner = boxed_matches[-1].group(1)
        m = _NUMBER_RE.search(inner)
        if m:
            val = _clean_to_float(m.group(0))
            if val is not None:
                return val

    tokens = list(_NUMBER_RE.finditer(text))
    if not tokens:
        return None
    return _clean_to_float(tokens[-1].group(0))


def extract_xml_answer(text: str) -> str:
    """Extract the content inside an ``<answer> ... </answer>`` block.

    The function uses a multiline, case-insensitive regex to locate the first
    ``<answer>...</answer>`` span and returns its inner text, stripped of
    leading and trailing whitespace. If no such block exists, an empty string
    is returned.

    Parameters
    ----------
    text:
        Full model output as a string, potentially containing XML-like tags.

    Returns
    -------
    answer:
        The stripped inner content of the ``<answer>`` tag, or ``""`` if no
        answer block is found.
    """
    m = _XML_ANSWER_RE.search(text)
    return m.group(1).strip() if m else ""


def extract_hash_answer(text: str) -> str | None:
    """Extract a GSM8K-style final answer following a ``####`` delimiter.

    This is intended for the original GSM8K format where the final answer is
    separated from the solution text by ``####`` (e.g., ``"... solution ... #### 42"``).

    Parameters
    ----------
    text:
        The GSM8K answer string containing the ``####`` delimiter.

    Returns
    -------
    answer:
        The substring following ``####``, stripped of whitespace, or ``None``
        if the delimiter is not present.
    """
    if "####" not in text:
        return None
    return text.split("####", 1)[1].strip()


# ---------------------------------------------------------------------------
# Datasets
# ---------------------------------------------------------------------------


def get_gsm8k_questions(split: str = "train") -> Dataset:
    """Load GSM8K and format records for chat-style conversation.

    The function reads the global CLI config via :func:`parse_args` to
    determine whether calibration is enabled, and injects the corresponding
    system prompt into each example.

    Parameters
    ----------
    split:
        Dataset split to load (e.g., ``"train"``, ``"test"``).

    Returns
    -------
    dataset:
        A :class:`datasets.Dataset` where each example has fields:
        ``"prompt"`` (list of chat messages) and ``"answer"`` (gold solution).
    """
    args = parse_args()
    calibration = args.core.calibration

    data = load_dataset("openai/gsm8k", "main")[split]
    data = data.map(
        lambda x: {
            "prompt": [
                {"role": "system", "content": build_system_prompt(calibration)},
                {"role": "user", "content": x["question"]},
            ],
            "answer": extract_hash_answer(x["answer"]),
        }
    )
    return data


def get_aime25_questions(split: str = "test") -> Dataset:
    """Load ``math-ai/aime25`` and format records for chat-style conversation.

    The function reads the global CLI config to determine whether calibration
    is enabled, and injects the corresponding system prompt.

    Parameters
    ----------
    split:
        Dataset split to load (typically ``"test"``).

    Returns
    -------
    dataset:
        A :class:`datasets.Dataset` with ``"prompt"`` and ``"answer"`` fields.
    """
    args = parse_args()
    calibration = args.core.calibration

    ds = load_dataset("math-ai/aime25", "default")[split]

    def _format(example):
        return {
            "prompt": [
                {"role": "system", "content": build_system_prompt(calibration)},
                {"role": "user", "content": example["problem"]},
            ],
            "answer": str(example["answer"]).strip(),
        }

    return ds.map(_format)


def get_math500_questions(split: str = "test") -> Dataset:
    """Load ``HuggingFaceH4/MATH-500`` and format for chat-style conversation.

    The function reads the global CLI config to determine whether calibration
    is enabled, and injects the corresponding system prompt.

    Parameters
    ----------
    split:
        Dataset split to load (typically ``"test"``).

    Returns
    -------
    dataset:
        A :class:`datasets.Dataset` with ``"prompt"`` and ``"answer"`` fields.
    """
    args = parse_args()
    calibration = args.core.calibration

    ds = load_dataset("HuggingFaceH4/MATH-500")[split]

    def _format(example):
        return {
            "prompt": [
                {"role": "system", "content": build_system_prompt(calibration)},
                {"role": "user", "content": example["problem"]},
            ],
            "answer": example["answer"].strip(),
        }

    return ds.map(_format)


def get_olympiadbench_questions(split: str = "train") -> Dataset:
    """Load ``Hothan/OlympiadBench`` and format for chat-style conversation.

    This loader uses the ``OE_TO_maths_en_COMP`` configuration. The
    ``final_answer`` field (which may be a string or list) is normalized to a
    list of strings, which is compatible with graders that support multiple
    correct answers.

    Parameters
    ----------
    split:
        Dataset split to load (e.g. ``"train"``, ``"validation"``, ``"test"``).

    Returns
    -------
    dataset:
        A :class:`datasets.Dataset` with:
        - ``"prompt"``: list of chat messages.
        - ``"answer"``: list of acceptable answers.
    """
    args = parse_args()
    calibration = args.core.calibration

    ds = load_dataset("Hothan/OlympiadBench", "OE_TO_maths_en_COMP")[split]

    def _format(example: Dict[str, Any]) -> Dict[str, Any]:
        fa = example["final_answer"]
        if isinstance(fa, str):
            answers = [fa.strip()]
        elif isinstance(fa, list):
            answers = [a.strip() for a in fa]
        else:
            raise TypeError(f"Unexpected final_answer type: {type(fa)}")

        return {
            "prompt": [
                {"role": "system", "content": build_system_prompt(calibration)},
                {"role": "user", "content": example["question"].strip()},
            ],
            "answer": answers,
        }

    return ds.map(_format)


def get_amc23_questions(split: str = "test") -> Dataset:
    """Load ``math-ai/amc23`` and format records for chat-style conversation.

    The function reads the global CLI config to determine whether calibration
    is enabled, and injects the corresponding system prompt.

    Parameters
    ----------
    split:
        Dataset split to load (typically ``"test"``).

    Returns
    -------
    dataset:
        A :class:`datasets.Dataset` with ``"prompt"`` and ``"answer"`` fields.
    """
    args = parse_args()
    calibration = args.core.calibration

    ds = load_dataset("math-ai/amc23")[split]

    def _format(example):
        return {
            "prompt": [
                {"role": "system", "content": build_system_prompt(calibration)},
                {"role": "user", "content": example["question"]},
            ],
            "answer": example["answer"].strip(),
        }

    return ds.map(_format)


def get_minervamath_questions(split: str = "test") -> Dataset:
    """Load ``math-ai/minervamath`` and format records for chat-style conversation.

    The function reads the global CLI config to determine whether calibration
    is enabled, and injects the corresponding system prompt.

    Parameters
    ----------
    split:
        Dataset split to load (typically ``"test"``).

    Returns
    -------
    dataset:
        A :class:`datasets.Dataset` with ``"prompt"`` and ``"answer"`` fields.
    """
    args = parse_args()
    calibration = args.core.calibration

    ds = load_dataset("math-ai/minervamath")[split]

    def _format(example):
        return {
            "prompt": [
                {"role": "system", "content": build_system_prompt(calibration)},
                {"role": "user", "content": example["question"]},
            ],
            "answer": example["answer"].strip(),
        }

    return ds.map(_format)


def get_aquarat_questions(split: str = "test") -> Dataset:
    """Load ``deepmind/aqua_rat`` and format records for chat-style conversation.

    The function:
    - Selects the correct option based on the provided answer letter.
    - Strips the multiple-choice label (e.g. ``"A)"``) from the final answer.
    - Injects the appropriate system prompt depending on calibration.

    Parameters
    ----------
    split:
        Dataset split to load (typically ``"test"``).

    Returns
    -------
    dataset:
        A :class:`datasets.Dataset` with standardized ``"prompt"`` and
        ``"answer"`` fields.
    """
    args = parse_args()
    calibration = args.core.calibration

    data = load_dataset("deepmind/aqua_rat")[split]

    def _format_example(x):
        correct_letter = x["correct"].strip()
        answer_text = None

        # options look like ["A) ...", "B) ...", ...]
        for opt in x["options"]:
            opt_stripped = opt.strip()
            # Match first character to the correct letter
            if opt_stripped and opt_stripped[0] == correct_letter:
                # Drop the "A)" / "B)" etc. label and leading spaces
                parts = opt_stripped.split(")", 1)
                answer_text = parts[1].strip() if len(parts) > 1 else opt_stripped
                break

        return {
            "prompt": [
                {"role": "system", "content": build_system_prompt(calibration)},
                {"role": "user", "content": x["question"]},
            ],
            "answer": answer_text,
        }

    data = data.map(_format_example)
    return data


def get_livemathbench_questions(split: str = "test") -> Dataset:
    """Load ``opencompass/LiveMathBench`` and format for chat-style conversation.

    This loader:
    - Uses the ``v202412_AMC_en`` configuration.
    - Strips surrounding ``$...$`` or ``$$...$$`` from answers where present.
    - Injects the appropriate system prompt depending on calibration.

    Parameters
    ----------
    split:
        Dataset split to load (typically ``"test"``).

    Returns
    -------
    dataset:
        A :class:`datasets.Dataset` with standardized ``"prompt"`` and
        ``"answer"`` fields.
    """
    args = parse_args()
    calibration = args.core.calibration

    data = load_dataset("opencompass/LiveMathBench", "v202412_AMC_en")[split]

    def strip_dollars(s: str) -> str:
        # remove surrounding $...$ or $$...$$
        s = s.strip()
        if s.startswith("$$") and s.endswith("$$"):
            return s[2:-2].strip()
        if s.startswith("$") and s.endswith("$"):
            return s[1:-1].strip()
        return s

    def _format_example(x):
        answer_clean = strip_dollars(x["answer"])

        return {
            "prompt": [
                {"role": "system", "content": build_system_prompt(calibration)},
                {"role": "user", "content": x["question"]},
            ],
            "answer": answer_clean,
        }

    data = data.map(_format_example)
    return data


def get_dapo_math_questions(split: str = "train") -> Dataset:
    """Load ``open-r1/DAPO-Math-17k-Processed`` and format for chat-style conversation.

    The loader:
    - Uses the ``"en"`` configuration.
    - Treats the ``"prompt"`` field as the user query.
    - Treats the ``"solution"`` field as the final answer.

    Parameters
    ----------
    split:
        Dataset split to load (e.g. ``"train"``, ``"validation"``, ``"test"``).

    Returns
    -------
    dataset:
        A :class:`datasets.Dataset` with standardized ``"prompt"`` and
        ``"answer"`` fields.
    """
    args = parse_args()
    calibration = args.core.calibration

    ds = load_dataset("open-r1/DAPO-Math-17k-Processed", "en")[split]

    def _format(example):
        return {
            "prompt": [
                {"role": "system", "content": build_system_prompt(calibration)},
                {"role": "user", "content": example["prompt"]},
            ],
            "answer": str(example["solution"]).strip(),
        }

    return ds.map(_format)
