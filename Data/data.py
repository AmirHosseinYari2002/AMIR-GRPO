from __future__ import annotations

from typing import Optional
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
    """Return the appropriate system prompt template."""

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

    Handles unicode minus, thousands separators, and trailing percent.
    Returns ``None`` on failure.
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

    Strategy:
      1) Prefer the **last** ``\boxed{...}`` region and take the first number inside.
      2) If none, fall back to the **last** number anywhere in the text.
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
    """Return the content inside `<answer> ... </answer>` if present, else empty string.

    Trims whitespace and supports multiline answers.
    """
    m = _XML_ANSWER_RE.search(text)
    return m.group(1).strip() if m else ""


def extract_hash_answer(text: str) -> str | None:
    """Extract the GSM8K-style final answer that follows a ``####`` delimiter."""
    if "####" not in text:
        return None
    return text.split("####", 1)[1].strip()


# ---------------------------------------------------------------------------
# Datasets
# ---------------------------------------------------------------------------

def get_gsm8k_questions(split: str = "train") -> Dataset:
    """Load GSM8K and format records for chat-style training.

    The function reads the global CLI args to determine whether calibration is
    enabled so it can inject the correct system prompt.
    """
    args = parse_args()
    calibration = getattr(args.core, "calibration", getattr(args, "calibration", False))

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
