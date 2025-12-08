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
    """Load GSM8K and format records for chat-style conversation.

    The function reads the global CLI args to determine whether calibration is
    enabled so it can inject the correct system prompt.
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
    """Load math-ai/aime25 and format records for chat-style conversation.

    The function reads the global CLI args to determine whether calibration is
    enabled so it can inject the correct system prompt.
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
    """Load HuggingFaceH4/MATH-500 and format records for chat-style conversation.

    The function reads the global CLI args to determine whether calibration is
    enabled so it can inject the correct system prompt.
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
    """Load Hothan/OlympiadBench (OE_TO_maths_en_COMP) and format for chat-style conversation.

    Uses 'question' as prompt and keeps 'final_answer' as a list[str].
    This is compatible with the grader that checks multiple correct answers.
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
    """Load math-ai/amc23 and format records for chat-style conversation.

    The function reads the global CLI args to determine whether calibration is
    enabled so it can inject the correct system prompt.
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
    """Load math-ai/minervamath and format records for chat-style conversation.

    The function reads the global CLI args to determine whether calibration is
    enabled so it can inject the correct system prompt.
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


def get_competition_math_questions(split: str = "train"):
    args = parse_args()
    calibration = args.core.calibration

    ds = load_dataset("qwedsacf/competition_math", split=split)

    # Keep Level 3–5
    ds = ds.filter(lambda ex: ex["level"] in {"Level 3", "Level 4", "Level 5"})

    # Regexes
    BOXED_RE = re.compile(r"\\boxed\s*\{([\s\S]*?)\}")   # match \boxed{ ... } across newlines
    NUM_RE   = re.compile(r"^-?\d+(?:\.\d+)?(?:e[+-]?\d+)?$", re.IGNORECASE)  # ints/decimals/sci
    FRAC_RE  = re.compile(r"^-?\s*\d+\s*/\s*\d+$")       # simple fractions like -a/b

    def extract_final(sol: str):
        m = BOXED_RE.search(sol)
        if not m:
            return None
        final = m.group(1).strip()

        # strip surrounding $ or trailing punctuation/spaces
        final = final.strip("$ ").rstrip(".,;")
        final = re.sub(r"\s+", "", final)

        # keep numbers and simple fractions only
        if NUM_RE.fullmatch(final) or FRAC_RE.fullmatch(final):
            return final
        return None

    def _format(example):
        final = extract_final(example["solution"])
        return {
            "prompt": [
                {"role": "system", "content": build_system_prompt(calibration)},
                {"role": "user",   "content": example["problem"]},
            ],
            "answer": final,
        }

    # Overwrite columns so trainer only sees what it needs
    ds = ds.map(_format, remove_columns=ds.column_names)
    # Now drop rows without a usable answer
    ds = ds.filter(lambda x: x["answer"] is not None)

    return ds


def get_aquarat_questions(split: str = "test") -> Dataset:
    """Load AQUA-RAT and format records for chat-style conversation.

    The function reads the global CLI args to determine whether calibration is
    enabled so it can inject the correct system prompt.
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
    """Load LiveMathBench and format records for chat-style conversation.

    The function reads the global CLI args to determine whether calibration is
    enabled so it can inject the correct system prompt.
    """
    args = parse_args()
    calibration = args.core.calibration

    data = load_dataset("opencompass/LiveMathBench", 'v202412_AMC_en')[split]

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
    """Load open-r1/DAPO-Math-17k-Processed (en) and format records for chat-style conversation.

    Uses the 'prompt' field as the user question and 'solution' as the final answer.
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
