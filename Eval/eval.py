from __future__ import annotations

from pathlib import Path
import torch

from config import parse_args
from Eval.eval_utils import evaluate_model_batched, load_model_and_tokenizer, load_model_and_tokenizer_from_hf
from Data.data import get_gsm8k_questions, get_aime25_questions, get_math500_questions, get_olympiadbench_questions, get_amc23_questions, get_minervamath_questions, get_aquarat_questions


def main() -> None:
    # Parse CLI into typed config
    args = parse_args()

    out_dir = Path(args.core.model_dir).expanduser()
    out = out_dir.as_posix()

    # ----------------------------
    # Dataset loading
    # ----------------------------
    if args.core.dataset_name.lower() == "gsm8k":
        test_dataset = get_gsm8k_questions(args.core.test_dataset_split)
    elif args.core.dataset_name.lower() == "aime25":
        test_dataset = get_aime25_questions(args.core.test_dataset_split)
    elif args.core.dataset_name.lower() == "math500":
        test_dataset = get_math500_questions(args.core.test_dataset_split)
    elif args.core.dataset_name.lower() == "olympiadbench":
        test_dataset = get_olympiadbench_questions()
    elif args.core.dataset_name.lower() == "amc23":
        test_dataset = get_amc23_questions(args.core.test_dataset_split)
    elif args.core.dataset_name.lower() == "aquarat":
        test_dataset = get_aquarat_questions(args.core.test_dataset_split)
    elif args.core.dataset_name.lower() == "minervamath":
        test_dataset = get_minervamath_questions(args.core.test_dataset_split)
    else:
        raise ValueError(f"Unknown dataset: {args.core.dataset_name}")

    # ----------------------------
    # Device & model loading
    # ----------------------------
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model, tokenizer = load_model_and_tokenizer(
        directory_path=out,
        device=device,
        load_in_4bit=bool(args.core.load_in_4bit),
    )
    model.eval()

    # ----------------------------
    # Evaluation
    # ----------------------------
    evaluate_model_batched(
        model=model,
        tokenizer=tokenizer,
        dataset=test_dataset,
        batch_size=8,
        device=device,
        progress=True,
    )


if __name__ == "__main__":
    main()
