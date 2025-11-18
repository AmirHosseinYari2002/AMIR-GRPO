from __future__ import annotations

from pathlib import Path
import torch

from config import get_parser
from Eval.eval_utils import evaluate_model_batched, load_model_and_tokenizer
from Data.data import get_gsm8k_questions, get_aime25_questions, get_math500_questions


def main() -> None:
    parser = get_parser()
    args = parser.parse_args()

    out_dir = Path(args.model_dir).expanduser()
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
    else:
        raise ValueError(f"Unknown dataset: {args.dataset_name}")

    # ----------------------------
    # Device & model loading
    # ----------------------------
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model, tokenizer = load_model_and_tokenizer(
        directory_path=out,
        device=device,
        load_in_4bit=bool(args.load_in_4bit),
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
