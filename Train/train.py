from __future__ import annotations

from pathlib import Path
import wandb

from config import parse_args, save_config_files
from Train.models import bf16_fp16_flags, load_ref_model, load_train_model
from Train.rewards import (
    brier_score,
    correctness_reward_func,
    expression_correctness_reward_func,
    int_reward_func,
    strict_format_reward_func,
    strict_format_reward_func_with_calib,
    xmlcount_reward_func,
)
from trl import GRPOConfig
from trl.trainer.grpo_trainer import GRPOTrainer

# Local: hybrid trainer that adds DPO regularization
from Train.trainer import GRPOWithDPOTrainer

# Datasets
from Data.data import get_gsm8k_questions 


# ---------------------------------------------------------------------------
# Config construction helpers
# ---------------------------------------------------------------------------

def build_grpo_config(args) -> GRPOConfig:
    """Build a :class:`GRPOConfig` from parsed CLI args (dataclass/namespace).

    Uses `bf16_fp16_flags()` to set precision flags appropriately.
    """
    bf16, fp16 = bf16_fp16_flags()
    return GRPOConfig(
        # Training
        learning_rate=args.training.learning_rate,
        weight_decay=args.training.weight_decay,
        max_grad_norm=args.training.max_grad_norm,
        per_device_train_batch_size=args.training.per_device_train_batch_size,
        gradient_accumulation_steps=args.training.gradient_accumulation_steps,
        max_steps=args.training.max_steps,
        seed=args.training.seed,
        # Scheduler & Optimizer
        lr_scheduler_type=args.sched_optim.lr_scheduler_type,
        warmup_ratio=args.sched_optim.warmup_ratio,
        optim=args.sched_optim.optim,
        adam_beta1=args.sched_optim.adam_beta1,
        adam_beta2=args.sched_optim.adam_beta2,
        # Precision
        bf16=bf16,
        fp16=fp16,
        # Generation
        num_generations=args.generation.num_generations,
        max_prompt_length=args.generation.max_prompt_length,
        max_completion_length=args.generation.max_completion_length,
        # Algorithm
        loss_type=args.algorithm.loss_type,
        epsilon=args.algorithm.epsilon,
        epsilon_high=args.algorithm.epsilon_high,
        mask_truncated_completions=bool(args.algorithm.mask_truncated_completions),
        scale_rewards=args.algorithm.scale_rewards,
        importance_sampling_level=args.algorithm.importance_sampling_level,
        # Logging / IO
        logging_steps=args.logging.logging_steps,
        save_steps=args.logging.save_steps,
        report_to=args.logging.report_to,
        run_name=f"{args.core.model_dir}/output",
        output_dir=args.core.model_dir,
    )


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    # Parse CLI into typed config
    args = parse_args()

    wandb.login(args.logging.wandb_api_key)

    # Output path
    out = Path(args.core.model_dir).expanduser()
    out.mkdir(parents=True, exist_ok=True)

    # Build trainer args
    training_args = build_grpo_config(args)

    # ----------------------------
    # Model loading
    # ----------------------------
    model_name = args.core.model_name
    lora_rank = args.core.lora_rank 
    max_seq_length = args.core.max_seq_length 
    load_in_4bit = bool(args.core.load_in_4bit)

    model, tokenizer = load_train_model(
        model_name=model_name,
        lora_rank=lora_rank,
        max_seq_length=max_seq_length,
        load_in_4bit=load_in_4bit,
    )

    # ----------------------------
    # Dataset Loading
    # ----------------------------
    dataset_name = args.core.dataset_name.lower()
    dataset_split = args.core.dataset_split

    if dataset_name == "gsm8k":
        train_dataset = get_gsm8k_questions(dataset_split)
    else:
        raise ValueError(f"Unknown dataset: {dataset_name}")

    # ----------------------------
    # Reference model (for DPO)
    # ----------------------------
    implicit_ref = args.dpo.implicit_ref
    ref_model = None
    if not implicit_ref:
        ref_model = load_ref_model(
            model_name=model_name,
            max_seq_length=max_seq_length,
            load_in_4bit=load_in_4bit,
        )

    # ----------------------------
    # Reward functions
    # ----------------------------
    use_calibration = args.core.calibration
    if use_calibration:
        reward_funcs = [
            brier_score,
            expression_correctness_reward_func,
            xmlcount_reward_func,
            strict_format_reward_func_with_calib,
            int_reward_func,
            correctness_reward_func,
        ]
    else:
        reward_funcs = [
            expression_correctness_reward_func,
            xmlcount_reward_func,
            strict_format_reward_func,
            int_reward_func,
            correctness_reward_func,
        ]

    # ----------------------------
    # Trainer selection & init
    # ----------------------------
    trainer_type = args.core.trainer_type
    TrainerCls = GRPOWithDPOTrainer if trainer_type == "grpo_dpo" else GRPOTrainer

    trainer_kwargs = dict(
        model=model,
        processing_class=tokenizer,
        reward_funcs=reward_funcs,
        args=training_args,
        train_dataset=train_dataset,
    )

    if TrainerCls is GRPOWithDPOTrainer:
        trainer_kwargs.update(
            ref_model=ref_model,
            lambda_pair=args.dpo.lambda_pair,
            pair_threshold=args.dpo.pair_threshold,
            beta_dpo=args.dpo.beta_dpo,
            pair_mining=args.dpo.pair_mining,
            max_pairs_per_group=args.dpo.max_pairs_per_group,
        )

    trainer = TrainerCls(**trainer_kwargs)
    trainer.train()

    # ----------------------------
    # Save artifacts
    # ----------------------------
    model.save_pretrained(out.as_posix())
    tokenizer.save_pretrained(out.as_posix())
    model.config.save_pretrained(out.as_posix())
    save_config_files(args, out)


if __name__ == "__main__":
    main()
