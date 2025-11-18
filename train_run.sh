#!/bin/bash
#SBATCH --job-name=grpo_3x_one_node             
#SBATCH --output=logs/%x_%j.master.out
#SBATCH --nodes=1
#SBATCH --ntasks=3 
#SBATCH --mem=90G 
#SBATCH --cpus-per-task=16
#SBATCH --gres=gpu:3
#SBATCH -p cscc-gpu-p
#SBATCH --time=24:00:00
#SBATCH --qos=cscc-gpu-qos
#SBATCH --exclude=gpu-01
#SBATCH --exclude=gpu-05


# # Load Miniconda and activate environment
# source ~/miniconda3/etc/profile.d/conda.sh
# conda activate /home/amir.yari/.conda/envs/py311

# python -m Train.train \
#   --model_name Qwen/Qwen2.5-7B-Instruct \
#   --trainer_type grpo \
#   --dataset_name gsm8k \
#   --model_dir /l/users/amir.yari/Qwen2.5-7B-Instruct-grpo \
#   --wandb_api_key 8dec9aef24361f033ede83f00a9a6ad914f53b3b


# -------- launches (3 parallel steps; each gets 1 GPU) --------

srun --ntasks=3 \
     --gpus-per-task=1 \
     --gpu-bind=single:1 \
     --kill-on-bad-exit=0 \
     -o logs/grpo_rank%t.out \
     bash -lc '
  # env per task
  source ~/miniconda3/etc/profile.d/conda.sh
  conda activate /home/amir.yari/.conda/envs/py311

  echo "Task $SLURM_PROCID has physical GPU(s): $SLURM_STEP_GPUS"

  case "$SLURM_PROCID" in
    0)
      python -m Train.train \
        --model_name Qwen/Qwen2.5-7B-Instruct \
        --trainer_type grpo \
        --dataset_name gsm8k \
        --model_dir /l/users/amir.yari/Qwen2.5-7B-Instruct-grpo \
        --wandb_api_key 8dec9aef24361f033ede83f00a9a6ad914f53b3b
      ;;
    1)
      python -m Train.train \
        --model_name meta-llama/Llama-3.2-3B-Instruct \
        --trainer_type grpo \
        --dataset_name gsm8k \
        --model_dir /l/users/amir.yari/Llama-3.2-3B-Instruct-grpo \
        --wandb_api_key 8dec9aef24361f033ede83f00a9a6ad914f53b3b
      ;;
    2)
      python -m Train.train \
        --model_name Qwen/Qwen2.5-7B-Instruct \
        --trainer_type grpo_dpo \
        --dataset_name gsm8k \
        --model_dir /l/users/amir.yari/Qwen2.5-3B-Instruct-grpo_dpo \
        --wandb_api_key 8dec9aef24361f033ede83f00a9a6ad914f53b3b
      ;;
  esac
'
