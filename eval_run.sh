#!/bin/bash
#SBATCH --job-name=grpo_module             
#SBATCH --output=output_module.log
#SBATCH --nodes=1
#SBATCH --ntasks=1 
#SBATCH --mem=40G 
#SBATCH --cpus-per-task=64
#SBATCH --gres=gpu:1
#SBATCH -p cscc-gpu-p
#SBATCH --time=24:00:00
#SBATCH --qos=cscc-gpu-qos
#SBATCH --exclude=gpu-01


# Load Miniconda and activate environment
source ~/miniconda3/etc/profile.d/conda.sh
conda activate /home/amir.yari/.conda/envs/py311

python -m Eval.eval \
  --trainer_type grpo --dataset_name gsm8k --model_dir /l/users/amir.yari/Qwen2.5-7B-Instruct-grpo-modular
