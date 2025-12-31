<div align="center">

# AMIR-GRPO: Inducing Implicit Preference Signals into GRPO

<p>
  <a href="https://amirhosseinyari2002.github.io/"><b>Amir Hossein Yari</b></a>
  ·
  <a href="https://www.fajrikoto.com/home"><b>Fajri Koto</b></a>
</p>

<p>
  <a href="#-key-results">Key Results</a> •
  <a href="#-installation-and-setup">Installation</a> •
  <a href="#-training">Training</a> •
  <a href="#-evaluation">Evaluation</a> •
  <a href="#-citation">Citation</a>
</p>

</div>

---

## ✨ What is AMIR-GRPO?

AMIR-GRPO is an extension of GRPO (Group Relative Policy Optimization) that leverages *implicit preference signals* derived from within-group reward rankings.  
It introduces a preference-style regularization term to strengthen learning signals—without requiring external human preference labels.

---

## 📈 Key Results



---

## 🛠️ Installation and Setup

```bash
# Clone the repository
git clone https://github.com/AmirHosseinYari2002/AMIR-GRPO.git
cd AMIR-GRPO

# Install necessary libraries
pip install -U pip
pip install -r requirements.txt
````

---

## 🚀 Training

Training is launched via `python -m Train.train` and the CLI mirrors the full configuration.

```bash
python -m Train.train \
  --model_name google/gemma-3-4b-it \
  --lora_rank 16 \
  --max_seq_length 2048 \
  --load_in_4bit 0 \
  --model_dir trained_model_directory \
  --dataset_name gsm8k \
  --dataset_split train \
  --trainer_type amir_grpo \
  --calibration \
  --learning_rate 5e-6 \
  --weight_decay 0.1 \
  --max_grad_norm 0.1 \
  --per_device_train_batch_size 1 \
  --gradient_accumulation_steps 4 \
  --max_steps 1000 \
  --seed 0 \
  --lr_scheduler_type cosine \
  --warmup_ratio 0.1 \
  --optim adamw_8bit \
  --adam_beta1 0.9 \
  --adam_beta2 0.99 \
  --num_generations 8 \
  --max_prompt_length 1024 \
  --max_completion_length 1024 \
  --loss_type grpo \
  --epsilon 0.20 \
  --epsilon_high 0.20 \
  --mask_truncated_completions 0 \
  --scale_rewards group \
  --importance_sampling_level token \
  --lambda_reg 0.01 \
  --reward_margin 2.0 \
  --beta_dpo 0.2 \
  --pair_mining all \
  --max_pairs_per_group None \
  --ref_free true \
  --logging_steps 1 \
  --save_steps 50 \
  --report_to wandb \
  --wandb_api_key your_wandb_api_key

```

---

## 📊 Evaluation

Evaluate the performance of the trained model using the Eval.eval script.

```bash
python -m Eval.eval \
  --dataset_name olympiadbench \
  --model_dir trained_model_directory
```

---


## 📝 Citation
