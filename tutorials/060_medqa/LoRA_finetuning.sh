#!/bin/bash

#SBATCH -c 1
#SBATCH -t 1-00:00
#SBATCH --mem=60G

#SBATCH -p kempner_h100
#SBATCH --account kempner_mzitnik_lab
#SBATCH --gres=gpu:1

#SBATCH -o  LoRA_finetuning_cui_mlp.out
#SBATCH -e LoRA_finetuning_cui_mlp.err

export PATH="/n/home01/ruthjohnson/.local/bin:$PATH"

module load cuda/11.8.0-fasrc01
module load python/3.10.9-fasrc01

source /n/home01/ruthjohnson/venv_dgl/bin/activate

python3 train_custom_model.py \
    --model_name_or_path "Henrychur/MMed-Llama-3-8B-EnIns" \
    --data_path "/n/holylfs06/LABS/mzitnik_lab/Lab/ruthjohnson/kg_paper_revision/medqa/MMedBench/Train_eng" \
    --output_dir "/n/home01/ruthjohnson/ruthjohnson/kg_paper_revision/medqa/ft_cui_results" \
    --bf16 True \
    --num_train_epochs 3 \
    --per_device_train_batch_size 1 \
    --gradient_accumulation_steps 10 \
    --evaluation_strategy "no" \
    --save_strategy "epoch" \
    --save_total_limit 3 \
    --learning_rate 2e-5 \
    --weight_decay 0. \
    --warmup_ratio 0.03 \
    --lr_scheduler_type "cosine" \
    --logging_steps 1 \
    --is_lora True\
    --local_rank 16\
    --target_modules "proj_layer_1" "proj_layer_2" "proj_layer_3" "proj_layer_4" 

# "q_proj" "v_proj" "proj_layer"

