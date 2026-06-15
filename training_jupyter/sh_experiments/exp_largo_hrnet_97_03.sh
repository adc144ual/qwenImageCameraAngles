#!/bin/bash
#SBATCH --job-name=exp_single_stage
#SBATCH --output=%x_%j.log
#SBATCH --error=%x_%j.err
#SBATCH --cpus-per-task=16
#SBATCH --mem=100G
#SBATCH --partition=gpu_volta
#SBATCH --gres=gpu:1

# module load python/3.10.10
# module load cuda/12.6.3
# VENV_DIR=~/envs/venv_jupyter

# source "$VENV_DIR/bin/activate"

# export CUDA_VISIBLE_DEVICES=0

LATENTS_DIR="/mnt/nas_dataset/antonio/qwen/datasets/dataset_local_latents_512_idx/"
# LATENTS_DIR="/nas/antoniodetoro/datasets/qwen/dataset_local_latents_512_idx/"
OUTPUT_DIR="/mnt/nas_dataset/antonio/qwen/OUTPUTS_EXPERIMENTOS/exp_single_stage_HRNet_97_03"
HRNET_PATH="/nas/antoniodetoro/qwen/Qwen-Image-Edit-Angles-2/src/hr_net/outputs/27kp_sucio_v1/hrnet_finetuned_best.pth"
TRANSFORMER_MODEL="/mnt/nas_dataset/antonio/qwen/Qwen-Fused-Angles"

EXP_JSON="/mnt/nas_dataset/antonio/qwen/experimentos/exp_largo/dataset_splits_uniformes_rareza_ususarios.json"   # JSON original se llama dataset_splits_uniformes_rareza_ususarios.json

mkdir -p $OUTPUT_DIR

python ../finetune.py \
    --reset_patience \
    --experiment_json $EXP_JSON \
    --latents_dir $LATENTS_DIR \
    --hrnet_model_path $HRNET_PATH \
    --output_dir $OUTPUT_DIR \
    --transformer_model $TRANSFORMER_MODEL \
    --batch_size 1 \
    --learning_rate 1e-4 \
    --heatmap_loss_weight 0.03 \
    --velocity_loss_weight 0.97 \
    --heatmap_loss_type weighted_mse \
    --inference_every 1 \
    --inference_steps 4 \
    --inference_samples 12 \
    --epochs 100 \
    --patience 3
    