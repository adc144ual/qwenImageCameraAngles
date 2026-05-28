#!/bin/bash

LATENTS_DIR="/home/antoniodetoro/nas_dataset/antonio/qwen/datasets/dataset_local_latents_512_idx/"
OUTPUT_DIR="/home/antoniodetoro/nas_dataset/antonio/qwen/OUTPUTS_EXPERIMENTOS/exp2_random_grupos_v1"
HRNET_PATH="/nas/antoniodetoro/qwen/Qwen-Image-Edit-Angles-2/src/hr_net/models/hrnet_finetuned_best.pth"
TRANSFORMER_MODEL="/home/antoniodetoro/nas_dataset/antonio/qwen/Qwen-Fused-Angles"

EXP_JSON_1="/home/antoniodetoro/nas_dataset/antonio/qwen/experimentos/exp2/exp2_grupo1_random.json"
EXP_JSON_2="/home/antoniodetoro/nas_dataset/antonio/qwen/experimentos/exp2/exp2_grupo2_random.json"
EXP_JSON_3="/home/antoniodetoro/nas_dataset/antonio/qwen/experimentos/exp2/exp2_grupo3_random.json"

COMMON_ARGS="
    --latents_dir $LATENTS_DIR
    --output_dir $OUTPUT_DIR
    --transformer_model $TRANSFORMER_MODEL
    
    --batch_size 2
    --learning_rate 1e-4
    --heatmap_loss_weight 0.0
    --velocity_loss_weight 1.0
    --heatmap_loss_type mse
    --inference_every 1
    --inference_steps 4
    --inference_samples 12
"

mkdir -p $OUTPUT_DIR

echo "========================================"
echo "ETAPA 1: $EXP_JSON_1"
echo "========================================"
python ../finetune.py \
    --experiment_json $EXP_JSON_1 \
    $COMMON_ARGS --epochs 10

if [ $? -ne 0 ]; then
    echo "ERROR en etapa 1. Abortando."
    exit 1
fi

echo "========================================"
echo "ETAPA 2: $EXP_JSON_2"
echo "========================================"
python ../finetune.py \
    --experiment_json $EXP_JSON_2 \
    $COMMON_ARGS --epochs 19

if [ $? -ne 0 ]; then
    echo "ERROR en etapa 2. Abortando."
    exit 1
fi

echo "========================================"
echo "ETAPA 3: $EXP_JSON_3"
echo "========================================"
python ../finetune.py \
    --experiment_json $EXP_JSON_3 \
    $COMMON_ARGS --epochs 29

if [ $? -ne 0 ]; then
    echo "ERROR en etapa 3. Abortando."
    exit 1
fi

echo "========================================"
echo "Experimento completado. Resultados en $OUTPUT_DIR"
echo "========================================"