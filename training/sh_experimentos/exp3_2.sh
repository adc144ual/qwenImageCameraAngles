#!/bin/bash

LATENTS_DIR="/mnt/nas_dataset/antonio/qwen/datasets/dataset_local_latents_512_idx/"
OUTPUT_DIR="/mnt/nas_dataset/antonio/qwen/OUTPUTS_EXPERIMENTOS/exp3_grupos_CV_v2"
HRNET_PATH="/nas/antoniodetoro/qwen/Qwen-Image-Edit-Angles-2/src/hr_net/models/hrnet_finetuned_best_mal.pth"
TRANSFORMER_MODEL="/mnt/nas_dataset/antonio/qwen/Qwen-Fused-Angles"

EXP_JSON_1="/mnt/nas_dataset/antonio/qwen/experimentos/exp3.2/exp4_fase1_facil.json"
EXP_JSON_2="/mnt/nas_dataset/antonio/qwen/experimentos/exp3.2/exp4_fase2_uniforme.json"
EXP_JSON_3="/mnt/nas_dataset/antonio/qwen/experimentos/exp3.2/exp4_fase3_dificil.json"

mkdir -p $OUTPUT_DIR

echo "========================================"
echo "Lanzando experimento multi-etapa"
echo "  Etapa 0: $EXP_JSON_1"
echo "  Etapa 1: $EXP_JSON_2"
echo "  Etapa 2: $EXP_JSON_3"
echo "========================================"

python ../finetune.py \
    --latents_dir $LATENTS_DIR \
    --output_dir $OUTPUT_DIR \
    --transformer_model $TRANSFORMER_MODEL \
    --hrnet_model_path $HRNET_PATH \
    --experiment_json $EXP_JSON_1 $EXP_JSON_2 $EXP_JSON_3 \
    --epochs 10 \
    --batch_size 2 \
    --learning_rate 1e-4 \
    --heatmap_loss_weight 0.0 \
    --velocity_loss_weight 1.0 \
    --heatmap_loss_type mse \
    --inference_every 1 \
    --inference_steps 4 \
    --inference_samples 12

if [ $? -ne 0 ]; then
    echo "ERROR en el entrenamiento. Abortando."
    exit 1
fi

echo "========================================"
echo "Experimento completado. Resultados en $OUTPUT_DIR"
echo "========================================"