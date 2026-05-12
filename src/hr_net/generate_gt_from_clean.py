#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Preparación de datos para fine-tuning de HRNet
===============================================

Lee una carpeta de imágenes y genera la estructura que consume 1_finetuning_hr_net_pose.py:

    dataset_hrnet/
    └── train_val/  (o test/, el script se lanza una vez por conjunto)
        ├── *.png       ← imágenes de entrenamiento copiadas
        └── *.npy       ← heatmaps GT (regular + flip)

Naming convention esperado en la carpeta de entrada:
    {base}_rgb_clean.png           → fuente GT;        GT = {base}_rgb.npy
    {base}_rgb_clean_flip.png      → imagen flip;      GT = {base}_rgb_flip.npy
    {base}_rgb_clean_aug1.png      → aumentada;        GT = {base}_rgb.npy
    {base}_rgb_clean_aug1_flip.png → aumentada + flip; GT = {base}_rgb_flip.npy
    ... (aug2, aug3 igual)
    {base}_rgb_noise_X.XX.png      → con ruido;        GT = {base}_rgb.npy

Uso:
    python 0_generate_gt_from_clean.py
    python 0_generate_gt_from_clean.py --input-dir ./imgs --output-dir ./dataset_hrnet/train_val
"""

import argparse
import logging
import re
import shutil
from pathlib import Path

import cv2
import numpy as np
import torch
from tqdm import tqdm

from hrnet_inference import PoseHRNet

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


def load_model(model_path: str, device: torch.device) -> PoseHRNet:
    """Cargar HRNet-W48 pre-entrenado."""
    model = PoseHRNet(width=48, num_joints=17)
    try:
        sd = torch.load(model_path, map_location=device, weights_only=False)
    except TypeError:
        sd = torch.load(model_path, map_location=device)
    if isinstance(sd, dict) and 'state_dict' in sd:
        sd = sd['state_dict']
    model.load_state_dict(sd, strict=True)
    model.eval()
    model.to(device)
    return model


def get_heatmaps(model, image: np.ndarray, device: torch.device,
                 input_size=(288, 384)) -> np.ndarray:
    """Preprocesar imagen y obtener heatmaps (igual que hrnet_inference.py)."""
    h, w = input_size
    resized = cv2.resize(image, (w, h), interpolation=cv2.INTER_LINEAR)
    mean = np.array([0.485, 0.456, 0.406], dtype=np.float32)
    std = np.array([0.229, 0.224, 0.225], dtype=np.float32)
    resized = cv2.cvtColor(resized, cv2.COLOR_BGR2RGB).astype(np.float32)
    resized = resized / 255.0
    resized = (resized - mean) / std
    tensor = torch.from_numpy(resized.transpose(2, 0, 1)).unsqueeze(0).float().to(device)

    with torch.no_grad():
        output = model(tensor)
    return output.squeeze(0).cpu().numpy()


# Pares de articulaciones simétricas (índices COCO, base 0) para flip correcto
FLIP_PAIRS = [(1, 2), (3, 4), (5, 6), (7, 8), (9, 10), (11, 12), (13, 14), (15, 16)]


def flip_heatmap(heatmap: np.ndarray) -> np.ndarray:
    """Voltea un heatmap (C, H, W) horizontalmente e intercambia canales izq↔der."""
    flipped = heatmap[:, :, ::-1].copy()
    for left, right in FLIP_PAIRS:
        flipped[left], flipped[right] = flipped[right].copy(), flipped[left].copy()
    return flipped


def main():
    parser = argparse.ArgumentParser(description='Preparar datos GT para fine-tuning de HRNet')
    parser.add_argument('--input-dir', type=str, default='/nas/antoniodetoro/datasets/qwen/hr_net_sucias/test',
                        help='Carpeta con todas las imágenes de entrada')
    parser.add_argument('--output-dir', type=str, default='/nas/antoniodetoro/datasets/qwen/hr_net_sucias_npy/test',
                        help='Carpeta de salida plana (imágenes + .npy juntos). '
                             'Ejecutar una vez por conjunto (train_val / test).')
    parser.add_argument('--model-path', type=str,
                        default='./models/pose_hrnet_w48_384x288.pth',
                        help='Modelo HRNet para generar heatmaps GT')
    args = parser.parse_args()

    input_dir  = Path(args.input_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    logger.info(f"Dispositivo: {device}")

    # ── 1. Clasificar imágenes ────────────────────────────────────────────────
    valid_ext = {'.png', '.jpg', '.jpeg'}
    all_files = sorted(f for f in input_dir.glob('*') if f.suffix.lower() in valid_ext)

    # Regex unificado: {base}_rgb_{variant}[_flip]
    # variant: clean | clean_aug1 | clean_aug2 | ... | noise_3.14
    pat = re.compile(r'^(.+)_rgb_(clean(?:_aug\d+)?|noise_[\d.]+)(_flip)?$')

    clean_map  = {}   # base → path de {base}_rgb_clean.png
    train_list = []   # (path, base, is_flip)

    for f in all_files:
        m = pat.match(f.stem)
        if not m:
            logger.warning(f"Nombre no reconocido, se omite: {f.name}")
            continue
        base    = m.group(1)
        variant = m.group(2)
        is_flip = m.group(3) is not None

        if variant == 'clean' and not is_flip:
            clean_map[base] = f
        train_list.append((f, base, is_flip))

    logger.info(f"Imágenes limpias (fuente GT): {len(clean_map)}")
    logger.info(f"Imágenes de entrenamiento totales: {len(train_list)}")

    # Filtrar solo las que tienen clean correspondiente
    train_list = [(p, b, flp) for p, b, flp in train_list if b in clean_map]
    bases_needed = sorted(clean_map.keys())
    logger.info(f"Imágenes válidas (con GT): {len(train_list)}")

    if not train_list:
        logger.error("No se encontraron pares válidos. Revisa --input-dir.")
        return

    # ── 2. Generar heatmaps GT (regular y flip) ───────────────────────────────
    logger.info("Cargando modelo HRNet...")
    model = load_model(args.model_path, device)

    logger.info("Generando heatmaps GT a partir de imágenes *_rgb_clean.png...")
    for base in tqdm(bases_needed, desc='Heatmaps GT'):
        clean_path    = clean_map[base]
        npy_path      = output_dir / f"{base}_rgb.npy"
        npy_flip_path = output_dir / f"{base}_rgb_flip.npy"

        if npy_path.exists() and npy_flip_path.exists():
            continue  # ya generados

        image = cv2.imread(str(clean_path))
        if image is None:
            logger.warning(f"No se pudo leer: {clean_path}")
            continue

        heatmap = get_heatmaps(model, image, device)
        np.save(str(npy_path), heatmap)
        np.save(str(npy_flip_path), flip_heatmap(heatmap))

    logger.info(f"✓ Heatmaps GT guardados en: {output_dir}")

    # ── 3. Copiar imágenes de entrenamiento ───────────────────────────────────
    # logger.info("Copiando imágenes de entrenamiento...")
    # for src_path, _base, _flip in tqdm(train_list, desc='Copiando imgs'):
    #     dst_path = output_dir / src_path.name
    #     if not dst_path.exists():
    #         shutil.copy2(str(src_path), str(dst_path))

    # logger.info(f"✓ Imágenes copiadas en: {output_dir}")

    # ── Resumen ───────────────────────────────────────────────────────────────
    logger.info("=" * 60)
    logger.info("PREPARACIÓN COMPLETADA")
    logger.info(f"  Imágenes de entrenamiento: {len(train_list)}")
    logger.info(f"  Heatmaps GT (regular + flip): {len(bases_needed) * 2}")
    logger.info(f"  Directorio de salida: {output_dir}")
    logger.info("=" * 60)
    logger.info("Para entrenar ejecuta:")
    logger.info("  python 1_finetuning_hr_net_pose.py --dataset-dir dataset_hrnet/")


if __name__ == '__main__':
    main()
