#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
HRNet Fine-Tuning Script
==========================================================
Descripción:
    Script para re-entrenar (fine-tune) el modelo HRNet utilizando
    imágenes ensuciadas (con ruido) y sus heatmaps originales generados
    previamente.

Asume que:
    - Ruta 1: Imágenes .jpg/.png (Ej: 00_15_1680259460407_rgb_noise_0.4.jpg)
    - Ruta 2: Heatmaps .npy (Ej: 00_15_1680259460407_rgb.npy)
"""

import os
import argparse
import csv
import logging
import re
from pathlib import Path

import cv2
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader, random_split
from tqdm import tqdm

# Importamos la arquitectura desde tu script original
# Asegúrate de que tu archivo subido se llame 'hrnet_inference.py'
try:
    from hrnet_inference import PoseHRNet
except ImportError:
    raise ImportError("No se pudo importar 'PoseHRNet'. Asegúrate de que el script original se llama 'hrnet_inference.py' y está en esta misma carpeta.")

# Configuración de logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


class NoiseHeatmapDataset(Dataset):
    """
    Dataset personalizado para mapear imágenes con ruido a sus heatmaps originales.
    """
    def __init__(self, imgs_dir: str, heatmaps_dir: str, input_size=(288, 384)):
        self.imgs_dir = Path(imgs_dir)
        self.heatmaps_dir = Path(heatmaps_dir)
        self.input_size = input_size
        
        # Buscar todas las imágenes válidas en ruta1
        valid_extensions = {'.jpg', '.jpeg', '.png'}
        self.img_paths = [p for p in self.imgs_dir.rglob('*') if p.suffix.lower() in valid_extensions]
        
        if not self.img_paths:
            raise ValueError(f"No se encontraron imágenes en {imgs_dir}")
            
        # Valores de normalización estándar de ImageNet (usados en HRNet)
        self.mean = np.array([0.485, 0.456, 0.406], dtype=np.float32)
        self.std = np.array([0.229, 0.224, 0.225], dtype=np.float32)

    def __len__(self):
        return len(self.img_paths)

    def __getitem__(self, idx):
        img_path = self.img_paths[idx]
        
        # LÓGICA DE EMPAREJAMIENTO
        # {base}_rgb_clean.png           → {base}_rgb.npy
        # {base}_rgb_clean_flip.png      → {base}_rgb_flip.npy
        # {base}_rgb_clean_aug1.png      → {base}_rgb.npy
        # {base}_rgb_clean_aug1_flip.png → {base}_rgb_flip.npy
        # {base}_rgb_noise_X.XX.png      → {base}_rgb.npy
        stem = img_path.stem
        m = re.match(r'^(.+)_rgb_(clean(?:_aug\d+)?|noise_[\d.]+)(_flip)?$', stem)
        if m is None:
            raise ValueError(f"Nombre de imagen no reconocido: {img_path.name}")
        base    = m.group(1)
        is_flip = m.group(3) is not None
        npy_name     = f"{base}_rgb_flip.npy" if is_flip else f"{base}_rgb.npy"
        heatmap_path = self.heatmaps_dir / npy_name
        
        if not heatmap_path.exists():
            raise FileNotFoundError(f"Falta el heatmap esperado: {heatmap_path}")

        # 1. Cargar y preprocesar imagen
        image = cv2.imread(str(img_path))
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        h, w = self.input_size
        image = cv2.resize(image, (w, h), interpolation=cv2.INTER_LINEAR)
        image = image.astype(np.float32) / 255.0
        image = (image - self.mean) / self.std
        
        # Formato canal primero (C, H, W)
        image_tensor = torch.from_numpy(image.transpose(2, 0, 1)).float()

        # 2. Cargar heatmap (.npy)
        heatmap = np.load(heatmap_path)
        heatmap_tensor = torch.from_numpy(heatmap).float()

        return image_tensor, heatmap_tensor


def load_pretrained_weights(model, weights_path, device):
    """Carga los pesos iniciales ignorando errores de 'weights_only'."""
    logger.info(f"Cargando pesos pre-entrenados desde {weights_path}")
    try:
        state_dict = torch.load(weights_path, map_location=device, weights_only=False)
    except TypeError:
        state_dict = torch.load(weights_path, map_location=device)
        
    if isinstance(state_dict, dict) and 'state_dict' in state_dict:
        state_dict = state_dict['state_dict']
        
    model.load_state_dict(state_dict, strict=True)
    return model


def evaluate(model, dataloader, criterion, device):
    """Evalúa el modelo en un dataloader y devuelve la pérdida promedio."""
    model.eval()
    total_loss = 0.0
    with torch.no_grad():
        for images, heatmaps in dataloader:
            images   = images.to(device, non_blocking=True)
            heatmaps = heatmaps.to(device, non_blocking=True)
            outputs  = model(images)
            loss     = criterion(outputs, heatmaps)
            total_loss += loss.item()
    return total_loss / len(dataloader)


def main():
    parser = argparse.ArgumentParser(description='Fine-Tuning de HRNet')
    parser.add_argument('--imgs-dir', type=str, required=True,
                        help='Directorio base de las imágenes (debe contener train_val/ y test/)')
    parser.add_argument('--heatmaps-dir', type=str, required=True,
                        help='Directorio base de los heatmaps .npy (debe contener train_val/ y test/)')
    parser.add_argument('--model-path', type=str, default='./models/pose_hrnet_w48_384x288.pth',
                        help='Ruta al modelo base para inicializar los pesos')
    parser.add_argument('--output-model', type=str, default='./models/hrnet_finetuned_best.pth',
                        help='Ruta donde guardar el mejor modelo (checkpoint por val_loss)')
    parser.add_argument('--output-stats', type=str, default='./training_stats.csv',
                        help='Ruta donde guardar las estadísticas de entrenamiento (CSV)')
    parser.add_argument('--epochs', type=int, default=100, help='Número máximo de épocas')
    parser.add_argument('--batch-size', type=int, default=12, help='Tamaño del batch')
    parser.add_argument('--lr', type=float, default=1e-4, help='Learning rate')
    parser.add_argument('--val-split', type=float, default=0.2,
                        help='Fracción de train_val destinada a validación (por defecto 0.2 = 20%%)')
    parser.add_argument('--early-stopping', type=int, default=10,
                        help='Épocas sin mejora en val_loss antes de detener el entrenamiento')
    parser.add_argument('--seed', type=int, default=42,
                        help='Semilla aleatoria para el split train/val reproducible')
    parser.add_argument('--gpus', type=int, nargs='+', default=None,
                        help='IDs de GPU a usar para DataParallel (ej: --gpus 0 1). '
                             'Si no se indica, se usa la GPU disponible o CPU.')

    args = parser.parse_args()

    # ── Definición de rutas separadas para imágenes y heatmaps ──
    imgs_dir = Path(args.imgs_dir)
    heatmaps_dir = Path(args.heatmaps_dir)

    train_imgs_dir = imgs_dir / 'train_val'
    train_heatmaps_dir = heatmaps_dir / 'train_val'
    
    test_imgs_dir = imgs_dir / 'test'
    test_heatmaps_dir = heatmaps_dir / 'test'

    # ── Dispositivo y paralelismo ──────────────────────────────────────────
    use_parallel = False
    if args.gpus is not None and len(args.gpus) > 1 and torch.cuda.is_available():
        device = torch.device(f'cuda:{args.gpus[0]}')
        use_parallel = True
        logger.info(f"Modo DataParallel activado en GPUs: {args.gpus}")
    elif torch.cuda.is_available():
        gpu_id = args.gpus[0] if args.gpus else 0
        device = torch.device(f'cuda:{gpu_id}')
        logger.info(f"Usando GPU: {gpu_id}")
    else:
        device = torch.device('cpu')
    logger.info(f"Iniciando fine-tuning en dispositivo: {device}")

    # ── 1. Datasets ──────────────────────────────────────────────────────────
    logger.info("Cargando dataset train_val...")
    full_trainval = NoiseHeatmapDataset(
        imgs_dir=str(train_imgs_dir),
        heatmaps_dir=str(train_heatmaps_dir),
        input_size=(288, 384),
    )

    n_total = len(full_trainval)
    n_val   = int(args.val_split * n_total)
    n_train = n_total - n_val
    logger.info(f"Total train_val: {n_total}  →  train: {n_train}, val: {n_val}")

    generator = torch.Generator().manual_seed(args.seed)
    train_dataset, val_dataset = random_split(full_trainval, [n_train, n_val], generator=generator)

    logger.info("Cargando dataset test...")
    test_dataset = NoiseHeatmapDataset(
        imgs_dir=str(test_imgs_dir),
        heatmaps_dir=str(test_heatmaps_dir),
        input_size=(288, 384),
    )
    logger.info(f"Total test: {len(test_dataset)}")

    num_workers  = 4
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True,
                              num_workers=num_workers, pin_memory=True)
    val_loader   = DataLoader(val_dataset,   batch_size=args.batch_size, shuffle=False,
                              num_workers=num_workers, pin_memory=True)
    test_loader  = DataLoader(test_dataset,  batch_size=args.batch_size, shuffle=False,
                              num_workers=num_workers, pin_memory=True)

    # ── 2. Modelo ────────────────────────────────────────────────────────────
    model = PoseHRNet(width=48, num_joints=17)
    if Path(args.model_path).exists():
        model = load_pretrained_weights(model, args.model_path, device)
    else:
        logger.warning(f"No se encontró {args.model_path}. Entrenando desde cero.")
    model = model.to(device)
    if use_parallel:
        model = nn.DataParallel(model, device_ids=args.gpus)
        logger.info(f"Modelo envuelto en DataParallel con device_ids={args.gpus}")

    # ── 3. Función de pérdida y optimizador ──────────────────────────────────
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)

    # ── 4. Variables de seguimiento ───────────────────────────────────────────
    stats            = []
    best_val_loss    = float('inf')
    patience_counter = 0
    output_path = Path(args.output_model)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    stats_path = Path(args.output_stats)
    stats_path.parent.mkdir(parents=True, exist_ok=True)

    # ── 5. Bucle de entrenamiento ─────────────────────────────────────────────
    logger.info("Comenzando el entrenamiento...")
    for epoch in range(args.epochs):
        # — Train —
        model.train()
        running_loss = 0.0
        progress_bar = tqdm(train_loader, desc=f"Época {epoch + 1}/{args.epochs} [train]")
        for images, heatmaps in progress_bar:
            images   = images.to(device, non_blocking=True)
            heatmaps = heatmaps.to(device, non_blocking=True)

            outputs = model(images)
            loss    = criterion(outputs, heatmaps)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            running_loss += loss.item()
            progress_bar.set_postfix({'loss': f'{loss.item():.6f}'})

        avg_train_loss = running_loss / len(train_loader)

        # — Validation —
        avg_val_loss = evaluate(model, val_loader, criterion, device)

        logger.info(
            f"Época [{epoch + 1}/{args.epochs}] — "
            f"Train Loss: {avg_train_loss:.6f} | Val Loss: {avg_val_loss:.6f}"
        )

        # — Guardar estadísticas de esta época —
        stats.append({
            'epoch':      epoch + 1,
            'train_loss': avg_train_loss,
            'val_loss':   avg_val_loss,
        })

        # — Checkpoint: guardar solo si val_loss mejora —
        if avg_val_loss < best_val_loss:
            best_val_loss    = avg_val_loss
            patience_counter = 0
            raw_model = model.module if isinstance(model, nn.DataParallel) else model
            torch.save({
                'epoch':      epoch + 1,
                'state_dict': raw_model.state_dict(),
                'optimizer':  optimizer.state_dict(),
                'train_loss': avg_train_loss,
                'val_loss':   avg_val_loss,
            }, str(output_path))
            logger.info(f"  → Mejor modelo guardado (val_loss={best_val_loss:.6f})")
        else:
            patience_counter += 1
            logger.info(
                f"  → Sin mejora en val_loss. "
                f"Paciencia: {patience_counter}/{args.early_stopping}"
            )
            if patience_counter >= args.early_stopping:
                logger.info("Early stopping activado. Finalizando entrenamiento anticipado.")
                break

    # ── 6. Guardar estadísticas en CSV ────────────────────────────────────────
    with open(stats_path, 'w', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=['epoch', 'train_loss', 'val_loss'])
        writer.writeheader()
        writer.writerows(stats)
    logger.info(f"Estadísticas de entrenamiento guardadas en: {stats_path}")

    # ── 7. Evaluación final en el conjunto de test ────────────────────────────
    logger.info("Evaluando el mejor modelo en el conjunto de test...")
    try:
        best_ckpt = torch.load(str(output_path), map_location=device, weights_only=False)
    except TypeError:
        best_ckpt = torch.load(str(output_path), map_location=device)
    raw_model = model.module if isinstance(model, nn.DataParallel) else model
    raw_model.load_state_dict(best_ckpt['state_dict'])
    test_loss = evaluate(raw_model, test_loader, criterion, device)
    logger.info(f"Test Loss (mejor modelo): {test_loss:.6f}")

    # Añadir resultado de test al CSV
    with open(stats_path, 'a', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=['epoch', 'train_loss', 'val_loss'])
        writer.writerow({'epoch': 'test', 'train_loss': '', 'val_loss': test_loss})

    logger.info("✅ Proceso finalizado.")


if __name__ == '__main__':
    main()