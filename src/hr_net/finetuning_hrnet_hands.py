#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
HRNet Fine-Tuning Script - 23 Keypoints (17 body + 6 hands)
==========================================================
Descripción:
    Script para re-entrenar (fine-tune) el modelo HRNet extendiendo
    de 17 a 23 keypoints, añadiendo puntos de manos sin perder
    conocimiento previo (evitando catastrophic forgetting).


Ejecución:
python finetuning_hrnet_hands.py --imgs-dir /nas/antoniodetoro/datasets/qwen/hr_net_with_hands/images
 --heatmaps-dir /nas/antoniodetoro/datasets/qwen/hr_net_with_hands/heatmaps/ 
 --model-path /nas/antoniodetoro/qwen/Qwen-Image-Edit-Angles-2/src/models/pose_hrnet_w48_384x288.pth 
 --output-model /nas/antoniodetoro/qwen/Qwen-Image-Edit-Angles-2/src/hr_net/outputs/2/hr_net_23kp_best.pth  
 --output-stats /nas/antoniodetoro/qwen/Qwen-Image-Edit-Angles-2/src/hr_net/outputs/2/training_stats.csv --epochs 100
 --lr-backbone 1e-5 --lr-head 1e-3  --alpha 3.0 --early-stopping 15 --freeze-until stage4  --augment  --num-brightness-augs 1 
 --gpus 1  --batch-size 128 --data-subset 1.0

Características:
    - Carga pesos preentrenados (17 kp) y extiende a 23
    - Congelación parcial del backbone
    - Loss ponderada (body vs hands)
    - LR diferenciados por capas
    - Augmentación: iluminación + flip horizontal
    - Métricas completas: train/val/test loss + accuracy
    - Notificaciones Telegram cuando mejora val_loss
    - Subsampling de datos para pruebas rápidas
"""

import os
import argparse
import csv
import logging
import re
from pathlib import Path
from typing import Optional

import cv2
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader, random_split, Subset
from tqdm import tqdm
from dotenv import load_dotenv
import urllib.request
import urllib.parse
import json

# Importar arquitectura HRNet
try:
    from hrnet_inference import PoseHRNet
except ImportError:
    raise ImportError("No se pudo importar 'PoseHRNet'. Asegúrate de que el script original se llama 'hrnet_inference.py' y está en esta misma carpeta.")

# Configuración de logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


# ══════════════════════════════════════════════════════════════════════════════
# TELEGRAM NOTIFIER
# ══════════════════════════════════════════════════════════════════════════════
def send_telegram(
    message: str,
    token: Optional[str] = None,
    chat_id: Optional[str] = None,
    parse_mode: str = "HTML",
) -> bool:
    """Envía un mensaje por Telegram Bot API. Devuelve True si tuvo éxito."""
    token = token or os.environ.get("TELEGRAM_BOT_TOKEN")
    chat_id = chat_id or os.environ.get("TELEGRAM_CHAT_ID")
    
    if not token or not chat_id:
        return False
    
    url = f"https://api.telegram.org/bot{token}/sendMessage"
    payload = urllib.parse.urlencode({
        "chat_id": chat_id,
        "text": message,
        "parse_mode": parse_mode,
    }).encode("utf-8")
    
    _preview = message.splitlines()[0][:80].replace("<b>", "").replace("</b>", "").replace("<code>", "").replace("</code>", "")
    
    try:
        req = urllib.request.Request(url, data=payload, method="POST")
        with urllib.request.urlopen(req, timeout=10) as resp:
            result = json.loads(resp.read().decode())
            if not result.get("ok"):
                logger.warning(f"[TELEGRAM] ✗ API respondió ok=false: {result}")
                print(f"[TELEGRAM] ✗ Fallo al enviar: {_preview!r}")
                return False
        logger.info(f"[TELEGRAM] ✓ Enviado: {_preview!r}")
        print(f"[TELEGRAM] ✓ {_preview}")
        return True
    except Exception as e:
        logger.warning(f"[TELEGRAM] ✗ Error: {e} | msg={_preview!r}")
        print(f"[TELEGRAM] ✗ Error enviando: {e}")
        return False


def notify_new_best_val(
    epoch: int,
    val_loss: float,
    prev_best: float,
    val_acc: float,
    test_loss: float,
    test_acc: float,
    token: Optional[str] = None,
    chat_id: Optional[str] = None,
    extra_context: str = "",
) -> bool:
    """Notificación de nuevo mejor val_loss."""
    improvement = prev_best - val_loss
    msg = (
        f"🏆 <b>Nuevo mejor modelo</b>\n"
        f"{extra_context}"
        f"Val Loss: <b>{val_loss:.6f}</b> (-{improvement:.6f})\n"
        f"Val Acc: {val_acc:.2f}%\n"
        f"Test Loss: {test_loss:.6f}\n"
        f"Test Acc: {test_acc:.2f}%\n"
        f"Época: {epoch}"
    )
    return send_telegram(msg, token=token, chat_id=chat_id)


# ══════════════════════════════════════════════════════════════════════════════
# DATASET CON AUGMENTACIÓN
# ══════════════════════════════════════════════════════════════════════════════
class NoiseHeatmapDataset(Dataset):
    """
    Dataset con augmentación:
    - Cambios de iluminación (brillo/contraste)
    - Flip horizontal (con heatmap correspondiente)
    """
    def __init__(self, imgs_dir: str, heatmaps_dir: str, input_size=(288, 384), 
                 augment=True, num_brightness_augs=2):
        self.imgs_dir = Path(imgs_dir)
        self.heatmaps_dir = Path(heatmaps_dir)
        self.input_size = input_size
        self.augment = augment
        self.num_brightness_augs = num_brightness_augs
        
        valid_extensions = {'.jpg', '.jpeg', '.png'}
        all_img_paths = [p for p in self.imgs_dir.rglob('*') if p.suffix.lower() in valid_extensions]
        
        if not all_img_paths:
            raise ValueError(f"No se encontraron imágenes en {imgs_dir}")
        
        self.img_paths = all_img_paths  # temporal para que _validate_pairs funcione
        valid_indices, n_total = self._validate_pairs()
        self.img_paths = [all_img_paths[i] for i in valid_indices]
        self._n_total_original = n_total
        self._n_valid = len(self.img_paths)
        
        if not self.img_paths:
            raise ValueError(f"No hay pares válidos en {imgs_dir}")
        
        # Valores de normalización ImageNet
        self.mean = np.array([0.485, 0.456, 0.406], dtype=np.float32)
        self.std = np.array([0.229, 0.224, 0.225], dtype=np.float32)
        
        # Multiplicador de samples por augmentación
        if self.augment:
            self.samples_per_img = (1 + self.num_brightness_augs) * 2
        else:
            self.samples_per_img = 1

    def _validate_pairs(self):
          
            # Valida todos los pares imagen/heatmap. Descarta los que:
            # - No tienen heatmap correspondiente
            # - El heatmap no tiene shape (23, H, W)
            # - Algún canal del heatmap está completamente vacío (suma == 0)
            
            # Devuelve lista de índices válidos de self.img_paths.
            
        valid_indices = []
        n_total = len(self.img_paths)
        
        for i, img_path in enumerate(tqdm(self.img_paths, desc="Validando pares", leave=False)):
            stem = img_path.stem
            m = re.match(r'^(.+)_rgb_(clean(?:_aug\d+)?|noise_[\d.]+)(_flip)?$', stem)
            if m is not None:
                base = m.group(1)
                is_flip = m.group(3) is not None
                npy_name = f"{base}_rgb_flip.npy" if is_flip else f"{base}_rgb.npy"
            else:
                if stem.endswith('_rgb'):
                    npy_name = f"{stem}.npy"
                else:
                    continue  # nombre no reconocido

            heatmap_path = self.heatmaps_dir / npy_name
            if not heatmap_path.exists():
                continue

            try:
                heatmap = np.load(heatmap_path)
            except Exception:
                continue

            # Validar shape
            if heatmap.ndim != 3 or heatmap.shape[0] != 23:
                continue

            # Validar que ningún canal esté completamente vacío
            if np.any(heatmap.sum(axis=(1, 2)) == 0):
                continue

            valid_indices.append(i)

        return valid_indices, n_total

    def __len__(self):
        return len(self.img_paths) * self.samples_per_img

    def __getitem__(self, idx):
        img_idx = idx // self.samples_per_img
        aug_variant = idx % self.samples_per_img
    
        img_path = self.img_paths[img_idx]

        
        # Emparejamiento con heatmap - VERSIÓN SIMPLIFICADA
        stem = img_path.stem
        
        # Intentar primero el patrón complejo (imágenes con sufijos)
        m = re.match(r'^(.+)_rgb_(clean(?:_aug\d+)?|noise_[\d.]+)(_flip)?$', stem)
        
        if m is not None:
            # Imagen con sufijos: {base}_rgb_{clean|noise_X.XX}[_flip]
            base = m.group(1)
            is_flip = m.group(3) is not None
            npy_name = f"{base}_rgb_flip.npy" if is_flip else f"{base}_rgb.npy"
        else:
            # Imagen simple: {base}_rgb.png
            # Intentar patrón simple
            if stem.endswith('_rgb'):
                base = stem  # El stem completo es la base
                npy_name = f"{base}.npy"
            else:
                raise ValueError(f"Nombre de imagen no reconocido: {img_path.name}")
        
        heatmap_path = self.heatmaps_dir / npy_name
        
        if not heatmap_path.exists():
            raise FileNotFoundError(f"Falta el heatmap esperado: {heatmap_path}")

        # Cargar imagen
        image = cv2.imread(str(img_path))
        if image is None:
            raise ValueError(f"No se pudo cargar la imagen: {img_path}")
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        # Cargar heatmap 
        heatmap = np.load(heatmap_path).astype(np.float32)
        
        # APLICAR AUGMENTACIÓN SOLO SI ESTÁ ACTIVADA
        if self.augment:
            # Aplicar augmentación según variant
            do_flip = aug_variant >= (self.samples_per_img // 2)
            brightness_idx = aug_variant % (self.samples_per_img // 2)
            
            if brightness_idx > 0:
                # Cambio de brillo/contraste
                alpha = np.random.uniform(0.7, 1.3)
                beta = np.random.randint(-30, 30)
                image = cv2.convertScaleAbs(image, alpha=alpha, beta=beta)
            
            if do_flip:
                # Flip horizontal
                image = cv2.flip(image, 1)
                heatmap = self._flip_heatmap(heatmap)
        
        # Resize y normalización (siempre se hace)
        h, w = self.input_size
        image = cv2.resize(image, (w, h), interpolation=cv2.INTER_LINEAR)
        image = image.astype(np.float32) / 255.0
        image = (image - self.mean) / self.std
        
        image_tensor = torch.from_numpy(image.transpose(2, 0, 1)).float()
        heatmap_tensor = torch.from_numpy(heatmap).float()

        return image_tensor, heatmap_tensor

    def _flip_heatmap(self, heatmap):
        """Flip horizontal del heatmap (23 canales)."""
        # Flip espacial
        flipped = np.flip(heatmap, axis=2).copy()
        
        # Swap de keypoints simétricos (HRNet COCO)
        # Índices: [left, right] pares que se intercambian
        swap_pairs = [
        (1, 2),   # ojos
        (3, 4),   # orejas
        (5, 6),   # hombros
        (7, 8),   # codos
        (9, 10),  # muñecas
        (11, 12), # caderas
        (13, 14), # rodillas
        (15, 16), # tobillos
    ]
        
        for left, right in swap_pairs:
            flipped[[left, right]] = flipped[[right, left]]
        
        # Para keypoints de manos (17-22), asumimos que también son simétricos
        # [17, 18, 19] mano izquierda → [20, 21, 22] mano derecha
        # flipped[[17, 18, 19, 20, 21, 22]] = flipped[[20, 21, 22, 17, 18, 19]] ESTO ESTÁ MAL

        flipped[[17, 19, 21, 18, 20, 22]] = flipped[[18, 20, 22, 17, 19, 21]]
        
        return flipped


def subsample_dataset(dataset, fraction, seed=42):
    """Crea un Subset con fracción de datos del dataset original."""
    if fraction is None or fraction >= 1.0:
        return dataset
    
    n_total = len(dataset)
    n_subset = max(1, int(n_total * fraction))
    
    generator = torch.Generator().manual_seed(seed)
    indices = torch.randperm(n_total, generator=generator)[:n_subset].tolist()
    
    return Subset(dataset, indices)


# ══════════════════════════════════════════════════════════════════════════════
# MODELO EXTENDIDO A 23 KEYPOINTS
# ══════════════════════════════════════════════════════════════════════════════
def load_pretrained_and_extend(model, weights_path, device, num_old_joints=17, num_new_joints=23):
    """
    Carga pesos preentrenados (17 kp) y extiende a 23 keypoints.
    
    - Copia pesos compatibles del backbone
    - Para final_layer:
        * Copia pesos de los 17 canales originales
        * Inicializa 6 nuevos canales con Kaiming normal
    """
    logger.info(f"Cargando pesos pre-entrenados desde {weights_path}")
    try:
        checkpoint = torch.load(weights_path, map_location=device, weights_only=False)
    except TypeError:
        checkpoint = torch.load(weights_path, map_location=device)
        
    if isinstance(checkpoint, dict) and 'state_dict' in checkpoint:
        state_dict_old = checkpoint['state_dict']
    else:
        state_dict_old = checkpoint
    
    # Cargar estado actual del modelo (23 kp)
    state_dict_new = model.state_dict()
    
    # Copiar pesos compatibles
    for name, param in state_dict_old.items():
        if name in state_dict_new:
            if state_dict_new[name].shape == param.shape:
                state_dict_new[name] = param
            elif 'final_layer' in name:
                # Extender final_layer
                if 'weight' in name:
                    # state_dict_old[name]: [17, C, 1, 1]
                    # state_dict_new[name]: [23, C, 1, 1]
                    state_dict_new[name][:num_old_joints] = param
                    # Inicializar nuevos canales
                    nn.init.kaiming_normal_(state_dict_new[name][num_old_joints:])
                    logger.info(f"  Extendido {name}: {param.shape} → {state_dict_new[name].shape}")
                elif 'bias' in name:
                    # [17] → [23]
                    state_dict_new[name][:num_old_joints] = param
                    nn.init.zeros_(state_dict_new[name][num_old_joints:])
                    logger.info(f"  Extendido {name}: {param.shape} → {state_dict_new[name].shape}")
            else:
                logger.warning(f"  Saltando {name}: shape mismatch {param.shape} vs {state_dict_new[name].shape}")
    
    model.load_state_dict(state_dict_new, strict=True)
    logger.info("Pesos cargados y extendidos correctamente")
    return model


def freeze_backbone_partial(model, freeze_until='stage4'):
    """
    Congela backbone hasta cierta etapa.
    
    - 'stage4': Congela TODO excepto final_layer (más conservador)
    - 'stage3': Congela hasta stage3, entrena stage4 + final_layer
    - 'stage2': Congela hasta stage2
    - 'none': No congela nada
    """
    if freeze_until == 'stage4':
        freeze_stages = ['conv1', 'bn1', 'conv2', 'bn2', 'layer1', 
                        'transition1', 'stage2', 'transition2', 'stage3', 
                        'transition3', 'stage4']
    elif freeze_until == 'stage3':
        freeze_stages = ['conv1', 'bn1', 'conv2', 'bn2', 'layer1', 
                        'transition1', 'stage2', 'transition2', 'stage3', 
                        'transition3']
    elif freeze_until == 'stage2':
        freeze_stages = ['conv1', 'bn1', 'conv2', 'bn2', 'layer1', 
                        'transition1', 'stage2', 'transition2']
    elif freeze_until == 'none':
        freeze_stages = []
    else:
        raise ValueError(f"freeze_until inválido: {freeze_until}")
    
    for name, param in model.named_parameters():
        for stage in freeze_stages:
            if name.startswith(stage):
                param.requires_grad = False
                break
    
    # Contar parámetros congelados/entrenables
    frozen = sum(p.numel() for p in model.parameters() if not p.requires_grad)
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    logger.info(f"Parámetros congelados: {frozen:,} | Entrenables: {trainable:,}")


# ══════════════════════════════════════════════════════════════════════════════
# MÉTRICAS: ACCURACY (PCK - Percentage of Correct Keypoints)
# ══════════════════════════════════════════════════════════════════════════════
def compute_pck(pred_heatmaps, gt_heatmaps, threshold=0.05):
    """
    Calcula PCK: porcentaje de keypoints predichos dentro de threshold.
    
    Criterio: argmax de heatmap predicho debe estar cerca del gt.
    """
    B, C, H, W = pred_heatmaps.shape
    
    # Extraer coordenadas de máximos
    pred_coords = []
    gt_coords = []
    
    for b in range(B):
        for c in range(C):
            pred_hm = pred_heatmaps[b, c].cpu().numpy()
            gt_hm = gt_heatmaps[b, c].cpu().numpy()
            
            # Argmax
            pred_y, pred_x = np.unravel_index(pred_hm.argmax(), pred_hm.shape)
            gt_y, gt_x = np.unravel_index(gt_hm.argmax(), gt_hm.shape)
            
            pred_coords.append([pred_x, pred_y])
            gt_coords.append([gt_x, gt_y])
    
    pred_coords = np.array(pred_coords)
    gt_coords = np.array(gt_coords)
    
    # Distancia euclidiana normalizada por tamaño del heatmap
    distances = np.linalg.norm(pred_coords - gt_coords, axis=1)
    normalized_dist = distances / np.sqrt(H**2 + W**2)
    
    # Porcentaje dentro del threshold
    correct = (normalized_dist < threshold).sum()
    total = len(distances)
    
    pck = 100.0 * correct / total
    return pck


import io
import base64
from PIL import Image

# Añadir esta función después de compute_pck:

def visualize_predictions(model, dataset, device, indices=[0, 1], input_size=(288, 384)):
    """
    Genera visualizaciones de predicciones sobre imágenes específicas del dataset.
    
    Args:
        model: Modelo a evaluar
        dataset: Dataset de donde sacar las imágenes
        device: Device donde está el modelo
        indices: Índices de las imágenes a visualizar
        input_size: Tamaño de entrada del modelo (H, W)
        
    Returns:
        Lista de imágenes PIL con keypoints superpuestos
    """
    model.eval()
    
    # Colores para diferentes tipos de keypoints
    # Body (0-16): azul, Hands (17-22): rojo
    colors_body = [(0, 255, 0)] * 17  # Verde para body
    colors_hands = [(255, 0, 0)] * 6   # Rojo para hands
    colors = colors_body + colors_hands
    
    # Conexiones de skeleton COCO (body)
    skeleton_body = [
        (0, 1), (0, 2),  # nariz -> ojos
        (1, 3), (2, 4),  # ojos -> orejas
        # (0, 5), (0, 6),  # nariz -> hombros (ESTA CONEXIÓN NO EXISTE EN COCO)
        (5, 7), (7, 9),  # brazo izq
        (6, 8), (8, 10), # brazo der
        (5, 6),          # hombros
        (5, 11), (6, 12),# hombros -> caderas
        (11, 12),        # caderas
        (11, 13), (13, 15), # pierna izq
        (12, 14), (14, 16), # pierna der
    ]
    
    # Conexiones de manos (simplificado: muñeca -> puntos de mano)
    # skeleton_hands = [
    #     (5, 17), (5, 18), (5, 19),  # muñeca izq -> dedos
    #     (6, 20), (6, 21), (6, 22),  # muñeca der -> dedos
    # ]

    skeleton_hands = [
        (9, 17), (9, 19), (9, 21),     # muñeca izq (COCO / HR NET) -> dedos(Mediapipe)
        (10, 18), (10, 20), (10, 22),  # muñeca der (COCO / HR NET) -> dedos(Mediapipe)
    ]
    
    skeleton = skeleton_body + skeleton_hands
    
    vis_images = []
    
    with torch.no_grad():
        for idx in indices:
            img_tensor, heatmap_gt = dataset[idx]
            img_batch = img_tensor.unsqueeze(0).to(device)
            heatmap_pred = model(img_batch)[0]
            
            img_np = img_tensor.cpu().numpy().transpose(1, 2, 0)
            mean = np.array([0.485, 0.456, 0.406])
            std = np.array([0.229, 0.224, 0.225])
            img_np = (img_np * std + mean) * 255.0
            img_np = np.clip(img_np, 0, 255).astype(np.uint8)
            img_vis = cv2.cvtColor(img_np, cv2.COLOR_RGB2BGR)
            
            H_hm, W_hm = heatmap_pred.shape[1:]
            H_img, W_img = img_np.shape[:2]
            
            keypoints = []
            for c in range(23):
                hm = heatmap_pred[c].cpu().numpy()
                y, x = np.unravel_index(hm.argmax(), hm.shape)
                x_img = int(x * W_img / W_hm)
                y_img = int(y * H_img / H_hm)
                conf = hm[y, x]
                keypoints.append((x_img, y_img, conf))
            
            for i, j in skeleton:
                if i < len(keypoints) and j < len(keypoints):
                    x1, y1, conf1 = keypoints[i]
                    x2, y2, conf2 = keypoints[j]
                    if conf1 > 0.1 and conf2 > 0.1:
                        color = (0, 255, 0) if i < 17 and j < 17 else (255, 0, 0)
                        cv2.line(img_vis, (x1, y1), (x2, y2), color, 2)
            
            for kp_idx, (x, y, conf) in enumerate(keypoints):
                if conf > 0.1:
                    cv2.circle(img_vis, (x, y), 4, colors[kp_idx], -1)
                    cv2.circle(img_vis, (x, y), 5, (255, 255, 255), 1)
            
            img_vis_rgb = cv2.cvtColor(img_vis, cv2.COLOR_BGR2RGB)
            vis_images.append(img_vis_rgb)  # numpy array, no PIL
    
    return vis_images

def make_collage(images, cols=5):
    """Crea un collage a partir de una lista de imágenes numpy (H, W, 3)."""
    rows = (len(images) + cols - 1) // cols
    h, w = images[0].shape[:2]
    
    collage = np.zeros((rows * h, cols * w, 3), dtype=np.uint8)
    for i, img in enumerate(images):
        r, c = divmod(i, cols)
        collage[r*h:(r+1)*h, c*w:(c+1)*w] = img
    
    return collage


def save_epoch_visualizations(images, output_dir, epoch):
    """Guarda imágenes individuales y collage por época."""
    epoch_dir = Path(output_dir) / f"epoch_{epoch:04d}"
    epoch_dir.mkdir(parents=True, exist_ok=True)
    
    for i, img in enumerate(images):
        img_bgr = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
        cv2.imwrite(str(epoch_dir / f"img_{i:02d}.png"), img_bgr)
    
    collage = make_collage(images, cols=5)
    collage_path = epoch_dir / "collage.png"
    collage_bgr = cv2.cvtColor(collage, cv2.COLOR_RGB2BGR)
    cv2.imwrite(str(collage_path), collage_bgr)
    
    return collage, collage_path


def send_telegram_with_images(
    message: str,
    collage: np.ndarray,
    token: Optional[str] = None,
    chat_id: Optional[str] = None,
    parse_mode: str = "HTML",
) -> bool:
    token = token or os.environ.get("TELEGRAM_BOT_TOKEN")
    chat_id = chat_id or os.environ.get("TELEGRAM_CHAT_ID")
    if not token or not chat_id:
        return False
    
    try:
        url_msg = f"https://api.telegram.org/bot{token}/sendMessage"
        payload_msg = urllib.parse.urlencode({
            "chat_id": chat_id,
            "text": message,
            "parse_mode": parse_mode,
        }).encode("utf-8")
        req = urllib.request.Request(url_msg, data=payload_msg, method="POST")
        with urllib.request.urlopen(req, timeout=10) as resp:
            result = json.loads(resp.read().decode())
            if not result.get("ok"):
                logger.warning(f"[TELEGRAM] ✗ Fallo enviando mensaje: {result}")
                return False
        
        # Convertir collage numpy a bytes PNG
        collage_pil = Image.fromarray(collage)
        img_byte_arr = io.BytesIO()
        collage_pil.save(img_byte_arr, format='PNG')
        img_byte_arr.seek(0)
        
        boundary = '----WebKitFormBoundary' + ''.join(
            np.random.choice(list('abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789'), 16)
        )
        body = (
            f'--{boundary}\r\n'
            f'Content-Disposition: form-data; name="chat_id"\r\n\r\n'
            f'{chat_id}\r\n'
            f'--{boundary}\r\n'
            f'Content-Disposition: form-data; name="photo"; filename="collage.png"\r\n'
            f'Content-Type: image/png\r\n\r\n'
        ).encode('utf-8')
        body += img_byte_arr.read()
        body += f'\r\n--{boundary}--\r\n'.encode('utf-8')
        
        url_photo = f"https://api.telegram.org/bot{token}/sendPhoto"
        req = urllib.request.Request(
            url_photo, data=body,
            headers={'Content-Type': f'multipart/form-data; boundary={boundary}'}
        )
        with urllib.request.urlopen(req, timeout=30) as resp:
            result = json.loads(resp.read().decode())
            if not result.get("ok"):
                logger.warning(f"[TELEGRAM] ✗ Fallo enviando collage: {result}")
                return False
        
        logger.info("[TELEGRAM] ✓ Collage enviado")
        return True
    
    except Exception as e:
        logger.warning(f"[TELEGRAM] ✗ Error: {e}")
        return False


# Modificar la sección de notificación en el bucle de entrenamiento:


def evaluate(model, dataloader, criterion, device, alpha=2.0):
    """
    Evalúa el modelo: loss total + accuracy (PCK).
    
    Returns:
        avg_loss, avg_acc, avg_loss_old, avg_loss_new
    """
    model.eval()
    total_loss = 0.0
    total_loss_old = 0.0
    total_loss_new = 0.0
    total_pck = 0.0
    n_batches = 0
    
    with torch.no_grad():
        for images, heatmaps in dataloader:
            images = images.to(device, non_blocking=True)
            heatmaps = heatmaps.to(device, non_blocking=True)
            
            outputs = model(images)
            
            # Loss ponderada
            loss_old = criterion(outputs[:, :17], heatmaps[:, :17])
            loss_new = criterion(outputs[:, 17:], heatmaps[:, 17:])
            loss = loss_old + alpha * loss_new
            
            # Accuracy (PCK)
            pck = compute_pck(outputs, heatmaps, threshold=0.05)
            
            total_loss += loss.item()
            total_loss_old += loss_old.item()
            total_loss_new += loss_new.item()
            total_pck += pck
            n_batches += 1
    
    avg_loss = total_loss / n_batches
    avg_loss_old = total_loss_old / n_batches
    avg_loss_new = total_loss_new / n_batches
    avg_pck = total_pck / n_batches
    
    return avg_loss, avg_pck, avg_loss_old, avg_loss_new


# ══════════════════════════════════════════════════════════════════════════════
# MAIN
# ══════════════════════════════════════════════════════════════════════════════
def main():
    parser = argparse.ArgumentParser(description='Fine-Tuning HRNet: 17→23 keypoints')
    parser.add_argument('--imgs-dir', type=str, required=True,
                        help='Directorio base de las imágenes (train_val/ y test/)')
    parser.add_argument('--heatmaps-dir', type=str, required=True,
                        help='Directorio base de los heatmaps .npy (train_val/ y test/)')
    parser.add_argument('--model-path', type=str, default='./models/pose_hrnet_w48_384x288.pth',
                        help='Ruta al modelo base (17 kp)')
    parser.add_argument('--output-model', type=str, default='./models/hrnet_23kp_best.pth',
                        help='Ruta donde guardar el mejor modelo')
    parser.add_argument('--output-stats', type=str, default='./training_stats_23kp.csv',
                        help='CSV con estadísticas de entrenamiento')
    parser.add_argument('--epochs', type=int, default=100, help='Número máximo de épocas')
    parser.add_argument('--batch-size', type=int, default=12, help='Tamaño del batch')
    parser.add_argument('--lr-backbone', type=float, default=1e-5, help='LR para stage4')
    parser.add_argument('--lr-head', type=float, default=1e-3, help='LR para final_layer')
    parser.add_argument('--alpha', type=float, default=2.0, 
                        help='Peso de loss para nuevos keypoints (loss = loss_old + alpha*loss_new)')
    parser.add_argument('--val-split', type=float, default=0.2, help='Fracción de train_val para validación')
    parser.add_argument('--early-stopping', type=int, default=10, help='Paciencia para early stopping')
    parser.add_argument('--seed', type=int, default=42, help='Semilla aleatoria')
    parser.add_argument('--gpus', type=int, nargs='+', default=None, help='IDs de GPU para DataParallel')
    parser.add_argument('--freeze-until', type=str, default='stage3', 
                        choices=['none', 'stage2', 'stage3', 'stage4'],
                        help='Hasta qué etapa congelar el backbone')
    parser.add_argument('--augment', action='store_true', help='Activar augmentación de datos')
    parser.add_argument('--num-brightness-augs', type=int, default=2, 
                        help='Número de variantes de brillo por imagen')
    parser.add_argument('--data-subset', type=float, default=None,
                        help='Porcentaje de datos a usar (0.0-1.0). Ej: 0.1 = 10%% del dataset')

    args = parser.parse_args()

    # Cargar .env para Telegram
    load_dotenv()
    telegram_token = os.environ.get("TELEGRAM_BOT_TOKEN")
    telegram_chat_id = os.environ.get("TELEGRAM_CHAT_ID")
    use_telegram = telegram_token and telegram_chat_id
    
    if use_telegram:
        logger.info("[TELEGRAM] ✓ Configurado")
    else:
        logger.info("[TELEGRAM] ✗ No configurado")

    # Rutas
    imgs_dir = Path(args.imgs_dir)
    heatmaps_dir = Path(args.heatmaps_dir)
    train_imgs_dir = imgs_dir / 'train_val'
    train_heatmaps_dir = heatmaps_dir / 'train_val'
    test_imgs_dir = imgs_dir / 'test'
    test_heatmaps_dir = heatmaps_dir / 'test'

    # Device
    use_parallel = False
    if args.gpus is not None and len(args.gpus) > 1 and torch.cuda.is_available():
        device = torch.device(f'cuda:{args.gpus[0]}')
        use_parallel = True
        logger.info(f"Modo DataParallel en GPUs: {args.gpus}")
    elif torch.cuda.is_available():
        gpu_id = args.gpus[0] if args.gpus else 0
        device = torch.device(f'cuda:{gpu_id}')
        logger.info(f"Usando GPU: {gpu_id}")
    else:
        device = torch.device('cpu')
    logger.info(f"Device: {device}")

    # ── Datasets ──
    logger.info("Cargando dataset train_val...")
    full_trainval = NoiseHeatmapDataset(
        imgs_dir=str(train_imgs_dir),
        heatmaps_dir=str(train_heatmaps_dir),
        input_size=(288, 384),
        augment=args.augment,
        num_brightness_augs=args.num_brightness_augs
    )
    logger.info(
        f"Train_val: {full_trainval._n_valid}/{full_trainval._n_total_original} pares válidos "
        f"({100*full_trainval._n_valid/full_trainval._n_total_original:.1f}%)"
    )

    # Aplicar subsampling si se especifica
    if args.data_subset is not None:
        logger.info(f"Aplicando subsampling: {args.data_subset*100:.1f}% de train_val")
        full_trainval = subsample_dataset(full_trainval, args.data_subset, seed=args.seed)
        logger.info(f"Train_val tras subsampling: {len(full_trainval)} samples")

    n_total = len(full_trainval)
    n_val = int(args.val_split * n_total)
    n_train = n_total - n_val
    logger.info(f"Total train_val: {n_total} → train: {n_train}, val: {n_val}")

    generator = torch.Generator().manual_seed(args.seed)
    train_dataset, val_dataset = random_split(full_trainval, [n_train, n_val], generator=generator)

    logger.info("Cargando dataset test...")
    full_test = NoiseHeatmapDataset(
        imgs_dir=str(test_imgs_dir),
        heatmaps_dir=str(test_heatmaps_dir),
        input_size=(288, 384),
        augment=False  # No augment en test
    )

    logger.info(
        f"Test: {full_test._n_valid}/{full_test._n_total_original} pares válidos "
        f"({100*full_test._n_valid/full_test._n_total_original:.1f}%)"
    )
    
    # Aplicar subsampling a test si se especifica
    if args.data_subset is not None:
        logger.info(f"Aplicando subsampling: {args.data_subset*100:.1f}% de test")
        test_dataset = subsample_dataset(full_test, args.data_subset, seed=args.seed)
        logger.info(f"Test tras subsampling: {len(test_dataset)} samples")
    else:
        test_dataset = full_test
    
    logger.info(f"Total test: {len(test_dataset)}")

    num_workers = 10
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True,
                              num_workers=num_workers, pin_memory=True, persistent_workers=True,
                              prefetch_factor=4 )
    
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False,
                            num_workers=num_workers, pin_memory=True, persistent_workers=True,
                            prefetch_factor=4)
    
    test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False,
                             num_workers=num_workers, pin_memory=True, persistent_workers=True,
                             prefetch_factor=4)

    # ── Modelo ──
    logger.info("Inicializando modelo con 23 keypoints...")
    model = PoseHRNet(width=48, num_joints=23)
    
    if Path(args.model_path).exists():
        model = load_pretrained_and_extend(
            model, args.model_path, device, num_old_joints=17, num_new_joints=23
        )
    else:
        logger.warning(f"No se encontró {args.model_path}. Entrenando desde cero.")
    
    # Congelar backbone parcialmente
    freeze_backbone_partial(model, freeze_until=args.freeze_until)
    
    model = model.to(device)
    if use_parallel:
        model = nn.DataParallel(model, device_ids=args.gpus)
        logger.info(f"Modelo envuelto en DataParallel")

    # ── Loss y Optimizer ──
    criterion = nn.MSELoss()
    
    # Grupos de parámetros con LR diferenciados
    raw_model = model.module if isinstance(model, nn.DataParallel) else model
    
    stage4_params = []
    head_params = []
    
    for name, param in raw_model.named_parameters():
        if not param.requires_grad:
            continue
        if 'stage4' in name:
            stage4_params.append(param)
        elif 'final_layer' in name:
            head_params.append(param)
    
    optimizer = torch.optim.Adam([
        {'params': stage4_params, 'lr': args.lr_backbone},
        {'params': head_params, 'lr': args.lr_head}
    ])
    
    logger.info(f"Optimizer: stage4 LR={args.lr_backbone}, head LR={args.lr_head}")

    # ── Variables de seguimiento ──
    stats = []
    best_val_loss = float('inf')
    patience_counter = 0
    output_path = Path(args.output_model)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    stats_path = Path(args.output_stats)
    stats_path.parent.mkdir(parents=True, exist_ok=True)

    # Contexto para notificaciones
    gpu_context = f"GPUs {args.gpus} | " if args.gpus else ""
    context_str = f"{gpu_context}BS={args.batch_size} LR={args.lr_head}\n"

    # Notificar inicio
    if use_telegram:
        subset_info = f"Subset: {args.data_subset*100:.1f}%\n" if args.data_subset else ""
        gpu_info = f"GPUs: {args.gpus}\n" if args.gpus else "GPU: CPU\n"
        
        send_telegram(
            f"🚀 <b>Inicio Fine-Tuning HRNet 23kp</b>\n"
            f"──────────────────────\n"
            f"{gpu_info}"
            f"Train: {n_train}, Val: {n_val}, Test: {len(test_dataset)}\n"
            f"{subset_info}"
            f"──────────────────────\n"
            f"<b>Hyperparams:</b>\n"
            f"• Batch size: {args.batch_size}\n"
            f"• Épocas: {args.epochs}\n"
            f"• LR backbone: {args.lr_backbone}\n"
            f"• LR head: {args.lr_head}\n"
            f"• Alpha: {args.alpha}\n"
            f"• Val split: {args.val_split}\n"
            f"• Freeze until: {args.freeze_until}\n"
            f"• Augment: {args.augment}\n"
            f"• Brightness augs: {args.num_brightness_augs}\n"
            f"• Early stopping: {args.early_stopping}\n"
            f"• Seed: {args.seed}",
            token=telegram_token,
            chat_id=telegram_chat_id
        )

    # ── Bucle de entrenamiento ──
    logger.info("Comenzando el entrenamiento...")
    for epoch in range(args.epochs):
        # ── Train ──
        model.train()
        running_loss = 0.0
        running_loss_old = 0.0
        running_loss_new = 0.0
        running_pck = 0.0
        n_batches_train = 0
        
        progress_bar = tqdm(train_loader, desc=f"Época {epoch + 1}/{args.epochs} [train]")
        for images, heatmaps in progress_bar:
            images = images.to(device, non_blocking=True)
            heatmaps = heatmaps.to(device, non_blocking=True)

            outputs = model(images)
            
            # Loss ponderada
            loss_old = criterion(outputs[:, :17], heatmaps[:, :17])
            loss_new = criterion(outputs[:, 17:], heatmaps[:, 17:])
            loss = loss_old + args.alpha * loss_new

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            # Métricas
            with torch.no_grad():
                pck = compute_pck(outputs, heatmaps, threshold=0.05)
            
            running_loss += loss.item()
            running_loss_old += loss_old.item()
            running_loss_new += loss_new.item()
            running_pck += pck
            n_batches_train += 1
            
            progress_bar.set_postfix({
                'loss': f'{loss.item():.6f}',
                'pck': f'{pck:.2f}%'
            })

        avg_train_loss = running_loss / n_batches_train
        avg_train_loss_old = running_loss_old / n_batches_train
        avg_train_loss_new = running_loss_new / n_batches_train
        avg_train_pck = running_pck / n_batches_train

        # ── Validation ──
        avg_val_loss, avg_val_pck, avg_val_loss_old, avg_val_loss_new = evaluate(
            model, val_loader, criterion, device, alpha=args.alpha
        )

        logger.info(
            f"Época [{epoch + 1}/{args.epochs}] — "
            f"Train Loss: {avg_train_loss:.6f} (old={avg_train_loss_old:.6f}, new={avg_train_loss_new:.6f}) | "
            f"Train PCK: {avg_train_pck:.2f}% | "
            f"Val Loss: {avg_val_loss:.6f} (old={avg_val_loss_old:.6f}, new={avg_val_loss_new:.6f}) | "
            f"Val PCK: {avg_val_pck:.2f}%"
        )

        # ── Test (cada época) ──
        avg_test_loss, avg_test_pck, avg_test_loss_old, avg_test_loss_new = evaluate(
            model, test_loader, criterion, device, alpha=args.alpha
        )
        
        logger.info(
            f"  Test Loss: {avg_test_loss:.6f} | Test PCK: {avg_test_pck:.2f}%"
        )

        # ── Guardar estadísticas ──
        stats.append({
            'epoch': epoch + 1,
            'train_loss': avg_train_loss,
            'train_loss_old': avg_train_loss_old,
            'train_loss_new': avg_train_loss_new,
            'train_acc': avg_train_pck,
            'val_loss': avg_val_loss,
            'val_loss_old': avg_val_loss_old,
            'val_loss_new': avg_val_loss_new,
            'val_acc': avg_val_pck,
            'test_loss': avg_test_loss,
            'test_loss_old': avg_test_loss_old,
            'test_loss_new': avg_test_loss_new,
            'test_acc': avg_test_pck,
        })

          # ── Guardar CSV incremental ──
        with open(stats_path, 'w', newline='', encoding='utf-8') as f:
            fieldnames = ['epoch', 'train_loss', 'train_loss_old', 'train_loss_new', 'train_acc',
                          'val_loss', 'val_loss_old', 'val_loss_new', 'val_acc',
                          'test_loss', 'test_loss_old', 'test_loss_new', 'test_acc']
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(stats)
        logger.info(f"Estadísticas guardadas en: {stats_path}")

        # ── Checkpoint si mejora val_loss ──
        if avg_val_loss < best_val_loss:
            improvement = best_val_loss - avg_val_loss
            best_val_loss = avg_val_loss
            patience_counter = 0
            
            raw_model_to_save = model.module if isinstance(model, nn.DataParallel) else model
            torch.save({
                'epoch': epoch + 1,
                'state_dict': raw_model_to_save.state_dict(),
                'optimizer': optimizer.state_dict(),
                'val_loss': avg_val_loss,
                'val_acc': avg_val_pck,
            }, str(output_path))
            
            logger.info(f"  → Mejor modelo guardado (val_loss={best_val_loss:.6f}, mejora={improvement:.6f})")
            
            # Notificación Telegram con visualizaciones
            if use_telegram:
                # Generar visualizaciones sobre imágenes de test
                logger.info("Generando visualizaciones para Telegram...")
                vis_images = visualize_predictions(
                    model=raw_model_to_save,
                    dataset=test_dataset,
                    device=device,
                    indices=[0, 1, 2, 3, 4, 5, 6, 7, 8, 9],  # Siempre las mismas 10 imágenes
                    input_size=(288, 384)
                )

                collage, collage_path = save_epoch_visualizations(
                    vis_images,
                    output_dir=Path(args.output_model).parent / "visualizations",
                    epoch=epoch + 1
                )
                logger.info(f"Visualizaciones guardadas en: {collage_path.parent}")
                
                msg = (
                    f"🏆 <b>Nuevo mejor modelo</b>\n"
                    f"{context_str}"
                    f"Val Loss: <b>{avg_val_loss:.6f}</b> (-{improvement:.6f})\n"
                    f"Val Acc: {avg_val_pck:.2f}%\n"
                    f"Época: {epoch + 1}"
                )
                
                send_telegram_with_images(
                    message=msg,
                    collage=collage,
                    token=telegram_token,
                    chat_id=telegram_chat_id
                )
        else:
            patience_counter += 1
            logger.info(
                f"  → Sin mejora en val_loss. "
                f"Paciencia: {patience_counter}/{args.early_stopping}"
            )
            if patience_counter >= args.early_stopping:
                logger.info("Early stopping activado. Finalizando entrenamiento.")
                break


    # ── Evaluación final en test con mejor modelo ──
    logger.info("Evaluando el mejor modelo en test...")
    try:
        best_ckpt = torch.load(str(output_path), map_location=device, weights_only=False)
    except TypeError:
        best_ckpt = torch.load(str(output_path), map_location=device)
    
    raw_model_final = model.module if isinstance(model, nn.DataParallel) else model
    raw_model_final.load_state_dict(best_ckpt['state_dict'])
    
    final_test_loss, final_test_pck, _, _ = evaluate(
        raw_model_final, test_loader, criterion, device, alpha=args.alpha
    )
    logger.info(f"Test Loss (mejor modelo): {final_test_loss:.6f} | PCK: {final_test_pck:.2f}%")

    # Notificación final
    if use_telegram:
        send_telegram(
            f"✅ <b>Entrenamiento finalizado</b>\n"
            f"Mejor Val Loss: {best_val_loss:.6f}\n"
            f"Test Loss: {final_test_loss:.6f}\n"
            f"Test PCK: {final_test_pck:.2f}%",
            token=telegram_token,
            chat_id=telegram_chat_id
        )

    logger.info("✅ Proceso finalizado.")


if __name__ == '__main__':
    main()