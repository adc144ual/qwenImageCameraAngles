#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
HRNet Fine-Tuning Script - 27 Keypoints (17 body + 6 hands + 4 feet)
==========================================================
Descripción:
    Script para re-entrenar (fine-tune) el modelo HRNet extendiendo
    de 17 a 27 keypoints, añadiendo puntos de manos y pies sin perder
    conocimiento previo (evitando catastrophic forgetting).

    Canales:
        0-16  : body (HRNet COCO)
        17-22 : manos (MediaPipe landmarks 17-22)
        23-26 : pies  (MediaPipe landmarks 29, 30, 31, 32)

Ejecución:
python finetuning_hrnet_hands_feet.py
 --imgs-dir /nas/antoniodetoro/datasets/qwen/hr_net_with_hands/images
 --heatmaps-dir /nas/antoniodetoro/datasets/qwen/hr_net_with_hands/heatmaps/
 --model-path /nas/antoniodetoro/qwen/Qwen-Image-Edit-Angles-2/src/models/pose_hrnet_w48_384x288.pth
 --output-model /nas/antoniodetoro/qwen/Qwen-Image-Edit-Angles-2/src/hr_net/outputs/3/hr_net_27kp_best.pth
 --output-stats /nas/antoniodetoro/qwen/Qwen-Image-Edit-Angles-2/src/hr_net/outputs/3/training_stats.csv
 --epochs 100 --lr-backbone 1e-5 --lr-head 1e-3
 --alpha-hands 3.0 --alpha-feet 3.0
 --early-stopping 15 --freeze-until stage4 --augment --num-brightness-augs 1
 --gpus 1 --batch-size 128 --data-subset 1.0
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

try:
    from hrnet_inference import PoseHRNet
except ImportError:
    raise ImportError("No se pudo importar 'PoseHRNet'. Asegúrate de que el script original se llama 'hrnet_inference.py' y está en esta misma carpeta.")

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Número de keypoints por grupo
N_BODY  = 17
N_HANDS = 6
N_FEET  = 4
N_TOTAL = N_BODY + N_HANDS + N_FEET  # 27

# Slices para cada grupo
SL_BODY  = slice(0, 17)
SL_HANDS = slice(17, 23)
SL_FEET  = slice(23, 27)


# ══════════════════════════════════════════════════════════════════════════════
# TELEGRAM NOTIFIER
# ══════════════════════════════════════════════════════════════════════════════
def send_telegram(
    message: str,
    token: Optional[str] = None,
    chat_id: Optional[str] = None,
    parse_mode: str = "HTML",
) -> bool:
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
                return False
        logger.info(f"[TELEGRAM] ✓ Enviado: {_preview!r}")
        print(f"[TELEGRAM] ✓ {_preview}")
        return True
    except Exception as e:
        logger.warning(f"[TELEGRAM] ✗ Error: {e} | msg={_preview!r}")
        return False


def send_telegram_with_images(
    message: str,
    collage: np.ndarray,
    token: Optional[str] = None,
    chat_id: Optional[str] = None,
    parse_mode: str = "HTML",
) -> bool:
    import io
    from PIL import Image

    token = token or os.environ.get("TELEGRAM_BOT_TOKEN")
    chat_id = chat_id or os.environ.get("TELEGRAM_CHAT_ID")
    if not token or not chat_id:
        return False
    try:
        url_msg = f"https://api.telegram.org/bot{token}/sendMessage"
        payload_msg = urllib.parse.urlencode({
            "chat_id": chat_id, "text": message, "parse_mode": parse_mode,
        }).encode("utf-8")
        req = urllib.request.Request(url_msg, data=payload_msg, method="POST")
        with urllib.request.urlopen(req, timeout=10) as resp:
            result = json.loads(resp.read().decode())
            if not result.get("ok"):
                return False

        collage_pil = Image.fromarray(collage)
        img_byte_arr = io.BytesIO()
        collage_pil.save(img_byte_arr, format='PNG')
        img_byte_arr.seek(0)

        boundary = '----WebKitFormBoundary' + ''.join(
            np.random.choice(list('abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789'), 16)
        )
        body = (
            f'--{boundary}\r\nContent-Disposition: form-data; name="chat_id"\r\n\r\n'
            f'{chat_id}\r\n'
            f'--{boundary}\r\nContent-Disposition: form-data; name="photo"; filename="collage.png"\r\n'
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
                return False
        logger.info("[TELEGRAM] ✓ Collage enviado")
        return True
    except Exception as e:
        logger.warning(f"[TELEGRAM] ✗ Error: {e}")
        return False


# ══════════════════════════════════════════════════════════════════════════════
# DATASET
# ══════════════════════════════════════════════════════════════════════════════
class NoiseHeatmapDataset(Dataset):
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

        self.img_paths = all_img_paths
        valid_indices, n_total = self._validate_pairs()
        self.img_paths = [all_img_paths[i] for i in valid_indices]
        self._n_total_original = n_total
        self._n_valid = len(self.img_paths)

        if not self.img_paths:
            raise ValueError(f"No hay pares válidos en {imgs_dir}")

        self.mean = np.array([0.485, 0.456, 0.406], dtype=np.float32)
        self.std  = np.array([0.229, 0.224, 0.225], dtype=np.float32)

        if self.augment:
            self.samples_per_img = (1 + self.num_brightness_augs) * 2
        else:
            self.samples_per_img = 1

    def _validate_pairs(self):
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
                    continue

            heatmap_path = self.heatmaps_dir / npy_name
            if not heatmap_path.exists():
                continue
            try:
                heatmap = np.load(heatmap_path)
            except Exception:
                continue

            # Validar shape: ahora 27 canales
            if heatmap.ndim != 3 or heatmap.shape[0] != N_TOTAL:
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
                raise ValueError(f"Nombre de imagen no reconocido: {img_path.name}")

        heatmap_path = self.heatmaps_dir / npy_name
        if not heatmap_path.exists():
            raise FileNotFoundError(f"Falta el heatmap esperado: {heatmap_path}")

        image = cv2.imread(str(img_path))
        if image is None:
            raise ValueError(f"No se pudo cargar la imagen: {img_path}")
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        heatmap = np.load(heatmap_path).astype(np.float32)

        if self.augment:
            do_flip = aug_variant >= (self.samples_per_img // 2)
            brightness_idx = aug_variant % (self.samples_per_img // 2)

            if brightness_idx > 0:
                alpha = np.random.uniform(0.7, 1.3)
                beta = np.random.randint(-30, 30)
                image = cv2.convertScaleAbs(image, alpha=alpha, beta=beta)

            if do_flip:
                image = cv2.flip(image, 1)
                heatmap = self._flip_heatmap(heatmap)

        h, w = self.input_size
        image = cv2.resize(image, (w, h), interpolation=cv2.INTER_LINEAR)
        image = image.astype(np.float32) / 255.0
        image = (image - self.mean) / self.std

        image_tensor = torch.from_numpy(image.transpose(2, 0, 1)).float()
        heatmap_tensor = torch.from_numpy(heatmap).float()
        return image_tensor, heatmap_tensor

    def _flip_heatmap(self, heatmap):
        """Flip horizontal del heatmap (27 canales)."""
        flipped = np.flip(heatmap, axis=2).copy()

        # Body: pares simétricos HRNet COCO
        body_swap_pairs = [
            (1, 2), (3, 4), (5, 6), (7, 8), (9, 10),
            (11, 12), (13, 14), (15, 16),
        ]
        for left, right in body_swap_pairs:
            flipped[[left, right]] = flipped[[right, left]]

        # Manos (canales 17-22): [17,19,21] izq ↔ [18,20,22] der
        flipped[[17, 19, 21, 18, 20, 22]] = flipped[[18, 20, 22, 17, 19, 21]]

        # Pies (canales 23-26):
        #   23 = MediaPipe 29 (heel izq)
        #   24 = MediaPipe 30 (heel der)
        #   25 = MediaPipe 31 (foot_index izq)
        #   26 = MediaPipe 32 (foot_index der)
        # ⚠️ Verificar antes de usar que 29/31 son izq y 30/32 son der en tu dataset
        flipped[[23, 25, 24, 26]] = flipped[[24, 26, 23, 25]]

        return flipped


def subsample_dataset(dataset, fraction, seed=42):
    if fraction is None or fraction >= 1.0:
        return dataset
    n_total = len(dataset)
    n_subset = max(1, int(n_total * fraction))
    generator = torch.Generator().manual_seed(seed)
    indices = torch.randperm(n_total, generator=generator)[:n_subset].tolist()
    return Subset(dataset, indices)


# ══════════════════════════════════════════════════════════════════════════════
# MODELO
# ══════════════════════════════════════════════════════════════════════════════
def load_pretrained_and_extend(model, weights_path, device, num_old_joints=17, num_new_joints=27):
    """
    Carga pesos preentrenados (17 kp) y extiende a num_new_joints.
    Los N_TOTAL - 17 canales nuevos de final_layer se inicializan con Kaiming.
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

    state_dict_new = model.state_dict()

    for name, param in state_dict_old.items():
        if name not in state_dict_new:
            continue
        if state_dict_new[name].shape == param.shape:
            state_dict_new[name] = param
        elif 'final_layer' in name:
            if 'weight' in name:
                state_dict_new[name][:num_old_joints] = param
                nn.init.kaiming_normal_(state_dict_new[name][num_old_joints:])
                logger.info(f"  Extendido {name}: {param.shape} → {state_dict_new[name].shape}")
            elif 'bias' in name:
                state_dict_new[name][:num_old_joints] = param
                nn.init.zeros_(state_dict_new[name][num_old_joints:])
                logger.info(f"  Extendido {name}: {param.shape} → {state_dict_new[name].shape}")
        else:
            logger.warning(f"  Saltando {name}: shape mismatch {param.shape} vs {state_dict_new[name].shape}")

    model.load_state_dict(state_dict_new, strict=True)
    logger.info("Pesos cargados y extendidos correctamente")
    return model


def freeze_backbone_partial(model, freeze_until='stage4'):
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

    frozen    = sum(p.numel() for p in model.parameters() if not p.requires_grad)
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    logger.info(f"Parámetros congelados: {frozen:,} | Entrenables: {trainable:,}")


# ══════════════════════════════════════════════════════════════════════════════
# MÉTRICAS
# ══════════════════════════════════════════════════════════════════════════════
def compute_pck(pred_heatmaps, gt_heatmaps, threshold=0.05):
    B, C, H, W = pred_heatmaps.shape
    pred_coords = []
    gt_coords   = []
    for b in range(B):
        for c in range(C):
            pred_hm = pred_heatmaps[b, c].cpu().numpy()
            gt_hm   = gt_heatmaps[b, c].cpu().numpy()
            pred_y, pred_x = np.unravel_index(pred_hm.argmax(), pred_hm.shape)
            gt_y,   gt_x   = np.unravel_index(gt_hm.argmax(),   gt_hm.shape)
            pred_coords.append([pred_x, pred_y])
            gt_coords.append([gt_x, gt_y])
    pred_coords = np.array(pred_coords)
    gt_coords   = np.array(gt_coords)
    distances = np.linalg.norm(pred_coords - gt_coords, axis=1)
    normalized_dist = distances / np.sqrt(H**2 + W**2)
    pck = 100.0 * (normalized_dist < threshold).sum() / len(distances)
    return pck


def compute_loss(criterion, outputs, heatmaps, alpha_hands, alpha_feet):
    """Loss ponderada con tres grupos: body, hands, feet."""
    loss_body  = criterion(outputs[:, SL_BODY],  heatmaps[:, SL_BODY])
    loss_hands = criterion(outputs[:, SL_HANDS], heatmaps[:, SL_HANDS])
    loss_feet  = criterion(outputs[:, SL_FEET],  heatmaps[:, SL_FEET])
    loss = loss_body + alpha_hands * loss_hands + alpha_feet * loss_feet
    return loss, loss_body, loss_hands, loss_feet


def evaluate(model, dataloader, criterion, device, alpha_hands, alpha_feet):
    model.eval()
    totals = dict(loss=0, body=0, hands=0, feet=0, pck=0)
    n_batches = 0
    with torch.no_grad():
        for images, heatmaps in dataloader:
            images   = images.to(device, non_blocking=True)
            heatmaps = heatmaps.to(device, non_blocking=True)
            outputs  = model(images)
            loss, l_body, l_hands, l_feet = compute_loss(
                criterion, outputs, heatmaps, alpha_hands, alpha_feet
            )
            pck = compute_pck(outputs, heatmaps, threshold=0.05)
            totals['loss']  += loss.item()
            totals['body']  += l_body.item()
            totals['hands'] += l_hands.item()
            totals['feet']  += l_feet.item()
            totals['pck']   += pck
            n_batches += 1
    return {k: v / n_batches for k, v in totals.items()}


# ══════════════════════════════════════════════════════════════════════════════
# VISUALIZACIÓN
# ══════════════════════════════════════════════════════════════════════════════
def visualize_predictions(model, dataset, device, indices, input_size=(288, 384)):
    model.eval()

    # Colores por grupo (BGR)
    colors = (
        [(0, 255, 0)]   * N_BODY   +   # verde  - body
        [(255, 0, 0)]   * N_HANDS  +   # azul   - manos
        [(0, 0, 255)] * N_FEET       # rojo - pies
    )

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

    skeleton_hands = [
        (9, 17), (9, 19), (9, 21),
        (10, 18), (10, 20), (10, 22),
    ]
    # tobillos COCO (15=izq, 16=der) → landmarks pie
    skeleton_feet = [
        (15, 23), (15, 25), (23, 25), # pie izq
        (16, 24), (16, 26), (24, 26), # pie der
    ]
    skeleton = skeleton_body + skeleton_hands + skeleton_feet

    vis_images = []
    mean = np.array([0.485, 0.456, 0.406])
    std  = np.array([0.229, 0.224, 0.225])

    with torch.no_grad():
        for idx in indices:
            img_tensor, _ = dataset[idx]
            img_batch     = img_tensor.unsqueeze(0).to(device)
            heatmap_pred  = model(img_batch)[0]

            img_np = img_tensor.cpu().numpy().transpose(1, 2, 0)
            img_np = np.clip((img_np * std + mean) * 255.0, 0, 255).astype(np.uint8)
            img_vis = cv2.cvtColor(img_np, cv2.COLOR_RGB2BGR)

            H_hm, W_hm = heatmap_pred.shape[1:]
            H_img, W_img = img_np.shape[:2]

            keypoints = []
            for c in range(N_TOTAL):
                hm = heatmap_pred[c].cpu().numpy()
                y, x = np.unravel_index(hm.argmax(), hm.shape)
                keypoints.append((
                    int(x * W_img / W_hm),
                    int(y * H_img / H_hm),
                    float(hm[y, x])
                ))

            for i, j in skeleton:
                x1, y1, c1 = keypoints[i]
                x2, y2, c2 = keypoints[j]
                if c1 > 0.1 and c2 > 0.1:
                    if i < N_BODY and j < N_BODY:
                        color = (0, 255, 0)
                    elif i < N_BODY + N_HANDS and j < N_BODY + N_HANDS:
                        color = (255, 0, 0)
                    else:
                        color = (0, 165, 255)
                    cv2.line(img_vis, (x1, y1), (x2, y2), color, 2)

            for kp_idx, (x, y, conf) in enumerate(keypoints):
                if conf > 0.1:
                    cv2.circle(img_vis, (x, y), 4, colors[kp_idx], -1)
                    cv2.circle(img_vis, (x, y), 5, (255, 255, 255), 1)

            vis_images.append(cv2.cvtColor(img_vis, cv2.COLOR_BGR2RGB))

    return vis_images


def make_collage(images, cols=5):
    rows = (len(images) + cols - 1) // cols
    h, w = images[0].shape[:2]
    collage = np.zeros((rows * h, cols * w, 3), dtype=np.uint8)
    for i, img in enumerate(images):
        r, c = divmod(i, cols)
        collage[r*h:(r+1)*h, c*w:(c+1)*w] = img
    return collage


def save_epoch_visualizations(images, output_dir, epoch):
    epoch_dir = Path(output_dir) / f"epoch_{epoch:04d}"
    epoch_dir.mkdir(parents=True, exist_ok=True)
    for i, img in enumerate(images):
        cv2.imwrite(str(epoch_dir / f"img_{i:02d}.png"), cv2.cvtColor(img, cv2.COLOR_RGB2BGR))
    collage = make_collage(images, cols=5)
    collage_path = epoch_dir / "collage.png"
    cv2.imwrite(str(collage_path), cv2.cvtColor(collage, cv2.COLOR_RGB2BGR))
    return collage, collage_path


# ══════════════════════════════════════════════════════════════════════════════
# MAIN
# ══════════════════════════════════════════════════════════════════════════════
def main():
    parser = argparse.ArgumentParser(description='Fine-Tuning HRNet: 17→27 keypoints')
    parser.add_argument('--imgs-dir',        type=str, required=True)
    parser.add_argument('--heatmaps-dir',    type=str, required=True)
    parser.add_argument('--model-path',      type=str, default='./models/pose_hrnet_w48_384x288.pth')
    parser.add_argument('--output-model',    type=str, default='./models/hrnet_27kp_best.pth')
    parser.add_argument('--output-stats',    type=str, default='./training_stats_27kp.csv')
    parser.add_argument('--epochs',          type=int,   default=100)
    parser.add_argument('--batch-size',      type=int,   default=12)
    parser.add_argument('--lr-backbone',     type=float, default=1e-5)
    parser.add_argument('--lr-head',         type=float, default=1e-3)
    parser.add_argument('--alpha-hands',     type=float, default=3.0,
                        help='Peso de loss para keypoints de manos')
    parser.add_argument('--alpha-feet',      type=float, default=3.0,
                        help='Peso de loss para keypoints de pies')
    parser.add_argument('--val-split',       type=float, default=0.2)
    parser.add_argument('--early-stopping',  type=int,   default=10)
    parser.add_argument('--seed',            type=int,   default=42)
    parser.add_argument('--gpus',            type=int, nargs='+', default=None)
    parser.add_argument('--freeze-until',    type=str, default='stage3',
                        choices=['none', 'stage2', 'stage3', 'stage4'])
    parser.add_argument('--augment',         action='store_true')
    parser.add_argument('--num-brightness-augs', type=int, default=2)
    parser.add_argument('--data-subset',     type=float, default=None)
    args = parser.parse_args()

    load_dotenv()
    telegram_token   = os.environ.get("TELEGRAM_BOT_TOKEN")
    telegram_chat_id = os.environ.get("TELEGRAM_CHAT_ID")
    use_telegram     = bool(telegram_token and telegram_chat_id)
    logger.info(f"[TELEGRAM] {'✓ Configurado' if use_telegram else '✗ No configurado'}")

    # ── Rutas ──
    imgs_dir      = Path(args.imgs_dir)
    heatmaps_dir  = Path(args.heatmaps_dir)

    # ── Device ──
    use_parallel = False
    if args.gpus is not None and len(args.gpus) > 1 and torch.cuda.is_available():
        device = torch.device(f'cuda:{args.gpus[0]}')
        use_parallel = True
    elif torch.cuda.is_available():
        device = torch.device(f'cuda:{args.gpus[0] if args.gpus else 0}')
    else:
        device = torch.device('cpu')
    logger.info(f"Device: {device}")

    # ── Datasets ──
    logger.info("Cargando dataset train_val...")
    full_trainval = NoiseHeatmapDataset(
        imgs_dir=str(imgs_dir / 'train_val'),
        heatmaps_dir=str(heatmaps_dir / 'train_val'),
        input_size=(288, 384),
        augment=args.augment,
        num_brightness_augs=args.num_brightness_augs,
    )
    logger.info(
        f"Train_val: {full_trainval._n_valid}/{full_trainval._n_total_original} pares válidos "
        f"({100*full_trainval._n_valid/full_trainval._n_total_original:.1f}%)"
    )

    if args.data_subset is not None:
        full_trainval = subsample_dataset(full_trainval, args.data_subset, seed=args.seed)
        logger.info(f"Train_val tras subsampling: {len(full_trainval)} samples")

    n_total = len(full_trainval)
    n_val   = int(args.val_split * n_total)
    n_train = n_total - n_val
    generator = torch.Generator().manual_seed(args.seed)
    train_dataset, val_dataset = random_split(full_trainval, [n_train, n_val], generator=generator)
    logger.info(f"Train: {n_train} | Val: {n_val}")

    logger.info("Cargando dataset test...")
    full_test = NoiseHeatmapDataset(
        imgs_dir=str(imgs_dir / 'test'),
        heatmaps_dir=str(heatmaps_dir / 'test'),
        input_size=(288, 384),
        augment=False,
    )
    logger.info(
        f"Test: {full_test._n_valid}/{full_test._n_total_original} pares válidos "
        f"({100*full_test._n_valid/full_test._n_total_original:.1f}%)"
    )
    test_dataset = subsample_dataset(full_test, args.data_subset, seed=args.seed) \
        if args.data_subset is not None else full_test
    logger.info(f"Test: {len(test_dataset)}")

    num_workers = 10
    loader_kwargs = dict(num_workers=num_workers, pin_memory=True,
                         persistent_workers=True, prefetch_factor=4)
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True,  **loader_kwargs)
    val_loader   = DataLoader(val_dataset,   batch_size=args.batch_size, shuffle=False, **loader_kwargs)
    test_loader  = DataLoader(test_dataset,  batch_size=args.batch_size, shuffle=False, **loader_kwargs)

    # ── Modelo ──
    logger.info(f"Inicializando modelo con {N_TOTAL} keypoints...")
    model = PoseHRNet(width=48, num_joints=N_TOTAL)

    if Path(args.model_path).exists():
        model = load_pretrained_and_extend(
            model, args.model_path, device, num_old_joints=17, num_new_joints=N_TOTAL
        )
    else:
        logger.warning(f"No se encontró {args.model_path}. Entrenando desde cero.")

    freeze_backbone_partial(model, freeze_until=args.freeze_until)
    model = model.to(device)

    if use_parallel:
        model = nn.DataParallel(model, device_ids=args.gpus)

    # ── Optimizer ──
    criterion  = nn.MSELoss()
    raw_model  = model.module if isinstance(model, nn.DataParallel) else model
    stage4_params = [p for n, p in raw_model.named_parameters() if p.requires_grad and 'stage4' in n]
    head_params   = [p for n, p in raw_model.named_parameters() if p.requires_grad and 'final_layer' in n]
    optimizer = torch.optim.Adam([
        {'params': stage4_params, 'lr': args.lr_backbone},
        {'params': head_params,   'lr': args.lr_head},
    ])
    logger.info(f"Optimizer: stage4 LR={args.lr_backbone}, head LR={args.lr_head}")

    # ── Tracking ──
    stats          = []
    best_val_loss  = float('inf')
    patience_counter = 0
    output_path    = Path(args.output_model)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    stats_path = Path(args.output_stats)
    stats_path.parent.mkdir(parents=True, exist_ok=True)
    context_str = f"BS={args.batch_size} LR={args.lr_head} α_hands={args.alpha_hands} α_feet={args.alpha_feet}\n"

    if use_telegram:
        subset_info = f"Subset: {args.data_subset*100:.1f}%\n" if args.data_subset else ""
        send_telegram(
            f"🚀 <b>Inicio Fine-Tuning HRNet 27kp</b>\n"
            f"──────────────────────\n"
            f"GPUs: {args.gpus}\n"
            f"Train: {n_train} | Val: {n_val} | Test: {len(test_dataset)}\n"
            f"{subset_info}"
            f"──────────────────────\n"
            f"<b>Hyperparams:</b>\n"
            f"• Batch size: {args.batch_size}\n"
            f"• Épocas: {args.epochs}\n"
            f"• LR backbone: {args.lr_backbone}\n"
            f"• LR head: {args.lr_head}\n"
            f"• Alpha hands: {args.alpha_hands}\n"
            f"• Alpha feet: {args.alpha_feet}\n"
            f"• Freeze until: {args.freeze_until}\n"
            f"• Augment: {args.augment} (brightness_augs={args.num_brightness_augs})\n"
            f"• Early stopping: {args.early_stopping}",
            token=telegram_token, chat_id=telegram_chat_id
        )

    # ── Bucle de entrenamiento ──
    logger.info("Comenzando entrenamiento...")
    for epoch in range(args.epochs):

        # Train
        model.train()
        t = dict(loss=0, body=0, hands=0, feet=0, pck=0)
        n_batches_train = 0

        pbar = tqdm(train_loader, desc=f"Época {epoch+1}/{args.epochs} [train]")
        for images, heatmaps in pbar:
            images   = images.to(device, non_blocking=True)
            heatmaps = heatmaps.to(device, non_blocking=True)
            outputs  = model(images)

            loss, l_body, l_hands, l_feet = compute_loss(
                criterion, outputs, heatmaps, args.alpha_hands, args.alpha_feet
            )
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            with torch.no_grad():
                pck = compute_pck(outputs, heatmaps, threshold=0.05)

            t['loss']  += loss.item()
            t['body']  += l_body.item()
            t['hands'] += l_hands.item()
            t['feet']  += l_feet.item()
            t['pck']   += pck
            n_batches_train += 1

            pbar.set_postfix({'loss': f"{loss.item():.5f}", 'pck': f"{pck:.1f}%"})

        tr = {k: v / n_batches_train for k, v in t.items()}

        # Val & Test
        val  = evaluate(model, val_loader,  criterion, device, args.alpha_hands, args.alpha_feet)
        test = evaluate(model, test_loader, criterion, device, args.alpha_hands, args.alpha_feet)

        logger.info(
            f"[{epoch+1}/{args.epochs}] "
            f"Train loss={tr['loss']:.5f} (body={tr['body']:.5f} hands={tr['hands']:.5f} feet={tr['feet']:.5f}) pck={tr['pck']:.1f}% | "
            f"Val loss={val['loss']:.5f} (body={val['body']:.5f} hands={val['hands']:.5f} feet={val['feet']:.5f}) pck={val['pck']:.1f}% | "
            f"Test loss={test['loss']:.5f} pck={test['pck']:.1f}%"
        )

        # CSV incremental
        stats.append({
            'epoch': epoch + 1,
            'train_loss': tr['loss'], 'train_loss_body': tr['body'],
            'train_loss_hands': tr['hands'], 'train_loss_feet': tr['feet'], 'train_pck': tr['pck'],
            'val_loss': val['loss'], 'val_loss_body': val['body'],
            'val_loss_hands': val['hands'], 'val_loss_feet': val['feet'], 'val_pck': val['pck'],
            'test_loss': test['loss'], 'test_loss_body': test['body'],
            'test_loss_hands': test['hands'], 'test_loss_feet': test['feet'], 'test_pck': test['pck'],
        })
        fieldnames = list(stats[0].keys())
        with open(stats_path, 'w', newline='', encoding='utf-8') as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(stats)

        # Checkpoint
        if val['loss'] < best_val_loss:
            improvement = best_val_loss - val['loss']
            best_val_loss = val['loss']
            patience_counter = 0

            raw_model_to_save = model.module if isinstance(model, nn.DataParallel) else model
            torch.save({
                'epoch': epoch + 1,
                'state_dict': raw_model_to_save.state_dict(),
                'optimizer': optimizer.state_dict(),
                'val_loss': val['loss'],
                'val_pck': val['pck'],
                'num_joints': N_TOTAL,
            }, str(output_path))
            logger.info(f"  → Mejor modelo guardado (val_loss={best_val_loss:.6f}, mejora={improvement:.6f})")

            if use_telegram:
                vis_images = visualize_predictions(
                    model=raw_model_to_save,
                    dataset=test_dataset,
                    device=device,
                    indices=list(range(min(10, len(test_dataset)))),
                )
                collage, collage_path = save_epoch_visualizations(
                    vis_images,
                    output_dir=Path(args.output_model).parent / "visualizations",
                    epoch=epoch + 1,
                )
                msg = (
                    f"🏆 <b>Nuevo mejor modelo</b>\n"
                    f"{context_str}"
                    f"Val Loss: <b>{val['loss']:.6f}</b> (-{improvement:.6f})\n"
                    f"  body={val['body']:.5f} hands={val['hands']:.5f} feet={val['feet']:.5f}\n"
                    f"Val PCK: {val['pck']:.2f}%\n"
                    f"Test Loss: {test['loss']:.6f} | PCK: {test['pck']:.2f}%\n"
                    f"Época: {epoch + 1}"
                )
                send_telegram_with_images(msg, collage, token=telegram_token, chat_id=telegram_chat_id)
        else:
            patience_counter += 1
            logger.info(f"  → Sin mejora. Paciencia: {patience_counter}/{args.early_stopping}")
            if patience_counter >= args.early_stopping:
                logger.info("Early stopping activado.")
                break

    # ── Evaluación final con mejor modelo ──
    logger.info("Evaluando mejor modelo en test...")
    try:
        best_ckpt = torch.load(str(output_path), map_location=device, weights_only=False)
    except TypeError:
        best_ckpt = torch.load(str(output_path), map_location=device)

    raw_model_final = model.module if isinstance(model, nn.DataParallel) else model
    raw_model_final.load_state_dict(best_ckpt['state_dict'])
    final = evaluate(raw_model_final, test_loader, criterion, device, args.alpha_hands, args.alpha_feet)
    logger.info(
        f"Test final — loss={final['loss']:.6f} "
        f"(body={final['body']:.5f} hands={final['hands']:.5f} feet={final['feet']:.5f}) "
        f"pck={final['pck']:.2f}%"
    )

    if use_telegram:
        send_telegram(
            f"✅ <b>Entrenamiento finalizado</b>\n"
            f"Mejor Val Loss: {best_val_loss:.6f}\n"
            f"Test Loss: {final['loss']:.6f}\n"
            f"  body={final['body']:.5f} hands={final['hands']:.5f} feet={final['feet']:.5f}\n"
            f"Test PCK: {final['pck']:.2f}%",
            token=telegram_token, chat_id=telegram_chat_id
        )

    logger.info("✅ Proceso finalizado.")


if __name__ == '__main__':
    main()