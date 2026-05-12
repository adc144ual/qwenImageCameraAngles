import numpy as np
import pandas as pd
import cv2
import torch
import argparse
from pathlib import Path
from tqdm import tqdm
import sys
import os
from dotenv import load_dotenv
import urllib.request
import urllib.parse
import json
import logging
from typing import Optional

log = logging.getLogger(__name__)

# --- Keypoints config ---
HAND_KEYPOINTS = [17, 18, 19, 20, 21, 22]
FEET_KEYPOINTS = [29, 30, 31, 32]

def get_keypoints(mode: str):
    if mode == 'hands':
        return HAND_KEYPOINTS
    elif mode == 'feet':
        return FEET_KEYPOINTS
    elif mode == 'both':
        return HAND_KEYPOINTS + FEET_KEYPOINTS
    else:
        raise ValueError(f"Modo desconocido: {mode}")


# --- Telegram Notifier ---
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
                log.warning(f"[TELEGRAM] ✗ API respondió ok=false: {result}")
                print(f"[TELEGRAM] ✗ Fallo al enviar: {_preview!r}")
                return False
        log.info(f"[TELEGRAM] ✓ Enviado: {_preview!r}")
        print(f"[TELEGRAM] ✓ {_preview}")
        return True
    except Exception as e:
        log.warning(f"[TELEGRAM] ✗ Error: {e} | msg={_preview!r}")
        print(f"[TELEGRAM] ✗ Error enviando: {e}")
        return False


def notify_progress(processed, total, split, token=None, chat_id=None):
    percentage = (processed / total) * 100
    msg = (
        f"📊 <b>Procesamiento {split}</b>\n"
        f"Progreso: <b>{processed}/{total}</b> ({percentage:.1f}%)"
    )
    return send_telegram(msg, token=token, chat_id=chat_id)


def notify_completion(total_processed, total_skipped, token=None, chat_id=None):
    msg = (
        f"✅ <b>Procesamiento completado</b>\n"
        f"Procesados: <b>{total_processed}</b>\n"
        f"Saltados: {total_skipped}"
    )
    return send_telegram(msg, token=token, chat_id=chat_id)


# --- Arquitectura HRNet ---
BN_MOMENTUM = 0.1

class Bottleneck(torch.nn.Module):
    expansion = 4
    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super().__init__()
        self.conv1 = torch.nn.Conv2d(inplanes, planes, 1, bias=False)
        self.bn1 = torch.nn.BatchNorm2d(planes, momentum=BN_MOMENTUM)
        self.conv2 = torch.nn.Conv2d(planes, planes, 3, stride=stride, padding=1, bias=False)
        self.bn2 = torch.nn.BatchNorm2d(planes, momentum=BN_MOMENTUM)
        self.conv3 = torch.nn.Conv2d(planes, planes * self.expansion, 1, bias=False)
        self.bn3 = torch.nn.BatchNorm2d(planes * self.expansion, momentum=BN_MOMENTUM)
        self.relu = torch.nn.ReLU(inplace=True)
        self.downsample = downsample

    def forward(self, x):
        residual = x
        out = self.relu(self.bn1(self.conv1(x)))
        out = self.relu(self.bn2(self.conv2(out)))
        out = self.bn3(self.conv3(out))
        if self.downsample is not None:
            residual = self.downsample(x)
        return self.relu(out + residual)

class BasicBlock(torch.nn.Module):
    expansion = 1
    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super().__init__()
        self.conv1 = torch.nn.Conv2d(inplanes, planes, 3, stride=stride, padding=1, bias=False)
        self.bn1 = torch.nn.BatchNorm2d(planes, momentum=BN_MOMENTUM)
        self.conv2 = torch.nn.Conv2d(planes, planes, 3, padding=1, bias=False)
        self.bn2 = torch.nn.BatchNorm2d(planes, momentum=BN_MOMENTUM)
        self.relu = torch.nn.ReLU(inplace=True)
        self.downsample = downsample

    def forward(self, x):
        residual = x
        out = self.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        if self.downsample is not None:
            residual = self.downsample(x)
        return self.relu(out + residual)

class HighResolutionModule(torch.nn.Module):
    def __init__(self, num_branches, num_channels, num_blocks, multi_scale_output=True):
        super().__init__()
        self.num_branches = num_branches
        self.num_channels = num_channels
        self.multi_scale_output = multi_scale_output
        self.branches = self._make_branches(num_branches, num_channels, num_blocks)
        self.fuse_layers = self._make_fuse_layers()
        self.relu = torch.nn.ReLU(inplace=True)

    def _make_one_branch(self, branch_idx, num_channels, num_blocks):
        layers = []
        for _ in range(num_blocks):
            layers.append(BasicBlock(num_channels[branch_idx], num_channels[branch_idx]))
        return torch.nn.Sequential(*layers)

    def _make_branches(self, num_branches, num_channels, num_blocks):
        branches = []
        for i in range(num_branches):
            branches.append(self._make_one_branch(i, num_channels, num_blocks))
        return torch.nn.ModuleList(branches)

    def _make_fuse_layers(self):
        num_branches = self.num_branches
        num_channels = self.num_channels
        fuse_layers = []
        for i in range(num_branches if self.multi_scale_output else 1):
            fuse_layer = []
            for j in range(num_branches):
                if j > i:
                    fuse_layer.append(torch.nn.Sequential(
                        torch.nn.Conv2d(num_channels[j], num_channels[i], 1, bias=False),
                        torch.nn.BatchNorm2d(num_channels[i], momentum=BN_MOMENTUM),
                    ))
                elif j == i:
                    fuse_layer.append(None)
                else:
                    conv_downsamples = []
                    for k in range(i - j):
                        if k == i - j - 1:
                            conv_downsamples.append(torch.nn.Sequential(
                                torch.nn.Conv2d(num_channels[j], num_channels[i], 3, stride=2, padding=1, bias=False),
                                torch.nn.BatchNorm2d(num_channels[i], momentum=BN_MOMENTUM),
                            ))
                        else:
                            conv_downsamples.append(torch.nn.Sequential(
                                torch.nn.Conv2d(num_channels[j], num_channels[j], 3, stride=2, padding=1, bias=False),
                                torch.nn.BatchNorm2d(num_channels[j], momentum=BN_MOMENTUM),
                                torch.nn.ReLU(inplace=True),
                            ))
                    fuse_layer.append(torch.nn.Sequential(*conv_downsamples))
            fuse_layers.append(torch.nn.ModuleList(fuse_layer))
        return torch.nn.ModuleList(fuse_layers)

    def forward(self, x):
        for i in range(self.num_branches):
            x[i] = self.branches[i](x[i])
        x_fuse = []
        for i in range(len(self.fuse_layers)):
            y = 0
            for j in range(self.num_branches):
                if i == j:
                    y = y + x[j]
                elif j > i:
                    y = y + torch.nn.functional.interpolate(
                        self.fuse_layers[i][j](x[j]),
                        size=x[i].shape[2:],
                        mode='bilinear',
                        align_corners=True
                    )
                else:
                    y = y + self.fuse_layers[i][j](x[j])
            x_fuse.append(self.relu(y))
        return x_fuse

class PoseHRNet(torch.nn.Module):
    def __init__(self, width=48, num_joints=17):
        super().__init__()
        C = width
        self.conv1 = torch.nn.Conv2d(3, 64, 3, stride=2, padding=1, bias=False)
        self.bn1 = torch.nn.BatchNorm2d(64, momentum=BN_MOMENTUM)
        self.conv2 = torch.nn.Conv2d(64, 64, 3, stride=2, padding=1, bias=False)
        self.bn2 = torch.nn.BatchNorm2d(64, momentum=BN_MOMENTUM)
        self.relu = torch.nn.ReLU(inplace=True)
        downsample = torch.nn.Sequential(
            torch.nn.Conv2d(64, 256, 1, bias=False),
            torch.nn.BatchNorm2d(256, momentum=BN_MOMENTUM),
        )
        self.layer1 = torch.nn.Sequential(
            Bottleneck(64, 64, downsample=downsample),
            Bottleneck(256, 64),
            Bottleneck(256, 64),
            Bottleneck(256, 64),
        )
        self.transition1 = torch.nn.ModuleList([
            torch.nn.Sequential(
                torch.nn.Conv2d(256, C, 3, padding=1, bias=False),
                torch.nn.BatchNorm2d(C, momentum=BN_MOMENTUM),
                torch.nn.ReLU(inplace=True),
            ),
            torch.nn.Sequential(torch.nn.Sequential(
                torch.nn.Conv2d(256, C * 2, 3, stride=2, padding=1, bias=False),
                torch.nn.BatchNorm2d(C * 2, momentum=BN_MOMENTUM),
                torch.nn.ReLU(inplace=True),
            )),
        ])
        self.stage2 = torch.nn.Sequential(
            HighResolutionModule(2, [C, C * 2], num_blocks=4),
        )
        self.transition2 = torch.nn.ModuleList([
            None, None,
            torch.nn.Sequential(torch.nn.Sequential(
                torch.nn.Conv2d(C * 2, C * 4, 3, stride=2, padding=1, bias=False),
                torch.nn.BatchNorm2d(C * 4, momentum=BN_MOMENTUM),
                torch.nn.ReLU(inplace=True),
            )),
        ])
        stage3_modules = []
        for i in range(4):
            stage3_modules.append(
                HighResolutionModule(3, [C, C * 2, C * 4], num_blocks=4, multi_scale_output=True)
            )
        self.stage3 = torch.nn.Sequential(*stage3_modules)
        self.transition3 = torch.nn.ModuleList([
            None, None, None,
            torch.nn.Sequential(torch.nn.Sequential(
                torch.nn.Conv2d(C * 4, C * 8, 3, stride=2, padding=1, bias=False),
                torch.nn.BatchNorm2d(C * 8, momentum=BN_MOMENTUM),
                torch.nn.ReLU(inplace=True),
            )),
        ])
        stage4_modules = []
        for i in range(3):
            multi_scale_output = True if i < 2 else False
            stage4_modules.append(
                HighResolutionModule(4, [C, C * 2, C * 4, C * 8], num_blocks=4,
                                     multi_scale_output=multi_scale_output)
            )
        self.stage4 = torch.nn.Sequential(*stage4_modules)
        self.final_layer = torch.nn.Conv2d(C, num_joints, 1)

    def forward(self, x):
        x = self.relu(self.bn1(self.conv1(x)))
        x = self.relu(self.bn2(self.conv2(x)))
        x = self.layer1(x)
        x_list = []
        for i in range(2):
            x_list.append(self.transition1[i](x))
        y_list = self.stage2[0](x_list)
        x_list = []
        for i in range(3):
            if self.transition2[i] is not None:
                x_list.append(self.transition2[i](y_list[-1]))
            else:
                x_list.append(y_list[i])
        y_list = x_list
        for module in self.stage3:
            y_list = module(y_list)
        x_list = []
        for i in range(4):
            if self.transition3[i] is not None:
                x_list.append(self.transition3[i](y_list[-1]))
            else:
                x_list.append(y_list[i])
        y_list = x_list
        for module in self.stage4:
            y_list = module(y_list)
        x = self.final_layer(y_list[0])
        return x


# --- Funciones auxiliares ---
def load_hrnet_model(model_path, device):
    model = PoseHRNet(width=48, num_joints=17)
    try:
        checkpoint = torch.load(model_path, map_location=device, weights_only=False)
    except TypeError:
        checkpoint = torch.load(model_path, map_location=device)
    if isinstance(checkpoint, dict) and 'state_dict' in checkpoint:
        checkpoint = checkpoint['state_dict']
    model.load_state_dict(checkpoint, strict=True)
    model.eval()
    model.to(device)
    return model


def preprocess_image_hrnet(image_bgr, target_size=(288, 384)):
    """Preprocesar imagen para HRNet"""
    # 1. Resize a 512x512
    img_512 = cv2.resize(image_bgr, (512, 512), interpolation=cv2.INTER_LINEAR)

    # 2. Resize a target_size (288, 384)
    h, w = target_size
    img_resized = cv2.resize(img_512, (w, h), interpolation=cv2.INTER_LINEAR)

    # 3. Normalización ImageNet
    mean = np.array([0.485, 0.456, 0.406], dtype=np.float32)
    std = np.array([0.229, 0.224, 0.225], dtype=np.float32)
    img_rgb = cv2.cvtColor(img_resized, cv2.COLOR_BGR2RGB).astype(np.float32)
    img_rgb = img_rgb / 255.0
    img_rgb = (img_rgb - mean) / std

    # 4. To tensor
    tensor = torch.from_numpy(img_rgb.transpose(2, 0, 1)).float()
    return tensor, img_resized


def transform_mediapipe_to_heatmap(x_norm, y_norm, heatmap_w, heatmap_h):
    """Transformar coords MediaPipe normalizadas a espacio heatmap HRNet"""
    # 1. De normalizado a píxeles 160x120
    x_160 = x_norm * 160
    y_120 = y_norm * 120

    # 2. Escalar a 512x512
    x_512 = x_160 * (512 / 160)
    y_512 = y_120 * (512 / 120)

    # 3. Escalar a 384x288
    x_384 = x_512 * (384 / 512)
    y_288 = y_512 * (288 / 512)

    # 4. Escalar a resolución heatmap
    x_hm = x_384 * (heatmap_w / 384)
    y_hm = y_288 * (heatmap_h / 288)
    return x_hm, y_hm


def create_mediapipe_heatmaps(df, keypoints, heatmap_shape, sigma=2.0):
    """Crear heatmaps de MediaPipe IDÉNTICOS a los de HRNet"""
    heatmap_h, heatmap_w = heatmap_shape
    mediapipe_heatmaps = []
    for kp_idx in keypoints:
        row = df[df['landmark_id'] == kp_idx]
        heatmap = np.zeros((heatmap_h, heatmap_w), dtype=np.float32)
        if not row.empty:
            x_norm = row['x'].values[0]
            y_norm = row['y'].values[0]

            x_hm, y_hm = transform_mediapipe_to_heatmap(x_norm, y_norm, heatmap_w, heatmap_h)

            # Generar gaussiana 2D igual que HRNet
            size = int(6 * sigma + 1)
            x0 = y0 = size // 2
            y_range = np.arange(0, size, 1, dtype=np.float32)
            x_range = np.arange(0, size, 1, dtype=np.float32)
            y_grid, x_grid = np.meshgrid(y_range, x_range, indexing='ij')

            gaussian = np.exp(-((x_grid - x0)**2 + (y_grid - y0)**2) / (2 * sigma**2))

            x_int, y_int = int(round(x_hm)), int(round(y_hm))

            x_start = max(0, x_int - x0)
            y_start = max(0, y_int - y0)
            x_end = min(heatmap_w, x_int + x0 + 1)
            y_end = min(heatmap_h, y_int + y0 + 1)

            g_x_start = max(0, x0 - x_int)
            g_y_start = max(0, y0 - y_int)
            g_x_end = g_x_start + (x_end - x_start)
            g_y_end = g_y_start + (y_end - y_start)

            if x_end > x_start and y_end > y_start:
                heatmap[y_start:y_end, x_start:x_end] = np.maximum(
                    heatmap[y_start:y_end, x_start:x_end],
                    gaussian[g_y_start:g_y_end, g_x_start:g_x_end]
                )
        mediapipe_heatmaps.append(heatmap)
    return np.array(mediapipe_heatmaps)


def collect_csv_image_pairs(csv_root, img_root, split):
    """Recolectar pares CSV-imagen para un split (train_val o test)"""
    pairs = []
    csv_split_path = Path(csv_root) / split
    img_split_path = Path(img_root) / split

    # Buscar en las carpetas 00_15, 00_16, 00_17
    for view_folder in ['00_15', '00_16', '00_17']:
        csv_view_path = csv_split_path / view_folder
        img_view_path = img_split_path / view_folder
        if not csv_view_path.exists():
            continue

        # Buscar todos los CSVs _rgb
        for csv_file in csv_view_path.glob('*_rgb.csv'):
            # Construir nombre de imagen correspondiente
            img_name = csv_file.stem + '.png'  # Cambiar .csv por .png
            img_file = img_view_path / img_name
            if img_file.exists():
                pairs.append({
                    'csv_path': csv_file,
                    'img_path': img_file,
                    'basename': csv_file.stem,   # nombre sin extensión
                    'split': split
                })
    return pairs


def process_batch(model, image_tensors, device):
    """Procesar batch de imágenes con HRNet"""
    batch = torch.stack(image_tensors).to(device)
    with torch.no_grad():
        heatmaps = model(batch)
    return heatmaps.cpu().numpy()


# --- Modo append: añade canales de pies a .npy existentes ---
def run_append_feet(args, npy_root, telegram_token, telegram_chat_id, use_telegram):
    """
    Carga cada .npy existente con 23 canales (17 HRNet + 6 manos) y
    añade 4 canales de pies (landmarks 29-32). Requiere CSV pero NO el modelo.
    Los .npy con 27 canales ya están completos y se saltan.
    Los .npy inexistentes se procesan desde cero (requiere modelo e imágenes).
    """
    all_pairs = []
    for split in ['train_val', 'test']:
        pairs = collect_csv_image_pairs(args.csv_root, args.img_root, split)
        all_pairs.extend(pairs)

    needs_full = []    # no existe el .npy → hay que generarlo desde cero
    needs_append = []  # existe con 23 canales → solo añadir pies
    already_done = []  # existe con 27 canales → saltar

    for pair in all_pairs:
        npy_path = npy_root / pair['split'] / f"{pair['basename']}.npy"
        if not npy_path.exists():
            needs_full.append(pair)
        else:
            existing = np.load(npy_path, mmap_mode='r')
            n_channels = existing.shape[0]
            if n_channels == 27:
                already_done.append(pair)
            elif n_channels == 23:
                needs_append.append(pair)
            else:
                print(f"[WARN] Shape inesperada {existing.shape} en {npy_path}, saltando")

    print(f"Append mode — ya completos: {len(already_done)}, "
          f"a append: {len(needs_append)}, a generar desde cero: {len(needs_full)}")

    if use_telegram:
        send_telegram(
            f"🚀 <b>Append pies iniciado</b>\n"
            f"Ya completos: {len(already_done)}\n"
            f"Append: {len(needs_append)}\n"
            f"Desde cero: {len(needs_full)}",
            token=telegram_token, chat_id=telegram_chat_id
        )

    total_processed = 0
    notification_interval = max(100, (len(needs_append) + len(needs_full)) // 10)

    # --- Paso 1: append sobre los que ya tienen 23 canales ---
    if needs_append:
        print(f"\nPaso 1/2: Appending pies a {len(needs_append)} archivos...")
        with tqdm(total=len(needs_append), desc="Append pies") as pbar:
            for pair in needs_append:
                npy_path = npy_root / pair['split'] / f"{pair['basename']}.npy"
                existing = np.load(npy_path)  # (23, H, W)
                heatmap_h, heatmap_w = existing.shape[1], existing.shape[2]

                df = pd.read_csv(pair['csv_path'])
                feet_heatmaps = create_mediapipe_heatmaps(
                    df, FEET_KEYPOINTS, (heatmap_h, heatmap_w)
                )  # (4, H, W)

                combined = np.concatenate([existing, feet_heatmaps], axis=0)  # (27, H, W)
                np.save(npy_path, combined)

                total_processed += 1
                pbar.update(1)

                if use_telegram and total_processed % notification_interval == 0:
                    notify_progress(total_processed, len(needs_append) + len(needs_full),
                                    "append_pies", token=telegram_token, chat_id=telegram_chat_id)

    # --- Paso 2: generar desde cero los que no existen ---
    if needs_full:
        print(f"\nPaso 2/2: Generando desde cero {len(needs_full)} archivos...")
        model = load_hrnet_model(args.model_path, args.device)
        img_root_out = npy_root.parent / 'images'

        i = 0
        with tqdm(total=len(needs_full), desc="Desde cero") as pbar:
            while i < len(needs_full):
                batch_pairs = needs_full[i:i + args.batch_size]
                batch_images_288x384 = []
                batch_tensors = []
                batch_dfs = []

                for pair in batch_pairs:
                    img_bgr = cv2.imread(str(pair['img_path']))
                    tensor, img_288x384 = preprocess_image_hrnet(img_bgr)
                    df = pd.read_csv(pair['csv_path'])
                    batch_images_288x384.append(img_288x384)
                    batch_tensors.append(tensor)
                    batch_dfs.append(df)

                hrnet_heatmaps_batch = process_batch(model, batch_tensors, args.device)

                for j, pair in enumerate(batch_pairs):
                    hrnet_heatmaps = hrnet_heatmaps_batch[j]
                    heatmap_h = hrnet_heatmaps.shape[1]
                    heatmap_w = hrnet_heatmaps.shape[2]

                    # Manos + pies
                    mp_heatmaps = create_mediapipe_heatmaps(
                        batch_dfs[j], HAND_KEYPOINTS + FEET_KEYPOINTS, (heatmap_h, heatmap_w)
                    )
                    combined = np.concatenate([hrnet_heatmaps, mp_heatmaps], axis=0)  # (27, H, W)

                    npy_path = npy_root / pair['split'] / f"{pair['basename']}.npy"
                    np.save(npy_path, combined)

                    img_out = img_root_out / pair['split'] / f"{pair['basename']}.png"
                    img_out.parent.mkdir(parents=True, exist_ok=True)
                    cv2.imwrite(str(img_out), batch_images_288x384[j])

                    total_processed += 1

                i += args.batch_size
                pbar.update(len(batch_pairs))

                if use_telegram and total_processed % notification_interval == 0:
                    notify_progress(total_processed, len(needs_append) + len(needs_full),
                                    "append_pies", token=telegram_token, chat_id=telegram_chat_id)

    if use_telegram:
        notify_completion(total_processed, len(already_done),
                          token=telegram_token, chat_id=telegram_chat_id)

    print(f"\nCompletado. Procesados: {total_processed}, ya completos (saltados): {len(already_done)}")


# --- Modo normal: generación desde cero ---
def run_normal(args, npy_root, img_root_out, telegram_token, telegram_chat_id, use_telegram):
    keypoints = get_keypoints(args.keypoints)
    n_mp_channels = len(keypoints)
    expected_channels = 17 + n_mp_channels  # HRNet + mediapipe

    all_pairs = []
    for split in ['train_val', 'test']:
        pairs = collect_csv_image_pairs(args.csv_root, args.img_root, split)
        all_pairs.extend(pairs)
        print(f"Split {split}: {len(pairs)} pares encontrados")

    if not args.overwrite:
        filtered_pairs = []
        for pair in all_pairs:
            npy_path = npy_root / pair['split'] / f"{pair['basename']}.npy"
            img_path = img_root_out / pair['split'] / f"{pair['basename']}.png"
            if not (npy_path.exists() and img_path.exists()):
                filtered_pairs.append(pair)
        skipped = len(all_pairs) - len(filtered_pairs)
        print(f"Modo continuar: {len(filtered_pairs)} a procesar, {skipped} ya existen")
        all_pairs = filtered_pairs

    if len(all_pairs) == 0:
        print("No hay pares para procesar")
        return

    print(f"Total: {len(all_pairs)} pares | keypoints: {args.keypoints} "
          f"({n_mp_channels} canales MP) → {expected_channels} canales totales")

    if use_telegram:
        send_telegram(
            f"🚀 <b>Iniciando procesamiento</b>\n"
            f"Total: {len(all_pairs)} muestras\n"
            f"Keypoints: {args.keypoints} ({expected_channels} canales)\n"
            f"Batch size: {args.batch_size}\n"
            f"Device: {args.device}",
            token=telegram_token, chat_id=telegram_chat_id
        )

    model = load_hrnet_model(args.model_path, args.device)

    i = 0
    total_processed = 0
    notification_interval = max(100, len(all_pairs) // 10)

    with tqdm(total=len(all_pairs), desc="Procesando") as pbar:
        while i < len(all_pairs):
            batch_pairs = all_pairs[i:i + args.batch_size]
            batch_images_288x384 = []
            batch_tensors = []
            batch_dfs = []

            for pair in batch_pairs:
                img_bgr = cv2.imread(str(pair['img_path']))
                tensor, img_288x384 = preprocess_image_hrnet(img_bgr)
                df = pd.read_csv(pair['csv_path'])
                batch_images_288x384.append(img_288x384)
                batch_tensors.append(tensor)
                batch_dfs.append(df)

            hrnet_heatmaps_batch = process_batch(model, batch_tensors, args.device)

            for j, pair in enumerate(batch_pairs):
                hrnet_heatmaps = hrnet_heatmaps_batch[j]
                heatmap_h, heatmap_w = hrnet_heatmaps.shape[1], hrnet_heatmaps.shape[2]

                mp_heatmaps = create_mediapipe_heatmaps(
                    batch_dfs[j], keypoints, (heatmap_h, heatmap_w)
                )
                combined = np.concatenate([hrnet_heatmaps, mp_heatmaps], axis=0)

                npy_path = npy_root / pair['split'] / f"{pair['basename']}.npy"
                np.save(npy_path, combined)

                img_out = img_root_out / pair['split'] / f"{pair['basename']}.png"
                cv2.imwrite(str(img_out), batch_images_288x384[j])

                total_processed += 1

            i += args.batch_size
            pbar.update(len(batch_pairs))

            if use_telegram and total_processed % notification_interval == 0:
                notify_progress(total_processed, len(all_pairs), "heatmaps",
                                token=telegram_token, chat_id=telegram_chat_id)

    if use_telegram:
        skipped = 0 if args.overwrite else (
            sum(len(collect_csv_image_pairs(args.csv_root, args.img_root, s))
                for s in ['train_val', 'test']) - len(all_pairs)
        )
        notify_completion(total_processed, skipped,
                          token=telegram_token, chat_id=telegram_chat_id)

    print("Procesamiento completado")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--csv_root', type=str, required=True)
    parser.add_argument('--img_root', type=str, required=True)
    parser.add_argument('--model_path', type=str, default="/nas/antoniodetoro/qwen/Qwen-Image-Edit-Angles-2/src/models/pose_hrnet_w48_384x288.pth")
    parser.add_argument('--output_root', type=str, required=True)
    parser.add_argument('--batch_size', type=int, default=8)
    parser.add_argument('--device', type=str, default='cuda')
    parser.add_argument('--overwrite', action='store_true',
                        help='Sobrescribir archivos existentes')
    parser.add_argument('--keypoints', type=str, default='hands',
                        choices=['hands', 'feet', 'both'],
                        help='Qué keypoints de MediaPipe incluir: '
                             'hands (17+6=23ch), feet (17+4=21ch), both (17+10=27ch). '
                             'Default: hands')
    parser.add_argument('--append_feet', action='store_true',
                        help='Modo append: añade canales de pies (landmarks 29-32 de mediapipe) a .npy existentes con '
                             '23 canales (17 HRNet + 6 manos) sin regenerar HRNet. '
                             'Resultado: 27 canales. Los ya completos (27ch) se saltan.')

    args = parser.parse_args()

    load_dotenv()
    telegram_token = os.environ.get("TELEGRAM_BOT_TOKEN")
    telegram_chat_id = os.environ.get("TELEGRAM_CHAT_ID")
    use_telegram = bool(telegram_token and telegram_chat_id)
    print(f"[TELEGRAM] {'✓ Configurado' if use_telegram else '✗ No configurado'}")

    output_root = Path(args.output_root)
    npy_root = output_root / 'heatmaps'
    img_root_out = output_root / 'images'

    for split in ['train_val', 'test']:
        (npy_root / split).mkdir(parents=True, exist_ok=True)
        (img_root_out / split).mkdir(parents=True, exist_ok=True)

    if args.append_feet:
        if args.keypoints != 'hands':
            print("[WARN] --append_feet ignora --keypoints, siempre añade FEET_KEYPOINTS")
        run_append_feet(args, npy_root, telegram_token, telegram_chat_id, use_telegram)
    else:
        run_normal(args, npy_root, img_root_out, telegram_token, telegram_chat_id, use_telegram)


if __name__ == '__main__':
    main()