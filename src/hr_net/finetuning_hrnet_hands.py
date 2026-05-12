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

# --- Telegram Notifier ---
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


def notify_progress(
    processed: int,
    total: int,
    split: str,
    token: Optional[str] = None,
    chat_id: Optional[str] = None,
) -> bool:
    """Notificación de progreso."""
    percentage = (processed / total) * 100
    msg = (
        f"📊 <b>Procesamiento {split}</b>\n"
        f"Progreso: <b>{processed}/{total}</b> ({percentage:.1f}%)"
    )
    return send_telegram(msg, token=token, chat_id=chat_id)


def notify_completion(
    total_processed: int,
    total_skipped: int,
    token: Optional[str] = None,
    chat_id: Optional[str] = None,
) -> bool:
    """Notificación de finalización."""
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
    """Cargar modelo HRNet"""
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


def create_mediapipe_heatmaps(df, hand_keypoints, heatmap_shape, sigma=2.0):
    """Crear heatmaps de MediaPipe IDÉNTICOS a los de HRNet"""
    heatmap_h, heatmap_w = heatmap_shape
    mediapipe_heatmaps = []
    
    for kp_idx in hand_keypoints:
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
                    'basename': csv_file.stem,  # nombre sin extensión
                    'split': split
                })
    
    return pairs


def process_batch(model, image_tensors, device):
    """Procesar batch de imágenes con HRNet"""
    batch = torch.stack(image_tensors).to(device)
    
    with torch.no_grad():
        heatmaps = model(batch)
    
    return heatmaps.cpu().numpy()



def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--csv_root', type=str, required=True,
                        help='Ruta raíz de CSVs')
    parser.add_argument('--img_root', type=str, required=True,
                        help='Ruta raíz de imágenes')
    parser.add_argument('--model_path', type=str, required=False, default="/nas/antoniodetoro/qwen/Qwen-Image-Edit-Angles-2/src/models/pose_hrnet_w48_384x288.pth",
                        help='Ruta del modelo HRNet')
    parser.add_argument('--output_root', type=str, required=True,
                        help='Ruta raíz de salida')
    parser.add_argument('--batch_size', type=int, default=8,
                        help='Batch size para HRNet')
    parser.add_argument('--device', type=str, default='cuda',
                        help='Device (cuda o cpu)')
    parser.add_argument('--overwrite', action='store_true',
                        help='Sobrescribir archivos existentes. Si no se especifica, se saltan los ya procesados')
    
    args = parser.parse_args()
    
    # Cargar .env si existe
    load_dotenv()
    telegram_token = os.environ.get("TELEGRAM_BOT_TOKEN")
    telegram_chat_id = os.environ.get("TELEGRAM_CHAT_ID")
    use_telegram = telegram_token and telegram_chat_id
    
    if use_telegram:
        print(f"[TELEGRAM] ✓ Configurado")
    else:
        print(f"[TELEGRAM] ✗ No configurado (falta .env o variables)")
    
    HAND_KEYPOINTS = [17, 18, 19, 20, 21, 22]
    
    # Crear carpetas de salida
    output_root = Path(args.output_root)
    npy_root = output_root / 'heatmaps'
    img_root = output_root / 'images'
    
    for split in ['train_val', 'test']:
        (npy_root / split).mkdir(parents=True, exist_ok=True)
        (img_root / split).mkdir(parents=True, exist_ok=True)
    
    # Cargar modelo
    print(f"Cargando modelo en {args.device}...")
    model = load_hrnet_model(args.model_path, args.device)
    
    # Recolectar pares para ambos splits
    all_pairs = []
    for split in ['train_val', 'test']:
        pairs = collect_csv_image_pairs(args.csv_root, args.img_root, split)
        all_pairs.extend(pairs)
        print(f"Split {split}: {len(pairs)} pares encontrados")
    
    # Filtrar pares ya procesados si no se usa --overwrite
    if not args.overwrite:
        filtered_pairs = []
        for pair in all_pairs:
            npy_output_path = npy_root / pair['split'] / f"{pair['basename']}.npy"
            img_output_path = img_root / pair['split'] / f"{pair['basename']}.png"
            
            if not (npy_output_path.exists() and img_output_path.exists()):
                filtered_pairs.append(pair)
        
        skipped = len(all_pairs) - len(filtered_pairs)
        print(f"Modo continuar: {len(filtered_pairs)} a procesar, {skipped} ya existen")
        all_pairs = filtered_pairs
    
    if len(all_pairs) == 0:
        print("No hay pares para procesar")
        return
    
    print(f"Total: {len(all_pairs)} pares a procesar")
    
    # Notificar inicio
    if use_telegram:
        send_telegram(
            f"🚀 <b>Iniciando procesamiento</b>\n"
            f"Total: {len(all_pairs)} muestras\n"
            f"Batch size: {args.batch_size}\n"
            f"Device: {args.device}",
            token=telegram_token,
            chat_id=telegram_chat_id
        )
    
    # Procesar en batches
    i = 0
    total_processed = 0
    notification_interval = max(100, len(all_pairs) // 10)  # Notificar cada 10% o mínimo cada 100
    
    with tqdm(total=len(all_pairs), desc="Procesando") as pbar:
        while i < len(all_pairs):
            batch_pairs = all_pairs[i:i + args.batch_size]
            
            # Cargar imágenes y CSVs del batch
            batch_images_bgr = []
            batch_images_288x384 = []
            batch_tensors = []
            batch_dfs = []
            
            for pair in batch_pairs:
                img_bgr = cv2.imread(str(pair['img_path']))
                tensor, img_288x384 = preprocess_image_hrnet(img_bgr)
                df = pd.read_csv(pair['csv_path'])
                
                batch_images_bgr.append(img_bgr)
                batch_images_288x384.append(img_288x384)
                batch_tensors.append(tensor)
                batch_dfs.append(df)
            
            # Procesar batch con HRNet
            hrnet_heatmaps_batch = process_batch(model, batch_tensors, args.device)
            
            # Procesar cada muestra del batch
            for j, pair in enumerate(batch_pairs):
                hrnet_heatmaps = hrnet_heatmaps_batch[j]
                heatmap_h, heatmap_w = hrnet_heatmaps.shape[1], hrnet_heatmaps.shape[2]
                
                # Crear heatmaps MediaPipe
                mediapipe_heatmaps = create_mediapipe_heatmaps(
                    batch_dfs[j], HAND_KEYPOINTS, (heatmap_h, heatmap_w)
                )
                
                # Combinar heatmaps
                combined_heatmaps = np.concatenate([hrnet_heatmaps, mediapipe_heatmaps], axis=0)
                
                # Guardar .npy
                npy_output_path = npy_root / pair['split'] / f"{pair['basename']}.npy"
                np.save(npy_output_path, combined_heatmaps)
                
                # Guardar imagen 288x384
                img_output_path = img_root / pair['split'] / f"{pair['basename']}.png"
                cv2.imwrite(str(img_output_path), batch_images_288x384[j])
                
                total_processed += 1
            
            i += args.batch_size
            pbar.update(len(batch_pairs))
            
            # Notificar progreso periódicamente
            if use_telegram and total_processed % notification_interval == 0:
                notify_progress(
                    total_processed,
                    len(all_pairs),
                    "heatmaps",
                    token=telegram_token,
                    chat_id=telegram_chat_id
                )
    
    # Notificar finalización
    if use_telegram:
        skipped = len(collect_csv_image_pairs(args.csv_root, args.img_root, 'train_val')) + \
                  len(collect_csv_image_pairs(args.csv_root, args.img_root, 'test')) - len(all_pairs)
        notify_completion(
            total_processed,
            skipped if not args.overwrite else 0,
            token=telegram_token,
            chat_id=telegram_chat_id
        )
    
    print("Procesamiento completado")


if __name__ == '__main__':
    main()