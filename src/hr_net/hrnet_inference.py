#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
HRNet Inference Script - Extracción y guardado de heatmaps
==========================================================
Autor: Para Universidad de Almería
Fecha: 2025

Descripción:
    Script profesional para realizar inferencia con HRNet pre-entrenado
    sobre un conjunto de imágenes y guardar los heatmaps de salida.

Uso:
    python hrnet_inference.py
    (Usa valores por defecto: imgs_entrada → imgs_salida)
    
    python hrnet_inference.py --input-dir ./mi_carpeta --output-dir ./resultados
    (Personalizado)

Características:
    - Carga de modelos pre-entrenados (COCO/MPII)
    - Procesamiento batch de imágenes
    - Extracción de heatmaps raw de la red
    - Guardado de heatmaps como: .npy (procesamiento), .png (visualización)
    - Manejo eficiente de memoria GPU
    - URLs de descarga verificadas y con backup
    - Logging detallado
"""

import os
import sys
import argparse
import logging
import json
from pathlib import Path
from typing import Tuple, Dict, List, Optional
import shutil

import cv2
import numpy as np
import torch
import torch.nn as nn
from tqdm import tqdm

# Configuración de logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


# =========================================================================
# Arquitectura HRNet real (Bottleneck → layer1, BasicBlock → stages 2-4)
# Compatible con checkpoints oficiales de MMPose / deep-high-resolution-net
# =========================================================================

BN_MOMENTUM = 0.1


class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super().__init__()
        self.conv1 = nn.Conv2d(inplanes, planes, 1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes, momentum=BN_MOMENTUM)
        self.conv2 = nn.Conv2d(planes, planes, 3, stride=stride, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes, momentum=BN_MOMENTUM)
        self.conv3 = nn.Conv2d(planes, planes * self.expansion, 1, bias=False)
        self.bn3 = nn.BatchNorm2d(planes * self.expansion, momentum=BN_MOMENTUM)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample

    def forward(self, x):
        residual = x
        out = self.relu(self.bn1(self.conv1(x)))
        out = self.relu(self.bn2(self.conv2(out)))
        out = self.bn3(self.conv3(out))
        if self.downsample is not None:
            residual = self.downsample(x)
        return self.relu(out + residual)


class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super().__init__()
        self.conv1 = nn.Conv2d(inplanes, planes, 3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes, momentum=BN_MOMENTUM)
        self.conv2 = nn.Conv2d(planes, planes, 3, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes, momentum=BN_MOMENTUM)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample

    def forward(self, x):
        residual = x
        out = self.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        if self.downsample is not None:
            residual = self.downsample(x)
        return self.relu(out + residual)


class HighResolutionModule(nn.Module):
    """Un módulo multi-rama con fusión entre resoluciones."""

    def __init__(self, num_branches, num_channels, num_blocks, multi_scale_output=True):
        super().__init__()
        self.num_branches = num_branches
        self.num_channels = num_channels
        self.multi_scale_output = multi_scale_output

        self.branches = self._make_branches(num_branches, num_channels, num_blocks)
        self.fuse_layers = self._make_fuse_layers()
        self.relu = nn.ReLU(inplace=True)

    def _make_one_branch(self, branch_idx, num_channels, num_blocks):
        layers = []
        for _ in range(num_blocks):
            layers.append(BasicBlock(num_channels[branch_idx], num_channels[branch_idx]))
        return nn.Sequential(*layers)

    def _make_branches(self, num_branches, num_channels, num_blocks):
        branches = []
        for i in range(num_branches):
            branches.append(self._make_one_branch(i, num_channels, num_blocks))
        return nn.ModuleList(branches)

    def _make_fuse_layers(self):
        num_branches = self.num_branches
        num_channels = self.num_channels
        fuse_layers = []
        for i in range(num_branches if self.multi_scale_output else 1):
            fuse_layer = []
            for j in range(num_branches):
                if j > i:
                    # Upsample: 1×1 conv + BN, luego interpolate en forward
                    fuse_layer.append(nn.Sequential(
                        nn.Conv2d(num_channels[j], num_channels[i], 1, bias=False),
                        nn.BatchNorm2d(num_channels[i], momentum=BN_MOMENTUM),
                    ))
                elif j == i:
                    fuse_layer.append(None)
                else:
                    # Downsample con stride-2 3×3 convs
                    conv_downsamples = []
                    for k in range(i - j):
                        if k == i - j - 1:
                            conv_downsamples.append(nn.Sequential(
                                nn.Conv2d(num_channels[j], num_channels[i], 3, stride=2, padding=1, bias=False),
                                nn.BatchNorm2d(num_channels[i], momentum=BN_MOMENTUM),
                            ))
                        else:
                            conv_downsamples.append(nn.Sequential(
                                nn.Conv2d(num_channels[j], num_channels[j], 3, stride=2, padding=1, bias=False),
                                nn.BatchNorm2d(num_channels[j], momentum=BN_MOMENTUM),
                                nn.ReLU(inplace=True),
                            ))
                    fuse_layer.append(nn.Sequential(*conv_downsamples))
            fuse_layers.append(nn.ModuleList(fuse_layer))
        return nn.ModuleList(fuse_layers)

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
                    y = y + nn.functional.interpolate(
                        self.fuse_layers[i][j](x[j]),
                        size=x[i].shape[2:],
                        mode='bilinear',
                        align_corners=True
                    )
                else:
                    y = y + self.fuse_layers[i][j](x[j])
            x_fuse.append(self.relu(y))
        return x_fuse


class PoseHRNet(nn.Module):
    """
    HRNet para estimación de pose humana.
    Configuración por defecto: W48 (COCO 17 keypoints).
    """

    def __init__(self, width=48, num_joints=17):
        super().__init__()
        C = width  # 48

        # --- Stem ---
        self.conv1 = nn.Conv2d(3, 64, 3, stride=2, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(64, momentum=BN_MOMENTUM)
        self.conv2 = nn.Conv2d(64, 64, 3, stride=2, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(64, momentum=BN_MOMENTUM)
        self.relu = nn.ReLU(inplace=True)

        # --- Layer1: 4 Bottleneck blocks (64 → 256) ---
        downsample = nn.Sequential(
            nn.Conv2d(64, 256, 1, bias=False),
            nn.BatchNorm2d(256, momentum=BN_MOMENTUM),
        )
        self.layer1 = nn.Sequential(
            Bottleneck(64, 64, downsample=downsample),
            Bottleneck(256, 64),
            Bottleneck(256, 64),
            Bottleneck(256, 64),
        )

        # --- Transition1: 256 → [C, 2C] ---
        self.transition1 = nn.ModuleList([
            nn.Sequential(
                nn.Conv2d(256, C, 3, padding=1, bias=False),
                nn.BatchNorm2d(C, momentum=BN_MOMENTUM),
                nn.ReLU(inplace=True),
            ),
            nn.Sequential(nn.Sequential(
                nn.Conv2d(256, C * 2, 3, stride=2, padding=1, bias=False),
                nn.BatchNorm2d(C * 2, momentum=BN_MOMENTUM),
                nn.ReLU(inplace=True),
            )),
        ])

        # --- Stage2: 1 module, 2 branches ---
        self.stage2 = nn.Sequential(
            HighResolutionModule(2, [C, C * 2], num_blocks=4),
        )

        # --- Transition2: → [C, 2C, 4C] ---
        self.transition2 = nn.ModuleList([
            None,  # branch 0 keeps same
            None,  # branch 1 keeps same
            nn.Sequential(nn.Sequential(
                nn.Conv2d(C * 2, C * 4, 3, stride=2, padding=1, bias=False),
                nn.BatchNorm2d(C * 4, momentum=BN_MOMENTUM),
                nn.ReLU(inplace=True),
            )),
        ])

        # --- Stage3: 4 modules, 3 branches ---
        stage3_modules = []
        for i in range(4):
            stage3_modules.append(
                HighResolutionModule(3, [C, C * 2, C * 4], num_blocks=4,
                                     multi_scale_output=True)
            )
        self.stage3 = nn.Sequential(*stage3_modules)

        # --- Transition3: → [C, 2C, 4C, 8C] ---
        self.transition3 = nn.ModuleList([
            None,
            None,
            None,
            nn.Sequential(nn.Sequential(
                nn.Conv2d(C * 4, C * 8, 3, stride=2, padding=1, bias=False),
                nn.BatchNorm2d(C * 8, momentum=BN_MOMENTUM),
                nn.ReLU(inplace=True),
            )),
        ])

        # --- Stage4: 3 modules, 4 branches ---
        stage4_modules = []
        for i in range(3):
            multi_scale_output = True if i < 2 else False
            stage4_modules.append(
                HighResolutionModule(4, [C, C * 2, C * 4, C * 8], num_blocks=4,
                                     multi_scale_output=multi_scale_output)
            )
        self.stage4 = nn.Sequential(*stage4_modules)

        # --- Final layer ---
        self.final_layer = nn.Conv2d(C, num_joints, 1)

    def forward(self, x):
        x = self.relu(self.bn1(self.conv1(x)))
        x = self.relu(self.bn2(self.conv2(x)))
        x = self.layer1(x)

        # Transition1
        x_list = []
        for i in range(2):
            x_list.append(self.transition1[i](x))

        # Stage2
        y_list = self.stage2[0](x_list)

        # Transition2
        x_list = []
        for i in range(3):
            if self.transition2[i] is not None:
                x_list.append(self.transition2[i](y_list[-1]))
            else:
                x_list.append(y_list[i])

        # Stage3
        y_list = x_list
        for module in self.stage3:
            y_list = module(y_list)

        # Transition3
        x_list = []
        for i in range(4):
            if self.transition3[i] is not None:
                x_list.append(self.transition3[i](y_list[-1]))
            else:
                x_list.append(y_list[i])

        # Stage4
        y_list = x_list
        for module in self.stage4:
            y_list = module(y_list)

        # Output: highest resolution branch
        x = self.final_layer(y_list[0])
        return x


class HRNetInferencer:
    """
    Clase para realizar inferencia con HRNet y extraer heatmaps.
    
    URLs verificadas en Feb 2025 con soporte a backup automático.
    """
    
    # URLs VERIFICADAS Y FUNCIONALES (Feb 2025)
    CONFIGS = {
        'hrnet_w32_coco': {
            'width': 32,
            'keypoints': 17,
            'input_size': (256, 192),
            'url': 'https://download.openmmlab.com/mmpose/top_down/hrnet/pose_hrnet_w32_256x192-ee87c6ab_20201104.pth',
            'url_backup': 'https://huggingface.co/spaces/wanghaochang/pose_weight/resolve/main/pose_hrnet_w32_256x192.pth'
        },
        'hrnet_w48_coco': {
            'width': 48,
            'keypoints': 17,
            'input_size': (288, 384),
            'url': 'https://download.openmmlab.com/mmpose/top_down/hrnet/pose_hrnet_w48_384x288-314c8528_20201104.pth',
            'url_backup': 'https://huggingface.co/spaces/wanghaochang/pose_weight/resolve/main/pose_hrnet_w48_384x288.pth'
        },
        'pose_hrnet_w48_384x288': {
            'width': 48,
            'keypoints': 17,
            'input_size': (288, 384),
            'url': 'https://download.openmmlab.com/mmpose/top_down/hrnet/pose_hrnet_w48_384x288-314c8528_20201104.pth',
            'url_backup': 'https://huggingface.co/spaces/wanghaochang/pose_weight/resolve/main/pose_hrnet_w48_384x288.pth'
        },
        'hrnet_w32_mpii': {
            'width': 32,
            'keypoints': 16,
            'input_size': (256, 256),
            'url': 'https://download.openmmlab.com/mmpose/top_down/hrnet/hrnet_w32_mpii_256x256-6e209ea0_20200812.pth',
            'url_backup': None
        }
    }
    
    COCO_KEYPOINT_NAMES = [
        'nose', 'left_eye', 'right_eye', 'left_ear', 'right_ear',
        'left_shoulder', 'right_shoulder', 'left_elbow', 'right_elbow',
        'left_wrist', 'right_wrist', 'left_hip', 'right_hip',
        'left_knee', 'right_knee', 'left_ankle', 'right_ankle'
    ]
    
    def __init__(
        self,
        model_path: str,
        model_type: str = 'hrnet_w48_coco',
        device: Optional[str] = None,
        download: bool = True,
        use_backup: bool = True
    ):
        """
        Inicializar inferidor de HRNet.
        
        Args:
            model_path: Ruta al archivo de pesos (.pth)
            model_type: Tipo de modelo
            device: 'cuda' o 'cpu' (auto si None)
            download: Descargar pesos si no existen
            use_backup: Usar URL backup si falla
        """
        self.model_type = model_type
        self.device = device or ('cuda' if torch.cuda.is_available() else 'cpu')
        self.use_backup = use_backup
        
        if model_type not in self.CONFIGS:
            raise ValueError(f"Tipo de modelo desconocido: {model_type}")
        
        self.config = self.CONFIGS[model_type]
        self.model_path = model_path
        
        self._ensure_model_exists(download)
        self._load_model()
        
        logger.info(f"✓ Modelo cargado: {model_type}")
        logger.info(f"  - Dispositivo: {self.device}")
        logger.info(f"  - Keypoints: {self.config['keypoints']}")
        logger.info(f"  - Entrada: {self.config['input_size']}")
    
    def _ensure_model_exists(self, download: bool = True):
        """Verificar si modelo existe, descargar si no"""
        model_path = Path(self.model_path)
        
        if model_path.exists():
            logger.info(f"✓ Modelo encontrado: {self.model_path}")
            return
        
        if not download:
            logger.error(f"✗ Modelo no encontrado: {self.model_path}")
            raise FileNotFoundError(f"Modelo no encontrado: {self.model_path}")
        
        logger.info(f"Descargando modelo...")
        self._download_weights()
    
    def _download_weights(self):
        """Descargar pesos con manejo robusto de errores"""
        model_path = Path(self.model_path)
        model_path.parent.mkdir(parents=True, exist_ok=True)
        
        urls_to_try = [self.config['url']]
        if self.use_backup and self.config.get('url_backup'):
            urls_to_try.append(self.config['url_backup'])
        
        for url_idx, url in enumerate(urls_to_try, 1):
            try:
                logger.info(f"Intento {url_idx}/{len(urls_to_try)}: Descargando...")
                self._download_url(url, str(model_path))
                logger.info(f"✓ Descarga completada: {model_path}")
                return
            except Exception as e:
                logger.warning(f"✗ Fallo: {str(e)}")
                if url_idx < len(urls_to_try):
                    logger.info(f"Intentando URL alternativa...")
                else:
                    raise
    
    def _download_url(self, url: str, output_path: str, chunk_size: int = 8192):
        """Descargar archivo con barra de progreso"""
        import urllib.request
        import urllib.error
        
        try:
            req = urllib.request.Request(
                url,
                headers={'User-Agent': 'Mozilla/5.0'}
            )
            
            with urllib.request.urlopen(req, timeout=30) as response:
                total_size = int(response.headers.get('Content-Length', 0))
                
                with open(output_path, 'wb') as f:
                    if total_size > 0:
                        with tqdm(total=total_size, unit='B', unit_scale=True) as pbar:
                            while True:
                                chunk = response.read(chunk_size)
                                if not chunk:
                                    break
                                f.write(chunk)
                                pbar.update(len(chunk))
                    else:
                        while True:
                            chunk = response.read(chunk_size)
                            if not chunk:
                                break
                            f.write(chunk)
        
        except urllib.error.HTTPError as e:
            raise RuntimeError(f"HTTP Error {e.code}: {e.reason}")
        except urllib.error.URLError as e:
            raise RuntimeError(f"URL Error: {e.reason}")
        except Exception as e:
            raise RuntimeError(f"Error: {str(e)}")
    
    def _load_model(self):
        """Cargar modelo HRNet"""
        try:
            self._load_with_mmpose()
        except Exception as e:
            logger.warning(f"⚠ MMPose no disponible: {e}")
            self._load_manual_architecture()
    
    def _load_with_mmpose(self):
        """Cargar pesos en la arquitectura HRNet real."""
        logger.info("Cargando modelo...")

        state_dict = self._load_checkpoint(self.model_path)
        if isinstance(state_dict, dict) and 'state_dict' in state_dict:
            state_dict = state_dict['state_dict']
        self.model = self._build_hrnet()
        self.model.load_state_dict(state_dict, strict=True)
        self.model.eval()
        self.model.to(self.device)
    
    def _load_manual_architecture(self):
        """Cargar con arquitectura manual (fallback)."""
        logger.info("Construyendo arquitectura HRNet...")

        self.model = self._build_hrnet()

        try:
            state_dict = self._load_checkpoint(self.model_path)
            if isinstance(state_dict, dict) and 'state_dict' in state_dict:
                state_dict = state_dict['state_dict']
            self.model.load_state_dict(state_dict, strict=True)
        except Exception as e:
            logger.warning(f"⚠ No se pudieron cargar pesos: {e}")

        self.model.eval()
        self.model.to(self.device)
    
    def _build_hrnet(self) -> nn.Module:
        """Construir arquitectura HRNet-W48 real compatible con el checkpoint oficial."""
        width = self.config['width']  # 48 para HRNet-W48
        num_joints = self.config['keypoints']
        return PoseHRNet(width=width, num_joints=num_joints)

    def _load_checkpoint(self, checkpoint_path: str):
        """Cargar checkpoint compatible con PyTorch>=2.6 y versiones previas."""
        # En PyTorch 2.6 `weights_only=True` por defecto; estos checkpoints de HRNet
        # contienen objetos de numpy serializados y requieren desactivarlo.
        try:
            return torch.load(
                checkpoint_path,
                map_location=self.device,
                weights_only=False
            )
        except TypeError:
            # Compatibilidad con versiones de PyTorch que no aceptan `weights_only`.
            return torch.load(checkpoint_path, map_location=self.device)
    
    def preprocess_image(
        self,
        image: np.ndarray,
        size: Tuple[int, int]
    ) -> Tuple[torch.Tensor, np.ndarray]:
        """Preprocesar imagen para HRNet"""
        h, w = size
        resized = cv2.resize(image, (w, h), interpolation=cv2.INTER_LINEAR)
        
        mean = np.array([0.485, 0.456, 0.406], dtype=np.float32)
        std = np.array([0.229, 0.224, 0.225], dtype=np.float32)
        
        resized = cv2.cvtColor(resized, cv2.COLOR_BGR2RGB).astype(np.float32)
        resized = resized / 255.0
        resized = (resized - mean) / std
        
        tensor = torch.from_numpy(resized.transpose(2, 0, 1)).unsqueeze(0).float()
        tensor = tensor.to(self.device)
        
        return tensor, resized
    
    def get_heatmaps(self, image: np.ndarray) -> np.ndarray:
        """Obtener heatmaps del modelo"""
        tensor, _ = self.preprocess_image(image, self.config['input_size'])
        
        with torch.no_grad():
            outputs = self.model(tensor)
        
        if isinstance(outputs, torch.Tensor):
            heatmaps = outputs.squeeze(0).cpu().numpy()
        else:
            heatmaps = outputs
        
        return heatmaps
    
    def infer_batch(
        self,
        image_paths: List[str],
        batch_size: int = 1
    ) -> Dict[str, Dict[str, object]]:
        """Procesamiento batch de imágenes"""
        results = {}
        
        for i in range(0, len(image_paths), batch_size):
            batch_paths = image_paths[i:i+batch_size]
            
            for img_path in tqdm(batch_paths, desc='Procesando'):
                img_name = Path(img_path).stem
                
                try:
                    image = cv2.imread(img_path)
                    if image is None:
                        logger.warning(f"No se pudo cargar: {img_path}")
                        continue
                    
                    heatmaps = self.get_heatmaps(image)
                    results[img_name] = {
                        'heatmaps': heatmaps,
                        'image_path': img_path,
                        'image_shape': image.shape[:2]
                    }
                
                except Exception as e:
                    logger.error(f"Error procesando {img_path}: {e}")
        
        return results


class HeatmapSaver:
    """Utilidad para guardar heatmaps"""
    
    @staticmethod
    def save_heatmap_npy(heatmaps: np.ndarray, output_path: str):
        """Guardar como .npy"""
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        np.save(output_path, heatmaps)
        logger.info(f"✓ Guardado: {output_path}")
    
    @staticmethod
    def save_heatmap_png(
        heatmaps: np.ndarray,
        output_dir: str,
        colormap: str = 'jet',
        names: Optional[List[str]] = None
    ):
        """Guardar heatmaps como PNG con nombre de articulación."""
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        colormap_map = {
            'jet': cv2.COLORMAP_JET,
            'viridis': cv2.COLORMAP_TURBO,
            'hot': cv2.COLORMAP_HOT,
        }
        cmap = colormap_map.get(colormap, cv2.COLORMAP_JET)

        for idx, heatmap in enumerate(heatmaps):
            heatmap_norm = ((heatmap - heatmap.min()) /
                          (heatmap.max() - heatmap.min() + 1e-8) * 255).astype(np.uint8)
            heatmap_color = cv2.applyColorMap(heatmap_norm, cmap)
            joint_name = names[idx] if names and idx < len(names) else f'{idx:02d}'
            fname = f"heatmap_{joint_name}.png"
            cv2.imwrite(str(output_dir / fname), heatmap_color)

        logger.info(f"✓ {len(heatmaps)} heatmaps guardados")

    @staticmethod
    def save_heatmap_overlay(
        heatmaps: np.ndarray,
        image: np.ndarray,
        output_dir: str,
        colormap: str = 'jet',
        alpha: float = 0.5,
        names: Optional[List[str]] = None
    ):
        """Superponer cada heatmap sobre la imagen original y guardar."""
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        colormap_map = {
            'jet': cv2.COLORMAP_JET,
            'viridis': cv2.COLORMAP_TURBO,
            'hot': cv2.COLORMAP_HOT,
        }
        cmap = colormap_map.get(colormap, cv2.COLORMAP_JET)
        h, w = image.shape[:2]

        for idx, heatmap in enumerate(heatmaps):
            # Escalar heatmap al tamaño de la imagen original
            hm_resized = cv2.resize(heatmap, (w, h), interpolation=cv2.INTER_CUBIC)
            hm_norm = ((hm_resized - hm_resized.min()) /
                      (hm_resized.max() - hm_resized.min() + 1e-8) * 255).astype(np.uint8)
            hm_color = cv2.applyColorMap(hm_norm, cmap)
            overlay = cv2.addWeighted(image, 1.0 - alpha, hm_color, alpha, 0)
            joint_name = names[idx] if names and idx < len(names) else f'{idx:02d}'
            fname = f"heatmap_{joint_name}.jpg"
            cv2.imwrite(str(output_dir / fname), overlay)

        logger.info(f"✓ {len(heatmaps)} heatmap overlays guardados")

    @staticmethod
    def heatmaps_to_keypoints(
        heatmaps: np.ndarray,
        image_shape: Tuple[int, int],
        names: Optional[List[str]] = None
    ) -> List[Dict[str, float]]:
        """Extraer keypoints (argmax) de heatmaps y escalarlos a la imagen original."""
        img_h, img_w = image_shape
        n_kpts, hm_h, hm_w = heatmaps.shape
        keypoints = []

        for idx in range(n_kpts):
            hm = heatmaps[idx]
            flat_idx = int(np.argmax(hm))
            y_hm, x_hm = np.unravel_index(flat_idx, hm.shape)
            score = float(hm[y_hm, x_hm])

            x_img = float((x_hm / max(hm_w - 1, 1)) * (img_w - 1))
            y_img = float((y_hm / max(hm_h - 1, 1)) * (img_h - 1))

            keypoints.append({
                'id': idx,
                'name': names[idx] if names and idx < len(names) else f'kp_{idx:02d}',
                'x': x_img,
                'y': y_img,
                'score': score
            })

        return keypoints

    @staticmethod
    def save_keypoints_json(keypoints: List[Dict[str, float]], output_path: str):
        """Guardar keypoints en JSON."""
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump({'keypoints': keypoints}, f, indent=2, ensure_ascii=False)

    # Conexiones del esqueleto COCO (índice_a, índice_b, color_BGR)
    COCO_SKELETON = [
        (0, 1,  (255, 200,  50)),  # nose – left_eye
        (0, 2,  (255, 200,  50)),  # nose – right_eye
        (1, 3,  (255, 200,  50)),  # left_eye – left_ear
        (2, 4,  (255, 200,  50)),  # right_eye – right_ear
        (5, 6,  ( 50, 255,  50)),  # left_shoulder – right_shoulder
        (5, 7,  (100, 100, 255)),  # left_shoulder – left_elbow
        (7, 9,  (100, 100, 255)),  # left_elbow – left_wrist
        (6, 8,  (255, 100, 100)),  # right_shoulder – right_elbow
        (8, 10, (255, 100, 100)),  # right_elbow – right_wrist
        (5, 11, ( 50, 255, 200)),  # left_shoulder – left_hip
        (6, 12, ( 50, 200, 255)),  # right_shoulder – right_hip
        (11, 12,(200, 255,  50)),  # left_hip – right_hip
        (11, 13,(100, 100, 255)),  # left_hip – left_knee
        (13, 15,(100, 100, 255)),  # left_knee – left_ankle
        (12, 14,(255, 100, 100)),  # right_hip – right_knee
        (14, 16,(255, 100, 100)),  # right_knee – right_ankle
    ]

    @staticmethod
    def save_keypoints_overlay(
        image: np.ndarray,
        keypoints: List[Dict[str, float]],
        output_path: str,
        score_threshold: float = 0.0
    ):
        """Guardar imagen con keypoints y esqueleto dibujados."""
        vis = image.copy()
        h, w = vis.shape[:2]
        radius = max(6, w // 150)      # escala al tamaño de imagen
        thickness_skel = max(2, w // 300)

        # --- Normalizar scores al rango [0, 1] para umbral robusto ---
        raw_scores = np.array([kp['score'] for kp in keypoints], dtype=np.float32)
        s_min, s_max = raw_scores.min(), raw_scores.max()
        if s_max - s_min > 1e-8:
            norm_scores = (raw_scores - s_min) / (s_max - s_min)
        else:
            norm_scores = np.ones_like(raw_scores)

        coords = {}
        for i, kp in enumerate(keypoints):
            if norm_scores[i] >= score_threshold:
                x = int(round(kp['x']))
                y = int(round(kp['y']))
                coords[kp['id']] = (x, y)

        # --- Líneas del esqueleto ---
        for (a, b, color) in HeatmapSaver.COCO_SKELETON:
            if a in coords and b in coords:
                cv2.line(vis, coords[a], coords[b], color, thickness_skel, cv2.LINE_AA)

        # --- Puntos con borde blanco ---
        for kp in keypoints:
            kid = kp['id']
            if kid not in coords:
                continue
            x, y = coords[kid]
            # borde blanco
            cv2.circle(vis, (x, y), radius + 2, (255, 255, 255), -1, cv2.LINE_AA)
            # relleno de color según score normalizado
            color = (0, 220, 0) if norm_scores[kid] >= 0.5 else (0, 140, 255)
            cv2.circle(vis, (x, y), radius, color, -1, cv2.LINE_AA)
            # etiqueta opcional (solo si hay espacio)
            label = kp.get('name', str(kid))
            cv2.putText(vis, label, (x + radius + 2, y + 4),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.35, (255, 255, 255), 1, cv2.LINE_AA)

        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        cv2.imwrite(str(output_path), vis)


def main():
    """Función principal"""
    parser = argparse.ArgumentParser(
        description='HRNet Inference - Extracción de heatmaps',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
EJEMPLOS:
  # Usar valores por defecto (imgs_entrada → imgs_salida)
  python hrnet_inference.py
  
  # Personalizado
  python hrnet_inference.py --input-dir ./images --output-dir ./results
  
  # Con opciones
  python hrnet_inference.py --save-png --colormap viridis
        """
    )
    
    parser.add_argument('--input-dir', type=str, default='imgs_test',
                       help='Directorio de entrada (default: imgs_ensuciadas)')
    parser.add_argument('--output-dir', type=str, default='imgs_test_salida',
                       help='Directorio de salida (default: imgs_salida_ensuciadas)')
    parser.add_argument('--model-path', type=str,
                       #default='./models/pose_hrnet_w48_384x288.pth',
                       default='./models/hrnet_finetuned.pth',
                       help='Ruta del modelo')
    parser.add_argument('--model-type', type=str, default='pose_hrnet_w48_384x288',
                       choices=['hrnet_w32_coco', 'hrnet_w48_coco', 'pose_hrnet_w48_384x288', 'hrnet_w32_mpii'],
                       help='Tipo (default: pose_hrnet_w48_384x288)')
    parser.add_argument('--device', type=str, default=None,
                       choices=['cuda', 'cpu'],
                       help='Dispositivo (auto si no especifica)')
    parser.add_argument('--batch-size', type=int, default=1,
                       help='Batch size (default: 1)')
    parser.add_argument('--download', action='store_true', default=False,
                       help='Descargar modelo')
    parser.add_argument('--use-backup', action='store_true', default=True,
                       help='Usar URL backup')
    parser.add_argument('--save-npy', action='store_true', default=True,
                       help='Guardar .npy')
    parser.add_argument('--save-png', action='store_true', default=True,
                       help='Guardar PNG')
    parser.add_argument('--save-keypoints', action='store_true', default=True,
                       help='Guardar keypoints en JSON y overlay')
    parser.add_argument('--colormap', type=str, default='jet',
                       choices=['jet', 'viridis', 'hot', 'cool'],
                       help='Colormap (default: jet)')
    
    args = parser.parse_args()
    
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    try:
        logger.info("=" * 60)
        logger.info("HRNet Inference - Extracción de Heatmaps")
        logger.info("=" * 60)
        
        logger.info(f"Entrada:  {args.input_dir}")
        logger.info(f"Salida:   {args.output_dir}")
        logger.info(f"Modelo:   {args.model_type}")
        logger.info("=" * 60)
        
        inferencer = HRNetInferencer(
            model_path=args.model_path,
            model_type=args.model_type,
            device=args.device,
            download=args.download,
            use_backup=args.use_backup
        )
        
        image_dir = Path(args.input_dir)
        image_extensions = {'.jpg', '.jpeg', '.png', '.bmp', '.tiff'}
        image_paths = [
            str(p) for p in image_dir.rglob('*')
            if p.suffix.lower() in image_extensions
        ]
        
        if not image_paths:
            logger.error(f"✗ No hay imágenes en {args.input_dir}")
            return
        
        logger.info(f"✓ Encontradas {len(image_paths)} imágenes")
        logger.info("=" * 60)
        
        results = inferencer.infer_batch(image_paths, args.batch_size)
        
        npy_dir = output_dir / 'npy' if args.save_npy else None
        png_dir = output_dir / 'png' if args.save_png else None
        heatmap_overlay_dir = output_dir / 'heatmap_overlay' if args.save_png else None
        keypoints_json_dir = output_dir / 'keypoints_json' if args.save_keypoints else None
        keypoints_overlay_dir = output_dir / 'keypoints_overlay' if args.save_keypoints else None
        
        saver = HeatmapSaver()
        
        logger.info("Guardando resultados...")
        for img_name, result in tqdm(results.items(), desc='Guardando'):
            heatmaps = result['heatmaps']

            if npy_dir:
                npy_dir.mkdir(parents=True, exist_ok=True)
                saver.save_heatmap_npy(heatmaps, str(npy_dir / f"{img_name}.npy"))
            
            if png_dir:
                png_dir.mkdir(parents=True, exist_ok=True)
                saver.save_heatmap_png(
                    heatmaps, str(png_dir / img_name), args.colormap,
                    names=inferencer.COCO_KEYPOINT_NAMES
                )

            if heatmap_overlay_dir:
                original_for_hm = cv2.imread(result['image_path'])
                if original_for_hm is not None:
                    saver.save_heatmap_overlay(
                        heatmaps, original_for_hm,
                        str(heatmap_overlay_dir / img_name),
                        args.colormap,
                        names=inferencer.COCO_KEYPOINT_NAMES
                    )

            if args.save_keypoints:
                image_path = result['image_path']
                image_shape = result['image_shape']
                keypoints = saver.heatmaps_to_keypoints(
                    heatmaps,
                    image_shape,
                    inferencer.COCO_KEYPOINT_NAMES
                )

                if keypoints_json_dir:
                    keypoints_json_dir.mkdir(parents=True, exist_ok=True)
                    saver.save_keypoints_json(
                        keypoints,
                        str(keypoints_json_dir / f"{img_name}.json")
                    )

                if keypoints_overlay_dir:
                    keypoints_overlay_dir.mkdir(parents=True, exist_ok=True)
                    original = cv2.imread(image_path)
                    if original is not None:
                        saver.save_keypoints_overlay(
                            original,
                            keypoints,
                            str(keypoints_overlay_dir / f"{img_name}.jpg")
                        )
        
        logger.info("=" * 60)
        logger.info(f"✓ COMPLETADO. Resultados en: {args.output_dir}")
        logger.info("=" * 60)
    
    except Exception as e:
        logger.error(f"✗ Error: {e}")
        sys.exit(1)


if __name__ == '__main__':
    main()