"""
Script de Fine-tuning en single GPU.

VERSIÓN MODIFICADA CON HEATMAP LOSS DE HRNET Y VALIDACIÓN

Fixes aplicados respecto a la versión base:
  T1+T2  — Loss cambiada a MSE directo sobre velocity (v_pred vs velocity_target).
            Elimina el VAE del training loop → gradientes limpios y estables.
            velocity_target = noise - target se usa ahora correctamente.
  T3     — Desnormalización corregida en inferencia/visualización:
            x_vae = x_norm * vae_std + vae_mean  (era al revés).
  T4     — pos_embed llamado con txt_seq_lens como lista de enteros por sample,
            no con kwarg max_txt_seq_len que no existe.
  T5     — Orden correcto: prepare_model_for_kbit_training ANTES de get_peft_model.
  T6     — target_modules ampliado con add_q_proj, add_k_proj, add_v_proj, to_add_out
            para cubrir el stream de texto en la atención conjunta.
  T7     — Optimizer creado sobre parámetros requires_grad=True, después de que
            QLoRA esté completamente configurado.
  T8     — timestep dividido por 1000 antes del forward del transformer,
            igual que hace la pipeline original.

MODIFICACIONES HEATMAP LOSS:
  H1     — Añadida arquitectura HRNet completa (PoseHRNet, Bottleneck, BasicBlock, etc.)
  H2     — Nueva función preprocess_image_for_hrnet() con normalización ImageNet
  H3     — Nueva función latents_to_images() para decodificar latentes a imágenes
  H4     — Clase CombinedLossFn que reemplaza VelocityLossFn:
            * Calcula x0_pred = noisy - t*v_pred (matemáticamente correcto)
            * Decodifica x0_pred con VAE
            * Calcula heatmaps con HRNet
            * Compara con GT heatmaps
            * Soporta dos tipos de loss: MSE simple y weighted MSE
  H5     — Modificado collate_latents() para extraer target_heatmaps de los .pt
  H6     — HRNet cargado en Rank 1 (mismo que VAE) y congelado
  H7     — Modificado training loop para pasar timesteps a la loss
  H8     — Añadidos argumentos CLI: --hrnet_model_path, --heatmap_loss_weight, etc.

NUEVAS MODIFICACIONES VALIDACIÓN:
  V1     — Split train/val con semilla fija (reproducible)
  V2     — Función validate() para calcular val_loss al final de cada época
  V3     — Val_loss guardado en CSV
  V4     — Argumento --val_split para controlar ratio de validación

Usage:
    python train_from_latents_1gpu.py \\
        --latents_dir "/ruta/precomputed_latents" \\
        --hrnet_model_path "./models/pose_hrnet_w48_384x288.pth" \\
        --output_dir "output_hrnet" \\
        --batch_size 4 \\
        --epochs 200 \\
        --learning_rate 1e-4 \\
        --heatmap_loss_weight 0.5 \\
        --velocity_loss_weight 0.5 \\
        --heatmap_loss_type "mse" \\
        --val_split 0.1



----------------------------------------------------------------

 python train_from_latents_1gpu.py --latents_dir "/data/antoniodetoro/qwen/dataset_local_latents_512_heatmaps/" --hrnet_model_path /nas/antoniodetoro/qwen/Qwen-Image-Edit-Angles-2/src/hr_net/models/hrnet_finetuned_best.pth --output_dir output_qwen_HRNet2 --batch_size 4 --epochs 20 --heatmap_loss_weight 0.5 --velocity_loss_weight 0.5 --heatmap_loss_type "mse"

 python train_from_latents_singleGPU_QLoRA_HRNet_Claude_Val.py --latents_dir "/data/antoniodetoro/qwen/dataset_local_latents_512_heatmaps/" --hrnet_model_path /nas/antoniodetoro/qwen/Qwen-Image-Edit-Angles-2/src/hr_net/models/hrnet_finetuned_best.pth --output_dir output_qwen_HRNet_single_GPU_prueba --batch_size 4 --epochs 20 --heatmap_loss_weight 0.5 --velocity_loss_weight 0.5 --heatmap_loss_type "mse"
"""

import os
import sys
import csv

current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
sys.path.append(parent_dir)

import argparse
from pathlib import Path
from dataclasses import dataclass
from typing import List, Tuple, Optional
from torch.utils.checkpoint import checkpoint

os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["HF_HOME"] = "/nas/antoniodetoro/qwen/hf_cache"
os.environ["TMPDIR"] = "/dev/shm"
os.environ["PYTHONNOUSERSITE"] = "1"

import torch
import logging
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as T
from PIL import Image
from torch.utils.data import Dataset, DataLoader, Subset  # NUEVO V1: añadido Subset
from tqdm import tqdm
from diffusers.optimization import get_scheduler
from diffusers import FlowMatchEulerDiscreteScheduler, QwenImageTransformer2DModel
from diffusers.models import AutoencoderKLQwenImage
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
from transformers import BitsAndBytesConfig
import numpy as np  # NUEVO V1: para split reproducible


logging.basicConfig(
    level=logging.INFO,
    force=True,
    format="%(message)s",
)
logger = logging.getLogger(__name__)


# =========================================================================
# ARQUITECTURA HRNET (igual que antes)
# =========================================================================

BN_MOMENTUM = 0.1


class Bottleneck(nn.Module):
    """Bloque Bottleneck para ResNet (usado en layer1 de HRNet)."""
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
    """Bloque básico para ResNet (usado en stages 2-4 de HRNet)."""
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
    """Módulo multi-rama con fusión entre resoluciones."""

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
                    fuse_layer.append(nn.Sequential(
                        nn.Conv2d(num_channels[j], num_channels[i], 1, bias=False),
                        nn.BatchNorm2d(num_channels[i], momentum=BN_MOMENTUM),
                    ))
                elif j == i:
                    fuse_layer.append(None)
                else:
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
    Configuración: W48 (COCO 17 keypoints).
    """

    def __init__(self, width=48, num_joints=17):
        super().__init__()
        C = width

        # Stem
        self.conv1 = nn.Conv2d(3, 64, 3, stride=2, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(64, momentum=BN_MOMENTUM)
        self.conv2 = nn.Conv2d(64, 64, 3, stride=2, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(64, momentum=BN_MOMENTUM)
        self.relu = nn.ReLU(inplace=True)

        # Layer1
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

        # Transition1
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

        # Stage2
        self.stage2 = nn.Sequential(
            HighResolutionModule(2, [C, C * 2], num_blocks=4),
        )

        # Transition2
        self.transition2 = nn.ModuleList([
            None,
            None,
            nn.Sequential(nn.Sequential(
                nn.Conv2d(C * 2, C * 4, 3, stride=2, padding=1, bias=False),
                nn.BatchNorm2d(C * 4, momentum=BN_MOMENTUM),
                nn.ReLU(inplace=True),
            )),
        ])

        # Stage3
        stage3_modules = []
        for i in range(4):
            stage3_modules.append(
                HighResolutionModule(3, [C, C * 2, C * 4], num_blocks=4,
                                     multi_scale_output=True)
            )
        self.stage3 = nn.Sequential(*stage3_modules)

        # Transition3
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

        # Stage4
        stage4_modules = []
        for i in range(3):
            multi_scale_output = True if i < 2 else False
            stage4_modules.append(
                HighResolutionModule(4, [C, C * 2, C * 4, C * 8], num_blocks=4,
                                     multi_scale_output=multi_scale_output)
            )
        self.stage4 = nn.Sequential(*stage4_modules)

        # Final layer
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

        # Output
        x = self.final_layer(y_list[0])
        return x


def load_hrnet_model(
    model_path: str,
    width: int = 48,
    num_joints: int = 17,
    device: torch.device = torch.device('cpu')
) -> PoseHRNet:
    """Carga HRNet pre-entrenado."""
    model = PoseHRNet(width=width, num_joints=num_joints)
    
    try:
        checkpoint = torch.load(model_path, map_location=device, weights_only=False)
    except TypeError:
        checkpoint = torch.load(model_path, map_location=device)
    
    if isinstance(checkpoint, dict) and 'state_dict' in checkpoint:
        checkpoint = checkpoint['state_dict']
    
    model.load_state_dict(checkpoint, strict=True)
    model.eval()
    model.to(device)
    
    logger.info(f"✓ HRNet cargado desde {model_path}")
    return model


# =========================================================================
# FIN ARQUITECTURA HRNET
# =========================================================================


@dataclass
class TrainingConfig:
    latents_dir: str = "../models/precomputed_latents_fix"
    output_dir: str = "../models/finetuned_pp"
    base_model: str = "Qwen/Qwen-Image-Edit-2509"
    transformer_model: str = "../models/Qwen-Fused-Angles"
    
    # Configuración HRNet
    hrnet_model_path: str = "./models/pose_hrnet_w48_384x288.pth"
    hrnet_input_size: Tuple[int, int] = (288, 384)
    
    epochs: int = 3
    batch_size: int = 4
    microbatches: int = 4
    lr: float = 1e-4
    lora_rank: int = 16
    lora_alpha: int = 32
    lora_dropout: float = 0.1
    gradient_accumulation_steps: int = 1
    inference_every: int = 10
    inference_steps: int = 4
    inference_samples: int = 2
    
    # Pesos de la loss combinada
    velocity_loss_weight: float = 0.5
    heatmap_loss_weight: float = 0.5
    heatmap_loss_type: str = "mse"
    
    # NUEVO V1: Configuración de validación
    val_split: float = 0.1  # Porcentaje de datos para validación
    val_seed: int = 42      # Semilla para split reproducible


# ---------------------------------------------------------------------------
# Dataset
# ---------------------------------------------------------------------------

class LatentsDataset(Dataset):
    def __init__(self, latents_dir, split="train"):
        self.split_dir = Path(latents_dir) / split
        self.files = sorted(list(self.split_dir.glob("*.pt")))
        if len(self.files) == 0:
            logger.warning(f"No files found in {self.split_dir}")

        logger.info(f"Scanning {len(self.files)} files to compute global_max_seq_len...")
        self.global_max_seq_len = 0
        for f in self.files:
            data = torch.load(f, weights_only=True)
            seq_len = data["prompt_embeds"].shape[1]
            if seq_len > self.global_max_seq_len:
                self.global_max_seq_len = seq_len
        logger.info(f"global_max_seq_len = {self.global_max_seq_len}")

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        return torch.load(self.files[idx], weights_only=True)


# NUEVO V1: Función para crear splits train/val reproducibles
def create_train_val_split(dataset: Dataset, val_split: float, seed: int = 42):
    """
    NUEVO V1: Crea splits train/val reproducibles con semilla fija.
    
    Args:
        dataset: Dataset completo
        val_split: Fracción de datos para validación (0.0-1.0)
        seed: Semilla para reproducibilidad
    
    Returns:
        train_subset, val_subset
    """
    dataset_size = len(dataset)
    indices = list(range(dataset_size))
    
    # Mezclar con semilla fija para reproducibilidad
    np.random.seed(seed)
    np.random.shuffle(indices)
    
    # Calcular punto de split
    split_idx = int(np.floor(val_split * dataset_size))
    
    train_indices = indices[split_idx:]
    val_indices = indices[:split_idx]
    
    train_subset = Subset(dataset, train_indices)
    val_subset = Subset(dataset, val_indices)
    
    logger.info(f"✓ Dataset split (seed={seed}): {len(train_subset)} train, {len(val_subset)} val")
    
    return train_subset, val_subset


def make_collate_latents(global_max_seq_len: int):
    """Collate que padea prompt_embeds y extrae target_heatmaps."""
    def collate_latents(batch):
        target_latents  = torch.cat([item["target_latents_packed"]  for item in batch], dim=0)
        source_latents  = torch.cat([item["source_latents_packed"]  for item in batch], dim=0)
        prompt_list     = [item["prompt_embeds"]      for item in batch]
        mask_list       = [item["prompt_embeds_mask"] for item in batch]
        
        # Extraer heatmaps GT
        target_heatmaps_list = []
        for item in batch:
            if "target_heatmaps" in item:
                target_heatmaps_list.append(item["target_heatmaps"])
            else:
                logger.warning("target_heatmaps no encontrado en batch item, usando ceros")
                target_heatmaps_list.append(torch.zeros(17, 72, 96, dtype=torch.float32))
        
        target_heatmaps = torch.cat(target_heatmaps_list, dim=0)

        padded_embeds, padded_masks = [], []
        for pe, pm in zip(prompt_list, mask_list):
            curr_len = pe.shape[1]
            if curr_len < global_max_seq_len:
                pad_len = global_max_seq_len - curr_len
                pe = F.pad(pe, (0, 0, 0, pad_len), value=0.0)
                pm = F.pad(pm, (0, pad_len),        value=0)
            padded_embeds.append(pe)
            padded_masks.append(pm)

        return {
            "target_latents_packed": target_latents,
            "source_latents_packed": source_latents,
            "prompt_embeds":         torch.cat(padded_embeds, dim=0),
            "prompt_embeds_mask":    torch.cat(padded_masks,  dim=0),
            "target_heatmaps":       target_heatmaps,
        }
    return collate_latents


# ---------------------------------------------------------------------------
# Helpers: pack / unpack latents
# ---------------------------------------------------------------------------

def unpack_latents(latents: torch.Tensor, height: int, width: int, vae_scale_factor: int = 8) -> torch.Tensor:
    """Invierte _pack_latents: (B, N_patches, z_dim*4) → (B, z_dim, 1, H, W)."""
    batch_size, num_patches, channels = latents.shape
    h = 2 * (int(height) // (vae_scale_factor * 2))
    w = 2 * (int(width)  // (vae_scale_factor * 2))
    latents = latents.view(batch_size, h // 2, w // 2, channels // 4, 2, 2)
    latents = latents.permute(0, 3, 1, 4, 2, 5)
    latents = latents.reshape(batch_size, channels // 4, 1, h, w)
    return latents


def latents_to_pil(
    latents_packed: torch.Tensor,
    vae: AutoencoderKLQwenImage,
    img_height: int,
    img_width: int,
) -> List[Image.Image]:
    """Desnormaliza + decodifica latentes empaquetados a lista de PIL."""
    spatial = unpack_latents(latents_packed.float(), img_height, img_width).to(vae.dtype)

    vae_mean = torch.tensor(vae.config.latents_mean).view(
        1, vae.config.z_dim, 1, 1, 1).to(spatial.device, spatial.dtype)
    vae_std  = torch.tensor(vae.config.latents_std).view(
        1, vae.config.z_dim, 1, 1, 1).to(spatial.device, spatial.dtype)
    z_raw = spatial * vae_std + vae_mean

    with torch.no_grad():
        decoded = vae.decode(z_raw, return_dict=False)[0]
    decoded = decoded[:, :, 0]

    pils = []
    for img_t in decoded:
        img_t = torch.clamp((img_t.cpu().float() * 0.5 + 0.5), 0.0, 1.0)
        pils.append(T.ToPILImage()(img_t))
    return pils


def latents_to_images(
    latents_packed: torch.Tensor,
    vae: AutoencoderKLQwenImage,
    img_height: int,
    img_width: int,
) -> torch.Tensor:
    """Desnormaliza + decodifica latentes empaquetados a tensor de imágenes."""
    spatial = unpack_latents(latents_packed.float(), img_height, img_width).to(vae.dtype)

    vae_mean = torch.tensor(vae.config.latents_mean).view(
        1, vae.config.z_dim, 1, 1, 1).to(spatial.device, spatial.dtype)
    vae_std  = torch.tensor(vae.config.latents_std).view(
        1, vae.config.z_dim, 1, 1, 1).to(spatial.device, spatial.dtype)
    z_raw = spatial * vae_std + vae_mean

    with torch.no_grad():
        decoded = vae.decode(z_raw, return_dict=False)[0]
    decoded = decoded[:, :, 0]
    
    return decoded


def preprocess_image_for_hrnet(
    image: torch.Tensor,
    target_size: Tuple[int, int] = (288, 384)
) -> torch.Tensor:
    """Preprocesa imagen para HRNet con normalización ImageNet."""
    # Convertir de [-1, 1] a [0, 1]
    image = (image + 1.0) / 2.0
    
    # Resize
    image = F.interpolate(image, size=target_size, mode='bilinear', align_corners=True)
    
    # Normalización ImageNet
    mean = torch.tensor([0.485, 0.456, 0.406], device=image.device).view(1, 3, 1, 1)
    std = torch.tensor([0.229, 0.224, 0.225], device=image.device).view(1, 3, 1, 1)
    
    image = (image - mean) / std
    return image


def compute_pck(
    pred_heatmaps: torch.Tensor,
    target_heatmaps: torch.Tensor,
    threshold: float = 0.2,
) -> float:
    """
    Calcula PCK (Percentage of Correct Keypoints).

    Un keypoint se considera correcto si la distancia euclídea entre la
    posición predicha y la GT (en coordenadas de heatmap) es menor que
    threshold * sqrt(H * W).

    Args:
        pred_heatmaps:   (B, J, H, W) heatmaps predichos.
        target_heatmaps: (B, J, H, W) heatmaps ground-truth.
        threshold:       fracción del tamaño del heatmap usada como radio.

    Returns:
        PCK en [0, 1].
    """
    B, J, H, W = pred_heatmaps.shape

    pred_flat   = pred_heatmaps.detach().view(B, J, -1).argmax(dim=-1)
    target_flat = target_heatmaps.detach().view(B, J, -1).argmax(dim=-1)

    pred_y   = (pred_flat   // W).float()
    pred_x   = (pred_flat   %  W).float()
    target_y = (target_flat // W).float()
    target_x = (target_flat %  W).float()

    dist    = torch.sqrt((pred_x - target_x) ** 2 + (pred_y - target_y) ** 2)
    thresh  = threshold * (H * W) ** 0.5
    correct = (dist < thresh).float()

    return correct.mean().item()


# ---------------------------------------------------------------------------
# Loss: CombinedLossFn
# ---------------------------------------------------------------------------

class CombinedLossFn:
    """Loss combinada: VelocityLoss + HeatmapLoss."""

    def __init__(
        self,
        vae: Optional[AutoencoderKLQwenImage] = None,
        hrnet: Optional[PoseHRNet] = None,
        hrnet_input_size: Tuple[int, int] = (288, 384),
        img_height: int = 512,
        img_width: int = 512,
        velocity_weight: float = 0.5,
        heatmap_weight: float = 0.5,
        heatmap_loss_type: str = "mse",
        save_dir: str = None
    ):
        self.vae = vae
        self.hrnet = hrnet
        self.hrnet_input_size = hrnet_input_size
        self.img_height = img_height
        self.img_width = img_width
        self.velocity_weight = velocity_weight
        self.heatmap_weight = heatmap_weight
        self.heatmap_loss_type = heatmap_loss_type
        
        self.save_dir = save_dir
        self.step_counter = 0
        if save_dir is not None:
            os.makedirs(save_dir, exist_ok=True)
        
        self.current_target_heatmaps = None
        self.current_timesteps = None
        self.last_pck = 0.0  # PCK del último forward
        
        # Pesos para weighted MSE
        self.keypoint_weights = torch.tensor([
            1.0,  # nose
            0.8, 0.8,  # eyes
            0.6, 0.6,  # ears
            1.5, 1.5,  # shoulders
            1.2, 1.2,  # elbows
            1.0, 1.0,  # wrists
            1.5, 1.5,  # hips
            1.2, 1.2,  # knees
            1.0, 1.0,  # ankles
        ], dtype=torch.float32)

    def set_batch_context(self, target_heatmaps: torch.Tensor, timesteps: torch.Tensor):
        """Actualiza contexto del batch actual."""
        self.current_target_heatmaps = target_heatmaps
        self.current_timesteps = timesteps

    def __call__(self, outputs: torch.Tensor, combined_target: torch.Tensor) -> torch.Tensor:
        """Calcula loss combinada."""
        device = outputs.device
        
        # 1. Velocity Loss
        v_target = combined_target[:, 0, :, :].float()
        v_pred = outputs.float()
        
        if v_pred.shape[1] > v_target.shape[1]:
            v_pred = v_pred[:, :v_target.shape[1], :]

        if torch.isnan(v_pred).any() or torch.isinf(v_pred).any():
            logger.warning(
                f"[LOSS step {self.step_counter}] NaN/Inf en v_pred. "
                f"max={v_pred.abs().nanmax().item():.2f}"
            )
            self.step_counter += 1
            return torch.tensor(0.0, device=device, requires_grad=True)

        velocity_loss_val = F.mse_loss(v_pred, v_target, reduction="mean")
        
        # 2. Heatmap Loss
        heatmap_loss_val = torch.tensor(0.0, device=device)
        
        if (self.vae is not None and 
            self.hrnet is not None and 
            self.current_target_heatmaps is not None and
            self.current_timesteps is not None):
            
            try:
                noisy_latents = combined_target[:, 1, :, :]
                
                t_normalized = self.current_timesteps.float() / 1000.0
                t_normalized = t_normalized.view(-1, 1, 1).to(v_pred.dtype)
                
                x0_pred = noisy_latents - t_normalized * v_pred
                
                decoded_images = latents_to_images(
                    x0_pred,
                    self.vae,
                    self.img_height,
                    self.img_width
                )
                
                hrnet_input = preprocess_image_for_hrnet(
                    decoded_images,
                    self.hrnet_input_size
                )
                
                with torch.no_grad():
                    pred_heatmaps = self.hrnet(hrnet_input)
                
                target_hm = self.current_target_heatmaps.to(device).float()
                
                if target_hm.shape[-2:] != pred_heatmaps.shape[-2:]:
                    target_hm = F.interpolate(
                        target_hm,
                        size=pred_heatmaps.shape[-2:],
                        mode='bilinear',
                        align_corners=True
                    )
                
                if self.heatmap_loss_type == "weighted_mse":
                    weights = self.keypoint_weights.to(device).view(1, 17, 1, 1)
                    diff = (pred_heatmaps - target_hm) ** 2
                    heatmap_loss_val = (weights * diff).mean()
                else:
                    heatmap_loss_val = F.mse_loss(
                        pred_heatmaps,
                        target_hm,
                        reduction="mean"
                    )

                # Accuracy: PCK sobre los heatmaps
                self.last_pck = compute_pck(pred_heatmaps, target_hm)

            except Exception as e:
                logger.warning(f"[LOSS step {self.step_counter}] Error en heatmap loss: {e}")
                import traceback
                traceback.print_exc()
                heatmap_loss_val = torch.tensor(0.0, device=device)
        
        # 3. Loss combinada
        total_loss = (
            self.velocity_weight * velocity_loss_val +
            self.heatmap_weight * heatmap_loss_val
        )
        
        if self.step_counter % 50 == 0:
            logger.info(
                f"[LOSS step {self.step_counter}] "
                f"Velocity: {velocity_loss_val.item():.6f}, "
                f"Heatmap: {heatmap_loss_val.item():.6f}, "
                f"PCK: {self.last_pck:.4f}, "
                f"Total: {total_loss.item():.6f} "
                f"(α={self.velocity_weight:.2f}, β={self.heatmap_weight:.2f})"
            )
        
        self.step_counter += 1
        return total_loss


# ---------------------------------------------------------------------------
# NUEVO V2: Función de validación
# ---------------------------------------------------------------------------

@torch.no_grad()
def validate(
    model,
    val_dataloader: DataLoader,
    loss_fn: CombinedLossFn,
    diff_scheduler,
    device: torch.device,
    dtype: torch.dtype = torch.bfloat16,
) -> Tuple[float, float]:
    """
    NUEVO V2: Calcula validation loss y PCK.

    Returns:
        (val_loss, val_pck)
    """
    model.eval()
    
    total_loss = 0.0
    total_pck  = 0.0
    num_batches = 0
    
    # Usar la misma semilla para reproducibilidad en validación
    val_gen = torch.Generator(device=device)
    val_gen.manual_seed(9999)  # Semilla diferente a training
    
    for batch in val_dataloader:
        target = batch["target_latents_packed"].to(device, dtype=dtype)
        source = batch["source_latents_packed"].to(device, dtype=dtype)
        prompt = batch["prompt_embeds"].to(device, dtype=dtype)
        mask = batch["prompt_embeds_mask"].to(device)
        target_heatmaps = batch["target_heatmaps"]
        
        bsz = target.shape[0]
        
        # Generar ruido y timesteps (reproducible con val_gen)
        timesteps = torch.randint(
            0, diff_scheduler.config.num_train_timesteps,
            (bsz,), generator=val_gen, device=device
        ).long()
        
        noise = torch.randn(target.shape, generator=val_gen, device=device, dtype=dtype)
        
        t_norm = (timesteps.float() / diff_scheduler.config.num_train_timesteps).to(dtype)
        t_norm = t_norm.view(-1, 1, 1)
        noisy = (1.0 - t_norm) * target + t_norm * noise
        
        velocity_target = noise - target
        timestep_norm = (timesteps.float() / 1000.0).to(dtype)

        latent_model_input = torch.cat([noisy, source], dim=1)
        v_pred = model(latent_model_input, prompt, mask, timestep_norm)

        if isinstance(loss_fn, CombinedLossFn):
            loss_fn.set_batch_context(target_heatmaps, timesteps)
        
        combined_target = torch.stack([velocity_target, noisy], dim=1)
        batch_loss = loss_fn(v_pred, combined_target)
        
        total_loss += batch_loss.item()
        if isinstance(loss_fn, CombinedLossFn):
            total_pck += loss_fn.last_pck
        num_batches += 1
    
    avg_loss = total_loss / num_batches if num_batches > 0 else 0.0
    avg_pck  = total_pck  / num_batches if num_batches > 0 else 0.0

    model.train()
    
    return avg_loss, avg_pck


# ---------------------------------------------------------------------------
# Model wrapper (1 GPU: forward completo sin split)
# ---------------------------------------------------------------------------

class QwenSingleGPUWrapper(nn.Module):
    """Wrapper de QwenImageTransformer2DModel para entrenamiento en 1 GPU."""

    def __init__(self, model: nn.Module):
        super().__init__()
        self.model = model

        if hasattr(model, "base_model") and hasattr(model.base_model, "model"):
            self.inner_model = model.base_model.model
        else:
            self.inner_model = model

        total_layers = len(self.inner_model.transformer_blocks)
        logger.info(f"QwenSingleGPUWrapper: {total_layers} bloques transformer (sin split).")

    def _block_checkpoint(self, block, h, e, mask, temb, r0, r1):
        def _fwd(h, e, mask, temb, r0, r1):
            if r0.numel() > r1.numel():
                img_rot, txt_rot = r0, r1
            else:
                img_rot, txt_rot = r1, r0
            return block(
                hidden_states=h,
                encoder_hidden_states=e,
                encoder_hidden_states_mask=mask,
                temb=temb,
                image_rotary_emb=(img_rot, txt_rot)
            )
        return checkpoint(_fwd, h, e, mask, temb, r0, r1, use_reentrant=False)

    def forward(self, hidden_states, encoder_hidden_states, encoder_hidden_states_mask, timestep):
        hidden_states = self.inner_model.img_in(hidden_states)
        timestep = timestep.to(hidden_states.dtype)
        encoder_hidden_states = self.inner_model.txt_norm(encoder_hidden_states)
        encoder_hidden_states = self.inner_model.txt_in(encoder_hidden_states)

        temb = self.inner_model.time_text_embed(timestep, hidden_states)
        temb = temb.to(dtype=hidden_states.dtype)

        B = hidden_states.shape[0]
        seq_len = hidden_states.shape[1]

        grid_sq = int(seq_len ** 0.5)
        if grid_sq * grid_sq == seq_len:
            img_shapes = [[(1, grid_sq, grid_sq)]] * B
        else:
            half = seq_len // 2
            grid_sq = int(half ** 0.5)
            img_shapes = [[(2, grid_sq, grid_sq)]] * B

        full_len = encoder_hidden_states.shape[1]

        image_rotary_emb = self.inner_model.pos_embed(
            img_shapes,
            max_txt_seq_len=full_len,
            device=hidden_states.device
        )
        r0, r1 = image_rotary_emb

        for block in self.inner_model.transformer_blocks:
            if self.training:
                if not hidden_states.requires_grad:
                    hidden_states.requires_grad_(True)
                if not encoder_hidden_states.requires_grad:
                    encoder_hidden_states.requires_grad_(True)
                encoder_hidden_states, hidden_states = self._block_checkpoint(
                    block, hidden_states, encoder_hidden_states,
                    encoder_hidden_states_mask, temb, r0, r1,
                )
            else:
                encoder_hidden_states, hidden_states = block(
                    hidden_states=hidden_states,
                    encoder_hidden_states=encoder_hidden_states,
                    encoder_hidden_states_mask=encoder_hidden_states_mask,
                    temb=temb,
                    image_rotary_emb=(r0, r1),
                )

        hidden_states = self.inner_model.norm_out(hidden_states, temb)
        hidden_states = self.inner_model.proj_out(hidden_states)

        return hidden_states


# ---------------------------------------------------------------------------
# Inferencia callback
# ---------------------------------------------------------------------------

@torch.no_grad()
def run_inference_callback(
    model, vae, base_scheduler, device,
    dataset, save_dir, epoch,
    img_height=1024, img_width=1024,
    num_steps=4, num_samples=2, dtype=torch.bfloat16,
):
    """Ejecuta inferencia y guarda imágenes."""
    model.eval()

    infer_dir = os.path.join(save_dir, f"inference_epoch_{epoch:03d}")
    os.makedirs(infer_dir, exist_ok=True)

    infer_scheduler = FlowMatchEulerDiscreteScheduler.from_config(base_scheduler.config)

    vae_mean = vae_std_tensor = None
    if vae is not None:
        vae_mean = torch.tensor(vae.config.latents_mean).view(
            1, vae.config.z_dim, 1, 1, 1).to(device, vae.dtype)
        vae_std_tensor = torch.tensor(vae.config.latents_std).view(
            1, vae.config.z_dim, 1, 1, 1).to(device, vae.dtype)

    # MODIFICADO V2: Acceder al dataset original si es Subset
    actual_dataset = dataset.dataset if isinstance(dataset, Subset) else dataset
    num_samples = min(num_samples, len(actual_dataset))
    logger.info(f"[Inference epoch {epoch}] {num_samples} imágenes, {num_steps} pasos")

    for sample_idx in range(num_samples):
        sample = actual_dataset[sample_idx]

        prompt = sample["prompt_embeds"]
        if prompt.dim() == 2:
            prompt = prompt.unsqueeze(0)
        prompt = prompt.to(device, dtype=dtype)

        mask = sample["prompt_embeds_mask"]
        if mask.dim() == 1:
            mask = mask.unsqueeze(0)
        mask = mask.to(device)

        target_packed = sample["target_latents_packed"]
        if target_packed.dim() == 2:
            target_packed = target_packed.unsqueeze(0)
        target_packed = target_packed.to(device, dtype=dtype)

        src_packed = sample.get("source_latents_packed")
        if src_packed is not None:
            if src_packed.dim() == 2:
                src_packed = src_packed.unsqueeze(0)
            src_packed = src_packed.to(device, dtype=dtype)

        latent_shape = target_packed.shape
        image_seq_len = latent_shape[1]

        if infer_scheduler.config.get("use_dynamic_shifting", False):
            base_seq   = infer_scheduler.config.get("base_image_seq_len", 256)
            max_seq    = infer_scheduler.config.get("max_image_seq_len", 4096)
            base_shift = infer_scheduler.config.get("base_shift", 0.5)
            max_shift  = infer_scheduler.config.get("max_shift", 1.15)
            m  = (max_shift - base_shift) / (max_seq - base_seq)
            b  = base_shift - m * base_seq
            mu = image_seq_len * m + b
            infer_scheduler.set_timesteps(num_steps, device=device, mu=mu)
        else:
            infer_scheduler.set_timesteps(num_steps, device=device)

        gen = torch.Generator(device=device).manual_seed(42 + sample_idx)
        latents = torch.randn(latent_shape, generator=gen, device=device, dtype=dtype)

        for t in infer_scheduler.timesteps:
            timestep = t.view(1).to(device)
            timestep_norm = (timestep.float() / 1000.0).to(dtype)

            if src_packed is not None:
                latent_model_input = torch.cat([latents, src_packed], dim=1)
            else:
                latent_model_input = latents

            v_pred = model(latent_model_input, prompt, mask, timestep_norm)
            v_pred = v_pred[:, :latent_shape[1], :]

            latents = infer_scheduler.step(
                v_pred.float(), t, latents.float()
            ).prev_sample.to(dtype)

        if vae is not None:
            try:
                def decode_packed(packed):
                    spatial = unpack_latents(packed.float(), img_height, img_width).to(vae.dtype)
                    z_raw   = spatial * vae_std_tensor + vae_mean
                    decoded = vae.decode(z_raw, return_dict=False)[0][:, :, 0]
                    return decoded

                pred_imgs = decode_packed(latents)
                gt_imgs   = decode_packed(target_packed)

                def to_pil(t_img):
                    img = torch.clamp(t_img[0].cpu().float() * 0.5 + 0.5, 0.0, 1.0)
                    return T.ToPILImage()(img)

                pred_pil = to_pil(pred_imgs)
                gt_pil   = to_pil(gt_imgs)
                w, h     = pred_pil.size

                if src_packed is not None:
                    src_imgs = decode_packed(src_packed)
                    src_pil  = to_pil(src_imgs)
                    combined = Image.new("RGB", (w * 3, h))
                    combined.paste(src_pil,  (0, 0))
                    combined.paste(pred_pil, (w, 0))
                    combined.paste(gt_pil,   (w * 2, 0))
                else:
                    combined = Image.new("RGB", (w * 2, h))
                    combined.paste(pred_pil, (0, 0))
                    combined.paste(gt_pil,   (w, 0))

                fname = os.path.join(infer_dir, f"sample_{sample_idx:02d}.png")
                combined.save(fname)
                logger.info(f"[Inference epoch {epoch}] Guardado: {fname}")

            except Exception as e:
                logger.warning(f"[Inference epoch {epoch}] Error sample {sample_idx}: {e}")

    model.train()
    logger.info(f"[Inference epoch {epoch}] Callback completado.")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--latents_dir", type=str, required=True)
    parser.add_argument("--output_dir", type=str, default="../models/finetuned_pp")
    parser.add_argument("--base_model", type=str, default="Qwen/Qwen-Image-Edit-2509")
    parser.add_argument("--transformer_model", type=str, default="../models/Qwen-Fused-Angles")
    
    # HRNet
    parser.add_argument("--hrnet_model_path", type=str, default="./models/pose_hrnet_w48_384x288.pth")
    parser.add_argument("--heatmap_loss_weight", type=float, default=0.5)
    parser.add_argument("--velocity_loss_weight", type=float, default=0.5)
    parser.add_argument("--heatmap_loss_type", type=str, default="mse", choices=["mse", "weighted_mse"])
    
    # Training
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument("--microbatches", type=int, default=4)
    parser.add_argument("--learning_rate", type=float, default=1e-4)
    parser.add_argument("--lora_rank", type=int, default=16)
    parser.add_argument("--lora_alpha", type=int, default=32)
    parser.add_argument("--lora_dropout", type=float, default=0.1)
    parser.add_argument("--inference_every", type=int, default=2)
    parser.add_argument("--inference_steps", type=int, default=4)
    parser.add_argument("--inference_samples", type=int, default=1)
    
    # NUEVO V4: Validación
    parser.add_argument("--val_split", type=float, default=0.1,
                       help="Fracción de datos para validación (0.0-1.0)")
    parser.add_argument("--val_seed", type=int, default=42,
                       help="Semilla para split train/val reproducible")
    
    args, _ = parser.parse_known_args()

    config = TrainingConfig(
        latents_dir=args.latents_dir,
        output_dir=args.output_dir,
        transformer_model=args.transformer_model,
        base_model=args.base_model,
        hrnet_model_path=args.hrnet_model_path,
        velocity_loss_weight=args.velocity_loss_weight,
        heatmap_loss_weight=args.heatmap_loss_weight,
        heatmap_loss_type=args.heatmap_loss_type,
        epochs=args.epochs,
        batch_size=args.batch_size,
        microbatches=args.microbatches,
        lr=args.learning_rate,
        lora_rank=args.lora_rank,
        lora_alpha=args.lora_alpha,
        lora_dropout=args.lora_dropout,
        inference_every=args.inference_every,
        inference_steps=args.inference_steps,
        inference_samples=args.inference_samples,
        val_split=args.val_split,  # NUEVO V4
        val_seed=args.val_seed,     # NUEVO V4
    )

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    dtype = torch.bfloat16

    logger.info(f"Training: 1 GPU ({device}), BS={config.batch_size}")
    logger.info(f"Heatmap Loss Weight: {config.heatmap_loss_weight}")
    logger.info(f"Velocity Loss Weight: {config.velocity_loss_weight}")
    logger.info(f"Heatmap Loss Type: {config.heatmap_loss_type}")
    logger.info(f"Validation Split: {config.val_split} (seed={config.val_seed})")  # NUEVO V4
    os.makedirs(config.output_dir, exist_ok=True)

    # Quantization
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16,
        bnb_4bit_use_double_quant=True,
    )

    # Cargar transformer
    logger.info(f"Cargando modelo desde {config.transformer_model} → {device}")
    transformer = QwenImageTransformer2DModel.from_pretrained(
        config.transformer_model,
        subfolder=None,
        quantization_config=bnb_config,
        torch_dtype=torch.bfloat16,
        device_map=str(device),
        attn_implementation="sdpa",
    )

    transformer = prepare_model_for_kbit_training(transformer, use_gradient_checkpointing=False)

    lora_config = LoraConfig(
        r=config.lora_rank,
        lora_alpha=config.lora_alpha,
        init_lora_weights=True,
        target_modules=["to_q", "to_k", "to_v", "to_out.0",
                       "add_q_proj", "add_k_proj", "add_v_proj", "to_add_out"],
        lora_dropout=config.lora_dropout,
    )
    transformer = get_peft_model(transformer, lora_config)
    transformer.print_trainable_parameters()

    logger.info("Creando QwenSingleGPUWrapper...")
    model = QwenSingleGPUWrapper(transformer)

    # ---------------------------------------------------------------------------
    # Resume: cargar checkpoint previo si existe
    # ---------------------------------------------------------------------------
    start_epoch = 0
    best_loss = float('inf')

    checkpoint_path = os.path.join(config.output_dir, "qwen_lora_best.pt")
    if os.path.exists(checkpoint_path):
        logger.info(f"Checkpoint detectado en {checkpoint_path}, cargando pesos LoRA...")
        try:
            ckpt = torch.load(checkpoint_path, map_location=device, weights_only=True)

            # Soporte para formato nuevo (dict con metadata) y antiguo (flat dict de tensores)
            if isinstance(ckpt, dict) and "lora_state_dict" in ckpt:
                lora_state = ckpt["lora_state_dict"]
                start_epoch = ckpt.get("epoch", -1) + 1
                best_loss = ckpt.get("best_loss", float('inf'))
                logger.info(
                    f"✓ Reanudando desde época {start_epoch} | "
                    f"Mejor val_loss anterior: {best_loss:.6f}"
                )
            else:
                # Formato antiguo: el checkpoint es directamente el state dict
                lora_state = ckpt
                logger.warning(
                    "Checkpoint en formato antiguo (sin metadata). "
                    "Reanudando desde época 0."
                )

            missing, unexpected = model.load_state_dict(lora_state, strict=False)
            loaded_keys = len(lora_state) - len(unexpected)
            logger.info(
                f"{loaded_keys}/{len(lora_state)} pesos LoRA cargados "
                f"({len(missing)} faltantes, {len(unexpected)} inesperados)"
            )
        except Exception as e:
            logger.error(
                f"Error cargando checkpoint: {e}. Iniciando desde cero."
            )
            start_epoch = 0
            best_loss = float('inf')
    else:
        logger.info("No se encontró checkpoint previo. Iniciando entrenamiento desde cero.")

    trainable_params = [p for p in model.parameters() if p.requires_grad]
    logger.info(f"{len(trainable_params)} grupos de parámetros entrenables")
    optimizer = torch.optim.AdamW(trainable_params, lr=config.lr, eps=1e-6)

    diff_scheduler = FlowMatchEulerDiscreteScheduler.from_pretrained(
        config.base_model, subfolder="scheduler"
    )

    # NUEVO V1: Crear dataset completo y splits train/val
    g = torch.Generator()
    g.manual_seed(42)

    full_dataset = LatentsDataset(config.latents_dir)
    
    # Split reproducible
    train_dataset, val_dataset = create_train_val_split(
        full_dataset,
        val_split=config.val_split,
        seed=config.val_seed
    )
    
    # DataLoaders
    train_dataloader = DataLoader(
        train_dataset,
        batch_size=config.batch_size,
        collate_fn=make_collate_latents(full_dataset.global_max_seq_len),
        drop_last=True,
        shuffle=True,
        generator=g,
        num_workers=4,
    )
    
    # NUEVO V1: Val DataLoader (sin shuffle para reproducibilidad)
    val_dataloader = DataLoader(
        val_dataset,
        batch_size=config.batch_size,
        collate_fn=make_collate_latents(full_dataset.global_max_seq_len),
        drop_last=False,  # Usar todos los datos de validación
        shuffle=False,    # Sin shuffle para reproducibilidad
        num_workers=4,
    )

    # VAE y HRNet
    logger.info(f"Cargando VAE en {device}...")
    vae = AutoencoderKLQwenImage.from_pretrained(
        config.base_model, subfolder="vae", torch_dtype=torch.float32
    ).to(device)
    vae.requires_grad_(False)
    vae.eval()
    
    hrnet = None
    if os.path.exists(config.hrnet_model_path):
        logger.info(f"Cargando HRNet desde {config.hrnet_model_path}...")
        hrnet = load_hrnet_model(config.hrnet_model_path, width=48, num_joints=17, device=device)
        hrnet.requires_grad_(False)
        hrnet.eval()
        logger.info("HRNet cargado y congelado correctamente")
    else:
        logger.warning("HRNet no encontrado, solo velocity loss")

    # Loss
    first_sample = torch.load(full_dataset.files[0], weights_only=True)
    img_resolution = first_sample.get("resolution", 1024)
    logger.info(f"Resolución inferida del dataset: {img_resolution}")
    
    loss_fn = CombinedLossFn(
        vae=vae,
        hrnet=hrnet,
        hrnet_input_size=config.hrnet_input_size,
        img_height=img_resolution,
        img_width=img_resolution,
        velocity_weight=config.velocity_loss_weight,
        heatmap_weight=config.heatmap_loss_weight,
        heatmap_loss_type=config.heatmap_loss_type,
        save_dir=os.path.join(config.output_dir, "loss_diagnostics")
    )

    logger.info("Listo para entrenar.")
    model.train()

    # CSV: cabecera solo si empezamos desde cero; si reanudamos, abrimos en modo append
    csv_file_path = os.path.join(config.output_dir, "training_metrics.csv")
    csv_mode = "a" if start_epoch > 0 else "w"
    with open(csv_file_path, mode=csv_mode, newline="") as f:
        writer = csv.writer(f)
        if csv_mode == "w":
            writer.writerow(["epoch", "train_loss", "val_loss", "train_pck", "val_pck"])

    for epoch in range(start_epoch, config.epochs):
        logger.info(f"Epoch {epoch} start")

        # TRAINING
        iterator = tqdm(train_dataloader, desc=f"Epoch {epoch}")
        avg_loss = 0.0
        avg_pck  = 0.0
        steps = 0

        for step, batch in enumerate(iterator):
            optimizer.zero_grad()

            target = batch["target_latents_packed"].to(device, dtype=dtype)
            source = batch["source_latents_packed"].to(device, dtype=dtype)
            prompt = batch["prompt_embeds"].to(device, dtype=dtype)
            mask = batch["prompt_embeds_mask"].to(device)
            target_heatmaps = batch["target_heatmaps"]

            bsz = target.shape[0]

            g_seed = 42 + epoch * 10000 + step
            gen = torch.Generator(device=device).manual_seed(g_seed)

            timesteps = torch.randint(
                0, diff_scheduler.config.num_train_timesteps,
                (bsz,), generator=gen, device=device
            ).long()

            noise = torch.randn(target.shape, generator=gen, device=device, dtype=dtype)

            t_norm = (timesteps.float() / diff_scheduler.config.num_train_timesteps).to(dtype)
            t_norm = t_norm.view(-1, 1, 1)
            noisy = (1.0 - t_norm) * target + t_norm * noise

            velocity_target = noise - target
            timestep_norm = (timesteps.float() / 1000.0).to(dtype)

            latent_model_input = torch.cat([noisy, source], dim=1)
            v_pred = model(latent_model_input, prompt, mask, timestep_norm)

            loss_fn.set_batch_context(target_heatmaps, timesteps)
            combined_target = torch.stack([velocity_target, noisy], dim=1)
            loss = loss_fn(v_pred, combined_target)

            loss.backward()
            torch.nn.utils.clip_grad_norm_(
                [p for p in model.parameters() if p.requires_grad],
                max_norm=1.0
            )
            optimizer.step()

            step_loss = loss.item()
            avg_loss += step_loss
            avg_pck  += loss_fn.last_pck
            steps += 1

        # NUEVO V2: VALIDATION al final de cada época
        logger.info(f"Ejecutando validación...")
        val_loss, val_pck = validate(
            model=model,
            val_dataloader=val_dataloader,
            loss_fn=loss_fn,
            diff_scheduler=diff_scheduler,
            device=device,
            dtype=dtype,
        )

        global_avg_loss = avg_loss / steps if steps > 0 else 0.0
        global_avg_pck  = avg_pck  / steps if steps > 0 else 0.0

        logger.info(
            f"Epoch {epoch} | "
            f"Train Loss: {global_avg_loss:.6f} | Val Loss: {val_loss:.6f} | "
            f"Train PCK: {global_avg_pck:.4f} | Val PCK: {val_pck:.4f} | "
            f"Mejor histórica: {best_loss:.6f}"
        )
        
        # NUEVO V3: Escribir train_loss, val_loss, train_pck y val_pck en CSV
        with open(csv_file_path, mode="a", newline="") as f:
            writer = csv.writer(f)
            writer.writerow([epoch, global_avg_loss, val_loss, global_avg_pck, val_pck])

        # Checkpoint (usar val_loss en lugar de train_loss)
        if val_loss < best_loss:  # MODIFICADO V3: usar val_loss
            best_loss = val_loss
            logger.info(f"🟢 ¡Nueva mejor val_loss! Guardando checkpoint...")

            local_lora = {
                k: v.cpu()
                for k, v in model.state_dict().items()
                if "lora" in k
            }
            save_path = os.path.join(config.output_dir, "qwen_lora_best.pt")
            torch.save({
                "lora_state_dict": local_lora,
                "epoch": epoch,
                "best_loss": best_loss,
            }, save_path)
            logger.info(f"LoRA guardado en: {save_path} (época {epoch}, val_loss={best_loss:.6f})")
        else:
            logger.info(f"⚪ Val_loss no mejoró. Saltando guardado.")

        # Callback de inferencia
        if config.inference_every > 0 and (epoch + 1) % config.inference_every == 0:
            run_inference_callback(
                model=model,
                vae=vae,
                base_scheduler=diff_scheduler,
                device=device,
                dataset=train_dataset,  # MODIFICADO V2: usar train_dataset
                save_dir=config.output_dir,
                epoch=epoch,
                img_height=img_resolution,
                img_width=img_resolution,
                num_steps=config.inference_steps,
                num_samples=config.inference_samples,
                dtype=dtype,
            )


if __name__ == "__main__":
    main()