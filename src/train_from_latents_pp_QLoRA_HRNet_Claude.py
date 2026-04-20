"""
Script de Fine-tuning con Pipeline Parallelism (2 GPUs) usando torch.distributed.pipelining.

VERSIÓN MODIFICADA CON HEATMAP LOSS DE HRNET

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

NUEVAS MODIFICACIONES (HEATMAP LOSS):
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

Usage:
    torchrun --nproc_per_node 2 train_from_latents_pp_QLoRA_HRNet.py \\
        --latents_dir "/ruta/precomputed_latents" \\
        --hrnet_model_path "./models/pose_hrnet_w48_384x288.pth" \\
        --output_dir "output_hrnet" \\
        --batch_size 4 \\
        --microbatches 4 \\
        --epochs 200 \\
        --learning_rate 1e-4 \\
        --heatmap_loss_weight 0.5 \\
        --velocity_loss_weight 0.5 \\
        --heatmap_loss_type "mse"


        --------------------------------------------------------------------------------------------------------
         torchrun --nproc_per_node 2 train_from_latents_pp_QLoRA_HRNet_Claude.py --latents_dir "/data/antoniodetoro/qwen/dataset_local_latents_512_heatmaps/" --hrnet_model_path /nas/antoniodetoro/qwen/Qwen-Image-Edit-Angles-2/src/hr_net/models/hrnet_finetuned_best.pth --output_dir output_qwen_HRNet --batch_size 4 --microbatches 4 --epochs 20 --heatmap_loss_weight 0.5 --velocity_loss_weight 0.5 --heatmap_loss_type "mse"

"""

import os
import sys
import csv  # Ya estaba

current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
sys.path.append(parent_dir)

import argparse
from pathlib import Path
from dataclasses import dataclass
from typing import List, Tuple, Optional  # MODIFICADO H1: añadido Optional
from torch.utils.checkpoint import checkpoint

os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["HF_HOME"] = "/nas/antoniodetoro/qwen/hf_cache"
os.environ["TMPDIR"] = "/dev/shm"
os.environ["PYTHONNOUSERSITE"] = "1"

import torch
import logging
import torch.nn as nn
import torch.nn.functional as F
import torch.distributed as dist
import torchvision.transforms as T
from PIL import Image
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm
from diffusers.optimization import get_scheduler
from diffusers import FlowMatchEulerDiscreteScheduler, QwenImageTransformer2DModel
from diffusers.models import AutoencoderKLQwenImage
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
from torch.distributed.pipelining import PipelineStage
from torch.distributed.pipelining.schedules import Schedule1F1B
from transformers import BitsAndBytesConfig


logging.basicConfig(
    level=logging.INFO,
    force=True,
    format="[Rank %(process)d] %(message)s",
)
logger = logging.getLogger(__name__)


# =========================================================================
# NUEVO H1: ARQUITECTURA HRNET COMPLETA
# Copiada del script hrnet_inference.py para cálculo de heatmaps
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
    Configuración: W48 (COCO 17 keypoints).
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


def load_hrnet_model(
    model_path: str,
    width: int = 48,
    num_joints: int = 17,
    device: torch.device = torch.device('cpu')
) -> PoseHRNet:
    """
    NUEVO H1: Carga HRNet pre-entrenado.
    
    Args:
        model_path: Ruta al checkpoint .pth
        width: Ancho del modelo (48 para W48)
        num_joints: Número de keypoints (17 para COCO)
        device: Device donde cargar
    
    Returns:
        Modelo HRNet cargado y en eval mode
    """
    model = PoseHRNet(width=width, num_joints=num_joints)
    
    try:
        # PyTorch >= 2.6 requiere weights_only=False para checkpoints de HRNet
        checkpoint = torch.load(model_path, map_location=device, weights_only=False)
    except TypeError:
        # Compatibilidad con versiones antiguas
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
    
    # NUEVO H1: Configuración HRNet
    hrnet_model_path: str = "./models/pose_hrnet_w48_384x288.pth"
    hrnet_input_size: Tuple[int, int] = (288, 384)  # (H, W) para HRNet
    
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
    
    # NUEVO H1: Pesos de la loss combinada
    velocity_loss_weight: float = 0.5  # alpha
    heatmap_loss_weight: float = 0.5   # beta
    heatmap_loss_type: str = "mse"     # "mse" o "weighted_mse"


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


def make_collate_latents(global_max_seq_len: int):
    """
    MODIFICADO H5: Collate que padea prompt_embeds y EXTRAE target_heatmaps.
    
    CAMBIO: Ahora extrae target_heatmaps de los archivos .pt y los incluye
    en el batch retornado.
    """
    def collate_latents(batch):
        target_latents  = torch.cat([item["target_latents_packed"]  for item in batch], dim=0)
        source_latents  = torch.cat([item["source_latents_packed"]  for item in batch], dim=0)
        prompt_list     = [item["prompt_embeds"]      for item in batch]
        mask_list       = [item["prompt_embeds_mask"] for item in batch]
        
        # NUEVO H5: Extraer heatmaps GT de los .pt
        target_heatmaps_list = []
        for item in batch:
            if "target_heatmaps" in item:
                target_heatmaps_list.append(item["target_heatmaps"])
            else:
                logger.warning("target_heatmaps no encontrado en batch item, usando ceros")
                # Placeholder: (17, 72, 96) según shape confirmado
                target_heatmaps_list.append(torch.zeros(17, 72, 96, dtype=torch.float32))
        
        target_heatmaps = torch.cat(target_heatmaps_list, dim=0)  # (B, 17, 72, 96)

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
            "target_heatmaps":       target_heatmaps,  # NUEVO H5
        }
    return collate_latents


# ---------------------------------------------------------------------------
# Helpers: pack / unpack latents
# ---------------------------------------------------------------------------

def unpack_latents(latents: torch.Tensor, height: int, width: int, vae_scale_factor: int = 8) -> torch.Tensor:
    """
    Invierte _pack_latents: (B, N_patches, z_dim*4) → (B, z_dim, 1, H, W).
    La dimensión de frame (=1) es necesaria para vae.decode de Qwen.
    """
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
    """
    Desnormaliza + decodifica latentes empaquetados a lista de PIL.
    FIX T3: desnormalización correcta.
    Los latentes guardados están normalizados: z_norm = (z_raw - mean) / std
    Para el VAE necesitamos z_raw:  z_raw = z_norm * std + mean
    """
    spatial = unpack_latents(latents_packed.float(), img_height, img_width).to(vae.dtype)

    # FIX T3: z_norm * std + mean  (NO dividir por std)
    vae_mean = torch.tensor(vae.config.latents_mean).view(
        1, vae.config.z_dim, 1, 1, 1).to(spatial.device, spatial.dtype)
    vae_std  = torch.tensor(vae.config.latents_std).view(
        1, vae.config.z_dim, 1, 1, 1).to(spatial.device, spatial.dtype)
    z_raw = spatial * vae_std + vae_mean

    with torch.no_grad():
        decoded = vae.decode(z_raw, return_dict=False)[0]  # (B, C, 1, H, W)
    decoded = decoded[:, :, 0]  # (B, C, H, W)

    pils = []
    for img_t in decoded:
        img_t = torch.clamp((img_t.cpu().float() * 0.5 + 0.5), 0.0, 1.0)
        pils.append(T.ToPILImage()(img_t))
    return pils


# NUEVO H3: Función para decodificar latentes a tensores de imágenes
def latents_to_images(
    latents_packed: torch.Tensor,
    vae: AutoencoderKLQwenImage,
    img_height: int,
    img_width: int,
) -> torch.Tensor:
    """
    NUEVO H3: Desnormaliza + decodifica latentes empaquetados a tensor de imágenes.
    
    Args:
        latents_packed: (B, Nv, 64) latentes empaquetados normalizados
        vae: VAE decoder
        img_height, img_width: resolución objetivo
    
    Returns:
        Tensor (B, 3, H, W) en rango [-1, 1] (salida directa del VAE)
    """
    spatial = unpack_latents(latents_packed.float(), img_height, img_width).to(vae.dtype)

    # Desnormalización: z_raw = z_norm * std + mean
    vae_mean = torch.tensor(vae.config.latents_mean).view(
        1, vae.config.z_dim, 1, 1, 1).to(spatial.device, spatial.dtype)
    vae_std  = torch.tensor(vae.config.latents_std).view(
        1, vae.config.z_dim, 1, 1, 1).to(spatial.device, spatial.dtype)
    z_raw = spatial * vae_std + vae_mean

    with torch.no_grad():
        decoded = vae.decode(z_raw, return_dict=False)[0]  # (B, C, 1, H, W)
    decoded = decoded[:, :, 0]  # (B, C, H, W)
    
    return decoded  # Ya está en rango [-1, 1]


# NUEVO H2: Función de preprocesamiento para HRNet
def preprocess_image_for_hrnet(
    image: torch.Tensor,
    target_size: Tuple[int, int] = (288, 384)
) -> torch.Tensor:
    """
    NUEVO H2: Preprocesa imagen para HRNet con normalización ImageNet.
    
    Args:
        image: Tensor (B, C, H, W) en rango [-1, 1] (salida del VAE)
        target_size: (height, width) para HRNet (288, 384 para W48)
    
    Returns:
        Tensor (B, C, H, W) normalizado para HRNet
    """
    # Paso 1: Convertir de [-1, 1] a [0, 1]
    image = (image + 1.0) / 2.0
    
    # Paso 2: Resize a tamaño esperado por HRNet
    image = F.interpolate(image, size=target_size, mode='bilinear', align_corners=True)
    
    # Paso 3: Normalización ImageNet (según script hrnet_inference.py)
    mean = torch.tensor([0.485, 0.456, 0.406], device=image.device).view(1, 3, 1, 1)
    std = torch.tensor([0.229, 0.224, 0.225], device=image.device).view(1, 3, 1, 1)
    
    image = (image - mean) / std
    return image


# ---------------------------------------------------------------------------
# Loss: NUEVA clase CombinedLossFn (REEMPLAZA VelocityLossFn)
# ---------------------------------------------------------------------------

class CombinedLossFn:
    """
    NUEVO H4: Loss combinada que reemplaza VelocityLossFn.
    
    Calcula:
        total_loss = alpha * velocity_loss + beta * heatmap_loss
    
    donde:
        velocity_loss = MSE(v_pred, v_target)  [loss original]
        heatmap_loss = MSE(HRNet(VAE(x0_pred)), target_heatmaps)  [nueva loss]
    
    Proceso:
        1. Calcula velocity loss (igual que antes)
        2. Reconstruye x0_pred = noisy - t * v_pred (matemáticamente correcto)
        3. Decodifica x0_pred con VAE → imágenes
        4. Preprocesa imágenes para HRNet
        5. Calcula heatmaps con HRNet (congelado)
        6. Compara con GT heatmaps (MSE o weighted MSE)
        7. Combina ambas losses con pesos configurables
    """

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
        """
        Args:
            vae: VAE decoder (solo en Rank 1)
            hrnet: HRNet modelo (solo en Rank 1, congelado)
            hrnet_input_size: (H, W) input size para HRNet
            img_height, img_width: Resolución de las imágenes del dataset
            velocity_weight: Peso alpha para velocity loss
            heatmap_weight: Peso beta para heatmap loss
            heatmap_loss_type: "mse" o "weighted_mse"
            save_dir: Directorio para guardar diagnósticos (opcional)
        """
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
        
        # Storage para heatmaps GT y timesteps (se actualizan desde fuera)
        self.current_target_heatmaps = None
        self.current_timesteps = None
        
        # NUEVO H4: Pesos para weighted MSE (prioriza joints centrales del cuerpo)
        # COCO keypoints: [nose, l_eye, r_eye, l_ear, r_ear, l_shoulder, r_shoulder, 
        #                  l_elbow, r_elbow, l_wrist, r_wrist, l_hip, r_hip, 
        #                  l_knee, r_knee, l_ankle, r_ankle]
        self.keypoint_weights = torch.tensor([
            1.0,  # nose
            0.8, 0.8,  # eyes
            0.6, 0.6,  # ears
            1.5, 1.5,  # shoulders (más importantes para estructura corporal)
            1.2, 1.2,  # elbows
            1.0, 1.0,  # wrists
            1.5, 1.5,  # hips (más importantes para estructura corporal)
            1.2, 1.2,  # knees
            1.0, 1.0,  # ankles
        ], dtype=torch.float32)

    def set_batch_context(self, target_heatmaps: torch.Tensor, timesteps: torch.Tensor):
        """
        NUEVO H4: Actualiza contexto del batch actual.
        Debe llamarse antes de cada schedule.step() en el training loop.
        
        Args:
            target_heatmaps: (B, 17, 72, 96) heatmaps GT del batch
            timesteps: (B,) timesteps del batch para reconstruir x0
        """
        self.current_target_heatmaps = target_heatmaps
        self.current_timesteps = timesteps

    def __call__(self, outputs: torch.Tensor, combined_target: torch.Tensor) -> torch.Tensor:
        """
        Calcula loss combinada.
        
        Args:
            outputs: v_pred del modelo (B, Seq, C)
            combined_target: (B, 2, Seq, C) con [v_target, noisy_core]
        
        Returns:
            loss total ponderada
        """
        device = outputs.device
        
        # ──────────────────────────────────────────────────────────────────
        # 1. VELOCITY LOSS (loss original, sin cambios)
        # ──────────────────────────────────────────────────────────────────
        v_target = combined_target[:, 0, :, :].float()  # (B, Seq, C)
        v_pred = outputs.float()
        
        # Recortar tokens de condición si el modelo los devuelve
        if v_pred.shape[1] > v_target.shape[1]:
            v_pred = v_pred[:, :v_target.shape[1], :]

        # Validación de NaN/Inf
        if torch.isnan(v_pred).any() or torch.isinf(v_pred).any():
            logger.warning(
                f"[LOSS step {self.step_counter}] NaN/Inf en v_pred. "
                f"max={v_pred.abs().nanmax().item():.2f}"
            )
            self.step_counter += 1
            return torch.tensor(0.0, device=device, requires_grad=True)

        velocity_loss_val = F.mse_loss(v_pred, v_target, reduction="mean")
        
        # ──────────────────────────────────────────────────────────────────
        # 2. HEATMAP LOSS (NUEVA)
        # ──────────────────────────────────────────────────────────────────
        heatmap_loss_val = torch.tensor(0.0, device=device)
        
        if (self.vae is not None and 
            self.hrnet is not None and 
            self.current_target_heatmaps is not None and
            self.current_timesteps is not None):
            
            try:
                # NUEVO H4a: Reconstruir x0_pred desde velocity (matemáticamente correcto)
                noisy_latents = combined_target[:, 1, :, :]  # (B, Seq, C)
                
                # Normalizar timesteps a [0, 1]
                # Los timesteps vienen como enteros [0, num_train_timesteps]
                # Asumiendo num_train_timesteps = 1000 (estándar en flow matching)
                t_normalized = self.current_timesteps.float() / 1000.0
                t_normalized = t_normalized.view(-1, 1, 1).to(v_pred.dtype)
                
                # x0_pred = noisy - t * v_pred
                # Esta es la imagen limpia predicha por el modelo
                x0_pred = noisy_latents - t_normalized * v_pred  # (B, Seq, C)
                
                # NUEVO H4b: Decodificar x0_pred a imágenes
                decoded_images = latents_to_images(
                    x0_pred,
                    self.vae,
                    self.img_height,
                    self.img_width
                )  # (B, 3, H, W) en [-1, 1]
                
                # NUEVO H4c: Preprocesar para HRNet (normalización ImageNet)
                hrnet_input = preprocess_image_for_hrnet(
                    decoded_images,
                    self.hrnet_input_size
                )  # (B, 3, 288, 384) normalizado
                
                # NUEVO H4d: Forward HRNet (congelado, sin gradientes)
                with torch.no_grad():
                    pred_heatmaps = self.hrnet(hrnet_input)  # (B, 17, 72, 96)
                
                # NUEVO H4e: Preparar GT heatmaps
                target_hm = self.current_target_heatmaps.to(device).float()
                
                # Verificar shapes y hacer resize si es necesario
                # (Deben ser (B, 17, 72, 96) ambos)
                if target_hm.shape[-2:] != pred_heatmaps.shape[-2:]:
                    target_hm = F.interpolate(
                        target_hm,
                        size=pred_heatmaps.shape[-2:],
                        mode='bilinear',
                        align_corners=True
                    )
                
                # NUEVO H4f: Calcular loss según tipo
                if self.heatmap_loss_type == "weighted_mse":
                    # Weighted MSE: dar más peso a joints importantes
                    weights = self.keypoint_weights.to(device).view(1, 17, 1, 1)
                    diff = (pred_heatmaps - target_hm) ** 2
                    heatmap_loss_val = (weights * diff).mean()
                else:  # "mse" (default)
                    heatmap_loss_val = F.mse_loss(
                        pred_heatmaps,
                        target_hm,
                        reduction="mean"
                    )
                
            except Exception as e:
                logger.warning(f"[LOSS step {self.step_counter}] Error en heatmap loss: {e}")
                import traceback
                traceback.print_exc()
                heatmap_loss_val = torch.tensor(0.0, device=device)
        
        # ──────────────────────────────────────────────────────────────────
        # 3. LOSS COMBINADA PONDERADA
        # ──────────────────────────────────────────────────────────────────
        total_loss = (
            self.velocity_weight * velocity_loss_val +
            self.heatmap_weight * heatmap_loss_val
        )
        
        # Logging cada 50 pasos
        if self.step_counter % 50 == 0:
            logger.info(
                f"[LOSS step {self.step_counter}] "
                f"Velocity: {velocity_loss_val.item():.6f}, "
                f"Heatmap: {heatmap_loss_val.item():.6f}, "
                f"Total: {total_loss.item():.6f} "
                f"(α={self.velocity_weight:.2f}, β={self.heatmap_weight:.2f})"
            )
        
        self.step_counter += 1
        return total_loss


# ---------------------------------------------------------------------------
# Distributed init
# ---------------------------------------------------------------------------

def init_distributed():
    rank       = int(os.environ["RANK"])
    world_size = int(os.environ["WORLD_SIZE"])
    local_rank = int(os.environ["LOCAL_RANK"])

    if torch.cuda.is_available():
        # Rank 0 → GPU 1 (24 GB), Rank 1 → GPU 0 (32 GB, tiene el VAE para inferencia)
        device_id = 1 if local_rank == 0 else 0
        torch.cuda.set_device(device_id)
        device = torch.device(f"cuda:{device_id}")
        backend = "nccl"
        logger.info(f"Rank {rank} mapeado a GPU {device_id}")
    else:
        device = torch.device("cpu")
        backend = "gloo"

    if not dist.is_initialized():
        dist.init_process_group(backend=backend)

    pp_group = dist.new_group()
    return rank, world_size, device, pp_group


# ---------------------------------------------------------------------------
# Model wrapper con pipeline split
# ---------------------------------------------------------------------------

class QwenSplitWrapper(nn.Module):
    """
    Divide QwenImageTransformer2DModel en dos mitades para pipeline parallelism.
    Rank 0: embeddings + bloques 0..N/2-1
    Rank 1: bloques N/2..N-1 + norm_out + proj_out
    """

    def __init__(self, model: nn.Module, rank: int, world_size: int):
        super().__init__()
        self.model      = model
        self.rank       = rank
        self.world_size = world_size

        # PEFT envuelve el modelo: PeftModel → LoraModel → QwenImageTransformer2DModel
        if hasattr(model, "base_model") and hasattr(model.base_model, "model"):
            self.inner_model = model.base_model.model
        else:
            self.inner_model = model

        total_layers = len(self.inner_model.transformer_blocks)
        split_layer  = total_layers // 2
        logger.info(f"Rank {rank}: {total_layers} bloques totales, split en {split_layer}.")

        if rank == 0:
            keep = list(self.inner_model.transformer_blocks[:split_layer])
            drop = list(self.inner_model.transformer_blocks[split_layer:])
            self.inner_model.transformer_blocks = nn.ModuleList(keep)
            for attr in ["norm_out", "proj_out"]:
                if hasattr(self.inner_model, attr):
                    setattr(self.inner_model, attr, None)
            for b in drop:
                del b

        elif rank == 1:
            keep = list(self.inner_model.transformer_blocks[split_layer:])
            drop = list(self.inner_model.transformer_blocks[:split_layer])
            self.inner_model.transformer_blocks = nn.ModuleList(keep)
            for attr in ["img_in", "txt_norm", "txt_in", "time_text_embed"]:
                if hasattr(self.inner_model, attr):
                    try:
                        delattr(self.inner_model, attr)
                    except AttributeError:
                        pass
            for b in drop:
                del b

    # ── Helpers ─────────────────────────────────────────────────────────────

    def _compute_img_shapes(self, hidden_states: torch.Tensor, batch_size: int) -> List:
        """
        Infiere img_shapes desde los tokens empaquetados.
        Si seq = 2 * N (dos imágenes concatenadas, noisy+source) → [(1,g,g),(1,g,g)]
        Si seq = N (una imagen sola) → [(1,g,g)]
        """
        seq_len   = hidden_states.shape[1]
        grid_sq   = int(seq_len ** 0.5)
        if grid_sq * grid_sq == seq_len:
            return [[(1, grid_sq, grid_sq)]] * batch_size
        half = seq_len // 2
        grid_sq = int(half ** 0.5)
        if grid_sq * grid_sq != half:
            raise ValueError(
                f"_compute_img_shapes: seq_len={seq_len} no es ni cuadrado perfecto "
                f"ni el doble de uno. No se puede inferir img_shapes."
            )
        return [[(1, grid_sq, grid_sq), (1, grid_sq, grid_sq)]] * batch_size

    def _block_checkpoint(self, block, h, e, mask, temb, r0, r1):
        """Wrapper para gradient checkpoint compatible con tensores complejos."""
        def _fwd(h, e, mask, temb, r0, r1):
            # FIX: Comparamos el número total de elementos para ser infalibles.
            # La imagen (2048 tokens) siempre tendrá más elementos que el texto (~227 tokens).
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

    # ── Forward ─────────────────────────────────────────────────────────────

    def forward(self, *inputs):
        if self.rank == 0:
            hidden_states, encoder_hidden_states, encoder_hidden_states_mask, timestep = inputs

            # Proyecciones de entrada
            hidden_states           = self.inner_model.img_in(hidden_states)
            timestep                = timestep.to(hidden_states.dtype)
            encoder_hidden_states   = self.inner_model.txt_norm(encoder_hidden_states)
            encoder_hidden_states   = self.inner_model.txt_in(encoder_hidden_states)

            # Timestep embedding
            # FIX T8: timestep ya viene dividido por 1000 desde el training loop.
            # QwenTimestepProjEmbeddings.forward(timestep, hidden_states) es la firma
            # correcta (sin guidance).
            temb = self.inner_model.time_text_embed(timestep, hidden_states)
            temb = temb.to(dtype=hidden_states.dtype)

            # RoPE
            # FIX T4 (Solución final): diffusers sigue esperando una lista de listas
            # de tuplas [[(frames, height, width)]] para la imagen, y un entero
            # para max_txt_seq_len.
            B = hidden_states.shape[0]
            seq_len = hidden_states.shape[1]
            
            grid_sq = int(seq_len ** 0.5)
            if grid_sq * grid_sq == seq_len:
                # Una sola imagen (T=1)
                img_shapes = [[(1, grid_sq, grid_sq)]] * B
            else:
                # Dos imágenes concatenadas en la secuencia: noisy + source (T=2)
                half = seq_len // 2
                grid_sq = int(half ** 0.5)
                img_shapes = [[(2, grid_sq, grid_sq)]] * B

            full_len = encoder_hidden_states.shape[1]

            image_rotary_emb = self.inner_model.pos_embed(
                img_shapes,               # Pasado como lista de tuplas nativa
                max_txt_seq_len=full_len,  # Nuevo argumento esperado por diffusers
                device=hidden_states.device
            )
            r0, r1 = image_rotary_emb

            # Bloques de la primera mitad
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

            # La máscara es int/bool; la casteamos a float para que PipelineStage
            # pueda manejar el tensor como activación flotante.
            mask_float = encoder_hidden_states_mask.to(dtype=hidden_states.dtype)
            return (hidden_states, encoder_hidden_states, mask_float, temb, r0.real.contiguous(), r0.imag.contiguous(), r1.real.contiguous(), r1.imag.contiguous())

        elif self.rank == 1:
            hidden_states, encoder_hidden_states, mask_float, temb, r0_real, r0_imag, r1_real, r1_imag = inputs
            encoder_hidden_states_mask = mask_float.to(torch.int64)
            r0 = torch.complex(r0_real, r0_imag)
            r1 = torch.complex(r1_real, r1_imag)

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

            # Conectar el grafo a los inputs para evitar que autograd los suelte.
            # Se hace en float32 con ×0 para no afectar valores ni causar overflow.
            anchor = torch.tensor(0.0, dtype=torch.float32, device=hidden_states.device)
            for inp in inputs:
                if isinstance(inp, torch.Tensor) and inp.requires_grad:
                    val = inp.float().mean()
                    if val.is_complex():
                        val = val.real
                    anchor = anchor + val * 0.0
            hidden_states = hidden_states + anchor.to(hidden_states.dtype)

            return hidden_states


# ---------------------------------------------------------------------------
# Inferencia callback
# ---------------------------------------------------------------------------

@torch.no_grad()
def run_inference_callback(
    model_split, vae, base_scheduler, device, rank,
    dataset, save_dir, epoch,
    img_height=1024, img_width=1024,
    num_steps=4, num_samples=2, dtype=torch.bfloat16,
):
    """
    Ejecuta un bucle de denoising completo usando los dos pipeline stages.
    Las imágenes se guardan en rank 1 (que tiene el VAE).
    """
    model_split.eval()

    infer_dir = os.path.join(save_dir, f"inference_epoch_{epoch:03d}")
    if rank == 1:
        os.makedirs(infer_dir, exist_ok=True)

    infer_scheduler = FlowMatchEulerDiscreteScheduler.from_config(base_scheduler.config)

    # Constantes VAE para desnormalización (solo rank 1)
    vae_mean = vae_std_tensor = None
    if rank == 1 and vae is not None:
        # FIX T3: z_raw = z_norm * std + mean
        vae_mean       = torch.tensor(vae.config.latents_mean).view(
            1, vae.config.z_dim, 1, 1, 1).to(device, vae.dtype)
        vae_std_tensor = torch.tensor(vae.config.latents_std).view(
            1, vae.config.z_dim, 1, 1, 1).to(device, vae.dtype)

    num_samples = min(num_samples, len(dataset))
    logger.info(f"[Inference epoch {epoch}] {num_samples} imágenes, {num_steps} pasos")

    for sample_idx in range(num_samples):
        sample = dataset[sample_idx]

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

        latent_shape = target_packed.shape  # (1, Nv, 64)
        image_seq_len = latent_shape[1]

        # Configurar scheduler con mu dinámico
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

        gen     = torch.Generator(device=device).manual_seed(42 + sample_idx)
        latents = torch.randn(latent_shape, generator=gen, device=device, dtype=dtype)

        for t in infer_scheduler.timesteps:
            timestep = t.view(1).to(device)
            # FIX T8: dividir por 1000 igual que en la pipeline original
            timestep_norm = (timestep.float() / 1000.0).to(dtype)

            if rank == 0:
                if src_packed is not None:
                    latent_model_input = torch.cat([latents, src_packed], dim=1)
                else:
                    latent_model_input = latents

                intermediate = model_split(latent_model_input, prompt, mask, timestep_norm)

                meta = []
                for t_tensor in intermediate:
                    meta.append({
                        "shape":      list(t_tensor.shape),
                        "dtype":      str(t_tensor.dtype),
                        "is_complex": t_tensor.is_complex(),
                    })
                dist.broadcast_object_list([meta], src=0)

                for t_tensor in intermediate:
                    if t_tensor.is_complex():
                        dist.send(t_tensor.real.contiguous().float(), dst=1)
                        dist.send(t_tensor.imag.contiguous().float(), dst=1)
                    else:
                        dist.send(t_tensor.contiguous(), dst=1)

                dist.recv(latents, src=1)

            elif rank == 1:
                container = [None]
                dist.broadcast_object_list(container, src=0)
                meta = container[0]

                received = []
                for info in meta:
                    shape  = info["shape"]
                    dt_str = info["dtype"].split(".")[-1]
                    dtype_t = getattr(torch, dt_str)
                    if info["is_complex"]:
                        real = torch.empty(shape, dtype=torch.float32, device=device)
                        imag = torch.empty(shape, dtype=torch.float32, device=device)
                        dist.recv(real, src=0)
                        dist.recv(imag, src=0)
                        received.append(torch.complex(real, imag))
                    else:
                        tensor = torch.empty(shape, dtype=dtype_t, device=device)
                        dist.recv(tensor, src=0)
                        received.append(tensor)

                v_pred = model_split(*received)
                v_pred = v_pred[:, :latent_shape[1], :]

                latents = infer_scheduler.step(
                    v_pred.float(), t, latents.float()
                ).prev_sample.to(dtype)

                dist.send(latents.contiguous(), dst=0)

        # Guardar imágenes (solo rank 1)
        if rank == 1 and vae is not None:
            try:
                def decode_packed(packed):
                    spatial = unpack_latents(packed.float(), img_height, img_width).to(vae.dtype)
                    # FIX T3: z_raw = z_norm * std + mean
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
                    combined.paste(src_pil,  (0,     0))
                    combined.paste(pred_pil, (w,     0))
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

    model_split.train()
    logger.info(f"[Inference epoch {epoch}] Callback completado.")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--latents_dir",       type=str, required=True)
    parser.add_argument("--output_dir",        type=str, default="../models/finetuned_pp")
    parser.add_argument("--base_model",        type=str, default="Qwen/Qwen-Image-Edit-2509")
    parser.add_argument("--transformer_model", type=str, default="../models/Qwen-Fused-Angles")
    
    # NUEVO H8: Argumentos para HRNet
    parser.add_argument("--hrnet_model_path",  type=str, default="./models/pose_hrnet_w48_384x288.pth",
                       help="Ruta al modelo HRNet pre-entrenado")
    parser.add_argument("--heatmap_loss_weight", type=float, default=0.5,
                       help="Peso beta de la heatmap loss (0.0-1.0)")
    parser.add_argument("--velocity_loss_weight", type=float, default=0.5,
                       help="Peso alpha de la velocity loss (0.0-1.0)")
    parser.add_argument("--heatmap_loss_type", type=str, default="mse",
                       choices=["mse", "weighted_mse"],
                       help="Tipo de loss para heatmaps: mse (simple) o weighted_mse (pondera joints)")
    
    parser.add_argument("--epochs",            type=int, default=3)
    parser.add_argument("--batch_size",        type=int, default=4)
    parser.add_argument("--microbatches",      type=int, default=4)
    parser.add_argument("--learning_rate",     type=float, default=1e-4)
    parser.add_argument("--lora_rank",         type=int, default=16)
    parser.add_argument("--lora_alpha",        type=int, default=32)
    parser.add_argument("--lora_dropout",      type=float, default=0.1)
    parser.add_argument("--inference_every",   type=int, default=2)
    parser.add_argument("--inference_steps",   type=int, default=4)
    parser.add_argument("--inference_samples", type=int, default=1)
    args, _ = parser.parse_known_args()

    config = TrainingConfig(
        latents_dir=args.latents_dir,
        output_dir=args.output_dir,
        transformer_model=args.transformer_model,
        base_model=args.base_model,
        
        # NUEVO H8: Configuración HRNet
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
    )

    rank, world_size, device, pp_group = init_distributed()
    dtype = torch.bfloat16   # bfloat16 más estable que float16 para flow matching

    if world_size != 2:
        if rank == 0:
            logger.error("Se requieren exactamente 2 GPUs. Usa torchrun --nproc_per_node=2")
        return

    if rank == 0:
        logger.info(f"Training: {world_size} GPUs, BS={config.batch_size}, micro={config.microbatches}")
        # NUEVO H8: Logging de configuración HRNet
        logger.info(f"Heatmap Loss Weight: {config.heatmap_loss_weight}")
        logger.info(f"Velocity Loss Weight: {config.velocity_loss_weight}")
        logger.info(f"Heatmap Loss Type: {config.heatmap_loss_type}")
        os.makedirs(config.output_dir, exist_ok=True)

    # ── 1. Quantization config ───────────────────────────────────────────────
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16,  # bfloat16 más estable que float16
        bnb_4bit_use_double_quant=True,
    )

    # ── 2. Cargar transformer cuantizado ─────────────────────────────────────
    logger.info(f"Rank {rank}: cargando modelo desde {config.transformer_model} → {device}")
    transformer = QwenImageTransformer2DModel.from_pretrained(
        config.transformer_model,
        subfolder=None,
        quantization_config=bnb_config,
        torch_dtype=torch.bfloat16,
        device_map=str(device),
        attn_implementation="sdpa",
    )

    # ── 3. FIX T5: prepare ANTES de get_peft_model ──────────────────────────
    # prepare_model_for_kbit_training congela los pesos base y castea LayerNorms
    # a float32. Si se hace después de get_peft_model puede congelar los adaptadores.
    transformer = prepare_model_for_kbit_training(
        transformer, use_gradient_checkpointing=False
    )

    # ── 4. FIX T6: LoRA en AMBOS streams de atención ────────────────────────
    # La versión anterior solo cubría el stream de imagen (to_q/k/v/out).
    # QwenDoubleStreamAttnProcessor usa también add_q/k/v_proj y to_add_out
    # para el stream de texto. Sin ellos el balance aprendido se rompe.
    lora_config = LoraConfig(
        r=config.lora_rank,
        lora_alpha=config.lora_alpha,
        init_lora_weights=True,    # zero-init: evita NaN al inicio
        target_modules=[
            # Stream imagen
            "to_q", "to_k", "to_v", "to_out.0",
            # FIX T6: stream texto (joint attention)
            "add_q_proj", "add_k_proj", "add_v_proj", "to_add_out",
        ],
        lora_dropout=config.lora_dropout,
    )
    transformer = get_peft_model(transformer, lora_config)

    if rank == 0:
        transformer.print_trainable_parameters()

    # ── 5. Model split ───────────────────────────────────────────────────────
    logger.info(f"Rank {rank}: dividiendo modelo...")
    model_split = QwenSplitWrapper(transformer, rank, world_size)

    # ── 6. FIX T7: optimizer DESPUÉS de toda la configuración QLoRA ──────────
    # Solo sobre parámetros entrenables (requires_grad=True), es decir los adaptadores LoRA.
    trainable_params = [p for p in model_split.parameters() if p.requires_grad]
    logger.info(f"Rank {rank}: {len(trainable_params)} grupos de parámetros entrenables")
    optimizer = torch.optim.AdamW(trainable_params, lr=config.lr, eps=1e-6)

    # ── 7. Scheduler de difusión ──────────────────────────────────────────────
    diff_scheduler = FlowMatchEulerDiscreteScheduler.from_pretrained(
        config.base_model, subfolder="scheduler"
    )

    # ── 8. Dataloader (Adelantado para obtener el batch de muestra) ───────────
    g = torch.Generator()
    g.manual_seed(42)

    dataset = LatentsDataset(config.latents_dir)
    dataloader = DataLoader(
        dataset,
        batch_size=config.batch_size,
        collate_fn=make_collate_latents(dataset.global_max_seq_len),
        drop_last=True,
        shuffle=True,
        generator=g,
        num_workers=4,
    )

    # ── 9. VAE y HRNet solo en rank 1 ────────────────────────────────────────
    vae = None
    hrnet = None  # NUEVO H6
    
    if rank == 1:
        logger.info(f"Rank {rank}: cargando VAE en {device}...")
        vae = AutoencoderKLQwenImage.from_pretrained(
            config.base_model, subfolder="vae", torch_dtype=torch.float32
        ).to(device)
        vae.requires_grad_(False)
        vae.eval()
        
        # NUEVO H6: Cargar HRNet en Rank 1 (mismo que VAE)
        if os.path.exists(config.hrnet_model_path):
            logger.info(f"Rank {rank}: cargando HRNet desde {config.hrnet_model_path}...")
            hrnet = load_hrnet_model(
                config.hrnet_model_path,
                width=48,
                num_joints=17,
                device=device
            )
            hrnet.requires_grad_(False)  # Congelar HRNet
            hrnet.eval()
            logger.info(f"Rank {rank}: HRNet cargado y congelado correctamente")
        else:
            logger.warning(
                f"Rank {rank}: HRNet no encontrado en {config.hrnet_model_path}, "
                f"solo se usará velocity loss"
            )

    # ── 10. Generar input_args (DRY-RUN) para PipelineStage ──────────────────
    logger.info(f"Rank {rank}: Generando dummy input_args para PipelineStage...")
    
    dummy_batch = next(iter(dataloader))
    # FIX: PipelineStage espera que el dummy input tenga el tamaño
    # de un solo micro-batch, no del batch global entero.
    mb_size = config.batch_size // config.microbatches
    
    d_target = dummy_batch["target_latents_packed"][:mb_size].to(device, dtype=dtype)
    d_prompt = dummy_batch["prompt_embeds"][:mb_size].to(device, dtype=dtype)
    d_mask   = dummy_batch["prompt_embeds_mask"][:mb_size].to(device)
    d_ts     = torch.zeros((mb_size,), device=device, dtype=dtype)

    if rank == 0:
        d_source = dummy_batch["source_latents_packed"][:mb_size].to(device, dtype=dtype)
        d_noisy  = torch.randn_like(d_target)
        d_latent = torch.cat([d_noisy, d_source], dim=1)
        input_args = (d_latent, d_prompt, d_mask, d_ts)
        
        # Simular salida del rank 0 para decirle las formas al rank 1
        with torch.no_grad():
            out_0 = model_split(*input_args)
        meta = [{"shape": list(t.shape), "dtype": str(t.dtype).split('.')[-1]} for t in out_0]
        dist.broadcast_object_list([meta], src=0)
    else:
        # Rank 1 recibe formas y crea tensores falsos
        container = [None]
        dist.broadcast_object_list(container, src=0)
        meta = container[0]
        
        input_args_list = []
        for info in meta:
            dtype_t = getattr(torch, info["dtype"])
            input_args_list.append(torch.zeros(info["shape"], dtype=dtype_t, device=device))
        input_args = tuple(input_args_list)

    # ── 11. Pipeline stage ────────────────────────────────────────────────────
    logger.info(f"Rank {rank}: init PipelineStage...")
    stage = PipelineStage(
        model_split,
        stage_index=rank,
        num_stages=world_size,
        device=device,
        input_args=input_args,  # <--- FIX: PARÁMETRO INYECTADO AQUÍ
        group=pp_group,
    )

    # ── 12. NUEVA Loss combinada (REEMPLAZA VelocityLossFn) ──────────────────
    # CAMBIO H4: Ahora usamos CombinedLossFn en lugar de VelocityLossFn
    # Inferir resolución desde el primer archivo del dataset
    first_sample = torch.load(dataset.files[0], weights_only=True)
    img_resolution = first_sample.get("resolution", 1024)
    logger.info(f"Resolución inferida del dataset: {img_resolution}")
    
    loss_fn_train = CombinedLossFn(
        vae=vae,
        hrnet=hrnet,
        hrnet_input_size=config.hrnet_input_size,
        img_height=img_resolution,
        img_width=img_resolution,
        velocity_weight=config.velocity_loss_weight,
        heatmap_weight=config.heatmap_loss_weight,
        heatmap_loss_type=config.heatmap_loss_type,
        save_dir=os.path.join(config.output_dir, "loss_diagnostics") if rank == 1 else None
    ) if rank == 1 else (lambda x, y: torch.tensor(0.0, device=device, requires_grad=True))

    schedule = Schedule1F1B(stage, n_microbatches=config.microbatches, loss_fn=loss_fn_train)

    logger.info(f"Rank {rank}: listo para entrenar.")
    model_split.train()
    
    best_loss = float('inf')  # Preparacion para guardar el mejor modelo

    # ── NUEVO: Inicializar CSV en Rank 0 ─────────────────────────────────────
    csv_file_path = os.path.join(config.output_dir, "training_metrics.csv")
    if rank == 0:
        with open(csv_file_path, mode="w", newline="") as f:
            writer = csv.writer(f)
            # Dejamos la columna val_loss preparada para el futuro
            writer.writerow(["epoch", "train_loss", "val_loss"])

    
    for epoch in range(config.epochs):
        if rank == 0:
            logger.info(f"Epoch {epoch} start")

        iterator = tqdm(dataloader, desc=f"Epoch {epoch}") if rank == 0 else dataloader
        avg_loss = 0.0
        steps    = 0

        for step, batch in enumerate(iterator):
            optimizer.zero_grad()

            target = batch["target_latents_packed"].to(device, dtype=dtype)  # (B, Nv, 64)
            prompt = batch["prompt_embeds"].to(device, dtype=dtype)           # (B, S, 3584)
            mask   = batch["prompt_embeds_mask"].to(device)                   # (B, S)
            
            # NUEVO H7: Extraer target_heatmaps del batch
            target_heatmaps = batch["target_heatmaps"]  # (B, 17, 72, 96) en CPU inicialmente

            bsz = target.shape[0]

            # Ruido y timesteps reproducibles por step
            g_seed = 42 + epoch * 10000 + step
            gen    = torch.Generator(device=device).manual_seed(g_seed)

            timesteps = torch.randint(
                0, diff_scheduler.config.num_train_timesteps,
                (bsz,), generator=gen, device=device
            ).long()

            noise = torch.randn(target.shape, generator=gen, device=device, dtype=dtype)

            # Flow matching forward process: x_t = (1-t)*x0 + t*noise
            t_norm   = (timesteps.float() / diff_scheduler.config.num_train_timesteps).to(dtype)
            t_norm   = t_norm.view(-1, 1, 1)
            noisy    = (1.0 - t_norm) * target + t_norm * noise

            # FIX T2: velocity_target se usa ahora en la loss
            velocity_target = noise - target  # (B, Nv, 64)

            # FIX T8: normalizar timestep a [0,1] antes del forward del transformer
            timestep_norm = (timesteps.float() / 1000.0).to(dtype)

            if rank == 0:
                source_r0 = batch["source_latents_packed"].to(device, dtype=dtype)
                latent_model_input = torch.cat([noisy, source_r0], dim=1)
                inputs = (latent_model_input, prompt, mask, timestep_norm)
                schedule.step(*inputs)
                torch.nn.utils.clip_grad_norm_(
                    [p for p in model_split.parameters() if p.requires_grad],
                    max_norm=1.0
                )
                optimizer.step()

            elif rank == 1:
                # NUEVO H7: Pasar contexto del batch a la loss function
                # Esto permite que la loss acceda a target_heatmaps y timesteps
                if isinstance(loss_fn_train, CombinedLossFn):
                    loss_fn_train.set_batch_context(
                        target_heatmaps=target_heatmaps,
                        timesteps=timesteps
                    )
                
                # FIX T1+T2: combined_target contiene v_target en canal 0
                # y noisy en canal 1 (para reconstruir x0_pred).
                combined_target = torch.stack([velocity_target, noisy], dim=1)
                losses = []
                schedule.step(target=combined_target, losses=losses)
                torch.nn.utils.clip_grad_norm_(
                    [p for p in model_split.parameters() if p.requires_grad],
                    max_norm=1.0
                )
                optimizer.step()

                if len(losses) > 0:
                    step_loss = torch.mean(torch.stack(losses)).item()
                    avg_loss += step_loss
                    steps    += 1

        # ── Calcular y Sincronizar Loss ──────────────────────────────────────
        # Solo Rank 1 tiene la pérdida real calculada. Preparamos el valor.
        epoch_loss_val = (avg_loss / steps) if (rank == 1 and steps > 0) else 0.0
        loss_tensor = torch.tensor(epoch_loss_val, device=device, dtype=torch.float32)
        
        # Transmitimos (broadcast) la pérdida desde el Rank 1 al Rank 0
        dist.broadcast(loss_tensor, src=1)
        global_avg_loss = loss_tensor.item()

        if rank == 0:
            logger.info(f"Epoch {epoch} | Loss: {global_avg_loss:.6f} | Mejor histórica: {best_loss:.6f}")
            # ── NUEVO: Escribir métricas de la época en el CSV ───────────────
            with open(csv_file_path, mode="a", newline="") as f:
                writer = csv.writer(f)
                # Si añades validación en el futuro, cambia el "" por la variable de val_loss
                writer.writerow([epoch, global_avg_loss, ""])

                
        # ── Checkpoint Condicional (Solo si mejora) ──────────────────────────
        if global_avg_loss < best_loss:
            best_loss = global_avg_loss
            
            if rank == 0:
                logger.info(f"🟢 ¡Nueva mejor loss! Procediendo a guardar...")

            local_state = model_split.state_dict()
            local_lora  = {k: v.cpu() for k, v in local_state.items() if "lora" in k}

            gathered = [None] * world_size if rank == 0 else None
            dist.gather_object(local_lora, gathered if rank == 0 else None, dst=0)

            if rank == 0:
                merged = {}
                for d in gathered:
                    merged.update(d)
                
                # Usamos un nombre fijo para sobreescribir siempre el anterior
                # y no llenar el disco de checkpoints intermedios
                save_path = os.path.join(config.output_dir, "qwen_lora_best.pt")
                torch.save(merged, save_path)
                logger.info(f"LoRA guardado en: {save_path}")
        else:
            if rank == 0:
                logger.info(f"⚪ La loss no mejoró. Saltando guardado de checkpoint.")

        dist.barrier()

        # ── Callback de inferencia ────────────────────────────────────────────
        if config.inference_every > 0 and (epoch + 1) % config.inference_every == 0:
            run_inference_callback(
                model_split=model_split,
                vae=vae,
                base_scheduler=diff_scheduler,
                device=device,
                rank=rank,
                dataset=dataset,
                save_dir=config.output_dir,
                epoch=epoch,
                img_height=img_resolution,
                img_width=img_resolution,
                num_steps=config.inference_steps,
                num_samples=config.inference_samples,
                dtype=dtype,
            )
            dist.barrier()

    dist.destroy_process_group()


if __name__ == "__main__":
    main()