"""
Script para pre-computar latentes y embeddings para Qwen-Image-Edit.
Permite ahorrar VRAM durante el entrenamiento al no cargar VAE ni TextEncoder.

Fixes aplicados respecto a la versión anterior:
  P1+P2 — eliminada doble normalización; _encode_vae_image ya normaliza internamente.
           Corregido view(1,4,...) → z_dim del VAE.
  P3     — squeeze(2) antes de _pack_latents para eliminar dim de frame.
  P4     — source images convertidas a PIL [0,255] antes de encode_prompt.
  P5     — source_latents_packed añadida al dict guardado.
  P6     — resolución por defecto 1024 para alinearse con el modelo base.
"""

import os
import sys

os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["HF_HOME"] = "/nas/antoniodetoro/qwen/hf_cache"
os.environ["TMPDIR"] = "/nas/antoniodetoro/qwen/tmp"
os.environ["PYTHONNOUSERSITE"] = "1"

import argparse
import torch
import torch.nn.functional as F
import numpy as np
import torchvision.transforms as T
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm
from accelerate import Accelerator
from accelerate.logging import get_logger
from accelerate.utils import set_seed
from dataclasses import dataclass
from PIL import Image
from pathlib import Path
from typing import Dict, List

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from qwenimage.pipeline_qwenimage_edit_plus import QwenImageEditPlusPipeline

logger = get_logger(__name__)


@dataclass
class PrecomputeConfig:
    dataset_root: str
    output_dir: str
    base_model: str = "Qwen/Qwen-Image-Edit-2509"
    # FIX P6: resolución 1024 alineada con el modelo base.
    # A 512 los RoPE y el VAE están fuera de distribución de preentrenamiento.
    resolution: int = 1024
    condition_resolution: int = 384   # tamaño que usa el text encoder para la imagen
    batch_size: int = 1
    max_samples: int = 10
    mixed_precision: str = "bf16"
    seed: int = 42
    dataloader_num_workers: int = 0


class MultiViewDataset(Dataset):
    """
    Dataset para imágenes multi-vista sincronizadas por timestamp.
    Genera pares de (imagen_origen, imagen_destino, prompt_rotación).
    Las imágenes se devuelven como PIL para que el caller decida el preprocesado.
    """

    CAMERA_ANGLES = {
        "00_17": 0,
        "00_16": 90,
        "00_15": -90,
    }

    ANGLE_PROMPTS = {
        0: {
            90:   "将镜头向右旋转90度 Rotate the camera 90 degrees to the right.",
            180:  "将镜头旋转180度 Rotate the camera 180 degrees.",
            -90:  "将镜头向左旋转90度 Rotate the camera 90 degrees to the left.",
        },
        90: {
            90:   "将镜头向右旋转90度 Rotate the camera 90 degrees to the right.",
            -90:  "将镜头向左旋转90度 Rotate the camera 90 degrees to the left.",
            -180: "将镜头旋转180度 Rotate the camera 180 degrees.",
        },
        -90: {
            90:   "将镜头向右旋转90度 Rotate the camera 90 degrees to the right.",
            -90:  "将镜头向左旋转90度 Rotate the camera 90 degrees to the left.",
            180:  "将镜头旋转180度 Rotate the camera 180 degrees.",
        },
    }

    def __init__(
        self,
        dataset_root: str,
        resolution: int = 1024,
        split: str = "train",
        train_ratio: float = 0.9,
    ):
        self.dataset_root = Path(dataset_root)
        self.resolution = resolution
        self.split = split

        # Las imágenes se cargan como PIL; el preprocesado ocurre en precompute_data.
        self.samples = self._build_sample_pairs()

        np.random.seed(42)
        indices = np.random.permutation(len(self.samples))
        split_idx = int(len(indices) * train_ratio)

        if split == "train":
            self.samples = [self.samples[i] for i in indices[:split_idx]]
        else:
            self.samples = [self.samples[i] for i in indices[split_idx:]]

        logger.info(f"Loaded {len(self.samples)} samples for {split} split")

    def _build_sample_pairs(self) -> List[Dict]:
        samples = []
        timestamp_data = {}

        base_dir = self.dataset_root / "train_val"
        if not base_dir.exists():
            logger.warning(f"No train_val folder at {self.dataset_root}, using root")
            base_dir = self.dataset_root

        for camera_dir in sorted(base_dir.glob("*")):
            if not camera_dir.is_dir():
                continue
            camera_id = camera_dir.name
            if camera_id not in self.CAMERA_ANGLES:
                continue
            images = sorted(camera_dir.glob("*_rgb.png"))
            if not images:
                images = sorted(camera_dir.glob("*_rgb*"))
            for img_path in images:
                parts = img_path.stem.split("_")
                if len(parts) >= 3:
                    timestamp = parts[2]
                    if timestamp not in timestamp_data:
                        timestamp_data[timestamp] = {}
                    timestamp_data[timestamp][camera_id] = img_path

        for timestamp, cameras in timestamp_data.items():
            for src_cam, src_angle in self.CAMERA_ANGLES.items():
                if src_cam not in cameras:
                    continue
                src_img = cameras[src_cam]
                for tgt_cam, tgt_angle in self.CAMERA_ANGLES.items():
                    if src_cam == tgt_cam or tgt_cam not in cameras:
                        continue
                    tgt_img = cameras[tgt_cam]
                    angle_diff = (tgt_angle - src_angle) % 360
                    if angle_diff > 180:
                        angle_diff -= 360
                    if src_angle in self.ANGLE_PROMPTS and angle_diff in self.ANGLE_PROMPTS[src_angle]:
                        prompt = self.ANGLE_PROMPTS[src_angle][angle_diff]
                    else:
                        prompt = f"将镜头旋转{angle_diff}度 Rotate the camera {angle_diff} degrees."
                    samples.append({
                        "src_path": str(src_img),
                        "tgt_path": str(tgt_img),
                        "src_angle": src_angle,
                        "tgt_angle": tgt_angle,
                        "angle_diff": angle_diff,
                        "prompt": prompt,
                        "timestamp": timestamp,
                    })
        return samples

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> Dict:
        sample = self.samples[idx]
        # Devolvemos PIL directamente; el preprocesado correcto ocurre en precompute_data.
        src_pil = Image.open(sample["src_path"]).convert("RGB")
        tgt_pil = Image.open(sample["tgt_path"]).convert("RGB")
        return {
            "source_pil": src_pil,
            "target_pil": tgt_pil,
            "prompt": sample["prompt"],
            "angle_diff": sample["angle_diff"],
        }


def collate_fn(examples):
    """Collate function: imágenes PIL se pasan como listas (no se pueden stack)."""
    return {
        "source_pils": [ex["source_pil"] for ex in examples],
        "target_pils": [ex["target_pil"] for ex in examples],
        "prompts":     [ex["prompt"] for ex in examples],
        "angle_diffs": torch.tensor([ex["angle_diff"] for ex in examples]),
    }


def pil_to_vae_tensor(pil_img: Image.Image, resolution: int, device, dtype) -> torch.Tensor:
    """
    Convierte una imagen PIL a tensor listo para el VAE.
    Salida: (1, C, 1, H, W) en rango [-1, 1], con la dimensión de frame requerida por
    AutoencoderKLQwenImage (que opera sobre video 5D).
    """
    transform = T.Compose([
        T.Resize((resolution, resolution), interpolation=T.InterpolationMode.BILINEAR),
        T.ToTensor(),                     # [0,1]
        T.Normalize([0.5], [0.5]),        # [-1,1]
    ])
    tensor = transform(pil_img).unsqueeze(0)          # (1, C, H, W)
    tensor = tensor.unsqueeze(2)                       # (1, C, 1, H, W)  ← dim de frame
    return tensor.to(device=device, dtype=dtype)


def pil_to_condition_pil(pil_img: Image.Image, condition_resolution: int) -> Image.Image:
    """
    Redimensiona a la resolución que usa el text encoder (384x384 por defecto).
    Se devuelve PIL porque Qwen2VLProcessor espera PIL/array en [0,255], NO tensores.
    FIX P4: la versión anterior pasaba tensores normalizados [-1,1] al processor,
    lo que producía embeddings completamente incorrectos (imagen casi negra).
    """
    return pil_img.resize((condition_resolution, condition_resolution), Image.BILINEAR)


@torch.no_grad()
def precompute_data(
    pipeline: QwenImageEditPlusPipeline,
    dataloader: DataLoader,
    accelerator: Accelerator,
    config: PrecomputeConfig,
    split: str = "train",
):
    """Pre-computa y guarda en disco latentes + embeddings para un split."""
    split_dir = os.path.join(config.output_dir, split)
    os.makedirs(split_dir, exist_ok=True)

    pipeline.vae.eval()
    pipeline.text_encoder.eval()

    device = accelerator.device
    # Dtype de almacenamiento: bf16 ahorra espacio sin perder precisión relevante.
    store_dtype = torch.bfloat16 if config.mixed_precision == "bf16" else torch.float16

    # El dtype del VAE debe coincidir con el del tensor de entrada.
    # El pipeline se cargó con torch_dtype=bfloat16, así que el VAE está en bfloat16.
    # Pasar float32 al VAE causaba: "Input type (float) and bias type (BFloat16) should be the same"
    vae_dtype = next(pipeline.vae.parameters()).dtype

    logger.info(f"Processing {split} split...")

    count = 0
    total_samples = min(len(dataloader), config.max_samples)
    progress_bar = tqdm(total=total_samples, disable=not accelerator.is_local_main_process)

    for i, batch in enumerate(dataloader):
        if count >= config.max_samples:
            break

        save_path = os.path.join(split_dir, f"batch_{i:05d}.pt")
        if os.path.exists(save_path):
            count += 1
            progress_bar.update(1)
            continue

        source_pils = batch["source_pils"]   # lista de PIL
        target_pils = batch["target_pils"]   # lista de PIL
        prompts     = batch["prompts"]
        angle_diffs = batch["angle_diffs"]

        # ── 1. TARGET LATENTS ────────────────────────────────────────────────
        # FIX P3: pasamos tensor 5D (B,C,1,H,W) y hacemos squeeze(2) antes de
        #         _pack_latents, que espera exactamente (B,C,H,W).
        # FIX P1+P2: _encode_vae_image ya aplica (latents - mean) / std
        #            internamente. NO renormalizamos aquí.
        target_latents_list = []
        for pil in target_pils:
            t_vae = pil_to_vae_tensor(pil, config.resolution, device, dtype=vae_dtype)
            # _encode_vae_image recibe (B, C, 1, H, W) y devuelve (B, z_dim, 1, H/8, W/8)
            # ya normalizado a distribución aproximada N(0,1).
            enc = pipeline._encode_vae_image(t_vae, generator=None)
            target_latents_list.append(enc)

        target_latents_5d = torch.cat(target_latents_list, dim=0)  # (B, z_dim, 1, H/8, W/8)

        # FIX P3: quitar dimensión de frame antes de packear.
        target_latents_4d = target_latents_5d.squeeze(2)            # (B, z_dim, H/8, W/8)
        B, C, H_lat, W_lat = target_latents_4d.shape
        target_latents_packed = pipeline._pack_latents(
            target_latents_4d, B, C, H_lat, W_lat
        )  # (B, (H/16)*(W/16), z_dim*4)

        # ── 2. SOURCE LATENTS (FIX P5: ahora también se guardan) ────────────
        source_latents_list = []
        for pil in source_pils:
            s_vae = pil_to_vae_tensor(pil, config.resolution, device, dtype=vae_dtype)
            enc = pipeline._encode_vae_image(s_vae, generator=None)
            source_latents_list.append(enc)

        source_latents_5d = torch.cat(source_latents_list, dim=0)
        source_latents_4d = source_latents_5d.squeeze(2)
        source_latents_packed = pipeline._pack_latents(
            source_latents_4d, B, C, H_lat, W_lat
        )  # (B, (H/16)*(W/16), z_dim*4)

        # ── 3. TEXT + VISION EMBEDDINGS ──────────────────────────────────────
        # FIX P4: encode_prompt internamente llama a Qwen2VLProcessor, que espera
        #         imágenes PIL en [0,255], NO tensores normalizados.
        #         pil_to_condition_pil devuelve PIL redimensionado, sin normalizar.
        condition_pils = [pil_to_condition_pil(p, config.condition_resolution) for p in source_pils]

        prompt_embeds, prompt_embeds_mask = pipeline.encode_prompt(
            prompt=prompts,
            image=condition_pils,      # PIL list ← correcto
            device=device,
            num_images_per_prompt=1,
        )
        # prompt_embeds:      (B, S_txt, 3584)
        # prompt_embeds_mask: (B, S_txt)

        # ── 4. GUARDAR ────────────────────────────────────────────────────────
        data_to_save = {
            # Latentes del target (imagen a generar), empaquetados y normalizados.
            "target_latents_packed":  target_latents_packed.to(dtype=store_dtype).cpu(),
            # FIX P5: latentes de la imagen fuente (condicionamiento visual del DiT).
            "source_latents_packed":  source_latents_packed.to(dtype=store_dtype).cpu(),
            # Embeddings del text encoder (ya incluyen la imagen de condición 384px).
            "prompt_embeds":          prompt_embeds.to(dtype=store_dtype).cpu(),
            "prompt_embeds_mask":     prompt_embeds_mask.cpu(),
            # Metadata útil para debugging / logging.
            "angle_diffs":            angle_diffs.cpu(),
            # Guardar la resolución usada para que el train script pueda inferir shapes.
            "resolution":             config.resolution,
        }

        torch.save(data_to_save, save_path)

        count += 1
        progress_bar.update(1)

        if i % 50 == 0:
            torch.cuda.empty_cache()

    progress_bar.close()
    logger.info(f"Saved {count} batches to {split_dir}")


def main():
    parser = argparse.ArgumentParser(description="Precompute latents for Qwen-Image-Edit")
    parser.add_argument("--dataset_root", type=str, required=True)
    parser.add_argument("--output_dir",   type=str, default="../models/precomputed_latents_fix_all")
    parser.add_argument("--batch_size",   type=int, default=1)
    parser.add_argument("--max_samples",  type=int, default=10)
    parser.add_argument("--resolution",   type=int, default=1024,
                        help="Resolución VAE. Usar 1024 para alinearse con el modelo base.")
    parser.add_argument("--condition_resolution", type=int, default=384,
                        help="Resolución de la imagen de condición para el text encoder.")
    args = parser.parse_args()

    config = PrecomputeConfig(
        dataset_root=args.dataset_root,
        output_dir=args.output_dir,
        batch_size=args.batch_size,
        max_samples=args.max_samples,
        resolution=args.resolution,
        condition_resolution=args.condition_resolution,
    )

    accelerator = Accelerator(mixed_precision=config.mixed_precision)
    set_seed(config.seed)

    logger.info("Loading pipeline model...")
    dtype = torch.bfloat16 if config.mixed_precision == "bf16" else torch.float16

    pipeline = QwenImageEditPlusPipeline.from_pretrained(
        config.base_model,
        torch_dtype=dtype,
    )

    logger.info("Deleting transformer to save memory...")
    del pipeline.transformer
    pipeline.transformer = None
    torch.cuda.empty_cache()

    pipeline.vae.to(accelerator.device)
    pipeline.text_encoder.to(accelerator.device)

    logger.info("Loading datasets...")
    train_dataset = MultiViewDataset(
        dataset_root=config.dataset_root,
        resolution=config.resolution,
        split="train",
    )
    train_dataloader = DataLoader(
        train_dataset,
        batch_size=config.batch_size,
        shuffle=False,
        num_workers=config.dataloader_num_workers,
        collate_fn=collate_fn,
    )
    precompute_data(pipeline, train_dataloader, accelerator, config, split="train")

    val_dataset = MultiViewDataset(
        dataset_root=config.dataset_root,
        resolution=config.resolution,
        split="val",
    )
    if len(val_dataset) > 0:
        val_dataloader = DataLoader(
            val_dataset,
            batch_size=1,
            shuffle=False,
            num_workers=config.dataloader_num_workers,
            collate_fn=collate_fn,
        )
        precompute_data(pipeline, val_dataloader, accelerator, config, split="val")

    logger.info("Done!")


if __name__ == "__main__":
    main()