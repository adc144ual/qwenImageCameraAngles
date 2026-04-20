"""
Script para pre-computar latentes y embeddings para Qwen-Image-Edit
Permite ahorrar VRAM durante el entrenamiento al no tener que cargar VAE y TextEncoder.
"""

import os
import sys

# --- CONFIGURACIÓN DE ENTORNO ---
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["HF_HOME"] = "/nas/antoniodetoro/qwen/hf_cache"
os.environ["TMPDIR"] = "/nas/antoniodetoro/qwen/tmp"
os.environ["PYTHONNOUSERSITE"] = "1"


import argparse
import torch
import torch.nn.functional as F
import numpy as np
from torchvision import transforms
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
from src.train_multiview_finetuning import MultiViewDataset, collate_fn

logger = get_logger(__name__)

@dataclass
class PrecomputeConfig:
    """Configuración de pre-computación"""
    dataset_root: str
    output_dir: str
    base_model: str = "Qwen/Qwen-Image-Edit-2509"
    resolution: int = 512
    batch_size: int = 1
    max_samples: int = 10 
    mixed_precision: str = "bf16"
    seed: int = 42
    dataloader_num_workers: int = 0



class MultiViewDataset(Dataset):
    """
    Dataset para imágenes multi-vista sincronizadas por timestamp.
    Genera pares de (imagen_origen, imagen_destino, prompt_rotación)
    """
    
    CAMERA_ANGLES = {
        "00_17": 0,     # Vista frontal
        "00_16": 90,    # Vista derecha (+90°)
        "00_15": -90,   # Vista izquierda (-90°)
    }
    
    ANGLE_PROMPTS = {
        0: {
            90: "将镜头向右旋转90度 Rotate the camera 90 degrees to the right.",
            180: "将镜头旋转180度 Rotate the camera 180 degrees.",
            -90: "将镜头向左旋转90度 Rotate the camera 90 degrees to the left.",
        },
        90: {
            90: "将镜头向右旋转90度 Rotate the camera 90 degrees to the right.", # This would be 180
            -90: "将镜头向左旋转90度 Rotate the camera 90 degrees to the left.", # Back to 0
            -180: "将镜头旋转180度 Rotate the camera 180 degrees.", # To -90
        },
        -90: {
            90: "将镜头向右旋转90度 Rotate the camera 90 degrees to the right.", # Back to 0
            -90: "将镜头向左旋转90度 Rotate the camera 90 degrees to the left.", # To 180 (not present in dataset but consistent relative angle)
            180: "将镜头旋转180度 Rotate the camera 180 degrees.", # To 90
        }
    }
    
    def __init__(
        self,
        dataset_root: str,
        resolution: int = 512,
        split: str = "train",
        train_ratio: float = 0.9
    ):
        self.dataset_root = Path(dataset_root)
        self.resolution = resolution
        self.split = split
        
        self.transform = transforms.Compose([
            transforms.Resize((resolution, resolution), interpolation=transforms.InterpolationMode.BILINEAR),
            transforms.ToTensor(),
            # Convierte de [0, 1] a [-1, 1]
            transforms.Normalize([0.5], [0.5]),
        ])
        
        # Escanear dataset y crear pares de entrenamiento
        self.samples = self._build_sample_pairs()
        
        # Split train/validation
        np.random.seed(42)
        indices = np.random.permutation(len(self.samples))
        split_idx = int(len(indices) * train_ratio)
        
        if split == "train":
            self.samples = [self.samples[i] for i in indices[:split_idx]]
        else:
            self.samples = [self.samples[i] for i in indices[split_idx:]]
        
        logger.info(f"Loaded {len(self.samples)} samples for {split} split")
    
    def _build_sample_pairs(self) -> List[Dict]:
        """
        Construye pares de imágenes sincronizadas por timestamp.
        Retorna lista de dicts con: src_path, tgt_path, src_angle, tgt_angle, prompt
        """
        samples = []
        
        # Estructura: {timestamp: {camera_id: image_path}}
        timestamp_data = {}
        
        # Determinar directorio base según split (usamos train_val para train/val y test para test si quisiéramos)
        # Por ahora asumimos que todo está en train_val y el split se hace por ratio
        base_dir = self.dataset_root / "train_val"
        if not base_dir.exists():
            # Fallback si no existe la carpeta train_val, intentar usar root directo
            logger.warning(f"No se encontró carpeta train_val en {self.dataset_root}, buscando en root")
            base_dir = self.dataset_root

        for camera_dir in sorted(base_dir.glob("*")):
            if not camera_dir.is_dir():
                continue
            
            camera_id = camera_dir.name  # "00_17", "00_16", "00_15"
            if camera_id not in self.CAMERA_ANGLES:
                continue
            
            # Obtener todas las imágenes RGB de esta cámara
            images = sorted(camera_dir.glob("*_rgb.png"))
            if not images:
                # Intentar también sin extensión .png por si acaso (el glob original tenía .png)
                images = sorted(camera_dir.glob("*_rgb*"))
                
            for img_path in images:
                # Extraer timestamp del nombre de archivo: CAM_TIMESTAMP_rgb...
                # Ejemplo: 00_15_1680174540683_rgb -> timestamp: 1680174540683
                parts = img_path.stem.split("_")
                # Buscamos el timestamp. Asumiendo formato timestamp es numérico largo.
                # Formatos vistos: 00_15_TIMESTAMP_rgb o XX_YY_TIMESTAMP_rgb
                # Partes: 0:"00", 1:"15", 2:"TIMESTAMP", 3:"rgb"
                
                if len(parts) >= 3:
                     # El timestamp suele estar en la posición 2 (índice base 0) para archivos como 00_15_1680174540683_rgb
                    timestamp = parts[2]
                    
                    if timestamp not in timestamp_data:
                        timestamp_data[timestamp] = {}
                    
                    timestamp_data[timestamp][camera_id] = img_path

        # Crear pares de entrenamiento
        for timestamp, cameras in timestamp_data.items():
            # Emparejar todas las combinaciones posibles para este timestamp
            for src_cam, src_angle in self.CAMERA_ANGLES.items():
                if src_cam not in cameras:
                    continue
                
                src_img = cameras[src_cam]
                
                for tgt_cam, tgt_angle in self.CAMERA_ANGLES.items():
                    if src_cam == tgt_cam:
                        continue
                    
                    if tgt_cam not in cameras:
                        continue
                    
                    tgt_img = cameras[tgt_cam]
                    
                    # Calcular ángulo relativo
                    angle_diff = (tgt_angle - src_angle) % 360
                    if angle_diff > 180:
                        angle_diff -= 360
                    
                    # Obtener prompt
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
                        "timestamp": timestamp
                    })
        
        return samples
    
    def __len__(self) -> int:
        return len(self.samples)
    
    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        sample = self.samples[idx]
        
        # Cargar imágenes
        src_img = Image.open(sample["src_path"]).convert("RGB")
        tgt_img = Image.open(sample["tgt_path"]).convert("RGB")
        
        # Aplicar transformaciones
        src_tensor = self.transform(src_img)
        tgt_tensor = self.transform(tgt_img)
        
        return {
            "source_image": src_tensor,
            "target_image": tgt_tensor,
            "prompt": sample["prompt"],
            "angle_diff": sample["angle_diff"],
        }


def collate_fn(examples):
    """Collate function para el DataLoader"""
    source_images = torch.stack([example["source_image"] for example in examples])
    target_images = torch.stack([example["target_image"] for example in examples])
    prompts = [example["prompt"] for example in examples]
    angle_diffs = torch.tensor([example["angle_diff"] for example in examples])
    
    return {
        "source_images": source_images,
        "target_images": target_images,
        "prompts": prompts,
        "angle_diffs": angle_diffs,
    }



def precompute_data(
    pipeline: QwenImageEditPlusPipeline,
    dataloader: DataLoader,
    accelerator: Accelerator,
    config: PrecomputeConfig,
    split: str = "train"
):
    """Pre-computa latentes y embeddings para un split"""
    split_dir = os.path.join(config.output_dir, split)
    os.makedirs(split_dir, exist_ok=True)
    
    # Asegurar modo evaluación
    pipeline.vae.eval()
    pipeline.text_encoder.eval()
    
    logger.info(f"Processing {split} split...")
    
    count = 0
    # Calcular total para la barra de progreso
    total_samples = min(len(dataloader), config.max_samples)
    progress_bar = tqdm(total=total_samples, disable=not accelerator.is_local_main_process)
    
    for i, batch in enumerate(dataloader):
        if count >= config.max_samples:
            break
            
        # Check if already processed (simple resume)
        save_path = os.path.join(split_dir, f"batch_{i:05d}.pt")
        if os.path.exists(save_path):
            count += 1
            progress_bar.update(1)
            continue

        with torch.no_grad():
            source_images = batch["source_images"]
            target_images = batch["target_images"]
            prompts = batch["prompts"]
            angle_diffs = batch["angle_diffs"]
            
            # Mover a device
            device = accelerator.device
            vae_dtype = next(pipeline.vae.parameters()).dtype
            target_images_vae = target_images.to(device, dtype=vae_dtype)
            
            # --- 1. Target Image Latents ---
            # Encode target image to latents
            # Qwen uses video-like 5D tensor (B, C, F, H, W) for input if possible, or handles 4D
            # Pipeline expects (B, C, H, W) usually but `_encode_vae_image` might expect frame dim
            # Check pipeline implementation or usage in original script:
            # target_images_vae.unsqueeze(2) in original script suggests adding frame dim.
            
            raw_latents = pipeline._encode_vae_image(
                target_images_vae.unsqueeze(2), 
                generator=None
            )
            
            # # Obtener estadísticas de la config del VAE
            # # Qwen usa estas estadísticas para llevar los latentes a distribución normal (Mean 0, Std 1)
            # _lm = torch.tensor(pipeline.vae.config.latents_mean)
            # _ls = torch.tensor(pipeline.vae.config.latents_std)
            # latent_channels = _lm.numel()
            # latents_mean = _lm.view(1, latent_channels, 1, 1, 1).to(device, dtype=vae_dtype)
            # latents_std = _ls.view(1, latent_channels, 1, 1, 1).to(device, dtype=vae_dtype)
            
            # # Aplicar la fórmula: (Raw - Mean) / Std
            # # Nota: En el train script se usaba inversa, aquí aplicamos la directa.
            # target_latents_norm = (raw_latents - latents_mean) / latents_std
            
            # Volver al tipo de dato de almacenamiento si quieres ahorrar espacio (opcional, pero seguro tras normalizar)
            target_latents_norm = raw_latents.to(dtype=pipeline.text_encoder.dtype)





            # Pack latents
            # (Batch, Channel, Frame, Height, Width) -> Packed formatting
            batch_size_curr, num_channels, _, height, width = target_latents_norm.shape
            target_latents_packed = pipeline._pack_latents(
                target_latents_norm, batch_size_curr, num_channels, height, width
            )

            # --- 1b. Source Image Latents (para visualización en entrenamiento) ---
            source_images_vae = source_images.to(device, dtype=vae_dtype)
            raw_source_latents = pipeline._encode_vae_image(
                source_images_vae.unsqueeze(2),
                generator=None
            )
            source_latents_norm = raw_source_latents.to(dtype=pipeline.text_encoder.dtype)
            bs_src, nc_src, _, h_src, w_src = source_latents_norm.shape
            source_latents_packed = pipeline._pack_latents(
                source_latents_norm, bs_src, nc_src, h_src, w_src
            )

            # --- 2. Source Image Conditioning + Text Embeddings ---
            source_images_encoder = source_images.to(device, dtype=pipeline.text_encoder.dtype)
            source_resized = F.interpolate(
                source_images_encoder,
                size=(384, 384),
                mode='bilinear',
                align_corners=False
            )
            
            # Get text embeddings
            # encode_prompt uses text_encoder inside
            prompt_embeds, prompt_embeds_mask = pipeline.encode_prompt(
                prompt=prompts,
                image=[source_resized[j] for j in range(source_resized.shape[0])],
                device=device,
                num_images_per_prompt=1,
            )
            
            # Guardar en disco (Mover a CPU)
            data_to_save = {
                "target_latents_packed": target_latents_packed.cpu(),
                "source_latents_packed": source_latents_packed.cpu(),
                "prompt_embeds": prompt_embeds.cpu(),
                "prompt_embeds_mask": prompt_embeds_mask.cpu(),
                "angle_diffs": angle_diffs.cpu(), 
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
    parser.add_argument("--output_dir", type=str, default="../models/precomputed_latents")
    parser.add_argument("--batch_size", type=int, default=1)
    parser.add_argument("--max_samples", type=int, default=10)
    
    args = parser.parse_args()
    
    config = PrecomputeConfig(
        dataset_root=args.dataset_root,
        output_dir=args.output_dir,
        batch_size=args.batch_size,
        max_samples=args.max_samples
    )
    
    # Initialize accelerator
    accelerator = Accelerator(mixed_precision=config.mixed_precision)
    set_seed(config.seed)
    
    # Load model parts
    logger.info("Loading pipeline model...")
    dtype = torch.bfloat16 if config.mixed_precision == "bf16" else torch.float16
    
    # Load pipeline
    pipeline = QwenImageEditPlusPipeline.from_pretrained(
        config.base_model,
        torch_dtype=dtype,
    )
    
    # Delete transformer
    logger.info("Deleting transformer to save memory...")
    del pipeline.transformer
    pipeline.transformer = None
    torch.cuda.empty_cache()
    
    # Move remaining models to device
    pipeline.vae.to(accelerator.device, dtype=torch.float32)  # VAE puede requerir precisión completa para estabilidad
    pipeline.text_encoder.to(accelerator.device)
    
    # Datasets
    logger.info("Loading datasets...")
    train_dataset = MultiViewDataset(
        dataset_root=config.dataset_root,
        resolution=config.resolution,
        split="train",
    )
    
    train_dataloader = DataLoader(
        train_dataset,
        batch_size=config.batch_size,
        shuffle=False, # Deterministic order for file naming
        num_workers=config.dataloader_num_workers,
        collate_fn=collate_fn,
    )
    
    precompute_data(pipeline, train_dataloader, accelerator, config, split="train")

    # Validation set
    val_dataset = MultiViewDataset(
        dataset_root=config.dataset_root,
        resolution=config.resolution,
        split="val"
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
