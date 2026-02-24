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
from torch.utils.data import DataLoader
from tqdm import tqdm
from accelerate import Accelerator
from accelerate.logging import get_logger
from accelerate.utils import set_seed
from dataclasses import dataclass



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
            target_images_vae = target_images.to(device, dtype=pipeline.vae.dtype)
            
            # --- 1. Target Image Latents ---
            # Encode target image to latents
            # Qwen uses video-like 5D tensor (B, C, F, H, W) for input if possible, or handles 4D
            # Pipeline expects (B, C, H, W) usually but `_encode_vae_image` might expect frame dim
            # Check pipeline implementation or usage in original script:
            # target_images_vae.unsqueeze(2) in original script suggests adding frame dim.
            
            target_latents = pipeline._encode_vae_image(
                target_images_vae.unsqueeze(2), 
                generator=None
            )
            
            # Pack latents
            # (Batch, Channel, Frame, Height, Width) -> Packed formatting
            batch_size_curr, num_channels, _, height, width = target_latents.shape
            target_latents_packed = pipeline._pack_latents(
                target_latents, batch_size_curr, num_channels, height, width
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
    pipeline.vae.to(accelerator.device)
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
