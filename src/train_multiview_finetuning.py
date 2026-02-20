"""
Script de Fine-tuning para Qwen-Image-Edit con Dataset Multi-Vista
Entrena el modelo para generar mejores rotaciones de cámara usando LoRA

Dataset Structure Expected:
/dataset_root/
    train_val/
        00_17/  # Cámara frontal (0 grados)
            00_17_timestamp_rgb.png
            ...
        00_16/  # Cámara derecha (+90 grados)
            00_16_timestamp_rgb.png
            ...
        00_15/  # Cámara izquierda (-90 grados)
            00_15_timestamp_rgb.png
            ...
    test/       # (Opcional) Estructura idéntica para validación/test
        00_17/
        00_16/
        00_15/

Notas:
- Las imágenes deben estar sincronizadas por timestamp.
- El script busca coincidencias de timestamp entre las carpetas de las cámaras.
- Se asume:
    - 00_17: Frontal (0°)
    - 00_16: Derecha (+90°)
    - 00_15: Izquierda (-90°)
"""

import os
import sys
import json
import argparse
from pathlib import Path
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass

# --- CONFIGURACIÓN DE ENTORNO ---
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
# os.environ["CUDA_VISIBLE_DEVICES"] = "0"  # Permitir ver todas las GPUs
os.environ["HF_HOME"] = "/nas/antoniodetoro/qwen/hf_cache"
os.environ["TMPDIR"] = "/nas/antoniodetoro/qwen/tmp"
os.environ["PYTHONNOUSERSITE"] = "1"

import torch
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image
import numpy as np
from tqdm import tqdm
from accelerate import Accelerator
from accelerate.logging import get_logger
from accelerate.utils import ProjectConfiguration, set_seed
from diffusers.optimization import get_scheduler
from diffusers import FlowMatchEulerDiscreteScheduler

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from qwenimage.pipeline_qwenimage_edit_plus import QwenImageEditPlusPipeline
from qwenimage.transformer_qwenimage import QwenImageTransformer2DModel

logger = get_logger(__name__)


@dataclass
class TrainingConfig:
    """Configuración de entrenamiento"""
    # Dataset
    dataset_root: str = "../datos/MultiViewVisibleThermalImagesHPE"
    output_dir: str = "../models/finetuned_multiview"
    
    # Modelo
    base_model: str = "Qwen/Qwen-Image-Edit-2509"
    transformer_model: str = "linoyts/Qwen-Image-Edit-Rapid-AIO"
    resolution: int = 512
    
    # Entrenamiento
    train_batch_size: int = 1
    gradient_accumulation_steps: int = 4
    num_train_epochs: int = 10
    learning_rate: float = 1e-4
    lr_scheduler: str = "cosine"
    lr_warmup_steps: int = 500
    max_grad_norm: float = 1.0
    
    # LoRA
    use_lora: bool = True
    lora_rank: int = 16
    lora_alpha: int = 32
    lora_dropout: float = 0.1
    
    # Logging
    logging_steps: int = 50
    save_steps: int = 500
    validation_steps: int = 250
    num_validation_samples: int = 4
    
    # Hardware
    mixed_precision: str = "bf16"  # "no", "fp16", "bf16"
    seed: int = 42
    dataloader_num_workers: int = 4


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


def enable_lora(transformer: QwenImageTransformer2DModel, config: TrainingConfig):
    """Habilita LoRA en las capas de atención del transformer"""
    from peft import LoraConfig, get_peft_model
    
    # Configurar LoRA solo para las capas de atención
    lora_config = LoraConfig(
        r=config.lora_rank,
        lora_alpha=config.lora_alpha,
        init_lora_weights="gaussian",
        target_modules=["to_q", "to_k", "to_v", "to_out.0", "add_q_proj", "add_k_proj", "add_v_proj"],
        lora_dropout=config.lora_dropout,
    )
    
    transformer = get_peft_model(transformer, lora_config)
    transformer.print_trainable_parameters()
    
    return transformer


def compute_loss(
    model_output: torch.Tensor,
    target: torch.Tensor,
    timesteps: torch.Tensor,
    noise_scheduler: FlowMatchEulerDiscreteScheduler,
) -> torch.Tensor:
    """
    Computa la pérdida de flow matching.
    En flow matching, el modelo predice el "flow" (velocidad) del proceso.
    """
    # MSE loss entre la predicción del modelo y el target
    loss = F.mse_loss(model_output.float(), target.float(), reduction="mean")
    return loss


def train_one_epoch(
    pipeline: QwenImageEditPlusPipeline,
    train_dataloader: DataLoader,
    optimizer: torch.optim.Optimizer,
    lr_scheduler,
    accelerator: Accelerator,
    config: TrainingConfig,
    epoch: int,
    global_step: int,
) -> int:
    """Entrena una época"""
    pipeline.transformer.train()
    
    progress_bar = tqdm(
        total=len(train_dataloader),
        disable=not accelerator.is_local_main_process,
        desc=f"Epoch {epoch}"
    )
    
    for step, batch in enumerate(train_dataloader):
        with accelerator.accumulate(pipeline.transformer):
            source_images = batch["source_images"]
            target_images = batch["target_images"]
            prompts = batch["prompts"]
            
            # Mover a device y dtype correcto
            source_images = source_images.to(accelerator.device, dtype=torch.float32)
            target_images = target_images.to(accelerator.device, dtype=torch.float32)
            
            # Preprocesar imágenes (en el dispositivo donde estén los modelos auxiliares)
            with torch.no_grad():
                # Encode target image to latents
                # Mover inputs al dispositivo del VAE
                vae_device = pipeline.vae.device
                target_images_vae = target_images.to(vae_device, dtype=pipeline.vae.dtype)
                
                target_latents = pipeline._encode_vae_image(
                    target_images_vae.unsqueeze(2),  # Add frame dimension
                    generator=None
                )
                
                # Mover latents de vuelta al dispositivo principal (donde está el transformer)
                target_latents = target_latents.to(accelerator.device)
                
                # Pack latents
                batch_size, num_channels, _, height, width = target_latents.shape
                target_latents_packed = pipeline._pack_latents(
                    target_latents, batch_size, num_channels, height, width
                )
                
                # Encode source image for conditioning
                source_images_encoder = source_images.to(vae_device, dtype=pipeline.text_encoder.dtype)
                source_resized = F.interpolate(
                    source_images_encoder,
                    size=(384, 384),
                    mode='bilinear',
                    align_corners=False
                )
                
                # Get text embeddings
                # encode_prompt puede requerir que el device se pase explícitamente o lo infiera
                prompt_embeds, prompt_embeds_mask = pipeline.encode_prompt(
                    prompt=prompts,
                    image=[source_resized[i] for i in range(source_resized.shape[0])],
                    device=pipeline.text_encoder.device,
                    num_images_per_prompt=1,
                )
                
                # Mover embeddings al dispositivo principal
                prompt_embeds = prompt_embeds.to(accelerator.device)
                prompt_embeds_mask = prompt_embeds_mask.to(accelerator.device)
            
            # Sample random timesteps
            timesteps = torch.randint(
                0, pipeline.scheduler.config.num_train_timesteps,
                (batch_size,),
                device=accelerator.device
            ).long()
            
            # Add noise to target latents (flow matching interpolation)
            noise = torch.randn_like(target_latents_packed)
            
            # Flow matching: interpolate between noise and target
            # x_t = (1 - t) * x_0 + t * noise
            timesteps_normalized = timesteps.float() / pipeline.scheduler.config.num_train_timesteps
            timesteps_normalized = timesteps_normalized.view(-1, 1, 1)
            
            noisy_latents = (1 - timesteps_normalized) * target_latents_packed + timesteps_normalized * noise
            
            # The model should predict the velocity: v = noise - x_0
            velocity_target = noise - target_latents_packed
            
            # Prepare image shapes for RoPE
            img_shapes = [[(1, height // 2, width // 2)]] * batch_size
            txt_seq_lens = prompt_embeds_mask.sum(dim=1).tolist()
            image_rotary_emb = pipeline.transformer.pos_embed(
                img_shapes, txt_seq_lens, device=accelerator.device
            )
            
            # Forward pass through transformer
            timestep_input = timesteps.float() / 1000.0
            
            model_output = pipeline.transformer(
                hidden_states=noisy_latents,
                timestep=timestep_input,
                encoder_hidden_states=prompt_embeds,
                encoder_hidden_states_mask=prompt_embeds_mask,
                image_rotary_emb=image_rotary_emb,
                return_dict=False,
            )[0]
            
            # Compute loss
            loss = compute_loss(
                model_output,
                velocity_target,
                timesteps,
                pipeline.scheduler,
            )
            
            # Backward pass
            accelerator.backward(loss)
            
            if accelerator.sync_gradients:
                accelerator.clip_grad_norm_(pipeline.transformer.parameters(), config.max_grad_norm)
            
            optimizer.step()
            lr_scheduler.step()
            optimizer.zero_grad()
        
        # Logging
        if accelerator.sync_gradients:
            progress_bar.update(1)
            global_step += 1
            
            if global_step % config.logging_steps == 0:
                logs = {
                    "loss": loss.detach().item(),
                    "lr": lr_scheduler.get_last_lr()[0],
                    "step": global_step,
                }
                progress_bar.set_postfix(**logs)
                accelerator.log(logs, step=global_step)
            
            # Save checkpoint
            if global_step % config.save_steps == 0:
                save_path = os.path.join(config.output_dir, f"checkpoint-{global_step}")
                accelerator.save_state(save_path)
                logger.info(f"Saved checkpoint to {save_path}")
        
        logs = {"loss": loss.detach().item()}
        progress_bar.set_postfix(**logs)
    
    progress_bar.close()
    return global_step


def validate(
    pipeline: QwenImageEditPlusPipeline,
    val_dataloader: DataLoader,
    accelerator: Accelerator,
    config: TrainingConfig,
    epoch: int,
):
    """Valida el modelo generando algunas imágenes"""
    pipeline.transformer.eval()
    
    logger.info("Running validation...")
    
    validation_images = []
    for i, batch in enumerate(val_dataloader):
        if i >= config.num_validation_samples:
            break
        
        source_images = batch["source_images"]
        prompts = batch["prompts"]
        
        with torch.inference_mode():
            # Generar imagen
            images = pipeline(
                image=[Image.fromarray((source_images[0].permute(1, 2, 0).cpu().numpy() * 255).astype(np.uint8))],
                prompt=prompts[0],
                num_inference_steps=6,
                height=config.resolution,
                width=config.resolution,
                generator=torch.Generator(device=accelerator.device).manual_seed(config.seed),
            ).images[0]
            
            validation_images.append(images)
    
    # Log images
    if accelerator.is_main_process:
        for i, img in enumerate(validation_images):
            img.save(os.path.join(config.output_dir, f"validation_epoch{epoch}_{i}.png"))
    
    pipeline.transformer.train()


def main():
    parser = argparse.ArgumentParser(description="Fine-tune Qwen-Image-Edit for multi-view camera rotation")
    parser.add_argument("--dataset_root", type=str, required=True, help="Root directory of the dataset")
    parser.add_argument("--output_dir", type=str, default="../models/finetuned_multiview", help="Output directory")
    parser.add_argument("--batch_size", type=int, default=1, help="Training batch size")
    parser.add_argument("--epochs", type=int, default=10, help="Number of training epochs")
    parser.add_argument("--learning_rate", type=float, default=1e-4, help="Learning rate")
    parser.add_argument("--resolution", type=int, default=512, help="Image resolution")
    parser.add_argument("--lora_rank", type=int, default=16, help="LoRA rank")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    
    args = parser.parse_args()
    
    # Setup config
    config = TrainingConfig(
        dataset_root=args.dataset_root,
        output_dir=args.output_dir,
        train_batch_size=args.batch_size,
        num_train_epochs=args.epochs,
        learning_rate=args.learning_rate,
        resolution=args.resolution,
        lora_rank=args.lora_rank,
        seed=args.seed,
    )
    
    # Setup accelerator
    accelerator_project_config = ProjectConfiguration(
        project_dir=config.output_dir,
        logging_dir=os.path.join(config.output_dir, "logs"),
    )
    
    accelerator = Accelerator(
        gradient_accumulation_steps=config.gradient_accumulation_steps,
        mixed_precision=config.mixed_precision,
        log_with="tensorboard",
        project_config=accelerator_project_config,
    )
    
    # Set seed
    set_seed(config.seed)
    
    # Create output directory
    os.makedirs(config.output_dir, exist_ok=True)
    
    # Save config
    if accelerator.is_main_process:
        with open(os.path.join(config.output_dir, "training_config.json"), "w") as f:
            json.dump(vars(config), f, indent=2)
    
    logger.info(f"Training configuration: {config}")
    
    # Load datasets
    logger.info("Loading datasets...")
    train_dataset = MultiViewDataset(
        dataset_root=config.dataset_root,
        resolution=config.resolution,
        split="train",
    )
    
    val_dataset = MultiViewDataset(
        dataset_root=config.dataset_root,
        resolution=config.resolution,
        split="val",
    )
    
    train_dataloader = DataLoader(
        train_dataset,
        batch_size=config.train_batch_size,
        shuffle=True,
        num_workers=config.dataloader_num_workers,
        collate_fn=collate_fn,
    )
    
    val_dataloader = DataLoader(
        val_dataset,
        batch_size=1,
        shuffle=False,
        num_workers=config.dataloader_num_workers,
        collate_fn=collate_fn,
    )
    
    # Load model
    logger.info("Loading model...")
    dtype = torch.bfloat16
    
    transformer = QwenImageTransformer2DModel.from_pretrained(
        config.transformer_model,
        subfolder="transformer",
        torch_dtype=dtype,
        low_cpu_mem_usage=True,
    )
    
    pipeline = QwenImageEditPlusPipeline.from_pretrained(
        config.base_model,
        transformer=transformer,
        torch_dtype=dtype,
    )
    
    # Enable LoRA
    if config.use_lora:
        logger.info("Enabling LoRA...")
        pipeline.transformer = enable_lora(pipeline.transformer, config)
    
    # Enable Gradient Checkpointing (CRITICAL for memory saving)
    pipeline.transformer.enable_gradient_checkpointing()
    logger.info("Gradient checkpointing enabled")
    
    # Freeze VAE and text encoder
    pipeline.vae.requires_grad_(False)
    pipeline.text_encoder.requires_grad_(False)
    
    # Setup optimizer
    optimizer = torch.optim.AdamW(
        pipeline.transformer.parameters(),
        lr=config.learning_rate,
        betas=(0.9, 0.999),
        weight_decay=0.01,
        eps=1e-8,
    )
    
    # Setup learning rate scheduler
    num_update_steps_per_epoch = len(train_dataloader) // config.gradient_accumulation_steps
    max_train_steps = config.num_train_epochs * num_update_steps_per_epoch
    
    lr_scheduler = get_scheduler(
        config.lr_scheduler,
        optimizer=optimizer,
        num_warmup_steps=config.lr_warmup_steps,
        num_training_steps=max_train_steps,
    )
    
    # Prepare with accelerator
    # Important: Do NOT move the transformer to device manually before prepare if using accelerate with multiple GPUs or offload
    pipeline.transformer, optimizer, train_dataloader, val_dataloader, lr_scheduler = accelerator.prepare(
        pipeline.transformer, optimizer, train_dataloader, val_dataloader, lr_scheduler
    )
    
    # Move other components to SECONDARY GPU (GPU 1) if available, otherwise keep on CPU or GPU 0
    secondary_device = torch.device("cuda:1") if torch.cuda.device_count() > 1 else accelerator.device
    logger.info(f"Moving VAE and Text Encoder to {secondary_device}")
    
    pipeline.vae.to(secondary_device, dtype=dtype)
    pipeline.text_encoder.to(secondary_device, dtype=dtype)
    
    # Training loop
    logger.info("***** Running training *****")
    logger.info(f"  Num examples = {len(train_dataset)}")
    logger.info(f"  Num Epochs = {config.num_train_epochs}")
    logger.info(f"  Instantaneous batch size per device = {config.train_batch_size}")
    logger.info(f"  Gradient Accumulation steps = {config.gradient_accumulation_steps}")
    logger.info(f"  Total optimization steps = {max_train_steps}")
    
    global_step = 0
    
    for epoch in range(config.num_train_epochs):
        global_step = train_one_epoch(
            pipeline=pipeline,
            train_dataloader=train_dataloader,
            optimizer=optimizer,
            lr_scheduler=lr_scheduler,
            accelerator=accelerator,
            config=config,
            epoch=epoch,
            global_step=global_step,
        )
        
        # Validation
        if epoch % 2 == 0:  # Validate every 2 epochs
            validate(
                pipeline=pipeline,
                val_dataloader=val_dataloader,
                accelerator=accelerator,
                config=config,
                epoch=epoch,
            )
        
        # Save epoch checkpoint
        if accelerator.is_main_process:
            save_path = os.path.join(config.output_dir, f"epoch-{epoch}")
            accelerator.save_state(save_path)
            logger.info(f"Saved epoch checkpoint to {save_path}")
    
    # Save final model
    accelerator.wait_for_everyone()
    if accelerator.is_main_process:
        # Save LoRA weights
        unwrapped_transformer = accelerator.unwrap_model(pipeline.transformer)
        unwrapped_transformer.save_pretrained(
            os.path.join(config.output_dir, "final_lora")
        )
        logger.info(f"Training complete! Final model saved to {config.output_dir}")
    
    accelerator.end_training()


if __name__ == "__main__":
    main()
