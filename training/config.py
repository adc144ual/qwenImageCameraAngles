from dataclasses import dataclass
from typing import Tuple


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