import logging
import os
import traceback
from typing import Optional, Tuple

import torch
import torch.nn.functional as F
from diffusers.models import AutoencoderKLQwenImage

from hrnet import PoseHRNet
from latent_utils import latents_to_images, preprocess_image_for_hrnet, compute_pck

logger = logging.getLogger(__name__)


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

        self.last_velocity_loss = 0.0
        self.last_heatmap_loss  = 0.0

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
        self.last_velocity_loss = velocity_loss_val.item()

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
                    self.img_width,
                    with_grad=True
                )

                hrnet_input = preprocess_image_for_hrnet(
                    decoded_images,
                    self.hrnet_input_size
                )

                # with torch.no_grad(): # ← SÍ queremos gradientes para el HRNet para que la pérdida de heatmap influya en el entrenamiento
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
                self.last_heatmap_loss  = heatmap_loss_val.item()

            except Exception as e:
                logger.warning(f"[LOSS step {self.step_counter}] Error en heatmap loss: {e}")
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