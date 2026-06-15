from typing import List, Tuple

import torch
import torch.nn.functional as F
import torchvision.transforms as T
from PIL import Image
from diffusers.models import AutoencoderKLQwenImage


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
    with_grad: bool = False,       # ← nuevo parámetro
) -> torch.Tensor:
    """Desnormaliza + decodifica latentes empaquetados a tensor de imágenes."""
    spatial = unpack_latents(latents_packed.float(), img_height, img_width).to(vae.dtype)

    vae_mean = torch.tensor(vae.config.latents_mean).view(
        1, vae.config.z_dim, 1, 1, 1).to(spatial.device, spatial.dtype)
    vae_std  = torch.tensor(vae.config.latents_std).view(
        1, vae.config.z_dim, 1, 1, 1).to(spatial.device, spatial.dtype)
    z_raw = spatial * vae_std + vae_mean

    if with_grad:
        decoded = vae.decode(z_raw, return_dict=False)[0]
    else:
        with torch.no_grad():
            decoded = vae.decode(z_raw, return_dict=False)[0]
    decoded = decoded[:, :, 0]

    return decoded


# def latents_to_images(  # antigua versión sin opción de gradientes
#     latents_packed: torch.Tensor,
#     vae: AutoencoderKLQwenImage,
#     img_height: int,
#     img_width: int,
# ) -> torch.Tensor:
#     """Desnormaliza + decodifica latentes empaquetados a tensor de imágenes."""
#     spatial = unpack_latents(latents_packed.float(), img_height, img_width).to(vae.dtype)

#     vae_mean = torch.tensor(vae.config.latents_mean).view(
#         1, vae.config.z_dim, 1, 1, 1).to(spatial.device, spatial.dtype)
#     vae_std  = torch.tensor(vae.config.latents_std).view(
#         1, vae.config.z_dim, 1, 1, 1).to(spatial.device, spatial.dtype)
#     z_raw = spatial * vae_std + vae_mean

#     with torch.no_grad():
#         decoded = vae.decode(z_raw, return_dict=False)[0]
#     decoded = decoded[:, :, 0]

#     return decoded


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