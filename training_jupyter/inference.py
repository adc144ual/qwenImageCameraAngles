import logging
import os

import torch
import torchvision.transforms as T
from diffusers import FlowMatchEulerDiscreteScheduler
from diffusers.models import AutoencoderKLQwenImage
from PIL import Image
from torch.utils.data import Subset

from latent_utils import unpack_latents

logger = logging.getLogger(__name__)


@torch.no_grad()
def run_inference_callback(
    model,
    vae: AutoencoderKLQwenImage,
    base_scheduler: FlowMatchEulerDiscreteScheduler,
    device: torch.device,
    dataset,
    save_dir: str,
    epoch: int,
    img_height: int = 1024,
    img_width: int = 1024,
    num_steps: int = 4,
    num_samples: int = 2,
    dtype: torch.dtype = torch.bfloat16,
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