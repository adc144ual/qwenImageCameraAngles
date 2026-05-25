from typing import Tuple

import torch
from torch.utils.data import DataLoader

from loss import CombinedLossFn


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