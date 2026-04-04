"""
Script de Fine-tuning con Pipeline Parallelism (2 GPUs) usando torch.distributed.pipelining.

Fixes aplicados respecto a la versión anterior:
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

Usage:
    torchrun --nproc_per_node 2 train_from_latents_pp_QLoRA.py \\
        --latents_dir "/ruta/precomputed_latents" \\
        --output_dir "output_test_fix" \\
        --batch_size 4 \\
        --microbatches 4 \\
        --epochs 200 \\
        --learning_rate 1e-4
"""

import os
import sys

current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
sys.path.append(parent_dir)

import argparse
from pathlib import Path
from dataclasses import dataclass
from typing import List, Tuple
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


@dataclass
class TrainingConfig:
    latents_dir: str = "../models/precomputed_latents_fix"
    output_dir: str = "../models/finetuned_pp"
    base_model: str = "Qwen/Qwen-Image-Edit-2509"
    transformer_model: str = "../models/Qwen-Fused-Angles"
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
    Collate que padea prompt_embeds a una longitud global fija.
    PipelineStage valida shapes contra el primer forward; todos los batches deben
    tener la misma forma.
    """
    def collate_latents(batch):
        target_latents  = torch.cat([item["target_latents_packed"]  for item in batch], dim=0)
        source_latents  = torch.cat([item["source_latents_packed"]  for item in batch], dim=0)
        prompt_list     = [item["prompt_embeds"]      for item in batch]
        mask_list       = [item["prompt_embeds_mask"] for item in batch]

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


# ---------------------------------------------------------------------------
# Loss: velocity MSE (FIX T1 + T2)
# ---------------------------------------------------------------------------

def velocity_loss(v_pred: torch.Tensor, v_target: torch.Tensor) -> torch.Tensor:
    """
    Loss de entrenamiento para flow matching.

    FIX T1+T2: la versión anterior calculaba velocity_target pero nunca la usaba
    y en su lugar hacía un MSE en espacio imagen a través del VAE, lo cual:
      - introducía gradientes extremadamente ruidosos para t alto
      - requería el VAE en GPU durante training (VRAM innecesaria)
      - la desnormalización era incorrecta (FIX T3)

    La loss correcta para flow matching es directamente:
        L = MSE(v_pred, v_target)
    donde v_target = noise - clean_latent.

    Si se quiere loss en x0 (alternativa válida y equivalente):
        x0_pred   = noisy - t * v_pred
        L         = MSE(x0_pred, clean_latent)
    Ambas son matemáticamente equivalentes salvo ponderación por t.
    Usamos la versión de velocidad porque es más estable con t alto.
    """
    # Recortar si el modelo devolvió tokens de condición adicionales
    if v_pred.shape[1] > v_target.shape[1]:
        v_pred = v_pred[:, :v_target.shape[1], :]
    return F.mse_loss(v_pred.float(), v_target.float(), reduction="mean")


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
        def _fwd(h_, e_, mask_, temb_, r0_, r1_):
            return block(
                hidden_states=h_,
                encoder_hidden_states=e_,
                encoder_hidden_states_mask=mask_,
                temb=temb_,
                image_rotary_emb=(r0_, r1_),
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
            # FIX T4: pos_embed espera (video_fhw, txt_seq_lens: List[int], device).
            # NO existe el kwarg max_txt_seq_len.
            B = hidden_states.shape[0]
            img_shapes   = self._compute_img_shapes(hidden_states, B)
            # txt_seq_lens: lista de longitudes reales (sin padding) por sample.
            txt_seq_lens = encoder_hidden_states_mask.sum(dim=1).long().tolist()
            # Si algún sample tiene máscara toda cero (padding total), usar longitud completa.
            full_len     = encoder_hidden_states.shape[1]
            txt_seq_lens = [max(l, 1) if l == 0 else l for l in txt_seq_lens]
            # Asegurar que los freqs cubran toda la secuencia incluido el padding.
            txt_seq_lens = [max(l, full_len) for l in txt_seq_lens]

            image_rotary_emb = self.inner_model.pos_embed(
                img_shapes, txt_seq_lens, device=hidden_states.device
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
            return (hidden_states, encoder_hidden_states, mask_float, temb, r0, r1)

        elif self.rank == 1:
            hidden_states, encoder_hidden_states, mask_float, temb, r0, r1 = inputs
            encoder_hidden_states_mask = mask_float.to(torch.int64)

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
# Loss wrapper para Schedule1F1B
# ---------------------------------------------------------------------------

class VelocityLossFn:
    """
    Loss de velocidad para flow matching, compatible con Schedule1F1B.

    FIX T1+T2: schedule.step recibe target = torch.stack([v_target, ...], dim=1)
    y la loss desempaqueta v_target para comparar con v_pred.
    Eliminamos el VAE del loop de training completamente.

    El target combinado tiene forma (B, 2, Seq, C):
        canal 0: v_target (velocity target = noise - clean_latent)
        canal 1: noisy_core  (solo para logging/diagnóstico, no para la loss)
    """

    def __init__(self, save_dir: str = None):
        self.save_dir     = save_dir
        self.step_counter = 0
        if save_dir is not None:
            os.makedirs(save_dir, exist_ok=True)

    def __call__(self, outputs: torch.Tensor, combined_target: torch.Tensor) -> torch.Tensor:
        # combined_target: (B, 2, Seq, C)
        v_target   = combined_target[:, 0, :, :].float()   # velocity target
        # noisy_core = combined_target[:, 1, :, :]  # disponible para diagnóstico

        v_pred = outputs.float()
        # Recortar tokens de condición si el modelo los devuelve
        if v_pred.shape[1] > v_target.shape[1]:
            v_pred = v_pred[:, :v_target.shape[1], :]

        if torch.isnan(v_pred).any() or torch.isinf(v_pred).any():
            logger.warning(
                f"[LOSS step {self.step_counter}] NaN/Inf en v_pred. "
                f"max={v_pred.abs().nanmax().item():.2f}"
            )
            self.step_counter += 1
            return torch.tensor(0.0, device=outputs.device, requires_grad=True)

        loss = F.mse_loss(v_pred, v_target, reduction="mean")
        self.step_counter += 1
        return loss


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

    # ── 8. Pipeline stage ────────────────────────────────────────────────────
    logger.info(f"Rank {rank}: init PipelineStage...")
    stage = PipelineStage(
        model_split,
        stage_index=rank,
        num_stages=world_size,
        device=device,
        group=pp_group,
    )

    # ── 9. VAE solo en rank 1 (para inferencia/visualización) ────────────────
    vae = None
    if rank == 1:
        logger.info(f"Rank {rank}: cargando VAE en {device}...")
        vae = AutoencoderKLQwenImage.from_pretrained(
            config.base_model, subfolder="vae", torch_dtype=torch.float32
        ).to(device)
        vae.requires_grad_(False)
        vae.eval()

    # ── 10. FIX T1+T2: Loss de velocidad, sin VAE en el loop de training ─────
    loss_fn_train = VelocityLossFn(
        save_dir=os.path.join(config.output_dir, "loss_diagnostics_rank1") if rank == 1 else None
    ) if rank == 1 else (lambda x, y: torch.tensor(0.0, device=device, requires_grad=True))

    schedule = Schedule1F1B(stage, n_microbatches=config.microbatches, loss_fn=loss_fn_train)

    # ── 11. Dataloader ────────────────────────────────────────────────────────
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

    # Inferir resolución desde el primer archivo del dataset
    first_sample = torch.load(dataset.files[0], weights_only=True)
    img_resolution = first_sample.get("resolution", 1024)
    logger.info(f"Resolución inferida del dataset: {img_resolution}")

    logger.info(f"Rank {rank}: listo para entrenar.")
    model_split.train()

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
                # FIX T1+T2: combined_target contiene v_target en canal 0
                # y noisy en canal 1 (solo para diagnóstico).
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

        if rank == 1 and steps > 0:
            logger.info(f"Epoch {epoch} | Loss: {avg_loss / steps:.6f}")

        # ── Checkpoint ───────────────────────────────────────────────────────
        local_state = model_split.state_dict()
        local_lora  = {k: v.cpu() for k, v in local_state.items() if "lora" in k}

        gathered = [None] * world_size if rank == 0 else None
        dist.gather_object(local_lora, gathered if rank == 0 else None, dst=0)

        if rank == 0:
            merged = {}
            for d in gathered:
                merged.update(d)
            save_path = os.path.join(config.output_dir, f"qwen_lora_epoch_{epoch}.pt")
            torch.save(merged, save_path)
            logger.info(f"LoRA guardado: {save_path}")
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