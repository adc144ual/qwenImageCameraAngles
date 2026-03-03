"""
Script de Fine-tuning con Pipeline Parallelism (2 GPUs) usando torch.distributed.pipelining.
Basado en train_from_latents.py y dividiendo_por_capas_GPU.py.

Usage:
    torchrun --nproc_per_node=2 train_from_latents_pp.py --latents_dir path/to/latents ...
"""

import os
import sys


# --- NUEVO: Añadir el directorio padre al path para encontrar 'qwenimage' ---
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
sys.path.append(parent_dir)
# ----------------------------------------------------------------------------

import argparse
from pathlib import Path
from dataclasses import dataclass
from typing import List, Tuple
from torch.utils.checkpoint import checkpoint

# --- CONFIGURACIÓN DE ENTORNO ---
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["HF_HOME"] = "/nas/antoniodetoro/qwen/hf_cache"
# os.environ["TMPDIR"] = "/nas/antoniodetoro/qwen/tmp"
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
    format="[Rank %(process)d] %(message)s"
    )

logger = logging.getLogger(__name__)

# Configuración
@dataclass
class TrainingConfig:
    latents_dir: str = "../models/precomputed_latents"
    output_dir: str = "../models/finetuned_pp"
    base_model: str = "Qwen/Qwen-Image-Edit-2509"
    transformer_model: str = "../models/Qwen-Fused-Angles"
    epochs: int = 3
    batch_size: int = 4 # Global batch size
    microbatches: int = 4 # Chunks per batch
    lr: float = 1e-4
    lora_rank: int = 16
    lora_alpha: int = 32
    lora_dropout: float = 0.1
    gradient_accumulation_steps: int = 1 # Not really used with 1F1B same way, but 1F1B does accum
    
class LatentsDataset(Dataset):
    def __init__(self, latents_dir, split="train"):
        self.split_dir = Path(latents_dir) / split
        self.files = sorted(list(self.split_dir.glob("*.pt")))
        if len(self.files) == 0:
            logger.warning(f"No files found in {self.split_dir}")

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        path = self.files[idx]
        return torch.load(path, weights_only=True)
    
class ImageSpaceLoss:
    def __init__(self, vae, img_height=512, img_width=512, save_dir="output_test_pp/images"):
        self.vae = vae
        self.img_height = img_height
        self.img_width = img_width
        self.save_dir = save_dir
        self.step_counter = 0
        
        # Crear la carpeta si no existe
        if self.save_dir is not None:
            os.makedirs(self.save_dir, exist_ok=True)
            
    def __call__(self, outputs, combined_target):
        # 1. Desempaquetar - FORZAR FLOAT32 para evitar overflow en fp16
        noisy_core = combined_target[:, 0, :, :].float()
        gt_latents_packed = combined_target[:, 1, :, :].float()
        t_norm = combined_target[:, 2, 0:1, 0:1].float()
        v_pred = outputs.float()
        
        # --- DIAGNÓSTICO: loguear dónde aparece NaN ---
        if torch.isnan(v_pred).any() or torch.isinf(v_pred).any():
            logger.warning(f"[LOSS step {self.step_counter}] NaN/Inf en v_pred (salida del modelo). "
                           f"max={v_pred.abs().max().item():.2f} nan={torch.isnan(v_pred).sum().item()}")
            # Si el modelo produce NaN, devolver 0 para no contaminar el grafo
            self.step_counter += 1
            return torch.tensor(0.0, device=outputs.device, requires_grad=True)
        
        # 2. Calcular x0 predicho matemáticamente (en float32 seguro)
        x0_pred = noisy_core - t_norm * v_pred
        
        # Clamp de seguridad antes del VAE
        x0_pred = torch.clamp(x0_pred, -100.0, 100.0)
        
        # 3. Convertir de Secuencia a Espacial
        x0_pred_spatial = unpack_latents(x0_pred, self.img_height, self.img_width, vae_scale_factor=8)
        gt_latents_spatial = unpack_latents(gt_latents_packed, self.img_height, self.img_width, vae_scale_factor=8)
        
        # 4. Normalización de Qwen
        x0_pred_spatial = x0_pred_spatial.to(self.vae.dtype)
        gt_latents_spatial = gt_latents_spatial.to(self.vae.dtype)
        
        latents_mean = torch.tensor(self.vae.config.latents_mean).view(1, self.vae.config.z_dim, 1, 1, 1).to(x0_pred_spatial.device, x0_pred_spatial.dtype)
        latents_std = 1.0 / torch.tensor(self.vae.config.latents_std).view(1, self.vae.config.z_dim, 1, 1, 1).to(x0_pred_spatial.device, x0_pred_spatial.dtype)
        
        x0_pred_norm = x0_pred_spatial / latents_std + latents_mean
        gt_latents_norm = gt_latents_spatial / latents_std + latents_mean
        
        # 5. Decodificar
        pred_image = self.vae.decode(x0_pred_norm, return_dict=False)[0][:, :, 0, :, :] # Eliminar la dimensión de frame que añadimos para VAE
        with torch.no_grad():
            gt_image = self.vae.decode(gt_latents_norm, return_dict=False)[0][:, :, 0]

        # --- DIAGNÓSTICO: detectar NaN tras el VAE ---
        if torch.isnan(pred_image).any() or torch.isinf(pred_image).any():
            logger.warning(f"[LOSS step {self.step_counter}] NaN/Inf tras VAE decode. "
                           f"x0_pred max={x0_pred.abs().max().item():.2f} "
                           f"x0_pred_norm max={x0_pred_norm.abs().max().item():.2f}")
            # Fallback: MSE directo en espacio de latentes, sin VAE
            self.step_counter += 1
            return F.mse_loss(x0_pred.float(), gt_latents_packed.float(), reduction="mean")
            
        # 6. GUARDAR LA IMAGEN CADA 50 STEPS (Ajusta este número si quieres más/menos imágenes)
        if self.save_dir is not None and self.step_counter % 50 == 0:
            # Tomamos la primera imagen del batch
            p_img = pred_image[0].detach().cpu().float()
            g_img = gt_image[0].detach().cpu().float()
            
            # Normalizar tensores al rango correcto de color [0, 1]
            p_img = torch.clamp((p_img / 2 + 0.5), 0, 1)
            g_img = torch.clamp((g_img / 2 + 0.5), 0, 1)
            
            p_pil = T.ToPILImage()(p_img)
            g_pil = T.ToPILImage()(g_img)
            
            # Crear una imagen combinada: Predicción (Izquierda) | Ground Truth (Derecha)
            w, h = p_pil.size
            combined = Image.new('RGB', (w * 2, h))
            combined.paste(p_pil, (0, 0))
            combined.paste(g_pil, (w, 0))
            
            # Guardar con formato step_0000.png, step_0050.png, etc.
            filename = os.path.join(self.save_dir, f"step_{self.step_counter:04d}.png")
            combined.save(filename)
            
        self.step_counter += 1
            
        # 7. Retornar la pérdida al Pipeline
        return F.mse_loss(pred_image.float(), gt_image.float(), reduction="mean")

def collate_latents(batch):
    target_latents = torch.cat([item["target_latents_packed"] for item in batch], dim=0)
    
    # Prompt embeds have variable length! We need to pad them.
    prompt_embeds_list = [item["prompt_embeds"] for item in batch]
    prompt_masks_list = [item["prompt_embeds_mask"] for item in batch]
    
    # Check max length in this batch
    max_len = max([pe.shape[1] for pe in prompt_embeds_list])
    
    padded_embeds = []
    padded_masks = []
    
    for i, (pe, pm) in enumerate(zip(prompt_embeds_list, prompt_masks_list)):
        # pe: [1, Seq, Dim]
        curr_len = pe.shape[1]
        
        # Debugging: check mask sum
        mask_sum = pm.sum()
        if mask_sum == 0:
            print(f"[COLLATE ERROR] Batch item {i}: prompt_embeds_mask sum is 0! Length {curr_len}")
            
        if curr_len < max_len:
            pad_len = max_len - curr_len
            # Pad embeds with zeros. F.pad tuple is (last_dim_left, last_dim_right, 2nd_last_left, 2nd_last_right...)
            # pe shape [1, Seq, Dim]. We want to pad Seq (dimension 1).
            # Last dim is Dim (index 2). No padding.
            # 2nd last is Seq (index 1). Padding pad_len at right.
            # 3rd last is Batch (index 0). No padding.
            pe_pad = F.pad(pe, (0, 0, 0, pad_len), value=0)
            
            # pm shape [1, Seq]. 
            # Last dim is Seq. Padding pad_len at right.
            pm_pad = F.pad(pm, (0, pad_len), value=0)
            
            padded_embeds.append(pe_pad)
            padded_masks.append(pm_pad)
        else:
            padded_embeds.append(pe)
            padded_masks.append(pm)
            
    prompt_embeds = torch.cat(padded_embeds, dim=0)
    prompt_embeds_mask = torch.cat(padded_masks, dim=0)

    return {
        "target_latents_packed": target_latents,
        "prompt_embeds": prompt_embeds,
        "prompt_embeds_mask": prompt_embeds_mask,
    }

def init_distributed():
    rank = int(os.environ["RANK"])
    world_size = int(os.environ["WORLD_SIZE"])
    local_rank = int(os.environ["LOCAL_RANK"])

    if torch.cuda.is_available():
        # LÓGICA DE INVERSIÓN:
        # Rank 0 (Capas 1-20) -> GPU 1 (24GB)
        # Rank 1 (Capas 21-40 + VAE) -> GPU 0 (32GB)
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

class QwenSplitWrapper(nn.Module):
    def __init__(self, model, rank, world_size):
        super().__init__()
        self.model = model
        self.rank = rank
        self.world_size = world_size
        
        # PEFT model wrapping necessitates accessing the underlying base model for surgery
        # PeftModel -> LoraModel -> QwenImageTransformer2DModel
        # We need to find where 'transformer_blocks' lives.
        
        if hasattr(model, "base_model") and hasattr(model.base_model, "model"):
             self.inner_model = model.base_model.model
        else:
             self.inner_model = model

        # Clean up model parts not needed on this rank
        total_layers = len(self.inner_model.transformer_blocks)
        split_layer = total_layers // 2 # Simple half split
        
        logger.info(f"Rank {rank}: Total layers {total_layers}. Split at {split_layer}.")
        
        if rank == 0:
            # Rank 0: Embeddings + Layers 0 to split_layer-1
            blocks_to_keep = self.inner_model.transformer_blocks[:split_layer]
            blocks_to_del = self.inner_model.transformer_blocks[split_layer:]
            
            self.inner_model.transformer_blocks = nn.ModuleList(blocks_to_keep)
            
            # Remove final layers
            if hasattr(self.inner_model, "norm_out"):
                del self.inner_model.norm_out
                self.inner_model.norm_out = None
            if hasattr(self.inner_model, "proj_out"):
                del self.inner_model.proj_out
                self.inner_model.proj_out = None
            
            # Explicitly clear deleted blocks to free memory
            for b in blocks_to_del:
                del b
            
        elif rank == 1:
            # Rank 1: Layers split_layer to end + Output
            blocks_to_keep = self.inner_model.transformer_blocks[split_layer:]
            blocks_to_del = self.inner_model.transformer_blocks[:split_layer]
            
            self.inner_model.transformer_blocks = nn.ModuleList(blocks_to_keep)
            
            # Remove embeddings
            for attr in ["img_in", "time_proj", "timestep_embedder", "txt_norm", "txt_in"]:
                if hasattr(self.inner_model, attr):
                    delattr(self.inner_model, attr)
            
            for b in blocks_to_del:
                del b
            
    def compute_img_shapes(self, hidden_states):
        # Infer img_shapes from hidden_states (packed)
        # B, Seq, C
        seq_len = hidden_states.shape[1]
        grid_size = int(seq_len ** 0.5)
        # Construct list[list[tuple]] matching batch
        bsz = hidden_states.shape[0]
        return [[(1, grid_size, grid_size)]] * bsz

    # def forward(self, *inputs):
    #     # inputs depends on Rank/Stage.
    #     # Rank 0 receives: noisy_core, prompt, mask, timesteps
    #     # Rank 1 receives: (hidden_states, encoder_hidden_states, encoder_hidden_states_mask, temb, rot0, rot1)
        
    #     if self.rank == 0:
    #         # Unpack initial inputs
    #         hidden_states, encoder_hidden_states, encoder_hidden_states_mask, timestep = inputs
            
    #         # --- Embedding Path (Rank 0 only) ---
    #         hidden_states = self.inner_model.img_in(hidden_states) # [B, Seq, Dim]
            
    #         timestep = timestep.to(hidden_states.dtype)
            
    #         encoder_hidden_states = self.inner_model.txt_norm(encoder_hidden_states)
    #         encoder_hidden_states = self.inner_model.txt_in(encoder_hidden_states)
            
    #         # Time & Text embeddings
    #         # Manual replication of time_text_embed logic if it's complex or method call
    #         if hasattr(self.inner_model, "time_text_embed"):
    #              temb = self.inner_model.time_text_embed(timestep, hidden_states, None)
    #         else:
    #              timesteps_proj = self.inner_model.time_proj(timestep)
    #              timesteps_emb = self.inner_model.timestep_embedder(timesteps_proj.to(dtype=hidden_states.dtype))
    #              temb = timesteps_emb # Simplify

    #         # Rotary Embeddings
    #         img_shapes = self.compute_img_shapes(hidden_states)
    #         text_seq_len = encoder_hidden_states_mask.sum(dim=1).max().item()
    #         # Ensure text_seq_len covers the full sequence length of encoder_hidden_states
    #         # If mask sum is 0 (all padding?), we still need valid freqs for the padded query
    #         full_text_len = encoder_hidden_states.shape[1]
    #         if text_seq_len < full_text_len:
    #              # If valid tokens are fewer than total tokens, we should probably generate freqs for ALL tokens
    #              # or at least enough to cover the query shape.
    #              # The Apply Rot Qwen function expects x and freqs to match. 
    #              # x is [B, S, H, D]. freqs is [S, D].
    #              # So we need freqs of length S (full_text_len).
    #              text_seq_len = full_text_len
            
    #         image_rotary_emb = self.inner_model.pos_embed(img_shapes, max_txt_seq_len=text_seq_len, device=hidden_states.device)
            
    #         # Run Partial Blocks
    #         for block in self.inner_model.transformer_blocks:
    #             encoder_hidden_states, hidden_states = block(
    #                 hidden_states=hidden_states,
    #                 encoder_hidden_states=encoder_hidden_states,
    #                 encoder_hidden_states_mask=encoder_hidden_states_mask,
    #                 temb=temb,
    #                 image_rotary_emb=image_rotary_emb
    #             )
            
    #         # Pack for Rank 1
    #         # encoder_hidden_states_mask is int/bool and causes issues with PipelineStage setting requires_grad
    #         # We cast it to float for transport, and cast back in Rank 1?
    #         # Or ensure it's not treated as activation. 
    #         # But the simplest workaround for "only Tensors of floating point dtype can require gradients"
    #         # is to cast it to float, and Rank 1 casts it back to bool/long.
    #         mask_float = encoder_hidden_states_mask.to(dtype=hidden_states.dtype)
            
    #         rot0, rot1 = image_rotary_emb
    #         return (hidden_states, encoder_hidden_states, mask_float, temb, rot0, rot1)

    #     elif self.rank == 1:
    #         # Unpack from Rank 0
    #         hidden_states, encoder_hidden_states, mask_float, temb, rot0, rot1 = inputs
    #         # Cast mask back
    #         encoder_hidden_states_mask = mask_float.to(torch.int64) # or bool depending on usage
            
    #         image_rotary_emb = (rot0, rot1)
            
    #         # Run Remaining Blocks
    #         for block in self.inner_model.transformer_blocks:
    #             encoder_hidden_states, hidden_states = block(
    #                 hidden_states=hidden_states,
    #                 encoder_hidden_states=encoder_hidden_states,
    #                 encoder_hidden_states_mask=encoder_hidden_states_mask,
    #                 temb=temb,
    #                 image_rotary_emb=image_rotary_emb
    #             )
                
    #         # Final Layer
    #         # norm_out is AdaLayerNormContinuous, needs conditioning_embedding (temb)
    #         hidden_states = self.inner_model.norm_out(hidden_states, temb)
    #         hidden_states = self.inner_model.proj_out(hidden_states)
            
    #         # Return prediction only?
    #         return hidden_states

    def forward(self, *inputs):
        # inputs depends on Rank/Stage.
        # Rank 0 receives: noisy_core, prompt, mask, timesteps
        # Rank 1 receives: (hidden_states, encoder_hidden_states, encoder_hidden_states_mask, temb, rot0, rot1)
        
        # Función auxiliar para que el checkpoint de PyTorch pueda trazar los gradientes de la tupla rotary_emb
        def make_custom_forward(block_module):
            def custom_forward(h, e, mask, t, r0, r1):
                return block_module(
                    hidden_states=h, 
                    encoder_hidden_states=e, 
                    encoder_hidden_states_mask=mask, 
                    temb=t, 
                    image_rotary_emb=(r0, r1)
                )
            return custom_forward

        if self.rank == 0:
            # Unpack initial inputs
            hidden_states, encoder_hidden_states, encoder_hidden_states_mask, timestep = inputs
            
            # --- Embedding Path (Rank 0 only) ---
            hidden_states = self.inner_model.img_in(hidden_states) # [B, Seq, Dim]
            
            timestep = timestep.to(hidden_states.dtype)
            
            encoder_hidden_states = self.inner_model.txt_norm(encoder_hidden_states)
            encoder_hidden_states = self.inner_model.txt_in(encoder_hidden_states)
            
            # Time & Text embeddings
            # NOTA: time_text_embed en Qwen acepta (timestep, hidden_states) o (timestep, hidden_states, guidance).
            # Pasar None como guidance puede producir NaN en algunos modelos, usamos la rama segura siempre.
            if hasattr(self.inner_model, "time_proj") and hasattr(self.inner_model, "timestep_embedder"):
                 timesteps_proj = self.inner_model.time_proj(timestep.float()).to(dtype=hidden_states.dtype)
                 temb = self.inner_model.timestep_embedder(timesteps_proj)
            elif hasattr(self.inner_model, "time_text_embed"):
                 # Intentar llamada sin guidance primero; si falla, pasar tensor de ceros
                 try:
                     temb = self.inner_model.time_text_embed(timestep, hidden_states)
                 except TypeError:
                     temb = self.inner_model.time_text_embed(timestep, hidden_states,
                                 torch.zeros(hidden_states.shape[0], dtype=hidden_states.dtype, device=hidden_states.device))
            else:
                 raise RuntimeError("No se encontró time_proj/timestep_embedder ni time_text_embed en el modelo.")
            temb = temb.to(dtype=hidden_states.dtype)

            # Rotary Embeddings
            img_shapes = self.compute_img_shapes(hidden_states)
            text_seq_len = encoder_hidden_states_mask.sum(dim=1).max().item()
            full_text_len = encoder_hidden_states.shape[1]
            if text_seq_len < full_text_len:
                 text_seq_len = full_text_len
            
            image_rotary_emb = self.inner_model.pos_embed(img_shapes, max_txt_seq_len=text_seq_len, device=hidden_states.device)
            
            # Run Partial Blocks with Checkpointing
            for block in self.inner_model.transformer_blocks:
                if self.training:
                    if not hidden_states.requires_grad:
                        hidden_states.requires_grad_(True)
                    if not encoder_hidden_states.requires_grad:
                        encoder_hidden_states.requires_grad_(True)

                    # Llamamos al checkpoint pasando los rotaries separados
                    encoder_hidden_states, hidden_states = checkpoint(
                        make_custom_forward(block),
                        hidden_states,
                        encoder_hidden_states,
                        encoder_hidden_states_mask,
                        temb,
                        image_rotary_emb[0], # rot0
                        image_rotary_emb[1], # rot1
                        use_reentrant=False
                    )
                else:
                    encoder_hidden_states, hidden_states = block(
                        hidden_states=hidden_states,
                        encoder_hidden_states=encoder_hidden_states,
                        encoder_hidden_states_mask=encoder_hidden_states_mask,
                        temb=temb,
                        image_rotary_emb=image_rotary_emb
                    )
            
            # Pack for Rank 1
            mask_float = encoder_hidden_states_mask.to(dtype=hidden_states.dtype)
            rot0, rot1 = image_rotary_emb
            return (hidden_states, encoder_hidden_states, mask_float, temb, rot0, rot1)

        elif self.rank == 1:
            # Unpack from Rank 0
            hidden_states, encoder_hidden_states, mask_float, temb, rot0, rot1 = inputs
            
            # Cast mask back
            encoder_hidden_states_mask = mask_float.to(torch.int64) # or bool depending on usage
            image_rotary_emb = (rot0, rot1)
            
            # Run Remaining Blocks with Checkpointing
            for block in self.inner_model.transformer_blocks:
                if self.training:
                    if not hidden_states.requires_grad:
                        hidden_states.requires_grad_(True)
                    if not encoder_hidden_states.requires_grad:
                        encoder_hidden_states.requires_grad_(True)
                    
                    # Llamamos al checkpoint usando rot0 y rot1 que hemos desempaquetado de Rank 0
                    encoder_hidden_states, hidden_states = checkpoint(
                        make_custom_forward(block),
                        hidden_states,
                        encoder_hidden_states,
                        encoder_hidden_states_mask,
                        temb,
                        rot0,
                        rot1,
                        use_reentrant=False
                    )
                else:
                    encoder_hidden_states, hidden_states = block(
                        hidden_states=hidden_states,
                        encoder_hidden_states=encoder_hidden_states,
                        encoder_hidden_states_mask=encoder_hidden_states_mask,
                        temb=temb,
                        image_rotary_emb=image_rotary_emb
                    )
                
           # Final Layer
            hidden_states = self.inner_model.norm_out(hidden_states, temb)
            hidden_states = self.inner_model.proj_out(hidden_states)

           # FIX DEFINITIVO 2.0: Atar TODOS los inputs originales al grafo
            # evitando que los tensores complejos contaminen el tipo de dato y prevenir desbordamiento (overflow) a NaN en fp16.
            dummy_loss = torch.tensor(0.0, dtype=hidden_states.dtype, device=hidden_states.device)
            for inp in inputs:
                if isinstance(inp, torch.Tensor) and inp.requires_grad:
                    # 1. Pasamos a float32 y usamos mean() en lugar de sum() para no superar 65.504
                    val = inp.mean()
                    # Si es un número complejo (rotary embeddings), nos quedamos solo con la parte real
                    if val.is_complex():
                        val = val.real

                    # 2. Multiplicamos por 0.0 AQUÍ (en float32 seguro) antes de pasarlo a float16
                    dummy_loss = dummy_loss + (val * 0.0).to(hidden_states.dtype)

            # Como dummy_loss ya es 0 puro, solo sumamos 0.0 * dummy_loss a hidden_states para mantener el grafo conectado sin afectar los valores ni causar overflow.
            hidden_states = hidden_states + 0.0 * dummy_loss
            
            # Return prediction only
            return hidden_states

def loss_fn(outputs, targets):
    return F.mse_loss(outputs.float(), targets.float(), reduction="mean")

# Copied from diffusers.pipelines.qwenimage.pipeline_qwenimage.QwenImagePipeline._unpack_latents
def unpack_latents(latents, height, width, vae_scale_factor):
    batch_size, num_patches, channels = latents.shape

    # VAE applies 8x compression on images but we must also account for packing which requires
    # latent height and width to be divisible by 2.
    height = 2 * (int(height) // (vae_scale_factor * 2))
    width = 2 * (int(width) // (vae_scale_factor * 2))

    latents = latents.view(batch_size, height // 2, width // 2, channels // 4, 2, 2)
    latents = latents.permute(0, 3, 1, 4, 2, 5)

    latents = latents.reshape(batch_size, channels // (2 * 2), 1, height, width)

    return latents

def make_image_space_loss_fn(vae, img_height=512, img_width=512):
    def loss_fn(outputs, combined_target):
        # 1. Desempaquetar el tensor combinado que PyTorch cortó en microbatches
        noisy_core = combined_target[:, 0, :, :]
        gt_latents_packed = combined_target[:, 1, :, :]
        
        # t_norm se expandió a [B, Seq, C], recuperamos su forma original [B, 1, 1]
        t_norm = combined_target[:, 2, 0:1, 0:1] 
        
        v_pred = outputs
        
        # 2. Calcular x0 predicho matemáticamente
        x0_pred = noisy_core - t_norm * v_pred
        
        # 3. Convertir de Secuencia a Espacial (Usa tu unpack_latents original aquí)
        x0_pred_spatial = unpack_latents(x0_pred, img_height, img_width, vae_scale_factor=8)
        gt_latents_spatial = unpack_latents(gt_latents_packed, img_height, img_width, vae_scale_factor=8)
        
        # 4. Normalización para Qwen
        x0_pred_spatial = x0_pred_spatial.to(vae.dtype)
        gt_latents_spatial = gt_latents_spatial.to(vae.dtype)
        
        latents_mean = torch.tensor(vae.config.latents_mean).view(1, vae.config.z_dim, 1, 1, 1).to(x0_pred_spatial.device, x0_pred_spatial.dtype)
        latents_std = 1.0 / torch.tensor(vae.config.latents_std).view(1, vae.config.z_dim, 1, 1, 1).to(x0_pred_spatial.device, x0_pred_spatial.dtype)
        
        x0_pred_norm = x0_pred_spatial / latents_std + latents_mean
        gt_latents_norm = gt_latents_spatial / latents_std + latents_mean
        
        # 5. Decodificar
        pred_image = vae.decode(x0_pred_norm, return_dict=False)[0]
        with torch.no_grad():
            gt_image = vae.decode(gt_latents_norm, return_dict=False)[0]
            
        # 6. Calcular Loss Perceptual / MSE
        return F.mse_loss(pred_image.float(), gt_image.float(), reduction="mean")
        
    return loss_fn

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--latents_dir", type=str, required=True)
    parser.add_argument("--output_dir", type=str, default="../models/finetuned_pp")
    parser.add_argument("--base_model", type=str, default="Qwen/Qwen-Image-Edit-2509")
    # CAMBIO: Apunta por defecto a tu carpeta local con el modelo fusionado
    parser.add_argument("--transformer_model", type=str, default="../models/Qwen-Fused-Angles")
    parser.add_argument("--epochs", type=int, default=3)
    # Global batch size must be divisible by microbatches
    parser.add_argument("--batch_size", type=int, default=4, help="Global batch size")
    parser.add_argument("--microbatches", type=int, default=4, help="Number of microbatches (chunks)")
    parser.add_argument("--learning_rate", type=float, default=1e-4)    
    parser.add_argument("--lora_rank", type=int, default=16) 
    parser.add_argument("--lora_alpha", type=int, default=32) 
    parser.add_argument("--lora_dropout", type=float, default=0.1) 
    
    args, unknown = parser.parse_known_args()
    
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
        lora_dropout=args.lora_dropout
    )
    
    rank, world_size, device, pp_group = init_distributed()
    
    # --- FIX: Definir dtype globalmente ---
    dtype = torch.float16
    # --------------------------------------

    if world_size != 2:
        if rank == 0: logger.error("This script requires exactly 2 GPUs. Use torchrun --nproc_per_node=2")
        return

    if rank == 0:
        logger.info(f"Starting Training on {world_size} GPUs. Global BS={config.batch_size}, Microbatches={config.microbatches}")
        os.makedirs(config.output_dir, exist_ok=True)

    # 1. Configuración de Quantization (QLoRA)
    # Cargar en 4 bits (NF4), cálculo en float16, doble cuantización para ahorro extra
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.float32,
        bnb_4bit_use_double_quant=True,
    )

    # 2. Cargar Transformer Fusionado directamente con Quantization
    # Usamos device_map=str(device) para que bitsandbytes cargue directo en VRAM 
    # y evitamos picos de RAM en CPU.
    logger.info(f"Rank {rank}: Loading quantized model from {config.transformer_model} directly to {device}...")
    
    # IMPORTANTE: Aseguramos torch_dtype=torch.float16 para que BnB reciba inputs compatibles (FP16/FP32).
    # Si la carpeta tiene safetensors en otro formato, esto fuerza el cast durante la carga antes de cuantizar.
    transformer = QwenImageTransformer2DModel.from_pretrained(
        config.transformer_model,
        subfolder=None,
        quantization_config=bnb_config,
        torch_dtype=torch.float16,
        device_map=str(device)
    )

    # 3. Model Surgery (Splitting) - Safely discards unused layers
    logger.info(f"Rank {rank}: Splitting model...")
    model_split = QwenSplitWrapper(transformer, rank, world_size)
    
    # 4. Preparar para entrenamiento k-bit
    # Congela base, castea LayerNorms a float32 para estabilidad.
    # use_gradient_checkpointing=False porque ya se hace manualmente en vuestro forward loop
    transformer = prepare_model_for_kbit_training(transformer, use_gradient_checkpointing=False)

    # 5. Add LoRA adapters (QLoRA)
    # La base está en int4, los adaptadores se inyectan en float32 y son entrenables.
    logger.info(f"Rank {rank}: Adding LoRA adapters (QLoRA)...")
    lora_config = LoraConfig(
        r=config.lora_rank,
        lora_alpha=config.lora_alpha,
        init_lora_weights=True,  # Zero-init: evita NaN por explosión en fp16 con init Gaussiana
        target_modules=["to_q", "to_k", "to_v", "to_out.0"],
        lora_dropout=config.lora_dropout,
    )
    
    # Esto modifica el modelo in-place añadiendo los adaptadores
    transformer = get_peft_model(transformer, lora_config)

    if rank == 0:
        transformer.print_trainable_parameters()

    # NO hacemos model_split.to(device) aquí porque los modelos de 4 bit ya están 
    # anclados al dispositivo de carga y moverlos explícitamente suele dar error.
    logger.info(f"Rank {rank}: Model is ready on GPU.")

    # Optimizer (Construct ONLY after moving/splitting to capture correct params)
    optimizer = torch.optim.AdamW(model_split.parameters(), lr=config.lr, eps=1e-06) # epsilon pequeño recomendado para fp16
    
    # Scheduler
    scheduler = FlowMatchEulerDiscreteScheduler.from_pretrained(config.base_model, subfolder="scheduler")
    
    # Pipeline Stage
    logger.info(f"Rank {rank}: Init PipelineStage...")
    stage = PipelineStage(
        model_split,
        stage_index=rank,
        num_stages=world_size,
        device=device,
        group=pp_group,
    )

    # =================================================================
    # AQUÍ CARGAMOS EL VAE (Justo antes de crear la Loss y el Schedule)
    # =================================================================
    vae = None
    if rank == 1:
        logger.info(f"Rank {rank}: Cargando VAE en GPU {device} para la loss perceptual...")
        vae = AutoencoderKLQwenImage.from_pretrained(
            config.base_model, 
            subfolder="vae", 
            torch_dtype=torch.float32
        ).to(device)
        vae.requires_grad_(False)
        vae.eval()

        loss_fn = ImageSpaceLoss(
            vae=vae,
            img_height=512,
            img_width=512, 
            save_dir=os.path.join(config.output_dir, "predictions_rank1")
        )
    else:
        loss_fn = lambda x, y: torch.tensor(0.0, device=device, requires_grad=True) # Dummy loss for Rank 0

    # Schedule
    schedule = Schedule1F1B(stage, n_microbatches=config.microbatches, loss_fn=loss_fn)
    
    g = torch.Generator()
    g.manual_seed(42) # Seed global para reproducibilidad (aunque cada paso también tiene su propio seed)

    # Dataloader
    dataset = LatentsDataset(config.latents_dir)
    dataloader = DataLoader(
        dataset, 
        batch_size=config.batch_size, 
        collate_fn=collate_latents, 
        drop_last=True, 
        shuffle=True, # Shuffle global,
        generator=g,
        num_workers=4
    )
    
    logger.info(f"Rank {rank}: Ready to train.")
    
    model_split.train()
    
    for epoch in range(config.epochs):
        if rank == 0: logger.info(f"Epoch {epoch} start")
        
        # tqdm only on rank 0
        desc = f"Epoch {epoch}"
        iterator = tqdm(dataloader, desc=desc) if rank == 0 else dataloader
        
        avg_loss = 0.0
        steps = 0
        
        for step, batch in enumerate(iterator):
            optimizer.zero_grad()
            
            # Prepare Data Slices
            target = batch["target_latents_packed"].to(device, dtype=dtype) # [GlobalBS, Seq, C]
            prompt = batch["prompt_embeds"].to(device, dtype=dtype)
            mask = batch["prompt_embeds_mask"].to(device)
            
            # Sync Random Number Gen for consistent noise
            # Best way: generate on CPU with fixed seed or broadcast?
            # CPU gen + to(device) is safest
            
            # Simple sync seed every step
            g_seed = 42 + epoch * 1000 + step
            gen = torch.Generator(device=device).manual_seed(g_seed)
            
            bsz = target.shape[0]
            timesteps = torch.randint(0, scheduler.config.num_train_timesteps, (bsz,), generator=gen, device=device).long()
            noise = torch.randn(target.shape, generator=gen, device=device, dtype=dtype)
            
            # Flow Match Inputs
            t_norm = (timesteps.float() / scheduler.config.num_train_timesteps).to(dtype)
            t_norm = t_norm.view(-1, 1, 1)
            noisy_core = (1 - t_norm) * target + t_norm * noise
            velocity_target = noise - target
            
            # --- MODO PIPELINE (2 GPUs) ---
            if rank == 0:
                inputs = (noisy_core, prompt, mask, timesteps) 
                schedule.step(*inputs)
                torch.nn.utils.clip_grad_norm_(model_split.parameters(), max_norm=1.0)
                optimizer.step()
                    
            elif rank == 1:
                t_norm_expanded = t_norm.expand_as(noisy_core)
                combined_target = torch.stack([noisy_core, target, t_norm_expanded], dim=1)
                losses = []
                schedule.step(target=combined_target, losses=losses)
                torch.nn.utils.clip_grad_norm_(model_split.parameters(), max_norm=1.0)
                optimizer.step()
                
                if len(losses) > 0:
                    step_loss = torch.mean(torch.stack(losses)).item()
                    avg_loss += step_loss
                    steps += 1
                
        if rank == 1 and steps > 0:
            logger.info(f"Epoch {epoch} Loss: {avg_loss/steps}")

       # ==========================================================
        # GUARDAR CHECKPOINT
        # ==========================================================
        
        local_state_dict = model_split.state_dict()
        local_lora_only = {
            k: v.cpu() for k, v in local_state_dict.items() if "lora" in k
        }

        gathered_lora_dicts = [None for _ in range(world_size)] if rank == 0 else None
        dist.gather_object(local_lora_only, gathered_lora_dicts if rank == 0 else None, dst=0)

        if rank == 0:
            merged_lora = {}
            for lora_dict in gathered_lora_dicts:
                merged_lora.update(lora_dict)
            save_path = os.path.join(config.output_dir, f"qwen_lora_epoch_{epoch}.pt")
            torch.save(merged_lora, save_path)
            logger.info(f"✅ Pesos de LoRA fusionados y guardados en: {save_path}")
        dist.barrier()
             
    # Destruir el grupo de procesos de red una vez terminadas todas las épocas
    dist.destroy_process_group()

if __name__ == "__main__":
    main()