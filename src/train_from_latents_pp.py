"""
Script de Fine-tuning con Pipeline Parallelism (2 GPUs) usando torch.distributed.pipelining.
Basado en train_from_latents.py y dividiendo_por_capas_GPU.py.

Usage:
    torchrun --nproc_per_node=2 train_from_latents_pp.py --latents_dir path/to/latents ...
"""

import os
import sys
import argparse
from pathlib import Path
from dataclasses import dataclass
from typing import List, Tuple
from torch.utils.checkpoint import checkpoint

# --- CONFIGURACIÓN DE ENTORNO ---
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["HF_HOME"] = "/nas/antoniodetoro/qwen/hf_cache"
os.environ["TMPDIR"] = "/nas/antoniodetoro/qwen/tmp"
os.environ["PYTHONNOUSERSITE"] = "1"

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.distributed as dist
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm
from diffusers.optimization import get_scheduler
from diffusers import FlowMatchEulerDiscreteScheduler, QwenImageTransformer2DModel
from peft import LoraConfig, get_peft_model
from torch.distributed.pipelining import PipelineStage
from torch.distributed.pipelining.schedules import Schedule1F1B

import logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Configuración
@dataclass
class TrainingConfig:
    latents_dir: str = "../models/precomputed_latents"
    output_dir: str = "../models/finetuned_pp"
    base_model: str = "Qwen/Qwen-Image-Edit-2509"
    transformer_model: str = "linoyts/Qwen-Image-Edit-Rapid-AIO"
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
    if "LOCAL_RANK" not in os.environ:
        # Fallback for single node manual run without torchrun (debugging)
        os.environ["LOCAL_RANK"] = "0"
        os.environ["RANK"] = "0"
        os.environ["WORLD_SIZE"] = "1"
        
    rank = int(os.environ["RANK"])
    world_size = int(os.environ["WORLD_SIZE"])
    local_rank = int(os.environ["LOCAL_RANK"])

    if torch.cuda.is_available():
        torch.cuda.set_device(local_rank)
        device = torch.device(f"cuda:{local_rank}")
        backend = "nccl"
    else:
        device = torch.device("cpu")
        backend = "gloo"

    if not dist.is_initialized():
        dist.init_process_group(backend=backend)
        
    # PP group is same as world for 2-GPU setup
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
            if hasattr(self.inner_model, "time_text_embed"):
                 temb = self.inner_model.time_text_embed(timestep, hidden_states, None)
            else:
                 timesteps_proj = self.inner_model.time_proj(timestep)
                 timesteps_emb = self.inner_model.timestep_embedder(timesteps_proj.to(dtype=hidden_states.dtype))
                 temb = timesteps_emb # Simplify

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
            # evitando que los tensores complejos contaminen el tipo de dato.
            dummy_loss = torch.tensor(0.0, dtype=hidden_states.dtype, device=hidden_states.device)
            for inp in inputs:
                if isinstance(inp, torch.Tensor) and inp.requires_grad:
                    val = inp.sum()
                    # Si es un número complejo (rotary embeddings), nos quedamos solo con la parte real
                    if val.is_complex():
                        val = val.real
                    
                    dummy_loss = dummy_loss + val.to(hidden_states.dtype)
            
            hidden_states = hidden_states + 0.0 * dummy_loss
            
            # Return prediction only
            return hidden_states

def loss_fn(outputs, targets):
    return F.mse_loss(outputs.float(), targets.float(), reduction="mean")

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--latents_dir", type=str, required=True)
    parser.add_argument("--output_dir", type=str, default="../models/finetuned_pp")
    parser.add_argument("--base_model", type=str, default="Qwen/Qwen-Image-Edit-2509")
    parser.add_argument("--transformer_model", type=str, default="linoyts/Qwen-Image-Edit-Rapid-AIO")
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
    if world_size != 2:
        if rank == 0: logger.error("This script requires exactly 2 GPUs (world_size=2).")
        return

    if rank == 0:
        logger.info(f"Starting PP Training on {world_size} GPUs. Global BS={config.batch_size}, Microbatches={config.microbatches}")
        os.makedirs(config.output_dir, exist_ok=True)

    # Load Model (CPU init to avoid VRAM spike)
    logger.info(f"Rank {rank}: Loading ALL model weights on CPU...")
    # NOTE: Loading full model on each rank is redundant but safest for structure correctness.
    # It consumes CPU RAM. If limited, we can initialize empty and load parts.
    dtype = torch.float16 # or bfloat16
    
    # 1. Load Base Model (CPU init)
    transformer = QwenImageTransformer2DModel.from_pretrained(
        config.transformer_model,
        subfolder="transformer",
        torch_dtype=dtype,
        low_cpu_mem_usage=True,
        device_map="cpu"
    )

    # 2. Split Model FIRST (This safely discards unused layers)
    logger.info(f"Rank {rank}: Splitting model...")
    model_split = QwenSplitWrapper(transformer, rank, world_size)
   
    
   # 3. Add LoRA adapters ONLY to the kept parts
    logger.info(f"Rank {rank}: Adding LoRA adapters...")
    lora_config = LoraConfig(
        r=config.lora_rank,
        lora_alpha=config.lora_alpha,
        init_lora_weights="gaussian",
        target_modules=["to_q", "to_k", "to_v", "to_out.0"],
        lora_dropout=config.lora_dropout,
    )
    transformer = get_peft_model(transformer, lora_config)


    # 4. Freeze non-lora
    for name, param in transformer.named_parameters():
             if "lora" not in name:
                 param.requires_grad = False
                 
    # 5. Move to GPU
    model_split.to(device)
    logger.info(f"Rank {rank}: Model moved to GPU.")


    # Optimizer (Construct ONLY after moving/splitting to capture correct params)
    optimizer = torch.optim.AdamW(model_split.parameters(), lr=config.lr)
    
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
    
    # Schedule
    schedule = Schedule1F1B(stage, n_microbatches=config.microbatches, loss_fn=loss_fn)
    
    # Dataloader
    dataset = LatentsDataset(config.latents_dir)
    dataloader = DataLoader(
        dataset, 
        batch_size=config.batch_size, 
        collate_fn=collate_latents, 
        drop_last=True, 
        shuffle=True, # Shuffle global
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
            
            if rank == 0:
                # Input for Pipeline (Will be chunked by Schedule1F1B)
                # Ensure batch dimension is 0
                inputs = (noisy_core, prompt, mask, timesteps) 
                schedule.step(*inputs)
                
            elif rank == 1:
                # Target for loss (Will be chunked by Schedule1F1B internally? 
                # Schedule1F1B usually doesn't chunk 'target' passed to step automatically? 
                # Wait, step(target=...) passes target to loss_fn.
                # Does loss_fn receive microbatch target or full target?
                # It receives microbatch target.
                # Does schedule chunk the target?
                # Yes, if it detects target as tensor matching batch dim.
                
                losses = []
                schedule.step(target=velocity_target, losses=losses)
                
                if len(losses) > 0:
                    step_loss = torch.mean(torch.stack(losses)).item()
                    avg_loss += step_loss
                    steps += 1
            
            optimizer.step()
            
            if rank == 1 and step % 10 == 0 and steps > 0:
                # Print loss? Hard to tqdm update from rank 1 if rank 0 holds bar.
                pass
                
        if rank == 1 and steps > 0:
            logger.info(f"Epoch {epoch} Loss: {avg_loss/steps}")

        # Save Checkpoint (Hacked: Save split parts)
        # Ideally reconstruct state_dict.
        # For simplicity, Rank 0 saves part0, Rank 1 saves part1.
        save_path = os.path.join(config.output_dir, f"epoch-{epoch}-rank-{rank}")
        logger.info(f"Saving checkpoint to {save_path}")
        # Need to save lora weights only ideally
        # model_split.parameters() includes only local params.
        torch.save(model_split.state_dict(), save_path + ".pt")
             
    dist.destroy_process_group()

if __name__ == "__main__":
    main()
