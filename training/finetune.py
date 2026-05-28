"""
Script de Fine-tuning en single GPU.

VERSIÓN MODIFICADA CON HEATMAP LOSS DE HRNET Y VALIDACIÓN

Fixes aplicados respecto a la versión base:
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

MODIFICACIONES HEATMAP LOSS:
  H1     — Añadida arquitectura HRNet completa (PoseHRNet, Bottleneck, BasicBlock, etc.)
  H2     — Nueva función preprocess_image_for_hrnet() con normalización ImageNet
  H3     — Nueva función latents_to_images() para decodificar latentes a imágenes
  H4     — Clase CombinedLossFn que reemplaza VelocityLossFn:
            * Calcula x0_pred = noisy - t*v_pred (matemáticamente correcto)
            * Decodifica x0_pred con VAE
            * Calcula heatmaps con HRNet
            * Compara con GT heatmaps
            * Soporta dos tipos de loss: MSE simple y weighted MSE
  H5     — Modificado collate_latents() para extraer target_heatmaps de los .pt
  H6     — HRNet cargado en Rank 1 (mismo que VAE) y congelado
  H7     — Modificado training loop para pasar timesteps a la loss
  H8     — Añadidos argumentos CLI: --hrnet_model_path, --heatmap_loss_weight, etc.

NUEVAS MODIFICACIONES VALIDACIÓN:
  V1     — Split train/val con semilla fija (reproducible)
  V2     — Función validate() para calcular val_loss al final de cada época
  V3     — Val_loss guardado en CSV
  V4     — Argumento --val_split para controlar ratio de validación

Usage:
    python train.py \\
        --latents_dir "/ruta/precomputed_latents" \\
        --hrnet_model_path "./models/pose_hrnet_w48_384x288.pth" \\
        --output_dir "output_hrnet" \\
        --batch_size 4 \\
        --epochs 200 \\
        --learning_rate 1e-4 \\
        --heatmap_loss_weight 0.5 \\
        --velocity_loss_weight 0.5 \\
        --heatmap_loss_type "mse" \\
        --val_split 0.1



----------------------------------------------------------------

 python train.py --latents_dir "/data/antoniodetoro/qwen/dataset_local_latents_512_heatmaps/" --hrnet_model_path /nas/antoniodetoro/qwen/Qwen-Image-Edit-Angles-2/src/hr_net/models/hrnet_finetuned_best.pth --output_dir output_qwen_HRNet_single_GPU --batch_size 2 --epochs 20 --heatmap_loss_weight 0.5 --velocity_loss_weight 0.5 --heatmap_loss_type "mse"

"""
# train.py
import os
import sys
import csv
import argparse
import logging

current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
sys.path.append(parent_dir)
sys.path.append(current_dir)

os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["HF_HOME"] = "/home/antoniodetoro/nas_dataset/antonio/.qwen/hf_cache"
os.environ["TMPDIR"] = "/dev/shm"
os.environ["PYTHONNOUSERSITE"] = "1"

import torch
from torch.utils.data import DataLoader
from tqdm import tqdm
from diffusers import FlowMatchEulerDiscreteScheduler, QwenImageTransformer2DModel
from diffusers.models import AutoencoderKLQwenImage
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
from transformers import BitsAndBytesConfig
import json
from dataclasses import asdict

from config import TrainingConfig
from dataset import build_dataloaders
from hrnet import load_hrnet_model
from inference import run_inference_callback
from loss import CombinedLossFn
from model import QwenSingleGPUWrapper
from validate import validate


logging.basicConfig(level=logging.INFO, force=True, format="%(message)s")
logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Argparse
# ---------------------------------------------------------------------------

def parse_args() -> TrainingConfig:
    parser = argparse.ArgumentParser()
    parser.add_argument("--latents_dir",       type=str, required=True)
    parser.add_argument("--experiment_json",   type=str, required=True)
    parser.add_argument("--output_dir",        type=str, default="../models/finetuned_pp")
    parser.add_argument("--base_model",        type=str, default="Qwen/Qwen-Image-Edit-2509")
    parser.add_argument("--transformer_model", type=str, default="../models/Qwen-Fused-Angles")
    
    #HRNet
    parser.add_argument("--hrnet_model_path",  type=str, default="./models/pose_hrnet_w48_384x288.pth")
    parser.add_argument("--heatmap_loss_weight",  type=float, default=0.5)
    parser.add_argument("--velocity_loss_weight", type=float, default=0.5)
    parser.add_argument("--heatmap_loss_type", type=str, default="mse", choices=["mse", "weighted_mse"])
    
    #Training
    parser.add_argument("--epochs",            type=int,   default=3)
    parser.add_argument("--batch_size",        type=int,   default=4)
    parser.add_argument("--learning_rate",     type=float, default=1e-4)
    parser.add_argument("--lora_rank",         type=int,   default=16)
    parser.add_argument("--lora_alpha",        type=int,   default=32)
    parser.add_argument("--lora_dropout",      type=float, default=0.1)
    parser.add_argument("--inference_every",   type=int,   default=2)
    parser.add_argument("--inference_steps",   type=int,   default=4)
    parser.add_argument("--inference_samples", type=int,   default=5)
    args, _ = parser.parse_known_args()

    return TrainingConfig(
        latents_dir=args.latents_dir,
        experiment_json=args.experiment_json,
        output_dir=args.output_dir,
        base_model=args.base_model,
        transformer_model=args.transformer_model,
        hrnet_model_path=args.hrnet_model_path,
        velocity_loss_weight=args.velocity_loss_weight,
        heatmap_loss_weight=args.heatmap_loss_weight,
        heatmap_loss_type=args.heatmap_loss_type,
        epochs=args.epochs,
        batch_size=args.batch_size,
        lr=args.learning_rate,
        lora_rank=args.lora_rank,
        lora_alpha=args.lora_alpha,
        lora_dropout=args.lora_dropout,
        inference_every=args.inference_every,
        inference_steps=args.inference_steps,
        inference_samples=args.inference_samples,
    )


# ---------------------------------------------------------------------------
# Setup de modelo, checkpoints
# ---------------------------------------------------------------------------

def build_model(config: TrainingConfig, device: torch.device) -> QwenSingleGPUWrapper:
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16,
        bnb_4bit_use_double_quant=True,
    )
    logger.info(f"Cargando modelo desde {config.transformer_model} → {device}")
    transformer = QwenImageTransformer2DModel.from_pretrained(
        config.transformer_model,
        subfolder=None,
        quantization_config=bnb_config,
        torch_dtype=torch.bfloat16,
        device_map=str(device),
        attn_implementation="sdpa",
        local_files_only=True,
    )
    transformer = prepare_model_for_kbit_training(transformer, use_gradient_checkpointing=False)
    lora_config = LoraConfig(
        r=config.lora_rank,
        lora_alpha=config.lora_alpha,
        init_lora_weights=True,
        target_modules=["to_q", "to_k", "to_v", "to_out.0",
                        "add_q_proj", "add_k_proj", "add_v_proj", "to_add_out"],
        lora_dropout=config.lora_dropout,
    )
    transformer = get_peft_model(transformer, lora_config)
    transformer.print_trainable_parameters()
    logger.info("Creando QwenSingleGPUWrapper...")
    return QwenSingleGPUWrapper(transformer)


def load_checkpoint(model, config, device, optimizer=None, diff_scheduler=None):
    checkpoint_path = os.path.join(config.output_dir, "qwen_lora_best.pt")
    if not os.path.exists(checkpoint_path):
        logger.info("No se encontró checkpoint previo. Iniciando desde cero.")
        return 0, float('inf')
    try:
        ckpt = torch.load(checkpoint_path, map_location=device, weights_only=False)
        if isinstance(ckpt, dict) and "lora_state_dict" in ckpt:
            lora_state  = ckpt["lora_state_dict"]
            start_epoch = ckpt.get("epoch", -1) + 1
            best_loss   = ckpt.get("best_loss", float('inf'))

            # Restaurar optimizador
            if optimizer is not None and "optimizer_state_dict" in ckpt:
                optimizer.load_state_dict(ckpt["optimizer_state_dict"])
                logger.info("✓ Estado del optimizador restaurado")

            # Restaurar scheduler si tiene estado
            if diff_scheduler is not None and "scheduler_state_dict" in ckpt and ckpt["scheduler_state_dict"] is not None:
                diff_scheduler.load_state_dict(ckpt["scheduler_state_dict"])
                logger.info("✓ Estado del scheduler restaurado")

            logger.info(f"✓ Reanudando desde época {start_epoch} | best_loss={best_loss:.6f}")
        else:
            lora_state, start_epoch, best_loss = ckpt, 0, float('inf')
            logger.warning("Checkpoint en formato antiguo. Reanudando desde época 0.")

        missing, unexpected = model.load_state_dict(lora_state, strict=False)
        logger.info(f"{len(lora_state)-len(unexpected)}/{len(lora_state)} pesos LoRA cargados")
        return start_epoch, best_loss
    except Exception as e:
        logger.error(f"Error cargando checkpoint: {e}. Iniciando desde cero.")
        return 0, float('inf')

def save_checkpoint(model, config, epoch, best_loss, optimizer, diff_scheduler, extra_name=None):
    local_lora = {k: v.cpu() for k, v in model.state_dict().items() if "lora" in k}
    ckpt_data  = {
        "lora_state_dict":      local_lora,
        "epoch":                epoch,
        "best_loss":            best_loss,
        "optimizer_state_dict": optimizer.state_dict(),
        "scheduler_state_dict": diff_scheduler.state_dict() if hasattr(diff_scheduler, 'state_dict') else None,
        "learning_rate":        config.lr,
    }

    save_path = os.path.join(config.output_dir, "qwen_lora_best.pt")
    torch.save(ckpt_data, save_path)
    logger.info(f"LoRA guardado en: {save_path} (época {epoch}, val_loss={best_loss:.6f})")

    if extra_name:
        ckpt_dir   = os.path.join(config.output_dir, "checkpoints")
        os.makedirs(ckpt_dir, exist_ok=True)
        named_path = os.path.join(ckpt_dir, f"{extra_name}_epoch{epoch:03d}.pt")
        torch.save(ckpt_data, named_path)
        logger.info(f"Copia guardada en: {named_path}")

def main():
    config = parse_args()
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    dtype  = torch.bfloat16

    logger.info(f"Training: 1 GPU ({device}), BS={config.batch_size}")
    logger.info(f"Experimento: {config.experiment_json}")
    logger.info(f"Heatmap Loss Weight: {config.heatmap_loss_weight}")
    logger.info(f"Experimento: {config.experiment_json}")
    logger.info(f"Velocity Loss Weight: {config.velocity_loss_weight}")
    logger.info(f"Heatmap Loss Type: {config.heatmap_loss_type}")


    os.makedirs(config.output_dir, exist_ok=True)

    #modelo
    model = build_model(config, device)

    

    trainable_params = [p for p in model.parameters() if p.requires_grad]
    logger.info(f"{len(trainable_params)} grupos de parámetros entrenables")


    optimizer = torch.optim.AdamW(trainable_params, lr=config.lr, eps=1e-6)

    diff_scheduler = FlowMatchEulerDiscreteScheduler.from_pretrained(
        config.base_model, subfolder="scheduler"
    )

    #Resume checkpoint si existe
    start_epoch, best_loss = load_checkpoint(model, config, device, optimizer, diff_scheduler)

    # Datasets y dataloaders desde JSON de experimento
    train_dataset, val_dataset, test_dataset, \
    train_dataloader, val_dataloader, test_dataloader = build_dataloaders(config)

    # VAE
    logger.info(f"Cargando VAE en {device}...")
    vae = AutoencoderKLQwenImage.from_pretrained(
        config.base_model, subfolder="vae", torch_dtype=torch.float32
    ).to(device)
    vae.requires_grad_(False)
    vae.eval()

    # HRNet
    hrnet = None
    if os.path.exists(config.hrnet_model_path):
        logger.info(f"Cargando HRNet desde {config.hrnet_model_path}...")
        hrnet = load_hrnet_model(config.hrnet_model_path, width=48, num_joints=17, device=device)
        hrnet.requires_grad_(False)
        hrnet.eval()
        logger.info("HRNet cargado y congelado.")
    else:
        logger.warning("HRNet no encontrado, solo velocity loss.")

    # Resolución inferida del primer .pt
    first_sample   = torch.load(train_dataset.files[0], weights_only=True)
    img_resolution = first_sample.get("resolution", 1024)
    logger.info(f"Resolución: {img_resolution}")

    loss_fn = CombinedLossFn(
        vae=vae, hrnet=hrnet,
        hrnet_input_size=config.hrnet_input_size,
        img_height=img_resolution, img_width=img_resolution,
        velocity_weight=config.velocity_loss_weight,
        heatmap_weight=config.heatmap_loss_weight,
        heatmap_loss_type=config.heatmap_loss_type,
        save_dir=os.path.join(config.output_dir, "loss_diagnostics"),
    )

    logger.info("Listo para entrenar.")
    model.train()


    # CSV: cabecera solo si empezamos desde cero; si reanudamos, abrimos en modo append
    csv_file_path = os.path.join(config.output_dir, "training_metrics.csv")
    csv_mode = "a" if start_epoch > 0 else "w"
    with open(csv_file_path, mode=csv_mode, newline="") as f:
        writer = csv.writer(f)
        if csv_mode == "w":
            writer.writerow(["epoch", "train_loss", "val_loss", "train_pck", "val_pck",
                 "train_velocity_loss", "train_heatmap_loss",
                 "val_velocity_loss", "val_heatmap_loss"])


        config_path = os.path.join(config.output_dir, "experiment_config.json")
        if not os.path.exists(config_path):  # no sobreescribir si se reanuda
            with open(config_path, "w") as f:
                json.dump(asdict(config), f, indent=2)
            logger.info(f"Configuración guardada en {config_path}")


    # Nombre del experimento para el checkpoint (sin extensión)
    exp_name = os.path.splitext(os.path.basename(config.experiment_json))[0]

    # -------------------------------------------------------------------------
    # Bucle de entrenamiento
    # -------------------------------------------------------------------------
    for epoch in range(start_epoch, config.epochs):
        logger.info(f"Epoch {epoch} start")
        #TRAINING

        iterator = tqdm(train_dataloader, desc=f"Epoch {epoch}")
      
        avg_loss, avg_pck, avg_velocity_loss, avg_heatmap_loss, steps = 0.0, 0.0, 0.0, 0.0, 0

        for step, batch in enumerate(iterator):
            optimizer.zero_grad()

            target = batch["target_latents_packed"].to(device, dtype=dtype)
            source = batch["source_latents_packed"].to(device, dtype=dtype)
            prompt = batch["prompt_embeds"].to(device, dtype=dtype)
            mask   = batch["prompt_embeds_mask"].to(device)
            target_heatmaps = batch["target_heatmaps"]

            bsz   = target.shape[0]
            g_seed = 42 + epoch * 10000 + step
            gen   = torch.Generator(device=device).manual_seed(g_seed)

            timesteps = torch.randint(
                0, diff_scheduler.config.num_train_timesteps,
                (bsz,), generator=gen, device=device
            ).long()
            noise = torch.randn(target.shape, generator=gen, device=device, dtype=dtype)

            t_norm = (timesteps.float() / diff_scheduler.config.num_train_timesteps).to(dtype).view(-1, 1, 1)
            noisy  = (1.0 - t_norm) * target + t_norm * noise

            velocity_target = noise - target
            timestep_norm   = (timesteps.float() / 1000.0).to(dtype)

            latent_model_input = torch.cat([noisy, source], dim=1)
            v_pred = model(latent_model_input, prompt, mask, timestep_norm)

            loss_fn.set_batch_context(target_heatmaps, timesteps)
            combined_target = torch.stack([velocity_target, noisy], dim=1)
            loss = loss_fn(v_pred, combined_target)

            loss.backward()

            ####################### Para revisar los gradientes antes de clipping ##############################
            total_grad = sum(
                p.grad.abs().sum().item()
                for p in model.parameters()
                if p.requires_grad and p.grad is not None
            )
            params_con_grad = sum(
                1 for p in model.parameters()
                if p.requires_grad and p.grad is not None and p.grad.abs().sum().item() > 0
            )
            total_params = sum(1 for p in model.parameters() if p.requires_grad)
            mem_allocated = torch.cuda.memory_allocated(device) / 1e9
            mem_reserved  = torch.cuda.memory_reserved(device) / 1e9
            mem_peak      = torch.cuda.max_memory_allocated(device) / 1e9
            logger.info(
                f"[Epoch {epoch} Step {step}] "
                f"Grad: {total_grad:.6f} | Params grad≠0: {params_con_grad}/{total_params} | "
                f"VRAM: {mem_allocated:.2f}/{mem_reserved:.2f}GB | Peak: {mem_peak:.2f}GB"
            )

            # Resetear el pico para el siguiente epoch
            torch.cuda.reset_peak_memory_stats(device)

            ####################################################################################################
            
            
            torch.nn.utils.clip_grad_norm_(
                [p for p in model.parameters() if p.requires_grad], max_norm=1.0
            )
            optimizer.step()

            avg_loss += loss.item()
            avg_pck  += loss_fn.last_pck
            avg_velocity_loss += loss_fn.last_velocity_loss
            avg_heatmap_loss  += loss_fn.last_heatmap_loss
            steps    += 1


        # Validación
        logger.info("Ejecutando validación...")
        val_loss, val_pck, val_vel_loss, val_heat_loss = validate(
            model=model, val_dataloader=val_dataloader,
            loss_fn=loss_fn, diff_scheduler=diff_scheduler,
            device=device, dtype=dtype,
        )

        global_avg_loss = avg_loss / steps if steps > 0 else 0.0
        global_avg_pck  = avg_pck  / steps if steps > 0 else 0.0

        logger.info(
            f"Epoch {epoch} | Train Loss: {global_avg_loss:.6f} | Val Loss: {val_loss:.6f} | "
            f"Train PCK: {global_avg_pck:.4f} | Val PCK: {val_pck:.4f} | Best: {best_loss:.6f}"
        )

        with open(csv_file_path, mode="a", newline="") as f:
            csv.writer(f).writerow([
            epoch, global_avg_loss, val_loss,
            global_avg_pck, val_pck,
            avg_velocity_loss / steps, avg_heatmap_loss / steps,
            val_vel_loss, val_heat_loss
        ])

        if val_loss < best_loss:
            best_loss = val_loss
            logger.info("🟢 Nueva mejor val_loss. Guardando checkpoint...")
            save_checkpoint(model, config, epoch, best_loss, optimizer, diff_scheduler, extra_name=exp_name)
        else:
            logger.info("⚪ Val_loss no mejoró.")

        # Callback de Inferencia sobre test (no sobre train)
        if config.inference_every > 0 and (epoch + 1) % config.inference_every == 0:
            run_inference_callback(
                model=model, vae=vae, base_scheduler=diff_scheduler,
                device=device, dataset=test_dataset,
                save_dir=config.output_dir, epoch=epoch,
                img_height=img_resolution, img_width=img_resolution,
                num_steps=config.inference_steps,
                num_samples=config.inference_samples,
                dtype=dtype,
            )
        # Siempre al final de cada época, copia nombrada en checkpoints/
        ckpt_dir   = os.path.join(config.output_dir, "checkpoints")
        os.makedirs(ckpt_dir, exist_ok=True)
        final_path = os.path.join(ckpt_dir, f"{exp_name}_final_epoch{epoch:03d}.pt")
        local_lora = {k: v.cpu() for k, v in model.state_dict().items() if "lora" in k}
        torch.save({
            "lora_state_dict":      local_lora,
            "epoch":                epoch,
            "best_loss":            best_loss,
            "optimizer_state_dict": optimizer.state_dict(),
            "scheduler_state_dict": diff_scheduler.state_dict() if hasattr(diff_scheduler, 'state_dict') else None,
            "learning_rate":        config.lr,
        }, final_path)
        logger.info(f"Checkpoint de etapa guardado en: {final_path}")


if __name__ == "__main__":
    main()