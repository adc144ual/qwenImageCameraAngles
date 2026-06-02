# evaluate.py
import os, sys, argparse, logging
current_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.dirname(current_dir))
sys.path.append(current_dir)

os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["HF_HOME"] = "/nas/antoniodetoro/qwen/hf_cache"
os.environ["TMPDIR"] = "/dev/shm"
os.environ["PYTHONNOUSERSITE"] = "1"

import torch
from diffusers import FlowMatchEulerDiscreteScheduler, QwenImageTransformer2DModel
from diffusers.models import AutoencoderKLQwenImage
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
from transformers import BitsAndBytesConfig

from config import TrainingConfig
from dataset import build_dataloaders
from hrnet import load_hrnet_model
from loss import CombinedLossFn
from model import QwenSingleGPUWrapper
from validate import validate

logging.basicConfig(level=logging.INFO, force=True, format="%(message)s")
logger = logging.getLogger(__name__)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint",        type=str, required=True)
    parser.add_argument("--experiment_json",   type=str, required=True)
    parser.add_argument("--latents_dir",       type=str, required=True)
    parser.add_argument("--base_model",        type=str, default="Qwen/Qwen-Image-Edit-2509")
    parser.add_argument("--transformer_model", type=str, default="../models/Qwen-Fused-Angles")
    parser.add_argument("--hrnet_model_path",  type=str, default="./models/pose_hrnet_w48_384x288.pth")
    parser.add_argument("--heatmap_loss_weight",  type=float, default=0.5)
    parser.add_argument("--velocity_loss_weight", type=float, default=0.5)
    parser.add_argument("--heatmap_loss_type",    type=str,   default="mse")
    parser.add_argument("--batch_size",        type=int,   default=2)
    parser.add_argument("--lora_rank",         type=int,   default=16)
    parser.add_argument("--lora_alpha",        type=int,   default=32)
    parser.add_argument("--lora_dropout",      type=float, default=0.1)
    return parser.parse_args()


def main():
    args   = parse_args()
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    dtype  = torch.bfloat16

    # Config mínima para build_dataloaders (solo necesita test split)
    config = TrainingConfig(
        latents_dir=args.latents_dir,
        experiment_json=args.experiment_json,
        output_dir=os.path.dirname(args.checkpoint),
        base_model=args.base_model,
        transformer_model=args.transformer_model,
        hrnet_model_path=args.hrnet_model_path,
        velocity_loss_weight=args.velocity_loss_weight,
        heatmap_loss_weight=args.heatmap_loss_weight,
        heatmap_loss_type=args.heatmap_loss_type,
        batch_size=args.batch_size,
        lora_rank=args.lora_rank,
        lora_alpha=args.lora_alpha,
        lora_dropout=args.lora_dropout,
    )

    # Solo test dataloader
    _, _, test_dataset, _, _, test_dataloader = build_dataloaders(config)

    first_sample   = torch.load(test_dataset.files[0], weights_only=True)
    img_resolution = first_sample.get("resolution", 1024)

    # Modelo
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True, bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16, bnb_4bit_use_double_quant=True,
    )
    transformer = QwenImageTransformer2DModel.from_pretrained(
        config.transformer_model, subfolder=None,
        quantization_config=bnb_config, torch_dtype=torch.bfloat16,
        device_map=str(device), attn_implementation="sdpa", local_files_only=True,
    )
    transformer = prepare_model_for_kbit_training(transformer, use_gradient_checkpointing=False)
    lora_config = LoraConfig(
        r=config.lora_rank, lora_alpha=config.lora_alpha,
        init_lora_weights=True,
        target_modules=["to_q", "to_k", "to_v", "to_out.0",
                        "add_q_proj", "add_k_proj", "add_v_proj", "to_add_out"],
        lora_dropout=config.lora_dropout,
    )
    transformer = get_peft_model(transformer, lora_config)
    model = QwenSingleGPUWrapper(transformer)

    # Cargar checkpoint
    ckpt = torch.load(args.checkpoint, map_location=device, weights_only=False)
    lora_state = ckpt["lora_state_dict"] if "lora_state_dict" in ckpt else ckpt
    missing, unexpected = model.load_state_dict(lora_state, strict=False)
    logger.info(f"Checkpoint cargado: {args.checkpoint}")
    logger.info(f"  LoRA keys: {len(lora_state)} | missing: {len(missing)} | unexpected: {len(unexpected)}")

    model.eval()

    # VAE
    vae = AutoencoderKLQwenImage.from_pretrained(
        args.base_model, subfolder="vae", torch_dtype=torch.float32
    ).to(device)
    vae.requires_grad_(False).eval()

    # HRNet
    hrnet = None
    if os.path.exists(args.hrnet_model_path):
        hrnet = load_hrnet_model(args.hrnet_model_path, width=48, num_joints=17, device=device)
        hrnet.requires_grad_(False).eval()

    diff_scheduler = FlowMatchEulerDiscreteScheduler.from_pretrained(
        args.base_model, subfolder="scheduler"
    )

    loss_fn = CombinedLossFn(
        vae=vae, hrnet=hrnet,
        hrnet_input_size=config.hrnet_input_size,
        img_height=img_resolution, img_width=img_resolution,
        velocity_weight=config.velocity_loss_weight,
        heatmap_weight=config.heatmap_loss_weight,
        heatmap_loss_type=config.heatmap_loss_type,
    )

    test_loss, test_pck, test_vel, test_heat = validate(
        model=model, val_dataloader=test_dataloader,
        loss_fn=loss_fn, diff_scheduler=diff_scheduler,
        device=device, dtype=dtype,
    )
    logger.info(f"TEST | Loss: {test_loss:.6f} | PCK: {test_pck:.4f} | Vel: {test_vel:.6f} | Heatmap: {test_heat:.6f}")
    import csv
    csv_path = os.path.join(os.path.dirname(args.checkpoint), "test_results.csv")
    with open(csv_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["checkpoint", "test_loss", "test_pck", "vel_loss", "heatmap_loss"])
        writer.writerow([args.checkpoint, test_loss, test_pck, test_vel, test_heat])
    logger.info(f"Resultados guardados en {csv_path}")

if __name__ == "__main__":
    main()