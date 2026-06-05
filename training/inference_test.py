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
from torch.utils.data import DataLoader
from diffusers import FlowMatchEulerDiscreteScheduler, QwenImageTransformer2DModel
from diffusers.models import AutoencoderKLQwenImage
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
from transformers import BitsAndBytesConfig

from config import TrainingConfig
from dataset import LatentsDatasetFromJSON, load_experiment_json, make_collate_latents
from model import QwenSingleGPUWrapper
from inference import run_inference_callback

logging.basicConfig(level=logging.INFO, force=True, format="%(message)s")
logger = logging.getLogger(__name__)


def build_test_dataloader(config):
    _, _, test_ts = load_experiment_json(config.experiment_json)
    test_dataset  = LatentsDatasetFromJSON(config.latents_dir, test_ts, split="test")
    collate_fn    = make_collate_latents(test_dataset.global_max_seq_len)
    test_dataloader = DataLoader(
        test_dataset,
        batch_size=config.batch_size,
        collate_fn=collate_fn,
        drop_last=False,
        shuffle=False,
        num_workers=4,
    )
    return test_dataset, test_dataloader


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint",        type=str, required=True)
    parser.add_argument("--experiment_json",   type=str, required=True)
    parser.add_argument("--latents_dir",       type=str, required=True)
    parser.add_argument("--base_model",        type=str, default="Qwen/Qwen-Image-Edit-2509")
    parser.add_argument("--transformer_model", type=str, default="../models/Qwen-Fused-Angles")
    parser.add_argument("--batch_size",        type=int,   default=4)
    parser.add_argument("--lora_rank",         type=int,   default=16)
    parser.add_argument("--lora_alpha",        type=int,   default=32)
    parser.add_argument("--lora_dropout",      type=float, default=0.1)
    parser.add_argument("--inference_steps",   type=int,   default=6)
    return parser.parse_args()


def main():
    args   = parse_args()
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    dtype  = torch.bfloat16

    checkpoint_dir = os.path.dirname(args.checkpoint)
    inference_dir  = os.path.join(checkpoint_dir, "inference_results_test")
    os.makedirs(inference_dir, exist_ok=True)

    config = TrainingConfig(
        latents_dir=args.latents_dir,
        experiment_json=args.experiment_json,
        output_dir=checkpoint_dir,
        base_model=args.base_model,
        transformer_model=args.transformer_model,
        batch_size=args.batch_size,
        lora_rank=args.lora_rank,
        lora_alpha=args.lora_alpha,
        lora_dropout=args.lora_dropout,
    )

    test_dataset, test_dataloader = build_test_dataloader(config)

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

    ckpt = torch.load(args.checkpoint, map_location=device, weights_only=False)
    lora_state = ckpt["lora_state_dict"] if "lora_state_dict" in ckpt else ckpt
    missing, unexpected = model.load_state_dict(lora_state, strict=False)
    logger.info(f"Checkpoint cargado: {args.checkpoint}")
    logger.info(f"  LoRA keys: {len(lora_state)} | missing: {len(missing)} | unexpected: {len(unexpected)}")

    model.eval()

    vae = AutoencoderKLQwenImage.from_pretrained(
        args.base_model, subfolder="vae", torch_dtype=torch.float32
    ).to(device)
    vae.requires_grad_(False).eval()


    diff_scheduler = FlowMatchEulerDiscreteScheduler.from_pretrained(
        args.base_model, subfolder="scheduler"
    )

    # Inferencia
    logger.info(f"Generando imágenes de inferencia en {inference_dir}...")
    run_inference_callback(
        model=model, vae=vae, base_scheduler=diff_scheduler,
        device=device, dataset=test_dataset,
        save_dir=inference_dir, epoch=0,
        img_height=img_resolution, img_width=img_resolution,
        num_steps=args.inference_steps,
        num_samples=len(test_dataset),
        dtype=dtype,
    )
    logger.info("Inferencia completada.")


if __name__ == "__main__":
    main()