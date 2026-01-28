import os

os.environ["HF_HOME"] = "/nas/antoniodetoro/qwen/hf_cache"
os.environ["TMPDIR"] = "/nas/antoniodetoro/qwen/tmp"
os.environ["PYTHONNOUSERSITE"] = "1"


import torch
from PIL import Image

import sys
import glob

# 1. Limpieza inicial
torch.cuda.empty_cache()
sys.path.append(os.getcwd())

from qwenimage.pipeline_qwenimage_edit_plus import QwenImageEditPlusPipeline
from qwenimage.transformer_qwenimage import QwenImageTransformer2DModel

# ---------------- CONFIG ----------------
BASE_MODEL = "Qwen/Qwen-Image-Edit-2509"
TRANSFORMER_MODEL = "linoyts/Qwen-Image-Edit-Rapid-AIO"
LORA_REPO = "dx8152/Qwen-Edit-2509-Multiple-angles"
LORA_WEIGHTS = "镜头转换.safetensors"
N_STEPS = 2

INPUT_DIR = "fotos_entrada/new"
OUTPUT_DIR = "fotos_salida"
os.makedirs(OUTPUT_DIR, exist_ok=True)

dtype = torch.bfloat16 
device = "cuda"

# Prompt sugerido (Mezcla Chino/Inglés para mejor activación del LoRA)
prompt = "将相机转向鸟瞰视角 Turn the camera to a bird's-eye view."

print("🚀 Cargando componentes en modo ahorro de memoria...")

try:
    # 2. Cargar Transformer (SIN .to(device))
    transformer = QwenImageTransformer2DModel.from_pretrained(
        TRANSFORMER_MODEL,
        subfolder="transformer",
        torch_dtype=dtype,
        low_cpu_mem_usage=True
    )

    # 3. Cargar Pipeline (SIN .to(device))
    pipe = QwenImageEditPlusPipeline.from_pretrained(
        BASE_MODEL,
        transformer=transformer,
        torch_dtype=dtype,
    )

    # --- ESTRATEGIA DE MEMORIA CLAVE ---
    # En lugar de subir todo a la GPU, esto mueve módulos de RAM a VRAM solo cuando se necesitan.
    pipe.enable_sequential_cpu_offload()
    # ----------------------------------

    # 4. Cargar LoRA (Sin fusionar para evitar picos de VRAM)
    print(f"📸 Aplicando adaptador de ángulos...")
    pipe.load_lora_weights(LORA_REPO, weight_name=LORA_WEIGHTS, adapter_name="angles")
    pipe.set_adapters(["angles"], adapter_weights=[0.9])

    # 5. Bucle de procesamiento
    valid_extensions = ("*.jpg", "*.jpeg", "*.png", "*.webp")
    image_files = []
    for ext in valid_extensions:
        image_files.extend(glob.glob(os.path.join(INPUT_DIR, ext)))

    if not image_files:
        print(f"⚠️ No se encontraron imágenes en {INPUT_DIR}")
        sys.exit()

    print(f"✅ {len(image_files)} imágenes encontradas. Iniciando...")

    for img_path in image_files:
        filename = os.path.basename(img_path)
        print(f"🖼️ Procesando: {filename}...")

        raw_image = Image.open(img_path).convert("RGB")
        
        # Usamos un generador local para mantener consistencia
        generator = torch.Generator(device="cuda").manual_seed(42)

        with torch.inference_mode():
            output = pipe(
                image=[raw_image],
                prompt=prompt,
                num_inference_steps=N_STEPS, 
                true_cfg_scale=1.0,
                generator=generator,
            ).images[0]
            
        # Guardar y limpiar caché de esta iteración
        output.save(os.path.join(OUTPUT_DIR, f"edit_{N_STEPS}_{filename}"))
        torch.cuda.empty_cache()

    print(f"\n✨ ¡PROCESO COMPLETADO! Resultados en: {OUTPUT_DIR}")

except Exception as e:
    print(f"❌ ERROR: {e}")
    import traceback
    traceback.print_exc()