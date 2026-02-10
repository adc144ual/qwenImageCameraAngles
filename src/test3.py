import torch
from PIL import Image
import os
import sys

# 1. Limpieza inicial de memoria
torch.cuda.empty_cache()

# Asegurar que el path local esté disponible
sys.path.append(os.getcwd())

from qwenimage.pipeline_qwenimage_edit_plus import QwenImageEditPlusPipeline
from qwenimage.transformer_qwenimage import QwenImageTransformer2DModel

# ---------------- CONFIG ----------------
BASE_MODEL = "Qwen/Qwen-Image-Edit-2509"
TRANSFORMER_MODEL = "linoyts/Qwen-Image-Edit-Rapid-AIO"
LORA_REPO = "dx8152/Qwen-Edit-2509-Multiple-angles"
LORA_WEIGHTS = "镜头转换.safetensors"

input_image_path = "disaster_girl.jpg"
output_image_path = "output_result.png"

# Cambiamos a bfloat16: es la clave para evitar imágenes negras en V100/A30
dtype = torch.bfloat16 
device = "cuda"

# --- VERIFICACIÓN DE GPU ---
if torch.cuda.is_available():
    vram = torch.cuda.get_device_properties(0).total_memory / 1e9
    print(f"🖥️ Utilizando GPU: {torch.cuda.get_device_name(0)} con {vram:.2f} GB de VRAM")
# ---------------------------

print("🚀 Cargando componentes en modo bfloat16...")

try:
    # 2. Cargar Transformer
    transformer = QwenImageTransformer2DModel.from_pretrained(
        TRANSFORMER_MODEL,
        subfolder="transformer",
        torch_dtype=dtype,
        low_cpu_mem_usage=True
    )

    # 3. Cargar Pipeline
    pipe = QwenImageEditPlusPipeline.from_pretrained(
        BASE_MODEL,
        transformer=transformer,
        torch_dtype=dtype,
    )

    # 4. Sequential Offload (Mantiene el uso de VRAM bajo control)
    pipe.enable_sequential_cpu_offload()

    # 5. Cargar LoRA de ángulos
    print(f"📸 Aplicando adaptador de ángulos...")
    pipe.load_lora_weights(
        LORA_REPO, 
        weight_name=LORA_WEIGHTS, 
        adapter_name="angles"
    )
    
    # Activamos el adaptador con un peso ligeramente menor para evitar artefactos
    pipe.set_adapters(["angles"], adapter_weights=[0.9])

    print("✅ Pipeline preparada. Iniciando generación...")

    # 6. Función de inferencia optimizada
    def run_edit(img_path, prompt):
        if not os.path.exists(img_path):
            raise FileNotFoundError(f"No se encuentra la imagen: {img_path}")
            
        raw_image = Image.open(img_path).convert("RGB")
        
        with torch.inference_mode():
            # Aumentamos steps a 12 para mejorar la calidad y evitar el negro
            # true_cfg_scale=1.0 es importante para este modelo específico
            result = pipe(
                image=[raw_image],
                prompt=prompt,
                num_inference_steps=12, 
                true_cfg_scale=1.0,
                generator=torch.Generator(device="cuda").manual_seed(42),
            ).images[0]
        return result

    # Prompt sugerido: Chino + Inglés ayuda al modelo a entender el LoRA
    # prompt = "将相机切换到仰视视角 Turn the camera to a worm's-eye view, low angle shot."
    prompt= "将相机转向鸟瞰视角 Turn the camera to a bird's-eye view."
    
    final_image = run_edit(input_image_path, prompt)
    
    # 7. Guardar y verificar
    final_image.save(output_image_path)
    print(f"✨ ¡PROCESO COMPLETADO!")
    print(f"📁 Imagen guardada en: {os.path.abspath(output_image_path)}")

except Exception as e:
    print(f"❌ ERROR: {e}")
    import traceback
    traceback.print_exc()