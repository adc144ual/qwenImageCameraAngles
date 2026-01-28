import os
import torch
from PIL import Image
import sys
import glob
import logging

# Configuración de Entorno
os.environ["HF_HOME"] = "/nas/antoniodetoro/qwen/hf_cache"
os.environ["TMPDIR"] = "/nas/antoniodetoro/qwen/tmp"
os.environ["PYTHONNOUSERSITE"] = "1"

# --- CONFIGURACIÓN DE LOGGING ---
LOG_FILE = "proceso_procesamiento.log"
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(LOG_FILE, encoding='utf-8'),
        logging.StreamHandler(sys.stdout)
    ]
)
logger = logging.getLogger(__name__)

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
N_STEPS = 6

# INPUT_DIR = "MultiViewVisibleThermalImagesHPE/test/00_16"
INPUT_DIR = "MultiViewVisibleImagesHPE_100/00_16/to_00_17"
OUTPUT_DIR = "MultiViewVisibleImagesHPE_100/00_16_17/to_00_15"
os.makedirs(OUTPUT_DIR, exist_ok=True)

dtype = torch.bfloat16 
device = "cuda"

# prompt = "将镜头向右旋转180度 Rotate the camera 180 degrees to the right."
prompt = "将镜头向左旋转90度 Rotate the camera 90 degrees to the left."

logger.info("🚀 Cargando componentes en modo ahorro de memoria...")

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

    # ESTRATEGIA DE MEMORIA
    pipe.enable_sequential_cpu_offload()

    # 4. Cargar LoRA
    logger.info(f"📸 Aplicando adaptador de ángulos: {LORA_WEIGHTS}")
    pipe.load_lora_weights(LORA_REPO, weight_name=LORA_WEIGHTS, adapter_name="angles")
    pipe.set_adapters(["angles"], adapter_weights=[0.9])

    # 5. Bucle de procesamiento con filtrado y límite de 100
    valid_extensions = ("*.jpg", "*.jpeg", "*.png", "*.webp")
    all_files = []
    for ext in valid_extensions:
        all_files.extend(glob.glob(os.path.join(INPUT_DIR, ext)))

    # FILTRO: Solo archivos que contengan "rgb" y ORDENADOS alfabéticamente
    image_files = sorted([f for f in all_files if "rgb" in os.path.basename(f).lower()])
    
    # SELECCIÓN: Solo los primeros 100
    image_files = image_files[:100]

    if not image_files:
        logger.warning(f"⚠️ No se encontraron imágenes que contengan 'rgb' en {INPUT_DIR}")
        sys.exit()

    logger.info(f"✅ {len(image_files)} imágenes seleccionadas. Iniciando proceso...")

    for img_path in image_files:
        filename = os.path.basename(img_path)
        logger.info(f"🖼️ Procesando: {filename}")

        try:
            raw_image = Image.open(img_path).convert("RGB")
            
            generator = torch.Generator(device="cuda").manual_seed(42)

            with torch.inference_mode():
                output = pipe(
                    image=[raw_image],
                    prompt=prompt,
                    num_inference_steps=N_STEPS, 
                    true_cfg_scale=1.0,
                    generator=generator,
                ).images[0]
                
            save_path = os.path.join(OUTPUT_DIR, f"edit_{N_STEPS}_{filename}")
            output.save(save_path)
            logger.info(f"💾 Guardado con éxito: {save_path}")
            
            torch.cuda.empty_cache()

        except Exception as e:
            logger.error(f"❌ Error procesando {filename}: {e}")

    logger.info(f"\n✨ ¡PROCESO COMPLETADO! Resultados en: {OUTPUT_DIR}")

except Exception as e:
    logger.critical(f"❌ ERROR CRÍTICO EN EL SCRIPT: {e}")
    import traceback
    logger.error(traceback.format_exc())



# import os
# import torch
# from PIL import Image
# import sys
# import glob
# import logging  # Nueva librería para el log

# # Configuración de Entorno
# os.environ["HF_HOME"] = "/nas/antoniodetoro/qwen/hf_cache"
# os.environ["TMPDIR"] = "/nas/antoniodetoro/qwen/tmp"
# os.environ["PYTHONNOUSERSITE"] = "1"

# # --- CONFIGURACIÓN DE LOGGING ---
# LOG_FILE = "proceso_procesamiento_16.log"
# logging.basicConfig(
#     level=logging.INFO,
#     format='%(asctime)s - %(levelname)s - %(message)s',
#     handlers=[
#         logging.FileHandler(LOG_FILE, encoding='utf-8'),
#         logging.StreamHandler(sys.stdout)
#     ]
# )
# logger = logging.getLogger(__name__)

# # 1. Limpieza inicial
# torch.cuda.empty_cache()
# sys.path.append(os.getcwd())

# from qwenimage.pipeline_qwenimage_edit_plus import QwenImageEditPlusPipeline
# from qwenimage.transformer_qwenimage import QwenImageTransformer2DModel

# # ---------------- CONFIG ----------------
# BASE_MODEL = "Qwen/Qwen-Image-Edit-2509"
# TRANSFORMER_MODEL = "linoyts/Qwen-Image-Edit-Rapid-AIO"
# LORA_REPO = "dx8152/Qwen-Edit-2509-Multiple-angles"
# LORA_WEIGHTS = "镜头转换.safetensors"
# N_STEPS = 6

# INPUT_DIR = "MultiViewVisibleThermalImagesHPE/test/00_16"
# OUTPUT_DIR = "MultiViewVisibleThermalImagesHPE_generated/test_00_16_fake"
# os.makedirs(OUTPUT_DIR, exist_ok=True)

# dtype = torch.bfloat16 
# device = "cuda"

# # prompt = "将相机转向鸟瞰视角 Turn the camera to a bird's-eye view."
# # prompt = "将镜头向右旋转90度 Rotate the camera 90 degrees to the right."
# prompt = "将镜头向右旋转180度 Rotate the camera 180 degrees to the right."
# # prompt= "将镜头向左旋转90度 Rotate the camera 90 degrees to the left."

# logger.info("🚀 Cargando componentes en modo ahorro de memoria...")

# try:
#     # 2. Cargar Transformer
#     transformer = QwenImageTransformer2DModel.from_pretrained(
#         TRANSFORMER_MODEL,
#         subfolder="transformer",
#         torch_dtype=dtype,
#         low_cpu_mem_usage=True
#     )

#     # 3. Cargar Pipeline
#     pipe = QwenImageEditPlusPipeline.from_pretrained(
#         BASE_MODEL,
#         transformer=transformer,
#         torch_dtype=dtype,
#     )

#     # ESTRATEGIA DE MEMORIA
#     pipe.enable_sequential_cpu_offload()

#     # 4. Cargar LoRA
#     logger.info(f"📸 Aplicando adaptador de ángulos: {LORA_WEIGHTS}")
#     pipe.load_lora_weights(LORA_REPO, weight_name=LORA_WEIGHTS, adapter_name="angles")
#     pipe.set_adapters(["angles"], adapter_weights=[0.9])

#     # 5. Bucle de procesamiento con filtrado
#     valid_extensions = ("*.jpg", "*.jpeg", "*.png", "*.webp")
#     all_files = []
#     for ext in valid_extensions:
#         all_files.extend(glob.glob(os.path.join(INPUT_DIR, ext)))

#     # FILTRO: Solo archivos que contengan "rgb" en el nombre (ignorando mayúsculas/minúsculas)
#     image_files = [f for f in all_files if "rgb" in os.path.basename(f).lower()]

#     if not image_files:
#         logger.warning(f"⚠️ No se encontraron imágenes que contengan 'rgb' en {INPUT_DIR}")
#         sys.exit()

#     logger.info(f"✅ {len(image_files)} imágenes aptas encontradas. Iniciando proceso...")

#     for img_path in image_files:
#         filename = os.path.basename(img_path)
#         logger.info(f"🖼️ Procesando: {filename}")

#         try:
#             raw_image = Image.open(img_path).convert("RGB")
            
#             generator = torch.Generator(device="cuda").manual_seed(42)

#             with torch.inference_mode():
#                 output = pipe(
#                     image=[raw_image],
#                     prompt=prompt,
#                     num_inference_steps=N_STEPS, 
#                     true_cfg_scale=1.0,
#                     generator=generator,
#                 ).images[0]
                
#             save_path = os.path.join(OUTPUT_DIR, f"edit_{N_STEPS}_{filename}")
#             output.save(save_path)
#             logger.info(f"💾 Guardado con éxito: {save_path}")
            
#             torch.cuda.empty_cache()

#         except Exception as e:
#             logger.error(f"❌ Error procesando {filename}: {e}")

#     logger.info(f"\n✨ ¡PROCESO COMPLETADO! Resultados en: {OUTPUT_DIR}")

# except Exception as e:
#     logger.critical(f"❌ ERROR CRÍTICO EN EL SCRIPT: {e}")
#     import traceback
#     logger.error(traceback.format_exc())