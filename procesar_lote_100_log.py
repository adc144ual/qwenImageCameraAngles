# ============================== Se generan X sólo desde la 17 a las 15 y 16, y empezando desde W imagen tras haber ordenado=======================================

import os
import torch
from PIL import Image
import sys
import glob
import logging

# --- CONFIGURACIÓN DE ENTORNO ---
os.environ["HF_HOME"] = "/nas/antoniodetoro/qwen/hf_cache"
os.environ["TMPDIR"] = "/nas/antoniodetoro/qwen/tmp"
os.environ["PYTHONNOUSERSITE"] = "1"

LOG_FILE = "proceso_multiview_00_17.log"
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[logging.FileHandler(LOG_FILE, encoding='utf-8'), logging.StreamHandler(sys.stdout)]
)
logger = logging.getLogger(__name__)

# --- PARÁMETROS DE FILTRADO (MODIFICA AQUÍ) ---
ORIGIN_VIEW = "00_17"                 # Vista de origen fija
NUM_IMAGES_TO_PROCESS = 10            # X: Cuántas imágenes generar
START_IMAGE_Y = "1680259600117_rgb"   # Y: Nombre base (sin 00_17_ ni extensión)

# --- CONFIGURACIÓN MODELOS ---
BASE_MODEL = "Qwen/Qwen-Image-Edit-2509"
TRANSFORMER_MODEL = "linoyts/Qwen-Image-Edit-Rapid-AIO"
LORA_REPO = "dx8152/Qwen-Edit-2509-Multiple-angles"
LORA_WEIGHTS = "镜头转换.safetensors"
N_STEPS = 6

# --- CONFIGURACIÓN RUTAS ---
BASE_INPUT_DIR = "MultiViewVisibleThermalImagesHPE/test"
BASE_OUTPUT_DIR = "MultiViewVisibleImagesHPE_Custom"
SUB_DIRS = ["00_15", "00_16", "00_17"]

# --- LÓGICA DE TRANSFORMACIÓN ---
TRANSFORMATIONS = {
    "00_17": {
        "00_16": "将镜头向右旋转90度 Rotate the camera 90 degrees to the right.",
        "00_15": "将镜头向左旋转90度 Rotate the camera 90 degrees to the left."
    }
}

def get_base_name(filename):
    """
    Extrae la parte común del nombre. 
    Ejemplo: '00_17_1680259600117_rgb.png' -> '1680259600117_rgb'
    """
    # Quitamos la extensión primero
    name_without_ext = os.path.splitext(filename)[0]
    parts = name_without_ext.split('_')
    # Si empieza por 00_XX_, devolvemos lo que sigue
    return "_".join(parts[2:]) if len(parts) > 2 else name_without_ext

def main():
    try:
        # 1. Filtrado y Sincronización
        logger.info(f"🔍 Sincronizando archivos desde {ORIGIN_VIEW}...")
        files_per_folder = {}
        
        for sd in SUB_DIRS:
            path = os.path.join(BASE_INPUT_DIR, sd)
            all_f = []
            for ext in ("*.jpg", "*.png", "*.jpeg"):
                all_f.extend(glob.glob(os.path.join(path, ext)))
            
            # Mapeamos { "nombre_base": "nombre_archivo_real.ext" }
            # Ejemplo: { "1680259600117_rgb": "00_17_1680259600117_rgb.png" }
            files_per_folder[sd] = {get_base_name(os.path.basename(f)): os.path.basename(f) for f in all_f if "rgb" in f.lower()}

        # Encontrar bases comunes y ordenar alfabéticamente
        common_bases = sorted(list(set(files_per_folder["00_15"].keys()) & 
                                   set(files_per_folder["00_16"].keys()) & 
                                   set(files_per_folder["00_17"].keys())))

        # 2. Selección de rango (X imágenes desde Y)
        if START_IMAGE_Y not in common_bases:
            logger.error(f"❌ No se encontró '{START_IMAGE_Y}' en los archivos comunes.")
            logger.info(f"Sugerencia: Revisa que el nombre no incluya '00_17_' ni '.png/.jpg'")
            return

        start_idx = common_bases.index(START_IMAGE_Y)
        selected_bases = common_bases[start_idx : start_idx + NUM_IMAGES_TO_PROCESS]

        logger.info(f"✅ Se procesarán {len(selected_bases)} imágenes desde la posición {start_idx}.")

        # 3. Carga de IA
        logger.info("🚀 Cargando modelos en GPU...")
        from qwenimage.pipeline_qwenimage_edit_plus import QwenImageEditPlusPipeline
        from qwenimage.transformer_qwenimage import QwenImageTransformer2DModel
        
        torch.cuda.empty_cache()
        transformer = QwenImageTransformer2DModel.from_pretrained(TRANSFORMER_MODEL, subfolder="transformer", torch_dtype=torch.bfloat16, low_cpu_mem_usage=True)
        pipe = QwenImageEditPlusPipeline.from_pretrained(BASE_MODEL, transformer=transformer, torch_dtype=torch.bfloat16)
        pipe.enable_sequential_cpu_offload()
        pipe.load_lora_weights(LORA_REPO, weight_name=LORA_WEIGHTS, adapter_name="angles")
        pipe.set_adapters(["angles"], adapter_weights=[0.9])

        # 4. Bucle de Procesamiento
        for base_name in selected_bases:
            real_filename = files_per_folder[ORIGIN_VIEW][base_name]
            img_path = os.path.join(BASE_INPUT_DIR, ORIGIN_VIEW, real_filename)
            
            raw_image = Image.open(img_path).convert("RGB")
            
            destinos = ["00_15", "00_16"]
            for target in destinos:
                prompt = TRANSFORMATIONS[ORIGIN_VIEW][target]
                
                # Carpeta de salida organizada
                save_dir = os.path.join(BASE_OUTPUT_DIR, ORIGIN_VIEW, f"to_{target}")
                os.makedirs(save_dir, exist_ok=True)
                
                logger.info(f"🎨 Generando: {real_filename} -> Vista {target}")

                with torch.inference_mode():
                    output = pipe(
                        image=[raw_image],
                        prompt=prompt,
                        num_inference_steps=N_STEPS,
                        true_cfg_scale=1.0,
                        generator=torch.Generator(device="cuda").manual_seed(42),
                    ).images[0]
                
                # Guardamos manteniendo una referencia al original
                save_name = f"Step_{N_STEPS}_fake_{target}_from_{real_filename}"
                output.save(os.path.join(save_dir, save_name))
                torch.cuda.empty_cache()

        logger.info(f"✨ ¡PROCESO COMPLETADO! Revisa la carpeta: {BASE_OUTPUT_DIR}")

    except Exception as e:
        logger.critical(f"❌ ERROR CRÍTICO: {e}")
        import traceback
        logger.error(traceback.format_exc())

if __name__ == "__main__":
    main()



# ======================== se generan 100 de todas a todas =====================================
# import os
# import torch
# from PIL import Image
# import sys
# import glob
# import logging

# # --- CONFIGURACIÓN DE ENTORNO ---
# os.environ["HF_HOME"] = "/nas/antoniodetoro/qwen/hf_cache"
# os.environ["TMPDIR"] = "/nas/antoniodetoro/qwen/tmp"
# os.environ["PYTHONNOUSERSITE"] = "1"

# LOG_FILE = "proceso_multiview_sincronizado.log"
# logging.basicConfig(
#     level=logging.INFO,
#     format='%(asctime)s - %(levelname)s - %(message)s',
#     handlers=[logging.FileHandler(LOG_FILE, encoding='utf-8'), logging.StreamHandler(sys.stdout)]
# )
# logger = logging.getLogger(__name__)

# # --- CONFIGURACIÓN MODELOS ---
# BASE_MODEL = "Qwen/Qwen-Image-Edit-2509"
# TRANSFORMER_MODEL = "linoyts/Qwen-Image-Edit-Rapid-AIO"
# LORA_REPO = "dx8152/Qwen-Edit-2509-Multiple-angles"
# LORA_WEIGHTS = "镜头转换.safetensors"
# N_STEPS = 6

# # --- CONFIGURACIÓN RUTAS ---
# BASE_INPUT_DIR = "MultiViewVisibleThermalImagesHPE/test"
# BASE_OUTPUT_DIR = "MultiViewVisibleImagesHPE_100"
# SUB_DIRS = ["00_15", "00_16", "00_17"]

# # --- LÓGICA DE TRANSFORMACIÓN ---
# # Estructura: [Origen][Destino] = (Prompt en Chino/Inglés)
# TRANSFORMATIONS = {
#     "00_15": {
#         "00_16": "将镜头向左旋转180度 Rotate the camera 180 degrees to the left.",
#         "00_17": "将镜头向右旋转90度 Rotate the camera 90 degrees to the right."
#     },
#     "00_16": {
#         "00_15": "将镜头向右旋转180度 Rotate the camera 180 degrees to the right.", # Inverso de 15->16
#         "00_17": "将镜头向左旋转90度 Rotate the camera 90 degrees to the left."
#     },
#     "00_17": {
#         "00_15": "将镜头向右旋转90度 Rotate the camera 90 degrees to the right.",
#         "00_16": "将镜头向左旋转90度 Rotate the camera 90 degrees to the left."
#     }
# }

# def get_base_name(filename):
#     # Elimina el prefijo 00_XX_ y devuelve el resto
#     parts = filename.split('_')
#     return "_".join(parts[2:]) if len(parts) > 2 else filename

# def main():
#     try:
#         # 1. Filtrado y Sincronización
#         logger.info("🔍 Sincronizando archivos entre carpetas...")
#         files_per_folder = {}
#         for sd in SUB_DIRS:
#             path = os.path.join(BASE_INPUT_DIR, sd)
#             all_f = []
#             for ext in ("*.jpg", "*.png", "*.jpeg"):
#                 all_f.extend(glob.glob(os.path.join(path, ext)))
            
#             # Filtrar por RGB, obtener nombre base y ordenar
#             rgb_files = sorted([os.path.basename(f) for f in all_f if "rgb" in os.path.basename(f).lower()])
#             files_per_folder[sd] = {get_base_name(f): f for f in rgb_files}

#         # Encontrar la intersección de las 3 carpetas (archivos comunes)
#         common_bases = sorted(list(set(files_per_folder["00_15"].keys()) & 
#                                 set(files_per_folder["00_16"].keys()) & 
#                                 set(files_per_folder["00_17"].keys())))[:100]

#         if len(common_bases) < 100:
#             logger.warning(f"⚠️ Solo se encontraron {len(common_bases)} archivos comunes. Se procesarán todos.")
#         else:
#             logger.info(f"✅ Se han seleccionado los primeros 100 archivos comunes.")

#         # 2. Carga de IA
#         logger.info("🚀 Cargando modelos...")
#         from qwenimage.pipeline_qwenimage_edit_plus import QwenImageEditPlusPipeline
#         from qwenimage.transformer_qwenimage import QwenImageTransformer2DModel
        
#         torch.cuda.empty_cache()
#         transformer = QwenImageTransformer2DModel.from_pretrained(TRANSFORMER_MODEL, subfolder="transformer", torch_dtype=torch.bfloat16, low_cpu_mem_usage=True)
#         pipe = QwenImageEditPlusPipeline.from_pretrained(BASE_MODEL, transformer=transformer, torch_dtype=torch.bfloat16)
#         pipe.enable_sequential_cpu_offload()
#         pipe.load_lora_weights(LORA_REPO, weight_name=LORA_WEIGHTS, adapter_name="angles")
#         pipe.set_adapters(["angles"], adapter_weights=[0.9])

#         # 3. Bucle de Procesamiento (100 archivos * 3 carpetas origen * 2 destinos cada una = 600 imágenes)
#         for base_name in common_bases:
#             for origin in SUB_DIRS:
#                 # Obtener el nombre de archivo real en esa carpeta (ej: 00_15_rgb_001.jpg)
#                 real_filename = files_per_folder[origin][base_name]
#                 img_path = os.path.join(BASE_INPUT_DIR, origin, real_filename)
                
#                 raw_image = Image.open(img_path).convert("RGB")
                
#                 # Procesar hacia los otros dos destinos
#                 destinos = [d for d in SUB_DIRS if d != origin]
#                 for target in destinos:
#                     prompt = TRANSFORMATIONS[origin][target]
                    
#                     # Crear estructura de carpetas: OUTPUT/ORIGEN/TARGET_TRANSFORM/
#                     save_dir = os.path.join(BASE_OUTPUT_DIR, origin, f"to_{target}")
#                     os.makedirs(save_dir, exist_ok=True)
                    
#                     logger.info(f"🎨 {origin} -> {target} | Archivo: {real_filename}")

#                     with torch.inference_mode():
#                         output = pipe(
#                             image=[raw_image],
#                             prompt=prompt,
#                             num_inference_steps=N_STEPS,
#                             true_cfg_scale=1.0,
#                             generator=torch.Generator(device="cuda").manual_seed(42),
#                         ).images[0]
                    
#                     save_path = os.path.join(save_dir, f"Step_{N_STEPS}_fake_{target}_from_{real_filename}")
#                     output.save(save_path)
#                     torch.cuda.empty_cache()

#         logger.info(f"✨ ¡PROCESO COMPLETADO! 600 imágenes generadas en {BASE_OUTPUT_DIR}")

#     except Exception as e:
#         logger.critical(f"❌ ERROR: {e}")
#         import traceback
#         logger.error(traceback.format_exc())

        
# if __name__ == "__main__":
#     main()
