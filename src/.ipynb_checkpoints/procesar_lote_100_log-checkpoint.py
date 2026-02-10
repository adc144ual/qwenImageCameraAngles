import os
import torch
from PIL import Image
import sys
import glob
import logging
import pandas as pd  # Añadido para manejo de CSV

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))


# --- CONFIGURACIÓN DE ENTORNO ---
os.environ["HF_HOME"] = "/nas/antoniodetoro/qwen/hf_cache"
os.environ["TMPDIR"] = "/nas/antoniodetoro/qwen/tmp"
os.environ["PYTHONNOUSERSITE"] = "1"

LOG_FILE = "proceso_multiview_frontal.log"
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[logging.FileHandler(LOG_FILE, encoding='utf-8'), logging.StreamHandler(sys.stdout)]
)
logger = logging.getLogger(__name__)

# --- PARÁMETROS DE FILTRADO ---
ORIGIN_VIEW = "00_17"
CSV_FRONTALIDAD = "resultados_frontalidad.csv"
NUM_IMAGES_TO_PROCESS = None  # Pon None para procesar todas las frontales

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

TRANSFORMATIONS = {
    "00_17": {
        "00_16": "将镜头向右旋转90度 Rotate the camera 90 degrees to the right.",
        "00_15": "将镜头向左旋转90度 Rotate the camera 90 degrees to the left."
    }
}

def get_base_name(filename):
    name_without_ext = os.path.splitext(filename)[0]
    parts = name_without_ext.split('_')
    return "_".join(parts[2:]) if len(parts) > 2 else name_without_ext

def get_frontal_bases(csv_path):
    """Lee el CSV y devuelve una lista de base_names que son frontales."""
    if not os.path.exists(csv_path):
        logger.error(f"❌ No se encuentra el archivo CSV: {csv_path}")
        return []
    
    df = pd.read_csv(csv_path)
    # Filtramos donde Frontal sea SI (ignorando mayúsculas/minúsculas)
    frontal_files = df[df['Frontal'].str.upper() == 'SI']['Imagen'].tolist()
    return [get_base_name(f) for f in frontal_files]

def main():
    try:
        # 1. Obtener bases frontales desde CSV
        logger.info(f"📄 Leyendo frontalidad desde {CSV_FRONTALIDAD}...")
        frontal_bases = get_frontal_bases(CSV_FRONTALIDAD)
        
        if not frontal_bases:
            logger.warning("⚠️ No se detectaron imágenes frontales para procesar.")
            return

        # 2. Sincronización de archivos en carpetas
        logger.info(f"🔍 Sincronizando archivos en las subcarpetas...")
        files_per_folder = {}
        for sd in SUB_DIRS:
            path = os.path.join(BASE_INPUT_DIR, sd)
            all_f = []
            for ext in ("*.jpg", "*.png", "*.jpeg"):
                all_f.extend(glob.glob(os.path.join(path, ext)))
            files_per_folder[sd] = {get_base_name(os.path.basename(f)): os.path.basename(f) for f in all_f if "rgb" in f.lower()}

        # 3. Intersección: Que existan en las 3 vistas Y sean frontales
        common_bases = sorted(list(
            set(files_per_folder["00_15"].keys()) & 
            set(files_per_folder["00_16"].keys()) & 
            set(files_per_folder["00_17"].keys()) & 
            set(frontal_bases)
        ))

        # Aplicar límite si se especifica
        if NUM_IMAGES_TO_PROCESS is not None:
            selected_bases = common_bases[:NUM_IMAGES_TO_PROCESS]
        else:
            selected_bases = common_bases

        logger.info(f"✅ Se han encontrado {len(selected_bases)} imágenes frontales comunes para procesar.")

        if not selected_bases:
            return

        # 4. Carga de IA
        logger.info("🚀 Cargando modelos en GPU...")
        from qwenimage.pipeline_qwenimage_edit_plus import QwenImageEditPlusPipeline
        from qwenimage.transformer_qwenimage import QwenImageTransformer2DModel
        
        torch.cuda.empty_cache()
        transformer = QwenImageTransformer2DModel.from_pretrained(TRANSFORMER_MODEL, subfolder="transformer", torch_dtype=torch.bfloat16, low_cpu_mem_usage=True)
        pipe = QwenImageEditPlusPipeline.from_pretrained(BASE_MODEL, transformer=transformer, torch_dtype=torch.bfloat16)
        pipe.enable_sequential_cpu_offload()
        pipe.load_lora_weights(LORA_REPO, weight_name=LORA_WEIGHTS, adapter_name="angles")
        pipe.set_adapters(["angles"], adapter_weights=[0.9])

        # 5. Bucle de Procesamiento
        for base_name in selected_bases:
            real_filename = files_per_folder[ORIGIN_VIEW][base_name]
            img_path = os.path.join(BASE_INPUT_DIR, ORIGIN_VIEW, real_filename)
            raw_image = Image.open(img_path).convert("RGB")
            
            for target in ["00_15", "00_16"]:
                prompt = TRANSFORMATIONS[ORIGIN_VIEW][target]
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