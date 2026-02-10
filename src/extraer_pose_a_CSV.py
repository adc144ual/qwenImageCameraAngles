import os, logging, sys, cv2
import mediapipe as mp
from datetime import datetime
from utils import save_landmarks_to_csv, get_landmarks_from_image, Landmark


# --- CONFIGURACIÓN DE LOGGING ---
log_filename = f"procesamiento_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(log_filename),
        logging.StreamHandler(sys.stdout)
    ]
)

# Configuración Mediapipe para GPU
os.environ["MEDIAPIPE_DISABLE_GPU"] = "0"
os.environ["CUDA_VISIBLE_DEVICES"] = "0"


# --- LÓGICA DE PROCESAMIENTO MULTI-VISTA ---

def process_dataset(input_root, output_root, regenerar=False):
    mp_pose = mp.solutions.pose
    subconjuntos = ['test', 'train_val']
    
    # Contadores para el resumen final
    stats = {"exito": 0, "fallo_pose": 0, "omitidos_sincro": 0, "errores_lectura": 0}
    
    with mp_pose.Pose(static_image_mode=True, min_detection_confidence=0.5) as pose:
        for sub in subconjuntos:
            sub_path = os.path.join(input_root, sub)
            if not os.path.exists(sub_path):
                logging.warning(f"Saltando subconjunto: {sub} (no existe)")
                continue

            logging.info(f"--- PROCESANDO {sub.upper()} ---")
            failed_timestamps = set()
            
            # Obtener carpetas de vistas (00_15, 00_16...) ordenadas
            vistas = sorted([d for d in os.listdir(sub_path) if os.path.isdir(os.path.join(sub_path, d))])

            for vista in vistas:
                vista_path = os.path.join(sub_path, vista)
                files = sorted(os.listdir(vista_path))
                
                for filename in files:
                    # Filtro: debe contener "rgb" y ser imagen
                    if "rgb" not in filename.lower() or not filename.lower().endswith(('.jpg', '.jpeg', '.png')):
                        continue

                    # LÓGICA DE EXCLUSIÓN: 
                    # Si el mismo timestamp (nombre archivo) falló antes en este subconjunto, saltar.
                    if filename in failed_timestamps:
                        stats["omitidos_sincro"] += 1
                        continue

                    csv_name = os.path.splitext(filename)[0] + ".csv"
                    csv_path = os.path.join(output_root, sub, vista, csv_name)

                    if os.path.exists(csv_path) and not regenerar:
                        continue

                    img_full_path = os.path.join(vista_path, filename)
                    image = cv2.imread(img_full_path)
                    
                    if image is None:
                        logging.error(f"Error de lectura: {img_full_path}")
                        stats["errores_lectura"] += 1
                        continue

                    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
                    landmarks = get_landmarks_from_image(image_rgb, pose)

                    if landmarks and "pose" in landmarks:
                        save_landmarks_to_csv(landmarks, csv_path)
                        stats["exito"] += 1
                    else:
                        logging.error(f"POCO CONTENIDO/SIN POSE: {sub}/{vista}/{filename}. Marcando timestamp para exclusión.")
                        failed_timestamps.add(filename)
                        stats["fallo_pose"] += 1


def synchronize_outputs_by_timestamp(output_root):
    """
    Sincroniza vistas ignorando el prefijo de la cámara (00_15, 00_16, etc.)
    """
    logging.info("Iniciando sincronización por TIMESTAMP...")
    subconjuntos = ['test', 'train_val']
    
    for sub in subconjuntos:
        sub_path = os.path.join(output_root, sub)
        if not os.path.exists(sub_path): continue
        
        vistas = sorted([d for d in os.listdir(sub_path) if os.path.isdir(os.path.join(sub_path, d))])
        if not vistas: continue
        
        # 1. Mapear timestamps por cada vista
        # Estructura: { 'timestamp': { 'vista': 'nombre_archivo_completo' } }
        catalog = {} 
        
        for vista in vistas:
            vista_dir = os.path.join(sub_path, vista)
            for f in os.listdir(vista_dir):
                # Extraemos el timestamp: de '00_15_12345_rgb.csv' sacamos '12345_rgb'
                # Dividimos por '_' y quitamos el primer elemento (la cámara)
                parts = f.split('_')
                timestamp_id = "_".join(parts[2:]) # '1680262517197_rgb.csv'
                
                if timestamp_id not in catalog:
                    catalog[timestamp_id] = {}
                catalog[timestamp_id][vista] = f

        # 2. Identificar qué timestamps están en TODAS las vistas
        num_vistas_requeridas = len(vistas)
        timestamps_validos = {ts for ts, vistas_encontradas in catalog.items() 
                             if len(vistas_encontradas) == num_vistas_requeridas}
        
        # 3. Borrar archivos que no tienen representación en todas las cámaras
        eliminados = 0
        total_mantenidos = len(timestamps_validos)
        
        for vista in vistas:
            vista_dir = os.path.join(sub_path, vista)
            for f in os.listdir(vista_dir):
                parts = f.split('_')
                timestamp_id = "_".join(parts[2:])
                
                if timestamp_id not in timestamps_validos:
                    os.remove(os.path.join(vista_dir, f))
                    eliminados += 1
        
        logging.info(f"Subconjunto {sub}:")
        logging.info(f"  - Timestamps comunes encontrados: {total_mantenidos}")
        logging.info(f"  - Archivos huérfanos eliminados: {eliminados}")

    # # Resumen final en el log
    # logging.info("="*30)
    # logging.info("RESUMEN DE PROCESAMIENTO")
    # logging.info(f"Imágenes con éxito:    {stats['exito']}")
    # logging.info(f"Fallos de pose:        {stats['fallo_pose']}")
    # logging.info(f"Omitidos por sincro:   {stats['omitidos_sincro']}")
    # logging.info(f"Errores de lectura:    {stats['errores_lectura']}")
    # logging.info("="*30)

# --- INICIO DEL SCRIPT ---
if __name__ == "__main__":
    # Ajusta estas rutas a tu entorno local
    DIRECTORIO_ENTRADA = "MultiViewVisibleThermalImagesHPE"
    DIRECTORIO_SALIDA = "Output_CSVs"

    logging.info(f"Iniciando extracción masiva. Log: {log_filename}")
    # process_dataset(DIRECTORIO_ENTRADA, DIRECTORIO_SALIDA, regenerar=False)

    synchronize_outputs_by_timestamp(DIRECTORIO_SALIDA)
    logging.info("Proceso finalizado correctamente.")