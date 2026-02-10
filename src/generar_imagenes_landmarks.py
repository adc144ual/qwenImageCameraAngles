import os
import sys
import pandas as pd
import cv2
import logging
import argparse
from utils import normalize_landmarks, plot_landmarks, Landmark

# Configuración básica de logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[logging.StreamHandler(sys.stdout)]
)

def load_pose_from_csv(csv_path):
    """
    Lee un archivo CSV y devuelve un diccionario con los landmarks de 'pose'.
    Ignora 'hands' y 'face'.
    """
    try:
        df = pd.read_csv(csv_path)
    except Exception as e:
        logging.error(f"Error leyendo CSV {csv_path}: {e}")
        return {}

    landmarks = {"pose": []}
    
    # Filtramos por tipo 'POSE' (según save_landmarks_to_csv en utils.py)
    if 'tipo' not in df.columns:
        return {}

    pose_rows = df[df['tipo'] == 'POSE']
    
    for _, row in pose_rows.iterrows():
        lm = Landmark(
            x=float(row['x']),
            y=float(row['y']),
            z=float(row.get('z', 0.0)),
            landmark_id=int(row['landmark_id'])
        )
        landmarks["pose"].append(lm)
        
    return landmarks

def process_csv_directory(input_dir, output_dir, image_size=512):
    """
    Recorre recursivamente el directorio de entrada, procesa los CSV y guarda las imágenes.
    """
    if not os.path.exists(input_dir):
        logging.error(f"El directorio de entrada no existe: {input_dir}")
        return

    count = 0
    errors = 0

    logging.info(f"Iniciando procesamiento desde: {input_dir}")
    logging.info(f"Guardando resultados en: {output_dir}")

    for root, dirs, files in os.walk(input_dir):
        for file in files:
            if not file.lower().endswith(".csv"):
                continue

            csv_path = os.path.join(root, file)
            
            # Calcular ruta de salida preservando estructura
            rel_path = os.path.relpath(root, input_dir)
            target_folder = os.path.join(output_dir, rel_path)
            os.makedirs(target_folder, exist_ok=True)
            
            output_filename = os.path.splitext(file)[0] + ".png"
            output_path = os.path.join(target_folder, output_filename)

            try:
                # 1. Cargar landmarks (filtrando solo POSE)
                landmarks = load_pose_from_csv(csv_path)
                
                if not landmarks.get("pose"):
                    # Si no hay landmarks de pose, saltamos
                    continue

                # 2. Normalizar landmarks
                # Usamos tam_target=100 para mantener consistencia con plot_landmarks(is_normalized=True)
                normalized_lms = normalize_landmarks(landmarks, tam_target=100, margin=5)

                # 3. Generar imagen (fondo negro por defecto en plot_landmarks)
                # tam define el tamaño final de la imagen en píxeles.
                img_result = plot_landmarks(normalized_lms, tam=image_size, is_normalized=True)

                # 4. Guardar imagen
                cv2.imwrite(output_path, img_result)
                count += 1
                
                if count % 100 == 0:
                    logging.info(f"Procesadas {count} imágenes...")

            except Exception as e:
                logging.error(f"Fallo al procesar {csv_path}: {e}")
                errors += 1

    logging.info("=" * 30)
    logging.info(f"Procesamiento completado.")
    logging.info(f"Imágenes generadas: {count}")
    logging.info(f"Errores: {errors}")
    logging.info("=" * 30)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generar imágenes de pose desde CSVs normalizados.")
    parser.add_argument("--input", "-i", type=str, default="MultiViewVisibleImagesHPE_CSV", 
                        help="Directorio raíz con los archivos CSV.")
    parser.add_argument("--output", "-o", type=str, default="MultiViewVisibleImagesHPE_Rendered", 
                        help="Directorio raíz donde guardar las imágenes.")
    parser.add_argument("--size", "-s", type=int, default=100, 
                        help="Tamaño de la imagen cuadrada de salida (default: 100).")

    args = parser.parse_args()

    # Ajuste de rutas si se ejecuta desde 'src/' pero los datos están en el root
    input_path = args.input
    if not os.path.exists(input_path) and os.path.exists(os.path.join("..", input_path)):
        input_path = os.path.join("..", input_path)
    
    output_path = args.output
    if not os.path.isabs(output_path) and input_path.startswith(".."):
        output_path = os.path.join("..", output_path)

    process_csv_directory(input_path, output_path, args.size)
