import os
import sys

# 1. Bloqueo total de logs de TensorFlow y MediaPipe
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
os.environ['GLOG_minloglevel'] = '3'

import cv2, csv
import mediapipe as mp
import numpy as np
from dataclasses import dataclass

@dataclass
class Landmark:
    x: float
    y: float
    landmark_id: int = None

def get_landmarks_from_image(image_path):
    # 2. Redirigimos stderr a dev/null temporalmente para silenciar el arranque de C++
    stderr = sys.stderr
    sys.stderr = open(os.devnull, 'w')
    
    try:
        mp_pose = mp.solutions.pose
        with mp_pose.Pose(static_image_mode=True, min_detection_confidence=0.5) as pose:
            image = cv2.imread(image_path)
            if image is None:
                return None, None
            
            image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            results = pose.process(image_rgb)
            
            all_lms = {}
            if results.pose_landmarks:
                all_lms["pose"] = [Landmark(lm.x, lm.y, i) for i, lm in enumerate(results.pose_landmarks.landmark)]
            return all_lms, image_rgb
    finally:
        # 3. Restauramos stderr
        sys.stderr = stderr

def normalize_landmarks(landmarks, tam_target=100, margin=5):
    all_points = []
    for key in landmarks:
        all_points.extend(landmarks[key])
    
    if not all_points: return landmarks

    min_x, max_x = min(lm.x for lm in all_points), max(lm.x for lm in all_points)
    min_y, max_y = min(lm.y for lm in all_points), max(lm.y for lm in all_points)

    width = (max_x - min_x) if max_x != min_x else 0.01
    height = (max_y - min_y) if max_y != min_y else 0.01

    available_space = tam_target - (2 * margin)
    scale = available_space / max(width, height)
    offset_x = margin + (available_space - width * scale) / 2
    offset_y = margin + (available_space - height * scale) / 2

    normalized_data = {}
    for k, list_lms in landmarks.items():
        normalized_data[k] = [
            Landmark(x=(lm.x - min_x) * scale + offset_x, 
                     y=(lm.y - min_y) * scale + offset_y, 
                     landmark_id=lm.landmark_id) 
            for lm in list_lms
        ]
    return normalized_data

def is_facing_camera(normalized_landmarks):
    pose = normalized_landmarks.get("pose", [])
    if len(pose) < 33: return "ERROR", "Sin puntos"

    nose = pose[0]
    l_sh, r_sh = pose[11], pose[12]
    l_hip, r_hip = pose[23], pose[24]

    # Lógica de detección basada en simetría y proporciones
    sh_width = abs(r_sh.x - l_sh.x)
    sh_center_x = (l_sh.x + r_sh.x) / 2
    nose_dev = abs(nose.x - sh_center_x) / (sh_width if sh_width > 0 else 1)
    
    torso_h = abs((l_sh.y + r_sh.y)/2 - (l_hip.y + r_hip.y)/2)
    sh_ratio = sh_width / (torso_h if torso_h > 0 else 1)

    es_frontal = nose_dev < 0.18 and sh_ratio > 0.45
    return ("SÍ" if es_frontal else "NO"), f"Dev:{nose_dev:.2f} Ratio:{sh_ratio:.2f}"

def analyze_folder(folder_path):
    valid_extensions = ('.png', '.jpg', '.jpeg', '.webp')
    files = [f for f in os.listdir(folder_path) 
             if "rgb" in f.lower() and f.lower().endswith(valid_extensions)]
    
    if not files:
        print("No se encontraron imágenes con 'rgb' en el nombre.")
        return

    # Nombre del archivo de salida
    script_dir = os.path.dirname(os.path.abspath(__file__))
    output_file = os.path.join(script_dir, "resultados_frontalidad.csv")
    
    # Preparamos la lista para guardar datos
    resultados_para_archivo = []

    print(f"\n{'ARCHIVO':<35} | {'FRONTAL':<8} | {'MÉTRICAS'}")
    print("-" * 75)

    for filename in files:
        path = os.path.join(folder_path, filename)
        lms, _ = get_landmarks_from_image(path)
        
        if lms and "pose" in lms:
            norm_lms = normalize_landmarks(lms)
            resultado, info = is_facing_camera(norm_lms)
        else:
            resultado, info = "ERROR", "Sin detección"
            
        # Imprimir en consola
        print(f"{filename[:35]:<35} | {resultado:<8} | {info}")
        
        # Guardar en la lista para el CSV
        resultados_para_archivo.append({
            "Imagen": filename,
            "Frontal": resultado,
            "Metricas": info
        })

    # Escribir los resultados en un archivo CSV
    with open(output_file, mode='w', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=["Imagen", "Frontal", "Metricas"])
        writer.writeheader()
        writer.writerows(resultados_para_archivo)

    print("-" * 75)
    print(f"✅ Análisis finalizado. Resultados guardados en: {output_file}")
# analyze_folder('tu_ruta')
# --- EJECUCIÓN ---
# Cambia 'tu_carpeta_de_imagenes' por la ruta real
analyze_folder('MultiViewVisibleImagesHPE_Custom/test/00_17/')