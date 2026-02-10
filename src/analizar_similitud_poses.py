import os
import glob
import pandas as pd
import numpy as np
import cv2
import mediapipe as mp
from dataclasses import dataclass
from scipy.spatial.distance import cosine
# Asegúrate de que utils.py tenga la Z opcional o usa el parche local si prefieres
from utils import Landmark, normalize_landmarks, get_landmarks_from_image

# --- LÓGICA DE MÉTRICAS ---
def get_metrics(pts_a, pts_b):
    # Convertimos las listas de objetos Landmark a arrays de numpy (N, 2)
    p1 = np.array([[l.x, l.y] for l in pts_a])
    p2 = np.array([[l.x, l.y] for l in pts_b])
    
    # 1. MPJPE: Distancia euclidiana media
    dist = np.linalg.norm(p1 - p2, axis=1)
    mpjpe = np.mean(dist)
    
    # 2. PCK@5: Porcentaje de puntos con error < 5 unidades
    pck5 = np.mean(dist < 5.0) * 100
    
    # Clasificación de calidad basada en tu criterio
    calidad = "crítico"
    if pck5 > 90: calidad = "excelente"
    elif pck5 >= 70: calidad = "aceptable"
    
    # 3. Similitud de Coseno (Orientación)
    cos_sim = 1 - cosine(p1.flatten(), p2.flatten())
    
    return {
        "mpjpe": mpjpe,
        "pck5": pck5,
        "cosine": cos_sim,
        "calidad": calidad,
        "worst_joint": np.argmax(dist)
    }

# --- PROCESADOR PRINCIPAL ---
def analizar_vistas(path_orig, path_gen, vista_id, csv_output):
    # static_image_mode=True es fundamental para precisión en imágenes sueltas
    pose_processor = mp.solutions.pose.Pose(static_image_mode=True, min_detection_confidence=0.5)
    resultados = []
    
    # Buscamos imágenes generadas
    gen_files = sorted(glob.glob(os.path.join(path_gen, "*.png")))
    print(f"Iniciando análisis de {len(gen_files)} imágenes para vista {vista_id}...")
    
    for g_path in gen_files:
        # Extraer timestamp
        filename = os.path.basename(g_path)
        try:
            timestamp = filename.split('_')[-2]
        except IndexError:
            continue
            
        # Buscar original equivalente
        o_name = f"{vista_id}_{timestamp}_rgb.png"
        o_path = os.path.join(path_orig, o_name)
        
        row = {
            "timestamp": timestamp,
            "ruta_original": o_path, 
            "ruta_generada": g_path, 
            "status": "OK",
            "mpjpe": None, "pck5": None, "cosine": None, "calidad": None, "worst_joint": None
        }
        
        if not os.path.exists(o_path):
            row["status"] = "ORIGINAL_NOT_FOUND"
            resultados.append(row)
            continue

        # Leer imágenes
        img_o_bgr = cv2.imread(o_path)
        img_g_bgr = cv2.imread(g_path)
        
        if img_o_bgr is None or img_g_bgr is None:
            row["status"] = "READ_ERROR"
            resultados.append(row)
            continue

        img_o = cv2.cvtColor(img_o_bgr, cv2.COLOR_BGR2RGB)
        img_g = cv2.cvtColor(img_g_bgr, cv2.COLOR_BGR2RGB)
        
        # Extraer Landmarks
        res_o_raw = get_landmarks_from_image(img_o, pose_processor)
        res_g_raw = get_landmarks_from_image(img_g, pose_processor)
        
        # Normalizar (accedemos a la llave "pose" si existe)
        lms_o_dict = normalize_landmarks(res_o_raw) if "pose" in res_o_raw else {}
        lms_g_dict = normalize_landmarks(res_g_raw) if "pose" in res_g_raw else {}
        
        lms_o = lms_o_dict.get("pose")
        lms_g = lms_g_dict.get("pose")
        
        if not lms_o or not lms_g:
            # Identificamos cuál falló para el reporte
            fail_msg = "BOTH" if not lms_o and not lms_g else ("ORIG" if not lms_o else "GEN")
            row["status"] = f"DETECTION_FAIL_{fail_msg}"
        else:
            # Calcular y actualizar métricas
            m = get_metrics(lms_o, lms_g)
            row.update(m)
            
        resultados.append(row)
    
    # Guardar resultados
    df = pd.DataFrame(resultados)
    df.to_csv(csv_output, index=False)
    print(f"✅ Análisis completado. Resultados guardados en: {csv_output}")

# --- EJECUCIÓN ---
if __name__ == "__main__":
    # Análisis Vista 15
    analizar_vistas(
        path_orig="/nas/antoniodetoro/qwen/Qwen-Image-Edit-Angles-2/datos/MultiViewVisibleImagesHPE_Custom/test/00_15", 
        path_gen="/nas/antoniodetoro/qwen/Qwen-Image-Edit-Angles-2/datos/MultiViewVisibleImagesHPE_Custom/00_17/to_00_15/", 
        vista_id="00_15", 
        csv_output="resultados_vista_15.csv"
    )
    
    # Análisis Vista 15 (puedes descomentar y ajustar rutas)
    # analizar_vistas(
    #     path_orig=".../test/00_15", 
    #     path_gen=".../to_00_15/", 
    #     vista_id="00_15", 
    #     csv_output="resultados_vista_15.csv"
    # )