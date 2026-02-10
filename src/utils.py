import os
import sys
import cv2, csv
import numpy as np
import pandas as pd
import mediapipe as mp
import warnings, logging
from dataclasses import dataclass

warnings.filterwarnings('ignore')
logging.getLogger('mediapipe').setLevel(logging.ERROR)
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

# Definimos las conexiones de MediaPipe
mp_holistic = mp.solutions.holistic
POSE_CONNECTIONS = mp_holistic.POSE_CONNECTIONS
HAND_CONNECTIONS = mp_holistic.HAND_CONNECTIONS

@dataclass
class Landmark:
    x: float
    y: float
    z: float = 0.0  # Añadido z ya que se usa en el CSV
    landmark_id: int = None

def get_landmarks_from_image(image_rgb, pose_processor):
    """Procesa la imagen RGB usando la instancia de Pose pasada."""
    results = pose_processor.process(image_rgb)
    all_lms = {}
    if results.pose_landmarks:
        all_lms["pose"] = [
            Landmark(lm.x, lm.y, lm.z, i) 
            for i, lm in enumerate(results.pose_landmarks.landmark)
        ]
    return all_lms

def save_landmarks_to_csv(landmarks, csv_path):
    data = []
    # El diccionario 'landmarks' puede tener varias llaves (pose, left_hand, etc.)
    for tipo, lms in landmarks.items():
        tag = {
            "pose": "POSE",
            "left_hand": "HAND_L",
            "right_hand": "HAND_R",
            "face": "FACE"
        }.get(tipo, tipo.upper())

        for lm in lms:
            data.append({
                'tipo': tag,
                'landmark_id': lm.landmark_id,
                'x': lm.x,
                'y': lm.y,
                'z': lm.z
            })
    
    if data:
        df = pd.DataFrame(data)
        os.makedirs(os.path.dirname(csv_path), exist_ok=True)
        df.to_csv(csv_path, index=False)


def plot_landmarks(landmarks, background_img=None, tam=100, is_normalized=True):

    # Usamos tam=100 por defecto para que la imagen final sea nítida
    if background_img is not None:
        img = background_img.copy()
    else:
        img = np.zeros((tam, tam, 3), dtype=np.uint8)
    h, w = img.shape[:2]

    def to_pixel(point):
        if is_normalized:
            # point.x está en rango 0-100
            return (int(point.x * w / 100), int(point.y * h / 100))
        return (int(point.x * w), int(point.y * h))

    # Dibujar conexiones

    lines = [("pose", POSE_CONNECTIONS), ("left_hand", HAND_CONNECTIONS), ("right_hand", HAND_CONNECTIONS)]

    for key, connections in lines:
        if key in landmarks:
            lms = landmarks[key]
            for start_idx, end_idx in connections:
                if start_idx < len(lms) and end_idx < len(lms):
                    cv2.line(img, to_pixel(lms[start_idx]), to_pixel(lms[end_idx]), (255, 255, 255), 1)

    # Dibujar puntos

    for p in landmarks.get("pose", []):
        cv2.circle(img, to_pixel(p), 1, (0, 0, 255), -1)

    for hand in ["left_hand", "right_hand"]:
        for p in landmarks.get(hand, []):
            cv2.circle(img, to_pixel(p), 1, (255, 0, 0), -1)
    return img


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