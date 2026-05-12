#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Script para Pre-Calcular Heatmaps de HRNet y Añadirlos a Archivos .pt
=====================================================================

VERSIÓN MEJORADA para archivos .pt generados con precompute_from_data.py
que no contienen nombres de imágenes, solo índices de batch.

Este script:
1. Reconstruye el orden del dataset original usando MultiViewDataset
2. Mapea cada batch_XXXXX.pt con las imágenes target correspondientes
3. Calcula heatmaps con HRNet para cada imagen target
4. Añade los heatmaps al archivo .pt bajo la clave "target_heatmaps"

Uso:
    python add_heatmaps_to_latents_v2.py \\
        --dataset_root "./dataset" \\
        --latents_dir "./precomputed_latents" \\
        --hrnet_model "./models/pose_hrnet_w48_384x288.pth" \\
        --split train \\
        --batch_size 1

IMPORTANTE: El batch_size debe coincidir con el usado en precompute_from_data.py
"""

import os
import sys
import argparse
import logging
from pathlib import Path
from typing import Optional, List, Dict

import cv2
import numpy as np
import torch
from tqdm import tqdm
from PIL import Image

# Importar MultiViewDataset del script de precompute
# Asumimos que está en el mismo directorio o ajustar path
current_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, current_dir)

# Copiar la clase MultiViewDataset aquí para independencia
class MultiViewDataset:
    """
    Dataset para imágenes multi-vista sincronizadas por timestamp.
    COPIA de la clase en precompute_from_data.py para reconstruir el orden.
    """

    CAMERA_ANGLES = {
        "00_17": 0,
        "00_16": 90,
        "00_15": -90,
    }

    ANGLE_PROMPTS = {
        0: {
            90:   "将镜头向右旋转90度 Rotate the camera 90 degrees to the right.",
            180:  "将镜头旋转180度 Rotate the camera 180 degrees.",
            -90:  "将镜头向左旋转90度 Rotate the camera 90 degrees to the left.",
        },
        90: {
            90:   "将镜头向右旋转90度 Rotate the camera 90 degrees to the right.",
            -90:  "将镜头向左旋转90度 Rotate the camera 90 degrees to the left.",
            -180: "将镜头旋转180度 Rotate the camera 180 degrees.",
        },
        -90: {
            90:   "将镜头向右旋转90度 Rotate the camera 90 degrees to the right.",
            -90:  "将镜头向左旋转90度 Rotate the camera 90 degrees to the left.",
            180:  "将镜头旋转180度 Rotate the camera 180 degrees.",
        },
    }

    def __init__(
        self,
        dataset_root: str,
        resolution: int = 1024,
        split: str = "train",
        train_ratio: float = 0.9,
    ):
        self.dataset_root = Path(dataset_root)
        self.resolution = resolution
        self.split = split

        self.samples = self._build_sample_pairs()

        np.random.seed(42)
        indices = np.random.permutation(len(self.samples))
        split_idx = int(len(indices) * train_ratio)

        if split == "train":
            self.samples = [self.samples[i] for i in indices[:split_idx]]
        else:
            self.samples = [self.samples[i] for i in indices[split_idx:]]

    def _build_sample_pairs(self) -> List[Dict]:
        samples = []
        timestamp_data = {}

        base_dir = self.dataset_root / "train_val"
        if not base_dir.exists():
            base_dir = self.dataset_root

        for camera_dir in sorted(base_dir.glob("*")):
            if not camera_dir.is_dir():
                continue
            camera_id = camera_dir.name
            if camera_id not in self.CAMERA_ANGLES:
                continue
            images = sorted(camera_dir.glob("*_rgb.png"))
            if not images:
                images = sorted(camera_dir.glob("*_rgb*"))
            for img_path in images:
                parts = img_path.stem.split("_")
                if len(parts) >= 3:
                    timestamp = parts[2]
                    if timestamp not in timestamp_data:
                        timestamp_data[timestamp] = {}
                    timestamp_data[timestamp][camera_id] = img_path

        for timestamp, cameras in timestamp_data.items():
            for src_cam, src_angle in self.CAMERA_ANGLES.items():
                if src_cam not in cameras:
                    continue
                src_img = cameras[src_cam]
                for tgt_cam, tgt_angle in self.CAMERA_ANGLES.items():
                    if src_cam == tgt_cam or tgt_cam not in cameras:
                        continue
                    tgt_img = cameras[tgt_cam]
                    angle_diff = (tgt_angle - src_angle) % 360
                    if angle_diff > 180:
                        angle_diff -= 360
                    if src_angle in self.ANGLE_PROMPTS and angle_diff in self.ANGLE_PROMPTS[src_angle]:
                        prompt = self.ANGLE_PROMPTS[src_angle][angle_diff]
                    else:
                        prompt = f"将镜头旋转{angle_diff}度 Rotate the camera {angle_diff} degrees."
                    samples.append({
                        "src_path": str(src_img),
                        "tgt_path": str(tgt_img),
                        "src_angle": src_angle,
                        "tgt_angle": tgt_angle,
                        "angle_diff": angle_diff,
                        "prompt": prompt,
                        "timestamp": timestamp,
                    })
        return samples

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> Dict:
        return self.samples[idx]


# Importar o copiar HRNet classes
try:
    from hrnet_inference import HRNetInferencer
except ImportError:
    logging.error(
        "No se pudo importar HRNetInferencer. "
        "Asegúrate de que hrnet_inference.py esté disponible."
    )
    sys.exit(1)


logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def add_heatmaps_to_batches(
    dataset_root: str,
    latents_dir: str,
    hrnet_model_path: str,
    split: str = 'train',
    batch_size: int = 1,
    model_type: str = 'pose_hrnet_w48_384x288',
    device: str = None,
    backup_original: bool = True,
    skip_existing: bool = True,
    train_ratio: float = 0.9,
):
    """
    Añade heatmaps de HRNet a archivos .pt batch_XXXXX.pt usando mapeo con dataset.
    
    Args:
        dataset_root: Directorio raíz del dataset original
        latents_dir: Directorio con los latentes pre-computados
        hrnet_model_path: Ruta al modelo HRNet
        split: Split del dataset (train/val)
        batch_size: Batch size usado en precompute (IMPORTANTE: debe coincidir)
        model_type: Tipo de modelo HRNet
        device: Device para HRNet (cuda/cpu)
        backup_original: Si hacer backup de los .pt originales
        skip_existing: Si saltar archivos que ya tienen heatmaps
        train_ratio: Ratio train/val usado en precompute
    """
    latents_path = Path(latents_dir) / split
    
    if not latents_path.exists():
        raise FileNotFoundError(f"No se encontró: {latents_path}")
    
    # 1. Reconstruir dataset original en el mismo orden
    logger.info(f"Reconstruyendo dataset original desde {dataset_root}...")
    dataset = MultiViewDataset(
        dataset_root=dataset_root,
        split=split,
        train_ratio=train_ratio,
    )
    logger.info(f"Dataset reconstruido: {len(dataset)} muestras")
    
    # 2. Inicializar HRNet
    logger.info(f"Cargando HRNet desde {hrnet_model_path}...")
    inferencer = HRNetInferencer(
        model_path=hrnet_model_path,
        model_type=model_type,
        device=device,
        download=False
    )
    
    # 3. Listar archivos .pt batch
    pt_files = sorted(list(latents_path.glob("batch_*.pt")))
    logger.info(f"Encontrados {len(pt_files)} archivos batch_*.pt en {latents_path}")
    
    if len(pt_files) == 0:
        logger.warning("No hay archivos batch_*.pt para procesar")
        return
    
    # 4. Crear directorio de backup si es necesario
    if backup_original:
        backup_dir = latents_path / "backup_before_heatmaps"
        backup_dir.mkdir(exist_ok=True)
        logger.info(f"Backups se guardarán en: {backup_dir}")
    
    # 5. Estadísticas
    processed = 0
    skipped = 0
    errors = 0
    
    # 6. Procesar cada batch
    for batch_idx, pt_file in enumerate(tqdm(pt_files, desc=f"Procesando {split}")):
        try:
            # Cargar datos existentes
            data = torch.load(pt_file, weights_only=True)
            
            # Verificar si ya tiene heatmaps
            if skip_existing and "target_heatmaps" in data:
                skipped += 1
                continue
            
            # Determinar cuántas muestras hay en este batch
            # Usar target_latents_packed para inferir batch_size real
            actual_batch_size = data["target_latents_packed"].shape[0]
            
            # Calcular índices del dataset correspondientes a este batch
            start_idx = batch_idx * batch_size
            end_idx = start_idx + actual_batch_size
            
            if end_idx > len(dataset):
                logger.warning(
                    f"Batch {batch_idx} excede dataset size "
                    f"({end_idx} > {len(dataset)}), truncando"
                )
                end_idx = len(dataset)
                actual_batch_size = end_idx - start_idx
            
            if actual_batch_size == 0:
                logger.warning(f"Batch {batch_idx} vacío, saltando")
                continue
            
            # Obtener las muestras correspondientes del dataset
            batch_samples = [dataset[i] for i in range(start_idx, end_idx)]
            
            # Calcular heatmaps para cada imagen target en el batch
            batch_heatmaps = []
            
            for sample_idx, sample in enumerate(batch_samples):
                tgt_img_path = sample["tgt_path"]
                
                # Cargar imagen
                image = cv2.imread(tgt_img_path)
                if image is None:
                    logger.warning(
                        f"No se pudo cargar imagen target: {tgt_img_path} "
                        f"(batch {batch_idx}, muestra {sample_idx})"
                    )
                    # Usar heatmaps vacíos como placeholder
                    batch_heatmaps.append(np.zeros((17, 72, 96), dtype=np.float32))
                    continue
                
                # Calcular heatmaps
                heatmaps = inferencer.get_heatmaps(image)  # (17, H, W) numpy
                batch_heatmaps.append(heatmaps)
            
            # Convertir lista a tensor
            if len(batch_heatmaps) != actual_batch_size:
                logger.error(
                    f"Mismatch en batch {batch_idx}: "
                    f"esperaba {actual_batch_size} heatmaps, obtuvo {len(batch_heatmaps)}"
                )
                errors += 1
                continue
            
            batch_heatmaps_tensor = torch.from_numpy(
                np.stack(batch_heatmaps, axis=0)
            ).float()  # (B, 17, H, W)
            
            # Hacer backup del original si es necesario
            if backup_original:
                backup_path = backup_dir / pt_file.name
                if not backup_path.exists():
                    torch.save(data, backup_path)
            
            # Añadir heatmaps a los datos
            data["target_heatmaps"] = batch_heatmaps_tensor
            
            # Guardar archivo actualizado
            torch.save(data, pt_file)
            processed += 1
            
            if processed % 50 == 0:
                logger.info(
                    f"Progreso: {processed}/{len(pt_files)} batches procesados, "
                    f"{skipped} saltados"
                )
        
        except Exception as e:
            logger.error(f"Error procesando {pt_file.name}: {e}")
            import traceback
            traceback.print_exc()
            errors += 1
    
    # Resumen final
    logger.info("=" * 60)
    logger.info("RESUMEN")
    logger.info("=" * 60)
    logger.info(f"Total batches:         {len(pt_files)}")
    logger.info(f"Procesados:            {processed}")
    logger.info(f"Saltados (ya tenían):  {skipped}")
    logger.info(f"Errores:               {errors}")
    logger.info("=" * 60)
    
    if processed > 0:
        logger.info(f"✓ Heatmaps añadidos correctamente a {processed} batches")
        if backup_original:
            logger.info(f"✓ Backups guardados en: {backup_dir}")
    else:
        logger.warning("⚠ No se procesó ningún batch")


def verify_heatmaps(
    latents_dir: str,
    split: str = 'train',
    sample_size: int = 5
):
    """
    Verifica que los heatmaps se hayan añadido correctamente.
    
    Args:
        latents_dir: Directorio con los latentes
        split: Split del dataset
        sample_size: Número de batches a verificar
    """
    latents_path = Path(latents_dir) / split
    pt_files = sorted(list(latents_path.glob("batch_*.pt")))[:sample_size]
    
    logger.info(f"Verificando {len(pt_files)} batches de muestra...")
    
    for pt_file in pt_files:
        data = torch.load(pt_file, weights_only=True)
        
        if "target_heatmaps" not in data:
            logger.error(f"✗ {pt_file.name}: NO tiene heatmaps")
            continue
        
        heatmaps = data["target_heatmaps"]
        
        if not isinstance(heatmaps, torch.Tensor):
            logger.error(f"✗ {pt_file.name}: heatmaps no es Tensor")
            continue
        
        shape = heatmaps.shape
        batch_size = data["target_latents_packed"].shape[0]
        
        if len(shape) != 4 or shape[0] != batch_size or shape[1] != 17:
            logger.error(
                f"✗ {pt_file.name}: shape incorrecta {shape}, "
                f"esperaba ({batch_size}, 17, H, W)"
            )
            continue
        
        logger.info(
            f"✓ {pt_file.name}: heatmaps OK, shape={shape}, "
            f"min={heatmaps.min():.4f}, max={heatmaps.max():.4f}, "
            f"batch_size={batch_size}"
        )


def main():
    parser = argparse.ArgumentParser(
        description='Añadir heatmaps de HRNet a archivos batch_*.pt de latentes',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
EJEMPLOS:
  # Procesamiento básico
  python add_heatmaps_to_latents_v2.py \\
      --dataset_root "./dataset" \\
      --latents_dir "./precomputed_latents" \\
      --hrnet_model "./models/pose_hrnet_w48_384x288.pth" \\
      --batch_size 1
  
  # Procesar solo split 'val'
  python add_heatmaps_to_latents_v2.py \\
      --dataset_root "./dataset" \\
      --latents_dir "./precomputed_latents" \\
      --hrnet_model "./models/pose_hrnet_w48_384x288.pth" \\
      --split val \\
      --batch_size 1
  
  # Con train_ratio diferente al default
  python add_heatmaps_to_latents_v2.py \\
      --dataset_root "./dataset" \\
      --latents_dir "./precomputed_latents" \\
      --hrnet_model "./models/pose_hrnet_w48_384x288.pth" \\
      --train_ratio 0.8
        """
    )
    
    parser.add_argument(
        '--dataset_root',
        type=str,
        required=True,
        help='Directorio raíz del dataset original (donde están las imágenes)'
    )
    parser.add_argument(
        '--latents_dir',
        type=str,
        required=True,
        help='Directorio con archivos batch_*.pt de latentes pre-computados'
    )
    parser.add_argument(
        '--hrnet_model',
        type=str,
        required=True,
        help='Ruta al modelo HRNet (.pth)'
    )
    parser.add_argument(
        '--split',
        type=str,
        default='train',
        choices=['train', 'val', 'test'],
        help='Split del dataset a procesar (default: train)'
    )
    parser.add_argument(
        '--batch_size',
        type=int,
        default=1,
        help='Batch size usado en precompute_from_data.py (IMPORTANTE: debe coincidir)'
    )
    parser.add_argument(
        '--model_type',
        type=str,
        default='pose_hrnet_w48_384x288',
        choices=['hrnet_w32_coco', 'hrnet_w48_coco', 'pose_hrnet_w48_384x288'],
        help='Tipo de modelo HRNet (default: pose_hrnet_w48_384x288)'
    )
    parser.add_argument(
        '--device',
        type=str,
        default=None,
        choices=['cuda', 'cpu'],
        help='Device para HRNet (default: auto)'
    )
    parser.add_argument(
        '--train_ratio',
        type=float,
        default=0.9,
        help='Train/val split ratio usado en precompute (default: 0.9)'
    )
    parser.add_argument(
        '--no-backup',
        action='store_true',
        help='NO hacer backup de archivos .pt originales'
    )
    parser.add_argument(
        '--force',
        action='store_true',
        help='Procesar incluso batches que ya tienen heatmaps'
    )
    parser.add_argument(
        '--verify',
        action='store_true',
        help='Solo verificar que los heatmaps existan (no procesar)'
    )
    parser.add_argument(
        '--verify-samples',
        type=int,
        default=5,
        help='Número de batches a verificar (default: 5)'
    )
    
    args = parser.parse_args()
    
    try:
        if args.verify:
            # Modo verificación
            logger.info("Modo verificación activado")
            verify_heatmaps(
                args.latents_dir,
                args.split,
                args.verify_samples
            )
        else:
            # Modo procesamiento
            add_heatmaps_to_batches(
                dataset_root=args.dataset_root,
                latents_dir=args.latents_dir,
                hrnet_model_path=args.hrnet_model,
                split=args.split,
                batch_size=args.batch_size,
                model_type=args.model_type,
                device=args.device,
                backup_original=not args.no_backup,
                skip_existing=not args.force,
                train_ratio=args.train_ratio,
            )
            
            # Verificar automáticamente después de procesar
            logger.info("")
            logger.info("Verificando batches procesados...")
            verify_heatmaps(
                args.latents_dir,
                args.split,
                args.verify_samples
            )
    
    except Exception as e:
        logger.error(f"Error fatal: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == '__main__':
    main()
