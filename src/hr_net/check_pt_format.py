#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Script de Diagnóstico para Archivos .pt de Latentes
===================================================

Verifica el contenido y formato de archivos batch_*.pt para asegurar
que tengan la estructura correcta antes del entrenamiento.

Uso:
    python check_pt_format.py --latents_dir "./precomputed_latents" --split train
"""

import argparse
import torch
from pathlib import Path
import logging

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(message)s'
)
logger = logging.getLogger(__name__)


def check_pt_files(latents_dir: str, split: str = "train", num_samples: int = 5):
    """Verifica el formato de archivos .pt."""
    
    latents_path = Path(latents_dir) / split
    
    if not latents_path.exists():
        logger.error(f"No se encontró: {latents_path}")
        return
    
    pt_files = sorted(list(latents_path.glob("batch_*.pt")))
    
    if len(pt_files) == 0:
        logger.error(f"No se encontraron archivos batch_*.pt en {latents_path}")
        return
    
    logger.info(f"Encontrados {len(pt_files)} archivos en total")
    logger.info(f"Verificando primeros {num_samples} archivos...\n")
    
    for i, pt_file in enumerate(pt_files[:num_samples]):
        logger.info("=" * 70)
        logger.info(f"Archivo {i+1}/{num_samples}: {pt_file.name}")
        logger.info("=" * 70)
        
        try:
            data = torch.load(pt_file, weights_only=True)
            
            # Campos esperados
            required_fields = [
                "target_latents_packed",
                "source_latents_packed",
                "prompt_embeds",
                "prompt_embeds_mask",
            ]
            
            optional_fields = [
                "target_heatmaps",
                "angle_diffs",
                "resolution",
            ]
            
            # Verificar campos requeridos
            logger.info("Campos encontrados:")
            for field in required_fields:
                if field in data:
                    tensor = data[field]
                    logger.info(f"  ✓ {field}: shape={tensor.shape}, dtype={tensor.dtype}")
                else:
                    logger.error(f"  ✗ {field}: FALTA (REQUERIDO)")
            
            # Verificar campos opcionales
            for field in optional_fields:
                if field in data:
                    value = data[field]
                    if isinstance(value, torch.Tensor):
                        logger.info(f"  ✓ {field}: shape={value.shape}, dtype={value.dtype}")
                    else:
                        logger.info(f"  ✓ {field}: {value} (tipo: {type(value).__name__})")
                else:
                    logger.info(f"  ○ {field}: no presente (opcional)")
            
            # Verificar consistencia de batch_size
            batch_sizes = {}
            for field in required_fields:
                if field in data and isinstance(data[field], torch.Tensor):
                    batch_sizes[field] = data[field].shape[0]
            
            if len(set(batch_sizes.values())) > 1:
                logger.warning(f"  ⚠ Batch sizes inconsistentes: {batch_sizes}")
            else:
                bs = list(batch_sizes.values())[0]
                logger.info(f"  ✓ Batch size consistente: {bs}")
            
            # Verificar heatmaps si existen
            if "target_heatmaps" in data:
                heatmaps = data["target_heatmaps"]
                expected_batch_size = data["target_latents_packed"].shape[0]
                
                if heatmaps.shape[0] != expected_batch_size:
                    logger.error(
                        f"  ✗ Heatmaps batch size {heatmaps.shape[0]} != "
                        f"expected {expected_batch_size}"
                    )
                
                if heatmaps.shape[1] != 17:
                    logger.error(
                        f"  ✗ Heatmaps tiene {heatmaps.shape[1]} keypoints, esperaba 17"
                    )
                
                logger.info(
                    f"  ✓ Heatmaps: min={heatmaps.min():.4f}, "
                    f"max={heatmaps.max():.4f}, mean={heatmaps.mean():.4f}"
                )
        
        except Exception as e:
            logger.error(f"  ✗ Error al cargar {pt_file.name}: {e}")
        
        logger.info("")
    
    # Resumen
    logger.info("=" * 70)
    logger.info("RESUMEN")
    logger.info("=" * 70)
    
    # Contar archivos con heatmaps
    with_heatmaps = 0
    without_heatmaps = 0
    
    for pt_file in pt_files:
        try:
            data = torch.load(pt_file, weights_only=True)
            if "target_heatmaps" in data:
                with_heatmaps += 1
            else:
                without_heatmaps += 1
        except:
            pass
    
    logger.info(f"Total archivos:              {len(pt_files)}")
    logger.info(f"Con target_heatmaps:         {with_heatmaps} ({with_heatmaps/len(pt_files)*100:.1f}%)")
    logger.info(f"Sin target_heatmaps:         {without_heatmaps} ({without_heatmaps/len(pt_files)*100:.1f}%)")
    
    if without_heatmaps > 0:
        logger.warning(
            f"\n⚠ {without_heatmaps} archivos sin heatmaps. "
            f"Ejecuta add_heatmaps_to_latents_v2.py antes de entrenar con HRNet loss."
        )
    else:
        logger.info(f"\n✓ Todos los archivos tienen heatmaps. Listo para entrenar!")


def main():
    parser = argparse.ArgumentParser(
        description='Verificar formato de archivos .pt de latentes'
    )
    parser.add_argument(
        '--latents_dir',
        type=str,
        required=True,
        help='Directorio con latentes pre-computados'
    )
    parser.add_argument(
        '--split',
        type=str,
        default='train',
        choices=['train', 'val', 'test'],
        help='Split a verificar (default: train)'
    )
    parser.add_argument(
        '--num_samples',
        type=int,
        default=5,
        help='Número de archivos a verificar en detalle (default: 5)'
    )
    
    args = parser.parse_args()
    
    check_pt_files(args.latents_dir, args.split, args.num_samples)


if __name__ == '__main__':
    main()
