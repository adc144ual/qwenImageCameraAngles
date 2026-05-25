import logging
from pathlib import Path
from typing import Tuple

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader, Subset

logger = logging.getLogger(__name__)


class LatentsDataset(Dataset):
    def __init__(self, latents_dir, split="train"):
        self.split_dir = Path(latents_dir) / split
        self.files = sorted(list(self.split_dir.glob("*.pt")))
        if len(self.files) == 0:
            logger.warning(f"No files found in {self.split_dir}")

        logger.info(f"Scanning {len(self.files)} files to compute global_max_seq_len...")
        self.global_max_seq_len = 0
        for f in self.files:
            data = torch.load(f, weights_only=True)
            seq_len = data["prompt_embeds"].shape[1]
            if seq_len > self.global_max_seq_len:
                self.global_max_seq_len = seq_len
        logger.info(f"global_max_seq_len = {self.global_max_seq_len}")

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        return torch.load(self.files[idx], weights_only=True)


def create_train_val_split(dataset: Dataset, val_split: float, seed: int = 42) -> Tuple[Subset, Subset]:
    """
    NUEVO V1: Crea splits train/val reproducibles con semilla fija.

    Args:
        dataset: Dataset completo
        val_split: Fracción de datos para validación (0.0-1.0)
        seed: Semilla para reproducibilidad

    Returns:
        train_subset, val_subset
    """
    dataset_size = len(dataset)
    indices = list(range(dataset_size))

    # Mezclar con semilla fija para reproducibilidad
    np.random.seed(seed)
    np.random.shuffle(indices)

    # Calcular punto de split
    split_idx = int(np.floor(val_split * dataset_size))

    train_indices = indices[split_idx:]
    val_indices = indices[:split_idx]

    train_subset = Subset(dataset, train_indices)
    val_subset = Subset(dataset, val_indices)

    logger.info(f"✓ Dataset split (seed={seed}): {len(train_subset)} train, {len(val_subset)} val")

    return train_subset, val_subset


def make_collate_latents(global_max_seq_len: int):
    """Collate que padea prompt_embeds y extrae target_heatmaps."""
    def collate_latents(batch):
        target_latents  = torch.cat([item["target_latents_packed"]  for item in batch], dim=0)
        source_latents  = torch.cat([item["source_latents_packed"]  for item in batch], dim=0)
        prompt_list     = [item["prompt_embeds"]      for item in batch]
        mask_list       = [item["prompt_embeds_mask"] for item in batch]

        # Extraer heatmaps GT
        target_heatmaps_list = []
        for item in batch:
            if "target_heatmaps" in item:
                target_heatmaps_list.append(item["target_heatmaps"])
            else:
                logger.warning("target_heatmaps no encontrado en batch item, usando ceros")
                target_heatmaps_list.append(torch.zeros(17, 72, 96, dtype=torch.float32))

        target_heatmaps = torch.cat(target_heatmaps_list, dim=0)

        padded_embeds, padded_masks = [], []
        for pe, pm in zip(prompt_list, mask_list):
            curr_len = pe.shape[1]
            if curr_len < global_max_seq_len:
                pad_len = global_max_seq_len - curr_len
                pe = F.pad(pe, (0, 0, 0, pad_len), value=0.0)
                pm = F.pad(pm, (0, pad_len),        value=0)
            padded_embeds.append(pe)
            padded_masks.append(pm)

        return {
            "target_latents_packed": target_latents,
            "source_latents_packed": source_latents,
            "prompt_embeds":         torch.cat(padded_embeds, dim=0),
            "prompt_embeds_mask":    torch.cat(padded_masks,  dim=0),
            "target_heatmaps":       target_heatmaps,
        }
    return collate_latents