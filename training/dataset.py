import json
import logging
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader, Subset

logger = logging.getLogger(__name__)


class LatentsDatasetFromJSON(Dataset):
    """
    Dataset que carga .pt a partir de un JSON de experimento.
    Cada timestamp del JSON puede tener hasta 6 .pt asociados.
    Train y val comparten el mismo índice (carpetas train/ y val/ se tratan como una sola).
    Test se busca exclusivamente en la carpeta test/.
    """

    def __init__(self, latents_dir: str, timestamps: List[int], split: str = "train"):
        self.latents_dir = Path(latents_dir)
        self.split = split

        # Construir índice timestamp → [.pt paths]
        self.ts_index = self._build_index(split)

        # Expandir timestamps a lista de .pt files
        self.files = []
        missing_ts = 0
        for ts in timestamps:
            pts = self.ts_index.get(ts, [])
            if not pts:
                missing_ts += 1
            self.files.extend(pts)

        if missing_ts > 0:
            logger.warning(f"[{split}] {missing_ts} timestamps sin .pt encontrado")

        # Calcular global_max_seq_len
        logger.info(f"[{split}] {len(self.files)} .pt files para {len(timestamps)} timestamps")
        logger.info(f"[{split}] Calculando global_max_seq_len...")
        self.global_max_seq_len = 0
        for f in self.files:
            data = torch.load(f, weights_only=True)
            seq_len = data["prompt_embeds"].shape[1]
            if seq_len > self.global_max_seq_len:
                self.global_max_seq_len = seq_len
        logger.info(f"[{split}] global_max_seq_len = {self.global_max_seq_len}")

    def _build_index(self, split: str) -> Dict[int, List[Path]]:
        """Escanea los directorios correspondientes y construye timestamp → [paths]."""
        index = {}

        if split == "test":
            dirs_to_scan = [self.latents_dir / "test"]
        else:
            # train y val comparten carpetas
            dirs_to_scan = [
                self.latents_dir / "train",
                self.latents_dir / "val",
            ]

        for scan_dir in dirs_to_scan:
            if not scan_dir.exists():
                logger.warning(f"Directorio no encontrado: {scan_dir}")
                continue
            for pt_path in scan_dir.glob("*.pt"):
                try:
                    data = torch.load(pt_path, map_location="cpu", weights_only=False)
                    ts = data.get("timestamp")
                    if ts is not None:
                        ts = int(ts)
                        if ts not in index:
                            index[ts] = []
                        index[ts].append(pt_path)
                except Exception as e:
                    logger.warning(f"Error leyendo {pt_path}: {e}")

        logger.info(f"[{split}] Índice construido: {len(index)} timestamps únicos")
        return index

    def __len__(self) -> int:
        return len(self.files)

    def __getitem__(self, idx: int):
        return torch.load(self.files[idx], weights_only=True)



def load_experiment_json(experiment_json: str) -> Tuple[List[int], List[int], List[int]]:
    """
    Lee el JSON de experimento y devuelve (train_timestamps, val_timestamps, test_timestamps).
    Cada timestamp aparece una vez aunque tenga 6 .pt asociados; el dataset los expande.
    """
    with open(experiment_json) as f:
        exp = json.load(f)

    def extract_timestamps(split_list):
        ts = []
        for user in split_list:
            ts.extend(user["timestamps"])
        return ts

    train_ts = extract_timestamps(exp.get("train", []))
    val_ts   = extract_timestamps(exp.get("val",   []))
    test_ts  = extract_timestamps(exp.get("test",  []))

    logger.info(f"Experimento: {exp.get('name', 'sin nombre')}")
    logger.info(f"Descripción: {exp.get('description', '')}")
    logger.info(f"Timestamps → train: {len(train_ts)} | val: {len(val_ts)} | test: {len(test_ts)}")

    return train_ts, val_ts, test_ts


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

def build_dataloaders(config):
    """Construye datasets y dataloaders desde el JSON de experimento."""
    train_ts, val_ts, test_ts = load_experiment_json(config.experiment_json)

    train_dataset = LatentsDatasetFromJSON(config.latents_dir, train_ts, split="train")
    val_dataset   = LatentsDatasetFromJSON(config.latents_dir, val_ts,   split="val")
    test_dataset  = LatentsDatasetFromJSON(config.latents_dir, test_ts,  split="test")

    # global_max_seq_len unificado entre los tres splits
    global_max_seq_len = max(
        train_dataset.global_max_seq_len,
        val_dataset.global_max_seq_len,
        test_dataset.global_max_seq_len,
    )
    logger.info(f"global_max_seq_len unificado: {global_max_seq_len}")

    collate_fn = make_collate_latents(global_max_seq_len)

    g = torch.Generator()
    g.manual_seed(42)

    train_dataloader = DataLoader(
        train_dataset,
        batch_size=config.batch_size,
        collate_fn=collate_fn,
        drop_last=True,
        shuffle=True,
        generator=g,
        num_workers=4,
    )
    val_dataloader = DataLoader(
        val_dataset,
        batch_size=config.batch_size,
        collate_fn=collate_fn,
        drop_last=False,
        shuffle=False,
        num_workers=4,
    )
    test_dataloader = DataLoader(
        test_dataset,
        batch_size=config.batch_size,
        collate_fn=collate_fn,
        drop_last=False,
        shuffle=False,
        num_workers=4,
    )

    return train_dataset, val_dataset, test_dataset, train_dataloader, val_dataloader, test_dataloader