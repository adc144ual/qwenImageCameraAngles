import json
import logging
from pathlib import Path
from typing import Dict, List, Tuple
from tqdm import tqdm

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

    Si se proporcionan synth_latents_dir y synth_timestamps, los .pt sintéticos
    se añaden a self.files junto con los reales (solo para train).
    Si no se proporcionan, el comportamiento es idéntico al original.
    """

    def __init__(self, latents_dir: str, timestamps: List[int], split: str = "train",
                 synth_latents_dir: str | None = None,
                 synth_timestamps: List[int] | None = None):
        self.latents_dir = Path(latents_dir)
        self.split = split

        # Directorio sintético (None si no se usa → backward compatible)
        self.synth_latents_dir = Path(synth_latents_dir) if synth_latents_dir else None

        # Construir índice timestamp → [.pt paths]
        self.ts_index = self._build_index(split)

        # Índice sintético: solo si se proporcionan ambos parámetros
        self.synth_ts_index: Dict[int, List[Path]] = {}
        if self.synth_latents_dir is not None and synth_timestamps:
            self.synth_ts_index = self._build_synth_index()

        # Expandir timestamps reales a lista de .pt files
        self.files = []
        missing_ts = 0
        for ts in timestamps:
            pts = self.ts_index.get(ts, [])
            if not pts:
                missing_ts += 1
            self.files.extend(pts)

        if missing_ts > 0:
            logger.warning(f"[{split}] {missing_ts} timestamps sin .pt encontrado")

        # Expandir timestamps sintéticos y añadirlos a self.files
        missing_synth = 0
        if synth_timestamps:
            for ts in synth_timestamps:
                pts = self.synth_ts_index.get(ts, [])
                if not pts:
                    missing_synth += 1
                self.files.extend(pts)
            if missing_synth > 0:
                logger.warning(f"[{split}] {missing_synth} timestamps sintéticos sin .pt encontrado")
            logger.info(f"[{split}] {len(self.files)} .pt files totales "
                        f"({len(timestamps)} ts reales + {len(synth_timestamps)} ts synth)")
        else:
            logger.info(f"[{split}] {len(self.files)} .pt files para {len(timestamps)} timestamps")

        # Calcular global_max_seq_len
        logger.info(f"[{split}] Calculando global_max_seq_len...")
        self.global_max_seq_len = 0
        for f in self.files:
            data = torch.load(f, weights_only=True)
            seq_len = data["prompt_embeds"].shape[1]
            if seq_len > self.global_max_seq_len:
                self.global_max_seq_len = seq_len
        logger.info(f"[{split}] global_max_seq_len = {self.global_max_seq_len}")

    def _build_index(self, split: str) -> Dict[int, List[Path]]:
        cache_path = self.latents_dir / f"ts_index_cache_{split}.json"

        if cache_path.exists():
            logger.info(f"[{split}] Cargando índice desde caché: {cache_path}")
            with open(cache_path) as f:
                raw = json.load(f)
            index = {int(k): [self.latents_dir / p for p in v] for k, v in raw.items()}
            logger.info(f"[{split}] Índice cargado: {len(index)} timestamps únicos")
            return index

        # Si no hay caché, construir leyendo .pt (lento)
        logger.warning(f"[{split}] Caché no encontrado en {cache_path}, construyendo índice leyendo .pt...")
        index = {}
        dirs_to_scan = [self.latents_dir / "test"] if split == "test" else [
            self.latents_dir / "train",
            self.latents_dir / "val",
        ]
        for scan_dir in dirs_to_scan:
            if not scan_dir.exists():
                logger.warning(f"Directorio no encontrado: {scan_dir}")
                continue
            for pt_path in tqdm(list(scan_dir.glob("*.pt")), desc=f"Indexando {scan_dir.name}"):
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

        # Guardar caché para próximas ejecuciones
        with open(cache_path, "w") as f:
            json.dump({k: [str(p) for p in v] for k, v in index.items()}, f)
        logger.info(f"[{split}] Caché guardado en {cache_path}")
        logger.info(f"[{split}] Índice construido: {len(index)} timestamps únicos")
        return index

    def _build_synth_index(self) -> Dict[int, List[Path]]:
        """
        Construye el índice timestamp → [.pt paths] para los sintéticos.
        Escanea synth_latents_dir/train/ (los sintéticos solo existen en train).
        Usa caché igual que _build_index para no releer en cada ejecución.
        """
        cache_path = self.synth_latents_dir / "ts_index_cache_synth.json"

        if cache_path.exists():
            logger.info(f"[synth] Cargando índice sintético desde caché: {cache_path}")
            with open(cache_path) as f:
                raw = json.load(f)
            index = {int(k): [self.synth_latents_dir / p for p in v] for k, v in raw.items()}
            logger.info(f"[synth] Índice cargado: {len(index)} timestamps únicos")
            return index

        logger.warning(f"[synth] Caché no encontrado en {cache_path}, construyendo índice leyendo .pt...")
        index = {}
        scan_dir = self.synth_latents_dir / "train"
        if not scan_dir.exists():
            logger.warning(f"[synth] Directorio no encontrado: {scan_dir}")
            return index

        for pt_path in tqdm(list(scan_dir.glob("*.pt")), desc="Indexando synth/train"):
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

        # Guardar caché
        with open(cache_path, "w") as f:
            json.dump({k: [str(p) for p in v] for k, v in index.items()}, f)
        logger.info(f"[synth] Caché guardado en {cache_path}")
        logger.info(f"[synth] Índice construido: {len(index)} timestamps únicos")
        return index

    def __len__(self) -> int:
        return len(self.files)

    def __getitem__(self, idx: int):
        return torch.load(self.files[idx], weights_only=True)


def load_experiment_json(experiment_json: str) -> Tuple[List[int], List[int], List[int], List[int]]:
    """
    Lee el JSON de experimento y devuelve (train_timestamps, val_timestamps, test_timestamps, train_synth_timestamps).
    Cada timestamp aparece una vez aunque tenga 6 .pt asociados; el dataset los expande.
    train_synth_timestamps estará vacío si el JSON no contiene el campo "synth" → backward compatible.
    """
    with open(experiment_json) as f:
        exp = json.load(f)

    def extract_timestamps(split_list):
        ts = []
        for item in split_list:
            if "timestamps" in item:
                ts.extend([int(t) for t in item["timestamps"]])
            elif "ts" in item:
                ts.append(int(item["ts"]))  # forzar a int
        return ts

    def extract_synth_timestamps(split_list):
        """Extrae timestamps del campo 'synth' de cada usuario. Vacío si no existe."""
        ts = []
        for item in split_list:
            synth = item.get("synth")
            if synth and "timestamps" in synth:
                ts.extend([int(t) for t in synth["timestamps"]])
        return ts

    train_ts       = extract_timestamps(exp.get("train", []))
    val_ts         = extract_timestamps(exp.get("val",   []))
    test_ts        = extract_timestamps(exp.get("test",  []))
    train_synth_ts = extract_synth_timestamps(exp.get("train", []))

    logger.info(f"Experimento: {exp.get('name', 'sin nombre')}")
    logger.info(f"Descripción: {exp.get('description', '')}")
    logger.info(f"Timestamps → train: {len(train_ts)} | synth: {len(train_synth_ts)} "
                f"| val: {len(val_ts)} | test: {len(test_ts)}")

    return train_ts, val_ts, test_ts, train_synth_ts


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
                logger.debug("target_heatmaps no encontrado en batch item, usando ceros")
                target_heatmaps_list.append(torch.zeros(27, 72, 96, dtype=torch.float32))

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
    train_ts, val_ts, test_ts, train_synth_ts = load_experiment_json(config.experiment_json)

    # synth_latents_dir es None si no está definido en config → backward compatible
    synth_dir = getattr(config, "synth_latents_dir", None)

    train_dataset = LatentsDatasetFromJSON(
        config.latents_dir, train_ts, split="train",
        synth_latents_dir=synth_dir,
        synth_timestamps=train_synth_ts if train_synth_ts else None,
    )
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