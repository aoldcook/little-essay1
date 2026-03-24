from __future__ import annotations

import json
from pathlib import Path
from typing import List, Sequence, Tuple

import numpy as np
from torch.utils.data import Dataset

from intra_sentence_model.span_feature_utils import FEATURE_ORDER, features_to_vector


class SpanInstanceDataset(Dataset):
    def __init__(self, rows: Sequence[dict]):
        self.instances = []
        for row in rows:
            for span in row.get("spans", []):
                features = span.get("features")
                label = span.get("label")
                if features is None or label is None:
                    continue
                self.instances.append(
                    {
                        "x": np.asarray(features_to_vector(features), dtype=np.float32),
                        "y": np.float32(label),
                    }
                )

    def __len__(self) -> int:
        return len(self.instances)

    def __getitem__(self, idx: int):
        item = self.instances[idx]
        return item["x"], item["y"]


def load_jsonl(path: Path) -> List[dict]:
    rows = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def build_xy(rows: Sequence[dict]) -> Tuple[np.ndarray, np.ndarray]:
    dataset = SpanInstanceDataset(rows)
    if len(dataset) == 0:
        raise ValueError("no span instances found in dataset")
    xs, ys = [], []
    for x, y in dataset:
        xs.append(x)
        ys.append(y)
    return np.stack(xs), np.asarray(ys, dtype=np.float32)


def build_feature_metadata() -> List[str]:
    return FEATURE_ORDER
