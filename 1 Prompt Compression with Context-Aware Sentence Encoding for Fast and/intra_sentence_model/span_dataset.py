from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, List, Sequence, Tuple

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


def group_key(row: dict) -> str:
    """Group identifier for leakage-free splitting.

    All spans generated from the same source example must land in the same split.
    `source_id` is written by generate_span_pseudo_labels.py; `example_id` has the
    form "<source_id>::<sent_idx>" so its prefix is a safe fallback.
    """
    source_id = str(row.get("source_id") or "").strip()
    if source_id:
        return source_id
    example_id = str(row.get("example_id") or "").strip()
    if example_id:
        return example_id.split("::")[0]
    return str(row.get("question") or "")[:256]


def split_rows_group_disjoint(
    rows: Sequence[dict],
    dev_ratio: float,
    seed: int = 42,
) -> Tuple[List[dict], List[dict]]:
    """Split at the SOURCE-EXAMPLE level, never at the span level.

    Fixes EVAL_VALIDITY_AUDIT.md finding C4: the previous implementation flattened
    every span from every row into one pool and then shuffled, so spans from the
    same sentence appeared in both train and dev. Dev was not independent of
    train and reported dev accuracy was optimistic.
    """
    import random as _random

    buckets: Dict[str, List[dict]] = {}
    for row in rows:
        buckets.setdefault(group_key(row), []).append(row)

    group_ids = sorted(buckets)
    _random.Random(seed).shuffle(group_ids)

    dev_group_count = int(len(group_ids) * dev_ratio)
    dev_ids = set(group_ids[:dev_group_count])

    train_rows: List[dict] = []
    dev_rows: List[dict] = []
    for group_id in group_ids:
        (dev_rows if group_id in dev_ids else train_rows).extend(buckets[group_id])

    # Invariant: a group never straddles the split.
    assert not ({group_key(r) for r in train_rows} & {group_key(r) for r in dev_rows}), (
        "group leakage detected between train and dev"
    )
    return train_rows, dev_rows


def build_xy_group_disjoint(
    rows: Sequence[dict],
    dev_ratio: float,
    seed: int = 42,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, Dict[str, int]]:
    """Group-disjoint train/dev tensors plus split statistics."""
    train_rows, dev_rows = split_rows_group_disjoint(rows, dev_ratio, seed)
    X_train, y_train = build_xy(train_rows)
    X_dev, y_dev = build_xy(dev_rows) if dev_rows else (np.empty((0, X_train.shape[1]), dtype=np.float32), np.empty((0,), dtype=np.float32))
    stats = {
        "num_source_groups": len({group_key(r) for r in rows}),
        "train_groups": len({group_key(r) for r in train_rows}),
        "dev_groups": len({group_key(r) for r in dev_rows}),
        "train_rows": len(train_rows),
        "dev_rows": len(dev_rows),
        "train_spans": int(len(X_train)),
        "dev_spans": int(len(X_dev)),
    }
    return X_train, y_train, X_dev, y_dev, stats
