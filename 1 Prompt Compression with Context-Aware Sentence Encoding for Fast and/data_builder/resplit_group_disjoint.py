"""Rebuild train/dev/test as GROUP-DISJOINT splits over the existing 5k pool.

Why this exists
---------------
The shipped split was drawn at the row level. The pool has 5000 rows but only
~3178 distinct source_ids and ~2853 distinct contexts, so the same passage
appears in several rows. Row-level sampling therefore put 13.1% of test rows on
a context that also occurs in training. For a CONTEXT COMPRESSION task that is
straightforward leakage: the Stage-1 encoder is fine-tuned on those passages,
so test-time selection is being scored on text the model was trained to encode.

This is the example-level analogue of EVAL_VALIDITY_AUDIT.md finding C4, which
covered span-level leakage inside the span model.

What it does
------------
Groups rows by the transitive closure of shared source_id and shared context
(two rows are in the same group if they share either), then assigns whole
groups to splits. A context can therefore never straddle the boundary. Group
assignment is stratified by dataset so the mix is preserved, and is
deterministic given --seed.

The default test target is 1000 rows: at 244 rows, EM/F1 differences below
roughly 5 points are inside the noise band for the ratio x seed x method grid,
which is not enough resolution for the headline table.

Usage:
    python -m data_builder.resplit_group_disjoint \
        --input_file data_builder/english_cqr_mixed_5k/all.jsonl \
        --output_dir data_builder/english_cqr_mixed_5k_grouped \
        --test_rows 1000 --dev_rows 500 --seed 42
"""

from __future__ import annotations

import argparse
import collections
import hashlib
import json
import random
from pathlib import Path
from typing import Dict, List, Sequence, Tuple


def load_jsonl(path: Path) -> List[dict]:
    rows = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def context_key(row: dict) -> str:
    """Normalised context hash: whitespace differences must not split a group."""
    text = " ".join(str(row.get("context") or "").split())
    return "ctx:" + hashlib.sha1(text.encode("utf-8")).hexdigest()


class UnionFind:
    def __init__(self) -> None:
        self.parent: Dict[str, str] = {}

    def find(self, item: str) -> str:
        self.parent.setdefault(item, item)
        root = item
        while self.parent[root] != root:
            root = self.parent[root]
        while self.parent[item] != root:  # path compression
            self.parent[item], item = root, self.parent[item]
        return root

    def union(self, a: str, b: str) -> None:
        ra, rb = self.find(a), self.find(b)
        if ra != rb:
            self.parent[rb] = ra


def build_groups(rows: Sequence[dict]) -> List[List[int]]:
    """Transitive closure over shared source_id and shared context."""
    uf = UnionFind()
    for idx, row in enumerate(rows):
        row_key = f"row:{idx}"
        uf.union(row_key, context_key(row))
        source_id = row.get("source_id")
        if source_id:
            uf.union(row_key, f"sid:{source_id}")

    buckets: Dict[str, List[int]] = collections.defaultdict(list)
    for idx in range(len(rows)):
        buckets[uf.find(f"row:{idx}")].append(idx)
    # Deterministic order: largest groups first, ties broken by first row index.
    return sorted(buckets.values(), key=lambda g: (-len(g), g[0]))


def dominant_dataset(rows: Sequence[dict], group: Sequence[int]) -> str:
    counts = collections.Counter(str(rows[i].get("dataset")) for i in group)
    return counts.most_common(1)[0][0]


def assign_groups(
    rows: Sequence[dict],
    groups: Sequence[Sequence[int]],
    test_rows: int,
    dev_rows: int,
    seed: int,
) -> Tuple[List[int], List[int], List[int], dict]:
    """Fill test then dev to their row targets, stratified by dataset.

    Groups are whole and indivisible, so the realised sizes land near the
    targets rather than exactly on them; the manifest records what was achieved.
    """
    by_dataset: Dict[str, List[Sequence[int]]] = collections.defaultdict(list)
    for group in groups:
        by_dataset[dominant_dataset(rows, group)].append(group)

    rng = random.Random(seed)
    for dataset in by_dataset:
        rng.shuffle(by_dataset[dataset])

    total = len(rows)
    test_idx: List[int] = []
    dev_idx: List[int] = []
    train_idx: List[int] = []

    for dataset, dataset_groups in sorted(by_dataset.items()):
        dataset_rows = sum(len(g) for g in dataset_groups)
        # Proportional quota so the dataset mix survives the split.
        test_quota = round(test_rows * dataset_rows / total)
        dev_quota = round(dev_rows * dataset_rows / total)

        filled_test = filled_dev = 0
        for group in dataset_groups:
            if filled_test < test_quota:
                test_idx.extend(group)
                filled_test += len(group)
            elif filled_dev < dev_quota:
                dev_idx.extend(group)
                filled_dev += len(group)
            else:
                train_idx.extend(group)

    stats = {
        "num_rows": total,
        "num_groups": len(groups),
        "largest_group": max(len(g) for g in groups) if groups else 0,
        "mean_group_size": round(total / max(len(groups), 1), 3),
        "target_test_rows": test_rows,
        "target_dev_rows": dev_rows,
        "seed": seed,
    }
    return sorted(train_idx), sorted(dev_idx), sorted(test_idx), stats


def verify_disjoint(rows: Sequence[dict], splits: Dict[str, Sequence[int]]) -> dict:
    """Hard check: no context and no source_id may cross a split boundary."""
    report: Dict[str, object] = {}
    keys = {
        "context": context_key,
        "source_id": lambda r: str(r.get("source_id") or ""),
    }
    for key_name, key_fn in keys.items():
        sets = {
            split: {key_fn(rows[i]) for i in idx if key_fn(rows[i])}
            for split, idx in splits.items()
        }
        for a, b in (("train", "test"), ("train", "dev"), ("dev", "test")):
            overlap = sets[a] & sets[b]
            report[f"{key_name}_{a}_{b}_overlap"] = len(overlap)
            if overlap:
                raise SystemExit(
                    f"LEAKAGE: {len(overlap)} {key_name} value(s) shared between "
                    f"{a} and {b}. The grouping is wrong; refusing to write splits."
                )
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input_file", type=str, required=True)
    parser.add_argument("--output_dir", type=str, required=True)
    parser.add_argument("--test_rows", type=int, default=1000)
    parser.add_argument("--dev_rows", type=int, default=500)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    rows = load_jsonl(Path(args.input_file))
    groups = build_groups(rows)
    train_idx, dev_idx, test_idx, stats = assign_groups(
        rows, groups, args.test_rows, args.dev_rows, args.seed
    )

    splits = {"train": train_idx, "dev": dev_idx, "test": test_idx}
    overlap_report = verify_disjoint(rows, splits)

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    for split, idx in splits.items():
        with (out_dir / f"{split}.jsonl").open("w", encoding="utf-8") as f:
            for i in idx:
                f.write(json.dumps(rows[i], ensure_ascii=False) + "\n")

    mix = {
        split: dict(collections.Counter(str(rows[i].get("dataset")) for i in idx))
        for split, idx in splits.items()
    }
    manifest = {
        "input_file": str(args.input_file),
        "grouping": "transitive closure over shared source_id and normalised context",
        "split_sizes": {split: len(idx) for split, idx in splits.items()},
        "dataset_mix": mix,
        "group_stats": stats,
        "disjointness_check": overlap_report,
    }
    (out_dir / "split_manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8"
    )

    print(json.dumps(manifest["split_sizes"], indent=2))
    print("groups:", stats["num_groups"], "largest:", stats["largest_group"])
    print("disjointness:", overlap_report)
    print("wrote:", out_dir)


if __name__ == "__main__":
    main()
