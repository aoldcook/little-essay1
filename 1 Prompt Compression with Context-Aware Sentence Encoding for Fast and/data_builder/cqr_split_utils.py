from __future__ import annotations

import hashlib
import json
import math
import random
from collections import Counter, defaultdict
from pathlib import Path
from typing import DefaultDict, Dict, Iterable, List, Sequence, Tuple


JsonDict = Dict[str, object]


def load_jsonl(path: Path) -> List[dict]:
    rows: List[dict] = []
    with path.open('r', encoding='utf-8-sig') as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def save_jsonl(path: Path, rows: Iterable[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('w', encoding='utf-8') as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + '\n')


def source_dataset(row: dict) -> str:
    explicit = str(row.get('dataset') or '').strip()
    if explicit:
        return explicit
    metadata = row.get('metadata') or {}
    if isinstance(metadata, dict):
        source_dataset_name = str(metadata.get('source_dataset') or '').strip()
        if source_dataset_name:
            return source_dataset_name
    source = row.get('source') or {}
    if isinstance(source, dict):
        source_dataset_name = str(source.get('dataset') or source.get('source_dataset') or '').strip()
        if source_dataset_name:
            return source_dataset_name
    return 'english_cqr'


def quality_label(row: dict) -> str:
    metadata = row.get('metadata') or {}
    if isinstance(metadata, dict):
        quality = str(metadata.get('quality') or '').strip()
        if quality:
            return quality
    source = row.get('source') or {}
    if isinstance(source, dict) and source.get('relaxed_salvage'):
        return 'silver'
    return 'gold'


def stable_group_id(row: dict) -> str:
    dataset = source_dataset(row)
    for key in ('source_id', 'raw_id', 'sample_id', 'id'):
        value = str(row.get(key) or '').strip()
        if value:
            return f'{dataset}::{value}'
    source = row.get('source') or {}
    if isinstance(source, dict):
        for key in ('raw_id', 'sample_id', 'id'):
            value = str(source.get(key) or '').strip()
            if value:
                return f'{dataset}::{value}'

    payload = {
        'question': str(row.get('question') or '').strip(),
        'context': str(row.get('context') or '').strip(),
        'answer': str(row.get('answer') or '').strip(),
    }
    digest = hashlib.md5(json.dumps(payload, ensure_ascii=False, sort_keys=True).encode('utf-8')).hexdigest()
    return f'{dataset}::hash::{digest}'


def dedupe_key(row: dict) -> str:
    payload = {
        'dataset': source_dataset(row),
        'question': str(row.get('question') or '').strip(),
        'context': str(row.get('context') or '').strip(),
        'positive_sentence': str(row.get('positive_sentence') or '').strip(),
        'answer': str(row.get('answer') or '').strip(),
        'negative_sentences': [str(item).strip() for item in (row.get('negative_sentences') or [])],
        'supporting_sentences': [str(item).strip() for item in (row.get('supporting_sentences') or [])],
        'quality': quality_label(row),
    }
    return hashlib.md5(json.dumps(payload, ensure_ascii=False, sort_keys=True).encode('utf-8')).hexdigest()


def dedupe_rows(rows: Sequence[dict]) -> Tuple[List[dict], int]:
    deduped: List[dict] = []
    seen = set()
    removed = 0
    for row in rows:
        key = dedupe_key(row)
        if key in seen:
            removed += 1
            continue
        seen.add(key)
        deduped.append(row)
    return deduped, removed


def bucket_key(row: dict, stratify_by_dataset: bool = True, stratify_by_quality: bool = True) -> str:
    parts: List[str] = []
    if stratify_by_dataset:
        parts.append(source_dataset(row))
    if stratify_by_quality:
        parts.append(quality_label(row))
    return '::'.join(parts) if parts else 'all'


def group_rows(
    rows: Sequence[dict],
    stratify_by_dataset: bool = True,
    stratify_by_quality: bool = True,
) -> Dict[str, List[List[dict]]]:
    grouped: DefaultDict[str, List[dict]] = defaultdict(list)
    for row in rows:
        grouped[stable_group_id(row)].append(row)

    buckets: DefaultDict[str, List[List[dict]]] = defaultdict(list)
    for group in grouped.values():
        key = bucket_key(group[0], stratify_by_dataset=stratify_by_dataset, stratify_by_quality=stratify_by_quality)
        buckets[key].append(group)
    return dict(buckets)


def allocate_counts(total: int, ratios: Sequence[float]) -> List[int]:
    raw = [ratio * total for ratio in ratios]
    counts = [math.floor(value) for value in raw]
    remaining = total - sum(counts)
    order = sorted(range(len(ratios)), key=lambda idx: raw[idx] - counts[idx], reverse=True)
    for idx in order[:remaining]:
        counts[idx] += 1
    return counts


def split_rows_by_group(
    rows: Sequence[dict],
    train_ratio: float,
    dev_ratio: float,
    test_ratio: float,
    seed: int = 42,
    stratify_by_dataset: bool = True,
    stratify_by_quality: bool = True,
) -> Dict[str, List[dict]]:
    ratio_sum = train_ratio + dev_ratio + test_ratio
    if abs(ratio_sum - 1.0) > 1e-6:
        raise ValueError(f'train/dev/test ratios must sum to 1.0, got {ratio_sum:.6f}')

    rng = random.Random(seed)
    buckets = group_rows(rows, stratify_by_dataset=stratify_by_dataset, stratify_by_quality=stratify_by_quality)
    splits = {'train': [], 'dev': [], 'test': []}

    for key, groups in buckets.items():
        shuffled = list(groups)
        rng.shuffle(shuffled)
        train_count, dev_count, test_count = allocate_counts(len(shuffled), (train_ratio, dev_ratio, test_ratio))
        assignments = (('train', train_count), ('dev', dev_count), ('test', test_count))
        cursor = 0
        for split_name, count in assignments:
            for group in shuffled[cursor : cursor + count]:
                splits[split_name].extend(group)
            cursor += count
        if cursor != len(shuffled):
            raise RuntimeError(f'bucket split mismatch for {key}')

    return splits


def summarize_rows(rows: Sequence[dict]) -> dict:
    return {
        'num_rows': len(rows),
        'dataset_counts': dict(Counter(source_dataset(row) for row in rows)),
        'quality_counts': dict(Counter(quality_label(row) for row in rows)),
    }


def summarize_splits(splits: Dict[str, Sequence[dict]]) -> dict:
    return {split_name: summarize_rows(rows) for split_name, rows in splits.items()}
