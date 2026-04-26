from __future__ import annotations

import argparse
import json
import random
from pathlib import Path
from typing import List

from cqr_split_utils import (
    dedupe_rows,
    load_jsonl,
    quality_label,
    save_jsonl,
    split_rows_by_group,
    summarize_rows,
)


def sample_rows(rows: List[dict], max_rows: int, seed: int) -> List[dict]:
    if max_rows <= 0 or len(rows) <= max_rows:
        return list(rows)
    rng = random.Random(seed)
    sampled = list(rows)
    rng.shuffle(sampled)
    return sampled[:max_rows]


def main() -> None:
    parser = argparse.ArgumentParser(description='Prepare gold/silver/train/dev/test CQR splits from a merged high-recall JSONL file.')
    parser.add_argument('--input_file', type=str, required=True)
    parser.add_argument('--output_dir', type=str, required=True)
    parser.add_argument('--train_ratio', type=float, default=0.8)
    parser.add_argument('--dev_ratio', type=float, default=0.1)
    parser.add_argument('--test_ratio', type=float, default=0.1)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--max_silver_train', type=int, default=0)
    parser.add_argument('--disable_dedupe', action='store_true')
    args = parser.parse_args()

    rows = load_jsonl(Path(args.input_file))
    deduped = rows
    removed_duplicates = 0
    if not args.disable_dedupe:
        deduped, removed_duplicates = dedupe_rows(rows)

    gold_rows = [row for row in deduped if quality_label(row) == 'gold']
    silver_rows = [row for row in deduped if quality_label(row) == 'silver']

    gold_splits = split_rows_by_group(
        gold_rows,
        train_ratio=args.train_ratio,
        dev_ratio=args.dev_ratio,
        test_ratio=args.test_ratio,
        seed=args.seed,
        stratify_by_dataset=True,
        stratify_by_quality=False,
    )

    silver_train_rows = sample_rows(silver_rows, args.max_silver_train, args.seed + 17)
    train_rows = list(gold_splits['train']) + list(silver_train_rows)

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    save_jsonl(output_dir / 'gold.jsonl', gold_rows)
    save_jsonl(output_dir / 'silver.jsonl', silver_rows)
    save_jsonl(output_dir / 'train_gold.jsonl', gold_splits['train'])
    save_jsonl(output_dir / 'train.jsonl', train_rows)
    save_jsonl(output_dir / 'dev.jsonl', gold_splits['dev'])
    save_jsonl(output_dir / 'test.jsonl', gold_splits['test'])

    summary = {
        'input_file': args.input_file,
        'output_dir': str(output_dir),
        'seed': args.seed,
        'ratios': {
            'train': args.train_ratio,
            'dev': args.dev_ratio,
            'test': args.test_ratio,
        },
        'removed_duplicates': removed_duplicates,
        'silver_train_policy': {
            'all_silver_to_train': args.max_silver_train <= 0,
            'max_silver_train': args.max_silver_train,
            'selected_silver_train': len(silver_train_rows),
        },
        'counts': {
            'all_rows': summarize_rows(deduped),
            'gold_rows': summarize_rows(gold_rows),
            'silver_rows': summarize_rows(silver_rows),
            'train_gold': summarize_rows(gold_splits['train']),
            'train': summarize_rows(train_rows),
            'dev': summarize_rows(gold_splits['dev']),
            'test': summarize_rows(gold_splits['test']),
        },
        'files': {
            'gold': str(output_dir / 'gold.jsonl'),
            'silver': str(output_dir / 'silver.jsonl'),
            'train_gold': str(output_dir / 'train_gold.jsonl'),
            'train': str(output_dir / 'train.jsonl'),
            'dev': str(output_dir / 'dev.jsonl'),
            'test': str(output_dir / 'test.jsonl'),
        },
    }
    with (output_dir / 'prepare_summary.json').open('w', encoding='utf-8') as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)

    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == '__main__':
    main()
