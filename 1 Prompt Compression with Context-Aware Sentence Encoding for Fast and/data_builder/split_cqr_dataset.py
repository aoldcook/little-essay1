from __future__ import annotations

import argparse
import json
from pathlib import Path

from cqr_split_utils import dedupe_rows, load_jsonl, save_jsonl, split_rows_by_group, summarize_rows, summarize_splits


def main() -> None:
    parser = argparse.ArgumentParser(description='Split a CQR JSONL file into grouped train/dev/test splits.')
    parser.add_argument('--input_file', type=str, required=True)
    parser.add_argument('--output_dir', type=str, required=True)
    parser.add_argument('--train_ratio', type=float, default=0.8)
    parser.add_argument('--dev_ratio', type=float, default=0.1)
    parser.add_argument('--test_ratio', type=float, default=0.1)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--disable_dedupe', action='store_true')
    parser.add_argument('--disable_dataset_stratify', action='store_true')
    parser.add_argument('--disable_quality_stratify', action='store_true')
    args = parser.parse_args()

    rows = load_jsonl(Path(args.input_file))
    deduped = rows
    removed_duplicates = 0
    if not args.disable_dedupe:
        deduped, removed_duplicates = dedupe_rows(rows)

    splits = split_rows_by_group(
        deduped,
        train_ratio=args.train_ratio,
        dev_ratio=args.dev_ratio,
        test_ratio=args.test_ratio,
        seed=args.seed,
        stratify_by_dataset=not args.disable_dataset_stratify,
        stratify_by_quality=not args.disable_quality_stratify,
    )

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    save_jsonl(output_dir / 'train.jsonl', splits['train'])
    save_jsonl(output_dir / 'dev.jsonl', splits['dev'])
    save_jsonl(output_dir / 'test.jsonl', splits['test'])

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
        'input_summary': summarize_rows(rows),
        'deduped_summary': summarize_rows(deduped),
        'split_summary': summarize_splits(splits),
    }
    with (output_dir / 'split_summary.json').open('w', encoding='utf-8') as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)

    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == '__main__':
    main()
