from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, List, Tuple

import matplotlib.pyplot as plt


def load_history(stage_dir: Path) -> List[Dict[str, float]]:
    history_file = stage_dir / 'training_history.json'
    if not history_file.exists():
        return []
    with history_file.open('r', encoding='utf-8') as f:
        return json.load(f)


def collect_loss_series(history: List[Dict[str, float]]) -> List[Tuple[str, List[float], List[float]]]:
    if not history:
        return []
    epochs = [float(item.get('epoch', idx + 1)) for idx, item in enumerate(history)]
    candidate_keys = [
        ('train_loss', 'train_loss'),
        ('train_total_loss', 'train_total_loss'),
        ('train_ctr_loss', 'train_ctr_loss'),
        ('train_mntp_loss', 'train_mntp_loss'),
    ]
    series: List[Tuple[str, List[float], List[float]]] = []
    for key, label in candidate_keys:
        values = [item.get(key) for item in history]
        if any(value is not None for value in values):
            clean_values = [float(value) if value is not None else float('nan') for value in values]
            series.append((label, epochs, clean_values))
    return series


def main() -> None:
    parser = argparse.ArgumentParser(description='Plot curriculum training loss curves from stage training_history.json files.')
    parser.add_argument('--run_dir', type=str, required=True)
    parser.add_argument('--output_file', type=str, default='')
    parser.add_argument('--title', type=str, default='Curriculum Training Loss')
    args = parser.parse_args()

    run_dir = Path(args.run_dir)
    output_file = Path(args.output_file) if args.output_file else run_dir / 'curriculum_loss.png'

    stages = [
        ('stage1_gold', 'Stage 1: Gold Warmup'),
        ('stage2_full', 'Stage 2: Full Finetune'),
    ]

    available = []
    for dirname, title in stages:
        stage_dir = run_dir / dirname
        history = load_history(stage_dir)
        if history:
            available.append((stage_dir, title, history))

    if not available:
        raise FileNotFoundError(f'No training_history.json found under {run_dir}')

    fig, axes = plt.subplots(len(available), 1, figsize=(10, 4 * len(available)), squeeze=False)
    fig.suptitle(args.title)

    for ax, (stage_dir, stage_title, history) in zip(axes[:, 0], available):
        series = collect_loss_series(history)
        if not series:
            ax.set_title(stage_title)
            ax.text(0.5, 0.5, 'No loss fields found', ha='center', va='center')
            ax.set_axis_off()
            continue

        for label, epochs, values in series:
            ax.plot(epochs, values, marker='o', linewidth=2, label=label)

        ax.set_title(stage_title)
        ax.set_xlabel('Epoch')
        ax.set_ylabel('Loss')
        ax.grid(True, alpha=0.3)
        ax.legend()

        best_metrics_file = stage_dir / 'best_metrics.json'
        if best_metrics_file.exists():
            with best_metrics_file.open('r', encoding='utf-8') as f:
                best_metrics = json.load(f)
            if 'epoch' in best_metrics:
                ax.axvline(float(best_metrics['epoch']), color='gray', linestyle='--', alpha=0.5)

    fig.tight_layout(rect=[0, 0, 1, 0.96])
    output_file.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_file, dpi=180, bbox_inches='tight')
    print(output_file)


if __name__ == '__main__':
    main()
