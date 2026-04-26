from __future__ import annotations

import argparse
import json
import random
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
from torch.utils.data import DataLoader

from context_aware_encoder_model.context_aware_sentence_encoder import (
    ContextAwareEncoderConfig,
    ContextAwareSentenceEncoder,
)
from context_aware_encoder_model.cqr_train_eval import CQRDataset, collate_fn, evaluate_sentence_ranking


def save_encoder_bundle(model: ContextAwareSentenceEncoder, config: ContextAwareEncoderConfig, output_dir: Path) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    model.encoder.save_pretrained(output_dir)
    model.tokenizer.save_pretrained(output_dir)
    with (output_dir / 'encoder_config.json').open('w', encoding='utf-8') as f:
        json.dump(vars(config), f, ensure_ascii=False, indent=2)


def better_metrics(current: Dict[str, float], best: Optional[Dict[str, float]]) -> bool:
    if best is None:
        return True
    current_key: Tuple[float, float, float] = (
        float(current['mrr']),
        float(current['top1_acc']),
        float(current['avg_margin']),
    )
    best_key: Tuple[float, float, float] = (
        float(best['mrr']),
        float(best['top1_acc']),
        float(best['avg_margin']),
    )
    return current_key > best_key


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument('--train_file', type=str, required=True)
    parser.add_argument('--dev_file', type=str, default='')
    parser.add_argument('--output_dir', type=str, required=True)
    parser.add_argument('--model_name', type=str, default='Qwen/Qwen3-Embedding-8B')
    parser.add_argument('--cache_dir', type=str, default='')
    parser.add_argument('--pooling_strategy', type=str, default='last_token', choices=['mean', 'last_token'])
    parser.add_argument('--epochs', type=int, default=3)
    parser.add_argument('--batch_size', type=int, default=4)
    parser.add_argument('--lr', type=float, default=2e-5)
    parser.add_argument('--max_length', type=int, default=512)
    parser.add_argument('--temperature', type=float, default=0.05)
    parser.add_argument('--seed', type=int, default=42)
    args = parser.parse_args()

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    config = ContextAwareEncoderConfig(
        model_name=args.model_name,
        max_length=args.max_length,
        temperature=args.temperature,
        device=device,
        cache_dir=args.cache_dir,
        pooling_strategy=args.pooling_strategy,
    )
    model = ContextAwareSentenceEncoder(config)

    train_dataset = CQRDataset(Path(args.train_file), config.marker_start, config.marker_end)
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, collate_fn=collate_fn)
    dev_dataset = CQRDataset(Path(args.dev_file), config.marker_start, config.marker_end) if args.dev_file else None

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr)
    model.train()

    history: List[Dict[str, float]] = []
    best_metrics: Optional[Dict[str, float]] = None
    best_state = None

    for epoch in range(1, args.epochs + 1):
        total_loss = 0.0
        total_count = 0
        for batch in train_loader:
            optimizer.zero_grad()
            loss = model.contrastive_loss(
                questions=batch['questions'],
                positive_marked_contexts=batch['positive_marked_contexts'],
                negative_marked_contexts=batch['negative_marked_contexts'],
            )
            loss.backward()
            optimizer.step()

            bs = len(batch['questions'])
            total_loss += float(loss.item()) * bs
            total_count += bs

        train_loss = total_loss / max(total_count, 1)
        record: Dict[str, float] = {'epoch': float(epoch), 'train_loss': train_loss}

        if dev_dataset is not None:
            dev_metrics = evaluate_sentence_ranking(model, dev_dataset)
            record.update({f'dev_{key}': value for key, value in dev_metrics.items()})
            print(
                f"epoch={epoch:02d} train_loss={train_loss:.4f} "
                f"dev_top1={dev_metrics['top1_acc']:.4f} dev_mrr={dev_metrics['mrr']:.4f} "
                f"dev_margin={dev_metrics['avg_margin']:.4f}"
            )
            if better_metrics(dev_metrics, best_metrics):
                best_metrics = dict(dev_metrics)
                best_metrics['epoch'] = float(epoch)
                best_metrics['train_loss'] = train_loss
                best_state = {k: v.detach().cpu() for k, v in model.state_dict().items()}
        else:
            print(f'epoch={epoch:02d} train_loss={train_loss:.4f}')

        history.append(record)

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    if dev_dataset is not None:
        save_encoder_bundle(model, config, output_dir / 'last')
        if best_state is not None:
            model.load_state_dict(best_state)
        save_encoder_bundle(model, config, output_dir)
        with (output_dir / 'best_metrics.json').open('w', encoding='utf-8') as f:
            json.dump(best_metrics or {}, f, ensure_ascii=False, indent=2)
    else:
        save_encoder_bundle(model, config, output_dir)

    with (output_dir / 'training_history.json').open('w', encoding='utf-8') as f:
        json.dump(history, f, ensure_ascii=False, indent=2)

    print('saved best model to', output_dir)
    if dev_dataset is not None:
        print('saved last model to', output_dir / 'last')


if __name__ == '__main__':
    main()

