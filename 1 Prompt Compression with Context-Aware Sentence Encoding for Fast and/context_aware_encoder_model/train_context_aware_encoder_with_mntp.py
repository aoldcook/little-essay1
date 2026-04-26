from __future__ import annotations

import argparse
import os
import json
import random
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from transformers import AutoModelForMaskedLM, AutoTokenizer

from context_aware_encoder_model.context_aware_sentence_encoder import default_hf_cache_dir
from context_aware_encoder_model.cqr_train_eval import CQRDataset, collate_fn, evaluate_sentence_ranking


@dataclass
class MultiTaskEncoderConfig:
    model_name: str = 'answerdotai/ModernBERT-base'
    max_length: int = 512
    temperature: float = 0.05
    mlm_probability: float = 0.15
    lambda_mntp: float = 1.0
    device: str = 'cuda'
    marker_start: str = '<sent_start>'
    marker_end: str = '<sent_end>'
    cache_dir: str = ''
    trust_remote_code: bool = True


class MultiTaskContextAwareSentenceEncoder(nn.Module):
    def __init__(self, config: MultiTaskEncoderConfig):
        super().__init__()
        self.config = config

        cache_dir = config.cache_dir or default_hf_cache_dir()
        os.environ.setdefault("HF_HOME", cache_dir)
        os.environ.setdefault("HF_HUB_CACHE", str(Path(cache_dir) / "hub"))
        os.environ.setdefault("TRANSFORMERS_CACHE", str(Path(cache_dir) / "transformers"))

        self.tokenizer = AutoTokenizer.from_pretrained(config.model_name, cache_dir=cache_dir, trust_remote_code=config.trust_remote_code)
        self.tokenizer.add_special_tokens(
            {'additional_special_tokens': [config.marker_start, config.marker_end]}
        )

        if self.tokenizer.pad_token is None and self.tokenizer.eos_token is not None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        self.mlm_model = AutoModelForMaskedLM.from_pretrained(config.model_name, cache_dir=cache_dir, trust_remote_code=config.trust_remote_code)
        self.mlm_model.resize_token_embeddings(len(self.tokenizer))
        self.backbone = getattr(self.mlm_model, self.mlm_model.base_model_prefix)

        self.start_id = self.tokenizer.convert_tokens_to_ids(config.marker_start)
        self.end_id = self.tokenizer.convert_tokens_to_ids(config.marker_end)
        self.device = torch.device(config.device)
        self.to(self.device)

    def mean_pool(self, hidden_states: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
        mask = attention_mask.unsqueeze(-1).float()
        summed = torch.sum(hidden_states * mask, dim=1)
        denom = torch.clamp(mask.sum(dim=1), min=1e-9)
        return summed / denom

    def encode_question(self, questions: Sequence[str]) -> torch.Tensor:
        batch = self.tokenizer(
            list(questions),
            padding=True,
            truncation=True,
            max_length=self.config.max_length,
            return_tensors='pt',
        ).to(self.device)
        outputs = self.backbone(**batch)
        pooled = self.mean_pool(outputs.last_hidden_state, batch['attention_mask'])
        return F.normalize(pooled, p=2, dim=1)

    def _find_marker_span(self, input_ids: torch.Tensor) -> Tuple[int, int]:
        ids = input_ids.detach().cpu().tolist()
        start_pos = ids.index(self.start_id)
        end_pos = ids.index(self.end_id)
        if end_pos <= start_pos + 1:
            raise ValueError('invalid marker span')
        return start_pos + 1, end_pos

    def encode_marked_contexts(self, questions: Sequence[str], marked_contexts: Sequence[str]) -> torch.Tensor:
        batch = self.tokenizer(
            list(questions),
            list(marked_contexts),
            padding=True,
            truncation=True,
            max_length=self.config.max_length,
            return_tensors='pt',
        ).to(self.device)

        outputs = self.backbone(**batch)
        hidden = outputs.last_hidden_state

        sent_vecs = []
        for batch_idx in range(hidden.size(0)):
            try:
                start, end = self._find_marker_span(batch['input_ids'][batch_idx])
                span_hidden = hidden[batch_idx, start:end, :]
                sent_vec = span_hidden.mean(dim=0)
            except ValueError:
                # Rare long-tail samples may lose markers after truncation; fall back to whole-sequence pooling.
                mask = batch['attention_mask'][batch_idx].unsqueeze(-1).float()
                sent_vec = torch.sum(hidden[batch_idx] * mask, dim=0) / torch.clamp(mask.sum(), min=1e-9)
            sent_vecs.append(sent_vec)
        sent_vecs = torch.stack(sent_vecs, dim=0)
        return F.normalize(sent_vecs, p=2, dim=1)

    def contrastive_loss(
        self,
        questions: Sequence[str],
        positive_marked_contexts: Sequence[str],
        negative_marked_contexts: Sequence[Sequence[str]],
    ) -> torch.Tensor:
        q_emb = self.encode_question(questions)
        pos_emb = self.encode_marked_contexts(questions, positive_marked_contexts)

        flat_neg_questions: List[str] = []
        flat_neg_contexts: List[str] = []
        for question, negs in zip(questions, negative_marked_contexts):
            for neg in negs:
                flat_neg_questions.append(question)
                flat_neg_contexts.append(neg)

        if flat_neg_contexts:
            neg_emb = self.encode_marked_contexts(flat_neg_questions, flat_neg_contexts)
            candidates = torch.cat([pos_emb, neg_emb], dim=0)
        else:
            candidates = pos_emb

        logits = torch.matmul(q_emb, candidates.T) / self.config.temperature
        targets = torch.arange(len(questions), device=self.device)
        return F.cross_entropy(logits, targets)

    def mntp_loss(self, contexts: Sequence[str]) -> torch.Tensor:
        batch = self.tokenizer(
            list(contexts),
            padding=True,
            truncation=True,
            max_length=self.config.max_length,
            return_tensors='pt',
        )
        input_ids = batch['input_ids']
        attention_mask = batch['attention_mask']

        labels = input_ids.clone()
        probability_matrix = torch.full(labels.shape, self.config.mlm_probability)
        special_tokens_mask = torch.tensor(
            [
                self.tokenizer.get_special_tokens_mask(val.tolist(), already_has_special_tokens=True)
                for val in labels
            ],
            dtype=torch.bool,
        )
        probability_matrix.masked_fill_(special_tokens_mask, value=0.0)
        masked_indices = torch.bernoulli(probability_matrix).bool()
        labels[~masked_indices] = -100

        indices_replaced = torch.bernoulli(torch.full(labels.shape, 0.8)).bool() & masked_indices
        mask_token_id = self.tokenizer.mask_token_id
        if mask_token_id is None:
            raise ValueError('Tokenizer has no mask token; choose a MLM-capable backbone.')
        input_ids[indices_replaced] = mask_token_id

        indices_random = (
            torch.bernoulli(torch.full(labels.shape, 0.5)).bool()
            & masked_indices
            & ~indices_replaced
        )
        random_words = torch.randint(len(self.tokenizer), labels.shape, dtype=torch.long)
        input_ids[indices_random] = random_words[indices_random]

        model_inputs = {
            'input_ids': input_ids.to(self.device),
            'attention_mask': attention_mask.to(self.device),
            'labels': labels.to(self.device),
        }
        outputs = self.mlm_model(**model_inputs)
        return outputs.loss


def save_encoder_bundle(model: MultiTaskContextAwareSentenceEncoder, config: MultiTaskEncoderConfig, output_dir: Path) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    model.mlm_model.save_pretrained(output_dir)
    model.tokenizer.save_pretrained(output_dir)
    with (output_dir / 'encoder_config.json').open('w', encoding='utf-8') as f:
        json.dump(asdict(config), f, ensure_ascii=False, indent=2)


def better_metrics(current: Dict[str, float], best: Optional[Dict[str, float]]) -> bool:
    if best is None:
        return True
    current_key = (float(current['mrr']), float(current['top1_acc']), float(current['avg_margin']))
    best_key = (float(best['mrr']), float(best['top1_acc']), float(best['avg_margin']))
    return current_key > best_key


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument('--train_file', type=str, required=True)
    parser.add_argument('--dev_file', type=str, default='')
    parser.add_argument('--output_dir', type=str, required=True)
    parser.add_argument('--model_name', type=str, default='answerdotai/ModernBERT-base')
    parser.add_argument('--cache_dir', type=str, default='')
    parser.add_argument('--epochs', type=int, default=10)
    parser.add_argument('--batch_size', type=int, default=4)
    parser.add_argument('--lr', type=float, default=2e-5)
    parser.add_argument('--max_length', type=int, default=512)
    parser.add_argument('--temperature', type=float, default=0.05)
    parser.add_argument('--mlm_probability', type=float, default=0.15)
    parser.add_argument('--lambda_mntp', type=float, default=1.0)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--log_every', type=int, default=200)
    args = parser.parse_args()

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    config = MultiTaskEncoderConfig(
        model_name=args.model_name,
        max_length=args.max_length,
        temperature=args.temperature,
        mlm_probability=args.mlm_probability,
        lambda_mntp=args.lambda_mntp,
        device=device,
        cache_dir=args.cache_dir,
    )

    train_dataset = CQRDataset(Path(args.train_file), config.marker_start, config.marker_end)
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, collate_fn=collate_fn)
    dev_dataset = CQRDataset(Path(args.dev_file), config.marker_start, config.marker_end) if args.dev_file else None

    model = MultiTaskContextAwareSentenceEncoder(config)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr)

    history: List[Dict[str, float]] = []
    best_metrics: Optional[Dict[str, float]] = None
    best_state = None

    model.train()
    for epoch in range(1, args.epochs + 1):
        total_ctr = 0.0
        total_mntp = 0.0
        total = 0
        num_batches = len(train_loader)

        for step, batch in enumerate(train_loader, start=1):
            optimizer.zero_grad()
            loss_ctr = model.contrastive_loss(
                questions=batch['questions'],
                positive_marked_contexts=batch['positive_marked_contexts'],
                negative_marked_contexts=batch['negative_marked_contexts'],
            )
            loss_mntp = model.mntp_loss(batch['contexts'])
            loss = loss_ctr + config.lambda_mntp * loss_mntp
            loss.backward()
            optimizer.step()

            bs = len(batch['questions'])
            total_ctr += float(loss_ctr.item()) * bs
            total_mntp += float(loss_mntp.item()) * bs
            total += bs

            if args.log_every > 0 and (step % args.log_every == 0 or step == num_batches):
                avg_ctr = total_ctr / max(total, 1)
                avg_mntp = total_mntp / max(total, 1)
                avg_total = avg_ctr + config.lambda_mntp * avg_mntp
                print(
                    f"epoch={epoch:02d} step={step:05d}/{num_batches:05d} "
                    f"ctr={avg_ctr:.4f} mntp={avg_mntp:.4f} total={avg_total:.4f}",
                    flush=True,
                )

        train_ctr = total_ctr / max(total, 1)
        train_mntp = total_mntp / max(total, 1)
        train_total = train_ctr + config.lambda_mntp * train_mntp
        record: Dict[str, float] = {
            'epoch': float(epoch),
            'train_ctr_loss': train_ctr,
            'train_mntp_loss': train_mntp,
            'train_total_loss': train_total,
        }

        if dev_dataset is not None:
            dev_metrics = evaluate_sentence_ranking(model, dev_dataset)
            record.update({f'dev_{key}': value for key, value in dev_metrics.items()})
            print(
                f"epoch={epoch:02d} ctr={train_ctr:.4f} mntp={train_mntp:.4f} total={train_total:.4f} "
                f"dev_top1={dev_metrics['top1_acc']:.4f} dev_mrr={dev_metrics['mrr']:.4f}"
            )
            if better_metrics(dev_metrics, best_metrics):
                best_metrics = dict(dev_metrics)
                best_metrics['epoch'] = float(epoch)
                best_metrics['train_total_loss'] = train_total
                best_state = {k: v.detach().cpu() for k, v in model.state_dict().items()}
        else:
            print(f'epoch={epoch:02d} ctr={train_ctr:.4f} mntp={train_mntp:.4f} total={train_total:.4f}')

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





