from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, List, Sequence

import torch
from torch.utils.data import Dataset

from context_aware_encoder_model.context_aware_sentence_encoder import build_marked_context_from_text


class CQRDataset(Dataset):
    def __init__(
        self,
        path: Path,
        marker_start: str,
        marker_end: str,
        use_window: bool = True,
        max_window_chars: int = 900,
    ):
        self.rows: List[Dict] = []
        with path.open('r', encoding='utf-8') as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                row = json.loads(line)
                pos_marked = build_marked_context_from_text(
                    row['context'],
                    row['positive_sentence'],
                    marker_start,
                    marker_end,
                    use_window=use_window,
                    max_chars=max_window_chars,
                )
                neg_marked = [
                    build_marked_context_from_text(
                        row['context'],
                        sent,
                        marker_start,
                        marker_end,
                        use_window=use_window,
                        max_chars=max_window_chars,
                    )
                    for sent in row.get('negative_sentences', [])
                ]
                self.rows.append(
                    {
                        'question': row['question'],
                        'context': row['context'],
                        'answer': row.get('answer', ''),
                        'positive_sentence': row['positive_sentence'],
                        'negative_sentences': row.get('negative_sentences', []),
                        'supporting_sentences': row.get('supporting_sentences', []),
                        'positive_marked_context': pos_marked,
                        'negative_marked_contexts': neg_marked,
                    }
                )

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(self, idx: int) -> Dict:
        return self.rows[idx]


def collate_fn(batch: List[Dict]) -> Dict[str, List[str]]:
    return {
        'questions': [item['question'] for item in batch],
        'contexts': [item['context'] for item in batch],
        'answers': [item['answer'] for item in batch],
        'positive_sentences': [item['positive_sentence'] for item in batch],
        'negative_sentences': [item['negative_sentences'] for item in batch],
        'supporting_sentences': [item['supporting_sentences'] for item in batch],
        'positive_marked_contexts': [item['positive_marked_context'] for item in batch],
        'negative_marked_contexts': [item['negative_marked_contexts'] for item in batch],
    }


def evaluate_sentence_ranking(model, dataset: CQRDataset) -> Dict[str, float]:
    model.eval()

    top1 = 0.0
    top2 = 0.0
    mrr = 0.0
    avg_margin = 0.0
    avg_candidates = 0.0

    with torch.no_grad():
        for row in dataset.rows:
            question = row['question']
            candidate_contexts = [row['positive_marked_context'], *row['negative_marked_contexts']]
            question_batch = [question] * len(candidate_contexts)
            q_emb = model.encode_question([question])
            cand_emb = model.encode_marked_contexts(question_batch, candidate_contexts)
            scores = torch.matmul(q_emb, cand_emb.T).squeeze(0)

            pos_score = float(scores[0].item())
            neg_scores = scores[1:]
            rank = 1 + int((neg_scores >= scores[0]).sum().item()) if neg_scores.numel() > 0 else 1

            top1 += 1.0 if rank == 1 else 0.0
            top2 += 1.0 if rank <= 2 else 0.0
            mrr += 1.0 / rank
            hardest_neg = float(torch.max(neg_scores).item()) if neg_scores.numel() > 0 else 0.0
            avg_margin += pos_score - hardest_neg
            avg_candidates += len(candidate_contexts)

    total = max(len(dataset), 1)
    return {
        'num_examples': float(len(dataset)),
        'top1_acc': top1 / total,
        'top2_acc': top2 / total,
        'mrr': mrr / total,
        'avg_margin': avg_margin / total,
        'avg_candidates': avg_candidates / total,
    }


