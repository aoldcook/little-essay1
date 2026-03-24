
from __future__ import annotations

import argparse
import json
import random
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Dict, List, Sequence

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
from transformers import AutoModelForMaskedLM, AutoTokenizer


def split_sentences(text: str) -> List[str]:
    import re
    sentences = re.split(r"(?<=[。！？.!?])\s*", text.strip())
    return [s.strip() for s in sentences if s.strip()]


def build_marked_context_from_text(
    context: str,
    target_sentence: str,
    marker_start: str,
    marker_end: str,
) -> str:
    sentences = split_sentences(context)
    target_norm = target_sentence.strip().replace(" ", "")
    target_idx = -1
    for i, s in enumerate(sentences):
        if s.strip().replace(" ", "") == target_norm:
            target_idx = i
            break
    if target_idx == -1:
        raise ValueError("target_sentence not found in context")

    parts = []
    for i, s in enumerate(sentences):
        if i == target_idx:
            parts.append(f"{marker_start}{s}{marker_end}")
        else:
            parts.append(s)
    return "".join(parts)


@dataclass
class MultiTaskEncoderConfig:
    model_name: str = "bert-base-chinese"
    max_length: int = 512
    temperature: float = 0.05
    mlm_probability: float = 0.15
    lambda_mntp: float = 1.0
    device: str = "cuda"
    marker_start: str = "<sent_start>"
    marker_end: str = "<sent_end>"


class MultiTaskContextAwareSentenceEncoder(nn.Module):
    """
    一个“对比学习 + MLM/MNTP近似”的多任务版本。

    说明：
    1. 对于 BERT / RoBERTa 一类双向编码器，论文中的 L_MNTP 可用标准 MLM 近似实现。
    2. 如果你后面换成 decoder-only backbone，可以再把这里改成更贴近原论文的 token prediction 形式。
    """

    def __init__(self, config: MultiTaskEncoderConfig):
        super().__init__()
        self.config = config

        self.tokenizer = AutoTokenizer.from_pretrained(config.model_name)
        self.tokenizer.add_special_tokens(
            {"additional_special_tokens": [config.marker_start, config.marker_end]}
        )

        self.mlm_model = AutoModelForMaskedLM.from_pretrained(config.model_name)
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
            return_tensors="pt",
        ).to(self.device)
        outputs = self.backbone(**batch)
        pooled = self.mean_pool(outputs.last_hidden_state, batch["attention_mask"])
        return F.normalize(pooled, p=2, dim=1)

    def _find_marker_span(self, input_ids: torch.Tensor):
        ids = input_ids.detach().cpu().tolist()
        start_pos = ids.index(self.start_id)
        end_pos = ids.index(self.end_id)
        if end_pos <= start_pos + 1:
            raise ValueError("invalid marker span")
        return start_pos + 1, end_pos

    def encode_marked_contexts(self, questions: Sequence[str], marked_contexts: Sequence[str]) -> torch.Tensor:
        batch = self.tokenizer(
            list(questions),
            list(marked_contexts),
            padding=True,
            truncation=True,
            max_length=self.config.max_length,
            return_tensors="pt",
        ).to(self.device)

        outputs = self.backbone(**batch)
        hidden = outputs.last_hidden_state

        sent_vecs = []
        for b in range(hidden.size(0)):
            start, end = self._find_marker_span(batch["input_ids"][b])
            span_hidden = hidden[b, start:end, :]
            sent_vecs.append(span_hidden.mean(dim=0))
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

        flat_neg_q = []
        flat_neg_ctx = []
        for q, negs in zip(questions, negative_marked_contexts):
            for neg in negs:
                flat_neg_q.append(q)
                flat_neg_ctx.append(neg)

        if flat_neg_ctx:
            neg_emb = self.encode_marked_contexts(flat_neg_q, flat_neg_ctx)
            candidates = torch.cat([pos_emb, neg_emb], dim=0)
        else:
            candidates = pos_emb

        logits = torch.matmul(q_emb, candidates.T) / self.config.temperature
        targets = torch.arange(len(questions), device=self.device)
        return F.cross_entropy(logits, targets)

    def mntp_loss(self, contexts: Sequence[str]) -> torch.Tensor:
        """
        用 BERT 风格 MLM 近似论文中的 L_MNTP。
        这里不对 question 做 mask，只对原始 context 做 token mask。
        """
        batch = self.tokenizer(
            list(contexts),
            padding=True,
            truncation=True,
            max_length=self.config.max_length,
            return_tensors="pt",
        )
        input_ids = batch["input_ids"]
        attention_mask = batch["attention_mask"]

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

        # 80% [MASK], 10% random, 10% original
        indices_replaced = torch.bernoulli(torch.full(labels.shape, 0.8)).bool() & masked_indices
        mask_token_id = self.tokenizer.mask_token_id
        if mask_token_id is None:
            raise ValueError("Tokenizer has no mask token; choose a MLM-capable backbone.")
        input_ids[indices_replaced] = mask_token_id

        indices_random = (
            torch.bernoulli(torch.full(labels.shape, 0.5)).bool()
            & masked_indices
            & ~indices_replaced
        )
        random_words = torch.randint(len(self.tokenizer), labels.shape, dtype=torch.long)
        input_ids[indices_random] = random_words[indices_random]

        batch = {
            "input_ids": input_ids.to(self.device),
            "attention_mask": attention_mask.to(self.device),
            "labels": labels.to(self.device),
        }
        outputs = self.mlm_model(**batch)
        return outputs.loss


class CQRMultiTaskDataset(Dataset):
    """
    输入 JSONL 格式：
    {
      "context": "...",
      "question": "...",
      "positive_sentence": "...",
      "negative_sentences": ["...", "..."]
    }
    """

    def __init__(self, path: Path, marker_start: str, marker_end: str):
        self.rows: List[Dict] = []
        with path.open("r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                row = json.loads(line)
                pos_marked = build_marked_context_from_text(
                    row["context"], row["positive_sentence"], marker_start, marker_end
                )
                neg_marked = [
                    build_marked_context_from_text(row["context"], s, marker_start, marker_end)
                    for s in row.get("negative_sentences", [])
                ]
                self.rows.append(
                    {
                        "context": row["context"],
                        "question": row["question"],
                        "positive_marked_context": pos_marked,
                        "negative_marked_contexts": neg_marked,
                    }
                )

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, idx):
        return self.rows[idx]


def collate_fn(batch: List[Dict]) -> Dict[str, List[str]]:
    return {
        "contexts": [x["context"] for x in batch],
        "questions": [x["question"] for x in batch],
        "positive_marked_contexts": [x["positive_marked_context"] for x in batch],
        "negative_marked_contexts": [x["negative_marked_contexts"] for x in batch],
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--train_file", type=str, required=True)
    parser.add_argument("--output_dir", type=str, required=True)
    parser.add_argument("--model_name", type=str, default="bert-base-chinese")
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument("--lr", type=float, default=2e-5)
    parser.add_argument("--max_length", type=int, default=512)
    parser.add_argument("--temperature", type=float, default=0.05)
    parser.add_argument("--mlm_probability", type=float, default=0.15)
    parser.add_argument("--lambda_mntp", type=float, default=1.0)
    args = parser.parse_args()

    random.seed(42)
    np.random.seed(42)
    torch.manual_seed(42)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    config = MultiTaskEncoderConfig(
        model_name=args.model_name,
        max_length=args.max_length,
        temperature=args.temperature,
        mlm_probability=args.mlm_probability,
        lambda_mntp=args.lambda_mntp,
        device=device,
    )

    dataset = CQRMultiTaskDataset(
        Path(args.train_file), config.marker_start, config.marker_end
    )
    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True, collate_fn=collate_fn)

    model = MultiTaskContextAwareSentenceEncoder(config)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr)

    model.train()
    for epoch in range(1, args.epochs + 1):
        total_ctr = 0.0
        total_mntp = 0.0
        total = 0

        for batch in loader:
            optimizer.zero_grad()

            loss_ctr = model.contrastive_loss(
                questions=batch["questions"],
                positive_marked_contexts=batch["positive_marked_contexts"],
                negative_marked_contexts=batch["negative_marked_contexts"],
            )
            loss_mntp = model.mntp_loss(batch["contexts"])
            loss = loss_ctr + config.lambda_mntp * loss_mntp

            loss.backward()
            optimizer.step()

            bs = len(batch["questions"])
            total_ctr += float(loss_ctr.item()) * bs
            total_mntp += float(loss_mntp.item()) * bs
            total += bs

        print(
            f"epoch={epoch:02d} "
            f"ctr={total_ctr / max(total,1):.4f} "
            f"mntp={total_mntp / max(total,1):.4f} "
            f"total={(total_ctr + config.lambda_mntp * total_mntp) / max(total,1):.4f}"
        )

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    model.mlm_model.save_pretrained(output_dir)
    model.tokenizer.save_pretrained(output_dir)

    with (output_dir / "encoder_config.json").open("w", encoding="utf-8") as f:
        json.dump(asdict(config), f, ensure_ascii=False, indent=2)

    print("saved to", output_dir)


if __name__ == "__main__":
    main()
