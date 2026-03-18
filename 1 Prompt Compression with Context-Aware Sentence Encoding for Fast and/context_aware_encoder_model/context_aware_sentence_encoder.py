from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Sequence, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoModel, AutoTokenizer


@dataclass
class ContextAwareEncoderConfig:
    model_name: str = "bert-base-chinese"
    max_length: int = 512
    temperature: float = 0.05
    device: str = "cpu"
    marker_start: str = "<sent_start>"
    marker_end: str = "<sent_end>"


class ContextAwareSentenceEncoder(nn.Module):
    """
    一个可直接落地的最小版本：
    - 问题单独编码为 query embedding
    - 目标句子在上下文中用特殊标记包围
    - 使用标记区间内的 token 平均池化作为 sentence-in-context embedding
    - 使用 InfoNCE 训练问题与正样本句子更接近、与负样本更远

    说明：
    1. 这是“可运行、可扩展”的骨架版本，适合先把方法跑通。
    2. 如果你后续要更贴近 CPC，可以再加 MLM/MNTP 辅助损失。
    """

    def __init__(self, config: ContextAwareEncoderConfig):
        super().__init__()
        self.config = config
        self.tokenizer = AutoTokenizer.from_pretrained(config.model_name)
        self.tokenizer.add_special_tokens(
            {"additional_special_tokens": [config.marker_start, config.marker_end]}
        )
        self.encoder = AutoModel.from_pretrained(config.model_name)
        self.encoder.resize_token_embeddings(len(self.tokenizer))
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
        outputs = self.encoder(**batch)
        pooled = self.mean_pool(outputs.last_hidden_state, batch["attention_mask"])
        return F.normalize(pooled, p=2, dim=1)

    def _find_marker_span(self, input_ids: torch.Tensor) -> Tuple[int, int]:
        ids = input_ids.tolist()
        try:
            start_pos = ids.index(self.start_id)
            end_pos = ids.index(self.end_id)
        except ValueError as exc:
            raise ValueError("marker tokens not found in input_ids; please check marked context construction") from exc
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
        outputs = self.encoder(**batch)
        hidden = outputs.last_hidden_state

        sent_vecs = []
        for b in range(hidden.size(0)):
            start, end = self._find_marker_span(batch["input_ids"][b])
            span_hidden = hidden[b, start:end, :]
            sent_vec = span_hidden.mean(dim=0)
            sent_vecs.append(sent_vec)
        sent_vecs = torch.stack(sent_vecs, dim=0)
        return F.normalize(sent_vecs, p=2, dim=1)

    def score_sentences(self, question: str, sentences: List[str], marked_contexts: List[str]) -> Tuple[List[float], torch.Tensor]:
        q_emb = self.encode_question([question])
        s_emb = self.encode_marked_contexts([question] * len(marked_contexts), marked_contexts)
        sims = torch.matmul(q_emb, s_emb.T).squeeze(0)
        return sims.detach().cpu().tolist(), s_emb.detach().cpu()

    def contrastive_loss(
        self,
        questions: Sequence[str],
        positive_marked_contexts: Sequence[str],
        negative_marked_contexts: Sequence[Sequence[str]],
    ) -> torch.Tensor:
        """
        使用 in-batch positives + 显式 negatives 的 InfoNCE。

        参数：
            questions: 长度 B
            positive_marked_contexts: 长度 B
            negative_marked_contexts: 长度 B，每个元素是若干个负样本 marked context
        """
        q_emb = self.encode_question(questions)                    # [B, d]
        pos_emb = self.encode_marked_contexts(questions, positive_marked_contexts)  # [B, d]

        flat_neg_questions: List[str] = []
        flat_negs: List[str] = []
        for q, negs in zip(questions, negative_marked_contexts):
            for neg in negs:
                flat_neg_questions.append(q)
                flat_negs.append(neg)

        if flat_negs:
            neg_emb = self.encode_marked_contexts(flat_neg_questions, flat_negs)    # [B*M, d]
            candidates = torch.cat([pos_emb, neg_emb], dim=0)                        # [B+B*M, d]
        else:
            candidates = pos_emb

        logits = torch.matmul(q_emb, candidates.T) / self.config.temperature          # [B, B+B*M]
        targets = torch.arange(len(questions), device=self.device)                    # 正样本在前 B 个位置的对角线
        return F.cross_entropy(logits, targets)


def split_sentences(text: str) -> List[str]:
    import re
    sentences = re.split(r'(?<=[。！？.!?])\s*', text.strip())
    return [s.strip() for s in sentences if s.strip()]


def build_marked_context(sentences: Sequence[str], target_index: int, marker_start: str, marker_end: str) -> str:
    parts: List[str] = []
    for i, sent in enumerate(sentences):
        if i == target_index:
            parts.append(f"{marker_start}{sent}{marker_end}")
        else:
            parts.append(sent)
    return " ".join(parts)


def build_marked_context_from_text(context: str, target_sentence: str, marker_start: str, marker_end: str) -> str:
    sentences = split_sentences(context)
    matched_idx = None
    for i, sent in enumerate(sentences):
        if sent == target_sentence:
            matched_idx = i
            break
    if matched_idx is None:
        raise ValueError("target sentence not found in context after split_sentences")
    return build_marked_context(sentences, matched_idx, marker_start, marker_end)
