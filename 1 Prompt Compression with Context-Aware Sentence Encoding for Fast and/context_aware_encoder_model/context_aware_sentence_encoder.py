from __future__ import annotations

import re
from dataclasses import dataclass
from typing import List, Sequence, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoModel, AutoTokenizer


@dataclass
class ContextAwareEncoderConfig:
    model_name: str = "bert-base-chinese"
    max_length: int = 512
    temperature: float = 0.05
    device: str = "cuda"
    marker_start: str = "<sent_start>"
    marker_end: str = "<sent_end>"


class ContextAwareSentenceEncoder(nn.Module):
    """
    一个最小可用的上下文感知句子编码器：
    - 问题单独编码得到 query embedding
    - 对“带 marker 的上下文”编码
    - 在 marker 圈出的 span 上做平均池化
    - 得到 sentence-in-context embedding
    """

    def __init__(self, config: ContextAwareEncoderConfig):
        super().__init__()
        self.config = config

        self.tokenizer = AutoTokenizer.from_pretrained(config.model_name)
        self.tokenizer.add_special_tokens(
            {"additional_special_tokens": [config.marker_start, config.marker_end]}
        )

        try:
            self.encoder = AutoModel.from_pretrained(
                config.model_name,
                attn_implementation="eager",
            )
        except TypeError:
            self.encoder = AutoModel.from_pretrained(config.model_name)
        self.encoder.resize_token_embeddings(len(self.tokenizer))

        self.start_id = self.tokenizer.convert_tokens_to_ids(config.marker_start)
        self.end_id = self.tokenizer.convert_tokens_to_ids(config.marker_end)

        self.device = torch.device(config.device)
        self.to(self.device)

    def mean_pool(self, hidden_states: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
        """
        hidden_states: [B, L, H]
        attention_mask: [B, L]
        return: [B, H]
        """
        mask = attention_mask.unsqueeze(-1).float()  # [B, L, 1]
        summed = torch.sum(hidden_states * mask, dim=1)  # [B, H]
        denom = torch.clamp(mask.sum(dim=1), min=1e-9)  # [B, 1]
        return summed / denom

    def encode_question(self, questions: Sequence[str]) -> torch.Tensor:
        """
        对问题做编码，输出归一化后的 query embedding。
        return: [B, H]
        """
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
        """
        在 token id 序列中找到 <sent_start> 和 <sent_end> 之间的 span。
        返回值是 [start, end) 风格，不包含 marker 本身。
        """
        ids = input_ids.detach().cpu().tolist()
        try:
            start_pos = ids.index(self.start_id)
            end_pos = ids.index(self.end_id)
        except ValueError as exc:
            raise ValueError(
                "marker tokens not found in input_ids; "
                "most likely the marked context was truncated because it is too long. "
                "Please use a shorter local context window."
            ) from exc

        if end_pos <= start_pos + 1:
            raise ValueError(
                "invalid marker span: end marker appears before start marker "
                "or the sentence span is empty."
            )
        return start_pos + 1, end_pos

    def encode_marked_contexts(
        self,
        questions: Sequence[str],
        marked_contexts: Sequence[str],
    ) -> torch.Tensor:
        """
        编码 (question, marked_context) 对，并提取 marker 内句子的上下文感知表示。
        return: [B, H]
        """
        batch = self.tokenizer(
            list(questions),
            list(marked_contexts),
            padding=True,
            truncation=True,
            max_length=self.config.max_length,
            return_tensors="pt",
        ).to(self.device)

        outputs = self.encoder(**batch)
        hidden = outputs.last_hidden_state  # [B, L, H]

        sent_vecs = []
        for b in range(hidden.size(0)):
            start, end = self._find_marker_span(batch["input_ids"][b])
            span_hidden = hidden[b, start:end, :]  # [span_len, H]
            sent_vec = span_hidden.mean(dim=0)     # [H]
            sent_vecs.append(sent_vec)

        sent_vecs = torch.stack(sent_vecs, dim=0)  # [B, H]
        return F.normalize(sent_vecs, p=2, dim=1)

    @torch.no_grad()
    def score_sentences(
        self,
        question: str,
        sentences: Sequence[str],
        marked_contexts: Sequence[str],
    ) -> Tuple[List[float], torch.Tensor]:
        """
        给一个问题和多个句子打分。
        返回：
        - similarities: List[float]
        - sentence_embeddings: [N, H]
        """
        if len(sentences) != len(marked_contexts):
            raise ValueError(
                f"len(sentences)={len(sentences)} but len(marked_contexts)={len(marked_contexts)}"
            )

        self.eval()

        q_emb = self.encode_question([question])  # [1, H]
        s_emb = self.encode_marked_contexts([question] * len(marked_contexts), marked_contexts)  # [N, H]

        sims = torch.matmul(q_emb, s_emb.T).squeeze(0)  # [N]
        return sims.detach().cpu().tolist(), s_emb.detach().cpu()

    @torch.no_grad()
    def attention_probe_scores(
        self,
        question: str,
        marked_contexts: Sequence[str],
        probe_layers: int = 2,
    ) -> List[float]:
        if not marked_contexts:
            return []

        self.eval()
        batch = self.tokenizer(
            [question] * len(marked_contexts),
            list(marked_contexts),
            padding=True,
            truncation=True,
            max_length=self.config.max_length,
            return_tensors="pt",
        ).to(self.device)

        outputs = self.encoder(**batch, output_attentions=True)
        attentions = getattr(outputs, "attentions", None)
        if not attentions:
            return [0.0 for _ in marked_contexts]

        last_layers = attentions[-probe_layers:]
        mean_attention = torch.stack([layer.mean(dim=1) for layer in last_layers], dim=0).mean(dim=0)
        special_ids = set(getattr(self.tokenizer, "all_special_ids", []))

        scores: List[float] = []
        for b in range(mean_attention.size(0)):
            start, end = self._find_marker_span(batch["input_ids"][b])
            span_indices = list(range(start, end))
            valid_indices = [
                idx
                for idx, input_id in enumerate(batch["input_ids"][b].detach().cpu().tolist())
                if int(batch["attention_mask"][b, idx].item()) == 1 and input_id not in special_ids
            ]
            if "token_type_ids" in batch:
                question_indices = [idx for idx in valid_indices if int(batch["token_type_ids"][b, idx].item()) == 0]
            else:
                question_indices = [idx for idx in valid_indices if idx < start]

            if not question_indices or not span_indices:
                scores.append(0.0)
                continue

            q_to_span = float(mean_attention[b, question_indices][:, span_indices].mean().item())
            global_to_span = float(mean_attention[b, valid_indices][:, span_indices].mean().item())
            scores.append(0.7 * q_to_span + 0.3 * global_to_span)

        return scores

    def contrastive_loss(
        self,
        questions: Sequence[str],
        positive_marked_contexts: Sequence[str],
        negative_marked_contexts: Sequence[Sequence[str]],
    ) -> torch.Tensor:
        """
        一个最小可用的 InfoNCE 对比学习损失：
        - query 更接近正样本句
        - query 远离负样本句
        """
        if len(questions) != len(positive_marked_contexts):
            raise ValueError("questions and positive_marked_contexts must have the same length")

        q_emb = self.encode_question(questions)  # [B, H]
        pos_emb = self.encode_marked_contexts(questions, positive_marked_contexts)  # [B, H]

        flat_neg_questions: List[str] = []
        flat_neg_contexts: List[str] = []
        for q, negs in zip(questions, negative_marked_contexts):
            for neg in negs:
                flat_neg_questions.append(q)
                flat_neg_contexts.append(neg)

        if len(flat_neg_contexts) > 0:
            neg_emb = self.encode_marked_contexts(flat_neg_questions, flat_neg_contexts)  # [K, H]
            candidates = torch.cat([pos_emb, neg_emb], dim=0)  # [B+K, H]
        else:
            candidates = pos_emb

        logits = torch.matmul(q_emb, candidates.T) / self.config.temperature  # [B, B+K]
        targets = torch.arange(len(questions), device=self.device)  # 正样本在前 B 个候选中的第 i 个
        return F.cross_entropy(logits, targets)


def split_sentences(text: str) -> List[str]:
    """
    中英混合的基础切句函数。
    """
    sentences = re.split(r"(?<=[。！？.!?])\s*", text.strip())
    return [s.strip() for s in sentences if s.strip()]


def build_marked_context(
    sentences: Sequence[str],
    target_index: int,
    marker_start: str,
    marker_end: str,
) -> str:
    """
    直接使用完整句子列表构造带 marker 的上下文：
    例如：A <sent_start>B</sent_end> C
    """
    parts = []
    for idx, s in enumerate(sentences):
        if idx == target_index:
            parts.append(f"{marker_start}{s}{marker_end}")
        else:
            parts.append(s)
    return "".join(parts)


def build_marked_context_window(
    sentences: Sequence[str],
    target_index: int,
    marker_start: str,
    marker_end: str,
    max_chars: int = 220,
) -> str:
    """
    只截取目标句附近的局部上下文窗口，避免完整上下文过长导致 marker 被截断。

    逻辑：
    - 一定保留目标句
    - 然后从左右两侧逐步扩展
    - 直到达到 max_chars 的字符预算

    注意：
    这里按“字符数”近似控制窗口大小，更适合中文最小可运行版本。
    """
    if target_index < 0 or target_index >= len(sentences):
        raise IndexError(f"target_index out of range: {target_index}")

    total_len = len(sentences[target_index]) + len(marker_start) + len(marker_end)
    selected = [target_index]

    left = target_index - 1
    right = target_index + 1

    while True:
        added = False

        if left >= 0 and total_len + len(sentences[left]) <= max_chars:
            selected.insert(0, left)
            total_len += len(sentences[left])
            left -= 1
            added = True

        if right < len(sentences) and total_len + len(sentences[right]) <= max_chars:
            selected.append(right)
            total_len += len(sentences[right])
            right += 1
            added = True

        if not added:
            break

    parts = []
    for idx in selected:
        s = sentences[idx]
        if idx == target_index:
            s = f"{marker_start}{s}{marker_end}"
        parts.append(s)

    return "".join(parts)


def build_marked_context_from_text(
    context: str,
    target_sentence: str,
    marker_start: str,
    marker_end: str,
    use_window: bool = False,
    max_chars: int = 220,
) -> str:
    """
    从原始文本和目标句文本出发构造带 marker 的上下文。

    默认 use_window=False，保持和旧训练脚本兼容；
    如果你后面想让训练和推理一致，可以改成 use_window=True。
    """
    sentences = split_sentences(context)
    target_norm = re.sub(r"\s+", "", target_sentence.strip())

    target_index = -1
    for i, s in enumerate(sentences):
        s_norm = re.sub(r"\s+", "", s.strip())
        if s_norm == target_norm:
            target_index = i
            break

    if target_index == -1:
        raise ValueError("target_sentence not found in context after sentence splitting")

    if use_window:
        return build_marked_context_window(
            sentences=sentences,
            target_index=target_index,
            marker_start=marker_start,
            marker_end=marker_end,
            max_chars=max_chars,
        )

    return build_marked_context(
        sentences=sentences,
        target_index=target_index,
        marker_start=marker_start,
        marker_end=marker_end,
    )

