from __future__ import annotations

import os
import re
from dataclasses import dataclass
from pathlib import Path
from typing import List, Sequence, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoModel, AutoTokenizer


DEFAULT_STAGE1_MODEL = "Qwen/Qwen3-Embedding-8B"
DEFAULT_QUERY_INSTRUCTION = "Given a question, retrieve context sentences that contain evidence needed to answer it."


def default_hf_cache_dir() -> str:
    return str(Path(__file__).resolve().parents[2] / "hf_cache")


@dataclass
class ContextAwareEncoderConfig:
    model_name: str = DEFAULT_STAGE1_MODEL
    max_length: int = 1024
    temperature: float = 0.05
    device: str = "cuda"
    marker_start: str = "<sent_start>"
    marker_end: str = "<sent_end>"
    cache_dir: str = ""
    trust_remote_code: bool = True
    pooling_strategy: str = "last_token"
    query_instruction: str = DEFAULT_QUERY_INSTRUCTION
    # Sentences of one context were previously encoded in a SINGLE batch, so a
    # context with N sentences allocated N x heads x max_length^2 attention
    # weights under eager attention. A 100-sentence context at max_length 1024
    # exhausts a 24 GB card. Encoding is chunked to this size instead; it changes
    # throughput only, never the resulting embeddings.
    # 64 rather than 16: marked context WINDOWS are ~900 chars (~250 tokens), not
    # the full max_length, so attention here costs ~16x less than the worst case.
    # At 16 the GPU sat at 17% utilisation and a single 1013-row cell had not
    # finished after 95 minutes -- the run was launch-bound on tiny batches.
    encode_batch_size: int = 64


class ContextAwareSentenceEncoder(nn.Module):
    def __init__(self, config: ContextAwareEncoderConfig):
        super().__init__()
        self.config = config

        cache_dir = config.cache_dir or default_hf_cache_dir()
        os.environ.setdefault("HF_HOME", cache_dir)
        os.environ.setdefault("HF_HUB_CACHE", str(Path(cache_dir) / "hub"))
        os.environ.setdefault("TRANSFORMERS_CACHE", str(Path(cache_dir) / "transformers"))

        self.tokenizer = AutoTokenizer.from_pretrained(
            config.model_name,
            cache_dir=cache_dir,
            trust_remote_code=config.trust_remote_code,
            padding_side="left",
        )
        added_tokens = 0
        if self.tokenizer.pad_token is None:
            if self.tokenizer.eos_token is not None:
                self.tokenizer.pad_token = self.tokenizer.eos_token
            else:
                added_tokens += self.tokenizer.add_special_tokens({"pad_token": "<pad>"})
        added_tokens += self.tokenizer.add_special_tokens(
            {"additional_special_tokens": [config.marker_start, config.marker_end]}
        )

        try:
            self.encoder = AutoModel.from_pretrained(
                config.model_name,
                cache_dir=cache_dir,
                trust_remote_code=config.trust_remote_code,
                attn_implementation="eager",
            )
        except TypeError:
            self.encoder = AutoModel.from_pretrained(
                config.model_name,
                cache_dir=cache_dir,
                trust_remote_code=config.trust_remote_code,
            )
        if added_tokens > 0:
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

    def last_token_pool(self, hidden_states: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
        left_padding = bool((attention_mask[:, -1].sum() == attention_mask.shape[0]).item())
        if left_padding:
            return hidden_states[:, -1]
        sequence_lengths = attention_mask.sum(dim=1) - 1
        batch_size = hidden_states.shape[0]
        return hidden_states[torch.arange(batch_size, device=hidden_states.device), sequence_lengths]

    def format_query(self, question: str) -> str:
        if not self.config.query_instruction:
            return question
        if question.lstrip().lower().startswith("instruct:"):
            return question
        return f"Instruct: {self.config.query_instruction}\nQuery: {question}"

    def encode_question(self, questions: Sequence[str]) -> torch.Tensor:
        encoded_questions = [self.format_query(question) for question in questions]
        batch = self.tokenizer(
            encoded_questions,
            padding=True,
            truncation=True,
            max_length=self.config.max_length,
            return_tensors="pt",
        ).to(self.device)

        outputs = self.encoder(**batch)
        if self.config.pooling_strategy == "last_token":
            pooled = self.last_token_pool(outputs.last_hidden_state, batch["attention_mask"])
        else:
            pooled = self.mean_pool(outputs.last_hidden_state, batch["attention_mask"])
        return F.normalize(pooled, p=2, dim=1)

    def _find_marker_span(self, input_ids: torch.Tensor) -> Tuple[int, int]:
        ids = input_ids.detach().cpu().tolist()
        try:
            start_pos = ids.index(self.start_id)
            end_pos = ids.index(self.end_id)
        except ValueError as exc:
            raise ValueError(
                "marker tokens not found in input_ids; the marked context was probably truncated. "
                "Use a shorter local context window or raise max_length."
            ) from exc

        if end_pos <= start_pos + 1:
            raise ValueError("invalid marker span: end marker appears before start marker or the span is empty")
        return start_pos + 1, end_pos

    def encode_marked_contexts(
        self,
        questions: Sequence[str],
        marked_contexts: Sequence[str],
    ) -> torch.Tensor:
        """Encode marked contexts in fixed-size chunks.

        Chunking is required, not an optimisation: encoding every sentence of a
        long context at once allocates batch x heads x len^2 attention weights
        and OOMs a 24 GB GPU on realistic inputs.
        """
        batch_size = max(1, int(getattr(self.config, "encode_batch_size", 16)))
        if len(marked_contexts) <= batch_size:
            return self._encode_marked_chunk(questions, marked_contexts)

        chunks = [
            self._encode_marked_chunk(
                questions[start: start + batch_size],
                marked_contexts[start: start + batch_size],
            )
            for start in range(0, len(marked_contexts), batch_size)
        ]
        return torch.cat(chunks, dim=0)

    def _encode_marked_chunk(
        self,
        questions: Sequence[str],
        marked_contexts: Sequence[str],
    ) -> torch.Tensor:
        encoded_questions = [self.format_query(question) for question in questions]
        batch = self.tokenizer(
            encoded_questions,
            list(marked_contexts),
            padding=True,
            truncation=True,
            max_length=self.config.max_length,
            return_tensors="pt",
        ).to(self.device)

        outputs = self.encoder(**batch)
        hidden = outputs.last_hidden_state

        sent_vecs = []
        for batch_idx in range(hidden.size(0)):
            try:
                start, end = self._find_marker_span(batch["input_ids"][batch_idx])
                span_hidden = hidden[batch_idx, start:end, :]
                sent_vec = span_hidden.mean(dim=0)
            except ValueError:
                mask = batch["attention_mask"][batch_idx].unsqueeze(-1).float()
                sent_vec = torch.sum(hidden[batch_idx] * mask, dim=0) / torch.clamp(mask.sum(), min=1e-9)
            sent_vecs.append(sent_vec)

        sent_vecs = torch.stack(sent_vecs, dim=0)
        return F.normalize(sent_vecs, p=2, dim=1)

    @torch.no_grad()
    def score_sentences(
        self,
        question: str,
        sentences: Sequence[str],
        marked_contexts: Sequence[str],
    ) -> Tuple[List[float], torch.Tensor]:
        if len(sentences) != len(marked_contexts):
            raise ValueError(
                f"len(sentences)={len(sentences)} but len(marked_contexts)={len(marked_contexts)}"
            )

        self.eval()
        if not sentences:
            return [], torch.empty((0, 0))
        q_emb = self.encode_question([question])
        s_emb = self.encode_marked_contexts([question] * len(marked_contexts), marked_contexts)
        sims = torch.matmul(q_emb, s_emb.T).squeeze(0)
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
        # Same OOM hazard as encode_marked_contexts, and worse: output_attentions
        # materialises layers x heads x len^2 per item.
        batch_size = max(1, int(getattr(self.config, "encode_batch_size", 16)))
        if len(marked_contexts) > batch_size:
            scores: List[float] = []
            for start in range(0, len(marked_contexts), batch_size):
                scores.extend(
                    self.attention_probe_scores(
                        question,
                        marked_contexts[start: start + batch_size],
                        probe_layers=probe_layers,
                    )
                )
            return scores

        encoded_question = self.format_query(question)
        batch = self.tokenizer(
            [encoded_question] * len(marked_contexts),
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
        for batch_idx in range(mean_attention.size(0)):
            try:
                start, end = self._find_marker_span(batch["input_ids"][batch_idx])
            except ValueError:
                scores.append(0.0)
                continue
            span_indices = list(range(start, end))
            valid_indices = [
                idx
                for idx, input_id in enumerate(batch["input_ids"][batch_idx].detach().cpu().tolist())
                if int(batch["attention_mask"][batch_idx, idx].item()) == 1 and input_id not in special_ids
            ]
            if "token_type_ids" in batch:
                question_indices = [idx for idx in valid_indices if int(batch["token_type_ids"][batch_idx, idx].item()) == 0]
            else:
                question_indices = [idx for idx in valid_indices if idx < start]

            if not question_indices or not span_indices:
                scores.append(0.0)
                continue

            q_to_span = float(mean_attention[batch_idx, question_indices][:, span_indices].mean().item())
            global_to_span = float(mean_attention[batch_idx, valid_indices][:, span_indices].mean().item())
            scores.append(0.7 * q_to_span + 0.3 * global_to_span)

        return scores

    def contrastive_loss(
        self,
        questions: Sequence[str],
        positive_marked_contexts: Sequence[str],
        negative_marked_contexts: Sequence[Sequence[str]],
    ) -> torch.Tensor:
        if len(questions) != len(positive_marked_contexts):
            raise ValueError("questions and positive_marked_contexts must have the same length")

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


def split_sentences(text: str) -> List[str]:
    stripped = text.strip()
    if not stripped:
        return []

    sentences: List[str] = []
    current: List[str] = []
    length = len(stripped)
    abbreviations = {
        "dr",
        "mr",
        "mrs",
        "ms",
        "prof",
        "inc",
        "ltd",
        "fig",
        "e.g",
        "i.e",
        "vs",
        "u.s",
        "u.k",
    }

    for idx, ch in enumerate(stripped):
        current.append(ch)
        if ch not in ".!?;":
            continue

        prev_char = stripped[idx - 1] if idx > 0 else ""
        next_char = stripped[idx + 1] if idx + 1 < length else ""
        if ch == ".":
            if prev_char.isdigit() and next_char.isdigit():
                continue
            prefix = "".join(current).strip().split()[-1].rstrip(".").lower() if current else ""
            if prefix in abbreviations:
                continue
            if next_char and next_char.islower():
                continue

        sentence = "".join(current).strip()
        if sentence:
            sentences.append(sentence)
        current = []

    tail = "".join(current).strip()
    if tail:
        sentences.append(tail)

    return sentences


def normalize_sentence_for_match(text: str) -> str:
    compact = re.sub(r"\s+", " ", text.strip().lower())
    return re.sub(r"[.!?;:]+$", "", compact)


def find_target_sentence_index(sentences: Sequence[str], target_sentence: str) -> int:
    target_norm = normalize_sentence_for_match(target_sentence)
    if not target_norm:
        return -1

    normalized_sentences = [normalize_sentence_for_match(sentence) for sentence in sentences]

    for index, sentence_norm in enumerate(normalized_sentences):
        if sentence_norm == target_norm:
            return index

    for index, sentence_norm in enumerate(normalized_sentences):
        if sentence_norm and (sentence_norm in target_norm or target_norm in sentence_norm):
            return index

    return -1


def build_marked_context(
    sentences: Sequence[str],
    target_index: int,
    marker_start: str,
    marker_end: str,
) -> str:
    parts = []
    for idx, sentence in enumerate(sentences):
        if idx == target_index:
            parts.append(f"{marker_start}{sentence}{marker_end}")
        else:
            parts.append(sentence)
    return " ".join(parts)


def build_marked_context_window(
    sentences: Sequence[str],
    target_index: int,
    marker_start: str,
    marker_end: str,
    max_chars: int = 900,
) -> str:
    if target_index < 0 or target_index >= len(sentences):
        raise IndexError(f"target_index out of range: {target_index}")

    total_len = len(sentences[target_index]) + len(marker_start) + len(marker_end)
    selected = [target_index]

    left = target_index - 1
    right = target_index + 1

    while True:
        added = False

        if left >= 0 and total_len + len(sentences[left]) + 1 <= max_chars:
            selected.insert(0, left)
            total_len += len(sentences[left]) + 1
            left -= 1
            added = True

        if right < len(sentences) and total_len + len(sentences[right]) + 1 <= max_chars:
            selected.append(right)
            total_len += len(sentences[right]) + 1
            right += 1
            added = True

        if not added:
            break

    parts = []
    for idx in selected:
        sentence = sentences[idx]
        if idx == target_index:
            sentence = f"{marker_start}{sentence}{marker_end}"
        parts.append(sentence)

    return " ".join(parts)


def build_marked_context_from_span(
    context: str,
    start: int,
    end: int,
    marker_start: str,
    marker_end: str,
) -> str:
    if start < 0 or end > len(context) or start >= end:
        raise ValueError("invalid span for marker insertion")
    return context[:start] + marker_start + context[start:end] + marker_end + context[end:]


def build_marked_context_from_span_window(
    context: str,
    start: int,
    end: int,
    marker_start: str,
    marker_end: str,
    max_chars: int = 900,
) -> str:
    if start < 0 or end > len(context) or start >= end:
        raise ValueError("invalid span for marker insertion")

    span_text = context[start:end]
    budget = max(max_chars - len(marker_start) - len(marker_end) - len(span_text), 0)
    left_budget = budget // 2
    right_budget = budget - left_budget

    left_start = max(0, start - left_budget)
    right_end = min(len(context), end + right_budget)

    if left_start == 0:
        right_end = min(len(context), right_end + (left_budget - (start - left_start)))
    if right_end == len(context):
        left_start = max(0, left_start - (right_budget - (right_end - end)))

    local_context = context[left_start:start] + marker_start + span_text + marker_end + context[end:right_end]
    return local_context.strip()


def build_marked_context_from_text(
    context: str,
    target_sentence: str,
    marker_start: str,
    marker_end: str,
    use_window: bool = False,
    max_chars: int = 900,
) -> str:
    sentences = split_sentences(context)
    target_index = find_target_sentence_index(sentences, target_sentence)

    if target_index != -1:
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

    raw_target = target_sentence.strip()
    fallback_candidates = []
    if raw_target:
        fallback_candidates.append(raw_target)
    stripped_target = re.sub(r"[.!?;:]+$", "", raw_target)
    if stripped_target and stripped_target not in fallback_candidates:
        fallback_candidates.append(stripped_target)

    for candidate in fallback_candidates:
        start = context.find(candidate)
        if start != -1:
            end = start + len(candidate)
            if use_window:
                return build_marked_context_from_span_window(
                    context=context,
                    start=start,
                    end=end,
                    marker_start=marker_start,
                    marker_end=marker_end,
                    max_chars=max_chars,
                )
            return build_marked_context_from_span(
                context=context,
                start=start,
                end=end,
                marker_start=marker_start,
                marker_end=marker_end,
            )

    raise ValueError("target_sentence not found in context after sentence splitting")

