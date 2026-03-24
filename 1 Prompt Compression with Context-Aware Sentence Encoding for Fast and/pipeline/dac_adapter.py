from __future__ import annotations

import re
from dataclasses import dataclass
from typing import List, Optional, Sequence

import torch
import torch.nn.functional as F
from transformers import AutoModelForMaskedLM


@dataclass
class DacCompressionConfig:
    fusion: str = "additive"
    alpha: float = 0.8
    max_dyn_steps: int = 6
    min_tokens: int = 6
    preserve_punct: bool = True
    avoid_consecutive: bool = True


class DacTokenAdapter:
    def __init__(self, sentence_encoder, config: DacCompressionConfig | None = None):
        self.sentence_encoder = sentence_encoder
        self.tokenizer = sentence_encoder.tokenizer
        self.encoder = sentence_encoder.encoder
        self.device = sentence_encoder.device
        self.max_length = sentence_encoder.config.max_length
        self.config = config or DacCompressionConfig()
        self.special_token_ids = set(getattr(self.tokenizer, "all_special_ids", []))
        self.available = False
        self.entropy_model = None

        try:
            self.entropy_model = AutoModelForMaskedLM.from_pretrained(
                sentence_encoder.config.model_name,
                attn_implementation="eager",
            ).to(self.device)
            self.entropy_model.eval()
            self.available = self.tokenizer.mask_token_id is not None
        except Exception:
            self.entropy_model = None
            self.available = False

    def normalize(self, tensor: torch.Tensor) -> torch.Tensor:
        if tensor.dim() == 0:
            tensor = tensor.unsqueeze(0)
        min_val = tensor.min()
        max_val = tensor.max()
        if float(max_val - min_val) > 1e-8:
            return (tensor - min_val) / (max_val - min_val)
        return torch.zeros_like(tensor)

    def _fuse_additive(self, losses: torch.Tensor, attention: torch.Tensor) -> torch.Tensor:
        loss_norm = self.normalize(losses)
        attn_norm = self.normalize(attention)
        return self.config.alpha * attn_norm + (1.0 - self.config.alpha) * loss_norm

    def _fuse_multiplicative(self, losses: torch.Tensor, attention: torch.Tensor) -> torch.Tensor:
        loss_norm = self.normalize(losses)
        attn_norm = self.normalize(attention)
        return loss_norm * attn_norm

    def _token_count(self, text: str) -> int:
        try:
            return len(self.tokenizer.tokenize(text))
        except Exception:
            return len(text)

    def compute_token_losses(self, text: str) -> tuple[Optional[torch.Tensor], Optional[List[List[int]]]]:
        if not self.available or not text.strip():
            return None, None

        encoding = self.tokenizer(
            text,
            add_special_tokens=False,
            truncation=True,
            max_length=self.max_length,
            return_tensors="pt",
            return_offsets_mapping=True,
        )
        if encoding["input_ids"].size(1) == 0:
            return None, None

        input_ids = encoding["input_ids"].to(self.device)
        attention_mask = encoding["attention_mask"].to(self.device)
        offsets = encoding["offset_mapping"][0].tolist()
        seq_len = input_ids.size(1)
        if seq_len < self.config.min_tokens:
            return None, offsets

        mask_token_id = self.tokenizer.mask_token_id
        target_ids = input_ids[0]
        losses = []
        chunk_size = 32
        position_ids = torch.arange(seq_len, device=self.device)

        with torch.no_grad():
            for start in range(0, seq_len, chunk_size):
                end = min(seq_len, start + chunk_size)
                batch_size = end - start
                masked_batch = input_ids.repeat(batch_size, 1)
                masked_batch[torch.arange(batch_size, device=self.device), position_ids[start:end]] = mask_token_id
                mask_batch = attention_mask.repeat(batch_size, 1)
                outputs = self.entropy_model(
                    input_ids=masked_batch,
                    attention_mask=mask_batch,
                )
                logits = outputs.logits[torch.arange(batch_size, device=self.device), position_ids[start:end]]
                batch_losses = F.cross_entropy(
                    logits,
                    target_ids[start:end],
                    reduction="none",
                )
                losses.append(batch_losses)

        return torch.cat(losses, dim=0), offsets

    def compute_token_attention(self, question: str, text: str) -> Optional[torch.Tensor]:
        if not text.strip() or not getattr(self.tokenizer, "is_fast", False):
            return None

        try:
            batch = self.tokenizer(
                question,
                text,
                truncation=True,
                max_length=self.max_length,
                return_tensors="pt",
                return_offsets_mapping=True,
            )
        except Exception:
            return None

        if not getattr(batch, "encodings", None):
            return None

        encoding = batch.encodings[0]
        sequence_ids = encoding.sequence_ids
        offsets = batch["offset_mapping"][0].tolist()
        input_ids = batch["input_ids"][0].tolist()
        model_inputs = {
            key: value.to(self.device)
            for key, value in batch.items()
            if key != "offset_mapping"
        }

        with torch.no_grad():
            outputs = self.encoder(**model_inputs, output_attentions=True)

        attentions = getattr(outputs, "attentions", None)
        if not attentions:
            return None

        attn = torch.stack([layer[0].mean(dim=0) for layer in attentions[-2:]], dim=0).mean(dim=0)
        question_indices = [
            idx
            for idx, (seq_id, offset, input_id) in enumerate(zip(sequence_ids, offsets, input_ids))
            if seq_id == 0 and offset[1] > offset[0] and input_id not in self.special_token_ids
        ]
        if not question_indices:
            return None

        token_scores = []
        for idx, (seq_id, offset, input_id) in enumerate(zip(sequence_ids, offsets, input_ids)):
            if seq_id != 1 or offset[1] <= offset[0] or input_id in self.special_token_ids:
                continue
            q_to_token = float(attn[question_indices][:, [idx]].mean().item())
            ctx_to_token = float(attn[:, [idx]].mean().item())
            token_scores.append(0.75 * q_to_token + 0.25 * ctx_to_token)

        if not token_scores:
            return None
        return torch.tensor(token_scores, dtype=torch.float32, device=self.device)

    def compute_fused_token_scores(self, question: str, text: str):
        losses, offsets = self.compute_token_losses(text)
        if losses is None or offsets is None:
            return None, None

        attention = self.compute_token_attention(question, text)
        if attention is None or attention.numel() != losses.numel():
            attention = torch.zeros_like(losses)

        if self.config.fusion == "multiplicative":
            fused = self._fuse_multiplicative(losses, attention)
        else:
            fused = self._fuse_additive(losses, attention)
        return fused, offsets

    def score_spans(self, question: str, text: str, spans) -> Optional[List[float]]:
        fused, offsets = self.compute_fused_token_scores(question, text)
        if fused is None or offsets is None:
            return None

        scores = []
        for span in spans:
            token_indices = [
                idx
                for idx, (start, end) in enumerate(offsets)
                if end > start and end > span.start and start < span.end
            ]
            if not token_indices:
                scores.append(0.0)
                continue
            scores.append(float(fused[token_indices].mean().item()))
        return scores

    def preserve_punctuation_mask(self, text: str, offsets: Sequence[Sequence[int]]) -> torch.Tensor:
        punct_pattern = re.compile(r"^\s*[^\w\s]+\s*$")
        preserve = []
        for start, end in offsets:
            token_text = text[start:end] if end > start else ""
            preserve.append(bool(punct_pattern.match(token_text)))
        return torch.tensor(preserve, dtype=torch.bool, device=self.device)

    def protected_token_mask(
        self,
        offsets: Sequence[Sequence[int]],
        protected_char_mask: Optional[Sequence[bool]],
    ) -> torch.Tensor:
        if protected_char_mask is None:
            return torch.zeros(len(offsets), dtype=torch.bool, device=self.device)

        protected = []
        for start, end in offsets:
            keep = False
            for pos in range(start, end):
                if pos < len(protected_char_mask) and protected_char_mask[pos]:
                    keep = True
                    break
            protected.append(keep)
        return torch.tensor(protected, dtype=torch.bool, device=self.device)

    def select_keep_indices(
        self,
        score: torch.Tensor,
        compress_ratio: float,
        punct_mask: torch.Tensor,
        protect_mask: torch.Tensor,
    ) -> torch.Tensor:
        total_tokens = score.numel()
        keep_count = max(1, int(total_tokens * (1 - compress_ratio)))
        working_score = score.clone()
        if self.config.preserve_punct:
            working_score = working_score + 1e5 * punct_mask.float()
        working_score = working_score + 1e5 * protect_mask.float()

        _, keep_indices = torch.topk(working_score.view(-1), k=keep_count, largest=True)
        keep_indices = torch.sort(keep_indices)[0]

        if not self.config.avoid_consecutive:
            return keep_indices

        all_indices = torch.arange(total_tokens, device=self.device)
        delete_indices = all_indices[~torch.isin(all_indices, keep_indices)]
        if delete_indices.numel() <= 1:
            return keep_indices

        differences = delete_indices[1:] - delete_indices[:-1]
        extra_keep_mask = torch.zeros_like(delete_indices, dtype=torch.bool)
        extra_keep_mask[1:] = differences == 1
        extra_keep_mask[0] = False

        for idx in range(1, extra_keep_mask.numel()):
            if extra_keep_mask[idx - 1]:
                extra_keep_mask[idx] = False

        rescued_indices = delete_indices[extra_keep_mask]
        if rescued_indices.numel() == 0:
            return keep_indices
        return torch.sort(torch.cat((keep_indices, rescued_indices)))[0]

    def reconstruct_text(self, text: str, offsets: Sequence[Sequence[int]], keep_indices: Sequence[int]) -> str:
        char_keep = [False] * len(text)
        for idx in keep_indices:
            start, end = offsets[int(idx)]
            for pos in range(start, end):
                if 0 <= pos < len(char_keep):
                    char_keep[pos] = True

        chars = []
        for pos, ch in enumerate(text):
            if char_keep[pos]:
                chars.append(ch)
                continue
            if ch.isspace() and (
                (pos > 0 and char_keep[pos - 1])
                or (pos + 1 < len(char_keep) and char_keep[pos + 1])
            ):
                chars.append(ch)
        return "".join(chars).strip()

    def compress(
        self,
        question: str,
        text: str,
        keep_ratio: float,
        protected_char_mask: Optional[Sequence[bool]] = None,
    ) -> Optional[dict]:
        if not self.available:
            return None

        current_text = text
        compress_ratio = max(0.0, 1.0 - keep_ratio)
        seq_len = self._token_count(text)
        if seq_len < self.config.min_tokens:
            return None

        dyn_steps = min(max(1, seq_len // 12), self.config.max_dyn_steps)
        real_ratio = 1.0 - (1.0 - compress_ratio) ** (1.0 / dyn_steps)

        for _ in range(dyn_steps):
            losses, offsets = self.compute_token_losses(current_text)
            if losses is None or offsets is None or losses.numel() < self.config.min_tokens:
                break

            attention = self.compute_token_attention(question, current_text)
            if attention is None or attention.numel() != losses.numel():
                attention = torch.zeros_like(losses)

            if self.config.fusion == "multiplicative":
                fused_score = self._fuse_multiplicative(losses, attention)
            else:
                fused_score = self._fuse_additive(losses, attention)

            punct_mask = self.preserve_punctuation_mask(current_text, offsets)
            protect_mask = self.protected_token_mask(offsets, protected_char_mask)
            keep_indices = self.select_keep_indices(fused_score, real_ratio, punct_mask, protect_mask)
            next_text = self.reconstruct_text(current_text, offsets, keep_indices.tolist())
            if not next_text or next_text == current_text:
                break
            current_text = next_text

        if current_text == text:
            return None

        return {
            "compressed_text": current_text,
            "dyn_steps": dyn_steps,
            "fusion": self.config.fusion,
            "alpha": self.config.alpha,
        }

