from __future__ import annotations

import re
from dataclasses import dataclass
from typing import List, Sequence, Tuple

import numpy as np
import torch

from intra_sentence_model.span_feature_utils import build_span_feature_dict, features_to_vector
from intra_sentence_model.span_model import load_span_model
from pipeline.dac_adapter import DacTokenAdapter


QUESTION_STOPWORDS = {
    "\u4ec0\u4e48",
    "\u4e3a\u4f55",
    "\u4e3a\u4ec0\u4e48",
    "\u5982\u4f55",
    "\u600e\u4e48",
    "\u54ea\u4e9b",
    "\u54ea\u4e2a",
    "\u591a\u5c11",
    "\u662f\u5426",
    "\u4ee5\u53ca",
    "\u8fd9\u4e2a",
    "\u90a3\u4e2a",
    "\u4e00\u79cd",
    "\u5173\u4e8e",
    "\u8bf7\u95ee",
    "the",
    "a",
    "an",
    "of",
    "for",
    "to",
    "in",
    "on",
    "with",
    "and",
    "or",
    "is",
    "are",
    "what",
    "why",
    "how",
}

CAUSE_HINTS = (
    "\u56e0\u4e3a",
    "\u539f\u56e0",
    "\u5bfc\u81f4",
    "\u56e0\u6b64",
    "\u6240\u4ee5",
    "\u7531\u4e8e",
    "\u673a\u5236",
    "\u4f7f\u5f97",
    "because",
    "due to",
    "therefore",
)
COMPARISON_HINTS = (
    "\u6bd4\u8f83",
    "\u533a\u522b",
    "\u4e0d\u540c",
    "\u76f8\u6bd4",
    "\u4f18\u4e8e",
    "\u52a3\u4e8e",
    "\u66f4",
    "less",
    "more",
    "than",
    "versus",
)
PROCEDURE_HINTS = (
    "\u6b65\u9aa4",
    "\u6d41\u7a0b",
    "\u9996\u5148",
    "\u7136\u540e",
    "\u63a5\u7740",
    "\u6700\u540e",
    "first",
    "then",
    "next",
    "finally",
)
NUMERIC_HINTS = ("\u591a\u5c11", "\u51e0", "\u6570\u503c", "\u6bd4\u4f8b", "\u53c2\u6570", "percent", "rate")
FACTOID_HINTS = (
    "\u8c01",
    "\u4f55\u65f6",
    "\u54ea\u91cc",
    "\u54ea\u4e00",
    "\u65f6\u95f4",
    "\u5730\u70b9",
    "when",
    "where",
    "who",
)
NEGATION_HINTS = ("\u4e0d", "\u6ca1", "\u65e0", "\u5e76\u975e", "\u4e0d\u662f", "cannot", "not", "without", "never")
EXAMPLE_HINTS = ("\u4f8b\u5982", "\u6bd4\u5982", "\u4e3e\u4f8b", "for example", "such as", "e.g.")
BOUNDARY_CHARS = set("\uff0c,\uff1b;\uff1a:\u3002\uff01\uff1f!?")
OPEN_BRACKETS = "([\uff08\u3010"
CLOSE_BRACKETS = ")]\uff09\u3011"
COMPLEX_QTYPES = {"cause", "comparison", "procedure"}


def normalize_scores(values: Sequence[float]) -> List[float]:
    arr = np.asarray(list(values), dtype=float)
    if arr.size == 0:
        return []
    if np.isclose(arr.max(), arr.min()):
        return [0.5 for _ in arr]
    norm = (arr - arr.min()) / (arr.max() - arr.min() + 1e-9)
    return norm.tolist()


def detect_question_type(question: str) -> str:
    q = question.strip()
    if any(k in q for k in ("\u4e3a\u4ec0\u4e48", "\u539f\u56e0", "\u5bfc\u81f4", "\u5982\u4f55\u5f71\u54cd", "\u673a\u5236")):
        return "cause"
    if any(k in q for k in ("\u533a\u522b", "\u4e0d\u540c", "\u5bf9\u6bd4", "\u6bd4\u8f83", "\u5f02\u540c")):
        return "comparison"
    if any(k in q for k in ("\u5982\u4f55", "\u600e\u4e48", "\u6b65\u9aa4", "\u6d41\u7a0b", "\u5b9e\u73b0")):
        return "procedure"
    if any(k in q for k in ("\u591a\u5c11", "\u51e0", "\u6570\u503c", "\u6bd4\u4f8b", "\u53c2\u6570")):
        return "numeric"
    if any(k in q for k in ("\u662f\u4ec0\u4e48", "\u5b9a\u4e49", "\u542b\u4e49", "\u6982\u5ff5")):
        return "definition"
    if any(k in q for k in ("\u8c01", "\u4f55\u65f6", "\u54ea\u91cc", "\u54ea\u4e00")):
        return "factoid"
    return "other"


def tokenize_query_terms(text: str) -> List[str]:
    pieces = re.findall(r"[A-Za-z0-9\-]+|[\u4e00-\u9fff]{2,8}", text.lower())
    return [piece for piece in pieces if piece not in QUESTION_STOPWORDS]


def query_overlap_score(question: str, text: str) -> float:
    q_terms = set(tokenize_query_terms(question))
    if not q_terms:
        return 0.0
    t_terms = set(tokenize_query_terms(text))
    if not t_terms:
        return 0.0
    return len(q_terms & t_terms) / max(len(q_terms), 1)


def task_anchor_score(question: str, text: str, question_type: str | None = None) -> float:
    q_type = question_type or detect_question_type(question)
    score = 0.0
    lowered = text.lower()

    if q_type == "cause" and any(h in lowered for h in CAUSE_HINTS):
        score += 0.45
    if q_type == "comparison" and any(h in lowered for h in COMPARISON_HINTS):
        score += 0.45
    if q_type == "procedure" and any(h in lowered for h in PROCEDURE_HINTS):
        score += 0.45
    if q_type == "numeric" and (any(h in lowered for h in NUMERIC_HINTS) or re.search(r"\d", text)):
        score += 0.45
    if q_type == "factoid" and (
        any(h in lowered for h in FACTOID_HINTS)
        or re.search(r"\d{4}|\d{1,2}[:\uff1a]\d{2}", text)
        or re.search(r"[A-Z][a-z]+", text)
    ):
        score += 0.45

    if re.search(r"\d", text):
        score += 0.15
    if any(h in lowered for h in NEGATION_HINTS):
        score += 0.15
    if re.search(r"[A-Za-z]{2,}|[\u4e00-\u9fff]{2,8}", text):
        score += 0.10

    return float(min(score, 1.0))


def compute_task_reward(question: str, text: str, question_type: str | None = None) -> float:
    q_type = question_type or detect_question_type(question)
    overlap = query_overlap_score(question, text)
    anchor = task_anchor_score(question, text, q_type)
    bonus = 0.08 if q_type in COMPLEX_QTYPES and len(text) > 12 else 0.0
    return float(min(0.55 * overlap + 0.35 * anchor + bonus, 1.0))


@dataclass
class IntraSentenceCompressionConfig:
    target_keep_ratio: float = 0.78
    min_keep_ratio: float = 0.55
    max_keep_ratio: float = 0.92
    attention_weight: float = 0.45
    anchor_weight: float = 0.30
    overlap_weight: float = 0.25
    reward_weight: float = 0.15
    probe_layers: int = 2
    min_sentence_chars: int = 18
    min_spans_to_compress: int = 2


@dataclass
class SpanUnit:
    text: str
    start: int
    end: int
    kind: str
    protected: bool = False


class DynamicSpanCompressor:
    def __init__(self, sentence_encoder, config: IntraSentenceCompressionConfig | None = None, span_model_dir: str | None = None):
        self.sentence_encoder = sentence_encoder
        self.encoder = sentence_encoder.encoder
        self.tokenizer = sentence_encoder.tokenizer
        self.device = sentence_encoder.device
        self.max_length = sentence_encoder.config.max_length
        self.config = config or IntraSentenceCompressionConfig()
        self.special_token_ids = set(getattr(self.tokenizer, "all_special_ids", []))
        self.dac_adapter = DacTokenAdapter(sentence_encoder)
        self.learned_span_model = None
        self.learned_span_metadata = None
        self.learned_span_threshold = 0.5
        if span_model_dir:
            self._load_trained_span_model(span_model_dir)

    def _load_trained_span_model(self, span_model_dir: str) -> None:
        try:
            model, metadata = load_span_model(span_model_dir, self.device)
        except Exception:
            self.learned_span_model = None
            self.learned_span_metadata = None
            self.learned_span_threshold = 0.5
            return
        self.learned_span_model = model
        self.learned_span_metadata = metadata
        self.learned_span_threshold = float(metadata.get("threshold", 0.5))

    def predict_learned_keep_scores(
        self,
        question: str,
        spans: Sequence[SpanUnit],
        question_type: str,
        sentence_score: float,
        keep_ratio: float,
        attention_scores: Sequence[float],
        dac_scores: Sequence[float],
    ) -> List[float]:
        if self.learned_span_model is None or not spans:
            return []

        sentence = "".join(span.text for span in spans)
        sentence_length = max(len(sentence), 1)
        sentence_token_length = max(self._count_tokens(sentence), 1)
        feature_rows = []
        for idx, span in enumerate(spans):
            feature_dict = build_span_feature_dict(
                question=question,
                span_text=span.text,
                span_kind=span.kind,
                span_index=idx,
                num_spans=len(spans),
                sentence_length=sentence_length,
                sentence_token_length=sentence_token_length,
                start=span.start,
                end=span.end,
                sentence_score=sentence_score,
                keep_ratio=keep_ratio,
                question_type=question_type,
                attention_score=attention_scores[idx] if idx < len(attention_scores) else 0.0,
                dac_score=dac_scores[idx] if idx < len(dac_scores) else 0.0,
                answer_overlap=0.0,
                answer_drop=0.0,
                protected=span.protected,
            )
            feature_rows.append(features_to_vector(feature_dict))

        x = torch.tensor(feature_rows, dtype=torch.float32, device=self.device)
        with torch.no_grad():
            logits = self.learned_span_model(x)
            probs = torch.sigmoid(logits).detach().cpu().tolist()
        return [float(p) for p in probs]

    def compress_sentences(
        self,
        question: str,
        sentences: Sequence[str],
        sentence_scores: Sequence[float],
    ) -> Tuple[List[str], dict]:
        if not sentences:
            return [], {"sentence_stats": [], "removed_span_count": 0}

        keep_ratios = self.allocate_sentence_keep_ratios(question, sentence_scores)
        compressed_sentences: List[str] = []
        sentence_stats = []
        removed_total = 0

        for sentence, keep_ratio, score in zip(sentences, keep_ratios, sentence_scores):
            compressed, stats = self.compress_sentence(question, sentence, keep_ratio, score)
            stats["sentence_score"] = float(score)
            stats["keep_ratio"] = float(keep_ratio)
            compressed_sentences.append(compressed)
            sentence_stats.append(stats)
            removed_total += int(stats["removed_span_count"])

        return compressed_sentences, {
            "sentence_stats": sentence_stats,
            "removed_span_count": removed_total,
        }

    def allocate_sentence_keep_ratios(
        self,
        question: str,
        sentence_scores: Sequence[float],
    ) -> List[float]:
        normalized_scores = normalize_scores(sentence_scores)
        question_type = detect_question_type(question)
        complexity_bonus = 0.04 if question_type in COMPLEX_QTYPES else 0.0

        ratios = []
        for score in normalized_scores:
            ratio = self.config.target_keep_ratio + 0.18 * score + complexity_bonus
            ratio = min(self.config.max_keep_ratio, max(self.config.min_keep_ratio, ratio))
            ratios.append(float(ratio))
        return ratios

    def compress_sentence(
        self,
        question: str,
        sentence: str,
        keep_ratio: float,
        sentence_score: float = 0.5,
    ) -> Tuple[str, dict]:
        original = sentence.strip()
        question_type = detect_question_type(question)

        if len(original) < self.config.min_sentence_chars:
            return original, self._build_sentence_stats(original, original, [], [])

        spans = self.split_sentence_into_spans(question, original, question_type)
        if len(spans) < self.config.min_spans_to_compress:
            return original, self._build_sentence_stats(original, original, spans, [])

        target_tokens = max(1, int(self._count_tokens(original) * keep_ratio))
        current_spans = spans[:]
        removed_spans: List[str] = []

        while len(current_spans) > 1 and self._count_span_tokens(current_spans) > target_tokens:
            ranked_candidates = self.rank_removal_candidates(question, current_spans, question_type, sentence_score, keep_ratio)
            if not ranked_candidates:
                break

            _, remove_idx, remove_text = ranked_candidates[0]
            removed_spans.append(remove_text)
            current_spans.pop(remove_idx)

        compressed = self.cleanup_sentence("".join(span.text for span in current_spans), original)
        if not compressed:
            compressed = original

        stats = self._build_sentence_stats(original, compressed, current_spans, removed_spans)
        if self.learned_span_model is not None:
            stats["compression_mode"] = "hybrid_learned_span_prune"
        else:
            stats["compression_mode"] = "dac_guided_span_prune" if self.dac_adapter.available else "span_prune"
        return compressed, stats

    def rank_removal_candidates(
        self,
        question: str,
        spans: Sequence[SpanUnit],
        question_type: str,
        sentence_score: float,
        keep_ratio: float,
    ) -> List[Tuple[float, int, str]]:
        attention_scores = normalize_scores(self.compute_span_attention_scores(question, spans))
        dac_scores = normalize_scores(self.compute_dac_span_scores(question, spans))
        overlap_scores = [query_overlap_score(question, span.text) for span in spans]
        anchor_scores = [task_anchor_score(question, span.text, question_type) for span in spans]
        reward_scores = normalize_scores([compute_task_reward(question, span.text, question_type) for span in spans])
        learned_keep_scores = self.predict_learned_keep_scores(
            question=question,
            spans=spans,
            question_type=question_type,
            sentence_score=sentence_score,
            keep_ratio=keep_ratio,
            attention_scores=attention_scores,
            dac_scores=dac_scores,
        )

        anchorful_count = sum(
            1
            for overlap, anchor in zip(overlap_scores, anchor_scores)
            if overlap > 0.0 or anchor >= 0.45
        )

        candidates = []
        dac_weight = 0.20 if any(score > 0.0 for score in dac_scores) else 0.0
        model_weight = 0.40 if learned_keep_scores else 0.0
        for idx, span in enumerate(spans):
            if span.protected:
                continue
            if len(spans) <= 3 and idx in {0, len(spans) - 1}:
                continue

            overlap = overlap_scores[idx]
            anchor = anchor_scores[idx]
            reward = reward_scores[idx]
            if anchorful_count <= 1 and (overlap > 0.0 or anchor >= 0.45):
                continue

            heuristic_importance = (
                self.config.attention_weight * attention_scores[idx]
                + dac_weight * dac_scores[idx]
                + self.config.anchor_weight * anchor
                + self.config.overlap_weight * overlap
                + self.config.reward_weight * reward
            )
            importance = heuristic_importance
            if learned_keep_scores:
                importance = (1.0 - model_weight) * heuristic_importance + model_weight * learned_keep_scores[idx]
            if span.kind == "parenthetical":
                importance -= 0.18
            elif span.kind == "example":
                importance -= 0.12
            elif span.kind == "tail":
                importance -= 0.05
            if self.is_low_value_filler(span.text):
                importance -= 0.08

            candidates.append((float(importance), idx, span.text))

        candidates.sort(key=lambda item: item[0])
        return candidates

    def split_sentence_into_spans(
        self,
        question: str,
        sentence: str,
        question_type: str,
    ) -> List[SpanUnit]:
        spans: List[SpanUnit] = []
        current: List[str] = []
        depth = 0
        start = 0

        for idx, ch in enumerate(sentence):
            current.append(ch)
            if ch in OPEN_BRACKETS:
                depth += 1
            elif ch in CLOSE_BRACKETS and depth > 0:
                depth -= 1

            if depth == 0 and ch in BOUNDARY_CHARS:
                text = "".join(current).strip()
                if text:
                    spans.append(self.build_span(question, text, start, idx + 1, question_type))
                current = []
                start = idx + 1

        if current:
            text = "".join(current).strip()
            if text:
                spans.append(self.build_span(question, text, start, len(sentence), question_type))

        return spans

    def build_protected_char_mask(
        self,
        question: str,
        sentence: str,
        spans: Sequence[SpanUnit],
        question_type: str,
    ) -> List[bool]:
        mask = [False] * len(sentence)
        keywords = set(tokenize_query_terms(question))
        if question_type == "cause":
            keywords.update(CAUSE_HINTS)
        elif question_type == "comparison":
            keywords.update(COMPARISON_HINTS)
        elif question_type == "procedure":
            keywords.update(PROCEDURE_HINTS)
        elif question_type == "numeric":
            keywords.update(NUMERIC_HINTS)
        elif question_type == "factoid":
            keywords.update(FACTOID_HINTS)

        for keyword in keywords:
            if not keyword:
                continue
            for match in re.finditer(re.escape(keyword), sentence, flags=re.IGNORECASE):
                for pos in range(match.start(), match.end()):
                    if 0 <= pos < len(mask):
                        mask[pos] = True

        for match in re.finditer(r"\d+", sentence):
            for pos in range(match.start(), match.end()):
                mask[pos] = True

        for span in spans:
            if span.protected and len(span.text) <= 12:
                for pos in range(span.start, min(span.end, len(mask))):
                    mask[pos] = True

        return mask

    def compute_dac_span_scores(
        self,
        question: str,
        spans: Sequence[SpanUnit],
    ) -> List[float]:
        sentence = "".join(span.text for span in spans)
        dac_scores = self.dac_adapter.score_spans(question, sentence, spans)
        if dac_scores is None:
            return [0.0 for _ in spans]
        return dac_scores

    def build_span(
        self,
        question: str,
        text: str,
        start: int,
        end: int,
        question_type: str,
    ) -> SpanUnit:
        stripped = text.strip()
        lowered = stripped.lower()

        kind = "content"
        if stripped.startswith(("(", "\uff08", "\u3010")) and stripped.endswith((")", "\uff09", "\u3011")):
            kind = "parenthetical"
        elif any(h in lowered for h in EXAMPLE_HINTS):
            kind = "example"
        elif start > 0 and len(stripped) <= 12:
            kind = "tail"

        overlap = query_overlap_score(question, stripped)
        anchor = task_anchor_score(question, stripped, question_type)
        protected = overlap > 0.0 or anchor >= 0.45
        if stripped.endswith(("\uff1a", ":")) and len(stripped) > 8:
            protected = True
        if re.search(r"\b(i|ii|iii)\b", lowered) or any(tag in stripped for tag in ("\u4e00\u662f", "\u4e8c\u662f", "\u4e09\u662f")):
            protected = True
        if kind == "parenthetical" and overlap == 0.0 and anchor < 0.45:
            protected = False

        return SpanUnit(
            text=stripped,
            start=start,
            end=end,
            kind=kind,
            protected=protected,
        )

    def compute_span_attention_scores(
        self,
        question: str,
        spans: Sequence[SpanUnit],
    ) -> List[float]:
        sentence = "".join(span.text for span in spans)
        if not sentence or not getattr(self.tokenizer, "is_fast", False):
            return [0.0 for _ in spans]

        try:
            batch = self.tokenizer(
                question,
                sentence,
                truncation=True,
                max_length=self.max_length,
                return_tensors="pt",
                return_offsets_mapping=True,
            )
        except (NotImplementedError, TypeError, ValueError):
            return [0.0 for _ in spans]

        if not getattr(batch, "encodings", None):
            return [0.0 for _ in spans]

        encoding = batch.encodings[0]
        sequence_ids = encoding.sequence_ids
        offset_mapping = batch["offset_mapping"][0].tolist()
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
            return [0.0 for _ in spans]

        last_layers = attentions[-self.config.probe_layers :]
        attn = torch.stack([layer[0].mean(dim=0) for layer in last_layers], dim=0).mean(dim=0)

        question_indices = [
            idx
            for idx, (seq_id, offsets, input_id) in enumerate(zip(sequence_ids, offset_mapping, input_ids))
            if seq_id == 0 and offsets[1] > offsets[0] and input_id not in self.special_token_ids
        ]
        if not question_indices:
            return [0.0 for _ in spans]

        scores = []
        for span in spans:
            span_indices = [
                idx
                for idx, (seq_id, offsets, input_id) in enumerate(zip(sequence_ids, offset_mapping, input_ids))
                if seq_id == 1
                and offsets[1] > offsets[0]
                and offsets[1] > span.start
                and offsets[0] < span.end
                and input_id not in self.special_token_ids
            ]
            if not span_indices:
                scores.append(0.0)
                continue

            q_to_span = float(attn[question_indices][:, span_indices].mean().item())
            span_self = float(attn[span_indices][:, span_indices].mean().item())
            scores.append(0.75 * q_to_span + 0.25 * span_self)

        return scores

    def is_low_value_filler(self, text: str) -> bool:
        lowered = text.lower()
        if len(text) <= 8:
            return False
        if any(h in lowered for h in EXAMPLE_HINTS):
            return True
        filler_markers = (
            "\u603b\u7684\u6765\u8bf4",
            "\u603b\u800c\u8a00\u4e4b",
            "\u6362\u53e5\u8bdd\u8bf4",
            "\u6362\u8a00\u4e4b",
            "\u603b\u4f53\u6765\u770b",
        )
        return any(marker in lowered for marker in filler_markers)

    def cleanup_sentence(self, text: str, original: str) -> str:
        cleaned = re.sub(r"\s+", " ", text).strip()
        cleaned = re.sub(r"\s+([\uff0c\u3002\uff01\uff1f\uff1b\uff1a,.;!?])", r"\1", cleaned)
        cleaned = re.sub(r"([\uff08(\u3010])\s+", r"\1", cleaned)
        cleaned = re.sub(r"\s+([\uff09)\u3011])", r"\1", cleaned)
        cleaned = re.sub(r"([\uff0c\uff1b\uff1a,;:]){2,}", lambda m: m.group(0)[0], cleaned)
        cleaned = cleaned.strip("\uff0c,\uff1b;\uff1a:")

        if original and original[-1] in "\u3002\uff01\uff1f.!?" and (not cleaned or cleaned[-1] not in "\u3002\uff01\uff1f.!?"):
            cleaned = cleaned + original[-1]
        return cleaned

    def _count_tokens(self, text: str) -> int:
        if not text.strip():
            return 0
        try:
            return max(1, len(self.tokenizer.tokenize(text)))
        except Exception:
            return max(1, len(text))

    def _count_span_tokens(self, spans: Sequence[SpanUnit]) -> int:
        return sum(self._count_tokens(span.text) for span in spans)

    def _build_sentence_stats(
        self,
        original: str,
        compressed: str,
        kept_spans: Sequence[SpanUnit],
        removed_spans: Sequence[str],
    ) -> dict:
        return {
            "original_sentence": original,
            "compressed_sentence": compressed,
            "removed_spans": list(removed_spans),
            "removed_span_count": len(removed_spans),
            "kept_spans": [span.text for span in kept_spans],
            "original_tokens": self._count_tokens(original),
            "compressed_tokens": self._count_tokens(compressed),
        }









