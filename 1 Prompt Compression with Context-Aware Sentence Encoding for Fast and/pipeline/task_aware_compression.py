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
    "a",
    "an",
    "and",
    "are",
    "as",
    "at",
    "be",
    "by",
    "can",
    "could",
    "do",
    "does",
    "did",
    "for",
    "from",
    "how",
    "in",
    "is",
    "it",
    "of",
    "on",
    "or",
    "should",
    "that",
    "the",
    "to",
    "was",
    "were",
    "what",
    "when",
    "where",
    "which",
    "who",
    "why",
    "with",
}
CAUSE_HINTS = (
    "because",
    "cause",
    "caused",
    "causes",
    "consequence",
    "consequences",
    "due to",
    "effect",
    "effects",
    "impact",
    "impacts",
    "lead to",
    "leads to",
    "mechanism",
    "result",
    "results",
    "risk",
    "risks",
    "therefore",
    "trigger",
    "triggers",
)
COMPARISON_HINTS = (
    "better",
    "compared",
    "comparison",
    "contrast",
    "differ",
    "difference",
    "different",
    "less",
    "more",
    "than",
    "versus",
    "vs",
    "whereas",
    "while",
)
PROCEDURE_HINTS = (
    "after",
    "before",
    "finally",
    "first",
    "method",
    "next",
    "process",
    "second",
    "step",
    "steps",
    "then",
    "workflow",
)
NUMERIC_HINTS = (
    "amount",
    "average",
    "count",
    "how many",
    "how much",
    "number",
    "percent",
    "percentage",
    "rate",
    "ratio",
    "threshold",
    "value",
)
FACTOID_HINTS = (
    "city",
    "country",
    "date",
    "location",
    "name",
    "person",
    "place",
    "time",
    "when",
    "where",
    "which",
    "who",
)
NEGATION_HINTS = ("cannot", "except", "lack", "lacks", "neither", "never", "no", "not", "without")
EXAMPLE_HINTS = ("e.g.", "example", "examples", "for example", "for instance", "such as")
OUTCOME_HINTS = ("consequence", "effect", "impact", "outcome", "result", "risk", "trigger")
EXPLANATORY_EVIDENCE_HINTS = (
    "decline",
    "evidence",
    "example",
    "examples",
    "include",
    "includes",
    "loss",
    "release",
    "releases",
    "risk",
    "risks",
    "severe",
    "stress",
)
BOUNDARY_CHARS = set(",;:")
OPEN_BRACKETS = "([{"
CLOSE_BRACKETS = ")]}"
COMPLEX_QTYPES = {"cause", "comparison", "procedure"}
TOKEN_RE = re.compile(r"[A-Za-z][A-Za-z0-9]*(?:-[A-Za-z0-9]+)*|\d+(?:\.\d+)?%?")


def normalize_scores(values: Sequence[float]) -> List[float]:
    arr = np.asarray(list(values), dtype=float)
    if arr.size == 0:
        return []
    if np.isclose(arr.max(), arr.min()):
        return [0.5 for _ in arr]
    norm = (arr - arr.min()) / (arr.max() - arr.min() + 1e-9)
    return norm.tolist()


def normalize_term(term: str) -> str:
    term = term.lower().strip("'\"")
    if len(term) > 4 and term.endswith("ies"):
        return term[:-3] + "y"
    for suffix in ("ing", "ed", "es", "s"):
        if len(term) > len(suffix) + 3 and term.endswith(suffix):
            return term[: -len(suffix)]
    return term


def contains_any(text: str, hints: Sequence[str]) -> bool:
    lowered = text.lower()
    return any(hint in lowered for hint in hints)


def detect_question_type(question: str) -> str:
    q = question.lower().strip()
    if contains_any(q, ("why", "cause", "reason", "impact", "effect", "consequence", "risk", "mechanism", "lead to")):
        return "cause"
    if contains_any(q, ("compare", "compared", "difference", "different", "versus", " vs ", "better", "worse")):
        return "comparison"
    if contains_any(q, ("how to", "steps", "process", "procedure", "workflow", "method", "implement")):
        return "procedure"
    if contains_any(q, NUMERIC_HINTS) or re.search(r"\b(how many|how much)\b", q):
        return "numeric"
    if re.search(r"\b(what is|what are|define|definition|meaning of)\b", q):
        return "definition"
    if re.search(r"\b(who|when|where|which|name)\b", q):
        return "factoid"
    return "other"


def tokenize_query_terms(text: str) -> List[str]:
    pieces = TOKEN_RE.findall(text.lower())
    terms = []
    for piece in pieces:
        norm = normalize_term(piece)
        if norm and norm not in QUESTION_STOPWORDS:
            terms.append(norm)
    return terms


def query_overlap_score(question: str, text: str) -> float:
    q_terms = set(tokenize_query_terms(question))
    if not q_terms:
        return 0.0
    t_terms = set(tokenize_query_terms(text))
    if not t_terms:
        return 0.0
    return len(q_terms & t_terms) / max(len(q_terms), 1)


def starts_with_example_marker(text: str) -> bool:
    stripped = text.strip().lower()
    return stripped.startswith(("for example", "for instance", "e.g.", "such as"))


def question_seeks_outcome_examples(question: str) -> bool:
    lowered = question.lower()
    list_hints = ("what", "which", "list", "name")
    return any(hint in lowered for hint in list_hints) and any(hint in lowered for hint in OUTCOME_HINTS)


def task_anchor_score(question: str, text: str, question_type: str | None = None) -> float:
    q_type = question_type or detect_question_type(question)
    score = 0.0
    lowered = text.lower()

    if q_type == "cause" and contains_any(lowered, CAUSE_HINTS):
        score += 0.45
    if q_type == "cause" and contains_any(lowered, EXPLANATORY_EVIDENCE_HINTS):
        score += 0.18
    if q_type == "comparison" and contains_any(lowered, COMPARISON_HINTS):
        score += 0.45
    if q_type == "procedure" and contains_any(lowered, PROCEDURE_HINTS):
        score += 0.45
    if q_type == "numeric" and (contains_any(lowered, NUMERIC_HINTS) or re.search(r"\d", text)):
        score += 0.45
    if q_type == "definition" and contains_any(lowered, ("defined as", "refers to", "is a", "is an", "means")):
        score += 0.35
    if q_type == "factoid" and (
        contains_any(lowered, FACTOID_HINTS)
        or re.search(r"\d{4}|\d{1,2}:\d{2}", text)
        or re.search(r"\b[A-Z][a-zA-Z0-9-]{2,}\b", text)
    ):
        score += 0.45

    if question_seeks_outcome_examples(question):
        if contains_any(lowered, OUTCOME_HINTS):
            score += 0.25
        if starts_with_example_marker(text):
            score += 0.30

    if re.search(r"\d", text):
        score += 0.15
    if contains_any(lowered, NEGATION_HINTS):
        score += 0.15
    if re.search(r"\b[A-Z][a-zA-Z0-9-]{2,}\b", text):
        score += 0.10
    if re.search(r"[A-Za-z]{3,}", text):
        score += 0.05

    return float(min(score, 1.0))


def compute_task_reward(question: str, text: str, question_type: str | None = None) -> float:
    q_type = question_type or detect_question_type(question)
    overlap = query_overlap_score(question, text)
    anchor = task_anchor_score(question, text, q_type)
    bonus = 0.08 if q_type in COMPLEX_QTYPES and len(tokenize_query_terms(text)) > 8 else 0.0
    lowered_question = question.lower()
    lowered_text = text.lower()
    if q_type == "cause":
        if contains_any(lowered_text, EXPLANATORY_EVIDENCE_HINTS):
            bonus += 0.10
        if re.search(r"\b(include|includes|such as|for example)\b", lowered_text):
            bonus += 0.10
        if contains_any(lowered_question, ("critical", "threshold", "risk", "tipping")) and contains_any(lowered_text, OUTCOME_HINTS):
            bonus += 0.10
    return float(min(0.50 * overlap + 0.35 * anchor + min(bonus, 0.28), 1.0))


@dataclass
class IntraSentenceCompressionConfig:
    target_keep_ratio: float = 0.56
    min_keep_ratio: float = 0.36
    max_keep_ratio: float = 0.76
    attention_weight: float = 0.25
    anchor_weight: float = 0.34
    overlap_weight: float = 0.34
    reward_weight: float = 0.22
    learned_keep_weight: float = 0.72
    learned_soft_protected_threshold: float = 0.28
    min_question_term_recall_after_prune: float = 0.72
    mismatch_penalty: float = 0.12
    filler_penalty: float = 0.16
    probe_layers: int = 2
    min_sentence_chars: int = 45
    min_spans_to_compress: int = 2
    min_evidence_list_items: int = 2


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

        sentence = " ".join(span.text for span in spans)
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

        keep_ratios = self.allocate_sentence_keep_ratios(question, sentences, sentence_scores)
        compressed_sentences: List[str] = []
        sentence_stats = []
        removed_total = 0

        for sentence, keep_ratio, score in zip(sentences, keep_ratios, sentence_scores):
            compressed, stats = self.compress_sentence(question, sentence, keep_ratio, score)
            stats["sentence_score"] = float(score)
            stats["keep_ratio"] = float(keep_ratio)
            if compressed.strip():
                compressed_sentences.append(compressed)
            sentence_stats.append(stats)
            removed_total += int(stats["removed_span_count"])

        return compressed_sentences, {
            "sentence_stats": sentence_stats,
            "removed_span_count": removed_total,
        }

    def compute_sentence_marginal_information_gains(
        self,
        question: str,
        sentences: Sequence[str],
        sentence_scores: Sequence[float],
    ) -> List[float]:
        if not sentences:
            return []

        relevance_scores = []
        for idx, sentence in enumerate(sentences):
            base_score = float(sentence_scores[idx]) if idx < len(sentence_scores) else 0.5
            overlap = query_overlap_score(question, sentence)
            anchor = task_anchor_score(question, sentence)
            relevance_scores.append(0.50 * base_score + 0.25 * overlap + 0.25 * anchor)

        mig_scores = []
        for idx, sentence in enumerate(sentences):
            sentence_terms = set(tokenize_query_terms(sentence))
            redundancy = 0.0
            for other_idx, other_sentence in enumerate(sentences):
                if idx == other_idx:
                    continue
                other_terms = set(tokenize_query_terms(other_sentence))
                if not sentence_terms or not other_terms:
                    continue
                overlap = len(sentence_terms & other_terms) / max(len(sentence_terms | other_terms), 1)
                redundancy = max(redundancy, overlap)
            mig_scores.append(float(relevance_scores[idx] - 0.45 * redundancy))
        return normalize_scores(mig_scores)

    def allocate_sentence_keep_ratios(
        self,
        question: str,
        sentences: Sequence[str],
        sentence_scores: Sequence[float],
    ) -> List[float]:
        normalized_scores = normalize_scores(sentence_scores)
        mig_scores = self.compute_sentence_marginal_information_gains(question, sentences, sentence_scores)
        question_type = detect_question_type(question)
        complexity_bonus = 0.02 if question_type in COMPLEX_QTYPES else 0.0

        ratios = []
        for idx, score in enumerate(normalized_scores):
            mig = mig_scores[idx] if idx < len(mig_scores) else 0.5
            ratio = self.config.target_keep_ratio + 0.05 * score + 0.04 * mig + complexity_bonus
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
        spans = self.refine_long_spans(question, spans, question_type)
        spans = self.apply_evidence_list_floor(question, spans, question_type)
        if len(spans) < self.config.min_spans_to_compress:
            return original, self._build_sentence_stats(original, original, spans, [])

        target_tokens = max(1, int(self._count_tokens(original) * keep_ratio))
        current_spans = spans[:]
        removed_spans: List[str] = []
        skipped_by_safety = 0

        while len(current_spans) > 1 and self._count_span_tokens(current_spans) > target_tokens:
            ranked_candidates = self.rank_removal_candidates(question, current_spans, question_type, sentence_score, keep_ratio)
            if not ranked_candidates:
                break

            selected_candidate = None
            for _, remove_idx, remove_text in ranked_candidates:
                if self.can_remove_span_safely(question, current_spans, remove_idx, question_type):
                    selected_candidate = (remove_idx, remove_text)
                    break
                skipped_by_safety += 1

            if selected_candidate is None:
                break

            remove_idx, remove_text = selected_candidate
            removed_spans.append(remove_text)
            current_spans.pop(remove_idx)

        compressed = self.cleanup_sentence(" ".join(span.text for span in current_spans), original)
        dropped_sentence = self.should_drop_sentence_after_prune(question, current_spans, question_type)
        if dropped_sentence:
            compressed = ""
        if not compressed and current_spans and not dropped_sentence:
            compressed = original

        stats = self._build_sentence_stats(original, compressed, current_spans, removed_spans)
        stats["safety_skipped_count"] = skipped_by_safety
        if not compressed:
            stats["dropped_sentence"] = True
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
        model_weight = self.config.learned_keep_weight if learned_keep_scores else 0.0
        for idx, span in enumerate(spans):
            soft_protected_candidate = False
            if span.protected:
                if (
                    not learned_keep_scores
                    or self.is_hard_protected_span(question, span, question_type)
                    or learned_keep_scores[idx] >= self.config.learned_soft_protected_threshold
                ):
                    continue
                soft_protected_candidate = True

            if span.protected and not soft_protected_candidate:
                continue

            overlap = overlap_scores[idx]
            anchor = anchor_scores[idx]
            reward = reward_scores[idx]
            allow_edge_removal = (
                len(spans) <= 3
                and idx in {0, len(spans) - 1}
                and overlap == 0.0
                and anchor < 0.45
                and (span.kind in {"tail", "source_attribution", "background_lead"} or self.is_temporal_background_span(span.text))
            )
            if span.kind in {"source_attribution", "background_lead"}:
                allow_edge_removal = True

            if len(spans) <= 3 and idx in {0, len(spans) - 1} and not allow_edge_removal:
                continue

            if anchorful_count <= 1 and (overlap > 0.0 or anchor >= 0.45) and not soft_protected_candidate:
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
            if overlap == 0.0 and anchor < 0.45:
                importance -= self.config.mismatch_penalty
            if self.is_temporal_background_span(span.text) and question_type not in {"numeric", "factoid"}:
                importance -= 0.10
            if span.kind == "parenthetical":
                importance -= 0.18
            elif span.kind == "source_attribution":
                importance -= 0.22
            elif span.kind == "background_lead":
                importance -= 0.16
            elif span.kind == "example":
                importance -= 0.10
            elif span.kind == "tail":
                importance -= 0.05
            if self.is_low_value_filler(span.text):
                importance -= self.config.filler_penalty

            candidates.append((float(importance), idx, span.text))

        candidates.sort(key=lambda item: item[0])
        return candidates

    def is_hard_protected_span(
        self,
        question: str,
        span: SpanUnit,
        question_type: str,
    ) -> bool:
        text = span.text
        lowered = text.lower()
        overlap = query_overlap_score(question, text)
        if overlap >= 0.35:
            return True
        if contains_any(lowered, NEGATION_HINTS) or contains_any(lowered, COMPARISON_HINTS):
            return True
        if re.search(r"\d", text) and (question_type in {"numeric", "factoid"} or re.search(r"\d", question)):
            return True
        if question_type == "cause" and contains_any(lowered, ("include", "includes", "because", "therefore", "due to")):
            return True
        if text.strip().endswith(":"):
            return True
        return False

    def can_remove_span_safely(
        self,
        question: str,
        spans: Sequence[SpanUnit],
        remove_idx: int,
        question_type: str,
    ) -> bool:
        if len(spans) <= 1 or remove_idx < 0 or remove_idx >= len(spans):
            return False

        span = spans[remove_idx]
        if self.is_hard_protected_span(question, span, question_type):
            return False

        remaining = [candidate for idx, candidate in enumerate(spans) if idx != remove_idx]
        if not remaining:
            return False

        before_question_terms = self.question_terms_present_in_spans(question, spans)
        if before_question_terms:
            after_question_terms = self.question_terms_present_in_spans(question, remaining)
            recall = len(before_question_terms & after_question_terms) / max(len(before_question_terms), 1)
            if recall < self.config.min_question_term_recall_after_prune:
                return False

        before_anchor_count = self.semantic_anchor_count(question, spans, question_type)
        after_anchor_count = self.semantic_anchor_count(question, remaining, question_type)
        if before_anchor_count > 0 and after_anchor_count == 0:
            return False

        if self.requires_evidence_list_floor(question, spans, question_type):
            before_items = self.evidence_item_count(question, spans, question_type)
            after_items = self.evidence_item_count(question, remaining, question_type)
            required_items = self.required_evidence_item_count(question, before_items)
            if before_items >= required_items and after_items < required_items:
                return False

        if question_type in {"definition", "numeric", "factoid"}:
            overlap = query_overlap_score(question, span.text)
            anchor = task_anchor_score(question, span.text, question_type)
            if overlap > 0.0 or anchor >= 0.45:
                return False

        if span.protected:
            learned_scores = self.predict_learned_keep_scores(
                question=question,
                spans=spans,
                question_type=question_type,
                sentence_score=0.5,
                keep_ratio=self.config.target_keep_ratio,
                attention_scores=[0.0 for _ in spans],
                dac_scores=[0.0 for _ in spans],
            )
            if learned_scores and learned_scores[remove_idx] >= self.config.learned_soft_protected_threshold:
                return False

        return True

    def question_terms_present_in_spans(self, question: str, spans: Sequence[SpanUnit]) -> set[str]:
        question_terms = set(tokenize_query_terms(question))
        if not question_terms:
            return set()
        span_terms = set()
        for span in spans:
            span_terms.update(tokenize_query_terms(span.text))
        return question_terms & span_terms

    def semantic_anchor_count(
        self,
        question: str,
        spans: Sequence[SpanUnit],
        question_type: str,
    ) -> int:
        count = 0
        for span in spans:
            lowered = span.text.lower()
            overlap = query_overlap_score(question, span.text)
            anchor = task_anchor_score(question, span.text, question_type)
            if overlap > 0.0 or anchor >= 0.45:
                count += 1
            elif question_type == "cause" and contains_any(lowered, EXPLANATORY_EVIDENCE_HINTS + CAUSE_HINTS):
                count += 1
            elif question_type == "procedure" and contains_any(lowered, PROCEDURE_HINTS):
                count += 1
        return count

    def requires_evidence_list_floor(
        self,
        question: str,
        spans: Sequence[SpanUnit],
        question_type: str,
    ) -> bool:
        if question_type != "cause":
            return False
        return any(
            contains_any(span.text.lower(), ("include", "includes", "including", "such as", "for example"))
            for span in spans
        )

    def required_evidence_item_count(self, question: str, available_items: int) -> int:
        required = self.config.min_evidence_list_items
        if question_seeks_outcome_examples(question):
            required = max(required, 3)
        return min(required, max(available_items, 0))

    def evidence_item_count(
        self,
        question: str,
        spans: Sequence[SpanUnit],
        question_type: str,
    ) -> int:
        count = 0
        for span in spans:
            lowered = span.text.lower()
            if self.is_structural_lead_span(span.text) and not contains_any(lowered, EXPLANATORY_EVIDENCE_HINTS):
                continue
            if query_overlap_score(question, span.text) > 0.0:
                count += 1
            elif task_anchor_score(question, span.text, question_type) >= 0.45:
                count += 1
            elif contains_any(lowered, EXPLANATORY_EVIDENCE_HINTS + OUTCOME_HINTS):
                count += 1
            elif len(tokenize_query_terms(span.text)) >= 2 and span.kind != "source_attribution":
                count += 1
        return count

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

    def refine_long_spans(
        self,
        question: str,
        spans: Sequence[SpanUnit],
        question_type: str,
    ) -> List[SpanUnit]:
        refined: List[SpanUnit] = []
        for span in spans:
            source_split = self.split_source_attribution_span(question, span, question_type)
            for source_span in source_split:
                clause_split = self.split_relative_or_contrast_clause(question, source_span, question_type)
                for clause_span in clause_split:
                    refined.extend(self.split_coordination_span(question, clause_span, question_type))
        return refined

    def split_source_attribution_span(
        self,
        question: str,
        span: SpanUnit,
        question_type: str,
    ) -> List[SpanUnit]:
        if self._count_tokens(span.text) < 14:
            return [span]
        pattern = re.compile(
            r"\b(reports?|reported|states?|stated|finds?|found|shows?|showed|suggests?|suggested|notes?|noted|argues?|argued)\s+that\b",
            flags=re.IGNORECASE,
        )
        match = pattern.search(span.text)
        if not match or match.end() >= len(span.text) - 8:
            return [span]

        lead_text = span.text[: match.end()].strip()
        evidence_text = span.text[match.end() :].strip()
        if self._count_tokens(evidence_text) < 6:
            return [span]

        lead = self.build_span(question, lead_text, span.start, span.start + match.end(), question_type, kind_override="source_attribution")
        evidence = self.build_span(question, evidence_text, span.start + match.end(), span.end, question_type)
        return [lead, evidence]

    def split_relative_or_contrast_clause(
        self,
        question: str,
        span: SpanUnit,
        question_type: str,
    ) -> List[SpanUnit]:
        if self._count_tokens(span.text) < 14:
            return [span]

        match = re.search(
            r",\s+(which|who|where|while|whereas|although|though|but|however)\b",
            span.text,
            flags=re.IGNORECASE,
        )
        if not match:
            return [span]

        lead_text = span.text[: match.start()].strip()
        tail_text = span.text[match.start() + 1 :].strip()
        if self._count_tokens(lead_text) < 5 or self._count_tokens(tail_text) < 5:
            return [span]

        lead = self.build_span(question, lead_text, span.start, span.start + match.start(), question_type)
        tail = self.build_span(question, tail_text, span.start + match.start() + 1, span.end, question_type, kind_override="tail")
        return [lead, tail]

    def split_coordination_span(
        self,
        question: str,
        span: SpanUnit,
        question_type: str,
    ) -> List[SpanUnit]:
        if self._count_tokens(span.text) < 11:
            return [span]

        lowered = span.text.lower()
        connector_count = len(re.findall(r"\b(?:and|or)\b", lowered))
        list_like = contains_any(lowered, ("include", "includes", "including", "such as", "consist of"))
        if connector_count < 2 and not list_like:
            return [span]

        connectors = list(re.finditer(r"\s+(and|or)\s+", span.text, flags=re.IGNORECASE))
        for match in reversed(connectors):
            lead_text = span.text[: match.start()].strip()
            tail_text = span.text[match.start() :].strip()
            if self._count_tokens(lead_text) < 6 or self._count_tokens(tail_text) < 3:
                continue
            if len(tokenize_query_terms(tail_text)) < 2 and not re.search(r"\d", tail_text):
                continue
            lead = self.build_span(question, lead_text, span.start, span.start + match.start(), question_type)
            tail = self.build_span(question, tail_text, span.start + match.start(), span.end, question_type, kind_override="tail")
            return [lead, tail]

        return [span]

    def apply_evidence_list_floor(
        self,
        question: str,
        spans: Sequence[SpanUnit],
        question_type: str,
    ) -> List[SpanUnit]:
        if question_type != "cause" or len(spans) < 3:
            return list(spans)

        lead_idx = None
        for idx, span in enumerate(spans):
            lowered = span.text.lower()
            if contains_any(lowered, ("include", "includes", "such as")) and contains_any(lowered, OUTCOME_HINTS + EXPLANATORY_EVIDENCE_HINTS):
                lead_idx = idx
                break
        if lead_idx is None:
            return list(spans)

        out = list(spans)
        evidence_indices = [
            idx
            for idx in range(lead_idx + 1, len(out))
            if len(tokenize_query_terms(out[idx].text)) >= 2 or contains_any(out[idx].text.lower(), EXPLANATORY_EVIDENCE_HINTS)
        ]
        keep_count = self.config.min_evidence_list_items
        if question_seeks_outcome_examples(question):
            keep_count = max(keep_count, 3)
        extra_keep_count = max(0, keep_count - 1)

        out[lead_idx].protected = True
        for rank, idx in enumerate(evidence_indices):
            span = out[idx]
            has_hard_anchor = (
                query_overlap_score(question, span.text) > 0.0
                or re.search(r"\d", span.text)
                or contains_any(span.text.lower(), NEGATION_HINTS + COMPARISON_HINTS)
            )
            span.protected = bool(rank < extra_keep_count or has_hard_anchor)
        return out

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

        for match in re.finditer(r"\d+(?:\.\d+)?%?", sentence):
            for pos in range(match.start(), match.end()):
                mask[pos] = True

        for span in spans:
            if span.protected and len(span.text) <= 24:
                for pos in range(span.start, min(span.end, len(mask))):
                    mask[pos] = True

        return mask

    def compute_dac_span_scores(
        self,
        question: str,
        spans: Sequence[SpanUnit],
    ) -> List[float]:
        sentence = " ".join(span.text for span in spans)
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
        kind_override: str | None = None,
    ) -> SpanUnit:
        stripped = text.strip()
        lowered = stripped.lower()
        outcome_question = question_seeks_outcome_examples(question)

        kind = kind_override or "content"
        if stripped.startswith(("(", "[", "{")) and stripped.endswith((")", "]", "}")):
            kind = "parenthetical"
        elif starts_with_example_marker(stripped) or contains_any(lowered, EXAMPLE_HINTS):
            kind = "example"
        elif start > 0 and len(tokenize_query_terms(stripped)) <= 4:
            kind = "tail"

        overlap = query_overlap_score(question, stripped)
        anchor = task_anchor_score(question, stripped, question_type)
        protected = overlap > 0.0 or anchor >= 0.45
        if outcome_question and kind == "example":
            protected = True
        if outcome_question and contains_any(stripped, OUTCOME_HINTS):
            protected = True
        if question_type == "cause" and contains_any(lowered, EXPLANATORY_EVIDENCE_HINTS):
            protected = protected or query_overlap_score(question, stripped) > 0.0
        if contains_any(lowered, NEGATION_HINTS) or contains_any(lowered, COMPARISON_HINTS):
            protected = True
        if stripped.endswith(":") and len(stripped) > 12:
            protected = True
        if re.search(r"\b(i|ii|iii|iv|v|first|second|third)\b", lowered):
            protected = True
        if kind == "parenthetical" and overlap == 0.0 and anchor < 0.45:
            protected = False
        if kind == "source_attribution" and question_type not in {"factoid"}:
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
        sentence = " ".join(span.text for span in spans)
        if not sentence or not getattr(self.tokenizer, "is_fast", False):
            return [0.0 for _ in spans]

        try:
            encoded_question = self.sentence_encoder.format_query(question) if hasattr(self.sentence_encoder, "format_query") else question
            batch = self.tokenizer(
                encoded_question,
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
        lowered = text.lower().strip()
        if len(tokenize_query_terms(text)) <= 4:
            return False
        filler_markers = (
            "as discussed above",
            "in general",
            "in other words",
            "more broadly",
            "overall",
            "taken together",
            "to summarize",
        )
        return any(marker in lowered for marker in filler_markers)

    def is_temporal_background_span(self, text: str) -> bool:
        lowered = text.lower()
        if re.search(r"\b(19|20)\d{2}\b", text):
            return True
        markers = ("historically", "in recent years", "over time", "previously", "since then")
        return any(marker in lowered for marker in markers)

    def is_structural_lead_span(self, text: str) -> bool:
        stripped = text.strip().lower()
        if stripped.endswith(":"):
            return True
        lead_markers = ("include", "includes", "consist of", "the following", "there are")
        return any(marker in stripped for marker in lead_markers)

    def should_drop_sentence_after_prune(
        self,
        question: str,
        kept_spans: Sequence[SpanUnit],
        question_type: str,
    ) -> bool:
        if len(kept_spans) != 1:
            return False
        text = kept_spans[0].text.strip()
        if not self.is_structural_lead_span(text):
            return False
        if query_overlap_score(question, text) > 0.0:
            return False
        if task_anchor_score(question, text, question_type) >= 0.45:
            return False
        return True

    def cleanup_sentence(self, text: str, original: str) -> str:
        cleaned = re.sub(r"\s+", " ", text).strip()
        cleaned = re.sub(r"\s+([,.;:!?])", r"\1", cleaned)
        cleaned = re.sub(r"([([{])\s+", r"\1", cleaned)
        cleaned = re.sub(r"\s+([)\]}])", r"\1", cleaned)
        cleaned = re.sub(r"([,;:]){2,}", lambda match: match.group(0)[0], cleaned)
        cleaned = cleaned.strip(" ,;:")

        if original and original[-1] in ".!?" and (not cleaned or cleaned[-1] not in ".!?"):
            cleaned = cleaned + original[-1]
        if cleaned and original[:1].isupper() and cleaned[:1].islower():
            cleaned = cleaned[:1].upper() + cleaned[1:]
        return cleaned

    def _count_tokens(self, text: str) -> int:
        if not text.strip():
            return 0
        try:
            return max(1, len(self.tokenizer.tokenize(text)))
        except Exception:
            return max(1, len(TOKEN_RE.findall(text)))

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
            "protected_kept_spans": [span.text for span in kept_spans if span.protected],
            "kept_span_kinds": [span.kind for span in kept_spans],
            "original_tokens": self._count_tokens(original),
            "compressed_tokens": self._count_tokens(compressed),
        }

