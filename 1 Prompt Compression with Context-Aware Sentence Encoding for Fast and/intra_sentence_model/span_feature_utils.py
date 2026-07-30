from __future__ import annotations

import re
from typing import Dict, List, Sequence

import numpy as np


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
    "due to",
    "effect",
    "impact",
    "lead to",
    "mechanism",
    "reason",
    "result",
    "risk",
    "therefore",
    "trigger",
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
PROCEDURE_HINTS = ("after", "before", "finally", "first", "method", "next", "process", "second", "step", "then")
NUMERIC_HINTS = ("amount", "average", "count", "how many", "how much", "number", "percent", "rate", "ratio", "threshold", "value")
FACTOID_HINTS = ("city", "country", "date", "location", "name", "person", "place", "time", "when", "where", "which", "who")
NEGATION_HINTS = ("cannot", "except", "lack", "lacks", "neither", "never", "no", "not", "without")
COMPLEX_QTYPES = {"cause", "comparison", "procedure"}
QUESTION_TYPES = [
    "cause",
    "comparison",
    "procedure",
    "numeric",
    "definition",
    "factoid",
    "other",
]
# Bumped whenever FEATURE_ORDER changes, so a stale checkpoint cannot be loaded
# against a mismatched feature vector. v2 removed the oracle features.
FEATURE_SCHEMA_VERSION = 2

# Features that are derived from the GOLD ANSWER. They are available while
# generating pseudo-labels but are structurally unavailable at inference time,
# where task_aware_compression.py could only ever pass 0.0. Training on them and
# then zeroing them was a train/test distribution mismatch that made the learned
# weights meaningless at deployment (EVAL_VALIDITY_AUDIT.md finding C5).
#
# They are therefore EXCLUDED from the model input. They remain computed and
# recorded in the feature dict for label construction and for analysis, but they
# must never re-enter FEATURE_ORDER.
ORACLE_ONLY_FEATURES = ("answer_overlap", "answer_drop")

FEATURE_ORDER = [
    "sentence_score",
    "keep_ratio",
    "num_spans",
    "span_index_norm",
    "is_first",
    "is_last",
    "relative_start",
    "relative_end",
    "char_len_norm",
    "token_len_norm",
    "kind_content",
    "kind_parenthetical",
    "kind_example",
    "kind_tail",
    "kind_source_attribution",
    "kind_background_lead",
    "query_overlap",
    "anchor_score",
    "task_reward",
    # answer_overlap / answer_drop deliberately absent -- see ORACLE_ONLY_FEATURES.
    "attention_score",
    "dac_score",
    "protected_flag",
    "qtype_cause",
    "qtype_comparison",
    "qtype_procedure",
    "qtype_numeric",
    "qtype_definition",
    "qtype_factoid",
    "qtype_other",
]
TOKEN_RE = re.compile(r"[A-Za-z][A-Za-z0-9]*(?:-[A-Za-z0-9]+)*|\d+(?:\.\d+)?%?")


def normalize_text(text: str) -> str:
    text = text.strip().lower()
    return re.sub(r"\s+", " ", text)


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


def tokenize_query_terms(text: str) -> List[str]:
    pieces = TOKEN_RE.findall(text.lower())
    return [normalize_term(piece) for piece in pieces if normalize_term(piece) not in QUESTION_STOPWORDS]


def tokenize_mixed(text: str) -> List[str]:
    return tokenize_query_terms(text)


def normalize_scores(values: Sequence[float]) -> List[float]:
    arr = np.asarray(list(values), dtype=float)
    if arr.size == 0:
        return []
    if np.isclose(arr.max(), arr.min()):
        return [0.5 for _ in arr]
    norm = (arr - arr.min()) / (arr.max() - arr.min() + 1e-9)
    return norm.tolist()


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

    if q_type == "cause" and contains_any(lowered, CAUSE_HINTS):
        score += 0.45
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
    return float(min(0.55 * overlap + 0.35 * anchor + bonus, 1.0))


def overlap_ratio(a: str, b: str) -> float:
    a_tokens = tokenize_mixed(a)
    b_tokens = tokenize_mixed(b)
    if not a_tokens or not b_tokens:
        return 0.0
    a_counts = {}
    b_counts = {}
    for token in a_tokens:
        a_counts[token] = a_counts.get(token, 0) + 1
    for token in b_tokens:
        b_counts[token] = b_counts.get(token, 0) + 1
    overlap = 0
    for token, count in a_counts.items():
        overlap += min(count, b_counts.get(token, 0))
    return overlap / max(len(a_tokens), 1)


def answer_overlap_score(answer: str, text: str) -> float:
    if not answer.strip() or not text.strip():
        return 0.0
    a_tokens = set(tokenize_mixed(answer))
    t_tokens = set(tokenize_mixed(text))
    if not a_tokens or not t_tokens:
        return 0.0
    return len(a_tokens & t_tokens) / max(len(a_tokens), 1)


def question_type_features(question_type: str) -> Dict[str, float]:
    features = {f"qtype_{name}": 0.0 for name in QUESTION_TYPES}
    normalized = question_type if question_type in QUESTION_TYPES else "other"
    features[f"qtype_{normalized}"] = 1.0
    return features


def kind_features(kind: str) -> Dict[str, float]:
    return {
        "kind_content": 1.0 if kind == "content" else 0.0,
        "kind_parenthetical": 1.0 if kind == "parenthetical" else 0.0,
        "kind_example": 1.0 if kind == "example" else 0.0,
        "kind_tail": 1.0 if kind == "tail" else 0.0,
        "kind_source_attribution": 1.0 if kind == "source_attribution" else 0.0,
        "kind_background_lead": 1.0 if kind == "background_lead" else 0.0,
    }


def approx_token_length(text: str) -> int:
    return max(len(TOKEN_RE.findall(text)), 1)


def build_span_feature_dict(
    question: str,
    span_text: str,
    span_kind: str,
    span_index: int,
    num_spans: int,
    sentence_length: int,
    sentence_token_length: int,
    start: int,
    end: int,
    sentence_score: float,
    keep_ratio: float,
    question_type: str,
    attention_score: float = 0.0,
    dac_score: float = 0.0,
    answer_overlap: float = 0.0,
    answer_drop: float = 0.0,
    protected: bool = False,
) -> Dict[str, float]:
    num_spans = max(num_spans, 1)
    sentence_length = max(sentence_length, 1)
    sentence_token_length = max(sentence_token_length, 1)
    features: Dict[str, float] = {
        "sentence_score": float(sentence_score),
        "keep_ratio": float(keep_ratio),
        "num_spans": float(num_spans),
        "span_index_norm": float(span_index / max(num_spans - 1, 1)),
        "is_first": 1.0 if span_index == 0 else 0.0,
        "is_last": 1.0 if span_index == num_spans - 1 else 0.0,
        "relative_start": float(start / sentence_length),
        "relative_end": float(end / sentence_length),
        "char_len_norm": float(len(span_text) / sentence_length),
        "token_len_norm": float(approx_token_length(span_text) / sentence_token_length),
        "query_overlap": float(query_overlap_score(question, span_text)),
        "anchor_score": float(task_anchor_score(question, span_text, question_type)),
        "task_reward": float(compute_task_reward(question, span_text, question_type)),
        "answer_overlap": float(answer_overlap),
        "answer_drop": float(answer_drop),
        "attention_score": float(attention_score),
        "dac_score": float(dac_score),
        "protected_flag": 1.0 if protected else 0.0,
    }
    features.update(kind_features(span_kind))
    features.update(question_type_features(question_type))
    return features


def features_to_vector(features: Dict[str, float]) -> List[float]:
    """Project a feature dict onto the model input vector.

    Only FEATURE_ORDER entries are included, so oracle-only features present in
    the dict never reach the model.
    """
    return [float(features[name]) for name in FEATURE_ORDER]


def assert_no_oracle_features(feature_order: Sequence[str]) -> None:
    """Guard against an oracle feature being reintroduced into the model input."""
    leaked = [name for name in ORACLE_ONLY_FEATURES if name in feature_order]
    if leaked:
        raise ValueError(
            f"Oracle-derived features {leaked} are in the model input vector. These are "
            "computed from the gold answer and are unavailable at inference time "
            "(they would be zeroed), so training on them is invalid. See "
            "EVAL_VALIDITY_AUDIT.md finding C5."
        )
