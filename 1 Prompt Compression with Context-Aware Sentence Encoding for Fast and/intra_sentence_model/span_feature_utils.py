from __future__ import annotations

import re
from typing import Dict, List, Sequence

import numpy as np


QUESTION_STOPWORDS = {
    "什么",
    "为何",
    "为什么",
    "如何",
    "怎么",
    "哪些",
    "哪个",
    "多少",
    "是否",
    "关于",
    "请问",
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
    "因为",
    "原因",
    "导致",
    "因此",
    "所以",
    "由于",
    "机制",
    "使得",
    "because",
    "due to",
    "therefore",
)
COMPARISON_HINTS = (
    "比较",
    "区别",
    "不同",
    "相比",
    "优于",
    "劣于",
    "更",
    "less",
    "more",
    "than",
    "versus",
)
PROCEDURE_HINTS = (
    "步骤",
    "流程",
    "首先",
    "然后",
    "接着",
    "最后",
    "first",
    "then",
    "next",
    "finally",
)
NUMERIC_HINTS = ("多少", "几", "数值", "比例", "参数", "percent", "rate")
FACTOID_HINTS = (
    "谁",
    "何时",
    "哪里",
    "哪一",
    "时间",
    "地点",
    "when",
    "where",
    "who",
)
NEGATION_HINTS = ("不", "没", "无", "并非", "不是", "cannot", "not", "without", "never")
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
    "query_overlap",
    "anchor_score",
    "task_reward",
    "answer_overlap",
    "answer_drop",
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


def normalize_text(text: str) -> str:
    text = text.strip().lower()
    text = re.sub(r"\s+", "", text)
    return text


def tokenize_query_terms(text: str) -> List[str]:
    pieces = re.findall(r"[A-Za-z0-9\-]+|[\u4e00-\u9fff]{2,8}", text.lower())
    return [piece for piece in pieces if piece not in QUESTION_STOPWORDS]


def tokenize_mixed(text: str) -> List[str]:
    text = normalize_text(text)
    zh_chars = re.findall(r"[\u4e00-\u9fff]", text)
    latin_words = re.findall(r"[a-z0-9]+", text)
    tokens = zh_chars + latin_words
    return [token for token in tokens if token and token not in QUESTION_STOPWORDS]


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
    if any(k in q for k in ("为什么", "原因", "导致", "如何影响", "机制")):
        return "cause"
    if any(k in q for k in ("区别", "不同", "对比", "比较", "异同")):
        return "comparison"
    if any(k in q for k in ("如何", "怎么", "步骤", "流程", "实现")):
        return "procedure"
    if any(k in q for k in ("多少", "几", "数值", "比例", "参数")):
        return "numeric"
    if any(k in q for k in ("是什么", "定义", "含义", "概念")):
        return "definition"
    if any(k in q for k in ("谁", "何时", "哪里", "哪一")):
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
        or re.search(r"\d{4}|\d{1,2}[:：]\d{2}", text)
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
    }


def approx_token_length(text: str) -> int:
    tokens = re.findall(r"[A-Za-z0-9]+|[\u4e00-\u9fff]", text)
    return max(len(tokens), 1)


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
    return [float(features[name]) for name in FEATURE_ORDER]
