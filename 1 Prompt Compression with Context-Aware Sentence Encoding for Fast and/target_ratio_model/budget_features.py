import math
import re
from typing import Dict, List

import numpy as np


QUESTION_TYPES = ["definition", "cause", "comparison", "procedure", "numeric", "factoid", "other"]
STOPWORDS = {
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
TOKEN_RE = re.compile(r"[A-Za-z][A-Za-z0-9]*(?:-[A-Za-z0-9]+)*|\d+(?:\.\d+)?%?")
ENTITY_PATTERN = re.compile(r"\b[A-Z][A-Za-z0-9-]{1,}\b|\d+(?:\.\d+)?%?")


def split_sentences(text: str) -> List[str]:
    stripped = text.strip()
    if not stripped:
        return []
    sentences: List[str] = []
    current: List[str] = []
    length = len(stripped)
    abbreviations = {"dr", "mr", "mrs", "ms", "prof", "inc", "ltd", "fig", "e.g", "i.e", "vs", "u.s", "u.k"}
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
            if prefix in abbreviations or (next_char and next_char.islower()):
                continue
        sentence = "".join(current).strip()
        if sentence:
            sentences.append(sentence)
        current = []
    tail = "".join(current).strip()
    if tail:
        sentences.append(tail)
    return sentences


def normalize_term(term: str) -> str:
    term = term.lower().strip("'\"")
    if len(term) > 4 and term.endswith("ies"):
        return term[:-3] + "y"
    for suffix in ("ing", "ed", "es", "s"):
        if len(term) > len(suffix) + 3 and term.endswith(suffix):
            return term[: -len(suffix)]
    return term


def tokenize(text: str) -> List[str]:
    return [tok for tok in (normalize_term(t) for t in TOKEN_RE.findall(text)) if tok and tok not in STOPWORDS]


def detect_question_type(question: str) -> str:
    q = question.lower().strip()
    if any(k in q for k in ["why", "cause", "reason", "lead to", "impact", "effect", "mechanism", "risk"]):
        return "cause"
    if any(k in q for k in ["compare", "compared", "difference", "different", "versus", " vs ", "better", "worse"]):
        return "comparison"
    if any(k in q for k in ["how to", "steps", "process", "procedure", "workflow", "method", "implement"]):
        return "procedure"
    if any(k in q for k in ["how many", "how much", "number", "percent", "percentage", "rate", "ratio", "threshold", "value"]):
        return "numeric"
    if re.search(r"\b(what is|what are|define|definition|meaning of)\b", q):
        return "definition"
    if re.search(r"\b(who|when|where|which|name)\b", q):
        return "factoid"
    return "other"


def count_entities(question: str) -> int:
    matches = ENTITY_PATTERN.findall(question)
    filtered = [match for match in matches if len(match.strip()) >= 2]
    return len(set(filtered))


def softmax(x: np.ndarray) -> np.ndarray:
    x = np.asarray(x, dtype=float)
    x = x - np.max(x)
    ex = np.exp(x)
    denom = ex.sum()
    if denom <= 0:
        return np.full_like(ex, 1.0 / len(ex))
    return ex / denom


def _safe_ratio(num: float, den: float) -> float:
    return float(num / den) if den else 0.0


def build_budget_features(question: str, context: str, similarities: List[float]) -> Dict[str, float]:
    sims = np.asarray(similarities, dtype=float)
    if sims.size == 0:
        sims = np.array([0.0], dtype=float)

    sims = np.sort(sims)[::-1]
    n = int(sims.size)
    probs = softmax(sims)
    entropy = float(-(probs * np.log(probs + 1e-12)).sum())
    entropy_norm = float(entropy / math.log(n + 1e-12)) if n > 1 else 0.0

    k3 = min(3, n)
    k20 = max(1, int(round(n * 0.2)))
    k50 = max(1, int(round(n * 0.5)))

    sentences = split_sentences(context)
    sentence_lens = np.array([len(tokenize(sentence)) for sentence in sentences], dtype=float) if sentences else np.array([0.0])
    q_tokens = tokenize(question)
    context_tokens = tokenize(context)

    q_type = detect_question_type(question)
    features: Dict[str, float] = {
        "num_sentences": float(n),
        "sim_max": float(sims[0]),
        "sim_min": float(sims[-1]),
        "sim_mean": float(np.mean(sims)),
        "sim_std": float(np.std(sims)),
        "sim_range": float(sims[0] - sims[-1]),
        "top1_gap": float(sims[0] - sims[1]) if n >= 2 else 0.0,
        "top2_gap": float(sims[1] - sims[2]) if n >= 3 else 0.0,
        "top3_mass": _safe_ratio(float(sims[:k3].sum()), float(sims.sum()) + 1e-9),
        "top20_mass": _safe_ratio(float(sims[:k20].sum()), float(sims.sum()) + 1e-9),
        "top50_mass": _safe_ratio(float(sims[:k50].sum()), float(sims.sum()) + 1e-9),
        "high_relevance_ratio": float(np.mean(sims >= 0.7)),
        "mid_relevance_ratio": float(np.mean(sims >= 0.5)),
        "entropy_norm": entropy_norm,
        "front_avg_drop": float(np.mean(np.abs(np.diff(sims[: min(5, n)])))) if n >= 2 else 0.0,
        "question_token_len": float(len(q_tokens)),
        "context_token_len": float(len(context_tokens)),
        "avg_sentence_token_len": float(np.mean(sentence_lens)),
        "max_sentence_token_len": float(np.max(sentence_lens)),
        "sentence_len_std": float(np.std(sentence_lens)),
        "question_entity_count": float(count_entities(question)),
        "is_multi_hop_like": float(any(k in question.lower() for k in ["why", "how", "compare", "difference", "impact", "relationship"])),
    }

    for qt in QUESTION_TYPES:
        features[f"qtype_{qt}"] = 1.0 if q_type == qt else 0.0

    return features


FEATURE_ORDER = list(
    build_budget_features(
        "Why does multi-hop question answering need bridge evidence?",
        "A multi-hop question links several entities. Bridge sentences connect those entities and keep the reasoning chain intact.",
        [0.82, 0.64],
    ).keys()
)


def features_to_vector(features: Dict[str, float]) -> np.ndarray:
    return np.array([float(features[name]) for name in FEATURE_ORDER], dtype=np.float32)
