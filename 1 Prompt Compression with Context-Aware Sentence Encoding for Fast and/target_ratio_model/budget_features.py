import math
import re
from typing import Dict, List

import numpy as np


QUESTION_TYPES = ["definition", "cause", "comparison", "procedure", "numeric", "factoid", "other"]


def split_sentences(text: str) -> List[str]:
    sentences = re.split(r'(?<=[。！？.!?])\s*', text.strip())
    return [s.strip() for s in sentences if s.strip()]


ENTITY_PATTERN = re.compile(r"[A-Za-z][A-Za-z0-9\-]{1,}|[\u4e00-\u9fff]{2,8}")


def detect_question_type(question: str) -> str:
    q = question.strip()
    if any(k in q for k in ["为什么", "原因", "导致", "如何影响", "机制"]):
        return "cause"
    if any(k in q for k in ["区别", "不同", "对比", "比较", "异同"]):
        return "comparison"
    if any(k in q for k in ["如何", "怎么", "步骤", "流程", "实现"]):
        return "procedure"
    if any(k in q for k in ["多少", "几", "数值", "比例", "参数"]):
        return "numeric"
    if any(k in q for k in ["是什么", "定义", "含义", "概念"]):
        return "definition"
    if any(k in q for k in ["谁", "何时", "哪里", "哪一"]):
        return "factoid"
    return "other"


def count_entities(question: str) -> int:
    matches = ENTITY_PATTERN.findall(question)
    filtered = [m for m in matches if len(m.strip()) >= 2]
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
    sentence_lens = np.array([len(s) for s in sentences], dtype=float) if sentences else np.array([0.0])

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
        "question_char_len": float(len(question)),
        "context_char_len": float(len(context)),
        "avg_sentence_char_len": float(np.mean(sentence_lens)),
        "max_sentence_char_len": float(np.max(sentence_lens)),
        "sentence_len_std": float(np.std(sentence_lens)),
        "question_entity_count": float(count_entities(question)),
        "is_multi_hop_like": float(any(k in question for k in ["为什么", "如何", "比较", "区别", "异同", "影响", "关系"])),
    }

    for qt in QUESTION_TYPES:
        features[f"qtype_{qt}"] = 1.0 if q_type == qt else 0.0

    return features


FEATURE_ORDER = list(build_budget_features("什么是CPC？", "CPC是一种压缩方法。它按句子筛选上下文。", [0.82, 0.64]).keys())


def features_to_vector(features: Dict[str, float]) -> np.ndarray:
    return np.array([float(features[name]) for name in FEATURE_ORDER], dtype=np.float32)
