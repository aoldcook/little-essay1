from __future__ import annotations

import math
from dataclasses import asdict, dataclass
from typing import Dict, List, Optional, Sequence

import numpy as np

from target_ratio_model.budget_features import build_budget_features, detect_question_type, split_sentences, tokenize


DEFAULT_RATIO_BUCKETS = [0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]
FORMULA_NAMES = [
    "entropy_spread",
    "top_gap",
    "length_density",
    "linguistic_complexity",
    "evidence_floor",
    "hybrid_li_dac",
]


@dataclass
class FormulaBudgetResult:
    formula_name: str
    ratio: float
    raw_ratio: float
    features: Dict[str, float]
    explanation: str

    def to_dict(self) -> Dict[str, object]:
        return asdict(self)


def _clip(value: float, low: float = 0.2, high: float = 1.0) -> float:
    return float(min(high, max(low, value)))


def _nearest_bucket(value: float, buckets: Sequence[float]) -> float:
    return float(min(buckets, key=lambda bucket: abs(float(bucket) - value)))


def _minmax(values: Sequence[float]) -> np.ndarray:
    arr = np.asarray(list(values), dtype=float)
    if arr.size == 0:
        return np.array([0.0], dtype=float)
    lo = float(arr.min())
    hi = float(arr.max())
    if math.isclose(lo, hi):
        return np.full_like(arr, 0.5, dtype=float)
    return (arr - lo) / (hi - lo + 1e-9)


def _linguistic_summary(linguistic_features: Optional[Sequence[dict]]) -> Dict[str, float]:
    if not linguistic_features:
        return {
            "ling_avg": 0.0,
            "ling_max": 0.0,
            "disc_max": 0.0,
            "coref_max": 0.0,
            "pa_max": 0.0,
            "dep_max": 0.0,
            "ling_high_ratio": 0.0,
        }
    finals = [float(row.get("final_score", 0.0)) for row in linguistic_features]
    return {
        "ling_avg": float(np.mean(finals)),
        "ling_max": float(np.max(finals)),
        "disc_max": float(max(float(row.get("discourse_score", 0.0)) for row in linguistic_features)),
        "coref_max": float(max(float(row.get("coref_score", 0.0)) for row in linguistic_features)),
        "pa_max": float(max(float(row.get("predicate_argument_score", 0.0)) for row in linguistic_features)),
        "dep_max": float(max(float(row.get("dependency_score", 0.0)) for row in linguistic_features)),
        "ling_high_ratio": float(np.mean([score >= 0.60 for score in finals])),
    }


def build_formula_features(
    question: str,
    context: str,
    similarities: Sequence[float],
    linguistic_features: Optional[Sequence[dict]] = None,
) -> Dict[str, float]:
    base = build_budget_features(question, context, list(similarities))
    sentences = split_sentences(context)
    lengths = [max(1, len(tokenize(sentence))) for sentence in sentences] or [1]
    norm_sims = _minmax(similarities)
    sorted_norm = sorted((float(x) for x in norm_sims), reverse=True)
    n = max(len(sorted_norm), 1)

    q_type = detect_question_type(question)
    q_complexity = {
        "cause": 0.18,
        "comparison": 0.18,
        "procedure": 0.16,
        "numeric": 0.12,
        "definition": 0.06,
        "factoid": 0.03,
        "other": 0.10,
    }.get(q_type, 0.10)
    top_gap = sorted_norm[0] - sorted_norm[1] if n >= 2 else sorted_norm[0]
    top3_mass = sum(sorted_norm[: min(3, n)]) / max(sum(sorted_norm), 1e-9)
    mid_relevance_ratio = float(np.mean([score >= 0.45 for score in norm_sims]))
    high_relevance_ratio = float(np.mean([score >= 0.70 for score in norm_sims]))

    sims_for_entropy = np.asarray(sorted_norm, dtype=float)
    exp_scores = np.exp(sims_for_entropy - sims_for_entropy.max())
    probs = exp_scores / max(float(exp_scores.sum()), 1e-9)
    entropy = float(-(probs * np.log(probs + 1e-12)).sum())
    entropy_norm = float(entropy / math.log(n + 1e-12)) if n > 1 else 0.0

    out = dict(base)
    out.update(
        {
            "q_complexity": q_complexity,
            "qtype_code": float(["definition", "cause", "comparison", "procedure", "numeric", "factoid", "other"].index(q_type))
            if q_type in {"definition", "cause", "comparison", "procedure", "numeric", "factoid", "other"}
            else 6.0,
            "norm_entropy": entropy_norm,
            "norm_top_gap": float(top_gap),
            "norm_top3_mass": float(top3_mass),
            "norm_mid_relevance_ratio": mid_relevance_ratio,
            "norm_high_relevance_ratio": high_relevance_ratio,
            "context_size": _clip(math.log1p(max(n, 1)) / math.log(16), 0.0, 1.0),
            "avg_len_norm": _clip(float(np.mean(lengths)) / 28.0, 0.0, 1.0),
            "max_len_norm": _clip(float(np.max(lengths)) / 40.0, 0.0, 1.0),
            "len_std_norm": _clip(float(np.std(lengths)) / 18.0, 0.0, 1.0),
        }
    )
    out.update(_linguistic_summary(linguistic_features))
    return out


def entropy_spread_formula(features: Dict[str, float]) -> float:
    return _clip(
        0.28
        + 0.34 * features["norm_entropy"]
        + 0.13 * features["norm_mid_relevance_ratio"]
        + features["q_complexity"]
        + 0.07 * features["context_size"]
        + 0.05 * features["len_std_norm"]
        - 0.10 * features["norm_top_gap"]
    )


def top_gap_formula(features: Dict[str, float]) -> float:
    return _clip(
        0.56
        + 0.18 * features["norm_entropy"]
        + 0.10 * features["norm_mid_relevance_ratio"]
        + features["q_complexity"]
        - 0.26 * features["norm_top_gap"]
        - 0.12 * features["norm_top3_mass"]
    )


def length_density_formula(features: Dict[str, float]) -> float:
    return _clip(
        0.30
        + 0.16 * features["context_size"]
        + 0.15 * features["avg_len_norm"]
        + 0.12 * features["len_std_norm"]
        + 0.16 * features["ling_avg"]
        + 0.08 * features["question_entity_count"]
        + features["q_complexity"]
    )


def linguistic_complexity_formula(features: Dict[str, float]) -> float:
    return _clip(
        0.24
        + 0.24 * features["ling_avg"]
        + 0.12 * features["disc_max"]
        + 0.12 * features["coref_max"]
        + 0.14 * features["pa_max"]
        + 0.10 * features["dep_max"]
        + 0.10 * features["ling_high_ratio"]
        + features["q_complexity"]
        - 0.08 * features["norm_top_gap"]
    )


def evidence_floor_formula(
    question: str,
    context: str,
    similarities: Sequence[float],
    features: Dict[str, float],
) -> float:
    sentences = split_sentences(context)
    if not sentences:
        return 0.4

    lengths = [max(1, len(tokenize(sentence))) for sentence in sentences]
    total_len = max(sum(lengths), 1)
    norm_sims = _minmax(similarities)
    ranked = sorted(range(len(sentences)), key=lambda idx: float(norm_sims[idx]), reverse=True)

    q_type = detect_question_type(question)
    min_evidence = {
        "cause": 2,
        "comparison": 2,
        "procedure": 3,
        "numeric": 2,
        "definition": 1,
        "factoid": 1,
        "other": 2,
    }.get(q_type, 2)
    best_score = float(max(norm_sims)) if len(norm_sims) else 0.0
    near_top_count = int(sum(float(score) >= best_score - 0.22 for score in norm_sims))
    entropy_count = int(math.ceil(len(sentences) * (0.20 + 0.35 * features["norm_entropy"])))
    k = max(min_evidence, min(max(near_top_count, entropy_count), len(sentences)))
    k = min(max(k, 1), len(sentences))

    kept_len = sum(lengths[idx] for idx in ranked[:k])
    margin = 0.03 + 0.04 * features["q_complexity"] + 0.04 * features["ling_high_ratio"]
    if features["norm_top_gap"] < 0.15:
        margin += 0.04
    return _clip(kept_len / total_len + margin)


def hybrid_li_dac_formula(
    question: str,
    context: str,
    similarities: Sequence[float],
    features: Dict[str, float],
) -> float:
    evidence_floor = evidence_floor_formula(question, context, similarities, features)
    return _clip(
        0.46 * entropy_spread_formula(features)
        + 0.24 * linguistic_complexity_formula(features)
        + 0.18 * length_density_formula(features)
        + 0.12 * evidence_floor
    )


def predict_formula_ratio(
    formula_name: str,
    question: str,
    context: str,
    similarities: Sequence[float],
    linguistic_features: Optional[Sequence[dict]] = None,
    ratio_buckets: Sequence[float] = DEFAULT_RATIO_BUCKETS,
) -> FormulaBudgetResult:
    if formula_name not in FORMULA_NAMES:
        raise ValueError(f"unknown formula_name={formula_name}; choose one of {FORMULA_NAMES}")

    features = build_formula_features(question, context, similarities, linguistic_features)
    if formula_name == "entropy_spread":
        raw = entropy_spread_formula(features)
        explanation = "raise ratio when relevance is spread across many sentences"
    elif formula_name == "top_gap":
        raw = top_gap_formula(features)
        explanation = "lower ratio when the best sentence is clearly separated"
    elif formula_name == "length_density":
        raw = length_density_formula(features)
        explanation = "raise ratio for long, uneven, dense contexts"
    elif formula_name == "linguistic_complexity":
        raw = linguistic_complexity_formula(features)
        explanation = "raise ratio when linguistic structure is dense or dependency-heavy"
    elif formula_name == "evidence_floor":
        raw = evidence_floor_formula(question, context, similarities, features)
        explanation = "estimate the minimum token budget needed to keep the top evidence set"
    else:
        raw = hybrid_li_dac_formula(question, context, similarities, features)
        explanation = "combine evidence floor, spread, length pressure, and LI-DAC linguistic complexity"

    return FormulaBudgetResult(
        formula_name=formula_name,
        raw_ratio=float(raw),
        ratio=_nearest_bucket(raw, ratio_buckets),
        features=features,
        explanation=explanation,
    )
