"""Downstream QA metrics: real EM / F1 / ROUGE-L, plus evidence recall.

These replace the lexical "answerability" proxy flagged as EVAL_VALIDITY_AUDIT.md
finding C2. EM and F1 follow the standard SQuAD normalisation so numbers are
comparable to published work. ROUGE-L is implemented locally (LCS-based) to
avoid a heavy dependency.

Vocabulary discipline used throughout the project:
  * `em`, `f1`, `rouge_l`      -> genuine downstream reader accuracy
  * `evidence_recall`          -> did the compressed context retain gold evidence
  * `lexical_coverage_*`       -> PROXY diagnostics only, never headline results
"""

from __future__ import annotations

import re
import string
from collections import Counter
from typing import Dict, List, Optional, Sequence

_ARTICLES = re.compile(r"\b(a|an|the)\b", re.UNICODE)
_PUNCT_TABLE = str.maketrans("", "", string.punctuation)
TOKEN_RE = re.compile(r"[A-Za-z][A-Za-z0-9]*(?:-[A-Za-z0-9]+)*|\d+(?:\.\d+)?%?")


def normalize_answer(text: str) -> str:
    """SQuAD normalisation: lowercase, strip punctuation/articles/extra space."""
    text = str(text or "").lower()
    text = text.translate(_PUNCT_TABLE)
    text = _ARTICLES.sub(" ", text)
    return " ".join(text.split())


def answer_tokens(text: str) -> List[str]:
    return normalize_answer(text).split()


def exact_match(prediction: str, ground_truth: str) -> float:
    return float(normalize_answer(prediction) == normalize_answer(ground_truth))


def token_f1(prediction: str, ground_truth: str) -> float:
    pred_tokens = answer_tokens(prediction)
    gold_tokens = answer_tokens(ground_truth)

    # Degenerate case: both empty counts as a match; one empty counts as a miss.
    if not pred_tokens or not gold_tokens:
        return float(pred_tokens == gold_tokens)

    common = Counter(pred_tokens) & Counter(gold_tokens)
    overlap = sum(common.values())
    if overlap == 0:
        return 0.0
    precision = overlap / len(pred_tokens)
    recall = overlap / len(gold_tokens)
    return 2 * precision * recall / (precision + recall)


def _lcs_length(a: Sequence[str], b: Sequence[str]) -> int:
    if not a or not b:
        return 0
    prev = [0] * (len(b) + 1)
    for token_a in a:
        curr = [0]
        for j, token_b in enumerate(b):
            if token_a == token_b:
                curr.append(prev[j] + 1)
            else:
                curr.append(max(prev[j + 1], curr[j]))
        prev = curr
    return prev[-1]


def rouge_l(prediction: str, ground_truth: str, beta: float = 1.2) -> float:
    """F-measure ROUGE-L over normalised tokens."""
    pred_tokens = answer_tokens(prediction)
    gold_tokens = answer_tokens(ground_truth)
    if not pred_tokens or not gold_tokens:
        return float(pred_tokens == gold_tokens)

    lcs = _lcs_length(pred_tokens, gold_tokens)
    if lcs == 0:
        return 0.0
    precision = lcs / len(pred_tokens)
    recall = lcs / len(gold_tokens)
    beta_sq = beta * beta
    return ((1 + beta_sq) * precision * recall) / (recall + beta_sq * precision)


def max_over_references(
    prediction: str,
    references: Sequence[str],
    metric_fn,
) -> float:
    """Standard multi-reference handling: take the best-scoring gold answer."""
    scores = [metric_fn(prediction, ref) for ref in references if str(ref).strip()]
    return max(scores) if scores else 0.0


def score_prediction(prediction: str, references: Sequence[str]) -> Dict[str, float]:
    """All downstream accuracy metrics for one prediction."""
    refs = [r for r in references if str(r).strip()]
    if not refs:
        return {"em": 0.0, "f1": 0.0, "rouge_l": 0.0, "has_reference": 0.0}
    return {
        "em": max_over_references(prediction, refs, exact_match),
        "f1": max_over_references(prediction, refs, token_f1),
        "rouge_l": max_over_references(prediction, refs, rouge_l),
        "has_reference": 1.0,
    }


def evidence_recall(evidence_sentences: Sequence[str], compressed_context: str) -> Optional[float]:
    """Fraction of gold evidence sentences substantially retained.

    A sentence counts as retained when >= `threshold` of its content tokens
    survive. Unlike the old proxy this is reported as its own metric and is
    never used to certify answerability.
    """
    refs = [s for s in evidence_sentences if str(s).strip()]
    if not refs:
        return None
    context_tokens = set(TOKEN_RE.findall(compressed_context.lower()))
    retained = 0
    for sentence in refs:
        sent_tokens = set(TOKEN_RE.findall(str(sentence).lower()))
        if not sent_tokens:
            continue
        if len(sent_tokens & context_tokens) / len(sent_tokens) >= 0.80:
            retained += 1
    return retained / len(refs)


def aggregate(rows: Sequence[Dict[str, float]], keys: Sequence[str]) -> Dict[str, Optional[float]]:
    """Mean over rows, ignoring None, returning None for fully-missing keys."""
    out: Dict[str, Optional[float]] = {}
    for key in keys:
        values = [
            float(row[key])
            for row in rows
            if row.get(key) is not None and not isinstance(row.get(key), bool)
        ]
        out[key] = float(sum(values) / len(values)) if values else None
    return out


def std(values: Sequence[float]) -> float:
    """Population std, used for multi-seed variance reporting (finding M1)."""
    clean = [float(v) for v in values if v is not None]
    if len(clean) < 2:
        return 0.0
    mean = sum(clean) / len(clean)
    return (sum((v - mean) ** 2 for v in clean) / len(clean)) ** 0.5
