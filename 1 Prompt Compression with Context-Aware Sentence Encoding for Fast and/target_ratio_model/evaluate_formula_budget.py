from __future__ import annotations

import argparse
import json
import random
import re
from pathlib import Path
from typing import Dict, Iterable, List, Sequence

import numpy as np

from pipeline.linguistic_information import build_linguistic_sentence_features
from pipeline.task_descriptor import build_task_descriptor
from target_ratio_model.budget_features import split_sentences
from target_ratio_model.formula_budget import DEFAULT_RATIO_BUCKETS, FORMULA_NAMES, predict_formula_ratio
from target_ratio_model.generate_pseudo_labels import evidence_coverage_score


TOKEN_RE = re.compile(r"[A-Za-z][A-Za-z0-9]*(?:-[A-Za-z0-9]+)*|\d+(?:\.\d+)?%?")
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


def token_len(text: str) -> int:
    return max(1, len(TOKEN_RE.findall(text)))


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


def lexical_similarity(question: str, sentence: str, rng: random.Random) -> float:
    q_terms = set(tokenize(question))
    s_terms = set(tokenize(sentence))
    if not s_terms:
        return 0.0
    overlap = len(q_terms & s_terms) / max(len(q_terms), 1)
    discourse_bonus = 0.08 if re.search(r"\b(because|therefore|include|includes|however|whereas|after|before|defined as|refers to)\b", sentence.lower()) else 0.0
    number_bonus = 0.06 if re.search(r"\d", question) and re.search(r"\d", sentence) else 0.0
    entity_bonus = 0.04 if re.search(r"\b[A-Z][A-Za-z0-9-]{2,}\b", sentence) else 0.0
    noise = rng.uniform(-0.035, 0.035)
    return float(min(1.0, max(0.0, 0.68 * overlap + discourse_bonus + number_bonus + entity_bonus + noise)))


def compress_with_scores(sentences: Sequence[str], similarities: Sequence[float], target_ratio: float) -> str:
    total_len = sum(token_len(sentence) for sentence in sentences)
    budget = max(1, int(total_len * target_ratio))
    ranked = sorted(range(len(sentences)), key=lambda idx: similarities[idx], reverse=True)
    selected: List[int] = []
    used = 0
    for idx in ranked:
        sent_len = token_len(sentences[idx])
        if used + sent_len <= budget:
            selected.append(idx)
            used += sent_len
    if not selected and ranked:
        selected = [ranked[0]]
    return " ".join(sentences[idx] for idx in sorted(selected))


def minimal_safe_ratio(sentences: Sequence[str], similarities: Sequence[float], evidence: Sequence[str], threshold: float) -> float:
    for ratio in DEFAULT_RATIO_BUCKETS:
        compressed = compress_with_scores(sentences, similarities, ratio)
        if evidence_coverage_score(compressed, list(evidence)) >= threshold:
            return ratio
    return 1.0


def build_case(case_id: int, rng: random.Random) -> Dict[str, object]:
    topics = [
        ("climate threshold", "1.5 degrees Celsius", "ecological tipping points", "ice-sheet loss", "permafrost methane"),
        ("battery recycling", "lithium recovery", "supply-chain pressure", "nickel waste", "cobalt contamination"),
        ("hospital triage", "early warning score", "patient deterioration", "oxygen saturation", "sepsis risk"),
        ("retrieval systems", "context reranking", "answer faithfulness", "bridge evidence", "duplicate passages"),
        ("urban flooding", "stormwater capacity", "infrastructure failure", "blocked drains", "surface runoff"),
        ("education policy", "attendance intervention", "learning recovery", "family support", "early warning signals"),
    ]
    topic, key_value, outcome, risk_a, risk_b = rng.choice(topics)
    qtype = ["cause", "numeric", "comparison", "procedure", "definition", "factoid"][case_id % 6]

    distractors = [
        f"Public discussion of {topic} often uses broad terminology that varies across reports.",
        f"Several organizations publish annual summaries, but their methods and audiences differ.",
        f"Historical background can help frame the issue, although it is not always needed for a direct answer.",
        f"Some teams also track cost, staffing, and implementation barriers in parallel.",
        f"The topic appears in policy documents, technical notes, and stakeholder briefings.",
    ]

    if qtype == "cause":
        question = f"Why is the {key_value} threshold considered critical for {topic}?"
        evidence = [
            f"Researchers report that crossing {key_value} sharply increases the probability of {outcome}.",
            f"These risks include {risk_a}, {risk_b}, severe service disruption, and stress on dependent systems.",
        ]
        sentences = [
            f"{topic.title()} is influenced by technical choices, institutional capacity, and long-term planning.",
            evidence[0],
            evidence[1],
            f"A mitigation plan can reduce exposure through monitoring, prevention, and targeted investment.",
            rng.choice(distractors),
        ]
    elif qtype == "numeric":
        value = rng.choice(["35%", "2.4 million", "72 hours", "1.7B tokens", "2024"])
        question = f"What exact value should be preserved when explaining {topic}?"
        evidence = [f"The report states that the key measured value for {topic} is {value}, including its unit and qualifier."]
        sentences = [
            rng.choice(distractors),
            f"Analysts describe {topic} using several approximate indicators.",
            evidence[0],
            f"Background discussion can be shortened if it does not change the interpretation of {value}.",
            f"Removing the unit would make the answer ambiguous.",
        ]
    elif qtype == "comparison":
        question = f"How does the new {topic} method differ from the baseline?"
        evidence = [
            f"The baseline method ranks passages mostly by keyword overlap and often repeats the same evidence.",
            f"However, the new method preserves bridge evidence and reduces redundant context before answering.",
        ]
        sentences = [
            f"{topic.title()} systems are usually evaluated with answer quality and latency.",
            evidence[0],
            evidence[1],
            rng.choice(distractors),
            f"Both methods can be deployed without changing the downstream language model.",
        ]
    elif qtype == "procedure":
        question = f"What steps are needed to apply the {topic} workflow?"
        evidence = [
            f"First, the workflow extracts candidate evidence sentences from the source context.",
            f"Then it scores dependencies, removes redundant background, and keeps protected spans.",
            f"Finally, the compressed context is checked against the question before downstream answering.",
        ]
        sentences = [
            rng.choice(distractors),
            evidence[0],
            evidence[1],
            evidence[2],
            f"Optional dashboards can show intermediate scores for auditing.",
        ]
    elif qtype == "definition":
        question = f"What does {topic} mean in this context?"
        evidence = [f"In this context, {topic} refers to a structured process for preserving answer-critical evidence while reducing irrelevant text."]
        sentences = [
            evidence[0],
            f"The phrase may be used differently in marketing and engineering documents.",
            rng.choice(distractors),
            f"Examples include keeping entities, numbers, causal links, and core predicates.",
            f"The system should avoid introducing facts not present in the source.",
        ]
    else:
        question = f"Which organization reported the main finding about {topic}?"
        org = rng.choice(["The Intergovernmental Panel", "The National Audit Office", "The Research Safety Board", "The Open Data Institute"])
        evidence = [f"{org} reported the main finding about {topic} after reviewing field evidence."]
        sentences = [
            rng.choice(distractors),
            f"Several regional groups discussed {topic} in workshops.",
            evidence[0],
            f"The finding was later cited in a public briefing.",
            f"Unrelated commentary focused on budgets and procurement timelines.",
        ]

    extra_count = rng.randint(0, 3)
    sentences = sentences + rng.sample(distractors, k=extra_count)
    rng.shuffle(sentences)
    similarities = [lexical_similarity(question, sentence, rng) for sentence in sentences]
    context = " ".join(sentences)
    descriptor = build_task_descriptor(question)
    linguistic_features = [feature.to_dict() for feature in build_linguistic_sentence_features(question, sentences, descriptor)]
    oracle = minimal_safe_ratio(sentences, similarities, evidence, threshold=0.85)
    return {
        "id": f"synthetic_{case_id:03d}",
        "question_type": qtype,
        "question": question,
        "context": context,
        "sentences": sentences,
        "similarities": similarities,
        "evidence": evidence,
        "oracle_ratio": oracle,
        "linguistic_features": linguistic_features,
    }


def write_jsonl(path: Path, rows: Iterable[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def evaluate_formula(case: dict, formula_name: str) -> Dict[str, float]:
    result = predict_formula_ratio(
        formula_name=formula_name,
        question=str(case["question"]),
        context=str(case["context"]),
        similarities=case["similarities"],
        linguistic_features=case.get("linguistic_features"),
    )
    compressed = compress_with_scores(case["sentences"], case["similarities"], result.ratio)
    coverage = evidence_coverage_score(compressed, case["evidence"])
    oracle = float(case["oracle_ratio"])
    return {
        "pred_ratio": float(result.ratio),
        "raw_ratio": float(result.raw_ratio),
        "oracle_ratio": oracle,
        "coverage": float(coverage),
        "success": float(coverage >= 0.85),
        "under_budget": float(result.ratio < oracle),
        "excess_ratio": float(max(result.ratio - oracle, 0.0)),
        "abs_error": float(abs(result.ratio - oracle)),
    }


def summarize(rows: Sequence[dict]) -> Dict[str, dict]:
    summary: Dict[str, dict] = {}
    for formula_name in FORMULA_NAMES:
        metrics = [evaluate_formula(row, formula_name) for row in rows]
        summary[formula_name] = {
            "success_rate": float(np.mean([m["success"] for m in metrics])),
            "under_budget_rate": float(np.mean([m["under_budget"] for m in metrics])),
            "mean_pred_ratio": float(np.mean([m["pred_ratio"] for m in metrics])),
            "mean_oracle_ratio": float(np.mean([m["oracle_ratio"] for m in metrics])),
            "mean_excess_ratio": float(np.mean([m["excess_ratio"] for m in metrics])),
            "mean_abs_error": float(np.mean([m["abs_error"] for m in metrics])),
            "mean_coverage": float(np.mean([m["coverage"] for m in metrics])),
        }
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate formula-based target ratio predictors on synthetic English QA contexts.")
    parser.add_argument("--num_cases", type=int, default=160)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output_jsonl", type=str, default="target_ratio_model/formula_ratio_synthetic_cases.jsonl")
    parser.add_argument("--summary_json", type=str, default="target_ratio_model/formula_ratio_eval_summary.json")
    args = parser.parse_args()

    rng = random.Random(args.seed)
    rows = [build_case(case_id, rng) for case_id in range(args.num_cases)]
    summary = summarize(rows)
    best_name = sorted(
        summary,
        key=lambda name: (
            -summary[name]["success_rate"],
            summary[name]["under_budget_rate"],
            summary[name]["mean_pred_ratio"],
            summary[name]["mean_abs_error"],
        ),
    )[0]

    write_jsonl(Path(args.output_jsonl), rows)
    summary_obj = {
        "num_cases": len(rows),
        "seed": args.seed,
        "best_formula": best_name,
        "metrics": summary,
    }
    Path(args.summary_json).write_text(json.dumps(summary_obj, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(summary_obj, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
