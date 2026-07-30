from __future__ import annotations

import argparse
import json
import random
import statistics
import sys
from collections import Counter
from pathlib import Path
from typing import Dict, Iterable, List, Sequence

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from data_builder.build_english_cqr_dataset import TOKEN_RE, split_sentences
from pipeline.compression_pipeline import ContextAwareCompressor, estimate_token_count
from pipeline.runtime_contract import (
    LEXICAL_FALLBACK_ID,
    EncoderContractError,
    resolve_encoder_source,
)
from repro.manifest import build_manifest, write_manifest


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


def load_jsonl(path: Path) -> List[dict]:
    rows = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def save_jsonl(path: Path, rows: Iterable[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def save_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def normalize_text(text: str) -> str:
    return " ".join(str(text or "").split())


def token_set(text: str) -> set[str]:
    return {token.lower() for token in TOKEN_RE.findall(text)}


def content_token_set(text: str) -> set[str]:
    return {token for token in token_set(text) if token not in QUESTION_STOPWORDS and len(token) > 2}


def token_recall(reference: str, candidate: str) -> float | None:
    ref_tokens = token_set(reference)
    if not ref_tokens:
        return None
    cand_tokens = token_set(candidate)
    return len(ref_tokens & cand_tokens) / max(len(ref_tokens), 1)


def average_recall(references: Sequence[str], candidate: str) -> float | None:
    scores = [token_recall(ref, candidate) for ref in references if str(ref).strip()]
    scores = [score for score in scores if score is not None]
    if not scores:
        return None
    return float(sum(scores) / len(scores))


def content_token_recall(reference: str, candidate: str) -> float | None:
    ref_tokens = content_token_set(reference)
    if not ref_tokens:
        return None
    cand_tokens = content_token_set(candidate)
    return len(ref_tokens & cand_tokens) / max(len(ref_tokens), 1)


def optional_max(values: Sequence[float | None]) -> float | None:
    clean = [float(value) for value in values if value is not None]
    if not clean:
        return None
    return max(clean)


def first_nonempty_list(values: object) -> List[str]:
    if not isinstance(values, list):
        return []
    return [normalize_text(item) for item in values if normalize_text(item)]


def sentence_key(sentence: str) -> str:
    return normalize_text(sentence).lower()


def build_candidate_context(row: dict, max_context_sentences: int, rng: random.Random) -> str:
    context = normalize_text(row.get("context"))
    context_sentences = split_sentences(context)
    positive = normalize_text(row.get("positive_sentence"))
    supports = first_nonempty_list(row.get("supporting_sentences"))
    negatives = first_nonempty_list(row.get("negative_sentences"))

    wanted = [sentence for sentence in [positive, *supports, *negatives] if sentence]
    wanted_keys = {sentence_key(sentence) for sentence in wanted}
    selected_indices = [
        idx for idx, sentence in enumerate(context_sentences) if sentence_key(sentence) in wanted_keys
    ]

    remaining = [idx for idx in range(len(context_sentences)) if idx not in set(selected_indices)]
    rng.shuffle(remaining)
    for idx in remaining:
        if len(selected_indices) >= max_context_sentences:
            break
        selected_indices.append(idx)

    selected_sentences = [context_sentences[idx] for idx in sorted(set(selected_indices))]
    present_keys = {sentence_key(sentence) for sentence in selected_sentences}
    for sentence in wanted:
        if sentence_key(sentence) not in present_keys:
            selected_sentences.append(sentence)
            present_keys.add(sentence_key(sentence))

    if not selected_sentences:
        selected_sentences = context_sentences[:max_context_sentences]
    return " ".join(selected_sentences)


def build_compressor(span_model_dir: str | None, args: argparse.Namespace, encoder_source: str) -> ContextAwareCompressor:
    compressor = ContextAwareCompressor(
        encoder_dir=encoder_source,
        budget_formula_name=args.budget_formula_name,
        span_model_dir=span_model_dir,
        use_attention_probe=True,
        attention_probe_weight=0.18,
        task_reward_weight=0.16,
        use_task_descriptor=True,
        task_descriptor_weight=0.14,
        use_sentence_dynamics=True,
        dynamic_attention_weight=0.12,
        information_density_weight=0.10,
        enable_linguistic_features=True,
        linguistic_feature_weight=0.18,
        enable_second_stage=True,
        second_stage_keep_ratio=args.second_stage_keep_ratio,
        second_stage_min_keep_ratio=args.second_stage_min_keep_ratio,
        second_stage_max_keep_ratio=args.second_stage_max_keep_ratio,
        allow_heuristic_fallback=args.allow_lexical_fallback,
    )
    if compressor.span_compressor is not None and span_model_dir is not None:
        compressor.span_compressor.config.learned_keep_weight = args.learned_keep_weight
        compressor.span_compressor.config.learned_soft_protected_threshold = args.learned_soft_protected_threshold
    return compressor


def summarize_tokens(sentences: Sequence[str]) -> int:
    return sum(estimate_token_count(sentence) for sentence in sentences)


def judge_answerability(
    row: dict,
    compressed_context: str,
    answer_threshold: float,
    evidence_threshold: float,
    question_threshold: float,
) -> dict:
    question = normalize_text(row.get("question"))
    answer = normalize_text(row.get("answer"))
    positive = normalize_text(row.get("positive_sentence"))
    supports = first_nonempty_list(row.get("supporting_sentences")) or ([positive] if positive else [])

    answer_coverage = token_recall(answer, compressed_context)
    positive_coverage = token_recall(positive, compressed_context)
    support_coverage = average_recall(supports, compressed_context)
    question_coverage = content_token_recall(question, compressed_context)
    evidence_coverage = optional_max([positive_coverage, support_coverage])

    # FAIL-CLOSED (audit finding C2): missing coverage previously defaulted to
    # True, so rows without an answer or question counted as "answerable" and
    # inflated the reported rate. Absent evidence is now never a pass.
    answer_ok = False if answer_coverage is None else answer_coverage >= answer_threshold
    evidence_ok = False if evidence_coverage is None else evidence_coverage >= evidence_threshold
    question_ok = False if question_coverage is None else question_coverage >= question_threshold
    answerable = bool(answer_ok and evidence_ok and question_ok)

    score_parts = [
        answer_coverage if answer_coverage is not None else 0.0,
        evidence_coverage if evidence_coverage is not None else 0.0,
        question_coverage if question_coverage is not None else 0.0,
    ]
    # Keys are deliberately prefixed `lexical_proxy_`: this measures whether
    # tokens survived, NOT whether a reader can answer. Real answerability comes
    # from evaluation/run_downstream_eval.py (EM/F1 with a frozen reader LLM).
    return {
        "lexical_proxy_answerable": answerable,
        "lexical_proxy_score": float(sum(score_parts) / len(score_parts)),
        "lexical_proxy_answer_ok": bool(answer_ok),
        "lexical_proxy_evidence_ok": bool(evidence_ok),
        "lexical_proxy_question_ok": bool(question_ok),
        "lexical_proxy_evidence_coverage": evidence_coverage,
        "lexical_proxy_question_coverage": question_coverage,
    }


def evaluate_row(row: dict, context: str, result: dict, variant: str, args: argparse.Namespace) -> dict:
    original_tokens = summarize_tokens(result["sentences"])
    stage1_tokens = summarize_tokens(result["selected_sentences"])
    stage2_tokens = summarize_tokens(result["compressed_sentences"])
    compressed_context = str(result.get("compressed_context") or "")

    positive = normalize_text(row.get("positive_sentence"))
    supports = first_nonempty_list(row.get("supporting_sentences")) or ([positive] if positive else [])
    answer = normalize_text(row.get("answer"))
    sentence_stats = result.get("second_stage_stats", {}).get("sentence_stats", [])
    modes = sorted({str(stat.get("compression_mode") or "") for stat in sentence_stats if stat.get("compression_mode")})
    safety_skipped_count = sum(int(stat.get("safety_skipped_count", 0)) for stat in sentence_stats)

    record = {
        "variant": variant,
        "target_ratio": float(result.get("target_ratio", 0.0)),
        "original_tokens": original_tokens,
        "stage1_tokens": stage1_tokens,
        "stage2_tokens": stage2_tokens,
        "stage1_ratio": stage1_tokens / max(original_tokens, 1),
        "final_ratio": stage2_tokens / max(original_tokens, 1),
        "selected_sentence_count": len(result.get("selected_sentences") or []),
        "removed_span_count": int(result.get("second_stage_stats", {}).get("removed_span_count", 0)),
        "safety_skipped_count": safety_skipped_count,
        "positive_coverage": token_recall(positive, compressed_context),
        "support_coverage": average_recall(supports, compressed_context),
        "answer_coverage": token_recall(answer, compressed_context),
        "compression_modes": modes,
        "compressed_context": compressed_context,
        "input_context": context,
    }
    record.update(
        judge_answerability(
            row=row,
            compressed_context=compressed_context,
            answer_threshold=args.answerability_answer_threshold,
            evidence_threshold=args.answerability_evidence_threshold,
            question_threshold=args.answerability_question_threshold,
        )
    )
    return record


def mean_optional(values: Sequence[float | None]) -> float | None:
    clean = [float(value) for value in values if value is not None]
    if not clean:
        return None
    return float(sum(clean) / len(clean))


def median(values: Sequence[float]) -> float:
    return float(statistics.median(values)) if values else 0.0


def summarize_variant(records: Sequence[dict]) -> dict:
    return {
        "num_rows": len(records),
        "avg_final_ratio": mean_optional([row["final_ratio"] for row in records]),
        "median_final_ratio": median([row["final_ratio"] for row in records]),
        "avg_stage1_ratio": mean_optional([row["stage1_ratio"] for row in records]),
        "avg_positive_coverage": mean_optional([row["positive_coverage"] for row in records]),
        "avg_support_coverage": mean_optional([row["support_coverage"] for row in records]),
        "avg_answer_coverage": mean_optional([row["answer_coverage"] for row in records]),
        "avg_lexical_proxy_score": mean_optional([row["lexical_proxy_score"] for row in records]),
        "lexical_proxy_answerable_rows": sum(1 for row in records if row.get("lexical_proxy_answerable")),
        "lexical_proxy_answerable_rate": (
            sum(1 for row in records if row.get("lexical_proxy_answerable")) / max(len(records), 1)
        ),
        "lexical_proxy_answer_ok_rate": (
            sum(1 for row in records if row.get("lexical_proxy_answer_ok")) / max(len(records), 1)
        ),
        "lexical_proxy_evidence_ok_rate": (
            sum(1 for row in records if row.get("lexical_proxy_evidence_ok")) / max(len(records), 1)
        ),
        "lexical_proxy_question_ok_rate": (
            sum(1 for row in records if row.get("lexical_proxy_question_ok")) / max(len(records), 1)
        ),
        "avg_removed_span_count": mean_optional([row["removed_span_count"] for row in records]),
        "avg_safety_skipped_count": mean_optional([row["safety_skipped_count"] for row in records]),
        "avg_selected_sentence_count": mean_optional([row["selected_sentence_count"] for row in records]),
        "mode_counts": dict(Counter(mode for row in records for mode in row.get("compression_modes", []))),
    }


def compare_pairs(rows: Sequence[dict]) -> dict:
    learned_better_ratio = 0
    learned_same_or_better_coverage = 0
    learned_regressed_coverage = 0
    learned_identical_context = 0
    final_ratio_delta = []
    positive_delta = []
    support_delta = []
    answer_delta = []
    answerability_score_delta = []
    efficient_wins = 0
    learned_same_or_better_answerable = 0

    for row in rows:
        heuristic = row["heuristic"]
        learned = row["learned"]
        ratio_delta = learned["final_ratio"] - heuristic["final_ratio"]
        pos_delta = (learned["positive_coverage"] or 0.0) - (heuristic["positive_coverage"] or 0.0)
        sup_delta = (learned["support_coverage"] or 0.0) - (heuristic["support_coverage"] or 0.0)
        ans_delta = (learned["answer_coverage"] or 0.0) - (heuristic["answer_coverage"] or 0.0)
        qa_delta = learned["lexical_proxy_score"] - heuristic["lexical_proxy_score"]

        final_ratio_delta.append(ratio_delta)
        positive_delta.append(pos_delta)
        support_delta.append(sup_delta)
        answer_delta.append(ans_delta)
        answerability_score_delta.append(qa_delta)

        if ratio_delta < -1e-9:
            learned_better_ratio += 1
        if pos_delta >= -0.01 and sup_delta >= -0.01 and ans_delta >= -0.01:
            learned_same_or_better_coverage += 1
        if min(pos_delta, sup_delta, ans_delta) < -0.03:
            learned_regressed_coverage += 1
        if learned["compressed_context"] == heuristic["compressed_context"]:
            learned_identical_context += 1
        if learned["lexical_proxy_answerable"] or not heuristic["lexical_proxy_answerable"]:
            learned_same_or_better_answerable += 1
        if ratio_delta <= 0.0 and pos_delta >= -0.01 and sup_delta >= -0.01 and ans_delta >= -0.01:
            efficient_wins += 1

    n = max(len(rows), 1)
    return {
        "avg_final_ratio_delta_learned_minus_heuristic": float(sum(final_ratio_delta) / n),
        "avg_positive_coverage_delta": float(sum(positive_delta) / n),
        "avg_support_coverage_delta": float(sum(support_delta) / n),
        "avg_answer_coverage_delta": float(sum(answer_delta) / n),
        "avg_lexical_proxy_score_delta": float(sum(answerability_score_delta) / n),
        "learned_lower_ratio_rows": learned_better_ratio,
        "learned_same_or_better_coverage_rows": learned_same_or_better_coverage,
        "learned_regressed_coverage_rows": learned_regressed_coverage,
        "learned_identical_context_rows": learned_identical_context,
        "learned_same_or_better_answerable_rows": learned_same_or_better_answerable,
        "learned_efficient_win_rows": efficient_wins,
        "learned_efficient_win_rate": efficient_wins / n,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Compare heuristic vs learned Stage-2 span compression on English CQR-style benchmark rows.")
    parser.add_argument("--input_file", type=str, default=str(PROJECT_ROOT / "data_builder" / "english_cqr_mixed_5k" / "test.jsonl"))
    parser.add_argument("--output_dir", type=str, default=str(PROJECT_ROOT / "intra_sentence_model" / "benchmark_outputs_english"))
    parser.add_argument("--span_model_dir", type=str, default=str(PROJECT_ROOT / "intra_sentence_model" / "outputs_english_feedback"))
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--encoder_dir", type=str, default=None,
                        help="Trained context-aware encoder checkpoint. Required unless "
                             "--allow_lexical_fallback is set.")
    parser.add_argument("--allow_lexical_fallback", action="store_true",
                        help="Explicitly run the NON-NEURAL lexical backend as an ablation "
                             "baseline. Results are labelled and must not be reported as "
                             "the main system.")
    # Default is now `full` (audit finding C3). `candidate` guarantees gold evidence
    # is present, which measures Stage-2 on an oracle-cleaned distribution the
    # deployed system never sees; it is a diagnostic, not a headline setting.
    parser.add_argument("--context_mode", choices=["candidate", "full"], default="full")
    parser.add_argument("--max_context_sentences", type=int, default=10)
    parser.add_argument("--budget_formula_name", type=str, default="entropy_spread")
    parser.add_argument("--second_stage_keep_ratio", type=float, default=0.52)
    parser.add_argument("--second_stage_min_keep_ratio", type=float, default=0.34)
    parser.add_argument("--second_stage_max_keep_ratio", type=float, default=0.72)
    parser.add_argument("--learned_keep_weight", type=float, default=0.72)
    parser.add_argument("--learned_soft_protected_threshold", type=float, default=0.28)
    parser.add_argument("--answerability_answer_threshold", type=float, default=0.50)
    parser.add_argument("--answerability_evidence_threshold", type=float, default=0.55)
    parser.add_argument("--answerability_question_threshold", type=float, default=0.25)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    rows = load_jsonl(Path(args.input_file))
    if args.limit > 0:
        rows = rows[: args.limit]

    # Encoder contract: refuse to silently degrade (audit findings C1 / H3).
    try:
        resolved = resolve_encoder_source(args.encoder_dir, args.allow_lexical_fallback)
    except EncoderContractError as exc:
        raise SystemExit(f"\nENCODER CONTRACT VIOLATION\n{exc}\n")
    encoder_source = LEXICAL_FALLBACK_ID if resolved.is_lexical else resolved.path

    print(
        "\nNOTE: every metric produced by this script is a LEXICAL COVERAGE PROXY\n"
        "(prefixed `lexical_proxy_`). It measures token survival, not whether a\n"
        "reader can answer. For genuine EM/F1 run:\n"
        "  python -m evaluation.run_downstream_eval --input_file ... --encoder_dir ...\n"
    )
    if args.context_mode == "candidate":
        print(
            "WARNING: --context_mode candidate builds contexts that GUARANTEE gold\n"
            "evidence is present. This overestimates Stage-2. Diagnostic use only.\n"
        )

    rng = random.Random(args.seed)
    heuristic_compressor = build_compressor(span_model_dir=None, args=args, encoder_source=encoder_source)
    learned_compressor = build_compressor(span_model_dir=args.span_model_dir, args=args, encoder_source=encoder_source)
    provenance = learned_compressor.provenance()
    print("system_label:", provenance.system_label())
    if provenance.lexical_fallback_used:
        print(
            "\n*** WARNING: NON-NEURAL lexical backend active. These numbers do NOT\n"
            "come from the trained context-aware encoder. ***\n"
        )

    detail_rows = []
    for idx, row in enumerate(rows):
        context = normalize_text(row.get("context")) if args.context_mode == "full" else build_candidate_context(row, args.max_context_sentences, rng)
        question = normalize_text(row.get("question"))
        if not question or not context:
            continue

        heuristic_result = heuristic_compressor.compress(question=question, context=context)
        learned_result = learned_compressor.compress(question=question, context=context)
        detail_rows.append(
            {
                "id": row.get("id") or row.get("source_id") or idx,
                "dataset": row.get("dataset"),
                "quality": (row.get("metadata") or {}).get("quality"),
                "question": question,
                "answer": row.get("answer", ""),
                "positive_sentence": row.get("positive_sentence", ""),
                "heuristic": evaluate_row(row, context, heuristic_result, "heuristic", args),
                "learned": evaluate_row(row, context, learned_result, "learned", args),
            }
        )

    heuristic_records = [row["heuristic"] for row in detail_rows]
    learned_records = [row["learned"] for row in detail_rows]
    summary = {
        "system_label": provenance.system_label(),
        "metric_semantics": (
            "All `lexical_proxy_*` fields are token-coverage PROXIES, not downstream "
            "answer accuracy. Do not report them as answerability."
        ),
        "runtime_provenance": provenance.to_dict(),
        "input_file": str(Path(args.input_file)),
        "num_input_rows": len(rows),
        "num_evaluated_rows": len(detail_rows),
        "context_mode": args.context_mode,
        "max_context_sentences": args.max_context_sentences,
        "second_stage_keep_ratio": args.second_stage_keep_ratio,
        "second_stage_min_keep_ratio": args.second_stage_min_keep_ratio,
        "second_stage_max_keep_ratio": args.second_stage_max_keep_ratio,
        "learned_keep_weight": args.learned_keep_weight,
        "learned_soft_protected_threshold": args.learned_soft_protected_threshold,
        "answerability_thresholds": {
            "answer": args.answerability_answer_threshold,
            "evidence": args.answerability_evidence_threshold,
            "question": args.answerability_question_threshold,
        },
        "heuristic": summarize_variant(heuristic_records),
        "learned": summarize_variant(learned_records),
        "comparison": compare_pairs(detail_rows),
        "dataset_counts": dict(Counter(str(row.get("dataset") or "") for row in detail_rows)),
        "quality_counts": dict(Counter(str(row.get("quality") or "") for row in detail_rows)),
    }

    output_dir = Path(args.output_dir)
    save_jsonl(output_dir / "stage2_english_benchmark_details.jsonl", detail_rows)
    save_json(output_dir / "stage2_english_benchmark_summary.json", summary)
    write_manifest(
        output_dir / "manifest.json",
        build_manifest(
            seeds={"seed": args.seed},
            datasets=[Path(args.input_file)],
            config=vars(args),
            provenance=provenance.to_dict(),
        ),
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
