"""Coverage-profile table for OUR method only, at fixed targets plus adaptive budget.

Reproduces the dimensions of the earlier project's comparison table (Actual CR,
Ans Cov, Support Cov, Hard Fact, QA Ans Rate, Utility/Token) so results are
comparable in shape, and reports REAL reader EM/F1 in the same rows so the proxy
and the ground truth can never drift apart unnoticed.

IMPORTANT -- what these columns are
-----------------------------------
Every coverage column is a LEXICAL PROXY computed without a reader. They measure
whether answer/evidence tokens survived compression, NOT whether a model can
answer. That is EVAL_VALIDITY_AUDIT.md finding C2, and it is exactly why the
downstream pipeline exists. They are reported here as diagnostics and for
continuity with the earlier table; EM/F1 from the frozen reader remain the only
citable accuracy numbers.

Metric definitions (stated explicitly because the earlier project's formulas were
not recoverable from its output table):

  Actual CR    100 * compressed_tokens / original_tokens.  Percent KEPT, so
               lower is more compression.
  Ans Cov      Mean over examples of the fraction of gold-answer content tokens
               still present in the compressed context.
  Support Cov  Same, over the supporting sentences.
  Hard Fact    Same, restricted to "hard" tokens -- numbers and capitalised
               entities drawn from the answer and supporting sentences. These are
               the tokens whose loss is unrecoverable by paraphrase.
  QA Ans Rate  Fraction of examples where EVERY gold-answer content token
               survived. A strict per-example answerability proxy, which is why
               it sits below Ans Cov.
  Utility/Tok  mean(Ans Cov, Support Cov, Hard Fact, QA Ans Rate) / Actual CR.

The Utility/Token definition is OURS. Several candidate formulas were checked
against the earlier table and none reproduced its values exactly, so this column
is internally consistent but NOT numerically comparable to the earlier results.
The other five columns follow the obvious definitions and should be broadly
comparable, though the earlier numbers came from a different dataset pool and a
split that leaked 13.1% of test contexts.

Usage:
    python -m evaluation.run_coverage_profile \
        --input_file data_builder/english_cqr_mixed_5k_grouped/test.jsonl \
        --encoder_dir context_aware_encoder_model/outputs_english/stage1_bge_small \
        --span_model_dir intra_sentence_model/outputs_span_gbm \
        --targets 0.20,0.40,0.60 --limit 1013
"""

from __future__ import annotations

import argparse
import json
import re
import sys
import time
from pathlib import Path
from typing import Dict, List, Optional, Sequence

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from evaluation.qa_metrics import TOKEN_RE, aggregate, normalize_answer, score_prediction
from evaluation.reader_client import QwenReader, ReaderConfig
from evaluation.run_downstream_eval import (
    build_compressors,
    gold_answers,
    gold_evidence,
    load_jsonl,
)
from pipeline.compression_pipeline import estimate_token_count
from pipeline.runtime_contract import (
    EncoderContractError,
    RuntimeProvenance,
    checkpoint_fingerprint,
    resolve_encoder_source,
)
from repro.manifest import build_manifest, write_manifest

STOPWORDS = {
    "a", "an", "and", "are", "as", "at", "be", "by", "for", "from", "in", "is",
    "it", "of", "on", "or", "that", "the", "to", "was", "were", "with",
}
# Numbers, and capitalised tokens of length >= 2. These are the tokens a
# paraphrase cannot recover, so losing them is qualitatively worse than losing
# a function word.
HARD_FACT_RE = re.compile(r"\b\d+(?:[.,]\d+)?%?\b|\b[A-Z][A-Za-z0-9-]{1,}\b")


def content_tokens(text: str) -> List[str]:
    return [t for t in normalize_answer(text).split() if t not in STOPWORDS]


def hard_fact_tokens(text: str) -> List[str]:
    return [normalize_answer(m.group(0)) for m in HARD_FACT_RE.finditer(str(text or ""))]


def coverage(reference_tokens: Sequence[str], candidate: str) -> Optional[float]:
    """Fraction of reference tokens present in the candidate (multiset-free)."""
    reference_tokens = [t for t in reference_tokens if t]
    if not reference_tokens:
        return None
    present = set(normalize_answer(candidate).split())
    return sum(1 for t in reference_tokens if t in present) / len(reference_tokens)


def profile_row(row: dict, compressed: str) -> Dict[str, Optional[float]]:
    answers = gold_answers(row)
    evidence = gold_evidence(row)

    answer_toks = content_tokens(" ".join(answers))
    support_toks = content_tokens(" ".join(evidence))
    hard_toks = hard_fact_tokens(" ".join(answers + evidence))

    ans_cov = coverage(answer_toks, compressed)
    return {
        "ans_cov": ans_cov,
        "support_cov": coverage(support_toks, compressed),
        "hard_fact": coverage(hard_toks, compressed),
        # Strict: every answer content token had to survive.
        "qa_ans_rate": None if ans_cov is None else float(ans_cov >= 1.0),
    }


def run_target(
    label: str,
    target: Optional[float],
    rows: Sequence[dict],
    ctx: Dict,
    batch_reader,
    method: str,
) -> Dict[str, object]:
    compressor = ctx["compressor_full"] if method == "ours_full" else ctx["compressor_stage1"]
    key = "compressed_context" if method == "ours_full" else "stage1_context"

    per_row: List[Dict[str, object]] = []
    pending = []
    for row in rows:
        question = " ".join(str(row.get("question") or "").split())
        context = " ".join(str(row.get("context") or "").split())
        if not question or not context:
            continue

        t0 = time.monotonic()
        if target is None:
            result = compressor.compress(question=question, context=context)
        else:
            result = compressor.compress(question=question, context=context, target_ratio=target)
        latency = time.monotonic() - t0
        compressed = str(result.get(key) or "")

        original_tokens = estimate_token_count(context)
        compressed_tokens = estimate_token_count(compressed)
        record: Dict[str, object] = {
            "id": row.get("id"),
            "dataset": row.get("dataset"),
            "target": target,
            "actual_cr": 100.0 * compressed_tokens / max(original_tokens, 1),
            "compress_latency_s": latency,
            "original_tokens": original_tokens,
            "compressed_tokens": compressed_tokens,
        }
        record.update({k: (None if v is None else 100.0 * v) for k, v in profile_row(row, compressed).items()})
        per_row.append(record)
        pending.append((record, question, compressed, gold_answers(row)))

    # Real reader accuracy in the same rows, so proxy and ground truth stay tied.
    scored = [p for p in pending if p[3]]
    if batch_reader is not None and scored:
        answers = batch_reader.answer_many([(q, c) for _, q, c, _ in scored])
        for (record, _, _, refs), answer in zip(scored, answers):
            record["prediction"] = answer
            ok = bool(str(answer).strip())
            record["reader_ok"] = ok
            if ok:
                record.update(score_prediction(str(answer), refs))
            else:
                # API failure, not a wrong answer: excluded from the mean.
                record["em"] = record["f1"] = record["rouge_l"] = None

    keys = ["actual_cr", "ans_cov", "support_cov", "hard_fact", "qa_ans_rate",
            "em", "f1", "rouge_l", "compress_latency_s"]
    summary = aggregate(per_row, keys)

    parts = [summary.get(k) for k in ("ans_cov", "support_cov", "hard_fact", "qa_ans_rate")]
    parts = [p for p in parts if p is not None]
    cr = summary.get("actual_cr") or 0.0
    summary["utility_per_token"] = (
        (sum(parts) / len(parts)) / cr if parts and cr > 0 else None
    )

    datasets = sorted({str(r.get("dataset")) for r in per_row})
    by_dataset = {
        name: aggregate([r for r in per_row if str(r.get("dataset")) == name], keys)
        for name in datasets
    }
    return {"label": label, "method": method, "target": target,
            "num_rows": len(per_row), "summary": summary,
            "by_dataset": by_dataset, "rows": per_row}


def fmt(value: Optional[float], places: int = 1) -> str:
    return "n/a" if value is None else f"{value:.{places}f}"


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--input_file", type=str, required=True)
    p.add_argument("--output_dir", type=str,
                   default=str(PROJECT_ROOT / "evaluation" / "outputs_coverage_profile"))
    p.add_argument("--encoder_dir", type=str, default=None)
    p.add_argument("--allow_lexical_fallback", action="store_true")
    p.add_argument("--span_model_dir", type=str, default=None)
    p.add_argument("--budget_formula_name", type=str, default="entropy_spread")
    p.add_argument("--targets", type=str, default="0.20,0.40,0.60")
    p.add_argument("--include_variable", action="store_true", default=True,
                   help="Also run the adaptive-budget setting (target chosen by the model).")
    p.add_argument("--methods", type=str, default="ours_full")
    p.add_argument("--limit", type=int, default=1013)
    p.add_argument("--second_stage_keep_ratio", type=float, default=0.52)
    p.add_argument("--second_stage_min_keep_ratio", type=float, default=0.34)
    p.add_argument("--second_stage_max_keep_ratio", type=float, default=0.72)
    p.add_argument("--reader_model", type=str, default=None)
    p.add_argument("--no_reader", action="store_true")
    p.add_argument("--reader_concurrency", type=int, default=16)
    p.add_argument("--reader_cache", type=str, default="")
    p.add_argument("--enable_dac", action="store_true", default=True)
    p.add_argument("--disable_dac", action="store_true")
    p.add_argument("--dac_salience_backend", type=str, default="causal", choices=["causal", "mlm"])
    p.add_argument("--dac_salience_model", type=str, default=None)
    p.add_argument("--dac_fusion", type=str, default="additive", choices=["additive", "multiplicative"])
    p.add_argument("--dac_alpha", type=float, default=0.8)
    p.add_argument("--dac_require_attention", action="store_true")
    p.add_argument("--dac_strict", action="store_true")
    args = p.parse_args()

    targets = [float(t) for t in str(args.targets).split(",") if str(t).strip()]
    methods = [m.strip() for m in str(args.methods).split(",") if m.strip()]

    reader = None
    batch_reader = None
    if not args.no_reader:
        reader = QwenReader(ReaderConfig.from_env(model=args.reader_model))
        smoke = reader.smoke_test()
        print("reader_smoke_test:", json.dumps(smoke, ensure_ascii=False))
        if not smoke["reachable"]:
            raise SystemExit(f"Reader unreachable: {smoke['error']}")
        from evaluation.cached_reader import CachedBatchReader

        cache_path = args.reader_cache or str(Path(args.output_dir) / "reader_cache.jsonl")
        batch_reader = CachedBatchReader(reader, cache_path=cache_path,
                                         concurrency=args.reader_concurrency)

    try:
        resolved = resolve_encoder_source(args.encoder_dir, args.allow_lexical_fallback)
    except EncoderContractError as exc:
        raise SystemExit(f"\nENCODER CONTRACT VIOLATION\n{exc}\n")

    ctx: Dict[str, object] = dict(build_compressors(args, resolved))
    active = ctx["compressor_full"]
    provenance = RuntimeProvenance(
        encoder_kind=resolved.kind,
        encoder_requested=resolved.requested,
        encoder_path=resolved.path,
        encoder_runtime=getattr(active, "encoder_runtime", ""),
        lexical_fallback_used=resolved.is_lexical,
        fallback_reason=resolved.reason if resolved.is_lexical else "",
        encoder_load_error=getattr(active, "encoder_load_error", None),
        checkpoint_fingerprint=checkpoint_fingerprint(resolved.path),
        span_model_dir=args.span_model_dir,
        span_model_active=bool(
            getattr(getattr(active, "span_compressor", None), "learned_span_model", None)
        ),
        budget_model_dir=None,
    )
    print("system_label:", provenance.system_label())

    rows = load_jsonl(Path(args.input_file))
    if args.limit > 0:
        rows = rows[: args.limit]

    results = []
    for method in methods:
        settings: List[tuple] = []
        if args.include_variable:
            settings.append(("variable", None))
        settings.extend([(f"{t:.2f}", t) for t in targets])
        for label, target in settings:
            res = run_target(label, target, rows, ctx, batch_reader, method)
            results.append(res)
            s = res["summary"]
            print(
                f"  {method:<11} target={label:<8} CR={fmt(s['actual_cr'])} "
                f"AnsCov={fmt(s['ans_cov'])} SupCov={fmt(s['support_cov'])} "
                f"Hard={fmt(s['hard_fact'])} QAAns={fmt(s['qa_ans_rate'])} "
                f"Util/Tok={fmt(s['utility_per_token'], 3)} | "
                f"EM={fmt(s['em'], 3)} F1={fmt(s['f1'], 3)}",
                flush=True,
            )

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    write_manifest(
        out_dir / "manifest.json",
        build_manifest(
            seeds={"seeds": [0]},
            datasets=[Path(args.input_file)],
            config=vars(args),
            provenance=provenance.to_dict(),
            extra={
                "reader": reader.config.public_dict() if reader else None,
                "reader_batching": batch_reader.provenance() if batch_reader else None,
                "metric_semantics": {
                    "coverage_columns": "LEXICAL PROXIES, not reader accuracy (finding C2)",
                    "actual_cr": "percent of tokens KEPT; lower = more compression",
                    "qa_ans_rate": "fraction of examples retaining EVERY answer content token",
                    "utility_per_token": "mean(coverage columns)/actual_cr -- our definition, "
                                         "not comparable to the earlier project's column",
                    "em/f1/rouge_l": "genuine downstream reader accuracy",
                },
            },
        ),
    )

    summary = {
        "system_label": provenance.system_label(),
        "rows": [{k: v for k, v in r.items() if k != "rows"} for r in results],
    }
    (out_dir / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    with (out_dir / "per_row.jsonl").open("w", encoding="utf-8") as f:
        for res in results:
            for record in res["rows"]:
                f.write(json.dumps(record, ensure_ascii=False) + "\n")

    # Markdown table in the shape of the earlier project's results.
    lines = [
        "| Method | Target | Actual CR | Ans Cov | Support Cov | Hard Fact | QA Ans Rate | Utility/Token | EM | F1 |",
        "|---|---|---|---|---|---|---|---|---|---|",
    ]
    for res in results:
        s = res["summary"]
        lines.append(
            f"| {res['method']} | {res['label']} | {fmt(s['actual_cr'])} | {fmt(s['ans_cov'])} | "
            f"{fmt(s['support_cov'])} | {fmt(s['hard_fact'])} | {fmt(s['qa_ans_rate'])} | "
            f"{fmt(s['utility_per_token'], 3)} | {fmt(s['em'], 3)} | {fmt(s['f1'], 3)} |"
        )
    (out_dir / "table.md").write_text("\n".join(lines) + "\n", encoding="utf-8")

    print("\n" + "\n".join(lines))
    print("\nwrote:", out_dir / "summary.json")
    print("wrote:", out_dir / "table.md")


if __name__ == "__main__":
    main()
