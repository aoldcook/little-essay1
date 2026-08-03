"""A/B test of evidence-cluster (bridge) coverage in Stage-1 selection.

Motivation, measured rather than assumed. At 8x compression on multi-hop QA:

    both supporting facts retained   17.2% of examples   F1 0.899
    exactly ONE of two retained      31.6% of examples   F1 0.451
    neither retained                 11.1% of examples   F1 0.195

Relevance ranking concentrates the budget on sentences that match the question
lexically, starving the bridge fact that shares few surface terms with it. MMR
removes redundancy but never guarantees COVERAGE of distinct evidence.

This reports F1 AND the bridge-retention distribution, because a gain that does
not come with a shift in how many supporting facts survive would mean the
mechanism is not what we claim, even if the number improves.

Contrast with the adaptive-ratio idea, which we measured and rejected: that
needed a per-example prediction (0.69 AUC) and its routing errors were sharply
asymmetric, so a learned policy scored 0.057 F1 BELOW a fixed ratio. Cluster
coverage is a structural constraint requiring no per-example prediction.
"""

from __future__ import annotations

import argparse
import collections
import json
import statistics
import sys
import time
from pathlib import Path
from typing import Dict, List, Optional, Sequence

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from evaluation.qa_metrics import aggregate, evidence_recall, normalize_answer, score_prediction
from evaluation.reader_client import QwenReader, ReaderConfig
from evaluation.run_coverage_profile import profile_row
from evaluation.run_downstream_eval import gold_answers, gold_evidence, load_jsonl
from pipeline.compression_pipeline import ContextAwareCompressor, estimate_token_count
from pipeline.runtime_contract import EncoderContractError, resolve_encoder_source

ENCODER_ONLY = dict(
    attention_probe_weight=0.0,
    task_reward_weight=0.0,
    task_descriptor_weight=0.0,
    dynamic_attention_weight=0.0,
    information_density_weight=0.0,
    linguistic_feature_weight=0.0,
    use_attention_probe=False,
    use_task_descriptor=False,
    use_sentence_dynamics=False,
    enable_linguistic_features=False,
)


def fact_retained(sentence: str, compressed: str, threshold: float = 0.8) -> bool:
    tokens = [t for t in normalize_answer(sentence).split() if len(t) > 2]
    if not tokens:
        return False
    present = set(normalize_answer(compressed).split())
    return sum(1 for t in tokens if t in present) / len(tokens) >= threshold


def run_arm(name: str, compressor, rows, ratio: float, batch_reader) -> Dict[str, object]:
    per_row: List[Dict[str, object]] = []
    pending = []
    for row in rows:
        question = " ".join(str(row.get("question") or "").split())
        context = " ".join(str(row.get("context") or "").split())
        if not question or not context:
            continue
        t0 = time.monotonic()
        result = compressor.compress(question=question, context=context, target_ratio=ratio)
        latency = time.monotonic() - t0
        compressed = str(result.get("stage1_context") or "")

        original = estimate_token_count(context)
        kept = estimate_token_count(compressed)
        support = [s for s in (row.get("supporting_sentences") or []) if str(s).strip()]
        retained = sum(1 for s in support if fact_retained(s, compressed))

        record: Dict[str, object] = {
            "arm": name,
            "id": row.get("id"),
            "dataset": row.get("dataset"),
            "achieved_ratio": kept / max(original, 1),
            "actual_cr": 100.0 * kept / max(original, 1),
            "compress_latency_s": latency,
            "evidence_recall": evidence_recall(gold_evidence(row), compressed),
            "num_support": len(support),
            "support_retained": retained,
            "all_support_retained": 1.0 if support and retained == len(support) else 0.0,
        }
        record.update(
            {k: (None if v is None else 100.0 * v) for k, v in profile_row(row, compressed).items()}
        )
        per_row.append(record)
        pending.append((record, question, compressed, gold_answers(row)))

    scored = [p for p in pending if p[3]]
    if batch_reader is not None and scored:
        answers = batch_reader.answer_many([(q, c) for _, q, c, _ in scored])
        for (record, _, _, refs), answer in zip(scored, answers):
            ok = bool(str(answer).strip())
            record["reader_ok"] = ok
            if ok:
                record.update(score_prediction(str(answer), refs))
            else:
                record["em"] = record["f1"] = record["rouge_l"] = None

    keys = ["em", "f1", "rouge_l", "evidence_recall", "achieved_ratio", "actual_cr",
            "ans_cov", "support_cov", "hard_fact", "qa_ans_rate",
            "all_support_retained", "compress_latency_s"]
    summary = aggregate(per_row, keys)

    multi = [r for r in per_row if int(r["num_support"] or 0) >= 2]
    dist = collections.Counter(
        f"{int(r['support_retained'])} of {int(r['num_support'])}" for r in multi
    )
    return {
        "arm": name,
        "num_rows": len(per_row),
        "scored_rows": sum(1 for r in per_row if r.get("f1") is not None),
        "summary": summary,
        "multi_hop_rows": len(multi),
        "retention_distribution": dict(dist.most_common(8)),
        "rows": per_row,
    }


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--input_file", type=str, required=True)
    p.add_argument("--output_dir", type=str,
                   default=str(PROJECT_ROOT / "evaluation" / "outputs_bridge_ablation"))
    p.add_argument("--encoder_dir", type=str, required=True)
    p.add_argument("--budget_formula_name", type=str, default="entropy_spread")
    p.add_argument("--ratios", type=str, default="0.125,0.25")
    p.add_argument("--limit", type=int, default=0)
    p.add_argument("--cluster_threshold", type=float, default=0.55)
    p.add_argument("--max_clusters", type=int, default=6)
    p.add_argument("--coverage_sentences", type=int, default=3)
    p.add_argument("--reader_model", type=str, default=None)
    p.add_argument("--reader_concurrency", type=int, default=24)
    p.add_argument("--reader_cache", type=str, default="")
    args = p.parse_args()

    ratios = [float(r) for r in args.ratios.split(",") if r.strip()]

    reader = QwenReader(ReaderConfig.from_env(model=args.reader_model))
    smoke = reader.smoke_test()
    print("reader_smoke_test:", json.dumps(smoke, ensure_ascii=False), flush=True)
    if not smoke["reachable"]:
        raise SystemExit(f"Reader unreachable: {smoke['error']}")
    from evaluation.cached_reader import CachedBatchReader

    cache_path = args.reader_cache or str(Path(args.output_dir) / "reader_cache.jsonl")
    batch_reader = CachedBatchReader(reader, cache_path=cache_path,
                                     concurrency=args.reader_concurrency)

    try:
        resolved = resolve_encoder_source(args.encoder_dir, False)
    except EncoderContractError as exc:
        raise SystemExit(f"\nENCODER CONTRACT VIOLATION\n{exc}\n")

    rows = load_jsonl(Path(args.input_file))
    if args.limit > 0:
        rows = rows[: args.limit]
    print(f"rows={len(rows)} ratios={ratios}", flush=True)

    results = []
    for bridge in (False, True):
        compressor = ContextAwareCompressor(
            encoder_dir=resolved.path,
            budget_formula_name=args.budget_formula_name,
            enable_second_stage=False,
            enable_dac=False,
            allow_heuristic_fallback=False,
            bridge_coverage=bridge,
            bridge_cluster_threshold=args.cluster_threshold,
            bridge_max_clusters=args.max_clusters,
            bridge_coverage_sentences=args.coverage_sentences,
            **ENCODER_ONLY,
        )
        for ratio in ratios:
            name = f"{'bridge' if bridge else 'baseline'}@{ratio:.3f}"
            res = run_arm(name, compressor, rows, ratio, batch_reader)
            results.append(res)
            s = res["summary"]
            print(
                f"  {name:<18} CR={s['actual_cr']:.1f} EM={s['em']:.3f} F1={s['f1']:.3f} "
                f"allSupport={s['all_support_retained']:.3f} "
                f"AnsCov={s['ans_cov']:.1f} lat={s['compress_latency_s']:.2f}s",
                flush=True,
            )
            print(f"      retention: {res['retention_distribution']}", flush=True)
        del compressor

    print("\n=== bridge vs baseline (matched target) ===")
    for ratio in ratios:
        b = next(r for r in results if r["arm"] == f"baseline@{ratio:.3f}")
        v = next(r for r in results if r["arm"] == f"bridge@{ratio:.3f}")
        bs, vs = b["summary"], v["summary"]
        print(f"  ratio {ratio:.3f}: F1 {bs['f1']:.3f} -> {vs['f1']:.3f} "
              f"({vs['f1'] - bs['f1']:+.3f})   CR {bs['actual_cr']:.1f} -> {vs['actual_cr']:.1f}   "
              f"allSupport {bs['all_support_retained']:.3f} -> {vs['all_support_retained']:.3f} "
              f"({vs['all_support_retained'] - bs['all_support_retained']:+.3f})")

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "summary.json").write_text(
        json.dumps({"arms": [{k: v for k, v in r.items() if k != "rows"} for r in results]},
                   ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    with (out_dir / "per_row.jsonl").open("w", encoding="utf-8") as f:
        for res in results:
            for record in res["rows"]:
                f.write(json.dumps(record, ensure_ascii=False) + "\n")
    print("\nwrote:", out_dir / "summary.json")


if __name__ == "__main__":
    main()
