"""Stage-1 component ablation: which signal actually produces the gain?

Stage 1 fuses seven signals. The encoder similarity is the RESIDUAL weight,
max(0.05, 1 - sum(component weights)), which under the shipped configuration is
only 0.12 -- the other 0.88 is hand-designed heuristics. The end-to-end result
(+0.203 F1 over sparse retrieval at 4x on extractive QA) therefore cannot be
attributed to the "context-aware sentence encoder" without this ablation.

Design decisions:

  * Run at 4x only. The advantage over topk_lexical is an inverted-U peaking
    there (+11pp / +22pp / +9pp of retained F1 at 2x / 4x / 8x), so 4x has the
    most headroom to resolve component differences.
  * Run on the EXTRACTIVE subset only. cqr_official and wikitext103_pseudo_cqr
    have ~20-word sentence golds scored against a short-answer prompt and sit
    near F1 0.26 for every method including the uncompressed upper bound; they
    add noise and no signal.
  * Leave-one-out arms RENORMALISE the surviving heuristics back to the original
    total. Without this, dropping a component silently raises the encoder's
    residual weight, so the arm would test two changes at once and a "component
    doesn't matter" result would be unreadable.

Usage:
    python -m evaluation.run_stage1_ablation \
        --input_file data_builder/english_cqr_mixed_5k_grouped/test_extractive.jsonl \
        --encoder_dir context_aware_encoder_model/outputs_english/stage1_bge_small \
        --ratio 0.25
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Dict, List, Optional

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from evaluation.qa_metrics import aggregate, evidence_recall, score_prediction
from evaluation.reader_client import QwenReader, ReaderConfig
from evaluation.run_coverage_profile import profile_row
from evaluation.run_downstream_eval import gold_answers, gold_evidence, load_jsonl
from pipeline.compression_pipeline import ContextAwareCompressor, estimate_token_count
from pipeline.runtime_contract import (
    EncoderContractError,
    checkpoint_fingerprint,
    resolve_encoder_source,
)
from repro.manifest import build_manifest, write_manifest

# The shipped configuration. Component weights sum to 0.88, leaving the encoder
# similarity a residual of 0.12.
BASE_WEIGHTS: Dict[str, float] = {
    "attention_probe_weight": 0.18,
    "task_reward_weight": 0.16,
    "task_descriptor_weight": 0.14,
    "dynamic_attention_weight": 0.12,
    "information_density_weight": 0.10,
    "linguistic_feature_weight": 0.18,
}
BASE_TOTAL = sum(BASE_WEIGHTS.values())

# Weight name -> the enable flags that must also be cleared when it is removed.
FLAGS_FOR: Dict[str, Dict[str, bool]] = {
    "attention_probe_weight": {"use_attention_probe": False},
    "task_descriptor_weight": {"use_task_descriptor": False},
    "dynamic_attention_weight": {"use_sentence_dynamics": False},
    "linguistic_feature_weight": {"enable_linguistic_features": False},
}


def arm_kwargs(name: str) -> Dict[str, object]:
    """Compressor kwargs for one ablation arm."""
    if name == "full":
        return dict(BASE_WEIGHTS)

    if name == "encoder_only":
        # Every heuristic off: the encoder similarity carries the whole score.
        kwargs: Dict[str, object] = {key: 0.0 for key in BASE_WEIGHTS}
        for flags in FLAGS_FOR.values():
            kwargs.update(flags)
        return kwargs

    if name == "heuristics_only":
        # Push the encoder to its 0.05 floor by scaling heuristics to 0.95.
        scale = 0.95 / BASE_TOTAL
        return {key: value * scale for key, value in BASE_WEIGHTS.items()}

    if name.startswith("no_"):
        target = {
            "no_attention": "attention_probe_weight",
            "no_task_reward": "task_reward_weight",
            "no_descriptor": "task_descriptor_weight",
            "no_dynamic": "dynamic_attention_weight",
            "no_density": "information_density_weight",
            "no_linguistic": "linguistic_feature_weight",
        }[name]
        remaining = BASE_TOTAL - BASE_WEIGHTS[target]
        # Renormalise so the encoder residual stays at 0.12 and this arm tests
        # exactly one change.
        scale = BASE_TOTAL / remaining
        kwargs = {
            key: (0.0 if key == target else value * scale)
            for key, value in BASE_WEIGHTS.items()
        }
        kwargs.update(FLAGS_FOR.get(target, {}))
        return kwargs

    raise SystemExit(f"unknown ablation arm: {name}")


ARMS = [
    "full",
    "encoder_only",
    "heuristics_only",
    "no_attention",
    "no_task_reward",
    "no_descriptor",
    "no_dynamic",
    "no_density",
    "no_linguistic",
]


def build_arm(name: str, encoder_arg: str, is_lexical: bool, budget_formula: str) -> ContextAwareCompressor:
    kwargs = arm_kwargs(name)
    return ContextAwareCompressor(
        encoder_dir=encoder_arg,
        budget_formula_name=budget_formula,
        enable_second_stage=False,  # Stage 1 only: Stage 2 is neutral end-to-end.
        allow_heuristic_fallback=is_lexical,
        enable_dac=False,           # DAC lives in Stage 2; irrelevant here.
        **kwargs,
    )


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
        record: Dict[str, object] = {
            "arm": name,
            "id": row.get("id"),
            "dataset": row.get("dataset"),
            "achieved_ratio": kept / max(original, 1),
            "actual_cr": 100.0 * kept / max(original, 1),
            "compression_x": original / max(kept, 1),
            "compress_latency_s": latency,
            "evidence_recall": evidence_recall(gold_evidence(row), compressed),
            # Stored so new metrics can be computed later without recompressing.
            "compressed_context": compressed,
        }
        # Lexical coverage columns (Ans Cov / Support Cov / Hard Fact / QA Ans
        # Rate). No model is involved: these measure token survival, not reader
        # comprehension. Reported beside EM/F1 so the two can be compared.
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

    keys = ["em", "f1", "rouge_l", "evidence_recall", "achieved_ratio",
            "actual_cr", "ans_cov", "support_cov", "hard_fact", "qa_ans_rate",
            "compression_x", "compress_latency_s"]
    summary = aggregate(per_row, keys)

    parts = [summary.get(k) for k in ("ans_cov", "support_cov", "hard_fact", "qa_ans_rate")]
    parts = [p for p in parts if p is not None]
    cr = summary.get("actual_cr") or 0.0
    summary["utility_per_token"] = (sum(parts) / len(parts)) / cr if parts and cr > 0 else None

    return {
        "arm": name,
        "weights": {k: round(float(v), 4) for k, v in arm_kwargs(name).items() if isinstance(v, (int, float))},
        "num_rows": len(per_row),
        "scored_rows": sum(1 for r in per_row if r.get("f1") is not None),
        "reader_failures": sum(1 for r in per_row if r.get("reader_ok") is False),
        "summary": summary,
        "rows": per_row,
    }


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--input_file", type=str, required=True)
    p.add_argument("--output_dir", type=str,
                   default=str(PROJECT_ROOT / "evaluation" / "outputs_stage1_ablation"))
    p.add_argument("--encoder_dir", type=str, default=None)
    p.add_argument("--allow_lexical_fallback", action="store_true")
    p.add_argument("--budget_formula_name", type=str, default="entropy_spread")
    p.add_argument("--ratio", type=float, default=0.25)
    p.add_argument("--arms", type=str, default=",".join(ARMS))
    p.add_argument("--limit", type=int, default=0)
    p.add_argument("--reader_model", type=str, default=None)
    p.add_argument("--reader_concurrency", type=int, default=24)
    p.add_argument("--reader_cache", type=str, default="")
    args = p.parse_args()

    arms = [a.strip() for a in args.arms.split(",") if a.strip()]
    for name in arms:
        arm_kwargs(name)  # validate before doing any work

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
        resolved = resolve_encoder_source(args.encoder_dir, args.allow_lexical_fallback)
    except EncoderContractError as exc:
        raise SystemExit(f"\nENCODER CONTRACT VIOLATION\n{exc}\n")
    encoder_arg = "lightweight_lexical_fallback" if resolved.is_lexical else resolved.path

    rows = load_jsonl(Path(args.input_file))
    if args.limit > 0:
        rows = rows[: args.limit]
    print(f"rows={len(rows)} ratio={args.ratio} arms={len(arms)}", flush=True)

    results = []
    for name in arms:
        compressor = build_arm(name, encoder_arg, resolved.is_lexical, args.budget_formula_name)
        res = run_arm(name, compressor, rows, args.ratio, batch_reader)
        results.append(res)
        s = res["summary"]
        def g(key, places=1):
            v = s.get(key)
            return "n/a" if v is None else f"{v:.{places}f}"

        print(
            f"  {name:<16} CR={g('actual_cr')} AnsCov={g('ans_cov')} "
            f"SupCov={g('support_cov')} Hard={g('hard_fact')} QAAns={g('qa_ans_rate')} "
            f"Util/Tok={g('utility_per_token', 3)} | EM={g('em', 3)} F1={g('f1', 3)} "
            f"lat={g('compress_latency_s', 2)}s",
            flush=True,
        )
        del compressor

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    write_manifest(
        out_dir / "manifest.json",
        build_manifest(
            seeds={"seeds": [0]},
            datasets=[Path(args.input_file)],
            config=vars(args),
            provenance={
                "encoder_kind": resolved.kind,
                "encoder_path": resolved.path,
                "checkpoint_fingerprint": checkpoint_fingerprint(resolved.path),
            },
            extra={
                "reader": reader.config.public_dict(),
                "base_weights": BASE_WEIGHTS,
                "encoder_residual_weight_in_full": round(max(0.05, 1 - BASE_TOTAL), 4),
                "note": "leave-one-out arms renormalise survivors to keep the encoder "
                        "residual constant, so each arm tests exactly one change",
            },
        ),
    )
    summary = {"ratio": args.ratio,
               "arms": [{k: v for k, v in r.items() if k != "rows"} for r in results]}
    (out_dir / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    with (out_dir / "per_row.jsonl").open("w", encoding="utf-8") as f:
        for res in results:
            for record in res["rows"]:
                f.write(json.dumps(record, ensure_ascii=False) + "\n")

    full = next((r for r in results if r["arm"] == "full"), None)
    if full:
        base_f1 = full["summary"]["f1"]
        print("\n=== delta vs full ===")
        for r in sorted(results, key=lambda r: r["summary"]["f1"] or 0):
            if r["arm"] == "full":
                continue
            print(f"  {r['arm']:<16} F1={r['summary']['f1']:.3f}  "
                  f"delta={r['summary']['f1'] - base_f1:+.3f}")
    print("\nwrote:", out_dir / "summary.json")


if __name__ == "__main__":
    main()
