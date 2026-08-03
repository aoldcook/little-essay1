"""Two experiments that decide what Stage 1 can claim.

After the Stage-1 ablation, the best configuration is encoder + MMR + budget:
every added heuristic cost accuracy (encoder_only F1 0.818 vs full 0.711 at 4x
on extractive QA). But context-aware sentence encoding is CPC's method, not
ours. Two questions remain, and each maps to one arm here.

E1 -- ENCODER PROVENANCE. Does our CQR fine-tuning beat the stock checkpoint?
     Arms `ours` and `stock` share everything except the weights. If stock
     matches ours, Stage 1 is a CPC reproduction and must be described as one.

     Caveat to report either way: the method inserts <sent_start>/<sent_end>
     marker tokens. Our checkpoint was fine-tuned with them; the stock arm gets
     randomly-initialised marker embeddings, because an off-the-shelf model has
     never seen them. That is inherent to applying this method to a stock
     encoder, not a handicap we imposed, but it means the comparison measures
     "fine-tuned with markers" vs "stock plus untrained markers".

E2 -- ADAPTIVE BUDGET. Is the learned per-example budget better than a fixed
     ratio? This is the only architectural delta over CPC still standing, and
     it has never been evaluated. The `auto` setting is scored against the
     encoder's OWN fixed-ratio curve, interpolated to the ratio auto actually
     achieved -- comparing it to a fixed ratio it did not hit would confound
     the compression axis with the quality axis.

Both stages beyond Stage 1 are disabled: Stage 2 was measured to sit on or
below the Stage-1 curve, and DAC lives inside Stage 2.
"""

from __future__ import annotations

import argparse
import json
import math
import sys
import time
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from evaluation.qa_metrics import aggregate, evidence_recall, score_prediction
from evaluation.reader_client import QwenReader, ReaderConfig
from evaluation.run_coverage_profile import profile_row
from evaluation.run_downstream_eval import gold_answers, gold_evidence, load_jsonl
from pipeline.compression_pipeline import ContextAwareCompressor, estimate_token_count
from pipeline.runtime_contract import EncoderContractError, resolve_encoder_source
from repro.manifest import build_manifest, write_manifest

# Winning configuration from the Stage-1 ablation: encoder similarity carries
# the whole sentence score.
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


def build(encoder_path: str, budget_formula: str) -> ContextAwareCompressor:
    return ContextAwareCompressor(
        encoder_dir=encoder_path,
        budget_formula_name=budget_formula,
        enable_second_stage=False,
        enable_dac=False,
        allow_heuristic_fallback=False,
        **ENCODER_ONLY,
    )


def run_setting(
    encoder_name: str,
    label: str,
    ratio: Optional[float],
    compressor,
    rows: Sequence[dict],
    batch_reader,
) -> Dict[str, object]:
    per_row: List[Dict[str, object]] = []
    pending = []
    for row in rows:
        question = " ".join(str(row.get("question") or "").split())
        context = " ".join(str(row.get("context") or "").split())
        if not question or not context:
            continue
        t0 = time.monotonic()
        if ratio is None:
            result = compressor.compress(question=question, context=context)
        else:
            result = compressor.compress(question=question, context=context, target_ratio=ratio)
        latency = time.monotonic() - t0
        compressed = str(result.get("stage1_context") or "")

        original = estimate_token_count(context)
        kept = estimate_token_count(compressed)
        record: Dict[str, object] = {
            "encoder": encoder_name,
            "setting": label,
            "id": row.get("id"),
            "dataset": row.get("dataset"),
            "achieved_ratio": kept / max(original, 1),
            "actual_cr": 100.0 * kept / max(original, 1),
            "compression_x": original / max(kept, 1),
            "compress_latency_s": latency,
            "evidence_recall": evidence_recall(gold_evidence(row), compressed),
            "compressed_context": compressed,
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
            "compression_x", "compress_latency_s"]
    summary = aggregate(per_row, keys)
    parts = [summary.get(k) for k in ("ans_cov", "support_cov", "hard_fact", "qa_ans_rate")]
    parts = [p for p in parts if p is not None]
    cr = summary.get("actual_cr") or 0.0
    summary["utility_per_token"] = (sum(parts) / len(parts)) / cr if parts and cr > 0 else None

    return {
        "encoder": encoder_name,
        "setting": label,
        "requested_ratio": ratio,
        "num_rows": len(per_row),
        "scored_rows": sum(1 for r in per_row if r.get("f1") is not None),
        "reader_failures": sum(1 for r in per_row if r.get("reader_ok") is False),
        "summary": summary,
        "rows": per_row,
    }


def interpolate_curve(points: List[Tuple[float, float]], x: float) -> Optional[float]:
    """Expected F1 at achieved-ratio `x`, interpolating in log-compression space.

    Compression quality falls roughly linearly in log2(compression factor), so
    interpolating there rather than in raw ratio avoids overstating the curve
    between widely spaced operating points.
    """
    pts = sorted((math.log2(1.0 / r), f) for r, f in points if r > 0)
    if len(pts) < 2:
        return None
    target = math.log2(1.0 / x) if x > 0 else None
    if target is None:
        return None
    if target <= pts[0][0]:
        (x0, y0), (x1, y1) = pts[0], pts[1]
    elif target >= pts[-1][0]:
        (x0, y0), (x1, y1) = pts[-2], pts[-1]
    else:
        for i in range(len(pts) - 1):
            if pts[i][0] <= target <= pts[i + 1][0]:
                (x0, y0), (x1, y1) = pts[i], pts[i + 1]
                break
    if x1 == x0:
        return y0
    return y0 + (y1 - y0) * (target - x0) / (x1 - x0)


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--input_file", type=str, required=True)
    p.add_argument("--output_dir", type=str,
                   default=str(PROJECT_ROOT / "evaluation" / "outputs_encoder_budget_study"))
    p.add_argument("--ours_encoder_dir", type=str, required=True)
    p.add_argument("--stock_encoder_dir", type=str, required=True)
    p.add_argument("--budget_formula_name", type=str, default="entropy_spread")
    p.add_argument("--ratios", type=str, default="0.5,0.25,0.125")
    p.add_argument("--limit", type=int, default=0)
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

    encoders = {"ours": args.ours_encoder_dir, "stock": args.stock_encoder_dir}
    for name, path in encoders.items():
        try:
            resolve_encoder_source(path, False)
        except EncoderContractError as exc:
            raise SystemExit(f"\nENCODER CONTRACT VIOLATION for {name}\n{exc}\n")

    rows = load_jsonl(Path(args.input_file))
    if args.limit > 0:
        rows = rows[: args.limit]
    print(f"rows={len(rows)} ratios={ratios} + auto", flush=True)

    results = []
    for enc_name, enc_path in encoders.items():
        compressor = build(enc_path, args.budget_formula_name)
        settings: List[Tuple[str, Optional[float]]] = [("auto", None)]
        settings += [(f"{r:.3f}", r) for r in ratios]
        for label, ratio in settings:
            res = run_setting(enc_name, label, ratio, compressor, rows, batch_reader)
            results.append(res)
            s = res["summary"]

            def g(k, places=1):
                v = s.get(k)
                return "n/a" if v is None else f"{v:.{places}f}"

            print(
                f"  {enc_name:<6} {label:<7} CR={g('actual_cr')} AnsCov={g('ans_cov')} "
                f"SupCov={g('support_cov')} Hard={g('hard_fact')} QAAns={g('qa_ans_rate')} "
                f"Util/Tok={g('utility_per_token', 3)} | EM={g('em', 3)} F1={g('f1', 3)} "
                f"lat={g('compress_latency_s', 2)}s",
                flush=True,
            )
        del compressor

    # ---- E1: encoder provenance ----
    print("\n=== E1: ours vs stock (matched settings) ===")
    for label in ["auto"] + [f"{r:.3f}" for r in ratios]:
        o = next((r for r in results if r["encoder"] == "ours" and r["setting"] == label), None)
        s_ = next((r for r in results if r["encoder"] == "stock" and r["setting"] == label), None)
        if not o or not s_:
            continue
        do, ds = o["summary"], s_["summary"]
        print(f"  {label:<7} ours F1={do['f1']:.3f} (CR {do['actual_cr']:.1f})  "
              f"stock F1={ds['f1']:.3f} (CR {ds['actual_cr']:.1f})  "
              f"delta={do['f1'] - ds['f1']:+.3f}")

    # ---- E2: adaptive budget vs the encoder's own fixed-ratio curve ----
    print("\n=== E2: adaptive budget vs fixed-ratio curve ===")
    for enc_name in encoders:
        fixed = [
            (r["summary"]["achieved_ratio"], r["summary"]["f1"])
            for r in results
            if r["encoder"] == enc_name and r["setting"] != "auto"
            and r["summary"]["f1"] is not None
        ]
        auto = next((r for r in results if r["encoder"] == enc_name and r["setting"] == "auto"), None)
        if not auto or len(fixed) < 2:
            continue
        a = auto["summary"]
        expected = interpolate_curve(fixed, a["achieved_ratio"])
        if expected is None:
            continue
        delta = a["f1"] - expected
        verdict = "ABOVE curve (adaptive helps)" if delta > 0.005 else (
            "BELOW curve (adaptive hurts)" if delta < -0.005 else "ON curve (no effect)")
        print(f"  {enc_name:<6} auto achieved={a['achieved_ratio']:.3f} "
              f"F1={a['f1']:.3f}  curve_expected={expected:.3f}  delta={delta:+.3f}  -> {verdict}")

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    write_manifest(
        out_dir / "manifest.json",
        build_manifest(
            seeds={"seeds": [0]},
            datasets=[Path(args.input_file)],
            config=vars(args),
            provenance={"encoders": encoders, "fusion": "encoder_only"},
            extra={
                "reader": reader.config.public_dict(),
                "note": "stock arm has randomly-initialised marker embeddings; it has "
                        "never seen <sent_start>/<sent_end>",
            },
        ),
    )
    (out_dir / "summary.json").write_text(
        json.dumps({"results": [{k: v for k, v in r.items() if k != "rows"} for r in results]},
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
