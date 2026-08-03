"""A/B test of single-pass sentence scoring against the marked-window path.

The marked-window path runs one encoder forward pass per sentence over a ~900
character window, so an N-sentence context costs roughly N * 250 token-positions
while the context itself is only L. Single-pass scoring encodes the context once
(chunked with 50% overlap when it exceeds the model window) and mean-pools each
sentence's token span, costing about L. It also conditions each sentence on the
FULL context rather than a fixed-width neighbourhood.

Risk worth stating: our encoder was fine-tuned WITH marker tokens, so span
pooling is mildly off-distribution for it. That is exactly what this measures --
if quality drops, the finding is that marker-based training does not transfer to
span pooling, which is informative either way.

Reports F1 and latency together, since a speedup that costs accuracy is only
interesting if the trade is favourable on the accuracy/compression curve.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from evaluation.reader_client import QwenReader, ReaderConfig
from evaluation.run_bridge_ablation import ENCODER_ONLY, run_arm
from evaluation.run_downstream_eval import load_jsonl
from pipeline.compression_pipeline import ContextAwareCompressor
from pipeline.runtime_contract import EncoderContractError, resolve_encoder_source


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--input_file", type=str, required=True)
    p.add_argument("--output_dir", type=str,
                   default=str(PROJECT_ROOT / "evaluation" / "outputs_singlepass_ablation"))
    p.add_argument("--encoder_dir", type=str, required=True)
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

    try:
        resolved = resolve_encoder_source(args.encoder_dir, False)
    except EncoderContractError as exc:
        raise SystemExit(f"\nENCODER CONTRACT VIOLATION\n{exc}\n")

    rows = load_jsonl(Path(args.input_file))
    if args.limit > 0:
        rows = rows[: args.limit]
    print(f"rows={len(rows)} ratios={ratios}", flush=True)

    results = []
    for single_pass in (False, True):
        compressor = ContextAwareCompressor(
            encoder_dir=resolved.path,
            budget_formula_name=args.budget_formula_name,
            enable_second_stage=False,
            enable_dac=False,
            allow_heuristic_fallback=False,
            single_pass_scoring=single_pass,
            **ENCODER_ONLY,
        )
        for ratio in ratios:
            name = f"{'single_pass' if single_pass else 'marked_window'}@{ratio:.3f}"
            res = run_arm(name, compressor, rows, ratio, batch_reader)
            results.append(res)
            s = res["summary"]
            print(
                f"  {name:<22} CR={s['actual_cr']:.1f} EM={s['em']:.3f} F1={s['f1']:.3f} "
                f"AnsCov={s['ans_cov']:.1f} QAAns={s['qa_ans_rate']:.1f} "
                f"allSupport={s['all_support_retained']:.3f} lat={s['compress_latency_s']:.3f}s",
                flush=True,
            )
        del compressor

    print("\n=== single-pass vs marked-window (matched target) ===")
    for ratio in ratios:
        b = next(r for r in results if r["arm"] == f"marked_window@{ratio:.3f}")
        v = next(r for r in results if r["arm"] == f"single_pass@{ratio:.3f}")
        bs, vs = b["summary"], v["summary"]
        speed = bs["compress_latency_s"] / max(vs["compress_latency_s"], 1e-9)
        print(f"  ratio {ratio:.3f}: F1 {bs['f1']:.3f} -> {vs['f1']:.3f} ({vs['f1'] - bs['f1']:+.3f})   "
              f"CR {bs['actual_cr']:.1f} -> {vs['actual_cr']:.1f}   "
              f"latency {bs['compress_latency_s']:.3f}s -> {vs['compress_latency_s']:.3f}s ({speed:.2f}x)")

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
