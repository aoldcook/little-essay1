"""Real downstream QA evaluation of context compression (audit findings C2/C3/H2/H3/M1).

What this fixes relative to the legacy Stage-2 benchmark:

  C2  Scores compressed contexts with a FROZEN READER LLM (EM/F1/ROUGE-L), not
      lexical token coverage. Coverage survives only as a labelled diagnostic.
  C3  Uses FULL, UNMODIFIED contexts by default. Oracle-seeded contexts are not
      constructible here at all; evidence retention is measured, never assumed.
      (BRIEF-Pro, Findings ACL 2026, Table 4 reports the same effect: expanding
      only oracle documents yields artificially clean contexts that overestimate
      performance.)
  H2  Runs several methods through the SAME reader and SAME prompt at MATCHED
      compression ratios, including a no-compression upper bound and trivial
      lower bounds. New compressors plug into COMPRESSORS.
  H3  Refuses to run unless the encoder backend was honoured exactly, or the
      lexical fallback was explicitly opted into and labelled as such.
  M1  Accepts --seeds for multi-seed runs and reports mean +/- std.

Example:
    python -m evaluation.run_downstream_eval \
        --input_file data_builder/english_cqr_mixed_5k/test.jsonl \
        --encoder_dir context_aware_encoder_model/outputs_english/stage2_full \
        --methods none,truncate,topk_lexical,ours_stage1,ours_full \
        --ratios 0.5,0.25,0.125 --limit 200 --seeds 42,43,44
"""

from __future__ import annotations

import argparse
import json
import random
import sys
import time
from pathlib import Path
from typing import Callable, Dict, List, Optional, Sequence

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from data_builder.build_english_cqr_dataset import split_sentences
from evaluation.qa_metrics import (
    TOKEN_RE,
    aggregate,
    evidence_recall,
    score_prediction,
    std,
)
from evaluation.reader_client import QwenReader, ReaderConfig
from pipeline.compression_pipeline import ContextAwareCompressor, estimate_token_count
from pipeline.runtime_contract import (
    EncoderContractError,
    RuntimeProvenance,
    checkpoint_fingerprint,
    resolve_encoder_source,
)
from repro.manifest import build_manifest, write_manifest

QUESTION_STOPWORDS = {
    "a", "an", "and", "are", "as", "at", "be", "by", "can", "could", "do", "does",
    "did", "for", "from", "how", "in", "is", "it", "of", "on", "or", "should",
    "that", "the", "to", "was", "were", "what", "when", "where", "which", "who",
    "why", "with",
}


# --------------------------------------------------------------------------
# Compression methods. Every method maps (question, context, ratio) -> text.
# --------------------------------------------------------------------------

def compress_none(question: str, context: str, ratio: float, ctx: Dict) -> str:
    """No compression: the upper bound on answer quality and on token cost."""
    return context


def compress_truncate(question: str, context: str, ratio: float, ctx: Dict) -> str:
    """Keep the leading sentences up to the budget. Trivial lower bound."""
    sentences = split_sentences(context)
    budget = max(1, int(round(sum(estimate_token_count(s) for s in sentences) * ratio)))
    kept, used = [], 0
    for sentence in sentences:
        cost = estimate_token_count(sentence)
        if used + cost > budget and kept:
            break
        kept.append(sentence)
        used += cost
    return " ".join(kept)


def compress_random(question: str, context: str, ratio: float, ctx: Dict) -> str:
    """Random sentences to budget, original order. Controls for 'any text helps'."""
    sentences = split_sentences(context)
    budget = max(1, int(round(sum(estimate_token_count(s) for s in sentences) * ratio)))
    rng: random.Random = ctx["rng"]
    order = list(range(len(sentences)))
    rng.shuffle(order)
    chosen, used = set(), 0
    for idx in order:
        cost = estimate_token_count(sentences[idx])
        if used + cost > budget and chosen:
            continue
        chosen.add(idx)
        used += cost
        if used >= budget:
            break
    return " ".join(sentences[i] for i in sorted(chosen))


def compress_topk_lexical(question: str, context: str, ratio: float, ctx: Dict) -> str:
    """Query-term-overlap sentence retrieval to budget. Cheap retrieval baseline.

    Stands in for BM25 Top-K; a true BM25 with corpus IDF should be added before
    publication (see H2 in the audit).
    """
    sentences = split_sentences(context)
    q_terms = {
        t.lower() for t in TOKEN_RE.findall(question)
        if t.lower() not in QUESTION_STOPWORDS
    }
    def score(sentence: str) -> float:
        s_terms = {t.lower() for t in TOKEN_RE.findall(sentence)}
        return len(q_terms & s_terms) / max(len(q_terms), 1) if q_terms else 0.0

    budget = max(1, int(round(sum(estimate_token_count(s) for s in sentences) * ratio)))
    ranked = sorted(range(len(sentences)), key=lambda i: score(sentences[i]), reverse=True)
    chosen, used = set(), 0
    for idx in ranked:
        cost = estimate_token_count(sentences[idx])
        if used + cost > budget and chosen:
            continue
        chosen.add(idx)
        used += cost
        if used >= budget:
            break
    return " ".join(sentences[i] for i in sorted(chosen))


def compress_dac_baseline(question: str, context: str, ratio: float, ctx: Dict) -> str:
    """Official DAC (ACL 2025), re-implemented. Task-agnostic: ignores the question.

    This is the published method as a baseline, NOT our question-aware salience
    feature. Keeping them separate is what lets the results table say honestly
    what DAC achieves versus what our adaptation of it achieves.
    """
    compressor = ctx.get("dac_baseline")
    if compressor is None:
        raise RuntimeError(
            "method 'dac' requires the DAC baseline compressor, which was not "
            "constructed. This is a wiring bug, not a runtime condition."
        )
    return compressor.compress(context=context, keep_ratio=ratio)


def compress_ours_stage1(question: str, context: str, ratio: float, ctx: Dict) -> str:
    result = ctx["compressor_stage1"].compress(
        question=question, context=context, target_ratio=ratio
    )
    return str(result.get("stage1_context") or "")


def compress_ours_full(question: str, context: str, ratio: float, ctx: Dict) -> str:
    result = ctx["compressor_full"].compress(
        question=question, context=context, target_ratio=ratio
    )
    return str(result.get("compressed_context") or "")


def compress_ours_auto_budget(question: str, context: str, ratio: float, ctx: Dict) -> str:
    """Our adaptive budget decides the ratio itself (ratio argument ignored).

    Reported at its own achieved ratio, so it must be compared on the
    accuracy-vs-achieved-ratio curve rather than at a matched ratio.
    """
    result = ctx["compressor_full"].compress(question=question, context=context)
    return str(result.get("compressed_context") or "")


COMPRESSORS: Dict[str, Callable[[str, str, float, Dict], str]] = {
    "none": compress_none,
    "truncate": compress_truncate,
    "random": compress_random,
    "topk_lexical": compress_topk_lexical,
    "dac": compress_dac_baseline,
    "ours_stage1": compress_ours_stage1,
    "ours_full": compress_ours_full,
    "ours_auto_budget": compress_ours_auto_budget,
}
# Methods whose achieved ratio is self-determined; not run per requested ratio.
SELF_BUDGETED = {"none", "ours_auto_budget"}


# --------------------------------------------------------------------------

def load_jsonl(path: Path) -> List[dict]:
    rows = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def gold_answers(row: dict) -> List[str]:
    answers = row.get("answers")
    if isinstance(answers, list) and answers:
        return [str(a) for a in answers if str(a).strip()]
    answer = str(row.get("answer") or "").strip()
    return [answer] if answer else []


def gold_evidence(row: dict) -> List[str]:
    evidence = [
        str(s) for s in (row.get("supporting_sentences") or []) if str(s).strip()
    ]
    positive = str(row.get("positive_sentence") or "").strip()
    if positive and positive not in evidence:
        evidence.append(positive)
    return evidence


def build_compressors(args: argparse.Namespace, resolved) -> Dict[str, object]:
    """Construct our compressor twice: Stage-1 only, and Stage-1+2."""
    encoder_arg = (
        "lightweight_lexical_fallback" if resolved.is_lexical else resolved.path
    )
    common = dict(
        encoder_dir=encoder_arg,
        budget_formula_name=args.budget_formula_name,
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
        # Fallback is pre-authorised only because resolve_encoder_source already
        # enforced the policy above; the contract decision is made once, there.
        allow_heuristic_fallback=resolved.is_lexical,
        enable_dac=not args.disable_dac,
        dac_salience_backend=args.dac_salience_backend,
        dac_salience_model=args.dac_salience_model,
        dac_fusion=args.dac_fusion,
        dac_alpha=args.dac_alpha,
        dac_require_attention=args.dac_require_attention,
        dac_strict=args.dac_strict,
    )
    stage1 = ContextAwareCompressor(enable_second_stage=False, **common)
    full = ContextAwareCompressor(
        enable_second_stage=True,
        second_stage_keep_ratio=args.second_stage_keep_ratio,
        second_stage_min_keep_ratio=args.second_stage_min_keep_ratio,
        second_stage_max_keep_ratio=args.second_stage_max_keep_ratio,
        span_model_dir=args.span_model_dir,
        **common,
    )
    return {"compressor_stage1": stage1, "compressor_full": full}


def dac_provenance(ctx: Dict) -> Optional[Dict[str, object]]:
    """Whether the DAC salience signal was actually live during the run.

    Recorded in the manifest so a "DAC-guided" result can never be reported
    without evidence that DAC loaded (audit finding D1: it used to disable
    itself silently and no artefact showed that).
    """
    compressor = ctx.get("compressor_full")
    span_compressor = getattr(compressor, "span_compressor", None)
    adapter = getattr(span_compressor, "dac_adapter", None)
    if adapter is None:
        return None
    record = dict(adapter.provenance())
    record["span_offset_mismatch_seen"] = getattr(
        span_compressor, "dac_offset_mismatch", None
    )
    return record


def evaluate_method(
    method: str,
    ratio: Optional[float],
    rows: Sequence[dict],
    reader: Optional[QwenReader],
    ctx: Dict,
) -> Dict[str, object]:
    fn = COMPRESSORS[method]
    per_row: List[Dict[str, object]] = []

    # Phase 1: compress everything. Compression latency is measured per row here,
    # where it is a property of the method; reader latency is a property of the
    # API and is reported in aggregate below rather than per row.
    pending: List[Dict[str, object]] = []
    for row in rows:
        question = " ".join(str(row.get("question") or "").split())
        context = " ".join(str(row.get("context") or "").split())
        refs = gold_answers(row)
        if not question or not context:
            continue

        t0 = time.monotonic()
        compressed = fn(question, context, ratio if ratio is not None else 1.0, ctx)
        compress_latency = time.monotonic() - t0

        original_tokens = estimate_token_count(context)
        compressed_tokens = estimate_token_count(compressed)

        record: Dict[str, object] = {
            "id": row.get("id"),
            "dataset": row.get("dataset"),
            "method": method,
            "requested_ratio": ratio,
            "original_tokens": original_tokens,
            "compressed_tokens": compressed_tokens,
            "achieved_ratio": compressed_tokens / max(original_tokens, 1),
            "compression_x": original_tokens / max(compressed_tokens, 1),
            "compress_latency_s": compress_latency,
            "evidence_recall": evidence_recall(gold_evidence(row), compressed),
            "has_reference": bool(refs),
        }
        per_row.append(record)
        pending.append({"record": record, "question": question,
                        "compressed": compressed, "refs": refs})

    # Phase 2: one batched, cached, concurrent pass over the reader. Serially this
    # was ~1013 calls per method-ratio cell at ~0.6 s each, i.e. hours per seed.
    # The reader is deterministic at temperature 0, so batching changes nothing
    # about the results.
    batch_reader = ctx.get("batch_reader")
    if reader is not None and batch_reader is not None:
        scored = [item for item in pending if item["refs"]]
        if scored:
            t0 = time.monotonic()
            answers = batch_reader.answer_many(
                [(item["question"], item["compressed"]) for item in scored]
            )
            reader_wall = time.monotonic() - t0
            for item, answer in zip(scored, answers):
                record = item["record"]
                record["prediction"] = answer
                ok = bool(str(answer).strip())
                record["reader_ok"] = ok
                if ok:
                    record.update(score_prediction(str(answer), item["refs"]))
                else:
                    # An empty answer means the API call failed after retries.
                    # Scoring it as a wrong answer would silently depress this
                    # method's numbers in proportion to how flaky the network
                    # was. Leave the metrics None so aggregate() skips the row,
                    # and surface the count instead.
                    record["em"] = record["f1"] = record["rouge_l"] = None
            for record in (item["record"] for item in scored):
                record["reader_batch_wall_s"] = reader_wall / max(len(scored), 1)

    metric_keys = [
        "em", "f1", "rouge_l", "evidence_recall", "achieved_ratio",
        "compression_x", "compress_latency_s", "reader_batch_wall_s",
        "reader_prompt_tokens", "compressed_tokens",
    ]
    # Per-dataset breakdown is mandatory, not optional. This pool mixes extractive
    # QA (HotpotQA, 2wikimqa: median gold 2 words) with sentence-response tasks
    # (cqr_official, wikitext103_pseudo_cqr: median gold ~20 words) while the
    # reader prompt asks for a short span. A pooled EM/F1 averages a task the
    # setup fits with one it structurally cannot score, and means nothing.
    datasets = sorted({str(r.get("dataset")) for r in per_row})
    by_dataset = {
        name: {
            "num_rows": sum(1 for r in per_row if str(r.get("dataset")) == name),
            **aggregate([r for r in per_row if str(r.get("dataset")) == name], metric_keys),
        }
        for name in datasets
    }

    reader_failures = sum(1 for r in per_row if r.get("reader_ok") is False)
    return {
        "method": method,
        "requested_ratio": ratio,
        "num_rows": len(per_row),
        "reader_failures": reader_failures,
        "scored_rows": sum(1 for r in per_row if r.get("f1") is not None),
        "summary": aggregate(per_row, metric_keys),
        "by_dataset": by_dataset,
        "rows": per_row,
    }


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Downstream QA evaluation of context compression with a frozen reader LLM."
    )
    p.add_argument("--input_file", type=str, default=None,
                   help="JSONL evaluation set. Required unless --smoke_test_only.")
    p.add_argument("--output_dir", type=str, default=str(PROJECT_ROOT / "evaluation" / "outputs_downstream"))
    p.add_argument("--encoder_dir", type=str, default=None,
                   help="Path to a trained context-aware encoder checkpoint.")
    p.add_argument("--allow_lexical_fallback", action="store_true",
                   help="Explicitly run the NON-NEURAL lexical baseline. Results are "
                        "labelled as such and must never be reported as the main system.")
    p.add_argument("--span_model_dir", type=str, default=None)
    p.add_argument("--budget_formula_name", type=str, default="entropy_spread")
    p.add_argument("--methods", type=str, default="none,truncate,topk_lexical,ours_full")
    p.add_argument("--ratios", type=str, default="0.5,0.25,0.125",
                   help="Target keep ratios (0.5=2x, 0.25=4x, 0.125=8x compression).")
    p.add_argument("--seeds", type=str, default="42")
    p.add_argument("--limit", type=int, default=200)
    p.add_argument("--second_stage_keep_ratio", type=float, default=0.52)
    p.add_argument("--second_stage_min_keep_ratio", type=float, default=0.34)
    p.add_argument("--second_stage_max_keep_ratio", type=float, default=0.72)
    p.add_argument("--reader_model", type=str, default=None)
    p.add_argument("--no_reader", action="store_true",
                   help="Skip the reader LLM (compression stats + evidence recall only). "
                        "Cannot produce EM/F1 and must not be used for headline claims.")
    p.add_argument("--smoke_test_only", action="store_true",
                   help="Verify reader credentials and model id, then exit.")
    p.add_argument("--reader_concurrency", type=int, default=16,
                   help="Parallel reader calls. The reader is deterministic at "
                        "temperature 0, so this changes throughput, not results.")
    p.add_argument("--reader_cache", type=str, default="",
                   help="JSONL cache of reader answers. Defaults to <output_dir>/reader_cache.jsonl. "
                        "Lets a re-run or an added seed reuse identical calls.")
    # ---- DAC token-salience signal (audit findings D1-D8) ----
    p.add_argument("--disable_dac", action="store_true",
                   help="Turn the DAC salience signal off deliberately (ablation arm). "
                        "Recorded as 'deliberately_disabled', distinct from a load failure.")
    p.add_argument("--dac_salience_backend", type=str, default="causal",
                   choices=["causal", "mlm"],
                   help="'causal' = single-pass shifted cross-entropy, faithful to DAC "
                        "(ACL 2025) and O(1) forwards. 'mlm' = per-token masking, "
                        "O(n) forwards, retained as an ablation arm only.")
    p.add_argument("--dac_salience_model", type=str, default=None,
                   help="LM providing token information content. Defaults to "
                        "Qwen/Qwen2-0.5B-Instruct for the causal backend and "
                        "roberta-base for mlm. A bare AutoModel checkpoint is "
                        "rejected when its head would be randomly initialised.")
    p.add_argument("--dac_fusion", type=str, default="additive",
                   choices=["additive", "multiplicative"])
    p.add_argument("--dac_alpha", type=float, default=0.8,
                   help="Weight on question-attention in additive fusion; (1-alpha) on MLM loss.")
    p.add_argument("--dac_require_attention", action="store_true",
                   help="Fail instead of renormalising onto the MLM term when "
                        "question-attention is unavailable.")
    p.add_argument("--dac_baseline_model", type=str, default="Qwen/Qwen2-0.5B-Instruct",
                   help="Causal LM for the task-agnostic DAC baseline (method 'dac'). "
                        "The reference implementation uses Qwen2-0.5B-Instruct.")
    p.add_argument("--dac_strict", action="store_true",
                   help="Raise if the DAC salience model cannot be loaded, instead of "
                        "printing a warning and continuing without it.")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    ratios = [float(r) for r in str(args.ratios).split(",") if str(r).strip()]
    seeds = [int(s) for s in str(args.seeds).split(",") if str(s).strip()]
    methods = [m.strip() for m in str(args.methods).split(",") if m.strip()]
    unknown = [m for m in methods if m not in COMPRESSORS]
    if unknown:
        raise SystemExit(f"unknown methods: {unknown}. available: {sorted(COMPRESSORS)}")

    # --smoke_test_only exits before any data is read, so --input_file is only
    # required for a real run. Validated here rather than by argparse so that
    # `--smoke_test_only` alone is a legal invocation.
    if args.smoke_test_only and args.no_reader:
        raise SystemExit(
            "--smoke_test_only and --no_reader are contradictory: the smoke test "
            "exists precisely to check that the reader is reachable."
        )
    if not args.smoke_test_only and not args.input_file:
        raise SystemExit(
            "--input_file is required for an evaluation run "
            "(omit it only when passing --smoke_test_only)."
        )

    # ---- Reader first: fail in seconds, not hours. ----
    reader = None
    if not args.no_reader:
        reader = QwenReader(ReaderConfig.from_env(model=args.reader_model))
        smoke = reader.smoke_test()
        print("reader_smoke_test:", json.dumps(smoke, ensure_ascii=False))
        if not smoke["reachable"]:
            raise SystemExit(
                f"Reader unreachable (model={smoke['model']}).\n"
                f"error: {smoke['error']}\n"
                "Check DASHSCOPE_API_KEY, DASHSCOPE_BASE_URL, and that the model id "
                "exists in your Bailian account."
            )
    if args.smoke_test_only:
        return

    # ---- Encoder contract: no silent degradation. ----
    needs_encoder = any(m.startswith("ours") for m in methods)
    resolved = None
    ctx: Dict[str, object] = {}
    provenance = RuntimeProvenance(
        encoder_kind="not_required",
        encoder_requested=str(args.encoder_dir),
    )
    if needs_encoder:
        try:
            resolved = resolve_encoder_source(args.encoder_dir, args.allow_lexical_fallback)
        except EncoderContractError as exc:
            raise SystemExit(f"\nENCODER CONTRACT VIOLATION\n{exc}\n")
        ctx.update(build_compressors(args, resolved))
        active = ctx["compressor_full"]
        provenance = RuntimeProvenance(
            encoder_kind=resolved.kind,
            encoder_requested=resolved.requested,
            encoder_path=resolved.path,
            encoder_runtime=getattr(active, "encoder_runtime", ""),
            lexical_fallback_used=resolved.is_lexical
            or getattr(active, "encoder_runtime", "") == "lightweight_lexical_fallback",
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
        if provenance.lexical_fallback_used:
            print(
                "\n*** WARNING: running the NON-NEURAL lexical fallback. These numbers "
                "do NOT come from the trained context-aware encoder and must be "
                "reported only as an ablation baseline. ***\n"
            )

    if reader is not None:
        from evaluation.cached_reader import CachedBatchReader

        cache_path = args.reader_cache or str(Path(args.output_dir) / "reader_cache.jsonl")
        ctx["batch_reader"] = CachedBatchReader(
            reader, cache_path=cache_path, concurrency=args.reader_concurrency
        )
        print("reader_batching:", json.dumps(ctx["batch_reader"].provenance(), ensure_ascii=False))

    if "dac" in methods:
        from pipeline.dac_baseline import DacBaselineCompressor, DacBaselineConfig

        ctx["dac_baseline"] = DacBaselineCompressor(
            DacBaselineConfig(
                model_name=args.dac_baseline_model,
                alpha=args.dac_alpha,
                fusion=args.dac_fusion,
            )
        )
        print("dac_baseline:", json.dumps(ctx["dac_baseline"].provenance(), ensure_ascii=False))

    rows = load_jsonl(Path(args.input_file))
    if args.limit > 0:
        rows = rows[: args.limit]

    # ---- Sweep ----
    results: List[Dict[str, object]] = []
    for seed in seeds:
        ctx["rng"] = random.Random(seed)
        random.seed(seed)
        for method in methods:
            method_ratios: List[Optional[float]] = (
                [None] if method in SELF_BUDGETED else list(ratios)
            )
            for ratio in method_ratios:
                res = evaluate_method(method, ratio, rows, reader, ctx)
                res["seed"] = seed
                results.append(res)
                s = res["summary"]
                print(
                    f"seed={seed} method={method:>16} ratio={ratio} "
                    f"achieved={_fmt(s['achieved_ratio'])} x={_fmt(s['compression_x'])} "
                    f"EM={_fmt(s['em'])} F1={_fmt(s['f1'])} "
                    f"evid={_fmt(s['evidence_recall'])} "
                    f"scored={res['scored_rows']} reader_fail={res['reader_failures']}",
                    flush=True,
                )
            if method == "dac" and ctx.get("dac_baseline") is not None:
                # Release the baseline's causal LM before the ours_* methods run.
                # Two ~0.5B models plus eager attention over long contexts does
                # not fit on a 24 GB card.
                ctx.pop("dac_baseline", None)
                import gc

                import torch as _torch

                gc.collect()
                if _torch.cuda.is_available():
                    _torch.cuda.empty_cache()

    # ---- Multi-seed aggregation (finding M1) ----
    grouped: Dict[str, List[Dict[str, object]]] = {}
    for res in results:
        grouped.setdefault(f"{res['method']}@{res['requested_ratio']}", []).append(res)

    across_seeds = {}
    for key, group in grouped.items():
        across_seeds[key] = {
            "num_seeds": len(group),
            "metrics": {
                metric: {
                    "mean": aggregate([g["summary"] for g in group], [metric])[metric],
                    "std": std([g["summary"][metric] for g in group]),
                }
                for metric in ("em", "f1", "rouge_l", "evidence_recall", "achieved_ratio", "compression_x")
            },
        }

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    manifest = build_manifest(
        seeds={"seeds": seeds},
        datasets=[Path(args.input_file)],
        config=vars(args),
        provenance=provenance.to_dict(),
        extra={
            "reader": reader.config.public_dict() if reader else None,
            "reader_stats": reader.stats.to_dict() if reader else None,
            "dac": dac_provenance(ctx),
            "metric_semantics": {
                "em/f1/rouge_l": "genuine downstream reader accuracy",
                "evidence_recall": "gold evidence retention in compressed context",
                "achieved_ratio": "compressed_tokens / original_tokens",
                "compression_x": "original_tokens / compressed_tokens",
            },
        },
    )
    write_manifest(out_dir / "manifest.json", manifest)

    summary = {
        "system_label": provenance.system_label(),
        "across_seeds": across_seeds,
        "per_run": [{k: v for k, v in r.items() if k != "rows"} for r in results],
    }
    (out_dir / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    with (out_dir / "per_row.jsonl").open("w", encoding="utf-8") as f:
        for res in results:
            for record in res["rows"]:
                record["seed"] = res["seed"]
                f.write(json.dumps(record, ensure_ascii=False) + "\n")

    print("\n" + json.dumps(summary["across_seeds"], ensure_ascii=False, indent=2))
    print("\nwrote:", out_dir / "summary.json")
    print("wrote:", out_dir / "manifest.json")
    print("wrote:", out_dir / "per_row.jsonl")


def _fmt(value: Optional[float]) -> str:
    return "n/a" if value is None else f"{value:.3f}"


if __name__ == "__main__":
    main()
