# Code Modification Specification — Step 1 of the remediation plan

Implements the first (blocking) tier of `EVAL_VALIDITY_AUDIT.md`: make the system
under test *identifiable*, make evaluation entry points *fail loudly*, and make
every run *reproducible*. Until this tier is in place, no downstream measurement
is interpretable, because a silent lexical fallback could be producing the numbers.

Status legend: `[x]` implemented in this change · `[ ]` specified, not yet built.

---

## 1. Encoder checkpoint loading contract

**New module:** `pipeline/runtime_contract.py`

| Component | Purpose |
|---|---|
| `EncoderContractError` | Fatal error for an unhonourable backend request. |
| `resolve_encoder_source(requested, allow_lexical_fallback)` | Validates *before* any model is built. Raises unless the request can be honoured exactly, or the lexical backend was explicitly opted into. |
| `describe_checkpoint_problem(path)` | Actionable reason a path is not a usable checkpoint (missing `encoder_config.json`, no weight files, etc.). Accepts a bare HF snapshot (`config.json` + weights). |
| `checkpoint_fingerprint(source)` | SHA-256 per weight file plus a combined digest, so a number can be tied to exact weights. Files > 256 MB use `head-8MB + size`; the mode is always recorded. |
| `RuntimeProvenance` | The permanent record: requested vs resolved backend, runtime, fallback flag + reason, load error, checkpoint fingerprint, span/budget model dirs. |
| `RuntimeProvenance.system_label()` | Returns `LEXICAL-FALLBACK(not-the-neural-system)` for any lexical run, so such numbers cannot be silently pasted into a results table. |
| `RuntimeProvenance.assert_neural()` | Hard gate for anything that claims to be a headline result. |

**Invariants enforced**

- `[x]` Requesting the lexical backend (`lexical://fallback` or the legacy
  `lightweight_lexical_fallback`) **requires** explicit opt-in.
- `[x]` A requested-but-unusable checkpoint **raises**; it never degrades.
- `[x]` Whatever ran is recorded and is embeddable in every results file.

**Verified:** all three no-opt-in paths raise (`no encoder specified`, legacy magic
string, nonexistent path); opt-in resolves and is labelled lexical; `assert_neural`
blocks lexical provenance.

## 2. Fail-loud evaluation entry points

| File | Change |
|---|---|
| `pipeline/compression_pipeline.py` | `[x]` Lexical branch now raises unless `allow_heuristic_fallback=True`. `[x]` New `ContextAwareCompressor.provenance()` returns a `RuntimeProvenance`. Constructor default was already `False`; callers were the problem. |
| `main.py` | `[x]` The silent 3-way chain (local ckpt → Qwen snapshot → lexical) now **exits with instructions** when no checkpoint is found. Opt in with `ALLOW_LEXICAL_FALLBACK=1`. `[x]` Prints `system_label`. |
| `intra_sentence_model/evaluate_stage2_english_benchmark.py` | `[x]` `--encoder_dir` added; hardcoded `"lightweight_lexical_fallback"` removed. `[x]` `--allow_lexical_fallback` (default **off**). `[x]` `--context_mode` default flipped `candidate` → **`full`** (finding C3), with a printed warning when `candidate` is used. `[x]` Proxy metrics renamed `qa_*` → `lexical_proxy_*`. `[x]` Fail-closed: missing coverage was `True`, now `False`. `[x]` Provenance + manifest written to output. |
| `intra_sentence_model/train_span_model.py` | `[x]` `--split_mode group` (**new default**) uses group-disjoint splitting; `span` reproduces the legacy leaky split behind a printed warning. |
| `intra_sentence_model/span_dataset.py` | `[x]` `group_key`, `split_rows_group_disjoint`, `build_xy_group_disjoint` — split by `source_id` **before** flattening spans, with an assertion that no group straddles the split. |

**Verified:** group-disjoint split on synthetic data (20 sources × 3 sentences × 3
spans) → 16 train / 4 dev groups, **zero overlap**, deterministic across runs,
144/36 spans. The `judge_answerability` fail-closed change is a deterministic
three-line `True`→`False` edit confirmed by diff (its module imports `torch`,
which is absent locally, so it was not executed here).

### Metric naming discipline (now enforced by name)

| Prefix | Meaning |
|---|---|
| `em`, `f1`, `rouge_l` | Genuine downstream reader accuracy. |
| `evidence_recall` | Gold evidence retention in the compressed context. |
| `lexical_proxy_*` | Token-survival **proxy**. Diagnostic only; never a headline claim. |

## 3. Reproducibility manifest

**New module:** `repro/manifest.py` (also runnable: `python -m repro.manifest`)

`[x]` Captures git commit/branch/**dirty flag** + dirty file list, installed
versions of `torch`/`transformers`/`numpy`/`scikit-learn`/`openai`, platform +
CUDA/GPU state, seeds, per-dataset SHA-256 + line counts, full resolved config,
and the `RuntimeProvenance`. Secrets are recorded only as
`DASHSCOPE_API_KEY_present: true|false` — never the value.

`[x]` Written by `evaluate_stage2_english_benchmark.py` and
`evaluation/run_downstream_eval.py`.
`[x]` `requirements.txt` added (was absent entirely — finding M2).

## 4. Real downstream evaluation harness (fixes C2, enables H2/M1)

**New package:** `evaluation/`

| File | Role |
|---|---|
| `env_loader.py` | Dependency-free `.env` loading. Exported env vars win over the file. Never logs values. |
| `qa_metrics.py` | SQuAD-normalised EM, token-F1, LCS ROUGE-L, multi-reference max, `evidence_recall`, multi-seed `std`. |
| `reader_client.py` | `QwenReader` over the Bailian/DashScope OpenAI-compatible endpoint. Key from `DASHSCOPE_API_KEY` **only**. `temperature=0`. One fixed prompt for all methods. Retries with jittered backoff. `smoke_test()` validates credentials + model id in ~3 s before a long sweep. |
| `run_downstream_eval.py` | The real benchmark: full unmodified contexts, matched ratios, one frozen reader, multi-seed. |

**Method registry** (`COMPRESSORS`): `none` (upper bound), `truncate`, `random`
(lower bounds), `topk_lexical`, `ours_stage1`, `ours_full`, `ours_auto_budget`.

**Verified:** EM/F1/ROUGE-L/multi-ref/evidence-recall all produce correct values
on hand-checked cases.

## 4b. Oracle-feature removal (C5) — implemented

`[x]` `answer_overlap` and `answer_drop` removed from `FEATURE_ORDER`
(31 → **29** features). They are still computed and stored in the feature dict for
label construction and analysis, but `features_to_vector()` projects only onto
`FEATURE_ORDER`, so they can no longer reach the model.

`[x]` `ORACLE_ONLY_FEATURES` + `assert_no_oracle_features()` guard, called at the
top of span training, so reintroduction fails immediately.

`[x]` `FEATURE_SCHEMA_VERSION = 2`, written into checkpoint metadata.
`load_span_model()` now **rejects** any checkpoint whose `feature_order` or
`input_dim` disagrees with the current code, listing the exact differing features.
Previously a v1 (oracle-trained, 31-dim) checkpoint would load and read every
feature at the wrong index, producing confident but meaningless scores.

`[x]` `DynamicSpanCompressor._load_trained_span_model()` no longer swallows
exceptions. An explicitly requested span model that fails to load used to silently
revert to the rule-based pruner, so a "learned" run could be entirely heuristic —
the same defect class as C1/H3. It now raises.

**Verified:** oracle values 0.99/0.88 provably absent from the 29-dim vector while
still present in the dict; guard fires on reintroduction; a simulated v1
checkpoint is rejected naming `['answer_overlap','answer_drop']`; `input_dim`
mismatch rejected.

**Consequence:** all existing span checkpoints are invalid and must be retrained
with `--split_mode group`. This is intended — they were trained on leaked splits
with oracle features.

## 5. Not yet done — next tier, in order
- `[ ]` **H1 — replace lexical pseudo-labels** with reader log-likelihood deltas
  (see `BRIEF_PRO_NOTES.md`, "Helpfulness"). Removes the circularity between
  labels and the metric.
- `[ ]` **H2 — external baselines**: BM25 (real IDF), embedding Top-K, CPC,
  LLMLingua-2, LongLLMLingua, RECOMP, EXIT, DAC. Plug into `COMPRESSORS`.
- `[ ]` **Efficiency accounting**: compressor FLOPs/latency reported separately
  from reader cost. This is where a cheap extractive method should win.
- `[ ]` **M3 — regenerate `project_technical_report.md`**, which still documents
  two files that do not exist (`build_cqr_with_filters.py`, `clean_cqr_dataset.py`).

## 6. Reproducing

```bash
pip install -r requirements.txt
cp .env.example .env        # then add your rotated DASHSCOPE_API_KEY

cd "1 Prompt Compression with Context-Aware Sentence Encoding for Fast and"
python -m repro.manifest                                    # environment record
python -m evaluation.run_downstream_eval --smoke_test_only   # verify reader + model id

python -m evaluation.run_downstream_eval \
    --input_file data_builder/english_cqr_mixed_5k/test.jsonl \
    --encoder_dir context_aware_encoder_model/outputs_english/stage2_full \
    --methods none,truncate,topk_lexical,ours_stage1,ours_full \
    --ratios 0.5,0.25,0.125 --limit 200 --seeds 42,43,44
```

Omitting `--encoder_dir` now fails with instructions instead of quietly measuring
the lexical heuristic. That single behavioural change is the point of this tier.
