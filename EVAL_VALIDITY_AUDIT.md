# Evaluation Validity Audit — Stage 2 span compression & benchmark

**Scope.** This audit examines whether the empirical results the project can currently
produce are *valid* — i.e., whether they measure what they claim to measure — at the
standard expected by a journal or applied-NLP conference review. It is deliberately
adversarial: it assumes a skeptical reviewer and asks where each number would fall apart.

**Verdict.** As of this audit, **no result the repository can currently produce is
publishable as-is.** The problems are not polish issues; they are validity issues that
would each independently cause a reject at any venue with competent reviewers. The good
news is that the *architecture* is real and the fixes are well-scoped. Below, findings are
ranked by severity, each with exact code locations, why a reviewer rejects on it, and the
required fix.

Files audited: `intra_sentence_model/evaluate_stage2_english_benchmark.py`,
`intra_sentence_model/generate_span_pseudo_labels.py`,
`intra_sentence_model/span_feature_utils.py`,
`intra_sentence_model/train_span_model.py`,
`intra_sentence_model/span_dataset.py`,
`pipeline/task_aware_compression.py`,
`data_builder/cqr_split_utils.py`, `main.py`.

---

## CRITICAL findings

### C1 — The core neural encoder is never actually used in any runnable result
- **Where:** `main.py:41` (`encoder_source = "lightweight_lexical_fallback"`);
  `evaluate_stage2_english_benchmark.py:168` (`encoder_dir="lightweight_lexical_fallback"`);
  `compression_pipeline.py:433-456` (silent fallback path);
  no `*.pt` / `encoder_config.json` exists in the repo (all gitignored).
- **What it means:** The paper's centerpiece — a *context-aware sentence encoder* trained
  with contrastive + MNTP objectives — is not what produces any demo output or benchmark
  number. Every runnable path resolves to a lexical (token-overlap) heuristic. The
  "context-aware encoding" contribution is currently supported by zero runnable evidence.
- **Why a reviewer rejects:** The headline method and the evaluated system are different
  systems. Any claim attributed to the encoder is unfalsifiable from the artifact.
- **Fix:** Train and *check in* (or release via a model host) at least one real encoder
  checkpoint; make the benchmark load it; report the lexical fallback only as an explicit
  ablation baseline, never as the main system. Fail loudly if a requested checkpoint is
  missing — see C7.

### C2 — "QA answerability" is a lexical-coverage proxy, not answer accuracy
- **Where:** `judge_answerability()` `evaluate_stage2_english_benchmark.py:197-233`; helper
  `token_recall` (:93-98) and `content_token_recall` (:109-114).
- **What it means:** "Answerable" is decided by whether answer/evidence/question *tokens*
  survive in the compressed text, thresholded at 0.50/0.55/0.25 (:393-395). This measures
  word survival, not whether a reader can produce the correct answer. It can move in the
  *opposite* direction from real EM/F1 (e.g., keeping the answer noun while deleting the
  negation or the comparison operator scores as "answerable" but is unanswerable in fact).
- **Aggravating detail:** `answer_ok` and `question_ok` default to **True** when coverage is
  `None` (:215, :217). Missing answer/question ⇒ counted as answerable. This inflates the
  answerable rate by construction.
- **Why a reviewer rejects:** The paper's central promise is "answerability-preserving."
  The metric that certifies it does not measure answerability.
- **Fix:** Replace with real downstream accuracy: feed each compressed context to a frozen
  reader LLM (Bailian/DashScope OpenAI-compatible endpoint is fine) and score EM/F1/ROUGE
  against gold answers. Keep token-coverage only as a secondary diagnostic, clearly labeled
  as a proxy. Report evidence recall separately.

### C3 — Stage-2 benchmark is oracle-seeded (gold evidence guaranteed present)
- **Where:** `build_candidate_context()` `evaluate_stage2_english_benchmark.py:134-163`;
  default `--context_mode candidate` (:385); force-append of gold at :156-159.
- **What it means:** The context handed to the compressor is constructed to *guarantee* the
  gold `positive_sentence` and every `supporting_sentence` are present (lines 141-159 select
  wanted sentences, then re-append any that got dropped). The compressor is thus evaluated on
  inputs where the retrieval/selection problem has already been solved in its favor. A `full`
  mode exists (:411) but is not the default and appears unused in reported numbers.
- **Why a reviewer rejects:** This is testing on an oracle-cleaned distribution the deployed
  system never sees. Stage-2's apparent value is inflated by an unknown, likely large margin;
  Stage-1's selection is never actually stress-tested.
- **Fix:** Evaluate on **full, unmodified contexts** as the primary setting. Report the
  oracle-seeded setting, if at all, only as a labeled diagnostic ("Stage-2 in isolation given
  perfect Stage-1"). Evidence recall must be measured, not assumed.

### C4 — Span train/dev split leaks across spans of the same example
- **Where:** `train_span_model.py:24-31` (`split_train_dev` shuffles flat indices);
  `span_dataset.py:14-27, 47-55` (`build_xy` flattens *all spans from all rows* into one pool
  before the split).
- **What it means:** The split is performed at the span-instance level after flattening, so
  multiple spans from the *same sentence* and the *same source example* are distributed across
  train and dev. Dev is not independent of train; reported `dev_acc` is optimistic.
- **Contrast:** The project already has correct group-disjoint splitting at the CQR level —
  `cqr_split_utils.py:61-80` (`stable_group_id`) and `:146-176` (`split_rows_by_group`). The
  span trainer simply does not use it.
- **Why a reviewer rejects:** Standard information-leakage flaw; invalidates the only
  quantitative number the span model reports.
- **Fix:** Split by `source_id` **before** flattening spans. Reuse `stable_group_id` /
  `split_rows_by_group`. Guarantee no `source_id` appears in more than one split. Better:
  inherit the split assignment from the CQR-level train/dev/test so Stage-1 and Stage-2 share
  disjoint example groups.

### C5 — Oracle features at train time, zeroed at inference; label defined by unavailable signal
- **Where:** `FEATURE_ORDER` includes `answer_overlap`, `answer_drop`
  (`span_feature_utils.py:112-114`); these are computed from the **gold answer** during
  pseudo-labeling (`generate_span_pseudo_labels.py:429, 446`); at inference they are hardcoded
  to **0.0** (`task_aware_compression.py:380-381`).
- **What it means:** Two input features are informative in training and structurally absent
  (constant 0) at test time — a train/test feature-distribution mismatch. Worse, the *label*
  itself is dominated by answer-derived signal: in the `feedback` policy
  (`generate_span_pseudo_labels.py:271-312`) the keep-score is
  `0.34*qa_feedback_drop + 0.20*answer_overlap + ...` — ~54% of the target comes from
  quantities computed with the gold answer, which the model cannot observe at inference.
- **Why a reviewer rejects:** The model is trained to predict a target defined mostly by
  information it will never have. Learned weights on the oracle features are meaningless at
  deployment; the classifier is partly fitting noise. This also silently overstates any
  offline accuracy.
- **Fix:** Remove `answer_overlap`/`answer_drop` from the feature vector entirely (they are
  never available at inference). Keep answer-derived quantities *only* as pseudo-label
  supervision, and make that explicit. Re-train and re-report. Then verify the model still
  helps when its inputs are exactly the inference-time features.

---

## HIGH findings

### H1 — Pseudo-labels are a lexical proxy of a lexical proxy (circular supervision)
- **Where:** `compute_answer_quality()` `generate_span_pseudo_labels.py:153-157` →
  `extractive_demo_answer()` (:143-150) returns the top-2 query-overlap sentences;
  `char_f1` (:107-125) scores them against the gold answer; `answer_drop` (:425) is the drop
  in this proxy when a span is removed.
- **What it means:** The "QA feedback" that drives labels is not a reader model. It is
  token-overlap sentence selection scored by token-overlap F1. This is the same lexical signal
  as C2, so the supervision and the evaluation are correlated by construction — the system is
  graded by a metric aligned with the labels it was trained on.
- **Fix:** Regenerate pseudo-labels using a real reader LLM's answer-quality delta (leave-one-
  span-out or counterfactual removal scored by EM/F1 from the reader), or at minimum validate
  that lexical-proxy labels agree with reader-LLM labels on a sampled subset and report that
  agreement.

### H2 — Benchmark has no external baselines and no no-compression control
- **Where:** whole of `evaluate_stage2_english_benchmark.py` — it compares only `heuristic`
  vs `learned` Stage-2 (`build_compressor` with/without `span_model_dir`, :406-407) via
  `compare_pairs` (:320-375), on coverage deltas.
- **What it means:** The "small Stage-2 improvement" is a within-system delta between two
  variants of the same span pruner, on a proxy metric, with no reference point. It cannot
  support any comparative claim ("better than CPC / LLMLingua-2 / DAC / Top-K").
- **Fix:** Build a comparison harness that runs, at matched compression ratios (2×/4×/8×):
  no-compression (upper bound), truncation and random-drop (lower bounds), BM25 / embedding
  Top-K, CPC, LLMLingua-2, LongLLMLingua, DAC, and this method — all scored by the *same*
  frozen reader on EM/F1/ROUGE + evidence recall + token cost + latency.

### H3 — Silent lexical fallback can mask a missing/broken model as "working"
- **Where:** `allow_heuristic_fallback=True` in benchmark (`:185`) and `main.py:69`;
  fallback path `compression_pipeline.py:433-456`.
- **What it means:** If a checkpoint is absent or fails to load, the pipeline silently
  degrades to the lexical heuristic and still emits plausible numbers. Combined with C1, this
  is how "the encoder works" can be believed while the encoder never ran.
- **Fix:** In all evaluation entry points, require an explicit `--allow_fallback` flag that
  defaults to **False**; otherwise hard-fail when a requested checkpoint is missing. Record
  the actual runtime backend (`encoder_runtime`) in every results file.

---

## MEDIUM findings

### M1 — Single seed, single config; deltas within noise
- **Where:** benchmark `--seed 42` (:384); span trainer seeds fixed to 42 (:70-72);
  `compare_pairs` treats deltas down to `-0.01`/`-0.03` as signal (:350-358).
- **Fix:** Report ≥3 seeds with mean ± std (and ideally a paired significance test) for every
  headline number. Deltas smaller than cross-seed std must be reported as "no significant
  difference," not as improvements.

### M2 — Dataset and trained artifacts are not reproducibly present
- **Where:** `.gitignore` excludes `*.pt`, `*.safetensors`, all `outputs*/`,
  `**/english_cqr_mixed_5k/*.jsonl`, all pseudo-label/budget JSONL. Only `examples.json` and
  `build_summary.json` remain in git.
- **What it means:** The 5,000-example dataset splits and every checkpoint are absent from the
  repository. The work is currently not reproducible by a third party, and the exact splits
  used for any reported number cannot be recovered.
- **Fix:** Provide a `make data` / `make train` pipeline that regenerates everything from a
  pinned seed and pinned source snapshots; publish frozen splits (hashes at minimum) and
  release checkpoints via a model host; add `requirements.txt`/`environment.yml` with pinned
  versions.

### M3 — Documentation contradicts the code
- **Where:** `project_technical_report.md` §3.3/§5.6 describe `build_cqr_with_filters.py` and
  `clean_cqr_dataset.py`, **neither of which exists**; Windows paths and "已验证通过" claims
  persist. Real builders are `build_mixed_english_cqr_training_data.py` /
  `build_english_cqr_dataset.py`.
- **Fix:** Regenerate the report from the current code; treat it as a build artifact, not
  hand-maintained prose. A reviewer who spots doc/code drift distrusts every other claim.

---

## What this means for the current draft claims

| Claimed contribution | Current evidentiary status |
|---|---|
| Context-aware sentence encoder improves selection | **Unsupported** — encoder never runs (C1) |
| Answerability-preserving compression | **Unsupported** — metric is lexical proxy (C2, H1) |
| Stage-2 span pruning adds value | **Confounded** — oracle-seeded input (C3), leaked split (C4), oracle features (C5) |
| Multi-signal control beats single-signal | **Untested** — no ablation vs strong baselines (H2) |
| Better than CPC / DAC / LLMLingua | **Untested** — no external baselines (H2) |

None of these are disproven — they are simply **not yet measured under valid conditions.**
The remediation below is what converts them into testable claims.

## Remediation order (strict dependencies)

1. **C1 + H3 + M2** — make a real checkpoint load, fail loudly on fallback, pin data/env.
   *Nothing downstream is interpretable until the system under test is the real system.*
2. **C4 + C5** — group-disjoint span split; delete oracle features; retrain. *Makes the span
   model's own numbers honest.*
3. **C2 + C3 + H1** — real reader-LLM evaluation on full contexts; regenerate labels or
   validate proxy against reader on a subset. *Makes the metric measure the claim.*
4. **H2 + M1** — external baselines at fixed 2×/4×/8×, ≥3 seeds with variance. *Makes the
   comparison fair and the deltas significant.*
5. **M3** — regenerate docs from code. *Restores reviewer trust.*

Only after 1–4 can the true contribution be stated. It may be smaller than the current
framing, or differently shaped (e.g., "calibrated adaptive budgets" may prove stronger than
"Stage-2 span pruning"); that determination must follow the measurements, not precede them.
