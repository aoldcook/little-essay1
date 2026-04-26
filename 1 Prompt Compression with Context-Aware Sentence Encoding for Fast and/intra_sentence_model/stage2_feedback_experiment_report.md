# Stage-2 Feedback Span Model Experiment

This run replaces the easy heuristic-only span pseudo labels with QA-feedback labels, hard disagreement cases, and negative sentence examples.

## Training Data

- Source split: `data_builder/english_cqr_mixed_5k/train.jsonl`
- Generated labels: `intra_sentence_model/span_pseudo_english_feedback_train.jsonl`
- Label policy: `feedback`
- Negative sentences: enabled, up to 2 per training row
- Sentence examples: 9,327
- Span instances: 29,049
- Label distribution: 13,819 keep / 15,230 drop
- Hard disagreement cases: 12,190

## Model

- Output: `intra_sentence_model/outputs_english_feedback`
- Best model: `span_model.best.pt`
- Final model: `span_model.pt`
- Dev loss: 0.3442
- Dev accuracy: 0.8258
- Feature dimension: 31

## Runtime Defaults

- `main.py` now prefers `intra_sentence_model/outputs_english_feedback`
- `learned_keep_weight`: 0.72
- `learned_soft_protected_threshold`: 0.28
- Strongly protected spans remain guarded unless the learned model gives a low keep score and the span is not a hard semantic anchor.
- Stage-2 now splits long spans more finely around source attribution, relative/contrast clauses, and list-like coordination.
- Stage-2 now uses a local safety gate before deleting a candidate span. The gate blocks deletion if it would remove hard anchors, drop too much question-term coverage, remove the last semantic anchor, break a cause/evidence list floor, or weaken factoid/numeric/definition spans.

## Benchmark: Default Stage-2 Budget

Command output: `intra_sentence_model/benchmark_outputs_english_feedback_t010_qa`

- Rows: 244
- Heuristic final ratio: 0.70497
- Feedback model final ratio: 0.69757
- Ratio delta: -0.00740
- Positive coverage delta: +0.00009
- Support coverage delta: +0.00002
- Answer coverage delta: -0.00211
- QA answerable rate: 0.77459 -> 0.77459
- Efficient win rate: 0.84426
- Same-or-better answerable rows: 244 / 244

## Benchmark: Stress Stage-2 Budget

Command output: `intra_sentence_model/benchmark_outputs_english_feedback_t010_qa_stress`

Stress config:

- `second_stage_keep_ratio`: 0.32
- `second_stage_min_keep_ratio`: 0.18
- `second_stage_max_keep_ratio`: 0.45

Results:

- Rows: 244
- Heuristic final ratio: 0.70393
- Feedback model final ratio: 0.69794
- Ratio delta: -0.00599
- Positive coverage delta: +0.00269
- Support coverage delta: +0.00242
- Answer coverage delta: -0.00211
- QA answerable rate: 0.77049 -> 0.77459
- Efficient win rate: 0.85656
- Same-or-better answerable rows: 244 / 244

## Reading

The feedback model gives a small but real gain: it removes more spans while preserving the same answerability under the default budget, and under the stronger Stage-2 stress budget it improves positive/support coverage and answerable rows versus the heuristic baseline. The remaining bottleneck is not the learned ranker alone; many sentences stop compressing because hard anchors and protected spans exhaust safe removal candidates.

## Guarded Protected-Span Relaxation

Command output:

- Default: `intra_sentence_model/benchmark_outputs_english_feedback_guarded_default_qa`
- Stress: `intra_sentence_model/benchmark_outputs_english_feedback_guarded_default_qa_stress`

Default Stage-2 budget:

- Rows: 244
- Heuristic final ratio: 0.70473
- Guarded feedback model final ratio: 0.70279
- Ratio delta: -0.00194
- Positive coverage delta: +0.00131
- Support coverage delta: +0.00128
- Answer coverage delta: +0.00084
- QA answerable rate: 0.77459 -> 0.77869
- Same-or-better answerable rows: 244 / 244
- Avg safety-skipped candidates: 1.67 per row for learned compression

Stress Stage-2 budget:

- Rows: 244
- Heuristic final ratio: 0.70397
- Guarded feedback model final ratio: 0.70249
- Ratio delta: -0.00148
- Positive coverage delta: +0.00252
- Support coverage delta: +0.00260
- Answer coverage delta: +0.00084
- QA answerable rate: 0.77049 -> 0.77869
- Same-or-better answerable rows: 244 / 244
- Avg safety-skipped candidates: 1.60 per row for learned compression

The earlier unguarded/tighter settings produced larger ratio reductions, but they allowed small answer-coverage drops. The guarded setting is intentionally more conservative: it only accepts extra compression when the local evidence and answerability checks remain intact.
