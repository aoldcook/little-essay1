# English CQR Mixed 5k

This directory contains the first-stage English CQR training set for the context-aware sentence encoder.

## Files

- `all.jsonl`: 5,000 mixed CQR-style rows.
- `train.jsonl`: 4,523 rows.
- `dev.jsonl`: 233 rows.
- `test.jsonl`: 244 rows.
- `build_summary.json`: source counts, split counts, quality counts, and build arguments.
- `examples.json`: small human-readable preview.

## Source Mix

- `cqr_official`: 2,000 gold rows from the official CPC/CQR dataset.
- `hotpotqa_supporting_facts`: 1,801 gold rows from HotpotQA sentence-level supporting facts.
- `longbench_*`: 399 silver rows from LongBench structure-adaptation samples.
- `wikitext103_pseudo_cqr`: 800 silver teacher-free pseudo-CQR rows from WikiText-103.

## Row Format

Each row is compatible with `context_aware_encoder_model/train_context_aware_encoder.py`:

```json
{
  "question": "...",
  "context": "...",
  "answer": "...",
  "positive_sentence": "...",
  "supporting_sentences": ["..."],
  "negative_sentences": ["...", "..."],
  "dataset": "...",
  "metadata": {
    "quality": "gold|silver",
    "construction": "..."
  }
}
```

## Rebuild Command

```powershell
python -m data_builder.build_mixed_english_cqr_training_data `
  --output_dir data_builder\english_cqr_mixed_5k `
  --cache_dir D:\python_project\LittleEssay1\hf_cache `
  --target_total 5000 `
  --cqr_rows 2000 `
  --hotpotqa_rows 1800 `
  --longbench_rows 400 `
  --wikitext_rows 800 `
  --seed 42
```

## Suggested Stage-1 Training Command

```powershell
python context_aware_encoder_model\train_context_aware_encoder.py `
  --train_file data_builder\english_cqr_mixed_5k\train.jsonl `
  --dev_file data_builder\english_cqr_mixed_5k\dev.jsonl `
  --output_dir context_aware_encoder_model\outputs_english\mixed_5k `
  --model_name D:\python_project\LittleEssay1\hf_cache\models--Qwen--Qwen3-Embedding-8B\snapshots\1d8ad4ca9b3dd8059ad90a75d4983776a23d44af `
  --cache_dir D:\python_project\LittleEssay1\hf_cache `
  --epochs 1 `
  --batch_size 1
```
