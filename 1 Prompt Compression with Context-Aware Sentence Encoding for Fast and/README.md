# English Prompt Compression for Downstream Black-box LLMs

This project now targets English prompt/context compression for QA and RAG-style downstream black-box LLMs. The previous Chinese benchmark path and Stage-3 generative rewriting path have been removed from the active pipeline.

## Active Pipeline

1. Stage-1: context-aware sentence selection
   - Qwen3 embedding backbone by default: `Qwen/Qwen3-Embedding-8B`
   - semantic sentence-query similarity
   - attention probing
   - task reward and task descriptor alignment
   - DAC-inspired sentence dynamics: dynamic attention plus information-density scoring
   - MMR selection under a token budget

2. Stage-2: task-aware intra-sentence span pruning
   - protected spans for entities, numbers, units, thresholds, negation, and answer-critical clauses
   - English question type rules: cause, comparison, procedure, numeric, definition, factoid, other
   - marginal-information-gain keep-ratio allocation
   - optional learned span model

Stage-3 generative recompression is no longer part of the active code path.

## Storage

Model cache defaults to:

```bash
D:\python_project\LittleEssay1\hf_cache
```

The code sets `HF_HOME`, `HF_HUB_CACHE`, and `TRANSFORMERS_CACHE` from `ContextAwareEncoderConfig.cache_dir` when the encoder loads.

## Direct Demo

```bash
python main.py
```

If no local English checkpoint exists under `context_aware_encoder_model/outputs_english/stage2_full`, the demo loads `Qwen/Qwen3-Embedding-8B` and caches it on D drive. Use a smaller compatible model if GPU memory is limited, for example `Qwen/Qwen3-Embedding-0.6B`.

## Build English CQR Data

Prepare raw JSON/JSONL files from English QA/RAG datasets, then run:

```bash
python -m data_builder.build_english_cqr_dataset ^
  --input_file data_builder/source_english/longbench_hotpotqa.jsonl --dataset_name longbench_hotpotqa ^
  --input_file data_builder/source_english/longbench_2wikimqa.jsonl --dataset_name longbench_2wikimqa ^
  --input_file data_builder/source_english/longbench_musique.jsonl --dataset_name longbench_musique ^
  --output_dir data_builder/english_cqr
```

Recommended sources:

- LongBench: `hotpotqa`, `2wikimqa`, `musique`, `multifieldqa_en`, `qasper`, `narrativeqa`, `triviaqa`
- HotpotQA, 2WikiMultiHopQA, MuSiQue, Qasper, Natural Questions, TriviaQA

## Train

```bash
python -m context_aware_encoder_model.train_context_aware_encoder ^
  --train_file data_builder/english_cqr/train.jsonl ^
  --dev_file data_builder/english_cqr/dev.jsonl ^
  --output_dir context_aware_encoder_model/outputs_english ^
  --model_name Qwen/Qwen3-Embedding-8B ^
  --cache_dir D:\python_project\LittleEssay1\hf_cache
```

For MLM/MNTP-style regularization, use an English masked-LM backbone such as `answerdotai/ModernBERT-base`; Qwen3 embedding models are the default for the plain Stage-1 encoder path.

## Key Docs

- `markdown/english_prompt_compression_experiment_plan.md`: revised experiment design, baselines, ablations, and paper mapping
- `data_builder/build_english_cqr_dataset.py`: English CQR dataset construction
- `pipeline/compression_pipeline.py`: two-stage compression entry point
- `pipeline/task_aware_compression.py`: English span pruning logic
