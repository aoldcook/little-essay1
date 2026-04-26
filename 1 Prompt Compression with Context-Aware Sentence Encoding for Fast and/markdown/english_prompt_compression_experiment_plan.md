# English Prompt Compression Experiment Plan

## Goal

The project now studies English prompt/context compression for downstream black-box LLM QA. The compressed prompt should use fewer input tokens while preserving the evidence needed by the downstream model and remaining readable English.

## Method

### Stage-1: Context-aware sentence selection

Inputs are an English question and a long English context. The selector scores each sentence with:

- semantic similarity from a context-aware encoder, defaulting to `Qwen/Qwen3-Embedding-8B`
- attention probing between the question and the marked sentence window
- task reward from English question type, query overlap, entities, numbers, and answer-style anchors
- task descriptor alignment
- DAC-inspired sentence dynamics:
  - `dynamic_attention_scores`: question-to-sentence and global-to-sentence attention mass
  - `information_density_scores`: lexical entropy plus focused attention entropy
- MMR redundancy control under a token budget

This is the direct place where the DAC idea is used in the first stage. DAC is originally token-level; here it is lifted to a sentence-deletion decision by scoring whether a sentence attracts focused attention and carries high non-redundant information density.

### Stage-2: Task-aware intra-sentence span pruning

The second stage keeps English text extractive and readable. It protects:

- named entities, dates, locations, numbers, units, thresholds, and ranges
- negation and exception phrases
- causal, comparison, procedural, definition, and factoid anchors
- bridge clauses for multi-hop questions

It prunes parenthetical, example, tail, and low-value filler spans only when they do not overlap the query or protected evidence.

### Removed Stage-3

Generative rewrite/recompression is no longer part of the active pipeline. RankCoT and FeedSum-style candidate rewrite/reranking are not used because the current target is extractive English compression with less hallucination risk and simpler reproducibility.

## Dataset Construction

Use `data_builder/build_english_cqr_dataset.py` to convert raw English QA/RAG datasets into CQR rows:

```json
{
  "id": "source_id::pos0",
  "dataset": "longbench_hotpotqa",
  "question": "...",
  "context": "...",
  "answer": "...",
  "positive_sentence": "...",
  "supporting_sentences": ["..."],
  "negative_sentences": ["...", "..."]
}
```

Construction rule:

1. Split context into English sentences.
2. Use explicit supporting facts when available.
3. Otherwise score sentences by question overlap, answer overlap, numbers/entities, and support-text fuzzy matching.
4. Pick top evidence sentences as positives.
5. Pick hard negatives with high question overlap but low answer/support overlap.
6. Filter examples with too few sentences, too short context, or too few negatives.

Recommended datasets:

- Primary benchmark: LongBench English QA tasks: `hotpotqa`, `2wikimqa`, `musique`, `multifieldqa_en`, `qasper`, `narrativeqa`, `triviaqa`.
- Training/support data: HotpotQA, 2WikiMultiHopQA, MuSiQue, Qasper, Natural Questions, TriviaQA.
- Avoid Chinese-only datasets for the new target: DuReader, CMRC2018, C3.

## Models

### Compressor backbones

- Main Stage-1 encoder: `Qwen/Qwen3-Embedding-8B`.
- Smaller ablation: `Qwen/Qwen3-Embedding-0.6B` or `Qwen/Qwen3-Embedding-4B`.
- MLM regularization option: `answerdotai/ModernBERT-base`, only for `train_context_aware_encoder_with_mntp.py`.

### Downstream black-box LLMs

Run compressed prompts on at least two answer models:

- strong API model: GPT-4.1 / GPT-4o / Claude Sonnet class model, depending on available API
- open local/inference model: Qwen3-8B-Instruct or Llama-3.1-8B-Instruct

The compressor should not depend on internals of the downstream answer model.

## Main Comparisons

1. No Compression: full context.
2. Fixed Ratio Random Sentences: random sentence selection at the same token budget.
3. BM25/keyword sentence selection.
4. Embedding Top-K: Qwen3 embedding similarity only.
5. Stage-1 Only: full Stage-1 score without Stage-2.
6. Stage-1 + Stage-2: full proposed extractive compressor.
7. LLMLingua / LongLLMLingua if available locally or through their package.
8. PISCO if checkpoint/data are available; otherwise cite as RAG compression baseline and compare conceptually.

## Ablations

Stage-1 ablations:

- remove attention probing
- remove task reward
- remove task descriptor alignment
- remove dynamic attention score
- remove information density score
- remove MMR redundancy control
- replace Qwen3 embedding with a smaller embedding model

Stage-2 ablations:

- remove protected spans
- remove MIG keep-ratio allocation
- remove learned span model
- remove DAC span score
- remove numeric/entity protection

Budget ablations:

- fixed keep ratios: 0.2, 0.3, 0.4, 0.5, 0.6
- learned budget predictor vs fixed ratio
- equal sentence keep ratio vs MIG-allocated keep ratio

## Metrics

Compression metrics:

- input token count
- compression ratio
- latency and cost reduction

Answer quality metrics:

- exact match / token F1 where datasets support it
- Rouge-L for long-form QA when appropriate
- LLM judge only as secondary, with full-context answer as reference

Faithfulness and evidence metrics:

- answer evidence recall: whether gold/support sentences survive Stage-1
- protected entity/number recall
- contradiction rate from answer judge or NLI model if available

Readability metrics:

- sentence grammaticality proxy from downstream answer failure rate
- human spot checks on compressed English contexts

## Innovation Points Worth Keeping

1. Sentence-level DAC transfer: combine focused attention and entropy-derived information density before sentence deletion.
2. Conditional marginal information gain: keep sentences that add task-conditioned evidence not already covered by selected sentences.
3. Evidence graph redundancy control: model sentence-sentence and token-token overlap as a graph, then penalize redundant clusters rather than only pairwise MMR.
4. Position-aware evidence packing: reorder or preserve selected evidence to reduce lost-in-the-middle risk, inspired by long-context prompt compression work.
5. Answerability-preserving bottleneck distillation: train budget/span labels by checking whether compressed context preserves the full-context answer.

## Paper Mapping

- DAC, ACL 2025: suitable. Used in Stage-1 sentence dynamics and Stage-2 span scoring.
- PISCO, Findings ACL 2025: suitable for sequence-level answerability distillation and RAG compression comparison.
- LongLLMLingua, ACL 2024: suitable for query-aware compression and position-aware evidence packing.
- LLMLingua, EMNLP 2023: suitable baseline for token-level compression and budget control.
- Selective Context, EMNLP 2023: suitable baseline for redundancy pruning.
- QUITO-X / information bottleneck framing: suitable as the theoretical objective, but keep implementation as measurable relevance, redundancy, entropy, and answerability terms.
- RankCoT / FeedSum: not currently suitable because Stage-3 rewrite was removed.
- SAP: partially suitable if we later add syntactic span boundaries; do not add now unless grammar degradation becomes a measured error.

## Minimum Reproducible Run

1. Build English CQR data.
2. Train/evaluate Stage-1 ranking on dev top-1, top-2, MRR, and margin.
3. Generate span pseudo labels on English CQR train split.
4. Train Stage-2 span model.
5. Run benchmark runner on LongBench English QA tasks with the comparison list above.
6. Report quality vs token ratio curves and ablation table.
