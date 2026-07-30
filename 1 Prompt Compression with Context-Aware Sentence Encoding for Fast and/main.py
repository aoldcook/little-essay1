import os
from pathlib import Path

from context_aware_encoder_model.context_aware_sentence_encoder import default_hf_cache_dir
from pipeline.compression_pipeline import ContextAwareCompressor, estimate_token_count
from pipeline.runtime_contract import LEXICAL_FALLBACK_ID


project_root = Path(__file__).resolve().parent
hf_cache_dir = default_hf_cache_dir()

local_encoder = project_root / "context_aware_encoder_model" / "outputs_english" / "stage2_full"
qwen_cache = Path(hf_cache_dir) / "models--Qwen--Qwen3-Embedding-8B"

def _find_complete_qwen_snapshot(cache_root):
    required_files = (
        "config.json",
        "tokenizer_config.json",
        "tokenizer.json",
        "model.safetensors.index.json",
        "model-00001-of-00004.safetensors",
        "model-00002-of-00004.safetensors",
        "model-00003-of-00004.safetensors",
        "model-00004-of-00004.safetensors",
    )
    for snapshot_dir in (cache_root / "snapshots").glob("*"):
        if not snapshot_dir.is_dir():
            continue
        try:
            if all((snapshot_dir / name).is_file() and (snapshot_dir / name).stat().st_size > 0 for name in required_files):
                return snapshot_dir
        except OSError:
            continue
    return None


qwen_snapshot = _find_complete_qwen_snapshot(qwen_cache)

# Encoder resolution is explicit and fails loudly (audit findings C1 / H3).
# Previously this chain silently ended at the lexical heuristic, so the demo
# appeared to exercise the trained encoder when it never did.
# Set ALLOW_LEXICAL_FALLBACK=1 to run the non-neural baseline on purpose.
allow_lexical_fallback = os.environ.get("ALLOW_LEXICAL_FALLBACK", "").strip() in {"1", "true", "yes"}
if (local_encoder / "encoder_config.json").exists():
    encoder_source = str(local_encoder)
elif qwen_snapshot is not None:
    encoder_source = str(qwen_snapshot)
elif allow_lexical_fallback:
    encoder_source = LEXICAL_FALLBACK_ID
else:
    raise SystemExit(
        "No trained context-aware encoder checkpoint was found.\n"
        f"  looked for : {local_encoder / 'encoder_config.json'}\n"
        f"  and        : a complete Qwen3-Embedding-8B snapshot under {qwen_cache}\n\n"
        "Refusing to fall back to the lexical heuristic silently: doing so would\n"
        "print numbers that do not come from the trained encoder.\n"
        "Either train/download a checkpoint, or run the non-neural baseline knowingly:\n"
        "  ALLOW_LEXICAL_FALLBACK=1 python main.py"
    )

budget_model = project_root / "target_ratio_model" / "outputs_english"
budget_model_dir = str(budget_model) if (budget_model / "metadata.json").exists() else None

span_model = project_root / "intra_sentence_model" / "outputs_english_feedback"
span_model_dir = str(span_model) if (span_model / "metadata.json").exists() else None

compressor = ContextAwareCompressor(
    encoder_dir=encoder_source,
    encoder_cache_dir=hf_cache_dir,
    budget_model_dir=budget_model_dir,
    budget_formula_name="entropy_spread",
    span_model_dir=span_model_dir,
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
    enable_second_stage=True,
    second_stage_keep_ratio=0.52,
    second_stage_min_keep_ratio=0.34,
    second_stage_max_keep_ratio=0.72,
    allow_heuristic_fallback=allow_lexical_fallback,
)

print("system_label:", compressor.provenance().system_label())

question = "Why is the 1.5 degrees Celsius warming threshold considered a critical climate tipping point?"
long_context = """
Global warming is driven primarily by human greenhouse gas emissions from fossil fuel combustion, land-use change, and industrial agriculture. The Intergovernmental Panel on Climate Change reports that warming beyond 1.5 degrees Celsius sharply increases the probability of irreversible ecological tipping points. These risks include accelerated ice-sheet loss, thawing permafrost that releases methane, severe coral reef decline, and stress on food and water systems. The Paris Agreement asks countries to pursue efforts to limit warming to 1.5 degrees Celsius, but current policies remain insufficient. Some mitigation strategies include renewable energy deployment, methane reduction, carbon capture, and nature-based restoration.
"""

result = compressor.compress(question=question, context=long_context)

print("encoder_source:", encoder_source)
print("encoder_runtime:", compressor.encoder_runtime)
if compressor.encoder_load_error:
    print("encoder_load_error:", compressor.encoder_load_error.splitlines()[-1][:300])
print("hf_cache_dir:", hf_cache_dir)
print("span_model_dir:", span_model_dir)
if compressor.span_compressor is not None:
    print("learned_span_model_active:", compressor.span_compressor.learned_span_model is not None)
    print("learned_keep_weight:", compressor.span_compressor.config.learned_keep_weight)
    print("learned_soft_protected_threshold:", compressor.span_compressor.config.learned_soft_protected_threshold)
print("target_ratio:", result["target_ratio"])
print("budget_formula:", result["budget_formula"])
print("selected_indices:", result["selected_indices"])
original_tokens = sum(estimate_token_count(sentence) for sentence in result["sentences"])
stage1_tokens = sum(estimate_token_count(sentence) for sentence in result["selected_sentences"])
stage2_tokens = sum(estimate_token_count(sentence) for sentence in result["compressed_sentences"])
print("original_tokens:", original_tokens)
print("stage1_tokens:", stage1_tokens)
print("stage2_tokens:", stage2_tokens)
print("stage1_ratio:", round(stage1_tokens / max(original_tokens, 1), 3))
print("final_ratio:", round(stage2_tokens / max(original_tokens, 1), 3))
print("task_descriptor:", result["task_descriptor"])
print("semantic_similarities:", result["semantic_similarities"])
print("attention_probe_scores:", result["attention_probe_scores"])
print("dynamic_attention_scores:", result["dynamic_attention_scores"])
print("information_density_scores:", result["information_density_scores"])
print("linguistic_scores:", result["linguistic_scores"])
print("task_rewards:", result["task_rewards"])
print("task_descriptor_scores:", result["task_descriptor_scores"])
print("selection_scores:", result["selection_scores"])
print("linguistic_features:")
for feature in result["linguistic_features"]:
    print(feature)
print("removed_span_count:", result["second_stage_stats"]["removed_span_count"])
print("stage1_context:")
print(result["stage1_context"])
print("compressed_context:")
print(result["compressed_context"])
print("original_sentence_count:", len(result["sentences"]))
print("stage1_sentence_count:", len(result["selected_sentences"]))
print("stage2_sentence_count:", len(result["compressed_sentences"]))

for idx, sentence in enumerate(result["selected_sentences"], start=1):
    print(f"stage1_sentence_{idx}:", sentence)

for idx, sentence_stat in enumerate(result["second_stage_stats"]["sentence_stats"], start=1):
    print(f"stage2_sentence_{idx}_compression_mode:", sentence_stat.get("compression_mode"))
    print(f"stage2_sentence_{idx}_removed_spans:", sentence_stat["removed_spans"])





