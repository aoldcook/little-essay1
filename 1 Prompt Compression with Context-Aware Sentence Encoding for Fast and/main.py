from pathlib import Path

from pipeline.compression_pipeline import ContextAwareCompressor

project_root = Path(__file__).resolve().parent
span_model_dir = project_root / "intra_sentence_model" / "outputs"
if not span_model_dir.exists():
    span_model_dir = project_root / "intra_sentence_model" / "outputs_demo"

compressor = ContextAwareCompressor(
    encoder_dir=str(project_root / "context_aware_encoder_model" / "outputs_mntp"),
    budget_model_dir=str(project_root / "target_ratio_model" / "outputs"),
    span_model_dir=str(span_model_dir),
    use_attention_probe=True,
    attention_probe_weight=0.25,
    task_reward_weight=0.15,
    enable_second_stage=True,
    second_stage_keep_ratio=0.5,
)

question = "CPC为什么能提升压缩效果？"
long_context = """
近年来，大语言模型（Large Language Models, LLMs）在自然语言处理领域取得了突破性进展。从早期的BERT、GPT系列，到如今的LLaMA、ChatGLM、Qwen等开源或闭源模型，其参数规模和推理能力不断提升。然而，随着模型能力的增强，对计算资源的需求也急剧上升，尤其是在处理长上下文输入时，推理延迟和显存占用成为实际部署中的主要瓶颈。
为应对这一挑战，研究者提出了多种优化策略。其中，上下文压缩（Context Compression）技术因其无需修改模型结构而备受关注。典型方法包括基于重要性评分的token剪枝（如LLMLingua）、利用小型代理模型进行摘要（如In-Context Learning with Summarization），以及最近提出的基于句子级选择的上下文感知压缩（CPC）。CPC方法通过训练一个专门的句子编码器，评估每个句子在特定问题下的相关性，并保留高分句子以重构压缩后的上下文，从而在保持回答质量的同时显著降低输入长度。
值得注意的是，CPC的有效性高度依赖于其训练数据的质量。论文作者构建了一个名为CQR（Context-aware Question-Relevance）的新数据集，该数据集要求正样本句子虽不直接包含答案，但能提供关键上下文线索，而负样本则需经严格验证确保完全无关。这种设计使得模型能够学习到更精细的语义依赖关系，而非简单的关键词匹配。
此外，压缩策略还需考虑下游任务的特性。例如，在问答任务中，保留包含实体、动作或因果关系的句子更为重要；而在摘要任务中，则需兼顾信息覆盖度与连贯性。因此，理想的压缩系统应具备任务自适应能力，甚至能根据输入动态调整压缩率——例如，对于信息密集的科研论文采用较低压缩比，而对于冗余较多的会议记录则可大幅裁剪。
尽管已有诸多进展，当前方法仍面临若干挑战：一是如何在极高压缩比（如保留10%内容）下维持语义完整性；二是如何将压缩过程与LLM的内部注意力机制对齐，以最小化信息损失；三是缺乏统一的评估基准，导致不同方法间难以公平比较。未来的研究或将结合强化学习、知识蒸馏或可微分压缩等方向，进一步提升效率与效果的平衡。
"""

result = compressor.compress(question=question, context=long_context)

print("target_ratio:", result["target_ratio"])
print("selected_indices:", result["selected_indices"])
print("semantic_similarities:", result["semantic_similarities"])
print("attention_probe_scores:", result["attention_probe_scores"])
print("task_rewards:", result["task_rewards"])
print("selection_scores:", result["selection_scores"])
print("removed_span_count:", result["second_stage_stats"]["removed_span_count"])
print("compressed_context:")
print(result["compressed_context"])
print("original_sentence_count:", len(long_context.split("。")))
print("stage1_sentence_count:", len(result["selected_sentences"]))
print("stage2_sentence_count:", len(result["compressed_sentences"]))

for idx, sentence_stat in enumerate(result["second_stage_stats"]["sentence_stats"], start=1):
    print(f"stage2_sentence_{idx}_removed_spans:", sentence_stat["removed_spans"])
