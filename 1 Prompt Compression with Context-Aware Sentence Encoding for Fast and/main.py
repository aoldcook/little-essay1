from pipeline.compression_pipeline import ContextAwareCompressor

compressor = ContextAwareCompressor(
    encoder_dir=r"D:\python_project\LittleEssay1\1 Prompt Compression with Context-Aware Sentence Encoding for Fast and\context_aware_encoder_model\outputs",
    budget_model_dir=r"D:\python_project\LittleEssay1\1 Prompt Compression with Context-Aware Sentence Encoding for Fast and\target_ratio_model\outputs",   # 如果你已经训练了预算预测器
)

question = "CPC为什么能提升压缩效果？"
context = (
    "CPC是一种句子级压缩方法。"
    "它通过上下文感知句子编码器计算每个句子与问题的相关性。"
    "然后保留最相关的句子，以减少输入长度。"
    "这种方法通常比简单的通用句向量排序更稳。"
)

result = compressor.compress(question=question, context=context)

print("target_ratio:", result["target_ratio"])
print("selected_indices:", result["selected_indices"])
print("similarities:", result["similarities"])
print("compressed_context:")
print(result["compressed_context"])