import numpy as np
import langchain
from langchain_core.prompts import PromptTemplate
from langchain_community.llms.tongyi import Tongyi
import time
import torch
from target_ratio import find_target_ratio



# --- 方案一：使用现有LLM ---
def compress_with_llm(context:str,question:str,model_name="qwen3-max"):
    """
    使用现有LLM对上下文进行压缩，目标压缩比为target_ratio。
    """

    #1. 初始化LLM
    model=Tongyi(model="qwen-max",api_key="sk-e716e7b1d65e4be5bbd222b75379ed5b")
    prompt_template=PromptTemplate.from_template(
        "Context: {sent},Question: {question},Please rate the relevance of this context sentence to the question on a scale from 0 to 1, where 1 means highly relevant and 0 means not relevant at all. Only output the number."
    )

    chain=prompt_template|model

    #2.对每个句子进行评分
    sentences=[s.strip() for s in context.split("。") if s.strip()]
    sentences = [s + '.' for s in sentences[:-1]] + [sentences[-1]]
    print(f"Assessing relevance of {len(sentences)} sentences using LLM...")

    scores_and_sentences=[]
    for sent in sentences:
        try:
            response=chain.invoke({"sent":sent,"question":question})
            if isinstance(response, (int, float)):
                score=float(response)
            else:
                response_text=str(response).strip()
                cleaned="".join(ch for ch in response_text if ch.isdigit() or ch==".")
                score=float(cleaned) if cleaned else 0.0
        except Exception as e:
            print(f"Error processing sentence: {sent}. Error: {e}")
            score=0.0
        scores_and_sentences.append((score, sent))

    #3.根据评分排序并选择前target_ratio比例的句子
    scores_and_sentences.sort(key=lambda x:x[0],reverse=True)

    total_token_original=len(context.split())
    target_ratio=find_target_ratio([score for score,_ in scores_and_sentences])
    target_tokens=int(total_token_original*target_ratio)
    
    compressed_parts=[]
    current_tokens=0

    for score,sent in scores_and_sentences:
        sent_tokens=len(sent.split())
        if current_tokens+sent_tokens<=target_tokens:
            compressed_parts.append(sent)
            current_tokens+=sent_tokens
        else:
            break

    return "".join(compressed_parts)

if __name__ == "__main__":
    # Example context and question
    long_context = """
    近年来，大语言模型（Large Language Models, LLMs）在自然语言处理领域取得了突破性进展。从早期的BERT、GPT系列，到如今的LLaMA、ChatGLM、Qwen等开源或闭源模型，其参数规模和推理能力不断提升。然而，随着模型能力的增强，对计算资源的需求也急剧上升，尤其是在处理长上下文输入时，推理延迟和显存占用成为实际部署中的主要瓶颈。
为应对这一挑战，研究者提出了多种优化策略。其中，上下文压缩（Context Compression）技术因其无需修改模型结构而备受关注。典型方法包括基于重要性评分的token剪枝（如LLMLingua）、利用小型代理模型进行摘要（如In-Context Learning with Summarization），以及最近提出的基于句子级选择的上下文感知压缩（CPC）。CPC方法通过训练一个专门的句子编码器，评估每个句子在特定问题下的相关性，并保留高分句子以重构压缩后的上下文，从而在保持回答质量的同时显著降低输入长度。
值得注意的是，CPC的有效性高度依赖于其训练数据的质量。论文作者构建了一个名为CQR（Context-aware Question-Relevance）的新数据集，该数据集要求正样本句子虽不直接包含答案，但能提供关键上下文线索，而负样本则需经严格验证确保完全无关。这种设计使得模型能够学习到更精细的语义依赖关系，而非简单的关键词匹配。
此外，压缩策略还需考虑下游任务的特性。例如，在问答任务中，保留包含实体、动作或因果关系的句子更为重要；而在摘要任务中，则需兼顾信息覆盖度与连贯性。因此，理想的压缩系统应具备任务自适应能力，甚至能根据输入动态调整压缩率——例如，对于信息密集的科研论文采用较低压缩比，而对于冗余较多的会议记录则可大幅裁剪。
尽管已有诸多进展，当前方法仍面临若干挑战：一是如何在极高压缩比（如保留10%内容）下维持语义完整性；二是如何将压缩过程与LLM的内部注意力机制对齐，以最小化信息损失；三是缺乏统一的评估基准，导致不同方法间难以公平比较。未来的研究或将结合强化学习、知识蒸馏或可微分压缩等方向，进一步提升效率与效果的平衡。
    """

    question = "CPC方法是如何利用CQR数据集来提升句子相关性判断的准确性的？"

    print("--- Comparing Compression Methods ---\n")
    compressed_context=compress_with_llm(long_context,question)
    print(f"Compressed Context (LLM):\n{compressed_context}\n")
    print(f"Original Context Tokens: {len(long_context.split())}")
    print(f"Compressed Context Tokens: {len(compressed_context.split())}")
