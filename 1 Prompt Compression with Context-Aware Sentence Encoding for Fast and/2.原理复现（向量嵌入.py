import re
import torch
from transformers import AutoTokenizer, AutoModel
from target_ratio import find_target_ratio

class EmbeddingBasedCompressor:
    def __init__(self,model_name="BAAI/bge-small-zh-v1.5", device=None) -> None:
       self.device=device or ("cuda" if torch.cuda.is_available() else "cpu")
       self.tokenizer=AutoTokenizer.from_pretrained(model_name)
       self.model=AutoModel.from_pretrained(model_name).to(self.device) 
       self.model.eval()

    def mean_pooling(self,model_output,attention_mask):
        token_embeddings = model_output[0]                  # (B, L, H)
        mask = attention_mask.unsqueeze(-1).float()         # (B, L, 1)

        masked_embeddings = token_embeddings * mask         # pad位置清零
        sum_embeddings = masked_embeddings.sum(dim=1)       # (B, H)
        valid_token_count = mask.sum(dim=1).clamp(min=1e-9) # (B, 1)

        return sum_embeddings / valid_token_count

    def encode(self,texts,max_length=512):
        """
        对文本进行编码，返回句向量。
        """
        encoded_input=self.tokenizer(
            texts,
            padding=True,
            truncation=True,
            max_length=max_length,
            return_tensors="pt"
        ).to(self.device)

        with torch.no_grad():
            model_output=self.model(**encoded_input)
        sentence_embeddings=self.mean_pooling(model_output,encoded_input["attention_mask"])
        return sentence_embeddings

    def split_sentences(self,text:str):
        """
        支持中英文混合断句
        """
        sentences = re.split(r'(?<=[。！？.!?])\s*', text.strip())
        sentences = [s.strip() for s in sentences if s.strip()]
        return sentences

    def compress(self,context:str,question:str,target_ratio:float=None):
        sentences=self.split_sentences(context)
        if not sentences:
            return context

        # BGE 系列推荐给 query 加前缀
        query_text = f"为这个句子生成表示以用于检索相关文章：{question}"
        all_texts=[query_text]+sentences
        embeddings=self.encode(all_texts)
        query_embedding=embeddings[0:1]# (1, dim)
        sentence_embeddings=embeddings[1:]# (num_sentences, dim)

        # 已经归一化，因此点积就是 cosine similarity
        similarities = torch.matmul(query_embedding, sentence_embeddings.T).squeeze(0).cpu().numpy()

        scores_and_sentences=list(zip(similarities,sentences))
        scores_and_sentences.sort(key=lambda x:x[0],reverse=True)

        # 自动决定 target_ratio
        if target_ratio is None:
            target_ratio = find_target_ratio([score for score, _ in scores_and_sentences])

        # 用更稳的“字符长度”估算压缩长度，避免中文 split() 不准
        total_len_original=sum(len(sent) for sent in sentences)
        target_len=max(1,int(total_len_original*target_ratio))

        selected_sentences=[]
        current_len=0

        for score,sent in scores_and_sentences:
            sent_len=len(sent)
            if current_len+sent_len<=target_len:
                selected_sentences.append(sent)
                current_len+=sent_len

        #按原文顺序恢复，避免语义跳跃
        selected_set=set(selected_sentences)
        selected_sentences=[s for s in sentences if s in selected_set]

        return "".join(selected_sentences)

if __name__ == "__main__":
    long_context = """
    近年来，大语言模型（Large Language Models, LLMs）在自然语言处理领域取得了突破性进展。从早期的BERT、GPT系列，到如今的LLaMA、ChatGLM、Qwen等开源或闭源模型，其参数规模和推理能力不断提升。然而，随着模型能力的增强，对计算资源的需求也急剧上升，尤其是在处理长上下文输入时，推理延迟和显存占用成为实际部署中的主要瓶颈。
    为应对这一挑战，研究者提出了多种优化策略。其中，上下文压缩（Context Compression）技术因其无需修改模型结构而备受关注。典型方法包括基于重要性评分的token剪枝（如LLMLingua）、利用小型代理模型进行摘要（如In-Context Learning with Summarization），以及最近提出的基于句子级选择的上下文感知压缩（CPC）。CPC方法通过训练一个专门的句子编码器，评估每个句子在特定问题下的相关性，并保留高分句子以重构压缩后的上下文，从而在保持回答质量的同时显著降低输入长度。
    值得注意的是，CPC的有效性高度依赖于其训练数据的质量。论文作者构建了一个名为CQR（Context-aware Question-Relevance）的新数据集，该数据集要求正样本句子虽不直接包含答案，但能提供关键上下文线索，而负样本则需经严格验证确保完全无关。这种设计使得模型能够学习到更精细的语义依赖关系，而非简单的关键词匹配。
    此外，压缩策略还需考虑下游任务的特性。例如，在问答任务中，保留包含实体、动作或因果关系的句子更为重要；而在摘要任务中，则需兼顾信息覆盖度与连贯性。因此，理想的压缩系统应具备任务自适应能力，甚至能根据输入动态调整压缩率——例如，对于信息密集的科研论文采用较低压缩比，而对于冗余较多的会议记录则可大幅裁剪。
    尽管已有诸多进展，当前方法仍面临若干挑战：一是如何在极高压缩比（如保留10%内容）下维持语义完整性；二是如何将压缩过程与LLM的内部注意力机制对齐，以最小化信息损失；三是缺乏统一的评估基准，导致不同方法间难以公平比较。未来的研究或将结合强化学习、知识蒸馏或可微分压缩等方向，进一步提升效率与效果的平衡。
    """

    question = "CPC方法是如何利用CQR数据集来提升句子相关性判断的准确性的？"

    compressor = EmbeddingBasedCompressor()
    compressed_context = compressor.compress(long_context, question)

    print("Compressed Context:")
    print(compressed_context)
    print()
    print("Original Length:", len(long_context))
    print("Compressed Length:", len(compressed_context))