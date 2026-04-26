# `context_aware_encoder_model` 代码详解

这份说明面向你当前项目中的 **`context_aware_encoder_model`** 文件夹，目标是把这一部分代码的：

- 总体功能
- 每个文件的职责
- 每个类 / 函数的作用
- 代码运行逻辑
- 为什么要这样写

讲清楚，方便你后续自己改代码、写论文方法章节和设计实验。

---

## 一、这个文件夹总体是在做什么

`context_aware_encoder_model` 这一层，负责解决你整个压缩系统里的第一个核心问题：

> **哪一个句子在当前问题下更重要？**

它不是直接做压缩率预测，也不是直接选择最终保留哪些句子，而是训练并使用一个：

> **上下文感知句子编码器（Context-Aware Sentence Encoder）**

这个编码器的设计思想是：

- 不把句子孤立编码
- 而是把句子放回完整上下文里编码
- 再从该句在上下文中的位置提取“sentence-in-context embedding”
- 最后和问题向量做相似度计算

因此，这一层输出的是：

- 每个句子的相关性分数
- 每个句子的上下文感知向量表示

后面的 `pipeline` 会继续使用这些分数和向量去做：

- 压缩率预测
- 预算约束下的句子选择
- 最终压缩文本构造

---

## 二、这个文件夹中两个主要文件的分工

### 1. `context_aware_sentence_encoder.py`

这是核心模型文件，负责：

- 定义配置类 `ContextAwareEncoderConfig`
- 定义模型类 `ContextAwareSentenceEncoder`
- 提供句子切分函数
- 提供“带句子标记的上下文构造函数”

它回答的问题是：

> **如何把一个句子编码成“它在上下文中的表示”？**

---

### 2. `train_context_aware_encoder.py`

这是训练脚本，负责：

- 读取训练数据
- 把正样本句 / 负样本句插回上下文并加标记
- 构造训练 batch
- 调用模型进行对比学习训练
- 保存训练好的 encoder 和 tokenizer

它回答的问题是：

> **如何训练这个上下文感知句子编码器？**

---

# 三、`context_aware_sentence_encoder.py` 详细说明

---

## 1. `ContextAwareEncoderConfig`

```python
@dataclass
class ContextAwareEncoderConfig:
    model_name: str = "Qwen/Qwen3-Embedding-8B"
    max_length: int = 512
    temperature: float = 0.05
    device: str = "cpu"
    marker_start: str = "<sent_start>"
    marker_end: str = "<sent_end>"
```

### 功能
这是模型配置类，用来统一管理句子编码器的超参数和运行配置。

### 每个字段的意义

#### `model_name`
底层 Transformer 编码器名称，默认是 `Qwen/Qwen3-Embedding-8B`。

作用：
- 决定用哪个预训练模型做 backbone
- 后续可以很容易替换成 RoBERTa、MacBERT、BGE backbone 等

#### `max_length`
编码时的最大长度。

作用：
- 控制 tokenizer 截断长度
- 防止超长文本导致显存爆炸
- 这里本质上是对“完整上下文 + 问题”做长度控制

#### `temperature`
InfoNCE 对比学习里的温度参数。

作用：
- 控制 softmax 的“尖锐程度”
- 温度越小，模型越强调把正样本和负样本拉开
- 常用于对比学习

#### `device`
指定模型运行在 `cpu` 还是 `cuda`。

#### `marker_start` / `marker_end`
特殊标记，默认是：

- `<sent_start>`
- `<sent_end>`

作用：
- 用来在完整上下文中圈出“当前要编码的句子”
- 让模型知道：这一段就是目标句 span

---

## 2. `ContextAwareSentenceEncoder`

```python
class ContextAwareSentenceEncoder(nn.Module):
```

这是整个文件最核心的类。

### 总体功能
它实现了一个最小可运行的上下文感知句子编码器。其核心流程是：

1. 单独编码问题，得到问题向量
2. 把某个句子放回原始上下文并用特殊标记圈出
3. 编码“问题 + 带标记上下文”
4. 在标记区间内做平均池化
5. 得到该句的 sentence-in-context embedding
6. 和问题向量做相似度计算
7. 训练时用对比学习让正句更近、负句更远

也就是说，它不是一个普通的“句向量模型”，而是一个：

> **query-aware + context-aware 的句子表示模型**

---

## 3. `__init__`

```python
def __init__(self, config: ContextAwareEncoderConfig):
```

### 主要做了什么

#### （1）保存配置
```python
self.config = config
```

#### （2）加载 tokenizer
```python
self.tokenizer = AutoTokenizer.from_pretrained(config.model_name)
```

#### （3）往 tokenizer 里加入特殊标记
```python
self.tokenizer.add_special_tokens(
    {"additional_special_tokens": [config.marker_start, config.marker_end]}
)
```

这一步非常关键。

因为后面构造输入时会出现：

```text
... 上文 <sent_start>目标句</sent_end> 下文 ...
```

如果不先把这些特殊 token 注册进 tokenizer，它们可能会被拆坏，或者根本不能稳定识别。

#### （4）加载底层编码器
```python
self.encoder = AutoModel.from_pretrained(config.model_name)
```

#### （5）调整词表大小
```python
self.encoder.resize_token_embeddings(len(self.tokenizer))
```

因为你新加了 `<sent_start>` 和 `<sent_end>`，所以 embedding 矩阵也必须扩容。

#### （6）记录特殊标记对应的 token id
```python
self.start_id = self.tokenizer.convert_tokens_to_ids(config.marker_start)
self.end_id = self.tokenizer.convert_tokens_to_ids(config.marker_end)
```

后面要靠它们在编码后的 token 序列中找目标句 span。

#### （7）设置设备
```python
self.device = torch.device(config.device)
self.to(self.device)
```

---

## 4. `mean_pool`

```python
def mean_pool(self, hidden_states, attention_mask):
```

### 功能
对一整段 token hidden states 做 masked mean pooling。

### 输入
- `hidden_states`: `[B, L, H]`
- `attention_mask`: `[B, L]`

### 输出
- `[B, H]`

### 为什么这样写
因为 Transformer 输出是 token 级别向量，而问题编码需要一个句子级表示，所以要把所有有效 token 平均起来。

这里使用 attention mask，是为了：

- 忽略 padding token
- 避免句长不一致时 pad 干扰向量

这属于标准的 masked mean pooling。

---

## 5. `encode_question`

```python
def encode_question(self, questions):
```

### 功能
把一个或多个问题编码成 query embedding。

### 过程
1. tokenizer 编码问题
2. 送入 Transformer
3. 对输出做 mean pooling
4. 做 L2 normalize

### 输出
返回形状：

\[
[B, d]
\]

其中：
- `B` 是 batch size
- `d` 是隐藏维度

### 为什么要 normalize
因为后面相似度计算用的是点积：

```python
torch.matmul(q_emb, s_emb.T)
```

如果先做了 L2 normalize，那么点积就等价于 cosine similarity。

---

## 6. `_find_marker_span`

```python
def _find_marker_span(self, input_ids):
```

### 功能
在 token 序列中找到 `<sent_start>` 和 `<sent_end>` 的位置。

### 为什么必须有这个函数
因为模型并不知道目标句是哪一段，你是通过特殊标记告诉它的。

所以编码完之后，还需要反过来找到：

- 起始标记位置
- 结束标记位置

再提取中间那一段 hidden states。

### 返回值
```python
(start_pos + 1, end_pos)
```

也就是说，真正取的是两标记之间的 token span，不包括标记本身。

### 为什么要做异常检查
如果找不到 marker，或者开始结束位置异常，说明：

- 你构造 marked context 的逻辑错了
- 或 tokenizer 没正确加入特殊 token

所以这里显式报错，方便排查。

---

## 7. `encode_marked_contexts`

```python
def encode_marked_contexts(self, questions, marked_contexts):
```

### 功能
对“问题 + 带标记上下文”进行编码，并提取被标记句子的上下文感知表示。

### 这是整个模型的关键函数

它不是简单编码句子文本，而是编码：

- 当前问题
- 完整上下文
- 其中某个句子被 marker 标出来

### 执行流程

#### 第一步：联合编码
```python
batch = self.tokenizer(
    list(questions),
    list(marked_contexts),
    ...
)
```

这里传的是双输入：

- 第一个输入：问题
- 第二个输入：带标记的上下文

这样底层编码器就能在 self-attention 里同时看到：
- 问题内容
- 完整上下文
- 哪个句子是目标句

#### 第二步：取 hidden states
```python
outputs = self.encoder(**batch)
hidden = outputs.last_hidden_state
```

#### 第三步：逐样本找 marker span
```python
start, end = self._find_marker_span(batch["input_ids"][b])
```

#### 第四步：对 marker span 内 token 做平均
```python
span_hidden = hidden[b, start:end, :]
sent_vec = span_hidden.mean(dim=0)
```

#### 第五步：归一化
```python
return F.normalize(sent_vecs, p=2, dim=1)
```

### 这一步背后的核心思想
这就是你要强调的：

> **sentence-in-context embedding**

也就是：

- 这个句子不再是孤立句向量
- 而是“在完整上下文和当前问题共同作用下”的句子向量

这正是它比普通 embedding 模型更适合压缩任务的地方。

---

## 8. `score_sentences`

```python
def score_sentences(self, question, sentences, marked_contexts):
```

### 功能
给一个问题和若干句子打相关性分数。

### 输入
- `question`: 单个问题
- `sentences`: 句子列表
- `marked_contexts`: 每个句子对应一个“该句被标记的上下文”

### 输出
- `similarities`: 每个句子的分数列表
- `s_emb`: 每个句子的上下文感知向量

### 内部逻辑

#### 先编码问题
```python
q_emb = self.encode_question([question])
```

得到 `[1, d]`

#### 再编码每个被标记句子
```python
s_emb = self.encode_marked_contexts([question] * len(marked_contexts), marked_contexts)
```

这里要把同一个问题复制多份，因为每个句子都要在“问题 + 对应 marked context”下编码一次。

#### 再计算相似度
```python
sims = torch.matmul(q_emb, s_emb.T).squeeze(0)
```

因为前面已经 normalize，所以这里点积就是 cosine similarity。

### 为什么这个接口很重要
它相当于把整个模型包装成了一个“句子打分器”，方便后续 pipeline 直接调用。

---

## 9. `contrastive_loss`

```python
def contrastive_loss(self, questions, positive_marked_contexts, negative_marked_contexts):
```

### 功能
计算对比学习损失，用来训练编码器。

### 训练目标
让模型学会：

- 问题向量更接近正样本句
- 问题向量远离负样本句

### 输入
- `questions`: 长度 B
- `positive_marked_contexts`: 长度 B
- `negative_marked_contexts`: 长度 B，每个样本可以有多个负例

### 执行过程

#### （1）编码问题
```python
q_emb = self.encode_question(questions)
```

#### （2）编码正样本
```python
pos_emb = self.encode_marked_contexts(questions, positive_marked_contexts)
```

#### （3）把所有负样本展开
```python
flat_neg_questions
flat_negs
```

因为原始输入是“每个问题对应多个负样本”，但编码器更适合批量处理平铺后的列表。

#### （4）编码负样本
```python
neg_emb = self.encode_marked_contexts(flat_neg_questions, flat_negs)
```

#### （5）把正负样本拼接成候选池
```python
candidates = torch.cat([pos_emb, neg_emb], dim=0)
```

#### （6）计算 query 对所有候选的相似度矩阵
```python
logits = torch.matmul(q_emb, candidates.T) / self.config.temperature
```

#### （7）构造监督目标
```python
targets = torch.arange(len(questions), device=self.device)
```

意思是：对第 i 个问题来说，前 B 个候选中的第 i 个就是它的正样本。

#### （8）做交叉熵
```python
return F.cross_entropy(logits, targets)
```

### 为什么这里用 InfoNCE
因为这个任务本质上就是一个“排序学习 / 对比学习”问题：

- 正句更相关
- 负句不相关

InfoNCE 很适合这种场景，而且实现简单、效果稳。

---

## 10. `split_sentences`

```python
def split_sentences(text: str) -> List[str]:
```

### 功能
把文本切成句子。

### 为什么必须统一
整个工程里：

- 训练数据构造
- 推理时句子打分
- 压缩选择

都必须基于同一套切句逻辑。  
否则会出现：

- 训练时一套句子边界
- 推理时另一套句子边界

最后导致标记句子找不到、相似度数量不一致等问题。

---

## 11. `build_marked_context`

```python
def build_marked_context(sentences, target_index, marker_start, marker_end):
```

### 功能
给定句子列表和目标句下标，构造“带标记上下文”。

例如：

原句列表：

```text
A, B, C
```

如果目标句是 `B`，构造后：

```text
A <sent_start>B<sent_end> C
```

### 为什么这样设计
因为模型不能天然知道“你要给哪一句打分”。  
所以必须显式地在上下文里圈出目标句。

---

## 12. `build_marked_context_from_text`

```python
def build_marked_context_from_text(context, target_sentence, marker_start, marker_end):
```

### 功能
从原始文本和目标句文本出发，自动构造带标记上下文。

### 执行过程
1. 先切句
2. 找到目标句在第几句
3. 调用 `build_marked_context`

### 这个函数主要给谁用
主要是给训练脚本用。  
因为训练数据通常存的是：

- 完整 context
- positive_sentence
- negative_sentence

而不是存目标句下标。

---

# 四、`train_context_aware_encoder.py` 详细说明

---

## 1. 这个脚本总体功能

这个脚本的目标是：

> **把训练数据中的 question / positive / negative 样本，变成可训练的 marked-context 对比学习任务，并训练一个句子编码器。**

---

## 2. `CQRDataset`

```python
class CQRDataset(Dataset):
```

### 功能
读取 JSONL 格式训练集，并把原始样本转换成模型需要的训练格式。

### 输入样本格式
每一行像这样：

```json
{
  "question": "...",
  "context": "...",
  "positive_sentence": "...",
  "negative_sentences": ["...", "..."]
}
```

### 为什么这样设计
因为你的任务不是普通句对相似度，而是：

- 同一个上下文里
- 有一个与问题相关的正句
- 有若干与问题不相关的负句

这和 CPC / CQR 风格的数据组织是一致的。

---

## 3. `CQRDataset.__init__`

### 做了什么
对每条样本：

#### 正句
```python
pos_marked = build_marked_context_from_text(...)
```

把正句插回上下文并加 marker。

#### 负句
```python
neg_marked = [...]
```

把每个负句也做同样处理。

### 最终存储
每条训练样本变成：

```python
{
    "question": ...,
    "positive_marked_context": ...,
    "negative_marked_contexts": ...
}
```

也就是说，训练数据在进入模型前，就已经转成“模型可直接使用的上下文感知格式”。

---

## 4. `__len__` 和 `__getitem__`

这两个是标准 PyTorch Dataset 接口：

- `__len__`：返回样本数
- `__getitem__`：返回单个样本

---

## 5. `collate_fn`

```python
def collate_fn(batch):
```

### 功能
把一个 batch 的样本整理成列表形式。

返回：

- `questions`
- `positive_marked_contexts`
- `negative_marked_contexts`

### 为什么单独写
因为负样本是“每个样本一个列表”，不是规则张量。  
所以默认 collate 可能不方便，手写更稳。

---

## 6. `main`

这是整个训练入口。

### （1）解析参数
包括：

- 训练文件
- 输出目录
- backbone 名称
- epoch
- batch size
- learning rate
- max_length
- temperature

这些都对应模型和训练超参数。

---

### （2）固定随机种子
```python
random.seed(42)
np.random.seed(42)
torch.manual_seed(42)
```

作用：
- 保持实验可复现
- 论文实验更稳定

---

### （3）构造配置和模型
```python
config = ContextAwareEncoderConfig(...)
model = ContextAwareSentenceEncoder(config)
```

---

### （4）构造数据集和 DataLoader
```python
dataset = CQRDataset(...)
loader = DataLoader(...)
```

---

### （5）构造优化器
```python
optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr)
```

AdamW 是 Transformer 微调里很常见的选择。

---

### （6）训练循环
核心就三步：

```python
loss = model.contrastive_loss(...)
loss.backward()
optimizer.step()
```

也就是说，这里训练的其实就是前面模型里定义好的 InfoNCE / 对比学习目标。

---

### （7）保存模型
训练完后保存三类东西：

#### a. 编码器参数
```python
model.encoder.save_pretrained(output_dir)
```

#### b. tokenizer
```python
model.tokenizer.save_pretrained(output_dir)
```

#### c. 配置
```python
encoder_config.json
```

### 为什么要分开保存
这样后面推理时就能重新加载：

- 模型权重
- tokenizer
- marker token 配置
- 设备、max_length 等参数

---

# 五、这部分代码的整体运行逻辑

如果把 `context_aware_encoder_model` 这一层串起来，它的完整流程是：

### 训练阶段
1. 读取 `question/context/positive/negative`
2. 把正句和负句都插回上下文并加 marker
3. 编码“问题 + 带标记上下文”
4. 提取 marker 区间内的句子表示
5. 用对比学习训练：
   - 问题更接近正句
   - 问题远离负句
6. 保存训练好的句子编码器

### 推理阶段
1. 输入一个问题和一段上下文
2. 切成若干句子
3. 依次给每句构造 marked context
4. 编码得到每句的 sentence-in-context embedding
5. 计算每句和问题的相似度
6. 输出句子分数给后续 pipeline 使用

---

# 六、为什么这一层要这样写

可以总结成三点：

### 1. 因为你的任务不是普通句向量匹配
如果直接拿 BGE / E5 去编码句子，它只能得到“孤立句向量”。

但你真正想知道的是：

> 这个句子在当前问题和当前上下文中，是否重要？

所以必须做 context-aware 编码。

---

### 2. 因为句子重要性往往依赖上下文
例如：
- 指代句本身没实体，但上下文能解释它指什么
- 过渡句本身没答案，但连接了推理链
- 因果句单独看普通，放回上下文里很关键

所以把句子放回上下文里编码，是合理且必要的。

---

### 3. 因为对比学习最适合句子级相关性训练
你的训练信号本质就是：

- 这句相关
- 这几句不相关

这非常适合用 InfoNCE / 对比学习建模，而不一定非得上复杂的生成式损失。

---

# 七、这部分代码在整个项目里的位置

它负责的是：

> **从“原始上下文”到“句子相关性分数”**

后面的压缩率预测和句子选择，都依赖它输出的分数分布。

所以你可以把 `context_aware_encoder_model` 理解成整个系统的：

> **句子打分模块 / relevance scorer**

