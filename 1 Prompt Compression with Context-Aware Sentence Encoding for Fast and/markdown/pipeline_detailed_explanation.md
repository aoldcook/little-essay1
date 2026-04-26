# `pipeline` 代码详解

这份说明面向你当前项目中的 **`pipeline`** 文件夹，重点解释：

- 总体功能是什么
- 每个类 / 函数在做什么
- 代码运行顺序是什么
- 为什么要这样设计

这个文件夹的核心作用是：

> **把训练好的句子编码器、预算预测器和句子选择逻辑串起来，真正输出压缩后的上下文。**

如果说 `context_aware_encoder_model` 解决的是：

> 哪一句重要？

那么 `pipeline` 解决的是：

> 在给定问题下，如何把“句子打分 → 压缩率预测 → 句子选择 → 压缩文本输出”整条链跑通？

---

## 一、`pipeline` 的总体功能

当前 `pipeline` 中最核心的文件是：

- `compression_pipeline.py`

它相当于整个项目中的“总调度器”。

### 它负责做三件事

1. **调用上下文感知句子编码器打分**
2. **调用预算预测器预测 target ratio**
3. **在预算约束下做句子选择并生成压缩结果**

换句话说，它不是训练代码，而是：

> **推理 / 压缩执行代码**

---

# 二、`compression_pipeline.py` 总体结构

这个文件里最关键的内容有三部分：

1. `select_with_mmr`
2. `BudgetPredictorAdapter`
3. `ContextAwareCompressor`

你可以把它们理解成三层：

- **选择器层**：`select_with_mmr`
- **预算适配层**：`BudgetPredictorAdapter`
- **系统总控层**：`ContextAwareCompressor`

---

# 三、函数与类详细说明

---

## 1. `select_with_mmr`

```python
def select_with_mmr(
    similarities,
    sentence_embeddings,
    sentences,
    target_ratio,
    lambda_relevance=0.7,
):
```

### 功能
这是一个 **MMR（Maximal Marginal Relevance）句子选择器**。

它的作用不是简单地取最高分句子，而是在预算约束下同时考虑：

- 句子和问题的相关性
- 句子之间的冗余度

### 为什么需要它
如果你只按相关性排序直接取 top 句，容易出现：

- 连续选中几个意思很像的句子
- 浪费预算在重复信息上
- 整体覆盖度变差

MMR 的核心思想就是：

> **既要选相关的句子，也要避免选过于重复的句子。**

---

### 输入参数解释

#### `similarities`
每个句子和问题的相关性分数。

这是上一步 context-aware encoder 的输出。

#### `sentence_embeddings`
每个句子的向量表示，用于计算句间相似度。

#### `sentences`
句子列表。

#### `target_ratio`
压缩率，例如 0.4 表示保留大约 40% 长度。

#### `lambda_relevance`
控制“相关性”和“去冗余”的权重平衡。

- 越接近 1：越偏向相关性
- 越小：越偏向多样性 / 去冗余

---

### 内部逻辑逐步说明

#### 第一步：算预算
```python
total_len = sum(len(s) for s in sentences)
budget = max(1, int(total_len * target_ratio))
```

这里是按**字符长度**近似预算，而不是按 token。

### 为什么这样写
因为这是中文最小可用版本里比较稳妥的近似方式：

- 比 `split()` 算词数更适合中文
- 不需要依赖 tokenizer 再重复算 token 数
- 实现简单，便于先跑通实验

---

#### 第二步：初始化
```python
selected = []
remaining = set(range(len(sentences)))
used_len = 0
```

- `selected`：已选句子索引
- `remaining`：还没选的句子
- `used_len`：已经使用的预算长度

---

#### 第三步：准备数值数组
```python
sims = np.asarray(similarities, dtype=float)
embs = sentence_embeddings.detach().cpu().numpy()
```

把 PyTorch tensor 转成 numpy，后面方便做快速计算。

---

#### 第四步：循环选句
```python
while remaining:
```

每轮都从剩余句子里找一个“最值得加入”的句子。

---

#### 第五步：预算检查
```python
if used_len + sent_len > budget:
    continue
```

如果加入这个句子会超出预算，就跳过。

### 这一步的意义
它保证最终选出来的句子集合一定满足压缩预算约束。

---

#### 第六步：计算 MMR 分数

如果还没选任何句子：

```python
score = rel
```

那第一句就只按相关性来选。

如果已经选过句子：

```python
red = max(float(np.dot(embs[i], embs[j])) for j in selected)
score = lambda_relevance * rel - (1.0 - lambda_relevance) * red
```

这里：

- `rel`：该句和问题的相关性
- `red`：该句和已选句子中最相似的一句之间的相似度
- `score`：综合分数

### 为什么是这个公式
这就是经典 MMR 的思想：

\[
\text{MMR}(s_i)
=
\lambda \cdot \text{Rel}(q, s_i)
-
(1-\lambda)\cdot \max_{s_j \in S}\text{Sim}(s_i, s_j)
\]

解释为：

- 第一项鼓励选“对问题有用”的句子
- 第二项惩罚选“和已选句子重复”的句子

---

#### 第七步：加入最佳句子
```python
selected.append(best_idx)
remaining.remove(best_idx)
used_len += len(sentences[best_idx])
```

---

#### 第八步：按原文顺序恢复
```python
return sorted(selected)
```

这里返回的是索引，并按索引排序，相当于恢复原文顺序。

### 为什么不是按选择顺序输出
因为 MMR 选择顺序不一定等于原文顺序。  
直接按选择顺序拼接会导致阅读跳跃。

按原索引排序能更自然地恢复语义流。

---

## 2. `BudgetPredictorAdapter`

```python
class BudgetPredictorAdapter:
```

### 功能
这是一个“适配器”，用来把你已经做好的预算预测器模块接进当前 pipeline。

它的作用不是训练预算模型，而是：

> **加载训练好的预算预测器，并把它包装成统一接口。**

---

### 为什么要单独写 Adapter
因为当前预算预测器代码原本是独立工程。  
而 pipeline 只需要做一件事：

- 给定 `question + context + similarities`
- 返回一个 `target_ratio`

所以最好的方式就是写个 Adapter，把已有工程的细节封装起来。

这样 pipeline 就不用关心：

- 预算模型内部怎么实现
- 用的是 MLP 还是别的模型
- 特征怎么提

它只调用一个方法：

```python
predict_ratio(...)
```

---

## 3. `BudgetPredictorAdapter.__init__`

```python
def __init__(self, budget_model_dir: str):
```

### 功能
加载训练好的预算预测器。

---

### 逻辑分解

#### （1）是否启用预算预测器
```python
self.enabled = budget_model_dir is not None
```

如果没传目录，就表示不用预算模型，后面会退化成固定 ratio。

这一步使 pipeline 更灵活：

- 可用 learned budget
- 也可不用 learned budget

---

#### （2）动态导入旧工程代码
```python
import sys
budget_dir = str(Path(budget_model_dir).resolve().parent)
if budget_dir not in sys.path:
    sys.path.insert(0, budget_dir)
```

然后：

```python
from budget_features import build_budget_features, features_to_vector
from budget_model import BudgetConfig, BudgetPredictorMLP, class_to_ratio
```

### 为什么这样写
因为预算预测器原本不是安装成正式包的，而是项目里的独立脚本模块。  
所以这里通过临时加入路径的方式，把它当包来导入。

这是工程上很常见的“低侵入复用旧代码”写法。

---

#### （3）读取 metadata
```python
with (model_dir / "metadata.json").open(...)
```

这里会恢复：

- ratio buckets
- 输入维度
- 隐藏层结构

### 为什么要读 metadata
因为推理时必须知道训练时模型的结构，否则无法正确重建网络。

---

#### （4）重建模型
```python
config = BudgetConfig(...)
self.model = BudgetPredictorMLP(config)
self.model.load_state_dict(...)
self.model.eval()
```

---

## 4. `BudgetPredictorAdapter.predict_ratio`

```python
def predict_ratio(self, question, context, similarities, fallback_ratio=0.4):
```

### 功能
对一个新样本预测压缩率。

### 逻辑

#### 如果没启用预算模型
```python
if not self.enabled:
    return fallback_ratio
```

### 这一步的意义
让 pipeline 有一个安全兜底：

- 没训练预算模型也能运行
- 默认压缩率设成 0.4

---

#### 否则先提特征
```python
feats = self.build_budget_features(...)
x = torch.tensor(self.features_to_vector(feats)).unsqueeze(0)
```

也就是把：

- 问题
- 上下文
- 句子分数分布

变成预算模型可以吃的输入向量。

---

#### 再做分类预测
```python
logits = self.model(x)
pred_cls = int(torch.argmax(logits, dim=1).item())
```

#### 最后映射回 ratio
```python
return float(self.class_to_ratio(pred_cls, self.ratio_buckets))
```

---

## 5. `ContextAwareCompressor`

```python
class ContextAwareCompressor:
```

### 功能
这是整个 pipeline 的总控类。

它负责把三件事串起来：

1. 加载训练好的句子编码器
2. 可选地加载预算预测器
3. 提供 `compress()` 接口，一次完成打分、预算预测、选择和输出

你可以把它理解成：

> **你整个压缩系统在推理时的统一入口**

---

## 6. `ContextAwareCompressor.__init__`

```python
def __init__(self, encoder_dir, budget_model_dir=None, device=None):
```

### 功能
加载所有推理时需要的模型。

---

### 逻辑分解

#### （1）确定设备
```python
device = device or ("cuda" if torch.cuda.is_available() else "cpu")
```

#### （2）读取句子编码器配置
```python
with (Path(encoder_dir) / "encoder_config.json").open(...)
```

### 为什么必须读这个文件
因为训练时保存的不仅是权重，还有：

- model_name
- marker token
- max_length
- temperature

推理时必须保持一致。

---

#### （3）根据配置重建 encoder
```python
cfg = ContextAwareEncoderConfig(**cfg_dict)
self.encoder = ContextAwareSentenceEncoder(cfg)
```

---

#### （4）加载训练好的模型和 tokenizer
```python
self.encoder.encoder = self.encoder.encoder.from_pretrained(encoder_dir)
self.encoder.tokenizer = self.encoder.tokenizer.from_pretrained(encoder_dir)
```

### 为什么 tokenizer 也要一起加载
因为训练时你往 tokenizer 里加了：

- `<sent_start>`
- `<sent_end>`

所以推理时必须加载同一个 tokenizer，否则 marker token 可能丢失。

---

#### （5）重新记录 marker token id
```python
self.encoder.start_id = ...
self.encoder.end_id = ...
```

---

#### （6）切换到推理模式
```python
self.encoder.eval()
```

---

#### （7）可选加载预算预测器
```python
self.budget_selector = BudgetPredictorAdapter(...) if budget_model_dir else None
```

这使得整个系统支持两种模式：

- 只用句子编码器 + 固定 ratio
- 句子编码器 + learned budget

---

## 7. `score_context`

```python
def score_context(self, question, context):
```

### 功能
对一段上下文中的所有句子打分。

### 输出
返回三样东西：

- `sentences`
- `similarities`
- `sent_embs`

### 具体流程

#### 第一步：切句
```python
sentences = split_sentences(context)
```

#### 第二步：为每个句子构造 marked context
```python
marked_contexts = [
    build_marked_context(sentences, i, ...)
    for i in range(len(sentences))
]
```

### 为什么每个句子都要构造一份 marked context
因为每次模型只能明确关注一个被 marker 圈出来的目标句。  
所以如果有 n 个句子，就要构造 n 个“目标句不同的上下文版本”。

#### 第三步：调用 encoder 打分
```python
similarities, sent_embs = self.encoder.score_sentences(...)
```

### 这一步的结果是什么
它给出了：

- 每个句子的 query-aware 分数
- 每个句子的上下文感知向量

这两个量后面都会用到：

- 分数给预算预测和句子选择
- 向量给 MMR 的冗余计算

---

## 8. `compress`

```python
def compress(
    self,
    question,
    context,
    target_ratio=None,
    lambda_relevance=0.7,
    fallback_ratio=0.4,
):
```

### 功能
这是整个系统最重要的对外接口。

它完成：

- 句子打分
- target ratio 决策
- 预算约束下句子选择
- 输出压缩结果

---

### 运行逻辑

#### 第一步：先打分
```python
sentences, similarities, sent_embs = self.score_context(question, context)
```

---

#### 第二步：确定压缩率
如果你手动传了 `target_ratio`，就用手动值。

否则：

- 如果有预算预测器，就让它预测
- 如果没有，就用默认 `fallback_ratio`

代码就是：

```python
if target_ratio is None:
    if self.budget_selector is not None:
        target_ratio = self.budget_selector.predict_ratio(...)
    else:
        target_ratio = fallback_ratio
```

### 为什么这样设计
因为这能兼容三种实验模式：

1. **固定 ratio**
2. **启发式 ratio**
3. **学习式 budget ratio**

这对论文实验和消融非常重要。

---

#### 第三步：在预算内选句
```python
selected_idx = select_with_mmr(...)
```

也就是调用前面的 MMR 选择器。

---

#### 第四步：拼接压缩结果
```python
compressed = "".join(sentences[i] for i in selected_idx)
```

### 为什么这里直接按索引拼接
因为 `selected_idx` 已经在 `select_with_mmr()` 里排序过，所以这里拼接后基本就是原文顺序。

---

#### 第五步：返回完整结果
```python
return {
    "question": question,
    "target_ratio": float(target_ratio),
    "sentences": sentences,
    "similarities": similarities,
    "selected_indices": selected_idx,
    "compressed_context": compressed,
}
```

### 为什么返回这么多信息
因为这不仅方便推理，也方便：

- 调试
- 可视化分析
- 写实验表格
- 做消融实验

例如你可以直接用这些字段分析：

- 预算预测是否合理
- 哪些句子被选中
- 分数分布形状怎样
- 压缩文本长度是多少

---

# 四、整个 `pipeline` 的运行顺序

如果把 `pipeline` 这一层完整串起来，它的推理流程就是：

### 1. 输入
- 问题 `question`
- 长文本 `context`

### 2. 切句
把上下文切成句子列表

### 3. 构造每个句子的 marked context
对每个句子构造一份“该句被 marker 圈出来的上下文”

### 4. 用训练好的句子编码器打分
得到：
- `similarities`
- `sentence embeddings`

### 5. 预测压缩率
如果启用预算预测器，则根据：
- 问题
- 上下文
- 句子分数分布

预测 `target_ratio`

否则使用固定 ratio

### 6. 用 MMR 在预算内选句
既考虑：
- relevance
- diversity / redundancy

又满足长度预算

### 7. 输出压缩结果
返回：
- 压缩后的文本
- 被选中的句子编号
- 每句分数
- 使用的 target ratio

---

# 五、为什么 pipeline 要这样写

---

## 1. 因为“打分”和“选择”不是一回事

很多初学实现会直接：

- 给句子打分
- 取 top-k
- 拼接完事

但这样的问题是：

- 会选到很多重复句
- 覆盖度差
- 难以适应不同预算

所以 pipeline 把这两件事分开：

- **句子编码器负责打分**
- **选择器负责在预算内做更合理的集合选择**

这比简单 top-k 更稳。

---

## 2. 因为预算本身应该是动态的

固定 ratio 很方便，但不够灵活。

有的样本：

- 信息集中
- 20% 就够

有的样本：

- 信息分散
- 需要 50% 才安全

所以 pipeline 专门留了 `BudgetPredictorAdapter` 这一层，让系统支持动态压缩率。

---

## 3. 因为工程上要尽量模块解耦

当前写法把系统拆成：

- 句子编码器模块
- 预算预测模块
- 选择器模块
- 总控 pipeline

这样后续做实验很方便：

- 换 encoder，不动预算模型
- 换预算模型，不动 encoder
- 换选择策略，不动前两层

这对论文消融实验尤其重要。

---

# 六、你在实验里可以怎么用这份 pipeline

当前这份 pipeline 支持很自然的几种实验方式：

---

## 情况 1：只测句子编码器 + 固定压缩率

```python
compressor.compress(question, context, target_ratio=0.4)
```

这时：
- 不依赖预算预测器
- 只测试上下文感知句子编码器效果

---

## 情况 2：句子编码器 + 学习式预算预测

```python
compressor = ContextAwareCompressor(
    encoder_dir="...",
    budget_model_dir="..."
)
```

然后：

```python
compressor.compress(question, context)
```

这时：
- 自动预测 `target_ratio`
- 自动执行完整压缩链

---

## 情况 3：只分析中间结果
例如：

```python
sentences, similarities, sent_embs = compressor.score_context(question, context)
```

这样你就能单独分析：

- 分数分布
- embedding 质量
- 哪些句子高分

这很适合论文里的案例分析。

---

# 七、整个 `pipeline` 在你项目中的角色

一句话概括：

> **`pipeline` 是把“训练好的句子编码器”和“训练好的预算预测器”真正落地成压缩系统的执行层。**

如果说：

- `context_aware_encoder_model` 是“句子重要性判断层”
- `target_ratio_model` 是“压缩预算决策层”

那么：

- `pipeline` 就是“压缩执行层”

它把三者真正串起来，变成一个完整可运行的压缩系统。

---

# 八、最终总结

`pipeline/compression_pipeline.py` 主要做了三件事：

### 1. 调用句子编码器打分
解决：
> 哪一句重要？

### 2. 调用预算预测器给出压缩率
解决：
> 压多少？

### 3. 用 MMR 在预算内选择句子
解决：
> 在预算约束下怎么选得更稳、更不重复？

所以这部分代码本质上就是你整个论文方法在推理阶段的“系统化实现”。
