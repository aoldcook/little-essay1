# 预算预测器示例工程代码讲解

本文档围绕示例工程中的三个核心文件展开说明：

- `budget_features.py`：特征提取
- `budget_model.py`：模型定义
- `predict_budget.py`：推理脚本

这三部分分别对应三件事：

1. 把“问题 + 上下文 + 相似度序列”转换成固定长度的数值特征向量。
2. 用一个小型 MLP 学习“特征 → 压缩率类别”的映射关系。
3. 在新样本上加载训练好的模型，输出推荐的 `target_ratio`。

---

# 一、特征提取：`budget_features.py`

这一部分的目标是：

> 把一个样本压缩成适合小模型学习的数值特征。

因为小 MLP 不能直接处理完整文本和变长相似度序列，所以这里先做人工特征工程。

## 1. 句子切分：`split_sentences`

```python
def split_sentences(text: str) -> List[str]:
    sentences = re.split(r'(?<=[。！？.!?])\s*', text.strip())
    return [s.strip() for s in sentences if s.strip()]
```

作用是把上下文切成句子列表。

例如：

```python
"CPC是一种压缩方法。它按句子筛选上下文。"
```

会变成：

```python
["CPC是一种压缩方法。", "它按句子筛选上下文。"]
```

这里很重要，因为后面很多特征都依赖“句子级统计”，例如：

- 句子数
- 平均句长
- 最长句长度
- 句长标准差

## 2. 问题类型识别：`detect_question_type`

```python
QUESTION_TYPES = ["definition", "cause", "comparison", "procedure", "numeric", "factoid", "other"]
```

这个函数会根据关键词，把问题粗分成几类：

- `definition`：定义类，如“是什么”
- `cause`：原因类，如“为什么”
- `comparison`：比较类，如“区别、异同”
- `procedure`：流程类，如“如何、步骤”
- `numeric`：数值类，如“多少、比例”
- `factoid`：事实类，如“谁、何时、哪里”
- `other`

例如：

- `什么是CPC？` → `definition`
- `CPC为什么有效？` → `cause`
- `CPC和LLMLingua有什么区别？` → `comparison`

为什么这个特征有用：

因为不同问题类型通常对应不同预算需求。比如：

- 定义题通常只要少数关键句
- 比较题往往需要同时保留两边信息
- 原因题、流程题往往需要更长的支撑链

所以这里实际上是在给预算预测器提供“任务复杂度”的先验。

## 3. 实体数量：`count_entities`

```python
ENTITY_PATTERN = re.compile(r"[A-Za-z][A-Za-z0-9\-]{1,}|[\u4e00-\u9fff]{2,8}")
```

这个函数用一个比较粗糙的规则，从问题里抽可能的实体短语并计数。

例如：

- `CPC和CQR之间有什么关系？`

会大致数出两个实体。

为什么这个特征有用：

如果一个问题涉及多个实体、多个对象、多个方法，通常意味着：

- 需要保留更多上下文
- 压缩不能太激进

这也是预算预测的重要信号。

## 4. softmax 和 entropy：把相似度分布“概率化”

```python
def softmax(x: np.ndarray) -> np.ndarray:
    ...
```

在 `build_budget_features` 中，会把相似度序列 `similarities` 排序后做 softmax：

```python
probs = softmax(sims)
entropy = float(-(probs * np.log(probs + 1e-12)).sum())
entropy_norm = float(entropy / math.log(n + 1e-12)) if n > 1 else 0.0
```

这是非常关键的一步。

### 直觉

如果相似度分布非常集中，例如：

```python
[0.95, 0.60, 0.20, 0.10]
```

说明前一两句特别重要，预算可以更小。

如果分布很平，例如：

```python
[0.78, 0.75, 0.72, 0.69]
```

说明很多句子都差不多重要，预算应该更大。

### 为什么用 entropy

熵衡量的是“分布有多分散”：

- 熵低：重要性集中
- 熵高：重要性分散

这比只看 top1 分数更稳。

## 5. 核心特征函数：`build_budget_features`

输入：

```python
build_budget_features(question, context, similarities)
```

输出：

```python
Dict[str, float]
```

也就是一组命名特征。下面按类别解释。

### A. 相似度分布特征

#### `num_sentences`

```python
"num_sentences": float(n)
```

句子数。句子越多，通常上下文越长，也更可能需要更大预算。

#### `sim_max / sim_min / sim_mean / sim_std / sim_range`

```python
"sim_max": float(sims[0]),
"sim_min": float(sims[-1]),
"sim_mean": float(np.mean(sims)),
"sim_std": float(np.std(sims)),
"sim_range": float(sims[0] - sims[-1]),
```

分别表示：

- 最大相似度
- 最小相似度
- 平均相似度
- 标准差
- 极差

这些特征描述整体分布形状。例如：

- `sim_max` 高，说明至少有很强的匹配句
- `sim_std` 大，说明句子间重要性差异大
- `sim_range` 大，说明高低句区分明显

#### `top1_gap / top2_gap`

```python
"top1_gap": float(sims[0] - sims[1]) if n >= 2 else 0.0,
"top2_gap": float(sims[1] - sims[2]) if n >= 3 else 0.0,
```

这两个特征很有解释性：

- `top1_gap` 大：第一句明显比第二句重要
- `top2_gap` 大：前两句后面开始掉下去

这在捕捉“头部是否陡峭”。如果头部非常陡，往往说明少量句子就够了。

#### `top3_mass / top20_mass / top50_mass`

```python
"top3_mass": _safe_ratio(float(sims[:k3].sum()), float(sims.sum()) + 1e-9),
"top20_mass": _safe_ratio(float(sims[:k20].sum()), float(sims.sum()) + 1e-9),
"top50_mass": _safe_ratio(float(sims[:k50].sum()), float(sims.sum()) + 1e-9),
```

这是“头部质量占比”。

例如 `top20_mass` 表示：

> 前 20% 句子的总相似度，占全部句子相似度总和的多少。

意义是：

- 如果前 20% 就占了很大比例，说明信息集中
- 如果前 20% 占比不高，说明信息分散

它本质上在回答：

> 重要信息是不是集中在少数句子里？

#### `high_relevance_ratio / mid_relevance_ratio`

```python
"high_relevance_ratio": float(np.mean(sims >= 0.7)),
"mid_relevance_ratio": float(np.mean(sims >= 0.5)),
```

表示：

- 高相似度句子占多少
- 中等以上相似度句子占多少

这两个特征反映“值得保留的句子有多少”。如果很多句子都在 0.7 以上，就不能压太狠。

#### `entropy_norm`

```python
"entropy_norm": entropy_norm,
```

归一化熵，是描述“相关性分布是否分散”的综合特征，非常重要。

#### `front_avg_drop`

```python
"front_avg_drop": float(np.mean(np.abs(np.diff(sims[: min(5, n)])))) if n >= 2 else 0.0,
```

这个特征看前几句分数下降得快不快。如果前 5 句分数落差大，说明信息集中在 very top 的几句里。

### B. 问题和上下文长度特征

```python
"question_char_len": float(len(question)),
"context_char_len": float(len(context)),
```

这里不是 token 长度，而是字符长度。为什么有用：

- 问题越长，通常越复杂
- 上下文越长，通常冗余和复杂性都更高

### C. 句长分布特征

```python
"avg_sentence_char_len": float(np.mean(sentence_lens)),
"max_sentence_char_len": float(np.max(sentence_lens)),
"sentence_len_std": float(np.std(sentence_lens)),
```

描述上下文句子的长度结构。它们有两个潜在作用：

1. 粗略反映文本风格，例如论文句子长、定义性文本句子短。
2. 反映压缩难度，如果句子都很长且差异很大，说明靠句级截断可能不够细。

### D. 问题结构特征

```python
"question_entity_count": float(count_entities(question)),
"is_multi_hop_like": float(any(k in question for k in ["为什么", "如何", "比较", "区别", "异同", "影响", "关系"])),
```

这里是在显式编码：

- 问题涉及多少实体
- 问题是否像多跳 / 比较 / 因果问题

`is_multi_hop_like` 虽然很粗糙，但很实用。因为这种问题常常不能只靠一两句回答。

### E. one-hot 的问题类别特征

```python
for qt in QUESTION_TYPES:
    features[f"qtype_{qt}"] = 1.0 if q_type == qt else 0.0
```

例如如果当前问题是 `cause`，就会生成：

```python
qtype_definition = 0
qtype_cause = 1
qtype_comparison = 0
...
```

这就是 one-hot 编码。

神经网络不能直接处理字符串 `"cause"`，所以要转成数字。

## 6. `FEATURE_ORDER` 和 `features_to_vector`

```python
FEATURE_ORDER = list(build_budget_features(...).keys())
```

这句的作用是固定特征顺序。

因为字典虽然有 key-value，但模型吃的是向量，不是字典。所以必须保证：

> 训练时第 1 维是什么，推理时第 1 维也必须还是它。

然后：

```python
def features_to_vector(features: Dict[str, float]) -> np.ndarray:
    return np.array([float(features[name]) for name in FEATURE_ORDER], dtype=np.float32)
```

就是把特征字典变成一个固定顺序的数值向量。

---

# 二、模型定义：`budget_model.py`

这一部分做三件事：

- 定义配置
- 定义 MLP 网络
- 定义损失函数和类别映射

## 1. 配置类：`BudgetConfig`

```python
@dataclass
class BudgetConfig:
    ratio_buckets: List[float]
    input_dim: int
    hidden_dims: List[int]
```

这个配置类描述模型结构，例如：

- `ratio_buckets = [0.2, 0.3, 0.4, 0.5, 0.6, 0.7]`
- `input_dim = 特征维度`
- `hidden_dims = [64, 32]`

所以这个模型不是输出连续值，而是输出一个压缩率类别。

## 2. 主模型：`BudgetPredictorMLP`

```python
class BudgetPredictorMLP(nn.Module):
```

这是一个多层感知机。

### 网络结构

```python
dims = [config.input_dim] + config.hidden_dims
```

如果输入维度是 28，隐藏层是 `[64, 32]`，那么网络就是：

\[
28 \rightarrow 64 \rightarrow 32 \rightarrow 6
\]

最后的 `6` 是因为有 6 个 ratio bucket。

### 每层的构造

```python
layers.append(nn.Linear(dims[i], dims[i + 1]))
layers.append(nn.ReLU())
layers.append(nn.Dropout(0.1))
```

每个隐藏层包含：

- 线性层
- ReLU 激活
- Dropout 0.1

最后一层：

```python
layers.append(nn.Linear(dims[-1], len(config.ratio_buckets)))
```

输出的是每个 ratio bucket 的一个 logit，例如：

```python
[1.2, 0.4, -0.1, 2.3, 0.7, -0.5]
```

这些还不是概率，要经过 softmax 才是概率。

## 3. 为什么这里做分类，不做回归

因为你的 `label_ratio` 本来就是通过扫描候选桶得到的，比如：

```python
0.2, 0.3, 0.4, 0.5, 0.6, 0.7
```

所以分类更自然。优点是：

- 稳定
- 好训练
- 不容易预测出离谱的中间值
- 和离线打标签方式一致

## 4. 损失函数：`BudgetLoss`

```python
class BudgetLoss(nn.Module):
```

这个设计是示例工程里比较值得注意的一点。

### 普通交叉熵部分

```python
ce_loss = self.ce(logits, targets)
```

先做标准分类训练。

### 额外的“低估预算惩罚”

```python
preds = torch.argmax(logits, dim=1)
under = torch.clamp(targets - preds, min=0).float()
return ce_loss + self.under_penalty * under.mean()
```

这里的逻辑是：

- 如果预测类别比真实类别更小，就算“低估预算”
- 低估越多，罚得越多

例如：

- 真实是 `0.6`
- 你预测成 `0.3`

这很危险，因为说明压得太狠，可能会损坏答案。

而反过来：

- 真实是 `0.3`
- 你预测成 `0.6`

虽然不够省 token，但至少不太会伤语义。

所以这个损失函数体现了任务偏好：

> 预算预测宁可稍微保守，也不要过度压缩。

## 5. `ratio_to_class` 和 `class_to_ratio`

```python
def ratio_to_class(ratio: float, ratio_buckets: List[float]) -> int:
```

把真实 ratio 变成类别索引，例如：

- `0.2 -> 0`
- `0.3 -> 1`
- `0.4 -> 2`

反过来：

```python
def class_to_ratio(cls_idx: int, ratio_buckets: List[float]) -> float:
```

把预测类别再映射回 ratio。训练和推理都需要。

## 6. `build_metadata`

```python
def build_metadata(config: BudgetConfig, feature_order: List[str]) -> Dict:
```

保存模型相关元信息：

- ratio buckets
- 输入维度
- 隐藏层设置
- 特征顺序

这是为了让推理时能按训练时的配置复原模型。

---

# 三、推理脚本：`predict_budget.py`

这个脚本的作用是：

> 加载训练好的预算预测器，对一个新样本预测 `target_ratio`。

## 1. 读取参数

```python
parser.add_argument("--model_dir", type=str, required=True)
parser.add_argument("--input_json", type=str, required=True)
```

你需要提供：

- `model_dir`：训练好的模型目录
- `input_json`：一个待预测样本

## 2. 加载元信息

```python
metadata = load_json(model_dir / "metadata.json")
config = BudgetConfig(
    ratio_buckets=metadata["ratio_buckets"],
    input_dim=metadata["input_dim"],
    hidden_dims=metadata["hidden_dims"],
)
```

这里从 `metadata.json` 恢复模型结构。

为什么要这样做：

因为推理时必须知道：

- 模型输出几个类别
- 输入多少维
- 隐藏层怎么搭

否则没法实例化出一样的网络。

## 3. 重建模型并加载权重

```python
model = BudgetPredictorMLP(config)
model.load_state_dict(torch.load(model_dir / "budget_predictor.pt", map_location="cpu"))
model.eval()
```

步骤是：

1. 按配置创建网络
2. 加载训练好的参数
3. 切到推理模式

`model.eval()` 会让 dropout 失效，保证推理稳定。

## 4. 读取输入样本

```python
sample = load_json(Path(args.input_json))
```

输入 JSON 里应该有：

- `question`
- `context`
- `similarities`

例如：

```json
{
  "question": "CPC为什么有效？",
  "context": "……",
  "similarities": [0.91, 0.88, 0.63, 0.40]
}
```

这里默认你已经有句子级相似度了。也就是说，这个推理脚本只负责“根据分布预测预算”，不负责“现算相似度”。

## 5. 提取特征并向量化

```python
features = build_budget_features(
    question=sample["question"],
    context=sample["context"],
    similarities=sample["similarities"],
)
x = torch.tensor(features_to_vector(features)).unsqueeze(0)
```

这里做了两件事。

### 第一步：提字典特征

输出类似：

```python
{
  "num_sentences": 12,
  "sim_max": 0.92,
  ...
}
```

### 第二步：变成张量

`unsqueeze(0)` 是为了增加 batch 维度。

原来：

\[
(input\_dim,)
\]

变成：

\[
(1,\ input\_dim)
\]

因为神经网络默认按 batch 输入。

## 6. 前向推理

```python
with torch.no_grad():
    logits = model(x)
    pred_cls = int(torch.argmax(logits, dim=1).item())
    pred_ratio = class_to_ratio(pred_cls, config.ratio_buckets)
    probs = torch.softmax(logits, dim=1).squeeze(0).tolist()
```

这里是完整推理流程。

### `logits = model(x)`

输出每个 ratio bucket 的得分。

### `argmax`

选分数最高的类别。

### `class_to_ratio`

把类别转成实际 ratio。

### `softmax`

把 logits 变成各类别概率，方便观察模型信心。

## 7. 打印输出

```python
print("predicted_ratio:", pred_ratio)
print("class_probabilities:")
...
print("top_features:")
...
```

输出三类信息：

### A. 最终预测压缩率

例如：

```python
predicted_ratio: 0.4
```

### B. 各个候选 ratio 的概率

例如：

```python
ratio=0.2 prob=0.12
ratio=0.3 prob=0.21
ratio=0.4 prob=0.45
...
```

这能让你看模型是在“明显偏向一个桶”，还是“多个桶都差不多”。

### C. 前几个特征值

```python
for name in FEATURE_ORDER[:10]:
```

这里只打印了前 10 个特征，主要用于调试。

严格说，这里打印的是“特征表前 10 个”，不是“最重要的 top 特征”。如果以后想看真正的特征重要性，需要额外做：

- SHAP
- permutation importance
- 或训练一个树模型比较

---

# 四、把三部分连起来看

整个流程非常清楚。

## 训练前准备

你先有一个样本：

- `question`
- `context`
- `similarities`
- `label_ratio`

## 特征提取

`budget_features.py` 做的是：

\[
(question,\ context,\ similarities) \rightarrow x
\]

把原始样本变成向量。

## 模型学习

`budget_model.py` 做的是：

\[
x \rightarrow \text{ratio class}
\]

学习从特征到预算的映射。

## 推理

`predict_budget.py` 做的是：

\[
(question,\ context,\ similarities) \rightarrow \hat r
\]

输出新样本该保留多少比例。

---

# 五、这套代码的功能边界

这份示例工程现在已经能做：

- 根据相似度分布预测压缩预算
- 替代手工 `find_target_ratio`
- 跑通一个小模型方案

但它暂时还没做这些事情：

- 不负责计算 `similarities`
- 不负责选择句子
- 不负责直接调用 LLM 评估答案质量
- 不负责端到端训练排序器和预算器

也就是说，它是：

> 一个预算预测模块

而不是完整的压缩系统。

---

# 六、阅读这三份代码时最应该抓住的主线

如果用一句话总结：

## `budget_features.py`

把“压缩难不难”变成数字描述。

## `budget_model.py`

学习这些数字描述和“最小安全预算”之间的关系。

## `predict_budget.py`

在新样本上调用这个关系，输出一个可用的 `target_ratio`。

---

# 七、阅读时最值得注意的小点

1. `entropy_norm`、`top20_mass`、`high_relevance_ratio` 这几个特征最接近原始 `find_target_ratio` 的数学直觉，是预算预测最核心的一组。
2. `BudgetLoss` 里“低估预算额外惩罚”这个设计很契合任务目标，论文里也容易解释为什么要这么做。
3. `predict_budget.py` 当前虽然读取了 `metadata["feature_order"]`，但实际向量化时还是直接用了 `budget_features.py` 里的 `FEATURE_ORDER`。在当前示例工程里没问题，因为训练和推理来自同一套代码；但以后如果你改了特征表，最好让推理严格使用保存下来的 `feature_order`，会更稳。

---

# 八、一句话总结

这套预算预测器工程的逻辑链条是：

1. 先把问题、上下文和相似度分布提取成一组可解释特征；
2. 再用一个小型分类模型预测“最小安全压缩率”；
3. 最后在新样本上输出推荐的 `target_ratio`，替代手工设计的 `find_target_ratio`。
