# budget_features.py 中 feature 的作用说明

这里的 `feature` 本质上就是：

> **把一条样本“压缩成多少比例才安全”这件事，用一组数值信号表达出来。**

因为后面的预算预测器是个小 MLP，它不能直接理解一整段 `question/context/similarities` 文本，所以 `budget_features.py` 的作用就是先把这些原始输入变成一个**固定长度的数值向量**，再交给模型学习。

---

## 1. feature 在整个流程里的位置

你的预算预测流程其实是：

\[
(question,\ context,\ similarities)
\rightarrow
features
\rightarrow
vector
\rightarrow
MLP
\rightarrow
predicted\ ratio
\]

也就是说：

- `question`：问题
- `context`：原始长文本
- `similarities`：问题和各句子的相关性分数
- `features`：从这三者里提炼出来的统计特征
- `predicted ratio`：最终建议保留比例

所以 `feature` 的作用不是“直接压缩文本”，而是：

> **告诉模型，这个样本是“容易压缩”还是“难压缩”。**

---

## 2. 为什么一定要有 feature

因为每个样本的句子数都不一样、上下文长度都不一样、相似度列表长度也不一样。  
神经网络不能直接吃这种变长结构做一个稳定的小模型预测，所以需要先抽象成固定维度的统计量。

比如：

- 句子数多不多
- 高分句集中不集中
- 问题是不是比较类/因果类
- 上下文是不是很长
- 相关句是不是很多

这些就是模型判断预算的依据。

---

## 3. 这些 feature 具体在判断什么

可以把 `budget_features.py` 里的特征分成 4 类来看。

---

### A. 相似度分布特征

这是最重要的一类，因为它直接反映：

> **关键信息是集中在少数句子里，还是分散在很多句子里。**

例如：

- `sim_max`
- `sim_mean`
- `sim_std`
- `sim_range`
- `top1_gap`
- `top20_mass`
- `high_relevance_ratio`
- `entropy_norm`
- `front_avg_drop`

它们分别在回答这些问题：

- 最高分句子有多突出？
- 前几句是不是拿走了大部分相关性？
- 高相关句子有多少？
- 相似度分布是“尖的”还是“平的”？

如果分布像这样：

```python
[0.95, 0.62, 0.21, 0.10]
```

说明前一两句特别关键，预算可以小一点。

如果分布像这样：

```python
[0.82, 0.80, 0.78, 0.76]
```

说明很多句子都差不多重要，预算应该大一点。

所以这类 feature 的作用是：

> **描述“信息集中度”和“压缩风险”。**

---

### B. 问题复杂度特征

这类特征用来告诉模型：

> **这个问题本身是不是需要更多上下文。**

例如：

- `question_char_len`
- `question_entity_count`
- `is_multi_hop_like`
- `qtype_definition`
- `qtype_cause`
- `qtype_comparison`
- `qtype_procedure`
- `qtype_numeric`
- `qtype_factoid`

它们表达的是：

- 问题长，通常更复杂
- 实体多，说明涉及对象多
- 比较题、因果题、流程题通常比定义题更难压缩
- “为什么/如何/比较”类题通常需要多句支撑，而不是一两句就够

所以这类 feature 的作用是：

> **让模型知道“题型”和“推理复杂度”。**

---

### C. 上下文结构特征

这类特征反映：

> **上下文本身的组织结构和压缩难度。**

例如：

- `num_sentences`
- `context_char_len`
- `avg_sentence_char_len`
- `max_sentence_char_len`
- `sentence_len_std`

它们的意义包括：

- 句子数多，通常上下文更长
- 上下文越长，预算往往越难压得很低
- 句子普遍很长，说明句级抽取的粒度可能较粗
- 句长波动大，说明上下文结构可能不均匀

所以这类 feature 的作用是：

> **告诉模型“原文长什么样、是否容易做句级压缩”。**

---

### D. one-hot 类型特征

这部分是：

```python
features[f"qtype_{qt}"] = 1.0 if q_type == qt else 0.0
```

比如问题是比较题，那可能就是：

```python
qtype_comparison = 1
qtype_definition = 0
qtype_cause = 0
...
```

作用就是把“问题类型”这个离散信息变成数值输入，让模型能用。

---

## 4. feature 不是最终答案，而是“判断依据”

这些 feature 本身不会直接告诉你：

```python
target_ratio = 0.4
```

它们只是给模型提供证据，比如：

- 高分句很集中
- 问题是定义类
- 句子数不多
- 熵很低

模型看到这些 feature 后，才学会：

> 这种样本通常可以压到 0.3 或 0.4

所以你可以把 feature 理解成：

> **预算预测器的输入信号。**

---

## 5. `features_to_vector` 又起什么作用

`build_budget_features()` 输出的是一个字典，比如：

```python
{
    "num_sentences": 12,
    "sim_max": 0.91,
    "entropy_norm": 0.43,
    ...
}
```

但神经网络不能直接吃字典，所以还要做一步：

```python
features_to_vector(features)
```

把它变成固定顺序的向量：

\[
[12,\ 0.91,\ 0.43,\ ...]
\]

这个向量才是真正送进 MLP 的输入。

---

## 6. 一句话总结

在 `budget_features.py` 里，feature 的作用就是：

> **把“问题难度 + 上下文结构 + 相关性分布”编码成一组固定长度的数值特征，让预算预测模型能够判断当前样本应该保留多少比例的文本。**

---

## 7. 更直白的理解

它其实就是在替模型回答这几个问题：

- 这个问题难不难？
- 这个上下文长不长？
- 重要信息是不是集中在少数句子里？
- 高相关句子多不多？
- 这个样本适不适合激进压缩？

然后模型根据这些信号预测一个 `target_ratio`。

---

## 8. 后续优化方向

如果你后面想让预算预测器更强，主要就是改 feature。  
因为这个文件决定了模型“看见什么”。

最值得继续加的 feature 通常有：

- 问题和 top-k 句子的实体重叠率
- 句间冗余度
- 是否存在桥接句
- 段落级结构特征
- 更细的任务类型特征

也就是说，这个文件其实就是整个预算预测器的“感知层”。
