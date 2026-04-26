# 二级压缩规则文档（当前实现）

## 1. 定位

当前二级压缩的目标不是生成式改写，而是：

- 在一级句子筛选之后，对保留句做句内 `span / clause` 级压缩
- 尽量保住问题主干、答案线索、任务相关例子
- 尽量删掉时间背景、解释性尾巴、低价值补充、结构壳子

当前实现是 **hybrid learned span prune**：

- 不是纯规则
- 也不是纯二分类器
- 而是 `规则 + attention + DAC + learned span model` 的混合裁剪

核心文件：

- `pipeline/task_aware_compression.py`
- `pipeline/compression_pipeline.py`

---

## 2. 整体流程

### 2.1 输入

二级压缩接收：

- `question`
- 一级保留下来的 `selected_sentences`
- 每个句子的句级分数 `sentence_scores`

入口函数：

- `DynamicSpanCompressor.compress_sentences`

### 2.2 处理顺序

每个保留句的处理顺序是：

1. 识别问题类型
2. 把句子切成 span
3. 给每个 span 做类型判定
4. 判定哪些 span 直接保护，不可删
5. 给当前句分配 keep ratio
6. 对可删 span 计算删除优先级
7. 反复删除最低优先级 span，直到满足预算
8. 若句子只剩结构壳子，则整句丢弃
9. 清理标点和空白，拼回压缩句

---

## 3. 问题类型与任务信号

### 3.1 问题类型识别

函数：`detect_question_type`

当前支持：

- `cause`
- `comparison`
- `procedure`
- `numeric`
- `definition`
- `factoid`
- `other`

识别方式是关键词规则，不是 learned classifier。

### 3.2 任务提示词

当前有这些提示词集合：

- `CAUSE_HINTS`
  - 如：`导致 / 触发 / 引发 / 后果 / 风险 / 临界点`
- `COMPARISON_HINTS`
- `PROCEDURE_HINTS`
- `NUMERIC_HINTS`
- `FACTOID_HINTS`
- `EXAMPLE_HINTS`
  - 如：`例如 / 比如 / 举例 / for example / such as`
- `OUTCOME_HINTS`
  - 如：`触发 / 引发 / 导致 / 造成 / 后果 / 风险 / 临界点 / 退化 / 解冻`

### 3.3 问题是否在问“哪些后果/例子”

辅助函数：

- `starts_with_example_marker`
- `question_seeks_outcome_examples`

当前判定逻辑：

- 问题里同时出现“列举类提示”和“结果类提示”时，认为该问题在问后果/风险/表现/结果类内容
- 此时以 `如 / 例如 / 比如` 开头的 span 会被特殊保护

---

## 4. 句内切分规则

函数：`split_sentence_into_spans`

### 4.1 切分边界

当前按以下符号切：

- `，`
- `；`
- `：`
- `。`
- `！`
- `？`
- 英文对应标点

### 4.2 括号保护

若处于括号内部：

- 不在括号深度未归零时切分
- 尽量保持括号内容整体

### 4.3 结果

切出来的最小单元不是 token，而是较粗粒度的短分句 / 从句 / 逗号片段。

---

## 5. Span 类型判定

函数：`build_span`

当前 span 类型：

- `content`
- `parenthetical`
- `example`
- `tail`

判定规则：

- 整段被括号包住：`parenthetical`
- 以 `如 / 例如 / 比如` 等举例标记开头，或命中例子提示词：`example`
- 非句首且长度较短（当前阈值约 12 字）：`tail`
- 其他：`content`

---

## 6. 哪些 span 会被直接保护

函数：`build_span`

以下 span 默认进入 `protected=True`：

- 与问题有显式词重叠的 span
- `task_anchor_score >= 0.45` 的 span
- 问题在问“哪些后果”时，类型为 `example` 的 span
- 问题在问“哪些后果”时，包含 `OUTCOME_HINTS` 的 span
- 以 `：` 结尾且本身较长的结构前导 span
- 命中 `一是 / 二是 / 三是` 这类枚举结构的 span

以下情况会取消保护：

- 若 span 是 `parenthetical`，且既无 overlap 又无足够 anchor，则不保护

这部分是当前二级压缩里最关键的“不可删”规则。

---

## 7. 任务相关分数

### 7.1 Query overlap

函数：`query_overlap_score`

含义：

- 取问题和 span 的关键词集合
- 计算交集比例

目标：

- 防止删掉和问题表面高度相关的局部短语

### 7.2 Anchor score

函数：`task_anchor_score`

含义：

- 看 span 是否命中当前问题类型的锚点词
- 对“问后果”的问题，再额外奖励结果主干词和举例前缀

当前行为：

- `cause` 问题中，`触发 / 引发 / 风险 / 临界点 / 后果` 这类主干 span 会显著升分
- “如亚马逊雨林退化...” 这种例子 span 也会额外升分

### 7.3 Task reward

函数：`compute_task_reward`

计算方式：

- `0.55 * overlap + 0.35 * anchor + bonus`

其中：

- `complex` 问题（`cause/comparison/procedure`）会额外加一个小 bonus

---

## 8. 每句 keep ratio 分配

函数：`allocate_sentence_keep_ratios`

当前默认配置来自 `IntraSentenceCompressionConfig`：

- `target_keep_ratio = 0.72`
- `min_keep_ratio = 0.50`
- `max_keep_ratio = 0.88`

句级分配公式：

- `ratio = target_keep_ratio + 0.12 * normalized_sentence_score + complexity_bonus`

其中：

- `complexity_bonus = 0.04`，仅对 `cause/comparison/procedure` 生效

最终 ratio 会截断到 `[0.50, 0.88]`。

### 8.1 需要注意

`main.py` 当前 demo 并没有用默认 `0.72`，而是显式传了：

- `second_stage_keep_ratio = 0.8`

这意味着：

- demo 比默认配置更保守
- 更偏向“少删、保语义”

---

## 9. 删除优先级计算

函数：`rank_removal_candidates`

### 9.1 输入特征

当前会综合：

- `attention_scores`
- `dac_scores`
- `overlap_scores`
- `anchor_scores`
- `reward_scores`
- `learned_keep_scores`（若加载了 span model）

### 9.2 启发式分数

启发式 importance 由下式组成：

- `0.25 * attention`
- `0.20 * dac`（仅 DAC 可用时）
- `0.34 * anchor`
- `0.34 * overlap`
- `0.22 * reward`

### 9.3 learned span model 融合

若已加载 `span_model.best.pt`，则再做混合：

- `importance = 0.45 * heuristic + 0.55 * learned_keep`

也就是说当前 learned span model 权重更高。

### 9.4 惩罚项

当前会额外减去以下惩罚：

- `mismatch_penalty = 0.12`
  - 当 span 与问题无 overlap 且 anchor 不足时
- 时间背景惩罚 `-0.10`
- `parenthetical` 惩罚 `-0.18`
- `example` 惩罚 `-0.12`
- `tail` 惩罚 `-0.05`
- `filler_penalty = 0.16`
  - 如：`总体来说 / 换句话说 / 不过在 / 在读写方面 / 此外 / 另一方面`

### 9.5 删除策略

- importance 越低，越优先删
- 每轮删一个最不重要 span
- 直到句子 token 数量不超过预算

---

## 10. 哪些 span 即使不 protected 也不轻易删

在 `rank_removal_candidates` 里有一个短句边界保护：

- 当句子切出来的 span 很少（`<= 3`）时
- 默认不删首尾 span

但有例外：

- 如果首尾 span 本身就是低相关尾巴
- 或者是明显时间背景 span
- 则允许进入删除候选

这条规则就是之前能把 `1969年之前使用传统汉字。` 删掉的原因之一。

---

## 11. 整句丢弃规则

函数：`should_drop_sentence_after_prune`

如果一个句子压缩后只剩一个 span，且这个 span：

- 是结构性前导
  - 如：`可分为三个时期：`
- 和问题没有 overlap
- anchor 不高

则整句直接丢弃。

这是为了避免留下：

- `可分为三类：`
- `主要包括：`
- `如下：`

这类没有语义主体的空壳句。

---

## 12. 清理规则

函数：`cleanup_sentence`

清理内容包括：

- 合并多余空格
- 去掉标点前空格
- 修复括号两侧空格
- 去掉重复逗号/冒号
- 若原句句末有终止标点，压缩结果没有时会补回终止标点

因此当前输出是抽取式压缩，不会主动改写成更自然的新句，只会在原片段基础上做拼接清理。

---

## 13. 当前实际配置总结

### 13.1 默认配置（代码默认）

- `target_keep_ratio = 0.72`
- `min_keep_ratio = 0.50`
- `max_keep_ratio = 0.88`
- `attention_weight = 0.25`
- `anchor_weight = 0.34`
- `overlap_weight = 0.34`
- `reward_weight = 0.22`
- `learned_keep_weight = 0.55`
- `mismatch_penalty = 0.12`
- `filler_penalty = 0.16`

### 13.2 当前 demo（main.py）

- `second_stage_keep_ratio = 0.8`
- 使用已训练的：`outputs_high_recall_v2/span_model.best.pt`

---

## 14. 当前已知限制

### 14.1 二级仍是“抽取式压缩”，不是改写

因此它能做到：

- 不把主干删掉
- 不把例子删坏
- 不保留明显噪音

但它不会主动把：

- `根据...报告，若...，将...，如...`

润色成更自然的生成式版本，例如：

- `IPCC第六次评估报告指出，若...，可能触发...`

### 14.2 时间背景规则当前有实现残留

`is_temporal_background_span` 目前的有效行为主要依赖：

- 检测是否含有两位以上数字

它原本还想依赖中文时间词列表，但当前实现里这部分有编码残留，后续最好单独清理。

### 14.3 字符级 protected mask 仍在文件里，但主路径未实际使用

`build_protected_char_mask` 当前没有进入主压缩决策主链路，主要逻辑仍然是 span 级规则。

---

## 15. 一句话总结

当前二级压缩的真实策略是：

- **先按标点切成短分句**
- **用问题类型和例子/后果规则判定哪些片段不能删**
- **再用 heuristic + DAC + learned span model 给剩余片段排序删除**
- **如果句子最后只剩结构壳子，就整句丢掉**

所以它本质上是：

- 任务对齐的抽取式句内压缩
- 不是纯关键词匹配
- 也不是生成式重写
