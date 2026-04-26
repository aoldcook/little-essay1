# 方法章节草稿：融合 Context-Aware Sentence Encoding 与自适应预算预测的语义保持型提示压缩

> 下面内容可直接作为论文第 3 章“方法”和第 4 章“实验设计”的初稿。文中方法名可暂定为 **CASC**（Context-Aware Semantic Compression），你也可以替换成自己的名称。

---

## 3 方法

### 3.1 任务定义

给定输入上下文
\[
C = \{s_i\}_{i=1}^{K}
\]
和问题
\[
q,
\]
其中 \(s_i\) 表示第 \(i\) 个句子，\(K\) 表示上下文中的句子数。我们的目标是在尽可能保留原始语义与下游任务性能的前提下，从原始上下文中抽取一个更短的压缩上下文
\[
\tilde C \subseteq C,
\]
使其长度显著小于原上下文，即
\[
|\tilde C| \ll |C|,
\]
同时保证目标大模型 \(\mathcal{M}\) 在压缩上下文上的输出质量尽可能接近完整上下文：
\[
\mathcal{Q}(\mathcal{M}(q, \tilde C)) \approx \mathcal{Q}(\mathcal{M}(q, C)).
\]

与 token 级压缩不同，本文采用**句子级压缩**作为主体粒度，以避免中间 token 删除带来的语法破坏与语义不连贯问题。进一步地，本文引入上下文感知句子编码器来判断“哪一句更重要”，并使用学习式预算预测器来决定“应该压缩到什么程度”。这一设计受到 CPC 工作的启发：CPC 通过 context-aware sentence encoder 对句子相关性进行排序，并在固定预算下选取高分句子，从而在 LongBench 和 ZeroSCROLLS 上取得了较强效果；同时，CPC 的消融结果表明，上下文感知编码、问题验证与负样本过滤对效果都有实质贡献。fileciteturn10file10 fileciteturn10file1

---

### 3.2 总体框架

本文方法由四个模块组成：

1. **训练样本构造模块**：从长文本中构造 \((C, q, p^+, n^- )\) 形式的训练样本；
2. **上下文感知句子编码器**：学习问题与句子在上下文中的相关表示；
3. **预算预测器**：预测当前样本的最小安全压缩率 \(\hat\tau\)；
4. **预算约束选择器**：在预算约束下综合相关性、覆盖度与冗余度完成句子选择。

#### 模块图描述（可直接转成论文图）

**图 1：方法总览图**

- 输入：长上下文 \(C\) 和问题 \(q\)
- 句子切分：得到候选句子集合 \(\{s_i\}_{i=1}^K\)
- Context-Aware Encoder：为每个句子生成上下文感知表示 \(\mathbf{z}_{i|C,q}\)
- Relevance Scorer：计算句子相关性分数 \(\alpha_i\)
- Budget Predictor：基于问题复杂度、上下文结构和分数分布预测压缩率 \(\hat\tau\)
- Coverage-aware Selector：在长度约束下选择句子子集 \(S^*\)
- 输出：压缩上下文 \(\tilde C\)
- 最终将 \((q, \tilde C)\) 输入目标 LLM 完成问答 / 摘要 / 信息提取

推荐将图画成两条路径：
- **训练路径**：样本构造 → 编码器训练 → 预算预测器训练
- **推理路径**：句子打分 → 预算预测 → 约束选择 → 压缩输出

---

### 3.3 训练样本构造

CPC 的核心启发之一，是其使用了专门的 Context-aware Question-Relevance（CQR）数据来训练句子编码器；其中正样本是“与问题相关、但通常不足以单独回答问题”的句子，负样本则是与问题无关的上下文句子，并通过问题验证、相似度过滤和 KL 过滤进行净化。其消融结果显示，去掉问题验证、相似度负样本过滤和 KL 负样本过滤都会造成性能下降。fileciteturn10file10 fileciteturn10file8

在此基础上，本文将训练样本扩展为三类正样本：

- **Direct positive**：直接包含答案线索的句子；
- **Bridge positive**：本句不足以独立回答，但对推理链、指代消解、因果关系有支撑作用；
- **Coverage positive**：对整体语义保持、摘要覆盖或主题保留有贡献的句子。

对每个训练样本，我们构造：
\[
(C, q, p^+, \{n_m^-\}_{m=1}^{M}),
\]
其中 \(p^+\) 可以来自上述三类正样本之一，\(n^-\) 为负样本句子集合。

#### 3.3.1 问题生成与验证

参照 CPC，可先从长文档中抽取候选句子 \(p\)，再利用大模型基于该句子及其上下文生成问题-答案对 \((q, a)\)。随后使用验证步骤确保该问题**不能只靠句子 \(p\) 单独回答**，而必须依赖上下文。CPC 就是通过这一步来避免模型退化为简单关键词匹配器。fileciteturn9file6

本文中，这一过程可形式化为：若对于句子 \(p\) 有
\[
\mathcal{V}(p, q, a)=0,
\]
则保留该样本，其中 \(\mathcal{V}\) 表示“句子本身是否足以独立回答问题”的验证器；返回 0 表示信息不足，说明该样本具有上下文依赖性。

#### 3.3.2 负样本过滤

参照 CPC，可先使用现成句向量模型计算问题与上下文句子的粗粒度相似度，筛出低于正样本相似度阈值的候选负样本，再通过删除句子前后答案分布变化的 KL 距离过滤掉“伪负样本”。CPC 使用了
\[
KL(P_C, P_{C\setminus s_j})
\]
来判断某个候选负样本是否实际上包含问题相关信息；若 KL 过高，则说明删除该句会显著改变答案分布，应将其从负样本中移除。fileciteturn7file11

本文可继续保留该思路，并额外加入一个**语义保持过滤器**：当删除候选句子导致 teacher summary 或 teacher answer 明显下降时，将其标注为 coverage positive，而不是 negative。

---

### 3.4 上下文感知句子编码器

#### 3.4.1 问题表示与句子表示

CPC 的一个关键设计，是不直接把句子独立编码，而是先对整个上下文进行编码，再在目标句子的 span 上做 pooling，从而得到“句子在上下文中的表示”。其表示形式为：
\[
\xi_{p,C}=\operatorname{norm}\left(\frac{1}{j-i+1}\sum_{t=i}^{j} \mathbf{z}_t\right),
\]
其中 \(i,j\) 是目标句子在上下文中的起止位置，\(\mathbf{z}_t\) 是编码器输出的 token 级表示。fileciteturn10file0

在本文中，我们使用共享编码器 \(f_\theta\) 构造两个表示：

1. **问题表示**：
\[
\mathbf{z}_q = \operatorname{norm}(\operatorname{MeanPool}(f_\theta(q))).
\]

2. **句子-上下文联合表示**：
先将目标句子 \(s_i\) 在上下文中用特殊标记包围，得到
\[
\bar C_i = \operatorname{Mark}(C, s_i),
\]
再编码
\[
\mathbf{H}_i = f_\theta([q; \bar C_i]),
\]
最后对被标记句子的 span 做平均池化：
\[
\mathbf{z}_{i|C,q}=\operatorname{norm}\left(\frac{1}{|I_i|}\sum_{t\in I_i}\mathbf{H}_{i,t}\right).
\]

其中 \(I_i\) 表示被标记句子的 token 索引集合。与 CPC 相比，本文在构造句子表示时显式引入问题 \(q\)，使句子表示不仅是 context-aware，也进一步是 query-aware。

#### 3.4.2 相关性分数

对于每个候选句子 \(s_i\)，其相关性分数定义为：
\[
\alpha_i = \cos(\mathbf{z}_q, \mathbf{z}_{i|C,q}).
\]

\(\alpha_i\) 越大，表示该句在给定问题下越重要。

---

### 3.5 编码器训练目标

CPC 使用了两类损失：
- 句子对比损失 \(L_{SC}\)，用于拉近问题与正样本句子、拉远问题与负样本句子；
- masked next token prediction 损失 \(L_{MNTP}\)，用于让模型学习整段上下文中的双向注意力。CPC 的消融显示，两者缺一都会明显掉点，说明上下文建模和对比学习都重要。fileciteturn10file0 fileciteturn10file4

本文保留这一设计，并将训练目标写为：
\[
\mathcal{L}_{enc}=\mathcal{L}_{ctr}+\lambda_{mlm}\mathcal{L}_{mlm}.
\]

#### 3.5.1 对比损失

对一个 batch 中的样本 \(b\)，设 \(\mathbf{z}_q^{(b)}\) 为问题表示，\(\mathbf{z}_+^{(b)}\) 为正样本句子表示，\(\mathcal{N}^{(b)}\) 为负样本集合（包括同样本负句与 batch 内其他样本的正句/负句），则使用 InfoNCE 形式：
\[
\mathcal{L}_{ctr}^{(b)}
=
-\log
\frac{\exp(\cos(\mathbf{z}_q^{(b)},\mathbf{z}_+^{(b)})/T)}
{\exp(\cos(\mathbf{z}_q^{(b)},\mathbf{z}_+^{(b)})/T)+
\sum_{\mathbf{z}\in\mathcal{N}^{(b)}}\exp(\cos(\mathbf{z}_q^{(b)},\mathbf{z})/T)}.
\]

整体损失为：
\[
\mathcal{L}_{ctr}=\frac{1}{B}\sum_{b=1}^{B}\mathcal{L}_{ctr}^{(b)}.
\]

#### 3.5.2 辅助上下文建模损失

参照 CPC，我们引入 masked language modeling / masked next token prediction 类型的辅助损失，让模型继续学习整段文本的上下文依赖：
\[
\mathcal{L}_{mlm}=
-\sum_{x_t\in \mathcal{M}(C)}\log p_\theta(x_t\mid C_{\setminus \mathcal{M}}).
\]

若使用 BERT / RoBERTa 等双向编码器实现，该项可以直接使用 MLM 头完成；若只想做最小可行版本，则可以先只训练 \(\mathcal{L}_{ctr}\)，再将 \(\mathcal{L}_{mlm}\) 作为第二阶段扩展。

---

### 3.6 学习式预算预测器

为避免使用固定压缩率或启发式 elbow，本文进一步引入预算预测器 \(g_\phi\)，根据当前样本的难度动态预测保留比例：
\[
\hat\tau = g_\phi(\mathbf{x}).
\]

其中 \(\mathbf{x}\) 由三类特征组成：

1. **相似度分布特征**：如 \(\max(\alpha)\)、top-k mass、熵、头部 gap、高相关句比例；
2. **问题复杂度特征**：问题长度、实体数、问题类型（定义/因果/比较/流程等）；
3. **上下文结构特征**：句子数、平均句长、上下文长度等。

训练标签 \(\tau^*\) 通过“最小安全预算”定义：
\[
\tau^* = \min_{\tau \in \mathcal{T}} \; \tau
\quad
\text{s.t.}
\quad
\mathcal{S}(\mathcal{M}(q, C_\tau), y_{ref}) \ge \gamma,
\]
其中 \(\mathcal{T}\) 为候选压缩率集合，\(C_\tau\) 表示压缩率为 \(\tau\) 的压缩上下文，\(y_{ref}\) 为 gold answer 或 full-context teacher answer，\(\mathcal{S}(\cdot)\) 为一致性评分函数，\(\gamma\) 为阈值。

这一模块与前述句子编码器形成互补：编码器解决“哪句重要”，预算预测器解决“压多少合适”。

---

### 3.7 预算约束下的覆盖感知句子选择

若只按 \(\alpha_i\) 取 top-k，容易出现两类问题：
- 选出多个高度重复的句子；
- 忽略某些桥接句或覆盖句。

因此，本文采用预算约束下的覆盖感知选择：
\[
S^* = \arg\max_{S \subseteq \{1,\dots,K\}}
\sum_{i\in S}\alpha_i
+\lambda \operatorname{Cov}(S,q)
-\mu \operatorname{Red}(S)
\]
\[
\text{s.t.}\quad \operatorname{Len}(S) \le \hat\tau \cdot \operatorname{Len}(C).
\]

其中：
- \(\operatorname{Cov}(S,q)\)：已选句子对问题实体/关键词/主题的覆盖度；
- \(\operatorname{Red}(S)\)：已选句子之间的冗余度；
- \(\operatorname{Len}(S)\)：长度预算，可按 token 或字符数近似。

实际实现中可采用 MMR 风格贪心：
\[
\operatorname{score}_{MMR}(s_i)
=
\lambda \alpha_i - (1-\lambda) \max_{s_j\in S_{sel}} \cos(\mathbf{z}_{i|C,q}, \mathbf{z}_{j|C,q}).
\]

在满足预算约束下，迭代选择分数最高的句子，最后再按原文顺序恢复，以保证压缩文本的可读性。

---

### 3.8 推理流程

推理时，给定 \((q, C)\)，完整流程如下：

1. 句子切分，得到 \(\{s_i\}_{i=1}^K\)；
2. 利用上下文感知句子编码器计算每个句子的 \(\alpha_i\)；
3. 由预算预测器得到 \(\hat\tau\)；
4. 在预算 \(\hat\tau\) 下使用覆盖感知选择器选句；
5. 按原文顺序恢复，得到压缩上下文 \(\tilde C\)；
6. 将 \((q, \tilde C)\) 输入目标 LLM 进行下游任务求解。

---

## 4 实验设计

### 4.1 数据集与任务设置

如果资源允许，建议沿用 CPC 的评测协议，覆盖：
- **LongBench**：SingleDoc、MultiDoc、FewShot、Code 等；
- **ZeroSCROLLS**：长文 QA 与摘要；
- **领域泛化集**：MeetingBank、PubMed、SummScreen、Krapivin 等。CPC 就是在 LongBench、ZeroSCROLLS 以及领域泛化任务上报告结果，并使用 Rouge、F1、Levenshtein 与关键词召回等指标。fileciteturn10file11 fileciteturn10file9

如果资源有限，建议至少保留三类任务：
- 单文档问答
- 多文档问答
- 长文本摘要

若你的论文以中文任务为主，也可以在中文长文 QA / 信息抽取数据上复现实验流程，但指标设计建议与上述保持一致。

---

### 4.2 对比试验设计

#### 4.2.1 外部基线

若可以复现现有方法，建议对比：
- Original Prompt
- BM25
- SBERT / BGE retrieval
- Selective-Context
- LLMLingua
- LLMLingua-2
- LongLLMLingua
- CPC

CPC 的原论文表明，在 LongBench 与 ZeroSCROLLS 上，其平均性能优于 LongLLMLingua，且平均/中位速度优势最高可达 27.5× / 10.93×；同时在更大 backbone 下，CPC 相比 LLMLingua-2 和 LongLLMLingua 也保持优势。fileciteturn10file13 fileciteturn8file5

#### 4.2.2 内部递进基线（最重要）

即使你无法完整复现全部外部基线，也必须做下面这组**内部对比**，因为这能直接证明你的贡献：

1. **Off-the-shelf embedding + fixed ratio**
2. **Off-the-shelf embedding + learned budget**
3. **Context-aware encoder + fixed ratio**
4. **Context-aware encoder + learned budget**
5. **Context-aware encoder + learned budget + coverage-aware selector**（完整方法）

这组对比能分别验证：
- 上下文感知编码器是否有效；
- 学习式预算预测是否有效；
- 覆盖感知选择是否进一步有效。

---

### 4.3 评价指标

建议报告四类指标：

1. **任务性能**
   - QA：F1 / EM
   - 摘要：ROUGE-1 / ROUGE-L / BERTScore
   - 信息抽取：Recall / F1

2. **压缩效率**
   - 平均保留 token
   - 压缩倍率
   - token 节省率

3. **推理效率**
   - 压缩器耗时
   - 目标 LLM 总耗时
   - 整体 wall-clock latency

4. **语义保持指标**
   - 与 full-context answer 的一致性
   - 与 gold answer 的相似度
   - teacher-student KL / embedding cosine（可选）

尤其建议画出 **rate–distortion curve**：
- 横轴：压缩率 / 保留 token 比例
- 纵轴：任务性能

这样可以直观看到在不同预算下各方法的性能保持情况。

---

### 4.4 消融实验设计

#### A. 编码器相关消融

1. 去掉 \(\mathcal{L}_{mlm}\)，只保留 \(\mathcal{L}_{ctr}\)
2. 去掉 \(\mathcal{L}_{ctr}\)，只保留 \(\mathcal{L}_{mlm}\)
3. 共享编码器 vs 双塔编码器
4. 句子独立编码 vs 上下文感知编码
5. 不同负样本数：2 / 4 / 8

CPC 的结果已经显示，\(L_{SC}+L_{MNTP}\) 最优；仅用 \(L_{MNTP}\) 或仅用 \(L_{SC}\) 都会降性能，而 2 个负样本优于 4 和 8。fileciteturn9file1

#### B. 数据构造消融

1. 不做问题验证
2. 不做相似度负样本过滤
3. 不做 KL 负样本过滤
4. 仅 direct positives
5. direct + bridge positives
6. direct + bridge + coverage positives

CPC 的消融显示，去掉问题验证、相似度过滤或 KL 过滤都会掉点，其中 KL 过滤影响最大。fileciteturn10file8

#### C. 预算预测消融

1. fixed ratio
2. elbow-based ratio
3. heuristic `find_target_ratio`
4. learned budget predictor

#### D. 选择器消融

1. top-k by relevance
2. relevance + original order
3. relevance + MMR
4. relevance + MMR + coverage term

#### E. 预算与顺序消融（可选）

1. 原文顺序恢复
2. 相关度顺序输出
3. 位置感知重排

---

### 4.5 建议的主结果表格

建议至少准备如下表格：

**表 1：主结果表**
- 方法
- QA / Summ / IE 指标
- 平均 token
- 压缩倍率
- 延迟

**表 2：内部模块递进表**
- off-the-shelf encoder
- + learned budget
- + context-aware encoder
- + coverage selector

**表 3：消融表**
- 损失函数
- 样本构造
- 负样本数
- 预算预测方式

**图 2：rate–distortion curve**

**图 3：不同上下文长度下的 latency 曲线**

---

## 5 实现建议

### 5.1 最小可行版本（推荐先做）

第一阶段先实现：
- context-aware encoder（仅 \(\mathcal{L}_{ctr}\)）
- learned budget predictor（你现有代码已具备）
- MMR 选择器

第二阶段再补：
- MLM / MNTP 辅助损失
- coverage positive
- KL 负样本过滤

这样可以更快得到第一组可发表结果。

### 5.2 与当前工程的对接方式

你当前已有：
- 句子切分
- embedding 打分
- `find_target_ratio` 学习化
- 伪标签生成脚本

因此最自然的融合方式是：

1. 用新训练的 context-aware encoder 替换原先 BGE/通用 embedding 打分器；
2. 将新的句子分数分布输入你现有的 budget predictor；
3. 将原先“按分数直接 top-k”升级为 MMR 选择器；
4. 评估完整方法在不同压缩率下的性能与耗时。

---

## 6 可直接放在论文贡献点里的表述

本文的主要贡献可以概括为：

1. 提出一种融合上下文感知句子编码与自适应预算预测的语义保持型提示压缩框架；
2. 在 CPC 的 CQR 思想基础上，扩展出 direct / bridge / coverage 三类正样本，以更好兼顾问答相关性与整体语义保持；
3. 设计预算约束下的覆盖感知句子选择机制，在保证压缩率的同时减少冗余并提升信息覆盖；
4. 通过系统的对比实验与消融实验，验证上下文感知编码、学习式预算预测与覆盖感知选择三者的独立贡献与协同效果。

