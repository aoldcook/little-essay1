# LittleEssay1 项目介绍书

## 1. 这项目到底在做什么

一句话版：

这个项目是在做一个“面向问答场景的两级上下文压缩系统”。

- 第一级先做句子级压缩：判断长上下文里哪些句子更值得保留。
- 中间再做预算预测：自动估计这次压缩应该保留多少内容。
- 第二级再做句内压缩：对保留下来的句子继续删掉低价值短语或插入语。
- 最终输出仍然是自然语言文本，所以它可以直接喂给黑盒 LLM，不需要改下游模型结构。

如果你把整个仓库当成一条流水线，它大致是这样：

1. `data_builder/` 先准备训练数据。
2. `context_aware_encoder_model/` 训练“句子级相关性编码器”。
3. `target_ratio_model/` 训练“压缩率预测器”。
4. `intra_sentence_model/` 训练“句内 span 保留/删除分类器”。
5. `pipeline/` 把这些模块串起来，推理时真正执行压缩。
6. `main.py` 是一个最小可运行示例。

## 2. 仓库地图

你现在这个仓库里，最值得优先理解的是下面这些目录：

```text
LittleEssay1/
├─ 1 Prompt Compression with Context-Aware Sentence Encoding for Fast and/
│  ├─ main.py
│  ├─ target_ratio.py
│  ├─ context_aware_encoder_model/
│  ├─ pipeline/
│  ├─ target_ratio_model/
│  ├─ intra_sentence_model/
│  ├─ data_builder/
│  └─ markdown/
├─ project_technical_report.md
└─ project_source_guide.md
```

可以这样理解各目录职责：

- `context_aware_encoder_model/`
  负责句子级编码和句子级训练，是整个系统的一级压缩基础。
- `pipeline/`
  负责推理主流程，把“句子打分 + 预算预测 + 句子选择 + 句内压缩”串起来。
- `target_ratio_model/`
  负责预测压缩率，也就是“这次保留多少内容”。
- `intra_sentence_model/`
  负责句内压缩特征、伪标签和 span 分类模型。
- `data_builder/`
  负责把 DuReader / C3 / CMRC 等原始问答数据转成你训练所需的 CQR 风格样本。
- `markdown/`
  是项目中已经写过的一些说明草稿，偏补充材料，不是主逻辑。

## 3. 建议阅读顺序

如果你现在“看不懂自己的项目”，我建议按这个顺序回看：

1. `main.py`
   先看系统怎么被真正调用。
2. `pipeline/compression_pipeline.py`
   看一级压缩、预算预测、二级压缩怎么串起来。
3. `context_aware_encoder_model/context_aware_sentence_encoder.py`
   看句子级相关性到底怎么算。
4. `pipeline/task_aware_compression.py`
   看句内 span 是怎么拆、怎么删的。
5. `target_ratio_model/budget_features.py` 和 `budget_model.py`
   看压缩率为什么能自动预测。
6. `intra_sentence_model/`
   看句内模型训练数据和训练逻辑。
7. `data_builder/convert_dureader_to_cqr.py`
   看训练数据最开始是怎么造出来的。

## 4. 系统主流程

### 4.1 推理流程

推理时主要走的是 `main.py -> pipeline/compression_pipeline.py`：

1. 输入 `question` 和长 `context`。
2. `split_sentences()` 先切句。
3. 对每个句子构造一个带 marker 的局部上下文窗口。
4. 句子编码器输出：
   - `semantic_similarities`
   - 可选 `attention_probe_scores`
5. 再算 `task_rewards`。
6. 三种分数融合成 `selection_scores`。
7. 若用户没手动给 `target_ratio`，预算模型自动预测。
8. 用 MMR 在预算内选句。
9. 对保留下来的句子继续做句内 span 压缩。
10. 拼回最终 `compressed_context`。

### 4.2 训练流程

训练链路一般是：

1. 原始 QA 数据先被 `data_builder/` 转成 CQR 风格样本。
2. 清洗、分割 train/dev/test。
3. 用这些数据训练句子级编码器。
4. 扫不同压缩率生成预算伪标签。
5. 训练预算预测器。
6. 从训练样本里生成句内 span 伪标签。
7. 训练句内 span 分类器。
8. 推理时把这几块都加载进 `pipeline/`。

## 5. 顶层文件说明

### `main.py`

定位：最小演示入口。

它做的事很直接：

- 自动找到项目根目录。
- 找句内模型目录，优先 `outputs`，没有就退回 `outputs_demo`。
- 实例化 `ContextAwareCompressor`。
- 构造一个演示问题和一段长上下文。
- 调 `compress()`。
- 把一级和二级压缩的关键统计都打印出来。

它没有定义函数，整个文件就是一个 demo 脚本。

### `target_ratio.py`

定位：压缩率规则基线。

函数：

- `find_target_ratio(similarities, min_ratio=0.2, max_ratio=0.8)`
  根据句子相似度分布的集中程度，启发式地估计应该保留多少内容。

这个文件比 `target_ratio_model/` 早，属于规则版 baseline。

### `1.原理复现(LLM版).py`

定位：最早期的基线脚本，用外部 LLM 给句子打分。

函数：

- `compress_with_llm(context, question, model_name="qwen3-max")`
  逐句调用 Tongyi/Qwen 打分，再按分数排序和预算截断。

注意：

- 它是实验脚本，不是主工程入口。
- 文件里目前写死了 Tongyi API key，这种写法不适合长期保留。

### `2.原理复现（向量嵌入.py`

定位：向量检索版 baseline。

类：

- `EmbeddingBasedCompressor`
  一个更像“传统检索压缩器”的基线实现。

方法：

- `__init__(model_name, device)`：加载 Hugging Face 编码器。
- `mean_pooling(model_output, attention_mask)`：对 token 表示做 masked mean pooling。
- `encode(texts, max_length=512)`：编码问题和句子，返回句向量。
- `split_sentences(text)`：中英混合切句。
- `compress(context, question, target_ratio=None)`：先编码问题和句子，再按相似度选句，最后按原顺序拼回文本。

## 6. `context_aware_encoder_model/` 详细说明

这是一级压缩的核心目录。

### 6.1 `context_aware_sentence_encoder.py`

定位：上下文感知句子编码器核心实现。

核心类：

- `ContextAwareEncoderConfig`
  用 `@dataclass` 管理模型配置，包括底座模型名、最大长度、温度系数、设备、句子边界 marker 等。
- `ContextAwareSentenceEncoder`
  真正的句子级编码模型。它不是把句子孤立编码，而是把目标句子重新放回上下文里，再抽取 marker 之间的表示。

`ContextAwareSentenceEncoder` 的方法：

- `__init__(config)`：加载 tokenizer 和 Transformer 编码器，注册 `<sent_start>` / `<sent_end>` 两个特殊 token，并记录它们的 token id。
- `mean_pool(hidden_states, attention_mask)`：把 token 级 hidden states 做掩码平均，得到句向量。
- `encode_question(questions)`：单独编码问题，得到归一化 query embedding。
- `_find_marker_span(input_ids)`：在 token 序列里找到 `<sent_start>` 和 `<sent_end>` 之间的真实句子范围。
- `encode_marked_contexts(questions, marked_contexts)`：编码 `(question, marked_context)` 对，并从 marker 包住的 span 上抽取句子的上下文感知表示。
- `score_sentences(question, sentences, marked_contexts)`：计算每个句子和问题的相似度分数，同时返回句向量。
- `attention_probe_scores(question, marked_contexts, probe_layers=2)`：从编码器后几层注意力里抽出“问题对目标句的关注程度”，作为辅助分数。
- `contrastive_loss(questions, positive_marked_contexts, negative_marked_contexts)`：用 InfoNCE 风格对比学习让 query 靠近正样本句、远离负样本句。

辅助函数：

- `split_sentences(text)`：按中英文句号、问号、分号切句。
- `normalize_sentence_for_match(text)`：规范化句子，便于在原文里匹配目标句。
- `find_target_sentence_index(sentences, target_sentence)`：在切句结果里定位某个目标句对应的索引。
- `build_marked_context(sentences, target_index, marker_start, marker_end)`：用整段上下文构造带 marker 的版本。
- `build_marked_context_window(sentences, target_index, marker_start, marker_end, max_chars=220)`：只取目标句附近一段局部上下文，避免长文本截断 marker。
- `build_marked_context_from_span(context, start, end, marker_start, marker_end)`：已知字符跨度时，直接在原文中插入 marker。
- `build_marked_context_from_span_window(context, start, end, marker_start, marker_end, max_chars=220)`：先截局部窗口，再插 marker。
- `build_marked_context_from_text(context, target_sentence, marker_start, marker_end, use_window=False, max_chars=220)`：给你原文和目标句文本，帮你构造带 marker 的上下文。

关键理解点：

- 这个文件解决的是“如何给句子打分”，不是最终压缩。
- 这里的句子表示是 `sentence-in-context`，不是孤立句向量。
- `attention_probe_scores()` 不是直接用下游大模型注意力，而是用当前编码器自己的注意力做 probe。

### 6.2 `cqr_train_eval.py`

定位：训练数据读取和开发集评估。

类：

- `CQRDataset`
  读取 CQR JSONL，自动把正负句转换成带 marker 的上下文。

方法：

- `__init__(path, marker_start, marker_end, use_window=True, max_window_chars=220)`：加载 JSONL 并构造训练样本。
- `__len__()`：返回样本数。
- `__getitem__(idx)`：返回某个样本。

函数：

- `collate_fn(batch)`：把一批样本整理成训练循环需要的字段字典。
- `evaluate_sentence_ranking(model, dataset)`：用正句和负句排名来评估模型，输出 top1、top2、MRR、margin 等指标。

### 6.3 `train_context_aware_encoder.py`

定位：标准对比学习训练脚本。

函数：

- `save_encoder_bundle(model, config, output_dir)`：保存编码器权重、tokenizer 和 `encoder_config.json`。
- `better_metrics(current, best)`：比较两次开发集指标，决定是否更新 best checkpoint。
- `main()`：解析参数、设随机种子、构建数据集、训练模型、保存 `best` 和 `last` 模型。

### 6.4 `train_context_aware_encoder_with_mntp.py`

定位：多任务增强训练脚本。

类：

- `MultiTaskEncoderConfig`：在标准配置上新增 `mlm_probability` 和 `lambda_mntp`。
- `MultiTaskContextAwareSentenceEncoder`：基于 `AutoModelForMaskedLM` 的多任务模型，既算句子对比损失，也算掩码语言模型损失。

方法：

- `__init__(config)`：加载 MLM 模型，并从中拿 backbone。
- `mean_pool(hidden_states, attention_mask)`：和标准版同理。
- `encode_question(questions)`：编码问题。
- `_find_marker_span(input_ids)`：找 marker span。
- `encode_marked_contexts(questions, marked_contexts)`：编码带 marker 上下文。
- `contrastive_loss(questions, positive_marked_contexts, negative_marked_contexts)`：句子级对比学习损失。
- `mntp_loss(contexts)`：近似 MLM/MNTP 的辅助损失，随机 mask token 再预测。

函数：

- `save_encoder_bundle(model, config, output_dir)`：保存多任务模型和配置。
- `better_metrics(current, best)`：比较开发集指标。
- `main()`：训练总入口，把对比损失和 MNTP 损失加权求和。

### 6.5 `train_context_aware_encoder_curriculum.py`

定位：课程式两阶段训练调度器。

函数：

- `resolve_optional_path(raw_value)`：把命令行字符串转成可选路径。
- `resolve_input_file(explicit_path, split_dir, default_name, label)`：优先用显式路径，否则去 split 目录里找默认文件名。
- `load_json_if_exists(path)`：读可选 JSON 文件，不存在就返回空字典。
- `build_stage_command(...)`：组装第一阶段或第二阶段训练命令。
- `run_stage(stage_name, command, output_dir, project_root)`：真正调用子进程执行训练，并生成阶段结果摘要。
- `main()`：生成 `curriculum_plan.json`，然后依次跑 `stage1_gold` 和 `stage2_full`。

### 6.6 `plot_curriculum_training_history.py`

定位：把课程训练的 loss 画出来。

函数：

- `load_history(stage_dir)`：读取阶段目录里的 `training_history.json`。
- `collect_loss_series(history)`：从训练记录里提取可画图的 loss 曲线。
- `main()`：读取多阶段历史并保存成 `curriculum_loss.png`。

### 6.7 `__init__.py`

定位：包标记文件，没有实际业务逻辑。

## 7. `pipeline/` 详细说明

这是推理主干目录。

### 7.1 `compression_pipeline.py`

定位：整个压缩系统的总入口。

关键函数：

- `select_with_mmr(similarities, sentence_embeddings, sentences, target_ratio, lambda_relevance=0.7)`：在预算约束下做 MMR 选句，同时考虑相关性和冗余度。
- `blend_sentence_scores(semantic_similarities, attention_probe_scores, task_rewards, attention_probe_weight, task_reward_weight)`：把语义分数、attention probe 分数和任务奖励融合成最终句子选择分数。

关键类：

- `BudgetPredictorAdapter`
  用来适配 `target_ratio_model/`。

方法：

- `__init__(budget_model_dir)`：加载预算模型和元数据。
- `predict_ratio(question, context, similarities, fallback_ratio=0.4)`：用预算模型预测保留比例；没有模型时回退到固定比例。

- `ContextAwareCompressor`
  项目里最重要的对外接口。

方法：

- `__init__(encoder_dir, budget_model_dir=None, device=None, window_max_chars=220, use_attention_probe=True, attention_probe_weight=0.25, task_reward_weight=0.15, attention_probe_layers=2, enable_second_stage=True, second_stage_keep_ratio=0.78, span_model_dir=None)`：加载编码器、预算预测器和句内压缩器。
- `score_context(question, context)`：对每个句子算一级分数，并返回句向量。
- `compress(question, context, target_ratio=None, lambda_relevance=0.7, fallback_ratio=0.4)`：执行完整压缩流程，返回包括中间统计在内的结果字典。

### 7.2 `task_aware_compression.py`

定位：二级压缩核心，也就是句内 span 级删减逻辑。

顶层辅助函数：

- `normalize_scores(values)`：归一化一组分数。
- `detect_question_type(question)`：用关键词判断问题类型。
- `tokenize_query_terms(text)`：从问题或 span 里抽关键词。
- `query_overlap_score(question, text)`：计算问题词与文本的重叠程度。
- `task_anchor_score(question, text, question_type=None)`：判断文本中是否出现因果、比较、流程、数字等任务锚点。
- `compute_task_reward(question, text, question_type=None)`：融合 query overlap 和 anchor score，形成一个任务奖励分数。

数据类：

- `IntraSentenceCompressionConfig`：管理句内压缩配置，例如 keep ratio、权重、最小句长等。
- `SpanUnit`：表示一句话里切出来的一个 span，记录文本、起止位置、类别、是否受保护。

核心类：`DynamicSpanCompressor`

方法：

- `__init__(sentence_encoder, config=None, span_model_dir=None)`：绑定句子编码器、DAC 适配器，以及可选的学习式 span 模型。
- `_load_trained_span_model(span_model_dir)`：尝试加载训练好的 span 分类器。
- `predict_learned_keep_scores(question, spans, question_type, sentence_score, keep_ratio, attention_scores, dac_scores)`：对每个 span 用学习式模型预测保留概率。
- `compress_sentences(question, sentences, sentence_scores)`：批量压缩多句，并汇总统计。
- `allocate_sentence_keep_ratios(question, sentence_scores)`：根据句子分数给不同句子分配不同 keep ratio。
- `compress_sentence(question, sentence, keep_ratio, sentence_score=0.5)`：压缩单句，是二级压缩的主入口。
- `rank_removal_candidates(question, spans, question_type, sentence_score, keep_ratio)`：给每个可删 span 排序，决定先删谁。
- `split_sentence_into_spans(question, sentence, question_type)`：按逗号、分号、括号等边界把一句话切成多个 span。
- `build_protected_char_mask(question, sentence, spans, question_type)`：构造字符级保护掩码，避免把问题关键词、数字、重要锚点误删。
- `compute_dac_span_scores(question, spans)`：调 `DacTokenAdapter` 给 span 计算 DAC 风格显著性分数。
- `build_span(question, text, start, end, question_type)`：把一段文本包装成 `SpanUnit`，同时判断它属于 content、example、tail、parenthetical 哪一类。
- `compute_span_attention_scores(question, spans)`：从编码器 attention 中估计 span 重要性。
- `is_low_value_filler(text)`：识别“总的来说、换句话说、例如”之类低价值补充语。
- `cleanup_sentence(text, original)`：把删减后的句子重新清理标点和空格。
- `_count_tokens(text)`：估算 token 数。
- `_count_span_tokens(spans)`：统计若干 span 的 token 数。
- `_build_sentence_stats(original, compressed, kept_spans, removed_spans)`：打包句内压缩统计结果。

### 7.3 `dac_adapter.py`

定位：把 DAC 风格的 token 显著性思路接到你当前系统里。

数据类：

- `DacCompressionConfig`：配置 token 级压缩过程，例如融合方式、是否保留标点、是否避免连续删除等。

核心类：`DacTokenAdapter`

方法：

- `__init__(sentence_encoder, config=None)`：尝试加载一个 MLM 模型，用来估计 token loss / entropy。
- `normalize(tensor)`：归一化张量到 0 到 1。
- `_fuse_additive(losses, attention)`：用加法方式融合 token loss 和 attention。
- `_fuse_multiplicative(losses, attention)`：用乘法方式融合 token loss 和 attention。
- `_token_count(text)`：统计 token 数。
- `compute_token_losses(text)`：遮住每个 token，观察预测损失，估计 token 的“删掉会不会痛”。
- `compute_token_attention(question, text)`：估计问题对文本中各 token 的注意力强度。
- `compute_fused_token_scores(question, text)`：融合 token loss 和 attention，得到 token 重要性。
- `score_spans(question, text, spans)`：把 token 分数聚合成 span 分数。
- `preserve_punctuation_mask(text, offsets)`：标记纯标点 token，避免把句子结构全部打坏。
- `protected_token_mask(offsets, protected_char_mask)`：把字符级保护掩码映射到 token 级。
- `select_keep_indices(score, compress_ratio, punct_mask, protect_mask)`：决定哪些 token 保留。
- `reconstruct_text(text, offsets, keep_indices)`：根据保留 token 反构文本。
- `compress(question, text, keep_ratio, protected_char_mask=None)`：迭代执行若干轮 token 压缩。

### 7.4 `__init__.py`

定位：包标记文件，无业务逻辑。

## 8. `target_ratio_model/` 详细说明

这是预算预测目录。

### 8.1 `budget_features.py`

定位：预算预测特征工程。

函数：

- `split_sentences(text)`：切句。
- `detect_question_type(question)`：把问题归到 definition、cause、comparison、procedure、numeric、factoid、other。
- `count_entities(question)`：粗略统计问题里的实体数。
- `softmax(x)`：把相似度分布归一化成概率。
- `_safe_ratio(num, den)`：安全除法，避免除零。
- `build_budget_features(question, context, similarities)`：提取预算模型用到的全部特征，比如相似度最大值、熵、头部质量、句长统计、问题类型、实体数等。
- `features_to_vector(features)`：按固定顺序把特征字典转成向量。

### 8.2 `budget_model.py`

定位：预算预测模型定义。

类：

- `BudgetConfig`：保存 ratio bucket、输入维度、隐藏层结构。
- `BudgetPredictorMLP`：一个多层感知机，输出每个压缩率 bucket 的 logits。
- `BudgetLoss`：在普通交叉熵基础上，对“预测过小压缩率”的情况额外加惩罚。

函数：

- `ratio_to_class(ratio, ratio_buckets)`：把真实 ratio 映射到最近的离散类别。
- `class_to_ratio(cls_idx, ratio_buckets)`：把类别 id 反解成 ratio。
- `build_metadata(config, feature_order)`：保存模型元信息。

### 8.3 `train_budget_predictor.py`

定位：预算模型训练脚本。

函数：

- `load_jsonl(path)`：读训练数据。
- `build_dataset(samples, ratio_buckets)`：从样本里提特征，构造 `X, y`。
- `main()`：训练 MLP 并保存 `budget_predictor.pt` 与 `metadata.json`。

### 8.4 `predict_budget.py`

定位：预算模型推理脚本。

函数：

- `load_json(path)`：读单条输入样本。
- `main()`：加载模型，输出预测压缩率和各 bucket 概率。

### 8.5 `generate_pseudo_labels.py`

定位：为预算模型生成伪标签。

辅助函数：

- `load_jsonl(path)`：读 JSONL。
- `save_jsonl(path, rows)`：写 JSONL。
- `parse_ratios(s)`：解析命令行传入的 ratio 列表。
- `compress_with_scores(context, sentences, similarities, target_ratio)`：按已知句子分数和目标压缩率生成压缩文本。
- `normalize_text(text)`：规范化文本。
- `tokenize_mixed(text)`：中英混合分词。
- `char_f1(pred, ref)`：字符级 F1。
- `jaccard_sim(a, b)`：Jaccard 相似度。
- `question_overlap_score(question, sentence)`：句子和问题词重叠。
- `extractive_demo_answer(question, context)`：一个简单的抽取式“模拟回答器”。

LLM 适配类：

- `LLMClient`
  负责兼容 OpenAI 风格和 Ark Responses 风格接口。

方法：

- `__init__(...)`：保存 API 配置。
- `_resolve_api_style(api_style, base_url)`：自动判断接口风格。
- `_build_request(system_prompt, user_prompt)`：组装请求体。
- `_extract_content(data)`：从响应里抽文本。
- `chat(system_prompt, user_prompt)`：真正发请求，带重试。

主逻辑函数：

- `build_answer_prompt(question, context)`：组装回答 prompt。
- `judge_consistency(pred_answer, teacher_answer, mode)`：选择字符 F1 或 Jaccard 作为回答一致性指标。
- `choose_label_ratio(question, context, similarities, ratios, threshold, answer_mode, judge_mode, client=None, gold_answer=None, full_answer=None, sleep_seconds=0.0)`：从多个 ratio 里找到“满足质量阈值的最小安全保留比例”。
- `main()`：整体伪标注流程入口。

### 8.6 `example_integration.py`

定位：预算模型接入示例。

类：

- `LearnedBudgetSelector`
  一个更轻量的预算预测封装。

方法：

- `__init__(model_dir)`：加载预算模型。
- `predict_ratio(question, context, similarities)`：输出预算预测结果。

函数：

- `compress_with_scores(context, sentences, similarities, target_ratio)`：简单按分数和预算做压缩。
- `demo(sentence_ranker, model_dir, context, question)`：演示怎么把预算模型接到已有句子排序器上。

### 8.7 `__init__.py`

定位：包标记文件，无业务逻辑。

## 9. `intra_sentence_model/` 详细说明

这是二级压缩训练相关目录。

### 9.1 `span_feature_utils.py`

定位：句内 span 特征工程。

函数：

- `normalize_text(text)`：文本规范化。
- `tokenize_query_terms(text)`：提取问题关键词。
- `tokenize_mixed(text)`：中英混合 token 化。
- `normalize_scores(values)`：归一化分数。
- `detect_question_type(question)`：判断问题类型。
- `query_overlap_score(question, text)`：问题和 span 的关键词重叠。
- `task_anchor_score(question, text, question_type=None)`：检测 span 是否含有因果、流程、数字等锚点。
- `compute_task_reward(question, text, question_type=None)`：生成任务奖励。
- `overlap_ratio(a, b)`：文本重叠比。
- `answer_overlap_score(answer, text)`：span 与答案的重叠度。
- `question_type_features(question_type)`：把问题类型转成 one-hot 特征。
- `kind_features(kind)`：把 span 类型转成 one-hot 特征。
- `approx_token_length(text)`：粗略统计 token 长度。
- `build_span_feature_dict(...)`：为单个 span 构建完整特征字典。
- `features_to_vector(features)`：按固定顺序转向量。

### 9.2 `span_model.py`

定位：句内 span 分类器定义。

类：

- `SpanClassifierConfig`：保存输入维度、隐藏层和 dropout。
- `SpanClassifierMLP`：输出单个 span 的保留 logit。

方法：

- `__init__(config)`：组网。
- `forward(x)`：前向传播。

函数：

- `build_metadata(config, feature_order, threshold)`：保存模型元数据。
- `load_span_model(model_dir, device)`：从磁盘加载 span 模型。

### 9.3 `span_dataset.py`

定位：span 训练数据读取。

类：

- `SpanInstanceDataset`
  把 JSONL 中的 span 样本展开成单条 `(x, y)` 实例。

方法：

- `__init__(rows)`：读取 span 特征和标签。
- `__len__()`：返回样本数。
- `__getitem__(idx)`：返回某个 span 实例。

函数：

- `load_jsonl(path)`：读 JSONL。
- `build_xy(rows)`：把整批样本转成 `X, y`。
- `build_feature_metadata()`：返回特征顺序。

### 9.4 `generate_span_pseudo_labels.py`

定位：给 span 模型造伪标签。

函数：

- `load_jsonl(path)`：读输入样本。
- `save_jsonl(path, rows)`：写输出样本。
- `split_sentences(text)`：切句。
- `char_f1(pred, ref)`：字符 F1。
- `extractive_demo_answer(question, context)`：抽取式简易回答器。
- `compute_answer_quality(question, context, answer)`：用 demo 回答器衡量答案质量。
- `load_encoder_from_dir(encoder_dir, device)`：从目录中恢复上下文感知编码器。
- `pick_training_sentences(row)`：从样本中挑适合拿来做 span 训练的句子。
- `build_pseudo_label(span, feature_dict, answer_drop)`：根据锚点分数、query overlap、answer drop 等启发式生成 span 标签。
- `main()`：整体伪标签生成入口。

### 9.5 `train_span_model.py`

定位：训练 span 分类器。

函数：

- `split_train_dev(X, y, dev_ratio, seed=42)`：切训练集和开发集。
- `evaluate(model, loader, device)`：评估 dev loss 和准确率。
- `main()`：训练模型并保存 `span_model.pt` 与 `metadata.json`。

### 9.6 `__init__.py`

定位：包标记文件，无业务逻辑。

## 10. `data_builder/` 详细说明

这是整个仓库里最杂也最重要的目录，因为训练数据是项目能不能跑起来的地基。

### 10.1 `build_cqr_with_filters.py`

定位：从候选 QA 样本构造更干净的 CQR 风格训练集。

函数：

- `split_sentences(text)`：切句。
- `save_jsonl(path, rows)`：写 JSONL。
- `load_jsonl(path)`：读 JSONL。
- `build_verification_prompt(text, question, answer)`：构造“单句是否足以回答问题”的验证 prompt。
- `heuristic_verification(text, question, answer)`：没有 LLM 时的离线兜底规则。
- `sim_filter_negatives(context, question, positive_sentence, embedder, beta=0.30)`：先用相似度筛一轮负句。
- `kl_filter_negatives(context, question, answer, candidate_negatives, kl_estimator, kl_threshold=4e-3)`：再用 KL 继续过滤负句。
- `build_records(rows, verification_mode, verifier_client, embedder, beta, negatives_per_positive, use_kl_filter, kl_estimator, kl_threshold)`：对一批原始样本执行验证、筛负句、生成 CQR 记录。
- `main()`：命令行入口。

类：

- `LLMClient`
  兼容 OpenAI / Ark 风格验证接口。
- `HFMeanEmbedder`
  用 HF 编码器做均值池化，给句子算向量。
- `AnswerKLEstimator`
  比较“删掉候选负句前后”答案分布的 KL 差异。

### 10.2 `clean_cqr_dataset.py`

定位：清洗和拆分 CQR 数据集。

函数：

- `split_sentences(text)`、`normalize_text(text)`、`tokenize_mixed(text)`、`overlap_ratio(a, b)`、`stable_hash(text)`、`find_exact_sentence(context_sentences, target)`、`save_jsonl(path, rows)`、`load_jsonl(path)`
  这些都是清洗阶段的基础工具。
- `split_by_context_hash(rows, train_size, dev_size, test_size)`：按上下文哈希分桶，避免同一 context 同时落到不同 split。
- `export_training_format(rows)`：导出最简训练格式。
- `resolve_paths(project_root, input_glob)`：把 glob 解析成路径列表。
- `load_all_rows(project_root, input_glob)`：批量加载匹配文件。
- `build_summary(raw_count, cleaned_rows, splits, stats)`：生成清洗摘要。
- `main()`：命令行入口。

类：

- `CleanerConfig`
  保存最小句数、负样本个数、上下文配额等清洗阈值。
- `CqrCleaner`
  真正执行清洗规则的类。

方法：

- `__init__(config)`：保存配置并初始化统计器。
- `clean_rows(rows)`：批量清洗，并处理重复 question-context 与上下文配额。
- `validate_and_normalize(row)`：对单条样本做字段合法性、正负句有效性、supporting 质量等检查。
- `looks_low_quality_question(question)`：判断是否是“根据上文”“是不是”等低质量问题。
- `is_positive_answer_like(positive_sentence, answer)`：判断正句是否已经太像答案句。
- `_deduplicate_preserve_order(items)`：保序去重。

### 10.3 `convert_dureader_to_cqr.py`

定位：把 DuReader 或扁平 QA 样本转换成 CQR 的主脚本，也是 `data_builder/` 的主干文件。

数据结构：

- `AnswerCandidate`：表示候选答案文本和起始位置。
- `SentenceSpan`：表示一句话在原文中的字符区间。
- `NormalizedSample`：统一后的中间样本结构，后续窗口选择和 LLM 审核都基于它。

接口与缓存：

- `LLMClient`：负责向裁判模型发请求。
- `JsonlCache`：一个很实用的小缓存层，避免同样的 prompt 重复调用 LLM。

文本与启发式函数：

- `normalize_space(text)`、`normalize_inline(text)`、`tokenize_mixed(text)`、`overlap_ratio(a, b)`、`contains_normalized(text, snippet)`、`question_factoid_risk(question)`：基础文本工具。
- `build_sentence_profiles(sentences, question, answer)`：为每个句子生成问题重叠、答案重叠、支持/负样本倾向等 profile。
- `heuristic_support_candidates(profiles, answer_indices, factoid_risk)`：根据 profile 挑 supporting 候选句。
- `heuristic_negative_candidates(profiles, answer_indices, support_indices)`：根据 profile 挑 negative 候选句。
- `positive_sentence_is_too_sufficient(sentence, question, answer)`：判断正句是否已经足以单句回答问题。
- `window_quality(window_sentences, question, answer, local_answer_indices)`：给一个窗口打质量分，用于挑最适合构造 CQR 的局部上下文。
- `stable_hash(text)`：文本哈希。

数据读取与规范化：

- `load_rows(path)`：兼容 `.jsonl`、普通 JSON、带 `data` 的 JSON。
- `save_jsonl_row(handle, row)`：逐行写 JSONL。
- `extract_answers(row)`：从不同答案字段格式里提答案候选。
- `pick_best_answer(context, answers)`：在候选答案里挑一个最适合当前上下文的答案。
- `iter_flat_samples(row, row_index)`：把扁平 QA 样本转成 `NormalizedSample`。
- `iter_document_samples(row, row_index, max_paragraphs_per_question)`：把 DuReader 文档式样本拆成段落级样本。
- `normalize_dureader_rows(rows, max_paragraphs_per_question)`：统一规整所有样本。

窗口构建：

- `split_sentences_with_spans(text)`：切句并保留原文位置。
- `locate_answer_sentence_indices(context, sentences, answer, answer_start, question)`：定位答案落在哪些句子里。
- `select_window_indices(num_sentences, center_index, target_sentences)`：围绕中心句扩展窗口。
- `build_windowed_context(sample, min_sentences, max_sentences, target_sentences)`：生成最优局部窗口，供 LLM 选择正负句。

LLM prompt 与解析：

- `build_selection_prompt(sample, window, allow_question_rewrite)`：构造第一阶段“选择/拒绝 CQR 样本”的 prompt。
- `build_verification_prompt(record)`：构造第二阶段“质检样本是否合格”的 prompt。
- `parse_json_response(text)`：从模型输出里提 JSON。
- `normalize_question_type(value, question)`：规范化问题类型标签。

修复与样本落盘：

- `valid_unique_indices(indices, num_sentences)`：过滤非法或重复索引。
- `repair_supporting_indices(supporting_indices, positive_index, negative_indices, window)`：修复 supporting 索引。
- `repair_negative_indices(negative_indices, supporting_indices, window)`：修复 negative 索引。
- `choose_repaired_positive_index(positive_index, supporting_indices, negative_indices, window, question, answer)`：修复后的正句兜底选择。
- `build_record_from_selection(sample, window, selection)`：把 LLM 输出转成最终 CQR 记录。
- `call_cached_json(client, cache, system_prompt, user_prompt)`：带缓存调用 LLM 并解析 JSON。
- `run_selection_and_verification(sample, window, client, cache, allow_question_rewrite)`：执行“先选择，再质检”的两阶段 LLM 审核。
- `load_processed_ids(path)`：断点续跑时跳过已处理样本。
- `build_summary(stats)`：生成摘要。
- `main()`：总入口。

主线记法：原始 QA -> 规范样本 -> 局部窗口 -> LLM 选/验 -> CQR 落盘。

### 10.4 `convert_c3_to_cqr.py`

定位：把 CLUE C3 数据集转成 CQR。

函数：

- `load_c3_dataset(input_path, hf_splits, cache_dir)`：从本地文件或 Hugging Face 读取 C3。
- `flatten_context_units(raw_context)`：把 C3 的 context 字段摊平成句子列表。
- `normalize_c3_rows(rows)`：转成统一的 `NormalizedSample`。
- `build_choice_augmented_question(sample)`：把选择题选项拼到问题里，增强判别信号。
- `build_c3_windowed_context(sample, min_sentences, max_sentences, target_sentences)`：为 C3 构造局部窗口。
- `build_selection_prompt_c3(sample, window, allow_question_rewrite)`：构造 C3 专属 selection prompt。
- `run_selection_and_verification_c3(sample, window, client, cache, allow_question_rewrite)`：执行选择和验证。
- `main()`：总入口。

### 10.5 `cqr_split_utils.py`

定位：通用分割和去重工具。

函数：

- `load_jsonl(path)`、`save_jsonl(path, rows)`：读写 JSONL。
- `source_dataset(row)`：猜样本来自哪个数据集。
- `quality_label(row)`：判断样本是 gold 还是 silver。
- `stable_group_id(row)`：给样本分组，保证同组样本不被拆开。
- `dedupe_key(row)`：生成去重键。
- `dedupe_rows(rows)`：去重。
- `bucket_key(row, stratify_by_dataset=True, stratify_by_quality=True)`：生成分层统计桶键。
- `group_rows(rows, stratify_by_dataset=True, stratify_by_quality=True)`：按 group_id 和 bucket 分组。
- `allocate_counts(total, ratios)`：把样本数按比例分配到 train/dev/test。
- `split_rows_by_group(rows, train_ratio, dev_ratio, test_ratio, seed=42, stratify_by_dataset=True, stratify_by_quality=True)`：分组分层切分样本。
- `summarize_rows(rows)`：汇总样本统计。
- `summarize_splits(splits)`：汇总多个 split 的统计。

### 10.6 `prepare_cqr_training_splits.py`

定位：把合并后的高召回数据集拆成 gold/silver/train/dev/test。

函数：

- `sample_rows(rows, max_rows, seed)`：从 silver 样本中抽样。
- `main()`：先去重，再切 gold split，并把部分 silver 拼到 train 里。

### 10.7 `split_cqr_dataset.py`

定位：更通用的 train/dev/test 切分脚本。

函数：

- `main()`：读输入、可选去重、按组切分并写结果。

### 10.8 `salvage_relaxed_cqr.py`

定位：把被 LLM 丢掉的样本再捞一遍，做 relaxed salvage。

函数：

- `load_dropped_audit(path)`：只读 audit 里被丢弃的样本。
- `maybe_int(value)`：尝试转 int。
- `classify_reason_bucket(reason, reason_mode)`：把丢弃原因归类。
- `should_salvage(reason, reason_mode)`：判断某种原因是否允许 salvage。
- `fallback_window(sample, min_sentences, max_sentences, target_sentences)`：正常窗口构建失败时的兜底窗口。
- `ensure_window(sample, dataset_kind, min_sentences, max_sentences, target_sentences)`：优先用正常窗口，失败再退回 fallback。
- `choose_positive(window, question, answer, allow_direct_answer)`：选择正句。
- `choose_supporting(window, positive_index, aggressive)`：选择 supporting 句。
- `choose_negatives(window, supporting, aggressive)`：选择 negative 句。
- `build_relaxed_selection(sample, window, aggressive)`：构造放宽版 selection。
- `salvage_record(sample, window, audit_row, aggressive)`：从 dropped 样本里尝试恢复 record。
- `infer_dataset_kind(explicit_kind, input_path, dropped_rows)`：推断样本来自 flat 还是 C3。
- `normalize_samples(rows, dataset_kind, max_paragraphs_per_question)`：规范化样本。
- `build_sample_lookup(samples)`：为 sample_id / row_index 建索引。
- `resolve_sample(audit_row, sample_by_key, sample_by_id)`：从 audit 记录回查原始样本。
- `build_job_summary(job, dataset_kind, stats)`：单个 salvage 任务摘要。
- `run_salvage_job(job)`：执行一个 salvage job。
- `build_aggregate_summary(job_summaries)`：合并多个 salvage 任务摘要。
- `load_job_specs(args)`：从命令行或 JSON 配置读多个 job。
- `main()`：总入口。

### 10.9 `export_cmrc2018_flat.py`

定位：把 CMRC2018 导成扁平 JSONL。

函数：

- `main()`：从 Hugging Face 下载数据集并导成统一格式。

### 10.10 `hf_staging/dureader.py`

定位：自定义 Hugging Face `datasets` 数据集加载器。

类：

- `DuReaderConfig`：保存数据源 URL。
- `DuReader`：继承 `datasets.GeneratorBasedBuilder` 的加载器。

方法：

- `DuReaderConfig.__init__(name, data_url, **kwargs)`：初始化数据集配置。
- `DuReader._info()`：定义字段 schema。
- `DuReader._split_generators(dl_manager)`：定义 train/dev/test 文件来源。
- `DuReader._generate_examples(data_file, split)`：根据 split 选择不同生成器。
- `DuReader._generate_robust_examples(data_file)`：生成 robust train/dev 样本。
- `DuReader._generate_robust_test_examples(data_file)`：生成 robust test 样本。
- `DuReader._generate_checklist_examples(data_file)`：生成 checklist train/dev 样本。
- `DuReader._generate_checklist_test_examples(data_file)`：生成 checklist test 样本。

### 10.11 `wait_and_summarize_generation.py`

定位：轮询多个生成任务，等它们跑完后汇总统计。

函数：

- `count_jsonl_lines(path)`：数输出 JSONL 行数。
- `count_audit(path)`：统计 kept / dropped / audit rows。
- `load_run_specs(path)`：读任务配置。
- `build_summary(run_specs)`：汇总所有 run 的统计。
- `all_summaries_exist(run_specs)`：判断所有 summary 文件是否已生成。
- `main()`：轮询等待并输出汇总文件。

备注：

- 这个文件当前有一个真实语法问题：`f.write("\\n".join(lines))` 那一行在源文件里被断开了，导致文件现在不能直接运行。

### 10.12 `source_datasets/dureader_hf/.../evaluate.py`

定位：官方风格评测脚本，主要是数据集自带工具，不是你主工程的核心。

常见函数：

- `_tokenize_chinese_chars(text)`：把中文按字切分。
- `_normalize(in_str)`：去标点和特殊符号。
- `find_lcs(s1, s2)`：求最长公共子序列。
- `evaluate(ref_ans, pred_ans, verbose=False)`：计算整体 F1 / EM。
- `calc_f1_score(answers, prediction)`：算单样本 F1。
- `calc_em_score(answers, prediction)`：算单样本 EM。
- `read_mrc_dataset(filename, tag=None)`：读取 checklist 版数据集。
- `read_model_prediction(filename)`：读取预测文件。

robust 版和 checklist 版逻辑类似，只是数据字段略有不同。

## 11. 其他目录和文件

### `markdown/`

这里是你已经写过的一些说明草稿，比如：

- `context_aware_encoder_model_detailed_explanation.md`
- `pipeline_detailed_explanation.md`
- `cpc_run_instructions.md`
- `window_mechanism_introduction.md`

它们偏“模块解释”和“论文草稿材料”，可以当补充阅读，不是主程序。

### 各种 `outputs/`、`outputs_*`、`outputs_demo/`

这类目录通常是模型权重、tokenizer、训练历史和配置文件，不是源码逻辑。

### `sample_*.jsonl`、`predict_input_example.json`

这些是示例输入或训练样例，主要用来 smoke test 和演示。

### `hf_cache/`

Hugging Face 数据或模型缓存，不是业务代码。

### `.idea/`

IDE 配置，无业务逻辑。

## 12. 这个项目里最容易看不懂的语法

### `from __future__ import annotations`

作用：推迟类型注解求值。

好处：

- 可以在类型提示里直接写 `str | None`。
- 互相引用类名时更省事。

### `@dataclass`

例如：

```python
@dataclass
class ContextAwareEncoderConfig:
    model_name: str = "bert-base-chinese"
```

意思：

- 这是一个“纯配置容器”。
- Python 会自动帮你生成 `__init__`、`__repr__` 等。

### `str | None`、`Sequence[str]`、`Tuple[List[str], dict]`

这些都是类型提示：

- `str | None`：这个值要么是字符串，要么是 `None`。
- `Sequence[str]`：表示“字符串序列”，可以是 list、tuple 等。
- `Tuple[List[str], dict]`：表示函数返回一个二元组，第一个元素是字符串列表，第二个是字典。

### `with torch.no_grad():`

表示这段代码只是推理，不训练，不用记录梯度，省显存、省算力。

### `model.eval()`

表示切到推理模式，关闭 dropout 的随机性。

### `**batch`

例如：

```python
outputs = self.encoder(**batch)
```

含义：把字典展开成关键字参数。

### `nn.Sequential(*layers)`

表示把一个 list 里的层按顺序拼成一个网络。

### 列表推导式 / 字典推导式

例如：

```python
[item["question"] for item in batch]
```

意思就是遍历 `batch`，取出每个元素的 `question`，最后组成新列表。

### `zip(...)`

表示同时并排遍历多个列表。

### `enumerate(...)`

表示遍历列表时同时拿到索引和元素。

### `yield`

在 `iter_flat_samples()`、`iter_document_samples()` 里出现。

含义：

- 这是一个生成器函数。
- 它不是一次性返回全部结果，而是“边生成边返回”。

### `Counter` 和 `defaultdict`

- `Counter`：用来计数。
- `defaultdict`：用来自动给新 key 提供默认值。

### `Path(...) / path.open(...)`

这是 `pathlib` 的写法，比传统字符串路径更稳。

### `state_dict()` / `load_state_dict()`

PyTorch 保存和加载权重的标准方式。

### `BCEWithLogitsLoss`

你在 span 模型里看到它。

意思是：

- 这个任务是二分类。
- 模型直接输出 logit，不先手写 `sigmoid`。
- loss 内部会自动做更稳定的数值计算。

## 13. 这个项目里几个最容易搞混的业务概念

- `positive_sentence` 不等于“答案句”。在 CQR 设定里，它更像关键线索句，但不能单句直接把答案说死。
- `supporting_sentences` 必须包含 `positive_sentence`。
- `negative_sentences` 不是“随机不相关句”，它们必须来自同一上下文，但不能支持答案。
- `target_ratio` 是保留比例，不是删掉比例。
- `second_stage_keep_ratio` 也是保留比例。
- `attention_probe_scores` 只是辅助信号，不是最终选择分数，也不是下游 LLM 的真实内部注意力。
- `MNTP` 在当前代码里更接近“M LM 风格辅助损失”，它的目标是增强上下文建模，而不是一个独立推理模块。

## 14. 我建议你以后怎么回看这个项目

如果以后你又忘了，我建议按这三个问题回看：

### 问题 1：这个项目怎么跑起来

看：

- `main.py`
- `pipeline/compression_pipeline.py`

### 问题 2：一级压缩为什么会这么打分

看：

- `context_aware_encoder_model/context_aware_sentence_encoder.py`
- `context_aware_encoder_model/cqr_train_eval.py`
- `context_aware_encoder_model/train_context_aware_encoder*.py`

### 问题 3：二级压缩为什么会删这些短语

看：

- `pipeline/task_aware_compression.py`
- `pipeline/dac_adapter.py`
- `intra_sentence_model/span_feature_utils.py`
- `intra_sentence_model/generate_span_pseudo_labels.py`

### 问题 4：训练数据从哪来

看：

- `data_builder/convert_dureader_to_cqr.py`
- `data_builder/convert_c3_to_cqr.py`
- `data_builder/clean_cqr_dataset.py`

### 问题 5：压缩率为什么不是写死的

看：

- `target_ratio.py`
- `target_ratio_model/budget_features.py`
- `target_ratio_model/generate_pseudo_labels.py`
- `target_ratio_model/train_budget_predictor.py`

## 15. 当前仓库里值得你额外注意的点

- 这是一个明显在持续演化的实验仓库，不是已经完全收敛的工业代码。
- `data_builder/` 里很多脚本是“数据工程脚本”，和主推理链路不同，不要混着看。
- 一些 `outputs_*`、`raw_batches/`、`training_splits/`、`hf_cache/` 是产物目录，不是核心源码。
- `1.原理复现(LLM版).py` 目前写了硬编码 API key，建议后面改成环境变量。
- `data_builder/wait_and_summarize_generation.py` 当前有语法错误，文档里已经标出来了。

## 16. 一页总结

如果只留一段给未来的你，我会这么写：

这个项目的主线是：

- 用 `context_aware_encoder_model/` 学一个“问题感知的句子编码器”。
- 用 `target_ratio_model/` 学一个“这次该压多少”的预算预测器。
- 用 `pipeline/compression_pipeline.py` 先做句子级筛选。
- 用 `pipeline/task_aware_compression.py` 再做句内 span 级裁剪。
- 用 `intra_sentence_model/` 支撑句内压缩的训练。
- 用 `data_builder/` 把原始 QA 数据加工成能训练这些模块的 CQR 风格数据。

所以它不是一个单点模型，而是一整条“数据构造 -> 编码器训练 -> 预算学习 -> 二级压缩 -> 推理整合”的工程链路。
