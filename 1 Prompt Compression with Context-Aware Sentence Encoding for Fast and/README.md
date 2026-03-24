# CPC 项目增强版

这个目录现在实现的是一个两级压缩原型：

1. 句子级压缩：`context-aware sentence encoder + learned budget + MMR selector`
2. 句内压缩：`task-aware dynamic span compression`

第二级压缩融合了四条思路：

- `TACO-RL`：把任务相关奖励信号引入压缩决策
- `DAC`：在句内做动态 attention-aware span 剪裁
- `Sentinel`：用 proxy encoder attention 做轻量 probing
- `Sentence-Anchored Gist Compression`：以句子/子句边界作为稳定压缩单元

## 目录说明

- `context_aware_encoder_model/`: 句子级编码器与训练脚本
- `target_ratio_model/`: 自适应预算预测器
- `pipeline/`: 端到端压缩管线
- `data_builder/`: CQR 风格数据构造脚本
- `main.py`: 最小可运行示例

## 直接运行

```bash
python main.py
```

运行后会输出：

- `semantic_similarities`: 原始语义相关分数
- `attention_probe_scores`: Sentinel 风格 attention probing 分数
- `task_rewards`: TACO-RL 风格任务奖励信号
- `selection_scores`: 句子级最终选择分数
- `removed_span_count`: 第二阶段句内压缩删除的 span 数量

## 训练脚本

```bash
python -m target_ratio_model.train_budget_predictor --train_file target_ratio_model/sample_budget_train.jsonl --output_dir target_ratio_model/outputs
python -m target_ratio_model.predict_budget --model_dir target_ratio_model/outputs --input_json target_ratio_model/predict_input_example.json
python -m context_aware_encoder_model.train_context_aware_encoder --train_file context_aware_encoder_model/sample_cqr_train.jsonl --output_dir context_aware_encoder_model/outputs
python -m context_aware_encoder_model.train_context_aware_encoder_with_mntp --train_file context_aware_encoder_model/sample_cqr_train.jsonl --output_dir context_aware_encoder_model/outputs_mntp
```

## 当前增强的核心入口

- `pipeline/compression_pipeline.py`: 两级压缩总入口
- `pipeline/task_aware_compression.py`: 句内动态 span 压缩
- `context_aware_encoder_model/context_aware_sentence_encoder.py`: attention probing 支持
