# Learned Budget Predictor Demo

这是一个最小可运行版本，用于把 `find_target_ratio` 从启发式规则升级成“可学习的小模型”。

## 目录说明

- `budget_features.py`: 从 question / context / similarities 提取预算预测特征
- `budget_model.py`: 小型 MLP + 带“低估预算惩罚”的损失函数
- `train_budget_predictor.py`: 训练脚本
- `predict_budget.py`: 推理脚本
- `example_integration.py`: 如何接到你当前压缩器中的示例
- `sample_budget_train.jsonl`: 示例训练数据
- `predict_input_example.json`: 示例推理输入

## 训练数据格式

每行一个 JSON：

```json
{
  "id": "ex01",
  "question": "...",
  "context": "...",
  "similarities": [0.93, 0.90, 0.87, 0.83],
  "label_ratio": 0.4
}
```

其中：
- `similarities` 是你已有 sentence ranker 输出的句子相关度分数（建议按降序）
- `label_ratio` 是最小安全预算标签，可来自 full-context teacher 扫描

## 安装

```bash
pip install numpy torch
```

## 训练

```bash
python train_budget_predictor.py \
  --train_file sample_budget_train.jsonl \
  --output_dir outputs
```

## 推理

```bash
python predict_budget.py \
  --model_dir outputs \
  --input_json predict_input_example.json
```

## 你后面如何换成真实标签

1. 用你现有的 embedding / LLM 打分器得到 `similarities`
2. 扫描多个 ratio，例如 0.2~0.7
3. 对每个 ratio 压缩上下文并让目标 LLM 回答
4. 找到“语义保持达标”的最小 ratio，写入 `label_ratio`
5. 用真实数据重新训练预算预测器

## 当前版本的定位

这是一个适合论文原型和 ablation 的第一版：
- 先验证“预算学习化”是否有效
- 再逐步替换特征、标签和网络结构
