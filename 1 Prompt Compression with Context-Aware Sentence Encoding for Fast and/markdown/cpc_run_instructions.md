# CPC 融合版代码运行说明

这份文档说明两个脚本的作用、输入输出格式，以及推荐的运行顺序。

---

## 一、文件说明

当前新增了两个核心脚本：

1. `data_builder/build_cqr_with_filters.py`  
   用于从原始候选样本构造更接近 CQR 风格的训练数据。

2. `context_aware_encoder_model/train_context_aware_encoder_with_mntp.py`  
   用于训练带有对比学习 + MLM/MNTP 近似损失的上下文感知句子编码器。

同时我给你准备了一份可以直接测试的数据：

- `data_builder/raw_candidates_more.jsonl`

---

## 二、原始候选数据格式

每条原始候选数据采用 JSONL 格式，一行一个 JSON 对象，字段如下：

```json
{
  "context": "...",
  "question": "...",
  "answer": "...",
  "positive_sentence": "..."
}
```

字段含义：

- `context`：完整上下文
- `question`：问题
- `answer`：标准答案或参考答案
- `positive_sentence`：上下文中与问题相关、希望作为正样本候选的句子

---

## 三、第一步：构造过滤后的 CQR 风格数据

### 1. 最基础运行方式

先用最简单的离线版本跑通，不启用 API 验证，也不启用 KL 过滤：

```bash
python -m data_builder.build_cqr_with_filters \
  --input_jsonl data_builder/raw_candidates_more.jsonl \
  --output_jsonl context_aware_encoder_model/cqr_filtered.jsonl
```

这一步会做：

- heuristic 版 question verification
- similarity-based negative filtering
- 不启用 KL filtering

输出文件：

```text
context_aware_encoder_model/cqr_filtered.jsonl
```

---

### 2. 如果你要启用 OpenAI-compatible question verification

```bash
python -m data_builder.build_cqr_with_filters \
  --input_jsonl data_builder/raw_candidates_more.jsonl \
  --output_jsonl context_aware_encoder_model/cqr_filtered.jsonl \
  --verification_mode openai_compatible \
  --base_url http://127.0.0.1:8000/v1 \
  --api_key YOUR_KEY \
  --verify_model YOUR_VERIFY_MODEL
```

说明：

- `base_url`：你的兼容 OpenAI 接口地址
- `api_key`：你的密钥
- `verify_model`：用于 verification 的模型名

---

### 3. 如果你要进一步启用 KL 过滤

```bash
python -m data_builder.build_cqr_with_filters \
  --input_jsonl data_builder/raw_candidates_more.jsonl \
  --output_jsonl context_aware_encoder_model/cqr_filtered.jsonl \
  --use_kl_filter \
  --kl_model Qwen/Qwen2.5-0.5B-Instruct
```

说明：

- `--use_kl_filter`：启用 KL-based negative filtering
- `--kl_model`：用来估计答案分布差异的因果语言模型

注意：这一步会明显更慢。

---

## 四、第二步：训练多任务句子编码器

在得到 `cqr_filtered.jsonl` 之后，再运行训练脚本：

```bash
python -m context_aware_encoder_model.train_context_aware_encoder_with_mntp \
  --train_file context_aware_encoder_model/cqr_filtered.jsonl \
  --output_dir context_aware_encoder_model/outputs_mntp \
  --model_name Qwen/Qwen3-Embedding-8B \
  --epochs 3 \
  --batch_size 4 \
  --lr 2e-5 \
  --max_length 512 \
  --lambda_mntp 1.0
```

参数说明：

- `train_file`：上一步生成的过滤后训练集
- `output_dir`：模型输出目录
- `model_name`：底座编码器
- `epochs`：训练轮数
- `batch_size`：批大小
- `lr`：学习率
- `max_length`：最大输入长度
- `lambda_mntp`：MLM/MNTP 近似损失的权重

训练完成后，模型会保存到：

```text
context_aware_encoder_model/outputs_mntp
```

---

## 五、推荐运行顺序

建议你按下面顺序做：

### 第一步
先用现成样例数据测试数据构造：

```bash
python -m data_builder.build_cqr_with_filters \
  --input_jsonl data_builder/raw_candidates_more.jsonl \
  --output_jsonl context_aware_encoder_model/cqr_filtered.jsonl
```

### 第二步
再用过滤后的数据训练 encoder：

```bash
python -m context_aware_encoder_model.train_context_aware_encoder_with_mntp \
  --train_file context_aware_encoder_model/cqr_filtered.jsonl \
  --output_dir context_aware_encoder_model/outputs_mntp
```

---

## 六、这两个脚本分别解决什么问题

### 1. `build_cqr_with_filters.py`
它解决的是**训练数据质量**问题。

如果没有这一步，正负样本可能不够干净，模型容易学成“表面关键词匹配器”。

它的价值在于：

- 过滤掉不符合上下文依赖假设的问题
- 筛选更干净的负样本
- 让训练数据更接近 CQR 风格

---

### 2. `train_context_aware_encoder_with_mntp.py`
它解决的是**训练目标不足**问题。

如果只做对比学习，模型主要学到“哪句更像问题”。  
加入 MLM/MNTP 近似损失后，模型还会学到“上下文内部的语言结构和语义依赖”。

它的价值在于：

- 提升 encoder 的上下文建模能力
- 更贴近你启发论文中的联合训练方式
- 为后续与 `BGE + learned budget` 做对比提供更强版本

---

## 七、补充建议

1. 先用默认参数跑通
2. 确认 `cqr_filtered.jsonl` 非空
3. 再逐步尝试：
   - OpenAI verification
   - KL filter
   - 更大的训练集
   - 更长训练轮数

这样更稳，不容易一上来就因为数据或算力问题卡住。

