# 可直接替换的重构代码 v2

把这个目录下的 3 个子目录复制到你的父目录：

- `target_ratio_model/`
- `context_aware_encoder_model/`
- `pipeline/`

并保留每个目录中的 `__init__.py`。

建议从父目录运行，例如：

```bash
python -m target_ratio_model.train_budget_predictor --train_file target_ratio_model/sample_budget_train.jsonl --output_dir target_ratio_model/outputs
python -m target_ratio_model.predict_budget --model_dir target_ratio_model/outputs --input_json target_ratio_model/predict_input_example.json
python -m context_aware_encoder_model.train_context_aware_encoder --train_file context_aware_encoder_model/sample_cqr_train.jsonl --output_dir context_aware_encoder_model/outputs
```
