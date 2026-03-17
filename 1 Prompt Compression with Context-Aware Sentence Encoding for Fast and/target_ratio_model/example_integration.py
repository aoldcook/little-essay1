"""
把预算预测器接到你的压缩器中的示例。
这里假设你已经有 sentence ranker，会输出 similarities。
"""

import json
from pathlib import Path

import torch

from budget_features import build_budget_features, split_sentences, features_to_vector
from budget_model import BudgetConfig, BudgetPredictorMLP, class_to_ratio


class LearnedBudgetSelector:
    def __init__(self, model_dir: str):
        model_dir = Path(model_dir)
        with (model_dir / "metadata.json").open("r", encoding="utf-8") as f:
            metadata = json.load(f)
        config = BudgetConfig(
            ratio_buckets=metadata["ratio_buckets"],
            input_dim=metadata["input_dim"],
            hidden_dims=metadata["hidden_dims"],
        )
        self.model = BudgetPredictorMLP(config)
        self.model.load_state_dict(torch.load(model_dir / "budget_predictor.pt", map_location="cpu"))
        self.model.eval()
        self.ratio_buckets = config.ratio_buckets

    def predict_ratio(self, question: str, context: str, similarities):
        features = build_budget_features(question, context, similarities)
        x = torch.tensor(features_to_vector(features)).unsqueeze(0)
        with torch.no_grad():
            logits = self.model(x)
            pred_cls = int(torch.argmax(logits, dim=1).item())
        return class_to_ratio(pred_cls, self.ratio_buckets)


def compress_with_scores(context: str, sentences, similarities, target_ratio: float):
    scores_and_sentences = list(zip(similarities, sentences))
    scores_and_sentences.sort(key=lambda x: x[0], reverse=True)

    total_len = sum(len(s) for s in sentences)
    target_len = max(1, int(total_len * target_ratio))

    selected = []
    cur_len = 0
    for score, sent in scores_and_sentences:
        if cur_len + len(sent) <= target_len:
            selected.append(sent)
            cur_len += len(sent)

    selected_set = set(selected)
    ordered = [s for s in sentences if s in selected_set]
    return "".join(ordered)


def demo(sentence_ranker, model_dir: str, context: str, question: str):
    sentences = split_sentences(context)
    similarities = sentence_ranker(question, sentences)

    budget_selector = LearnedBudgetSelector(model_dir)
    target_ratio = budget_selector.predict_ratio(question, context, similarities)
    compressed = compress_with_scores(context, sentences, similarities, target_ratio)

    return {
        "target_ratio": target_ratio,
        "compressed_context": compressed,
        "original_sentences": len(sentences),
        "kept_chars": len(compressed),
        "original_chars": len(context),
    }
