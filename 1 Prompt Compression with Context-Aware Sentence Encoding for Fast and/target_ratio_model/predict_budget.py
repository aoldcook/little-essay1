import argparse
import json
from pathlib import Path

import torch

from budget_features import FEATURE_ORDER, build_budget_features, features_to_vector
from budget_model import BudgetConfig, BudgetPredictorMLP, class_to_ratio


def load_json(path: Path):
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_dir", type=str, required=True,default='outputs')
    parser.add_argument("--input_json", type=str, required=True,default='predict_input_example.json')
    args = parser.parse_args()

    model_dir = Path(args.model_dir)
    metadata = load_json(model_dir / "metadata.json")
    config = BudgetConfig(
        ratio_buckets=metadata["ratio_buckets"],
        input_dim=metadata["input_dim"],
        hidden_dims=metadata["hidden_dims"],
    )

    model = BudgetPredictorMLP(config)
    model.load_state_dict(torch.load(model_dir / "budget_predictor.pt", map_location="cpu"))
    model.eval()

    sample = load_json(Path(args.input_json))
    features = build_budget_features(
        question=sample["question"],
        context=sample["context"],
        similarities=sample["similarities"],
    )
    x = torch.tensor(features_to_vector(features)).unsqueeze(0)

    with torch.no_grad():
        logits = model(x)
        pred_cls = int(torch.argmax(logits, dim=1).item())
        pred_ratio = class_to_ratio(pred_cls, config.ratio_buckets)
        probs = torch.softmax(logits, dim=1).squeeze(0).tolist()

    print("predicted_ratio:", pred_ratio)
    print("class_probabilities:")
    for ratio, p in zip(config.ratio_buckets, probs):
        print(f"  ratio={ratio:.1f} prob={p:.4f}")
    print("top_features:")
    for name in FEATURE_ORDER[:10]:
        print(f"  {name}: {features[name]:.4f}")


if __name__ == "__main__":
    main()
