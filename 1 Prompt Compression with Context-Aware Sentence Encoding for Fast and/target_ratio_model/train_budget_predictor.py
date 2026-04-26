import argparse
import json
import random
from pathlib import Path
from typing import List

import numpy as np
import torch
from torch.utils.data import DataLoader, TensorDataset

from target_ratio_model.budget_features import FEATURE_ORDER, build_budget_features, features_to_vector
from target_ratio_model.budget_model import (
    BudgetConfig,
    BudgetLoss,
    BudgetPredictorMLP,
    build_metadata,
    ratio_to_class,
)


DEFAULT_BUCKETS = [0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]


def load_jsonl(path: Path) -> List[dict]:
    data = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                data.append(json.loads(line))
    return data


def build_dataset(samples: List[dict], ratio_buckets: List[float]):
    X, y = [], []
    for sample in samples:
        feats = build_budget_features(
            question=sample["question"],
            context=sample["context"],
            similarities=sample["similarities"],
        )
        X.append(features_to_vector(feats))
        y.append(ratio_to_class(sample["label_ratio"], ratio_buckets))
    return np.stack(X), np.array(y, dtype=np.int64)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--train_file", type=str, required=True)
    parser.add_argument("--output_dir", type=str, required=True)
    parser.add_argument("--epochs", type=int, default=120)
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--lr", type=float, default=1e-3)
    args = parser.parse_args()

    random.seed(42)
    np.random.seed(42)
    torch.manual_seed(42)

    samples = load_jsonl(Path(args.train_file))
    ratio_buckets = DEFAULT_BUCKETS
    X, y = build_dataset(samples, ratio_buckets)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    X_tensor = torch.tensor(X, dtype=torch.float32)
    y_tensor = torch.tensor(y, dtype=torch.long)

    dataset = TensorDataset(X_tensor, y_tensor)
    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True)

    config = BudgetConfig(
        ratio_buckets=ratio_buckets,
        input_dim=X.shape[1],
        hidden_dims=[64, 32],
    )
    model = BudgetPredictorMLP(config).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    criterion = BudgetLoss(under_penalty=0.8)

    model.train()
    for epoch in range(1, args.epochs + 1):
        running_loss = 0.0
        correct = 0
        total = 0
        for xb, yb in loader:
            xb, yb = xb.to(device), yb.to(device)
            optimizer.zero_grad()
            logits = model(xb)
            loss = criterion(logits, yb)
            loss.backward()
            optimizer.step()

            running_loss += float(loss.item()) * xb.size(0)
            preds = torch.argmax(logits, dim=1)
            correct += int((preds == yb).sum().item())
            total += int(yb.size(0))

        if epoch == 1 or epoch % 20 == 0 or epoch == args.epochs:
            print(f"epoch={epoch:03d} loss={running_loss/total:.4f} acc={correct/total:.4f}")

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    torch.save(model.state_dict(), output_dir / "budget_predictor.pt")
    with (output_dir / "metadata.json").open("w", encoding="utf-8") as f:
        json.dump(build_metadata(config, FEATURE_ORDER), f, ensure_ascii=False, indent=2)

    print("saved:", output_dir / "budget_predictor.pt")
    print("saved:", output_dir / "metadata.json")


if __name__ == "__main__":
    main()

