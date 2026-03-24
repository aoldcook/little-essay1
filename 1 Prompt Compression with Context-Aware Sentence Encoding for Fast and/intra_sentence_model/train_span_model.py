from __future__ import annotations

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import argparse
import json
import random
from typing import Tuple

import numpy as np
import torch
from torch.utils.data import DataLoader, TensorDataset

from intra_sentence_model.span_dataset import build_xy, load_jsonl
from intra_sentence_model.span_feature_utils import FEATURE_ORDER
from intra_sentence_model.span_model import SpanClassifierConfig, SpanClassifierMLP, build_metadata


def split_train_dev(X: np.ndarray, y: np.ndarray, dev_ratio: float, seed: int = 42) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    indices = np.arange(len(X))
    rng = np.random.default_rng(seed)
    rng.shuffle(indices)
    dev_size = int(len(indices) * dev_ratio)
    dev_idx = indices[:dev_size]
    train_idx = indices[dev_size:]
    return X[train_idx], y[train_idx], X[dev_idx], y[dev_idx]


def evaluate(model: SpanClassifierMLP, loader: DataLoader, device: str) -> Tuple[float, float]:
    model.eval()
    total_loss = 0.0
    total = 0
    correct = 0
    criterion = torch.nn.BCEWithLogitsLoss()
    with torch.no_grad():
        for xb, yb in loader:
            xb = xb.to(device)
            yb = yb.to(device)
            logits = model(xb)
            loss = criterion(logits, yb)
            probs = torch.sigmoid(logits)
            preds = (probs >= 0.5).float()
            total_loss += float(loss.item()) * xb.size(0)
            correct += int((preds == yb).sum().item())
            total += int(yb.size(0))
    return total_loss / max(total, 1), correct / max(total, 1)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--train_file", type=str, required=True)
    parser.add_argument("--output_dir", type=str, required=True)
    parser.add_argument("--epochs", type=int, default=60)
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--dev_ratio", type=float, default=0.1)
    args = parser.parse_args()

    random.seed(42)
    np.random.seed(42)
    torch.manual_seed(42)

    rows = load_jsonl(Path(args.train_file))
    X, y = build_xy(rows)
    X_train, y_train, X_dev, y_dev = split_train_dev(X, y, args.dev_ratio)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    train_dataset = TensorDataset(torch.tensor(X_train, dtype=torch.float32), torch.tensor(y_train, dtype=torch.float32))
    dev_dataset = TensorDataset(torch.tensor(X_dev, dtype=torch.float32), torch.tensor(y_dev, dtype=torch.float32))
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
    dev_loader = DataLoader(dev_dataset, batch_size=args.batch_size)

    config = SpanClassifierConfig(input_dim=X.shape[1], hidden_dims=[64, 32], dropout=0.1)
    model = SpanClassifierMLP(config).to(device)

    pos = max(float(y_train.sum()), 1.0)
    neg = max(float(len(y_train) - y_train.sum()), 1.0)
    pos_weight = torch.tensor([neg / pos], dtype=torch.float32, device=device)
    criterion = torch.nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)

    best_dev_loss = float("inf")
    best_state = None
    for epoch in range(1, args.epochs + 1):
        model.train()
        total_loss = 0.0
        total = 0
        for xb, yb in train_loader:
            xb = xb.to(device)
            yb = yb.to(device)
            optimizer.zero_grad()
            logits = model(xb)
            loss = criterion(logits, yb)
            loss.backward()
            optimizer.step()
            total_loss += float(loss.item()) * xb.size(0)
            total += int(xb.size(0))

        train_loss = total_loss / max(total, 1)
        dev_loss, dev_acc = evaluate(model, dev_loader, device)
        if dev_loss < best_dev_loss:
            best_dev_loss = dev_loss
            best_state = {k: v.detach().cpu() for k, v in model.state_dict().items()}

        if epoch == 1 or epoch % 10 == 0 or epoch == args.epochs:
            print(f"epoch={epoch:03d} train_loss={train_loss:.4f} dev_loss={dev_loss:.4f} dev_acc={dev_acc:.4f}")

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    if best_state is not None:
        model.load_state_dict(best_state)
    torch.save(model.state_dict(), output_dir / "span_model.pt")
    with (output_dir / "metadata.json").open("w", encoding="utf-8") as f:
        json.dump(build_metadata(config, FEATURE_ORDER, threshold=0.5), f, ensure_ascii=False, indent=2)

    print("saved:", output_dir / "span_model.pt")
    print("saved:", output_dir / "metadata.json")


if __name__ == "__main__":
    main()
