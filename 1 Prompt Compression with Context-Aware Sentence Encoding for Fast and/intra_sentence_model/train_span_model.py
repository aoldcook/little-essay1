from __future__ import annotations

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import argparse
import json
import random
from typing import Any, Tuple

import numpy as np
import torch
from torch.utils.data import DataLoader, TensorDataset

from intra_sentence_model.span_dataset import build_xy, build_xy_group_disjoint, load_jsonl
from intra_sentence_model.span_feature_utils import (
    FEATURE_ORDER,
    FEATURE_SCHEMA_VERSION,
    assert_no_oracle_features,
)
from intra_sentence_model.span_model import SpanClassifierConfig, SpanClassifierMLP, build_metadata


def split_train_dev_span_level(X: np.ndarray, y: np.ndarray, dev_ratio: float, seed: int = 42) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """DEPRECATED: leaks spans from one example across train and dev.

    Retained only to reproduce historical (optimistic) numbers for comparison.
    See EVAL_VALIDITY_AUDIT.md finding C4 and use --split_mode group instead.
    """
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


def save_json(path: Path, payload: Any) -> None:
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--train_file", type=str, required=True)
    parser.add_argument("--output_dir", type=str, required=True)
    parser.add_argument("--epochs", type=int, default=60)
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--dev_ratio", type=float, default=0.1)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--split_mode", choices=["group", "span"], default="group",
                        help="'group' splits by source example (leakage-free, required for "
                             "reportable numbers). 'span' reproduces the legacy leaky split.")
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    # Fail before training if a gold-answer-derived feature is in the input vector.
    assert_no_oracle_features(FEATURE_ORDER)
    print(f"feature_schema_version={FEATURE_SCHEMA_VERSION} num_features={len(FEATURE_ORDER)}")

    rows = load_jsonl(Path(args.train_file))

    # The dac_score column is only meaningful if the whole training file was
    # generated under one DAC regime. A file mixing DAC-on and DAC-off rows has
    # two different feature distributions stacked in one column.
    dac_flags = {bool(r.get("dac_active")) for r in rows if "dac_active" in r}
    if len(dac_flags) > 1:
        raise SystemExit(
            "Training file mixes dac_active=True and dac_active=False rows, so the "
            "dac_score feature has two different distributions in one column. "
            "Regenerate pseudo-labels under a single DAC configuration."
        )
    dac_active = next(iter(dac_flags)) if dac_flags else None

    label_policies = {str(r.get("label_policy")) for r in rows if r.get("label_policy")}
    if len(label_policies) > 1:
        raise SystemExit(
            f"Training file mixes label policies {sorted(label_policies)}. The target "
            "would mean different things on different rows. Regenerate under one policy."
        )
    label_policy = next(iter(label_policies)) if label_policies else None
    reader_models = {str(r.get("label_reader_model")) for r in rows if r.get("label_reader_model")}
    label_reader_model = next(iter(reader_models)) if len(reader_models) == 1 else None
    print(f"label_policy={label_policy} label_reader_model={label_reader_model}")
    if label_policy in {"heuristic", "feedback"}:
        print(
            "\n*** WARNING: this policy derives the label from a weighted sum over the "
            "SAME features the model receives as input, so a high dev accuracy means "
            "the model rediscovered that formula, not that it learned anything. Use "
            "--label_policy reader for a target the model cannot see. ***\n"
        )
    dac_models = {
        str(r.get("dac_salience_model")) for r in rows if r.get("dac_salience_model")
    }
    dac_salience_model = next(iter(dac_models)) if len(dac_models) == 1 else None
    if dac_active is None:
        print(
            "\n*** WARNING: pseudo-labels carry no 'dac_active' flag (generated before "
            "DAC was functional). The checkpoint will record dac_active=None and will "
            "be REFUSED at inference when DAC is enabled. Regenerate pseudo-labels to "
            "train a DAC-aware span model. ***\n"
        )
    else:
        print(f"dac_active={dac_active} dac_salience_model={dac_salience_model}")

    # A file can claim dac_active=True while the dac_score column is constant --
    # e.g. if the generator dropped the sentence argument, so every span scored
    # 0.0 and normalisation mapped the degenerate range to a uniform 0.5. That
    # trains a dead input while the checkpoint advertises a live DAC signal.
    if dac_active:
        dac_values = {
            round(float(span["features"]["dac_score"]), 6)
            for row in rows
            for span in row.get("spans", [])
            if "dac_score" in span.get("features", {})
        }
        if len(dac_values) <= 1:
            raise SystemExit(
                f"dac_active=True but dac_score takes only {len(dac_values)} distinct "
                f"value(s) {sorted(dac_values)[:3]} across the whole file, i.e. it is a "
                "constant column carrying no information.\n\n"
                "This usually means pseudo-label generation called "
                "compute_dac_span_scores without the sentence argument. Regenerate the "
                "pseudo-labels, or train with DAC genuinely disabled."
            )

    if args.split_mode == "group":
        X_train, y_train, X_dev, y_dev, split_stats = build_xy_group_disjoint(
            rows, args.dev_ratio, seed=args.seed
        )
        X = X_train
        print("split_mode=group (leakage-free) split_stats=", json.dumps(split_stats))
    else:
        print(
            "\n*** WARNING: --split_mode span shuffles individual spans, so spans from "
            "the same source example appear in BOTH train and dev. Dev accuracy will be "
            "optimistic (audit finding C4). Use --split_mode group for reportable "
            "numbers. ***\n"
        )
        X, y = build_xy(rows)
        X_train, y_train, X_dev, y_dev = split_train_dev_span_level(
            X, y, args.dev_ratio, seed=args.seed
        )
        split_stats = {"split_mode": "span_level_LEAKY"}

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

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_path = output_dir / "last_checkpoint.pt"
    history_path = output_dir / "training_history.json"
    best_metrics_path = output_dir / "best_metrics.json"
    best_model_path = output_dir / "span_model.best.pt"

    start_epoch = 1
    best_dev_loss = float("inf")
    best_state = None
    best_metrics: dict[str, Any] = {}
    history: list[dict[str, float]] = []

    if args.resume and checkpoint_path.exists():
        checkpoint = torch.load(checkpoint_path, map_location=device)
        model.load_state_dict(checkpoint["model_state"])
        optimizer.load_state_dict(checkpoint["optimizer_state"])
        start_epoch = int(checkpoint.get("epoch", 0)) + 1
        best_dev_loss = float(checkpoint.get("best_dev_loss", float("inf")))
        best_state = checkpoint.get("best_state")
        best_metrics = checkpoint.get("best_metrics", {})
        history = checkpoint.get("history", [])
        print(f"resuming from epoch={start_epoch:03d} best_dev_loss={best_dev_loss:.4f}")

    if start_epoch > args.epochs:
        print(f"nothing to do: start_epoch={start_epoch:03d} exceeds epochs={args.epochs:03d}")
    else:
        for epoch in range(start_epoch, args.epochs + 1):
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
            history.append(
                {
                    "epoch": epoch,
                    "train_loss": float(train_loss),
                    "dev_loss": float(dev_loss),
                    "dev_acc": float(dev_acc),
                }
            )

            if dev_loss < best_dev_loss:
                best_dev_loss = dev_loss
                best_state = {k: v.detach().cpu() for k, v in model.state_dict().items()}
                best_metrics = {
                    "epoch": epoch,
                    "train_loss": float(train_loss),
                    "dev_loss": float(dev_loss),
                    "dev_acc": float(dev_acc),
                    "device": device,
                    "batch_size": args.batch_size,
                    "lr": args.lr,
                    "num_examples": int(len(rows)),
                    "train_size": int(len(X_train)),
                    "dev_size": int(len(X_dev)),
                }
                torch.save(best_state, best_model_path)

            checkpoint_payload = {
                "epoch": epoch,
                "model_state": model.state_dict(),
                "optimizer_state": optimizer.state_dict(),
                "best_dev_loss": best_dev_loss,
                "best_state": best_state,
                "best_metrics": best_metrics,
                "history": history,
                "args": vars(args),
            }
            torch.save(checkpoint_payload, checkpoint_path)
            save_json(history_path, history)
            save_json(best_metrics_path, best_metrics)
            print(f"epoch={epoch:03d} train_loss={train_loss:.4f} dev_loss={dev_loss:.4f} dev_acc={dev_acc:.4f}", flush=True)

    if best_state is not None:
        model.load_state_dict(best_state)
    torch.save(model.state_dict(), output_dir / "span_model.pt")
    with (output_dir / "metadata.json").open("w", encoding="utf-8") as f:
        json.dump(
            build_metadata(
                config,
                FEATURE_ORDER,
                threshold=0.5,
                dac_active=dac_active,
                dac_salience_model=dac_salience_model,
                label_policy=label_policy,
                label_reader_model=label_reader_model,
            ),
            f,
            ensure_ascii=False,
            indent=2,
        )

    print("saved:", output_dir / "span_model.pt")
    print("saved:", output_dir / "metadata.json")
    if best_model_path.exists():
        print("saved:", best_model_path)
    if best_metrics:
        print("best_dev_loss=", f"{best_dev_loss:.4f}")


if __name__ == "__main__":
    main()
