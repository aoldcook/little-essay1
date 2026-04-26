from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Tuple

import json
import torch
import torch.nn as nn


@dataclass
class SpanClassifierConfig:
    input_dim: int
    hidden_dims: List[int]
    dropout: float = 0.1


class SpanClassifierMLP(nn.Module):
    def __init__(self, config: SpanClassifierConfig):
        super().__init__()
        dims = [config.input_dim] + list(config.hidden_dims)
        layers = []
        for i in range(len(dims) - 1):
            layers.append(nn.Linear(dims[i], dims[i + 1]))
            layers.append(nn.ReLU())
            layers.append(nn.Dropout(config.dropout))
        layers.append(nn.Linear(dims[-1], 1))
        self.net = nn.Sequential(*layers)
        self.config = config

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x).squeeze(-1)


def build_metadata(config: SpanClassifierConfig, feature_order: List[str], threshold: float) -> Dict:
    return {
        "input_dim": config.input_dim,
        "hidden_dims": config.hidden_dims,
        "dropout": config.dropout,
        "feature_order": feature_order,
        "threshold": threshold,
    }


def load_span_model(model_dir: str | Path, device: str | torch.device) -> Tuple[SpanClassifierMLP, Dict]:
    model_dir = Path(model_dir)
    metadata = json.loads((model_dir / "metadata.json").read_text(encoding="utf-8"))
    config = SpanClassifierConfig(
        input_dim=int(metadata["input_dim"]),
        hidden_dims=list(metadata["hidden_dims"]),
        dropout=float(metadata.get("dropout", 0.1)),
    )
    model = SpanClassifierMLP(config)
    model_path = model_dir / "span_model.best.pt"
    if not model_path.exists():
        model_path = model_dir / "span_model.pt"
    state = torch.load(model_path, map_location=device)
    model.load_state_dict(state)
    model.to(device)
    model.eval()
    return model, metadata

