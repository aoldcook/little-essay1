from dataclasses import dataclass
from typing import Dict, List

import torch
import torch.nn as nn


@dataclass
class BudgetConfig:
    ratio_buckets: List[float]
    input_dim: int
    hidden_dims: List[int]


class BudgetPredictorMLP(nn.Module):
    def __init__(self, config: BudgetConfig):
        super().__init__()
        layers = []
        dims = [config.input_dim] + config.hidden_dims
        for i in range(len(dims) - 1):
            layers.append(nn.Linear(dims[i], dims[i + 1]))
            layers.append(nn.ReLU())
            layers.append(nn.Dropout(0.1))
        layers.append(nn.Linear(dims[-1], len(config.ratio_buckets)))
        self.net = nn.Sequential(*layers)
        self.config = config

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class BudgetLoss(nn.Module):
    """
    在普通交叉熵之外，对“预测过小”加额外惩罚。
    """
    def __init__(self, under_penalty: float = 0.8):
        super().__init__()
        self.ce = nn.CrossEntropyLoss()
        self.under_penalty = under_penalty

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        ce_loss = self.ce(logits, targets)
        preds = torch.argmax(logits, dim=1)
        under = torch.clamp(targets - preds, min=0).float()
        return ce_loss + self.under_penalty * under.mean()


def ratio_to_class(ratio: float, ratio_buckets: List[float]) -> int:
    diffs = [abs(ratio - b) for b in ratio_buckets]
    return int(min(range(len(diffs)), key=lambda i: diffs[i]))


def class_to_ratio(cls_idx: int, ratio_buckets: List[float]) -> float:
    return float(ratio_buckets[int(cls_idx)])


def build_metadata(config: BudgetConfig, feature_order: List[str]) -> Dict:
    return {
        "ratio_buckets": config.ratio_buckets,
        "input_dim": config.input_dim,
        "hidden_dims": config.hidden_dims,
        "feature_order": feature_order,
    }
