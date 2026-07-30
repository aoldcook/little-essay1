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


class SpanModelSchemaError(RuntimeError):
    """Raised when a checkpoint's feature schema does not match the current code."""


def build_metadata(
    config: SpanClassifierConfig,
    feature_order: List[str],
    threshold: float,
    dac_active: bool | None = None,
    dac_salience_model: str | None = None,
) -> Dict:
    from intra_sentence_model.span_feature_utils import FEATURE_SCHEMA_VERSION

    return {
        "input_dim": config.input_dim,
        "hidden_dims": config.hidden_dims,
        "dropout": config.dropout,
        "feature_order": feature_order,
        "feature_schema_version": FEATURE_SCHEMA_VERSION,
        "threshold": threshold,
        # `dac_score` is a model input whose DISTRIBUTION depends on whether the
        # DAC salience model was live when the training features were generated.
        # Recorded so the loader can refuse a DAC-off model at DAC-on inference.
        "dac_active": dac_active,
        "dac_salience_model": dac_salience_model,
    }


def load_span_model(
    model_dir: str | Path,
    device: str | torch.device,
    runtime_dac_active: bool | None = None,
) -> Tuple[SpanClassifierMLP, Dict]:
    """Load a span classifier, refusing any checkpoint whose features misalign.

    Without this check a checkpoint trained under an older FEATURE_ORDER loads
    happily and reads every feature at the wrong index -- producing confident,
    meaningless scores. Schema v1 checkpoints in particular were trained with the
    oracle features (finding C5) and are not usable.
    """
    from intra_sentence_model.span_feature_utils import (
        FEATURE_ORDER,
        FEATURE_SCHEMA_VERSION,
        assert_no_oracle_features,
    )

    model_dir = Path(model_dir)
    metadata = json.loads((model_dir / "metadata.json").read_text(encoding="utf-8"))

    ckpt_order = list(metadata.get("feature_order") or [])
    ckpt_version = metadata.get("feature_schema_version")
    if ckpt_order and ckpt_order != list(FEATURE_ORDER):
        missing = [f for f in FEATURE_ORDER if f not in ckpt_order]
        extra = [f for f in ckpt_order if f not in FEATURE_ORDER]
        raise SpanModelSchemaError(
            f"Span model at {model_dir} was trained with a different feature schema "
            f"(checkpoint v{ckpt_version}, code v{FEATURE_SCHEMA_VERSION}).\n"
            f"  in code but not checkpoint : {missing}\n"
            f"  in checkpoint but not code : {extra}\n\n"
            "Loading it would silently read features at the wrong indices. Retrain:\n"
            "  python -m intra_sentence_model.train_span_model --split_mode group ..."
        )
    assert_no_oracle_features(ckpt_order or FEATURE_ORDER)

    expected_dim = len(FEATURE_ORDER)
    if int(metadata["input_dim"]) != expected_dim:
        raise SpanModelSchemaError(
            f"Span model input_dim={metadata['input_dim']} but the current feature "
            f"vector has {expected_dim} dimensions. Retrain the span model."
        )

    # `dac_score` is feature index 20. A model trained while DAC was disabled saw
    # that column as a constant 0.0, so its learned weight for it is meaningless
    # once real salience values start arriving -- the same train/inference
    # distribution mismatch as the oracle features (finding C5), reached by a
    # different route. Refuse the combination rather than degrade quietly.
    if runtime_dac_active is not None:
        ckpt_dac = metadata.get("dac_active")
        if ckpt_dac is None and runtime_dac_active:
            raise SpanModelSchemaError(
                f"Span model at {model_dir} records no 'dac_active' flag, so it predates "
                "DAC being functional and was trained with dac_score == 0.0 throughout. "
                "Running it with DAC enabled feeds a feature distribution it never saw.\n\n"
                "Either retrain with DAC enabled, or pass --disable_dac to reproduce the "
                "conditions this checkpoint was trained under."
            )
        if ckpt_dac is not None and bool(ckpt_dac) != bool(runtime_dac_active):
            raise SpanModelSchemaError(
                f"Span model at {model_dir} was trained with dac_active={bool(ckpt_dac)} "
                f"but inference is running with dac_active={bool(runtime_dac_active)}. "
                "The dac_score feature would be out of distribution. Match the training "
                "configuration, or retrain."
            )

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

