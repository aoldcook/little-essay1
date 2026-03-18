from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import torch

from context_aware_encoder_model.context_aware_sentence_encoder import (
    ContextAwareEncoderConfig,
    ContextAwareSentenceEncoder,
    build_marked_context,
    split_sentences,
)


def select_with_mmr(
    similarities: List[float],
    sentence_embeddings: torch.Tensor,
    sentences: Sequence[str],
    target_ratio: float,
    lambda_relevance: float = 0.7,
) -> List[int]:
    """
    一个简单可用的 MMR 句子选择器。
    预算这里默认按字符数近似，更适合中文最小版本实验。
    """
    total_len = sum(len(s) for s in sentences)
    budget = max(1, int(total_len * target_ratio))

    selected: List[int] = []
    remaining = set(range(len(sentences)))
    used_len = 0

    sims = np.asarray(similarities, dtype=float)
    embs = sentence_embeddings.detach().cpu().numpy()

    while remaining:
        best_idx = None
        best_score = -1e18
        for i in remaining:
            sent_len = len(sentences[i])
            if used_len + sent_len > budget:
                continue

            rel = sims[i]
            if not selected:
                score = rel
            else:
                red = max(float(np.dot(embs[i], embs[j])) for j in selected)
                score = lambda_relevance * rel - (1.0 - lambda_relevance) * red

            if score > best_score:
                best_score = score
                best_idx = i

        if best_idx is None:
            break

        selected.append(best_idx)
        remaining.remove(best_idx)
        used_len += len(sentences[best_idx])

    return sorted(selected)


class BudgetPredictorAdapter:
    """
    适配 target_ratio_model 工程。
    如果你不想依赖该模型，可以把这里替换成固定 ratio 或你自己的模型。
    """
    def __init__(self, budget_model_dir: str):
        self.enabled = budget_model_dir is not None
        if not self.enabled:
            return

        from target_ratio_model.budget_features import build_budget_features, features_to_vector
        from target_ratio_model.budget_model import BudgetConfig, BudgetPredictorMLP, class_to_ratio

        self.build_budget_features = build_budget_features
        self.features_to_vector = features_to_vector
        self.class_to_ratio = class_to_ratio

        model_dir = Path(budget_model_dir)
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

    def predict_ratio(self, question: str, context: str, similarities: List[float], fallback_ratio: float = 0.4) -> float:
        if not self.enabled:
            return fallback_ratio
        feats = self.build_budget_features(question=question, context=context, similarities=similarities)
        x = torch.tensor(self.features_to_vector(feats)).unsqueeze(0)
        with torch.no_grad():
            logits = self.model(x)
            pred_cls = int(torch.argmax(logits, dim=1).item())
        return float(self.class_to_ratio(pred_cls, self.ratio_buckets))


class ContextAwareCompressor:
    def __init__(self, encoder_dir: str, budget_model_dir: Optional[str] = None, device: Optional[str] = None):
        device = device or ("cuda" if torch.cuda.is_available() else "cpu")

        with (Path(encoder_dir) / "encoder_config.json").open("r", encoding="utf-8") as f:
            cfg_dict = json.load(f)
        cfg_dict["device"] = device
        cfg = ContextAwareEncoderConfig(**cfg_dict)

        self.encoder = ContextAwareSentenceEncoder(cfg)
        self.encoder.encoder = self.encoder.encoder.from_pretrained(encoder_dir)
        self.encoder.tokenizer = self.encoder.tokenizer.from_pretrained(encoder_dir)
        self.encoder.start_id = self.encoder.tokenizer.convert_tokens_to_ids(cfg.marker_start)
        self.encoder.end_id = self.encoder.tokenizer.convert_tokens_to_ids(cfg.marker_end)
        self.encoder.to(self.encoder.device)
        self.encoder.eval()

        self.budget_selector = BudgetPredictorAdapter(budget_model_dir) if budget_model_dir else None

    def score_context(self, question: str, context: str) -> Tuple[List[str], List[float], torch.Tensor]:
        sentences = split_sentences(context)
        marked_contexts = [
            build_marked_context(sentences, i, self.encoder.config.marker_start, self.encoder.config.marker_end)
            for i in range(len(sentences))
        ]
        similarities, sent_embs = self.encoder.score_sentences(question, sentences, marked_contexts)
        return sentences, similarities, sent_embs

    def compress(
        self,
        question: str,
        context: str,
        target_ratio: Optional[float] = None,
        lambda_relevance: float = 0.7,
        fallback_ratio: float = 0.4,
    ) -> Dict:
        sentences, similarities, sent_embs = self.score_context(question, context)

        if target_ratio is None:
            if self.budget_selector is not None:
                target_ratio = self.budget_selector.predict_ratio(question, context, similarities, fallback_ratio)
            else:
                target_ratio = fallback_ratio

        selected_idx = select_with_mmr(
            similarities=similarities,
            sentence_embeddings=sent_embs,
            sentences=sentences,
            target_ratio=target_ratio,
            lambda_relevance=lambda_relevance,
        )
        compressed = "".join(sentences[i] for i in selected_idx)

        return {
            "question": question,
            "target_ratio": float(target_ratio),
            "sentences": sentences,
            "similarities": similarities,
            "selected_indices": selected_idx,
            "compressed_context": compressed,
        }
