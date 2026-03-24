from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import torch

from context_aware_encoder_model.context_aware_sentence_encoder import (
    ContextAwareEncoderConfig,
    ContextAwareSentenceEncoder,
    build_marked_context_window,
    split_sentences,
)
from pipeline.task_aware_compression import (
    DynamicSpanCompressor,
    IntraSentenceCompressionConfig,
    compute_task_reward,
    normalize_scores,
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
    在预算约束下，同时考虑：
    - 与问题的相关性
    - 与已选句子的冗余度
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


def blend_sentence_scores(
    semantic_similarities: Sequence[float],
    attention_probe_scores: Sequence[float],
    task_rewards: Sequence[float],
    attention_probe_weight: float,
    task_reward_weight: float,
) -> List[float]:
    semantic_norm = normalize_scores(semantic_similarities)
    attention_norm = normalize_scores(attention_probe_scores) if attention_probe_scores else [0.0] * len(semantic_norm)
    task_norm = normalize_scores(task_rewards) if task_rewards else [0.0] * len(semantic_norm)

    effective_attention = attention_probe_weight if attention_probe_scores else 0.0
    effective_task = task_reward_weight if task_rewards else 0.0
    semantic_weight = max(0.05, 1.0 - effective_attention - effective_task)

    denom = semantic_weight + effective_attention + effective_task
    return [
        float(
            (
                semantic_weight * semantic_norm[i]
                + effective_attention * attention_norm[i]
                + effective_task * task_norm[i]
            )
            / denom
        )
        for i in range(len(semantic_norm))
    ]


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

    def predict_ratio(
        self,
        question: str,
        context: str,
        similarities: List[float],
        fallback_ratio: float = 0.4,
    ) -> float:
        if not self.enabled:
            return fallback_ratio

        feats = self.build_budget_features(
            question=question,
            context=context,
            similarities=similarities,
        )
        x = torch.tensor(self.features_to_vector(feats)).unsqueeze(0)

        with torch.no_grad():
            logits = self.model(x)
            pred_cls = int(torch.argmax(logits, dim=1).item())

        return float(self.class_to_ratio(pred_cls, self.ratio_buckets))


class ContextAwareCompressor:
    def __init__(
        self,
        encoder_dir: str,
        budget_model_dir: Optional[str] = None,
        device: Optional[str] = None,
        window_max_chars: int = 220,
        use_attention_probe: bool = True,
        attention_probe_weight: float = 0.25,
        task_reward_weight: float = 0.15,
        attention_probe_layers: int = 2,
        enable_second_stage: bool = True,
        second_stage_keep_ratio: float = 0.78,
        span_model_dir: Optional[str] = None,
    ):
        device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.window_max_chars = window_max_chars
        self.use_attention_probe = use_attention_probe
        self.attention_probe_weight = attention_probe_weight
        self.task_reward_weight = task_reward_weight
        self.attention_probe_layers = attention_probe_layers

        with (Path(encoder_dir) / "encoder_config.json").open("r", encoding="utf-8") as f:
            cfg_dict = json.load(f)

        cfg_dict["device"] = device
        cfg_dict["model_name"] = encoder_dir
        allowed_keys = {
            "model_name",
            "max_length",
            "temperature",
            "device",
            "marker_start",
            "marker_end",
        }
        cfg_dict = {k: v for k, v in cfg_dict.items() if k in allowed_keys}

        cfg = ContextAwareEncoderConfig(**cfg_dict)
        self.encoder = ContextAwareSentenceEncoder(cfg)
        self.encoder.start_id = self.encoder.tokenizer.convert_tokens_to_ids(cfg.marker_start)
        self.encoder.end_id = self.encoder.tokenizer.convert_tokens_to_ids(cfg.marker_end)
        self.encoder.to(self.encoder.device)
        self.encoder.eval()

        self.budget_selector = BudgetPredictorAdapter(budget_model_dir) if budget_model_dir else None
        self.span_compressor = None
        if enable_second_stage:
            span_config = IntraSentenceCompressionConfig(
                target_keep_ratio=second_stage_keep_ratio,
                probe_layers=attention_probe_layers,
            )
            self.span_compressor = DynamicSpanCompressor(self.encoder, span_config, span_model_dir=span_model_dir)

    def score_context(self, question: str, context: str) -> Tuple[List[str], Dict[str, List[float]], torch.Tensor]:
        """
        对整段上下文中的每个句子打分。
        这里使用“局部窗口版 marked context”，避免长文本时 marker 被截断。
        """
        sentences = split_sentences(context)

        marked_contexts = [
            build_marked_context_window(
                sentences=sentences,
                target_index=i,
                marker_start=self.encoder.config.marker_start,
                marker_end=self.encoder.config.marker_end,
                max_chars=self.window_max_chars,
            )
            for i in range(len(sentences))
        ]

        semantic_similarities, sent_embs = self.encoder.score_sentences(
            question=question,
            sentences=sentences,
            marked_contexts=marked_contexts,
        )
        attention_scores = (
            self.encoder.attention_probe_scores(
                question=question,
                marked_contexts=marked_contexts,
                probe_layers=self.attention_probe_layers,
            )
            if self.use_attention_probe
            else [0.0 for _ in sentences]
        )
        task_rewards = [compute_task_reward(question, sentence) for sentence in sentences]
        selection_scores = blend_sentence_scores(
            semantic_similarities=semantic_similarities,
            attention_probe_scores=attention_scores,
            task_rewards=task_rewards,
            attention_probe_weight=self.attention_probe_weight,
            task_reward_weight=self.task_reward_weight,
        )

        return sentences, {
            "semantic_similarities": semantic_similarities,
            "attention_probe_scores": attention_scores,
            "task_rewards": task_rewards,
            "selection_scores": selection_scores,
        }, sent_embs

    def compress(
        self,
        question: str,
        context: str,
        target_ratio: Optional[float] = None,
        lambda_relevance: float = 0.7,
        fallback_ratio: float = 0.4,
    ) -> Dict:
        """
        完整压缩流程：
        1. 句子打分
        2. 预测 target_ratio（如果没手动指定）
        3. 在预算内做 MMR 选择
        4. 对保留句做句内动态压缩
        5. 输出压缩结果
        """
        sentences, score_dict, sent_embs = self.score_context(question, context)
        semantic_similarities = score_dict["semantic_similarities"]
        selection_scores = score_dict["selection_scores"]

        if target_ratio is None:
            if self.budget_selector is not None:
                target_ratio = self.budget_selector.predict_ratio(
                    question=question,
                    context=context,
                    similarities=semantic_similarities,
                    fallback_ratio=fallback_ratio,
                )
            else:
                target_ratio = fallback_ratio

        selected_idx = select_with_mmr(
            similarities=selection_scores,
            sentence_embeddings=sent_embs,
            sentences=sentences,
            target_ratio=target_ratio,
            lambda_relevance=lambda_relevance,
        )

        selected_sentences = [sentences[i] for i in selected_idx]
        second_stage_stats = {"sentence_stats": [], "removed_span_count": 0}
        compressed_sentences = selected_sentences
        if self.span_compressor is not None and selected_sentences:
            compressed_sentences, second_stage_stats = self.span_compressor.compress_sentences(
                question=question,
                sentences=selected_sentences,
                sentence_scores=[selection_scores[i] for i in selected_idx],
            )

        compressed = "".join(compressed_sentences)

        return {
            "question": question,
            "target_ratio": float(target_ratio),
            "sentences": sentences,
            "similarities": semantic_similarities,
            "semantic_similarities": semantic_similarities,
            "attention_probe_scores": score_dict["attention_probe_scores"],
            "task_rewards": score_dict["task_rewards"],
            "selection_scores": selection_scores,
            "selected_indices": selected_idx,
            "selected_sentences": selected_sentences,
            "compressed_sentences": compressed_sentences,
            "second_stage_stats": second_stage_stats,
            "compressed_context": compressed,
        }
