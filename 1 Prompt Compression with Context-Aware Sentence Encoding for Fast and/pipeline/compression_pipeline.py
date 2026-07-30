from __future__ import annotations

import json
import math
import re
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
from pipeline.dac_adapter import DacCompressionConfig
from pipeline.task_aware_compression import (
    DynamicSpanCompressor,
    IntraSentenceCompressionConfig,
    compute_task_reward,
    normalize_scores,
)
from pipeline.linguistic_information import (
    LinguisticFeatureConfig,
    build_linguistic_sentence_features,
)
from pipeline.task_descriptor import build_task_descriptor, compute_task_descriptor_alignment
from pipeline.runtime_contract import (
    LEGACY_LEXICAL_IDS,
    EncoderContractError,
    RuntimeProvenance,
    checkpoint_fingerprint,
)
from target_ratio_model.formula_budget import predict_formula_ratio


CONTENT_STOPWORDS = {
    "a",
    "an",
    "and",
    "are",
    "as",
    "at",
    "be",
    "by",
    "can",
    "could",
    "do",
    "does",
    "did",
    "for",
    "from",
    "how",
    "in",
    "is",
    "it",
    "of",
    "on",
    "or",
    "that",
    "the",
    "to",
    "was",
    "were",
    "what",
    "when",
    "where",
    "which",
    "who",
    "why",
    "with",
}
TOKEN_RE = re.compile(r"[A-Za-z][A-Za-z0-9]*(?:-[A-Za-z0-9]+)*|\d+(?:\.\d+)?%?")


def estimate_token_count(text: str) -> int:
    tokens = TOKEN_RE.findall(text)
    return max(len(tokens), 1 if text.strip() else 0)


def join_sentences(sentences: Sequence[str]) -> str:
    return " ".join(sentence.strip() for sentence in sentences if sentence and sentence.strip()).strip()


def lexical_information_density(text: str) -> float:
    tokens = [tok.lower() for tok in TOKEN_RE.findall(text) if tok.lower() not in CONTENT_STOPWORDS]
    if not tokens:
        return 0.0

    counts: Dict[str, int] = {}
    for token in tokens:
        counts[token] = counts.get(token, 0) + 1

    probs = [count / len(tokens) for count in counts.values()]
    entropy = -sum(prob * math.log(prob + 1e-12) for prob in probs)
    entropy_norm = entropy / math.log(len(counts) + 1e-12) if len(counts) > 1 else 0.0
    length_factor = min(1.0, len(tokens) / 24.0)
    return float(entropy_norm * length_factor)


def select_with_budget_aware_beam(
    similarities: List[float],
    sentence_embeddings: torch.Tensor,
    sentences: Sequence[str],
    target_ratio: float,
    lambda_relevance: float = 0.7,
    beam_size: int = 64,
) -> List[int]:
    total_len = sum(estimate_token_count(sentence) for sentence in sentences)
    budget = max(1, int(total_len * target_ratio))

    sims = np.asarray(similarities, dtype=float)
    if len(sentences) == 0:
        return []
    if sims.size != len(sentences):
        raise ValueError(f"len(similarities)={sims.size} but len(sentences)={len(sentences)}")

    lengths = [estimate_token_count(sentence) for sentence in sentences]
    embs = sentence_embeddings.detach().cpu().numpy()
    if embs.ndim != 2 or embs.shape[0] != len(sentences):
        pairwise = np.zeros((len(sentences), len(sentences)), dtype=float)
    else:
        pairwise = np.matmul(embs, embs.T)
        pairwise = np.clip(pairwise, 0.0, 1.0)

    ranked_indices = sorted(
        range(len(sentences)),
        key=lambda idx: (float(sims[idx]), float(sims[idx]) / math.sqrt(max(lengths[idx], 1))),
        reverse=True,
    )

    def evaluate(selected: Tuple[int, ...], used_len: int) -> float:
        if not selected:
            return 0.0

        rel_values = sims[list(selected)]
        rel_sum = float(rel_values.sum())
        rel_mean = float(rel_values.mean())
        budget_util = min(1.0, used_len / max(budget, 1))

        redundancy = 0.0
        pair_count = 0
        for pos, left in enumerate(selected):
            for right in selected[pos + 1 :]:
                redundancy += float(pairwise[left, right])
                pair_count += 1
        if pair_count:
            redundancy /= pair_count

        density_bonus = 0.08 * rel_sum / math.sqrt(max(used_len, 1))
        budget_bonus = 0.04 * math.sqrt(budget_util)
        return (
            lambda_relevance * rel_sum
            + (1.0 - lambda_relevance) * rel_mean
            - (1.0 - lambda_relevance) * redundancy
            + density_bonus
            + budget_bonus
        )

    states: List[Tuple[Tuple[int, ...], int, float]] = [(tuple(), 0, 0.0)]
    best_state: Tuple[Tuple[int, ...], int, float] = (tuple(), 0, -1e18)

    for idx in ranked_indices:
        next_states = list(states)
        sent_len = lengths[idx]
        for selected, used_len, _ in states:
            new_len = used_len + sent_len
            if new_len > budget:
                continue

            new_selected = tuple(sorted(selected + (idx,)))
            score = evaluate(new_selected, new_len)
            next_states.append((new_selected, new_len, score))
            if score > best_state[2]:
                best_state = (new_selected, new_len, score)

        dedup: Dict[Tuple[int, ...], Tuple[Tuple[int, ...], int, float]] = {}
        for selected, used_len, score in next_states:
            prev = dedup.get(selected)
            if prev is None or score > prev[2]:
                dedup[selected] = (selected, used_len, score)
        states = sorted(dedup.values(), key=lambda item: item[2], reverse=True)[:beam_size]

    if best_state[0]:
        return list(best_state[0])

    return [int(np.argmax(sims))]


def select_with_mmr(
    similarities: List[float],
    sentence_embeddings: torch.Tensor,
    sentences: Sequence[str],
    target_ratio: float,
    lambda_relevance: float = 0.7,
) -> List[int]:
    return select_with_budget_aware_beam(
        similarities=similarities,
        sentence_embeddings=sentence_embeddings,
        sentences=sentences,
        target_ratio=target_ratio,
        lambda_relevance=lambda_relevance,
    )

def blend_sentence_scores(
    semantic_similarities: Sequence[float],
    attention_probe_scores: Sequence[float],
    task_rewards: Sequence[float],
    task_descriptor_scores: Sequence[float],
    dynamic_attention_scores: Sequence[float],
    information_density_scores: Sequence[float],
    linguistic_scores: Sequence[float],
    attention_probe_weight: float,
    task_reward_weight: float,
    task_descriptor_weight: float,
    dynamic_attention_weight: float,
    information_density_weight: float,
    linguistic_feature_weight: float,
) -> List[float]:
    semantic_norm = normalize_scores(semantic_similarities)
    n = len(semantic_norm)

    components: List[Tuple[List[float], float]] = []
    active_weight = 0.0

    def add_component(values: Sequence[float], weight: float) -> None:
        nonlocal active_weight
        if weight <= 0.0 or len(values) != n:
            return
        components.append((normalize_scores(values), float(weight)))
        active_weight += float(weight)

    add_component(attention_probe_scores, attention_probe_weight)
    add_component(task_rewards, task_reward_weight)
    add_component(task_descriptor_scores, task_descriptor_weight)
    add_component(dynamic_attention_scores, dynamic_attention_weight)
    add_component(information_density_scores, information_density_weight)
    add_component(linguistic_scores, linguistic_feature_weight)

    semantic_weight = max(0.05, 1.0 - active_weight)
    denominator = semantic_weight + sum(weight for _, weight in components)

    blended = []
    for idx in range(n):
        score = semantic_weight * semantic_norm[idx]
        for values, weight in components:
            score += weight * values[idx]
        blended.append(float(score / max(denominator, 1e-9)))
    return blended



class SimpleTokenizer:
    is_fast = False
    mask_token_id = None
    all_special_ids: List[int] = []

    def tokenize(self, text: str) -> List[str]:
        return TOKEN_RE.findall(text)

    def convert_tokens_to_ids(self, token: str) -> int:
        return abs(hash(token)) % 100000


class LightweightSentenceEncoder:
    def __init__(self, marker_start: str = "<sent_start>", marker_end: str = "<sent_end>", max_length: int = 1024):
        self.tokenizer = SimpleTokenizer()
        self.device = torch.device("cpu")
        self.encoder = None
        self.start_id = self.tokenizer.convert_tokens_to_ids(marker_start)
        self.end_id = self.tokenizer.convert_tokens_to_ids(marker_end)
        self.config = type(
            "LightweightEncoderConfig",
            (),
            {
                "model_name": "lightweight_lexical_fallback",
                "marker_start": marker_start,
                "marker_end": marker_end,
                "max_length": max_length,
                "cache_dir": "",
                "trust_remote_code": False,
            },
        )()

    def to(self, device):
        self.device = torch.device("cpu")
        return self

    def eval(self):
        return self

    def format_query(self, question: str) -> str:
        return question

    def _vectorize(self, text: str, dim: int = 256) -> torch.Tensor:
        vector = torch.zeros(dim, dtype=torch.float32)
        for token in TOKEN_RE.findall(text.lower()):
            if token in CONTENT_STOPWORDS:
                continue
            vector[hash(token) % dim] += 1.0
        norm = torch.norm(vector, p=2)
        return vector / norm if float(norm.item()) > 0.0 else vector

    def _lexical_score(self, question: str, sentence: str) -> float:
        q_terms = {token for token in TOKEN_RE.findall(question.lower()) if token not in CONTENT_STOPWORDS}
        s_terms = {token for token in TOKEN_RE.findall(sentence.lower()) if token not in CONTENT_STOPWORDS}
        if not q_terms or not s_terms:
            return 0.0
        overlap = len(q_terms & s_terms) / max(len(q_terms), 1)
        density = lexical_information_density(sentence)
        numeric_bonus = 0.08 if re.search(r"\d", question) and re.search(r"\d", sentence) else 0.0
        return float(min(1.0, 0.72 * overlap + 0.20 * density + numeric_bonus))

    def score_sentences(self, question: str, sentences: Sequence[str], marked_contexts: Sequence[str]) -> Tuple[List[float], torch.Tensor]:
        scores = [self._lexical_score(question, sentence) for sentence in sentences]
        embeddings = torch.stack([self._vectorize(sentence) for sentence in sentences], dim=0) if sentences else torch.empty((0, 256))
        return scores, embeddings

    def attention_probe_scores(self, question: str, marked_contexts: Sequence[str], probe_layers: int = 2) -> List[float]:
        scores = []
        for marked_context in marked_contexts:
            start = marked_context.find(self.config.marker_start)
            end = marked_context.find(self.config.marker_end)
            if start == -1 or end == -1 or end <= start:
                scores.append(0.0)
                continue
            sentence = marked_context[start + len(self.config.marker_start) : end]
            scores.append(self._lexical_score(question, sentence))
        return scores

class BudgetPredictorAdapter:
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
        budget_formula_name: Optional[str] = None,
        device: Optional[str] = None,
        encoder_cache_dir: Optional[str] = None,
        window_max_chars: int = 900,
        use_attention_probe: bool = True,
        attention_probe_weight: float = 0.18,
        task_reward_weight: float = 0.16,
        use_task_descriptor: bool = True,
        task_descriptor_weight: float = 0.14,
        use_sentence_dynamics: bool = True,
        dynamic_attention_weight: float = 0.12,
        information_density_weight: float = 0.10,
        enable_linguistic_features: bool = True,
        linguistic_feature_weight: float = 0.18,
        enable_discourse_centrality: bool = True,
        enable_coreference_preservation: bool = True,
        enable_predicate_argument_skeleton: bool = True,
        enable_linguistic_information_density: bool = True,
        enable_redundancy_marginal_gain: bool = True,
        enable_inter_sentence_dependency: bool = True,
        attention_probe_layers: int = 2,
        enable_second_stage: bool = True,
        second_stage_keep_ratio: float = 0.56,
        second_stage_min_keep_ratio: float = 0.36,
        second_stage_max_keep_ratio: float = 0.76,
        span_model_dir: Optional[str] = None,
        allow_heuristic_fallback: bool = False,
        enable_dac: bool = True,
        dac_salience_model: Optional[str] = None,
        dac_fusion: str = "additive",
        dac_alpha: float = 0.8,
        dac_require_attention: bool = False,
        dac_strict: bool = False,
    ):
        device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.window_max_chars = window_max_chars
        self.use_attention_probe = use_attention_probe
        self.attention_probe_weight = attention_probe_weight
        self.task_reward_weight = task_reward_weight
        self.use_task_descriptor = use_task_descriptor
        self.task_descriptor_weight = task_descriptor_weight
        self.use_sentence_dynamics = use_sentence_dynamics
        self.dynamic_attention_weight = dynamic_attention_weight
        self.information_density_weight = information_density_weight
        self.enable_linguistic_features = enable_linguistic_features
        self.linguistic_feature_weight = linguistic_feature_weight
        self.linguistic_config = LinguisticFeatureConfig(
            enable_discourse_centrality=enable_discourse_centrality,
            enable_coreference_preservation=enable_coreference_preservation,
            enable_predicate_argument_skeleton=enable_predicate_argument_skeleton,
            enable_information_density=enable_linguistic_information_density,
            enable_redundancy_marginal_gain=enable_redundancy_marginal_gain,
            enable_inter_sentence_dependency=enable_inter_sentence_dependency,
        )
        self.attention_probe_layers = attention_probe_layers
        self.budget_formula_name = budget_formula_name

        # Encoder loading contract (audit findings C1 / H3): the lexical backend is
        # a legitimate ablation baseline but it is NOT the trained context-aware
        # encoder, so selecting it -- explicitly or by degradation -- must always be
        # a deliberate, recorded decision rather than a silent default.
        if encoder_dir in LEGACY_LEXICAL_IDS:
            if not allow_heuristic_fallback:
                raise EncoderContractError(
                    f"The lexical fallback backend ({encoder_dir!r}) is not the trained "
                    "context-aware encoder and cannot be selected implicitly.\n"
                    "Pass allow_heuristic_fallback=True to run it knowingly as a "
                    "non-neural ablation baseline, or supply a real checkpoint path."
                )
            self.encoder_load_error = None
            self.encoder_runtime = "lightweight_lexical_fallback"
            self.encoder = LightweightSentenceEncoder()
        else:
            cfg_dict = self._build_encoder_config(
                encoder_source=encoder_dir,
                device=device,
                cache_dir=encoder_cache_dir,
            )
            cfg = ContextAwareEncoderConfig(**cfg_dict)
            self.encoder_load_error = None
            self.encoder_runtime = "transformers"
            try:
                self.encoder = ContextAwareSentenceEncoder(cfg)
                self.encoder.start_id = self.encoder.tokenizer.convert_tokens_to_ids(cfg.marker_start)
                self.encoder.end_id = self.encoder.tokenizer.convert_tokens_to_ids(cfg.marker_end)
                self.encoder.to(self.encoder.device)
                self.encoder.eval()
            except Exception as exc:
                if not allow_heuristic_fallback:
                    raise
                self.encoder_load_error = str(exc)
                self.encoder_runtime = "lightweight_lexical_fallback"
                self.encoder = LightweightSentenceEncoder(
                    marker_start=cfg.marker_start,
                    marker_end=cfg.marker_end,
                    max_length=cfg.max_length,
                )

            
        self.budget_selector = BudgetPredictorAdapter(budget_model_dir) if budget_model_dir else None
        self.span_compressor = None
        if enable_second_stage:
            span_config = IntraSentenceCompressionConfig(
                target_keep_ratio=second_stage_keep_ratio,
                min_keep_ratio=second_stage_min_keep_ratio,
                max_keep_ratio=second_stage_max_keep_ratio,
                probe_layers=attention_probe_layers,
            )
            dac_config = DacCompressionConfig(
                fusion=dac_fusion,
                alpha=dac_alpha,
                require_attention=dac_require_attention,
            )
            if dac_salience_model:
                dac_config.salience_model_name = dac_salience_model
            self.span_compressor = DynamicSpanCompressor(
                self.encoder,
                span_config,
                span_model_dir=span_model_dir,
                dac_config=dac_config,
                dac_strict=dac_strict,
                enable_dac=enable_dac,
            )

        self.encoder_requested = str(encoder_dir)
        self.span_model_dir = span_model_dir
        self.budget_model_dir = budget_model_dir

    def provenance(self) -> RuntimeProvenance:
        """Record which backend actually ran, for embedding in results files.

        Any results file that omits this cannot be attributed to a system, which
        is how lexical-fallback numbers were previously mistaken for neural ones.
        """
        is_lexical = self.encoder_runtime == "lightweight_lexical_fallback"
        return RuntimeProvenance(
            encoder_kind="lexical_fallback" if is_lexical else "context_aware_encoder",
            encoder_requested=self.encoder_requested,
            encoder_path=None if is_lexical else self.encoder_requested,
            encoder_runtime=self.encoder_runtime,
            lexical_fallback_used=is_lexical,
            fallback_reason=(
                "load failure -> degraded" if self.encoder_load_error else
                ("explicitly requested" if is_lexical else "")
            ),
            encoder_load_error=self.encoder_load_error,
            checkpoint_fingerprint=checkpoint_fingerprint(
                None if is_lexical else self.encoder_requested
            ),
            span_model_dir=self.span_model_dir,
            span_model_active=bool(
                getattr(getattr(self, "span_compressor", None), "learned_span_model", None)
            ),
            budget_model_dir=self.budget_model_dir,
        )

    @staticmethod
    def _build_encoder_config(encoder_source: str, device: str, cache_dir: Optional[str]) -> Dict[str, object]:
        source_path = Path(encoder_source)
        if source_path.exists() and (source_path / "encoder_config.json").exists():
            with (source_path / "encoder_config.json").open("r", encoding="utf-8") as f:
                cfg_dict = json.load(f)
            cfg_dict["model_name"] = str(source_path)
        else:
            cfg_dict = {"model_name": encoder_source}

        cfg_dict["device"] = device
        if cache_dir:
            cfg_dict["cache_dir"] = cache_dir

        allowed_keys = {
            "model_name",
            "max_length",
            "temperature",
            "device",
            "marker_start",
            "marker_end",
            "cache_dir",
            "trust_remote_code",
            "pooling_strategy",
            "query_instruction",
        }
        return {key: value for key, value in cfg_dict.items() if key in allowed_keys}

    def compute_sentence_dynamics(
        self,
        question: str,
        sentences: Sequence[str],
        marked_contexts: Sequence[str],
    ) -> Tuple[List[float], List[float]]:
        lexical_scores = [lexical_information_density(sentence) for sentence in sentences]
        attention_scores = [0.0 for _ in sentences]
        attention_focus_scores = [0.0 for _ in sentences]

        if not self.use_sentence_dynamics or not marked_contexts:
            return attention_scores, lexical_scores

        try:
            encoded_question = self.encoder.format_query(question) if hasattr(self.encoder, "format_query") else question
            batch = self.encoder.tokenizer(
                [encoded_question] * len(marked_contexts),
                list(marked_contexts),
                padding=True,
                truncation=True,
                max_length=self.encoder.config.max_length,
                return_tensors="pt",
            )
            encodings = getattr(batch, "encodings", None)
            batch = batch.to(self.encoder.device)
            with torch.no_grad():
                outputs = self.encoder.encoder(**batch, output_attentions=True)
        except Exception:
            return attention_scores, lexical_scores

        attentions = getattr(outputs, "attentions", None)
        if not attentions:
            return attention_scores, lexical_scores

        last_layers = attentions[-self.attention_probe_layers :]
        mean_attention = torch.stack([layer.mean(dim=1) for layer in last_layers], dim=0).mean(dim=0)
        special_ids = set(getattr(self.encoder.tokenizer, "all_special_ids", []))

        input_ids_batch = batch["input_ids"].detach().cpu().tolist()
        attention_mask_batch = batch["attention_mask"].detach().cpu().tolist()
        token_type_batch = batch.get("token_type_ids")
        token_type_values = token_type_batch.detach().cpu().tolist() if token_type_batch is not None else None

        for batch_idx, input_ids in enumerate(input_ids_batch):
            try:
                start, end = self.encoder._find_marker_span(batch["input_ids"][batch_idx])
            except ValueError:
                continue

            mask_values = attention_mask_batch[batch_idx]
            valid_indices = [
                idx
                for idx, input_id in enumerate(input_ids)
                if mask_values[idx] == 1 and input_id not in special_ids
            ]
            span_indices = [
                idx
                for idx in range(start, end)
                if idx < len(input_ids) and mask_values[idx] == 1 and input_ids[idx] not in special_ids
            ]
            if not valid_indices or not span_indices:
                continue

            sequence_ids = None
            if encodings is not None and batch_idx < len(encodings):
                sequence_ids = encodings[batch_idx].sequence_ids

            if sequence_ids:
                question_indices = [
                    idx
                    for idx in valid_indices
                    if idx < len(sequence_ids) and sequence_ids[idx] == 0
                ]
            elif token_type_values is not None:
                question_indices = [
                    idx for idx in valid_indices if token_type_values[batch_idx][idx] == 0 and idx < start
                ]
            else:
                question_indices = [idx for idx in valid_indices if idx < start]

            if not question_indices:
                continue

            attn = mean_attention[batch_idx]
            q_to_span = float(attn[question_indices][:, span_indices].mean().item())
            global_to_span = float(attn[valid_indices][:, span_indices].mean().item())
            span_self = float(attn[span_indices][:, span_indices].mean().item())
            attention_scores[batch_idx] = 0.65 * q_to_span + 0.20 * global_to_span + 0.15 * span_self

            incoming = attn[question_indices][:, span_indices].mean(dim=0).clamp_min(0)
            total = incoming.sum()
            if float(total.item()) > 0.0 and incoming.numel() > 1:
                probs = incoming / total
                entropy = -torch.sum(probs * torch.log(probs + 1e-12))
                entropy_norm = float(entropy.item() / math.log(incoming.numel()))
                attention_focus_scores[batch_idx] = max(0.0, 1.0 - entropy_norm)

        information_scores = [
            float(0.55 * lexical_scores[idx] + 0.45 * attention_focus_scores[idx])
            for idx in range(len(sentences))
        ]
        return attention_scores, information_scores

    def score_context(self, question: str, context: str) -> Tuple[List[str], Dict[str, object], torch.Tensor]:
        sentences = split_sentences(context)
        descriptor = build_task_descriptor(question) if self.use_task_descriptor else None

        marked_contexts = [
            build_marked_context_window(
                sentences=sentences,
                target_index=idx,
                marker_start=self.encoder.config.marker_start,
                marker_end=self.encoder.config.marker_end,
                max_chars=self.window_max_chars,
            )
            for idx in range(len(sentences))
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
        dynamic_attention_scores, information_density_scores = self.compute_sentence_dynamics(
            question=question,
            sentences=sentences,
            marked_contexts=marked_contexts,
        )
        task_rewards = [compute_task_reward(question, sentence) for sentence in sentences]
        task_descriptor_scores = (
            [compute_task_descriptor_alignment(descriptor, sentence) for sentence in sentences]
            if descriptor is not None
            else [0.0 for _ in sentences]
        )
        linguistic_features = (
            build_linguistic_sentence_features(
                question=question,
                sentences=sentences,
                descriptor=descriptor,
                config=self.linguistic_config,
            )
            if self.enable_linguistic_features
            else []
        )
        linguistic_scores = [feature.final_score for feature in linguistic_features] if linguistic_features else [0.0 for _ in sentences]
        selection_scores = blend_sentence_scores(
            semantic_similarities=semantic_similarities,
            attention_probe_scores=attention_scores,
            task_rewards=task_rewards,
            task_descriptor_scores=task_descriptor_scores,
            dynamic_attention_scores=dynamic_attention_scores,
            information_density_scores=information_density_scores,
            linguistic_scores=linguistic_scores,
            attention_probe_weight=self.attention_probe_weight,
            task_reward_weight=self.task_reward_weight,
            task_descriptor_weight=self.task_descriptor_weight,
            dynamic_attention_weight=self.dynamic_attention_weight,
            information_density_weight=self.information_density_weight,
            linguistic_feature_weight=self.linguistic_feature_weight,
        )

        return sentences, {
            "semantic_similarities": semantic_similarities,
            "attention_probe_scores": attention_scores,
            "task_rewards": task_rewards,
            "task_descriptor_scores": task_descriptor_scores,
            "dynamic_attention_scores": dynamic_attention_scores,
            "information_density_scores": information_density_scores,
            "linguistic_scores": linguistic_scores,
            "linguistic_features": [feature.to_dict() for feature in linguistic_features],
            "task_descriptor": descriptor.to_dict() if descriptor is not None else None,
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
        sentences, score_dict, sent_embs = self.score_context(question, context)
        semantic_similarities = score_dict["semantic_similarities"]
        selection_scores = score_dict["selection_scores"]

        if target_ratio is None:
            budget_formula = None
            if self.budget_formula_name:
                budget_formula = predict_formula_ratio(
                    formula_name=self.budget_formula_name,
                    question=question,
                    context=context,
                    similarities=semantic_similarities,
                    linguistic_features=score_dict.get("linguistic_features"),
                )
                target_ratio = budget_formula.ratio
            elif self.budget_selector is not None:
                target_ratio = self.budget_selector.predict_ratio(
                    question=question,
                    context=context,
                    similarities=semantic_similarities,
                    fallback_ratio=fallback_ratio,
                )
            else:
                target_ratio = fallback_ratio
                budget_formula = None
        else:
            budget_formula = None

        selected_idx = select_with_budget_aware_beam(
            similarities=selection_scores,
            sentence_embeddings=sent_embs,
            sentences=sentences,
            target_ratio=target_ratio,
            lambda_relevance=lambda_relevance,
        )

        selected_sentences = [sentences[idx] for idx in selected_idx]
        selected_scores = [selection_scores[idx] for idx in selected_idx]
        second_stage_stats = {"sentence_stats": [], "removed_span_count": 0}
        compressed_sentences = selected_sentences
        if self.span_compressor is not None and selected_sentences:
            compressed_sentences, second_stage_stats = self.span_compressor.compress_sentences(
                question=question,
                sentences=selected_sentences,
                sentence_scores=selected_scores,
            )

        stage1_context = join_sentences(selected_sentences)
        compressed_context = join_sentences(compressed_sentences)

        return {
            "question": question,
            "target_ratio": float(target_ratio),
            "budget_formula": budget_formula.to_dict() if budget_formula is not None else None,
            "sentences": sentences,
            "similarities": semantic_similarities,
            "semantic_similarities": semantic_similarities,
            "attention_probe_scores": score_dict["attention_probe_scores"],
            "dynamic_attention_scores": score_dict["dynamic_attention_scores"],
            "information_density_scores": score_dict["information_density_scores"],
            "linguistic_scores": score_dict["linguistic_scores"],
            "linguistic_features": score_dict["linguistic_features"],
            "task_rewards": score_dict["task_rewards"],
            "task_descriptor": score_dict["task_descriptor"],
            "task_descriptor_scores": score_dict["task_descriptor_scores"],
            "selection_scores": selection_scores,
            "selected_indices": selected_idx,
            "selected_sentences": selected_sentences,
            "stage1_context": stage1_context,
            "compressed_sentences": compressed_sentences,
            "stage2_sentences": compressed_sentences,
            "second_stage_stats": second_stage_stats,
            "second_stage_context": compressed_context,
            "compressed_context": compressed_context,
        }






