"""DAC-style token salience: masked-LM information content fused with question attention.

Design notes (see EVAL_VALIDITY_AUDIT.md findings C6, D1-D8):

D1  The salience model is now DECOUPLED from the Stage-1 sentence encoder. The
    Stage-1 default is Qwen3-Embedding, a causal embedding model that has (a) no
    mask token and (b) no entry in transformers' MaskedLM mapping. Pointing this
    adapter at it therefore disabled DAC unconditionally, so every "DAC-guided"
    run silently degraded to plain span pruning. `salience_model_name` now
    defaults to a real encoder-MLM with trained head weights.

D3  Loss tokens (salience tokenizer, text alone) and attention tokens (encoder
    tokenizer, question+text pair) come from DIFFERENT tokenizers and are no
    longer assumed to be positionally aligned. Attention is projected onto
    CHARACTER offsets of `text` and then re-aggregated onto salience tokens, so
    alignment is correct by construction. When the attention term is genuinely
    unavailable the fusion RENORMALISES onto the loss term and records the fact,
    instead of substituting zeros and pretending attention contributed.

D4  `select_keep_indices` used to append "rescued" tokens on top of a full topk
    budget, so the achieved keep-ratio exceeded the requested one. Rescues are
    now compensated, keeping |keep| == keep_count exactly.

D6  `compress` reused a protected_char_mask indexed against the ORIGINAL text on
    every iteration, silently protecting the wrong characters from step 2 on.
    The mask is now carried forward through each reconstruction.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence, Tuple

import torch
import torch.nn.functional as F
from transformers import AutoModelForCausalLM, AutoModelForMaskedLM, AutoTokenizer


# Official DAC (ACL 2025, arXiv 2507.11942) derives token information content
# from a small CAUSAL LM in a SINGLE forward pass: shift the logits, take
# per-token cross-entropy. That is O(1) forwards for the whole sequence.
#
# The "mlm" backend below masks each position in turn instead, which needs O(n)
# forwards and measures bidirectional cloze difficulty rather than left-to-right
# surprisal. It is retained only as an ablation arm; "causal" is the default
# because it is both faithful to the cited method and orders of magnitude
# cheaper.
DEFAULT_SALIENCE_MODEL_CAUSAL = "Qwen/Qwen2-0.5B-Instruct"
DEFAULT_SALIENCE_MODEL_MLM = "roberta-base"
SALIENCE_BACKENDS = ("causal", "mlm")

# Parameter-name fragments belonging to a masked-LM prediction head. If any of
# these are reported missing when loading, the head was randomly initialised.
_LM_HEAD_KEY_HINTS = ("cls.predictions", "lm_head", "predictions.decoder", "mlm_head", "vocab_transform")

# Below this fraction of `text` covered by the attention pass we treat the
# attention term as unavailable rather than as mostly-zeros.
_MIN_ATTENTION_COVERAGE = 0.60


def _lm_head_keys_missing(missing_keys: Sequence[str]) -> List[str]:
    return [
        key for key in (missing_keys or [])
        if any(hint in key for hint in _LM_HEAD_KEY_HINTS)
    ]


@dataclass
class DacCompressionConfig:
    fusion: str = "additive"
    alpha: float = 0.8
    max_dyn_steps: int = 6
    min_tokens: int = 6
    preserve_punct: bool = True
    avoid_consecutive: bool = True
    # "causal" = single-pass shifted cross-entropy (faithful to DAC, O(1)
    # forwards). "mlm" = per-token masking (O(n) forwards), ablation only.
    salience_backend: str = "causal"
    # Empty means "pick the default for the chosen backend".
    salience_model_name: str = ""
    # Raise instead of renormalising when the question-attention term is missing.
    require_attention: bool = False

    def resolved_salience_model(self) -> str:
        if self.salience_model_name:
            return self.salience_model_name
        return (
            DEFAULT_SALIENCE_MODEL_CAUSAL
            if self.salience_backend == "causal"
            else DEFAULT_SALIENCE_MODEL_MLM
        )


class DacTokenAdapter:
    """Token salience = fuse(masked-LM information content, question attention).

    The masked-LM term needs one forward pass per masked position, so it is
    O(n) forwards for an n-token input. It is applied per SENTENCE (short n),
    not per context, which keeps this tractable -- but it is still by far the
    most expensive signal in the pipeline and must be reported as such in any
    latency table.
    """

    def __init__(
        self,
        sentence_encoder,
        config: DacCompressionConfig | None = None,
        dac_model_name: str | None = None,
        strict: bool = False,
        verbose: bool = True,
    ):
        self.sentence_encoder = sentence_encoder
        self.config = config or DacCompressionConfig()

        # Encoder side: used ONLY for question->token attention.
        self.encoder = sentence_encoder.encoder
        self.encoder_tokenizer = sentence_encoder.tokenizer
        self.device = sentence_encoder.device
        self.encoder_max_length = sentence_encoder.config.max_length
        self.special_token_ids = set(getattr(self.encoder_tokenizer, "all_special_ids", []))

        # Salience side: an independent LM, causal by default.
        self.backend = self.config.salience_backend
        if self.backend not in SALIENCE_BACKENDS:
            raise ValueError(
                f"unknown salience_backend {self.backend!r}; expected one of {SALIENCE_BACKENDS}"
            )
        self.salience_model_name = dac_model_name or self.config.resolved_salience_model()
        self.salience_tokenizer = None
        self.entropy_model = None
        self.max_length = self.encoder_max_length

        self.available = False
        self.unavailable_reason: Optional[str] = None
        # Populated per call so provenance reflects what actually happened.
        self.attention_available: Optional[bool] = None
        self.attention_unavailable_reason: Optional[str] = None

        self._load_salience_model(strict=strict, verbose=verbose)

    # ------------------------------------------------------------------
    # Loading
    # ------------------------------------------------------------------

    def _load_salience_model(self, strict: bool, verbose: bool) -> None:
        try:
            self.salience_tokenizer = AutoTokenizer.from_pretrained(
                self.salience_model_name,
                cache_dir=getattr(self.sentence_encoder.config, "cache_dir", "") or None,
                use_fast=True,
            )
        except Exception as exc:
            self.unavailable_reason = (
                f"could not load salience tokenizer {self.salience_model_name!r}: "
                f"{type(exc).__name__}: {exc}"
            )
            self._report(strict, verbose)
            return

        if self.backend == "mlm" and self.salience_tokenizer.mask_token_id is None:
            self.unavailable_reason = (
                f"salience tokenizer {self.salience_model_name!r} has no mask token, so "
                "per-token masked-LM scoring is impossible. Use an encoder-MLM "
                "(e.g. roberta-base, bert-base-uncased), or switch to the causal backend."
            )
            self._report(strict, verbose)
            return

        if not getattr(self.salience_tokenizer, "is_fast", False):
            self.unavailable_reason = (
                f"salience tokenizer {self.salience_model_name!r} is not a fast tokenizer; "
                "character offsets are required for span alignment."
            )
            self._report(strict, verbose)
            return

        kwargs = dict(
            cache_dir=getattr(self.sentence_encoder.config, "cache_dir", "") or None,
            output_loading_info=True,
        )
        loader = (
            AutoModelForCausalLM if self.backend == "causal" else AutoModelForMaskedLM
        )
        loading_info = None
        try:
            try:
                self.entropy_model, loading_info = loader.from_pretrained(
                    self.salience_model_name, attn_implementation="eager", **kwargs
                )
            except TypeError:
                self.entropy_model, loading_info = loader.from_pretrained(
                    self.salience_model_name, **kwargs
                )
        except Exception as exc:
            self.entropy_model = None
            self.unavailable_reason = (
                f"could not load a {self.backend} LM from {self.salience_model_name!r}: "
                f"{type(exc).__name__}: {exc}"
            )
            self._report(strict, verbose)
            return

        # The load succeeded, but did the LM head actually come from the checkpoint?
        # Only meaningful when the head is a separate parameter set. Causal models
        # frequently TIE lm_head to the input embeddings, in which case the head
        # keys are legitimately absent from the checkpoint and flagging them would
        # disable a perfectly good model.
        tied = bool(getattr(getattr(self.entropy_model, "config", None), "tie_word_embeddings", False))
        if not tied:
            missing = _lm_head_keys_missing((loading_info or {}).get("missing_keys", []))
            if missing:
                self.entropy_model = None
                self.unavailable_reason = (
                    f"LM head was randomly initialised when loading "
                    f"{self.salience_model_name!r} (missing: {missing[:4]}). DAC salience "
                    f"would be noise, so it is disabled. Point --dac_salience_model at a "
                    f"checkpoint that publishes trained head weights."
                )
                self._report(strict, verbose)
                return

        # roberta-base caps at 512 positions; causal models advertise far more but
        # we never need the whole window for a single sentence.
        default_cap = 2048 if self.backend == "causal" else 512
        model_cap = int(getattr(self.salience_tokenizer, "model_max_length", default_cap) or default_cap)
        if model_cap > 100_000:  # sentinel used by some tokenizers
            model_cap = default_cap
        self.max_length = max(8, min(self.encoder_max_length, model_cap))

        self.entropy_model = self.entropy_model.to(self.device)
        self.entropy_model.eval()
        self.available = True

    def _report(self, strict: bool, verbose: bool) -> None:
        message = f"DAC token salience DISABLED: {self.unavailable_reason}"
        if strict:
            raise RuntimeError(message)
        if verbose:
            print(f"[dac_adapter] WARNING: {message}", flush=True)

    def provenance(self) -> Dict[str, object]:
        """What actually ran. Belongs in the run manifest."""
        return {
            "dac_available": bool(self.available),
            "dac_salience_backend": self.backend,
            "dac_salience_model": self.salience_model_name,
            "dac_unavailable_reason": self.unavailable_reason,
            "dac_fusion": self.config.fusion,
            "dac_alpha": self.config.alpha,
            "dac_attention_available": self.attention_available,
            "dac_attention_unavailable_reason": self.attention_unavailable_reason,
            "dac_max_length": self.max_length,
        }

    # ------------------------------------------------------------------
    # Scoring primitives
    # ------------------------------------------------------------------

    def normalize(self, tensor: torch.Tensor) -> torch.Tensor:
        if tensor.dim() == 0:
            tensor = tensor.unsqueeze(0)
        min_val = tensor.min()
        max_val = tensor.max()
        if float(max_val - min_val) > 1e-8:
            return (tensor - min_val) / (max_val - min_val)
        return torch.zeros_like(tensor)

    def _token_count(self, text: str) -> int:
        tokenizer = self.salience_tokenizer or self.encoder_tokenizer
        try:
            return len(tokenizer.tokenize(text))
        except Exception:
            # Whitespace words, NOT len(text): characters would inflate the count
            # by ~5x and saturate dyn_steps.
            return len(text.split())

    def compute_token_losses(
        self, text: str
    ) -> Tuple[Optional[torch.Tensor], Optional[List[List[int]]]]:
        """Per-token information content, with character offsets into `text`."""
        if self.backend == "causal":
            return self._causal_token_losses(text)
        return self._mlm_token_losses(text)

    def _causal_token_losses(
        self, text: str
    ) -> Tuple[Optional[torch.Tensor], Optional[List[List[int]]]]:
        """Shifted cross-entropy from ONE forward pass (official DAC's get_ppl).

        logits[t] predicts token t+1, so the raw loss vector has length n-1 and
        describes tokens 1..n-1. Token 0 is unconditioned and has no loss; it is
        assigned the mean of the rest so the returned tensor stays 1:1 with
        `offsets` and callers need no special case.
        """
        if not self.available or not text.strip():
            return None, None

        encoding = self.salience_tokenizer(
            text,
            add_special_tokens=False,
            truncation=True,
            max_length=self.max_length,
            return_tensors="pt",
            return_offsets_mapping=True,
        )
        seq_len = encoding["input_ids"].size(1)
        if seq_len == 0:
            return None, None
        offsets = encoding["offset_mapping"][0].tolist()
        if seq_len < self.config.min_tokens or seq_len < 2:
            return None, offsets

        input_ids = encoding["input_ids"].to(self.device)
        attention_mask = encoding["attention_mask"].to(self.device)

        with torch.no_grad():
            outputs = self.entropy_model(input_ids=input_ids, attention_mask=attention_mask)
            logits = outputs.logits
            shift_logits = logits[..., :-1, :].contiguous()
            shift_labels = input_ids[..., 1:].contiguous()
            losses = F.cross_entropy(
                shift_logits.view(-1, shift_logits.size(-1)),
                shift_labels.view(-1),
                reduction="none",
            )

        mean_loss = losses.mean() if losses.numel() else torch.zeros(1, device=self.device)
        aligned = torch.cat([mean_loss.reshape(1), losses], dim=0)
        return aligned, offsets

    def _mlm_token_losses(
        self, text: str
    ) -> Tuple[Optional[torch.Tensor], Optional[List[List[int]]]]:
        """Masked-LM cross-entropy per token: O(n) forward passes. Ablation only."""
        if not self.available or not text.strip():
            return None, None

        encoding = self.salience_tokenizer(
            text,
            add_special_tokens=False,
            truncation=True,
            max_length=self.max_length,
            return_tensors="pt",
            return_offsets_mapping=True,
        )
        if encoding["input_ids"].size(1) == 0:
            return None, None

        input_ids = encoding["input_ids"].to(self.device)
        attention_mask = encoding["attention_mask"].to(self.device)
        offsets = encoding["offset_mapping"][0].tolist()
        seq_len = input_ids.size(1)
        if seq_len < self.config.min_tokens:
            return None, offsets

        mask_token_id = self.salience_tokenizer.mask_token_id
        target_ids = input_ids[0]
        losses = []
        chunk_size = 32
        position_ids = torch.arange(seq_len, device=self.device)

        with torch.no_grad():
            for start in range(0, seq_len, chunk_size):
                end = min(seq_len, start + chunk_size)
                batch_size = end - start
                masked_batch = input_ids.repeat(batch_size, 1)
                masked_batch[torch.arange(batch_size, device=self.device), position_ids[start:end]] = mask_token_id
                mask_batch = attention_mask.repeat(batch_size, 1)
                outputs = self.entropy_model(
                    input_ids=masked_batch,
                    attention_mask=mask_batch,
                )
                logits = outputs.logits[torch.arange(batch_size, device=self.device), position_ids[start:end]]
                batch_losses = F.cross_entropy(
                    logits,
                    target_ids[start:end],
                    reduction="none",
                )
                losses.append(batch_losses)

        return torch.cat(losses, dim=0), offsets

    def compute_char_attention(self, question: str, text: str) -> Optional[List[float]]:
        """Question-attention projected onto CHARACTER positions of `text`.

        Returning a per-character array makes the result independent of the
        encoder's tokenisation, so it can be re-aggregated onto salience tokens
        (or spans) without any positional-alignment assumption.
        """
        self.attention_available = False
        self.attention_unavailable_reason = None

        if not text.strip():
            self.attention_unavailable_reason = "empty text"
            return None
        if not getattr(self.encoder_tokenizer, "is_fast", False):
            self.attention_unavailable_reason = "encoder tokenizer is not fast (no offsets)"
            return None

        try:
            encoded_question = (
                self.sentence_encoder.format_query(question)
                if hasattr(self.sentence_encoder, "format_query")
                else question
            )
            batch = self.encoder_tokenizer(
                encoded_question,
                text,
                truncation="only_first",
                max_length=self.encoder_max_length,
                return_tensors="pt",
                return_offsets_mapping=True,
            )
        except Exception as exc:
            self.attention_unavailable_reason = f"tokenisation failed: {type(exc).__name__}: {exc}"
            return None

        if not getattr(batch, "encodings", None):
            self.attention_unavailable_reason = "tokenizer returned no encodings"
            return None

        encoding = batch.encodings[0]
        sequence_ids = encoding.sequence_ids
        offsets = batch["offset_mapping"][0].tolist()
        input_ids = batch["input_ids"][0].tolist()
        model_inputs = {
            key: value.to(self.device)
            for key, value in batch.items()
            if key != "offset_mapping"
        }

        with torch.no_grad():
            outputs = self.encoder(**model_inputs, output_attentions=True)

        attentions = getattr(outputs, "attentions", None)
        if not attentions:
            self.attention_unavailable_reason = (
                "encoder returned no attentions (needs attn_implementation='eager')"
            )
            return None

        attn = torch.stack([layer[0].mean(dim=0) for layer in attentions[-2:]], dim=0).mean(dim=0)
        question_indices = [
            idx
            for idx, (seq_id, offset, input_id) in enumerate(zip(sequence_ids, offsets, input_ids))
            if seq_id == 0 and offset[1] > offset[0] and input_id not in self.special_token_ids
        ]
        if not question_indices:
            self.attention_unavailable_reason = "no question tokens identified"
            return None

        char_total = [0.0] * len(text)
        char_count = [0] * len(text)
        covered = 0
        for idx, (seq_id, offset, input_id) in enumerate(zip(sequence_ids, offsets, input_ids)):
            if seq_id != 1 or offset[1] <= offset[0] or input_id in self.special_token_ids:
                continue
            q_to_token = float(attn[question_indices][:, [idx]].mean().item())
            ctx_to_token = float(attn[:, [idx]].mean().item())
            score = 0.75 * q_to_token + 0.25 * ctx_to_token
            start, end = int(offset[0]), min(int(offset[1]), len(text))
            for pos in range(start, end):
                char_total[pos] += score
                char_count[pos] += 1
                covered += 1

        if covered == 0:
            self.attention_unavailable_reason = "no context tokens survived filtering"
            return None

        non_space = max(sum(1 for ch in text if not ch.isspace()), 1)
        coverage = min(1.0, covered / non_space)
        if coverage < _MIN_ATTENTION_COVERAGE:
            # The pair encoding truncated most of the context. Mostly-zero
            # attention is worse than no attention: renormalise onto loss.
            self.attention_unavailable_reason = (
                f"attention covered only {coverage:.2f} of the text (truncated); "
                f"below the {_MIN_ATTENTION_COVERAGE:.2f} threshold"
            )
            return None

        self.attention_available = True
        return [
            (char_total[pos] / char_count[pos]) if char_count[pos] else 0.0
            for pos in range(len(text))
        ]

    @staticmethod
    def _aggregate_chars(
        char_values: Sequence[float], start: int, end: int
    ) -> Optional[float]:
        lo, hi = max(0, int(start)), min(len(char_values), int(end))
        if hi <= lo:
            return None
        window = [char_values[pos] for pos in range(lo, hi)]
        return sum(window) / len(window) if window else None

    def _attention_for_offsets(
        self, char_attention: Sequence[float], offsets: Sequence[Sequence[int]]
    ) -> torch.Tensor:
        values = []
        for start, end in offsets:
            mean = self._aggregate_chars(char_attention, start, end)
            values.append(0.0 if mean is None else mean)
        return torch.tensor(values, dtype=torch.float32, device=self.device)

    def _fuse(
        self,
        losses: torch.Tensor,
        attention: Optional[torch.Tensor],
    ) -> torch.Tensor:
        loss_norm = self.normalize(losses)
        if attention is None:
            if self.config.require_attention:
                raise RuntimeError(
                    "DAC question-attention unavailable "
                    f"({self.attention_unavailable_reason}) and require_attention=True."
                )
            # Renormalise onto the loss term rather than fusing against zeros.
            return loss_norm
        attn_norm = self.normalize(attention)
        if self.config.fusion == "multiplicative":
            return loss_norm * attn_norm
        return self.config.alpha * attn_norm + (1.0 - self.config.alpha) * loss_norm

    def compute_fused_token_scores(
        self, question: str, text: str
    ) -> Tuple[Optional[torch.Tensor], Optional[List[List[int]]]]:
        losses, offsets = self.compute_token_losses(text)
        if losses is None or offsets is None:
            return None, None

        char_attention = self.compute_char_attention(question, text)
        attention = (
            self._attention_for_offsets(char_attention, offsets)
            if char_attention is not None
            else None
        )
        return self._fuse(losses, attention), offsets

    def score_spans(self, question: str, text: str, spans) -> Optional[List[float]]:
        """Mean fused salience per span.

        `spans` must carry .start/.end offsets INTO `text`. Passing a different
        string than the one the spans were built from silently misattributes
        salience, so callers must hand over the original sentence.
        """
        fused, offsets = self.compute_fused_token_scores(question, text)
        if fused is None or offsets is None:
            return None

        scores = []
        for span in spans:
            token_indices = [
                idx
                for idx, (start, end) in enumerate(offsets)
                if end > start and end > span.start and start < span.end
            ]
            if not token_indices:
                scores.append(0.0)
                continue
            scores.append(float(fused[token_indices].mean().item()))
        return scores

    # ------------------------------------------------------------------
    # Iterative token-level compression
    # ------------------------------------------------------------------

    def preserve_punctuation_mask(self, text: str, offsets: Sequence[Sequence[int]]) -> torch.Tensor:
        punct_pattern = re.compile(r"^\s*[^\w\s]+\s*$")
        preserve = []
        for start, end in offsets:
            token_text = text[start:end] if end > start else ""
            preserve.append(bool(punct_pattern.match(token_text)))
        return torch.tensor(preserve, dtype=torch.bool, device=self.device)

    def protected_token_mask(
        self,
        offsets: Sequence[Sequence[int]],
        protected_char_mask: Optional[Sequence[bool]],
    ) -> torch.Tensor:
        if protected_char_mask is None:
            return torch.zeros(len(offsets), dtype=torch.bool, device=self.device)

        protected = []
        for start, end in offsets:
            keep = False
            for pos in range(start, end):
                if pos < len(protected_char_mask) and protected_char_mask[pos]:
                    keep = True
                    break
            protected.append(keep)
        return torch.tensor(protected, dtype=torch.bool, device=self.device)

    def select_keep_indices(
        self,
        score: torch.Tensor,
        compress_ratio: float,
        punct_mask: torch.Tensor,
        protect_mask: torch.Tensor,
    ) -> torch.Tensor:
        """Top-k keep selection that HONOURS the requested budget.

        Punctuation/protected tokens are force-kept. The `avoid_consecutive`
        rescue then trades tokens rather than appending them, so the achieved
        keep-count equals the requested keep-count (finding D4).
        """
        total_tokens = score.numel()
        keep_count = max(1, int(round(total_tokens * (1 - compress_ratio))))
        keep_count = min(keep_count, total_tokens)

        forced = protect_mask.clone()
        if self.config.preserve_punct:
            forced = forced | punct_mask

        working_score = score.clone().float()
        working_score = working_score + 1e5 * forced.float()

        _, topk = torch.topk(working_score.view(-1), k=keep_count, largest=True)
        keep_mask = torch.zeros(total_tokens, dtype=torch.bool, device=score.device)
        keep_mask[topk] = True

        if not self.config.avoid_consecutive:
            return torch.nonzero(keep_mask, as_tuple=False).view(-1).sort()[0]

        # Rescue the first token of each run of >=2 consecutive deletions, then
        # pay for each rescue by dropping the lowest-scoring non-forced keep.
        deleted = torch.nonzero(~keep_mask, as_tuple=False).view(-1)
        if deleted.numel() <= 1:
            return torch.nonzero(keep_mask, as_tuple=False).view(-1).sort()[0]

        rescues: List[int] = []
        run_len = 1
        for pos in range(1, deleted.numel()):
            if int(deleted[pos]) == int(deleted[pos - 1]) + 1:
                run_len += 1
                if run_len == 2:
                    rescues.append(int(deleted[pos - 1]))
            else:
                run_len = 1

        for rescue_idx in rescues:
            droppable = torch.nonzero(keep_mask & ~forced, as_tuple=False).view(-1)
            if droppable.numel() == 0:
                break
            victim = droppable[torch.argmin(score[droppable])]
            if float(score[victim]) >= float(score[rescue_idx]):
                continue  # trade would lower total salience; skip this rescue
            keep_mask[victim] = False
            keep_mask[rescue_idx] = True

        return torch.nonzero(keep_mask, as_tuple=False).view(-1).sort()[0]

    def reconstruct_text(
        self,
        text: str,
        offsets: Sequence[Sequence[int]],
        keep_indices: Sequence[int],
    ) -> Tuple[str, List[int]]:
        """Return the kept text AND the original char positions it came from.

        The position list lets callers carry a character-aligned mask (e.g. the
        protected-entity mask) forward across iterations (finding D6).
        """
        char_keep = [False] * len(text)
        for idx in keep_indices:
            start, end = offsets[int(idx)]
            for pos in range(start, end):
                if 0 <= pos < len(char_keep):
                    char_keep[pos] = True

        chars: List[str] = []
        positions: List[int] = []
        for pos, ch in enumerate(text):
            if char_keep[pos]:
                chars.append(ch)
                positions.append(pos)
                continue
            if ch.isspace() and (
                (pos > 0 and char_keep[pos - 1])
                or (pos + 1 < len(char_keep) and char_keep[pos + 1])
            ):
                chars.append(ch)
                positions.append(pos)

        joined = "".join(chars)
        stripped = joined.strip()
        # Keep `positions` aligned to `stripped` after strip().
        lead = len(joined) - len(joined.lstrip())
        trail = len(joined) - len(joined.rstrip())
        positions = positions[lead: len(positions) - trail] if trail else positions[lead:]
        return stripped, positions

    def compress(
        self,
        question: str,
        text: str,
        keep_ratio: float,
        protected_char_mask: Optional[Sequence[bool]] = None,
    ) -> Optional[dict]:
        if not self.available:
            return None

        current_text = text
        # Mask stays aligned to current_text across iterations (finding D6).
        current_protect: Optional[List[bool]] = (
            list(protected_char_mask[: len(text)]) + [False] * max(0, len(text) - len(protected_char_mask))
            if protected_char_mask is not None
            else None
        )

        compress_ratio = max(0.0, 1.0 - keep_ratio)
        seq_len = self._token_count(text)
        if seq_len < self.config.min_tokens:
            return None

        dyn_steps = min(max(1, seq_len // 12), self.config.max_dyn_steps)
        real_ratio = 1.0 - (1.0 - compress_ratio) ** (1.0 / dyn_steps)
        steps_run = 0

        for _ in range(dyn_steps):
            losses, offsets = self.compute_token_losses(current_text)
            if losses is None or offsets is None or losses.numel() < self.config.min_tokens:
                break

            char_attention = self.compute_char_attention(question, current_text)
            attention = (
                self._attention_for_offsets(char_attention, offsets)
                if char_attention is not None
                else None
            )
            fused_score = self._fuse(losses, attention)

            punct_mask = self.preserve_punctuation_mask(current_text, offsets)
            protect_mask = self.protected_token_mask(offsets, current_protect)
            keep_indices = self.select_keep_indices(fused_score, real_ratio, punct_mask, protect_mask)
            next_text, kept_positions = self.reconstruct_text(
                current_text, offsets, keep_indices.tolist()
            )
            if not next_text or next_text == current_text:
                break

            if current_protect is not None:
                current_protect = [
                    current_protect[pos] if pos < len(current_protect) else False
                    for pos in kept_positions
                ]
            current_text = next_text
            steps_run += 1

        if current_text == text:
            return None

        return {
            "compressed_text": current_text,
            "dyn_steps": dyn_steps,
            "steps_run": steps_run,
            "fusion": self.config.fusion,
            "alpha": self.config.alpha,
            "attention_available": self.attention_available,
        }
