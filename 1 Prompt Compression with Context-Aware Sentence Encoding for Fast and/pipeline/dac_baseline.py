"""Faithful re-implementation of DAC as a COMPARISON BASELINE.

DAC: A Dynamic Attention-aware Approach for Task-Agnostic Prompt Compression
(ACL 2025, arXiv 2507.11942, https://github.com/QQQ-yi/DAC)

This is deliberately separate from pipeline/dac_adapter.py. The adapter is our
question-aware salience FEATURE; this module is the published method, run as an
independent baseline so the results table has an honest DAC row.

Faithful to the reference implementation (compressor.py):
  * Token information content from a small CAUSAL LM in ONE forward pass:
    shift_logits -> per-token cross-entropy (their `get_ppl`).
  * Attention summed COLUMN-WISE over every layer and head, i.e. how much the
    rest of the sequence attends to each token. No question is involved --
    DAC is task-agnostic.
  * Additive fusion: alpha * attn_norm + (1 - alpha) * ppl_norm, alpha = 0.8.
  * Iterative compression over several dynamic steps, punctuation preserved,
    isolated single-token deletions avoided.

Two documented deviations, both forced by engineering limits rather than choice:
  * Long contexts are processed in windows (default 1024 tokens). Materialising
    attention for a full 4k context costs layers x heads x n^2 floats, which
    does not fit alongside the rest of the pipeline. Attention is therefore
    within-window. Recorded in provenance as `windowed`.
  * We follow their CODE for multiplicative fusion, torch.mul(ppl, attn), not
    their docstring's `attn * (1/ppl)`; the two disagree in the reference repo.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence, Tuple

import torch
import torch.nn.functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer

DEFAULT_DAC_MODEL = "Qwen/Qwen2-0.5B-Instruct"


@dataclass
class DacBaselineConfig:
    model_name: str = DEFAULT_DAC_MODEL
    alpha: float = 0.8
    fusion: str = "additive"
    max_dyn_steps: int = 6
    min_tokens: int = 6
    preserve_punct: bool = True
    avoid_consecutive: bool = True
    window_tokens: int = 1024


class DacBaselineCompressor:
    """Task-agnostic DAC compression of a whole context."""

    def __init__(self, config: DacBaselineConfig | None = None, device: str | None = None):
        self.config = config or DacBaselineConfig()
        self.device = torch.device(
            device or ("cuda" if torch.cuda.is_available() else "cpu")
        )
        self.tokenizer = AutoTokenizer.from_pretrained(self.config.model_name, use_fast=True)
        if not getattr(self.tokenizer, "is_fast", False):
            raise RuntimeError(
                f"{self.config.model_name} has no fast tokenizer; character offsets are "
                "required to rebuild the compressed text."
            )
        try:
            self.model = AutoModelForCausalLM.from_pretrained(
                self.config.model_name, attn_implementation="eager"
            )
        except TypeError:
            self.model = AutoModelForCausalLM.from_pretrained(self.config.model_name)
        self.model.to(self.device).eval()
        self.special_token_ids = set(getattr(self.tokenizer, "all_special_ids", []))
        self.windowed = False

    def provenance(self) -> Dict[str, object]:
        return {
            "baseline": "dac_official_reimpl",
            "reference": "ACL 2025, arXiv 2507.11942",
            "model": self.config.model_name,
            "alpha": self.config.alpha,
            "fusion": self.config.fusion,
            "window_tokens": self.config.window_tokens,
            "windowed_last_call": self.windowed,
            "task_agnostic": True,
        }

    # ------------------------------------------------------------------

    @staticmethod
    def _normalize(tensor: torch.Tensor) -> torch.Tensor:
        if tensor.numel() == 0:
            return tensor
        lo, hi = tensor.min(), tensor.max()
        if float(hi - lo) > 1e-8:
            return (tensor - lo) / (hi - lo)
        return torch.zeros_like(tensor)

    def _score_window(self, input_ids: torch.Tensor) -> torch.Tensor:
        """Fused per-token score for one window. input_ids: [1, n]."""
        attention_mask = torch.ones_like(input_ids)
        with torch.no_grad():
            outputs = self.model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                output_attentions=True,
            )
            logits = outputs.logits
            shift_logits = logits[..., :-1, :].contiguous()
            shift_labels = input_ids[..., 1:].contiguous()
            ppl = F.cross_entropy(
                shift_logits.view(-1, shift_logits.size(-1)),
                shift_labels.view(-1),
                reduction="none",
            )
            # Token 0 is unconditioned; give it the mean so ppl aligns 1:1.
            ppl = torch.cat([ppl.mean().reshape(1), ppl], dim=0)

            # Column-sum attention across every layer and head (their loop).
            column_sum = None
            for layer_attn in outputs.attentions:
                # layer_attn: [1, heads, n, n] -> sum over heads and over rows.
                per_layer = layer_attn[0].sum(dim=0).sum(dim=0)
                column_sum = per_layer if column_sum is None else column_sum + per_layer
            attn = column_sum if column_sum is not None else torch.zeros_like(ppl)

        ppl_norm = self._normalize(ppl)
        attn_norm = self._normalize(attn)
        if self.config.fusion == "multiplicative":
            # Matches their CODE (torch.mul(ppl, attn)), not their docstring.
            return ppl_norm * attn_norm
        return self.config.alpha * attn_norm + (1.0 - self.config.alpha) * ppl_norm

    def _score_tokens(self, text: str) -> Tuple[Optional[torch.Tensor], Optional[List[List[int]]]]:
        encoding = self.tokenizer(
            text, add_special_tokens=False, return_tensors="pt", return_offsets_mapping=True
        )
        input_ids = encoding["input_ids"]
        offsets = encoding["offset_mapping"][0].tolist()
        total = input_ids.size(1)
        if total < self.config.min_tokens:
            return None, offsets

        window = max(self.config.min_tokens, self.config.window_tokens)
        self.windowed = total > window

        scores: List[torch.Tensor] = []
        for start in range(0, total, window):
            chunk = input_ids[:, start : start + window].to(self.device)
            if chunk.size(1) < 2:
                scores.append(torch.zeros(chunk.size(1), device=self.device))
                continue
            scores.append(self._score_window(chunk))
        return torch.cat(scores, dim=0), offsets

    # ------------------------------------------------------------------

    def _punct_mask(self, text: str, offsets: Sequence[Sequence[int]]) -> torch.Tensor:
        pattern = re.compile(r"^\s*[^\w\s]+\s*$")
        flags = [
            bool(pattern.match(text[s:e])) if e > s else False for s, e in offsets
        ]
        return torch.tensor(flags, dtype=torch.bool, device=self.device)

    def _select_keep(self, score: torch.Tensor, keep_ratio: float, punct: torch.Tensor) -> List[int]:
        total = score.numel()
        keep_count = max(1, min(total, int(round(total * keep_ratio))))

        working = score.clone().float()
        forced = punct if self.config.preserve_punct else torch.zeros_like(punct)
        working = working + 1e5 * forced.float()

        _, topk = torch.topk(working.view(-1), k=keep_count, largest=True)
        keep_mask = torch.zeros(total, dtype=torch.bool, device=score.device)
        keep_mask[topk] = True

        if self.config.avoid_consecutive:
            deleted = torch.nonzero(~keep_mask, as_tuple=False).view(-1)
            run = 1
            rescues: List[int] = []
            for i in range(1, deleted.numel()):
                if int(deleted[i]) == int(deleted[i - 1]) + 1:
                    run += 1
                    if run == 2:
                        rescues.append(int(deleted[i - 1]))
                else:
                    run = 1
            # Trade rather than append, so the ratio stays where it was asked to be.
            for idx in rescues:
                droppable = torch.nonzero(keep_mask & ~forced, as_tuple=False).view(-1)
                if droppable.numel() == 0:
                    break
                victim = droppable[torch.argmin(score[droppable])]
                if float(score[victim]) >= float(score[idx]):
                    continue
                keep_mask[victim] = False
                keep_mask[idx] = True

        return torch.nonzero(keep_mask, as_tuple=False).view(-1).sort()[0].tolist()

    @staticmethod
    def _rebuild(text: str, offsets: Sequence[Sequence[int]], keep: Sequence[int]) -> str:
        char_keep = [False] * len(text)
        for idx in keep:
            s, e = offsets[int(idx)]
            for pos in range(s, min(e, len(text))):
                char_keep[pos] = True
        out: List[str] = []
        for pos, ch in enumerate(text):
            if char_keep[pos]:
                out.append(ch)
            elif ch.isspace() and (
                (pos > 0 and char_keep[pos - 1])
                or (pos + 1 < len(char_keep) and char_keep[pos + 1])
            ):
                out.append(ch)
        return "".join(out).strip()

    def compress(self, context: str, keep_ratio: float) -> str:
        """Compress `context` down to approximately `keep_ratio` of its tokens."""
        text = context.strip()
        if not text or keep_ratio >= 1.0:
            return text

        token_total = len(self.tokenizer.tokenize(text))
        if token_total < self.config.min_tokens:
            return text

        dyn_steps = min(max(1, token_total // 12), self.config.max_dyn_steps)
        # Geometric decomposition: per-step keep ratio compounds to keep_ratio.
        step_keep = keep_ratio ** (1.0 / dyn_steps)

        current = text
        for _ in range(dyn_steps):
            score, offsets = self._score_tokens(current)
            if score is None or offsets is None or score.numel() < self.config.min_tokens:
                break
            if score.numel() != len(offsets):
                # Windowing must never desynchronise scores from offsets.
                raise RuntimeError(
                    f"DAC baseline: {score.numel()} scores for {len(offsets)} tokens"
                )
            punct = self._punct_mask(current, offsets)
            keep = self._select_keep(score, step_keep, punct)
            nxt = self._rebuild(current, offsets, keep)
            if not nxt or nxt == current:
                break
            current = nxt
        return current
