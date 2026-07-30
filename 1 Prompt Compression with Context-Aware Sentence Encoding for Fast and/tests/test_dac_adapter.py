"""Regression tests for the DAC adapter fixes (audit findings D3, D4, D6, D8).

These exercise the pure/tensor logic without loading any pretrained model, so
they run on CPU in under a second. Model loading itself is covered by the
availability assertions in tests/test_dac_enabled.py, which needs network access.

Run:  python -m pytest tests/test_dac_adapter.py -v
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest
import torch

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from pipeline.dac_adapter import DacCompressionConfig, DacTokenAdapter


def make_adapter(**config_kwargs) -> DacTokenAdapter:
    """A DacTokenAdapter with only the attributes the pure methods need.

    Bypasses __init__ deliberately: these tests target the selection and
    reconstruction logic, not model loading.
    """
    adapter = object.__new__(DacTokenAdapter)
    adapter.config = DacCompressionConfig(**config_kwargs)
    adapter.device = torch.device("cpu")
    adapter.attention_available = None
    adapter.attention_unavailable_reason = None
    return adapter


# ---------------------------------------------------------------------------
# D4: the keep budget must be honoured exactly.
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("compress_ratio", [0.1, 0.25, 0.5, 0.75, 0.9])
def test_select_keep_indices_honours_budget(compress_ratio):
    adapter = make_adapter(avoid_consecutive=True, preserve_punct=False)
    torch.manual_seed(0)
    n = 40
    score = torch.rand(n)
    punct = torch.zeros(n, dtype=torch.bool)
    protect = torch.zeros(n, dtype=torch.bool)

    keep = adapter.select_keep_indices(score, compress_ratio, punct, protect)
    expected = max(1, int(round(n * (1 - compress_ratio))))
    assert keep.numel() == expected, (
        f"requested keep={expected} but got {keep.numel()}: the avoid_consecutive "
        "rescue is inflating the budget again (finding D4)"
    )


def test_select_keep_indices_budget_holds_with_rescues():
    """A score pattern that forces many consecutive deletions still respects budget."""
    adapter = make_adapter(avoid_consecutive=True, preserve_punct=False)
    n = 30
    # Descending score: topk keeps a prefix, so all deletions are consecutive.
    score = torch.arange(n, dtype=torch.float32).flip(0)
    punct = torch.zeros(n, dtype=torch.bool)
    protect = torch.zeros(n, dtype=torch.bool)

    keep = adapter.select_keep_indices(score, 0.5, punct, protect)
    assert keep.numel() == 15
    assert keep.unique().numel() == keep.numel(), "duplicate indices returned"


def test_protected_and_punct_tokens_are_force_kept():
    adapter = make_adapter(avoid_consecutive=False, preserve_punct=True)
    n = 20
    score = torch.zeros(n)  # all equally unattractive
    punct = torch.zeros(n, dtype=torch.bool)
    punct[3] = True
    protect = torch.zeros(n, dtype=torch.bool)
    protect[11] = True

    keep = adapter.select_keep_indices(score, 0.9, punct, protect)  # keep only 2
    kept = set(keep.tolist())
    assert 3 in kept and 11 in kept, "forced tokens were dropped"


# ---------------------------------------------------------------------------
# D6: reconstruct_text must report positions so masks stay aligned.
# ---------------------------------------------------------------------------

def test_reconstruct_text_positions_match_output():
    adapter = make_adapter()
    text = "Alpha beta gamma delta"
    # Token offsets for the four words.
    offsets = [[0, 5], [6, 10], [11, 16], [17, 22]]
    kept, positions = adapter.reconstruct_text(text, offsets, [0, 2])

    assert len(positions) == len(kept), (
        "positions must be 1:1 with the returned text so a character-aligned "
        "mask can be carried forward (finding D6)"
    )
    assert "".join(text[p] for p in positions) == kept
    assert "Alpha" in kept and "gamma" in kept and "beta" not in kept


def test_protected_mask_survives_iteration():
    """Carrying the mask through positions keeps protection on the same words."""
    adapter = make_adapter()
    text = "Keep 42 percent always drop this filler"
    protect = [False] * len(text)
    start = text.index("42")
    for pos in range(start, start + len("42 percent")):
        protect[pos] = True

    offsets = []
    cursor = 0
    for word in text.split(" "):
        offsets.append([cursor, cursor + len(word)])
        cursor += len(word) + 1

    keep_idx = [0, 1, 2, 3]  # drop "drop this filler"
    kept, positions = adapter.reconstruct_text(text, offsets, keep_idx)
    carried = [protect[p] if p < len(protect) else False for p in positions]

    protected_chars = "".join(
        ch for ch, flag in zip(kept, carried) if flag
    )
    assert "42" in protected_chars and "percent" in protected_chars, (
        f"protection drifted after reconstruction: protected={protected_chars!r}"
    )


# ---------------------------------------------------------------------------
# D3: attention is aggregated by character offset, never by token position.
# ---------------------------------------------------------------------------

def test_aggregate_chars_averages_window():
    adapter = make_adapter()
    values = [0.0, 1.0, 2.0, 3.0, 4.0]
    assert adapter._aggregate_chars(values, 1, 3) == pytest.approx(1.5)
    assert adapter._aggregate_chars(values, 0, 5) == pytest.approx(2.0)
    # Out-of-range windows must report "no data", not silently return 0.0.
    assert adapter._aggregate_chars(values, 9, 12) is None
    assert adapter._aggregate_chars(values, 3, 3) is None


def test_attention_for_offsets_is_offset_driven():
    adapter = make_adapter()
    # Char attention: first 5 chars hot, rest cold.
    char_attn = [1.0] * 5 + [0.0] * 15
    offsets = [[0, 5], [6, 11], [12, 20]]
    out = adapter._attention_for_offsets(char_attn, offsets)
    assert out[0].item() == pytest.approx(1.0)
    assert out[1].item() == pytest.approx(0.0)
    assert out[2].item() == pytest.approx(0.0)


def test_fuse_renormalises_when_attention_missing():
    """Missing attention must renormalise onto loss, not fuse against zeros."""
    adapter = make_adapter(alpha=0.8, fusion="additive", require_attention=False)
    losses = torch.tensor([1.0, 3.0, 2.0])
    fused = adapter._fuse(losses, None)
    expected = adapter.normalize(losses)
    assert torch.allclose(fused, expected), (
        "with no attention the fused score should equal the normalised loss, "
        "not 0.2 * loss (finding D3)"
    )


def test_fuse_can_fail_loud_on_missing_attention():
    adapter = make_adapter(require_attention=True)
    adapter.attention_unavailable_reason = "truncated"
    with pytest.raises(RuntimeError, match="require_attention"):
        adapter._fuse(torch.tensor([1.0, 2.0]), None)


def test_fuse_uses_alpha_when_attention_present():
    adapter = make_adapter(alpha=0.75, fusion="additive")
    losses = torch.tensor([0.0, 1.0])
    attention = torch.tensor([1.0, 0.0])
    fused = adapter._fuse(losses, attention)
    # normalize maps both onto [0,1]; alpha weights the attention term.
    assert fused[0].item() == pytest.approx(0.75)
    assert fused[1].item() == pytest.approx(0.25)


# ---------------------------------------------------------------------------
# D7: token-count fallback must not return character counts.
# ---------------------------------------------------------------------------

def test_token_count_fallback_uses_words_not_chars():
    adapter = make_adapter()

    class Exploding:
        def tokenize(self, text):
            raise RuntimeError("no tokenizer")

    adapter.salience_tokenizer = Exploding()
    adapter.encoder_tokenizer = Exploding()
    text = "one two three four five"
    assert adapter._token_count(text) == 5, (
        "fallback must count words; returning len(text) inflates the count ~5x "
        "and saturates dyn_steps (finding D7)"
    )
