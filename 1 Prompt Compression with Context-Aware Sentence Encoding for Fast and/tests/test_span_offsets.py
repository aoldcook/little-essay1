"""Span offsets must describe span.text exactly (audit finding D8).

Every consumer of SpanUnit indexes the sentence with .start/.end:
DAC salience token selection, build_protected_char_mask, and span
reconstruction. When offsets drift from .text, protection lands on the wrong
characters and salience is attributed to the wrong tokens -- silently, because
nothing downstream re-checks the correspondence.

Run:  python -m pytest tests/test_span_offsets.py -v
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from pipeline.task_aware_compression import (
    DynamicSpanCompressor,
    IntraSentenceCompressionConfig,
)


class WhitespaceTokenizer:
    def tokenize(self, text: str):
        return text.split()


def make_compressor() -> DynamicSpanCompressor:
    """A DynamicSpanCompressor with only what the span splitters touch.

    Bypasses __init__ so no encoder or DAC model is loaded: these tests target
    offset arithmetic, which is pure string handling.
    """
    compressor = object.__new__(DynamicSpanCompressor)
    compressor.config = IntraSentenceCompressionConfig()
    compressor.tokenizer = WhitespaceTokenizer()
    return compressor


SENTENCES = [
    # The case that first exposed the bug: source attribution + long tail.
    "Sebnem Arsu (New York Times -- link has text and video) notes the increase "
    "in Iraq refugees to the country since October 31st, which the ministry "
    "reported that officials had not anticipated.",
    # Relative clause after a comma.
    "The reactor exceeded 550 degrees, which the operators had not expected, "
    "and the coolant pump failed shortly afterwards during the night shift.",
    # Coordination with several connectors.
    "The report includes rising costs and delayed shipments and reduced staffing "
    "across the three regional distribution centres described in the appendix.",
    # Attribution pattern that triggers the source-attribution splitter.
    "The committee reported that the emissions had fallen by 12 percent since "
    "the previous audit, although the methodology changed midway through.",
    # Leading/trailing whitespace and multiple spaces between clauses.
    "   Costs rose sharply.    Revenue fell,  however the margin held steady "
    "because the supplier absorbed the difference.   ",
    # Parenthetical nesting.
    "Growth (measured as year-over-year change (YoY) in constant currency) "
    "slowed to 3 percent, which analysts attributed to weaker demand overseas.",
]


@pytest.mark.parametrize("sentence", SENTENCES)
def test_span_offsets_describe_span_text(sentence):
    compressor = make_compressor()
    spans = compressor.split_sentence_into_spans(sentence, sentence, "cause")
    spans = compressor.refine_long_spans(sentence, spans, "cause")
    spans = compressor.apply_evidence_list_floor(sentence, spans, "cause")

    assert spans, "expected at least one span"
    for span in spans:
        assert 0 <= span.start <= span.end <= len(sentence), (
            f"span {span.start}:{span.end} out of range for len={len(sentence)}"
        )
        assert sentence[span.start:span.end] == span.text, (
            "offsets must slice out exactly span.text\n"
            f"  sentence[{span.start}:{span.end}] = {sentence[span.start:span.end]!r}\n"
            f"  span.text                        = {span.text!r}"
        )


@pytest.mark.parametrize("sentence", SENTENCES)
def test_spans_do_not_overlap_and_stay_ordered(sentence):
    compressor = make_compressor()
    spans = compressor.split_sentence_into_spans(sentence, sentence, "cause")
    spans = compressor.refine_long_spans(sentence, spans, "cause")

    ordered = sorted(spans, key=lambda s: s.start)
    for earlier, later in zip(ordered, ordered[1:]):
        assert earlier.end <= later.start, (
            f"spans overlap: {earlier.start}:{earlier.end} and {later.start}:{later.end}. "
            "Overlapping spans double-count tokens in salience aggregation."
        )


def test_span_text_is_stripped():
    """A span whose text carries edge whitespace cannot satisfy the offset invariant."""
    compressor = make_compressor()
    sentence = "   Costs rose sharply.    Revenue fell steadily afterwards.   "
    spans = compressor.split_sentence_into_spans(sentence, sentence, "cause")
    for span in spans:
        assert span.text == span.text.strip(), f"unstripped span text: {span.text!r}"
        assert span.text, "empty span emitted"
