from __future__ import annotations

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import argparse
import json
from typing import Iterable, List, Optional, Sequence

from context_aware_encoder_model.context_aware_sentence_encoder import (
    ContextAwareEncoderConfig,
    ContextAwareSentenceEncoder,
)
from intra_sentence_model.span_feature_utils import (
    answer_overlap_score,
    build_span_feature_dict,
    detect_question_type,
    normalize_scores,
    overlap_ratio,
    query_overlap_score,
    tokenize_mixed,
)
from pipeline.task_aware_compression import DynamicSpanCompressor, IntraSentenceCompressionConfig


def load_jsonl(path: Path) -> List[dict]:
    rows = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def save_jsonl(path: Path, rows: Iterable[dict]) -> None:
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def split_sentences(text: str) -> List[str]:
    import re
    return [s.strip() for s in re.split(r"(?<=[。！？.!?])\s*", text.strip()) if s.strip()]


def char_f1(pred: str, ref: str) -> float:
    pred_toks = tokenize_mixed(pred)
    ref_toks = tokenize_mixed(ref)
    if not pred_toks or not ref_toks:
        return 0.0
    pred_counts = {}
    ref_counts = {}
    for tok in pred_toks:
        pred_counts[tok] = pred_counts.get(tok, 0) + 1
    for tok in ref_toks:
        ref_counts[tok] = ref_counts.get(tok, 0) + 1
    overlap = 0
    for tok, count in pred_counts.items():
        overlap += min(count, ref_counts.get(tok, 0))
    if overlap == 0:
        return 0.0
    precision = overlap / max(len(pred_toks), 1)
    recall = overlap / max(len(ref_toks), 1)
    return 2 * precision * recall / max(precision + recall, 1e-12)


def extractive_demo_answer(question: str, context: str) -> str:
    sentences = split_sentences(context)
    if not sentences:
        return ""
    scored = [(query_overlap_score(question, sent), idx, sent) for idx, sent in enumerate(sentences)]
    scored.sort(key=lambda x: (x[0], -len(x[2])), reverse=True)
    top = [sent for _, _, sent in scored[:2]]
    return " ".join(top)


def compute_answer_quality(question: str, context: str, answer: str) -> float:
    if not answer.strip():
        return 0.0
    pred = extractive_demo_answer(question, context)
    return char_f1(pred, answer)


def load_encoder_from_dir(encoder_dir: str, device: str) -> ContextAwareSentenceEncoder:
    encoder_dir_path = Path(encoder_dir)
    cfg_dict = json.loads((encoder_dir_path / "encoder_config.json").read_text(encoding="utf-8"))
    cfg_dict["device"] = device
    cfg_dict["model_name"] = str(encoder_dir_path)
    cfg_dict = {
        k: v
        for k, v in cfg_dict.items()
        if k in {"model_name", "max_length", "temperature", "device", "marker_start", "marker_end"}
    }
    config = ContextAwareEncoderConfig(**cfg_dict)
    model = ContextAwareSentenceEncoder(config)
    model.eval()
    return model


def pick_training_sentences(row: dict) -> List[str]:
    context = row.get("context", "")
    context_sentences = split_sentences(context)
    seen = set()
    out = []
    for sentence in row.get("supporting_sentences", []) or []:
        sentence = str(sentence).strip()
        if not sentence:
            continue
        if sentence not in context_sentences:
            continue
        key = sentence.strip()
        if key in seen:
            continue
        seen.add(key)
        out.append(sentence)
    positive = str(row.get("positive_sentence", "")).strip()
    if positive and positive in context_sentences and positive not in seen:
        out.append(positive)
    return out


def build_pseudo_label(
    span,
    feature_dict: dict,
    answer_drop: float,
) -> tuple[int, float]:
    score = (
        0.28 * feature_dict["anchor_score"]
        + 0.22 * feature_dict["query_overlap"]
        + 0.16 * feature_dict["task_reward"]
        + 0.14 * feature_dict["answer_overlap"]
        + 0.14 * answer_drop
        + 0.03 * feature_dict["attention_score"]
        + 0.03 * feature_dict["dac_score"]
    )
    if span.kind == "parenthetical":
        score -= 0.12
    elif span.kind == "example":
        score -= 0.08
    elif span.kind == "tail":
        score -= 0.04
    label = 1 if span.protected or answer_drop >= 0.18 or score >= 0.46 else 0
    return label, float(score)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_file", type=str, required=True)
    parser.add_argument("--output_file", type=str, required=True)
    parser.add_argument("--encoder_dir", type=str, required=True)
    parser.add_argument("--device", type=str, default="cpu")
    parser.add_argument("--default_keep_ratio", type=float, default=0.78)
    parser.add_argument("--min_sentence_chars", type=int, default=18)
    args = parser.parse_args()

    rows = load_jsonl(Path(args.input_file))
    encoder = load_encoder_from_dir(args.encoder_dir, args.device)
    compressor = DynamicSpanCompressor(
        encoder,
        IntraSentenceCompressionConfig(
            target_keep_ratio=args.default_keep_ratio,
            min_sentence_chars=args.min_sentence_chars,
        ),
    )

    output_rows = []
    for row_idx, row in enumerate(rows):
        question = str(row.get("question", "")).strip()
        context = str(row.get("context", "")).strip()
        answer = str(row.get("answer", "")).strip()
        if not question or not context:
            continue

        candidate_sentences = pick_training_sentences(row)
        if not candidate_sentences:
            continue

        question_type = str(row.get("question_type", "")).strip() or detect_question_type(question)
        base_answer_quality = compute_answer_quality(question, context, answer)

        for sent_idx, sentence in enumerate(candidate_sentences):
            if len(sentence.strip()) < args.min_sentence_chars:
                continue

            spans = compressor.split_sentence_into_spans(question, sentence, question_type)
            if len(spans) < 2:
                continue

            attention_scores = normalize_scores(compressor.compute_span_attention_scores(question, spans))
            dac_scores = normalize_scores(compressor.compute_dac_span_scores(question, spans))
            if not attention_scores:
                attention_scores = [0.0 for _ in spans]
            if not dac_scores:
                dac_scores = [0.0 for _ in spans]

            span_rows = []
            sentence_score = float(row.get("sentence_score", 0.75 if sentence == row.get("positive_sentence") else 0.68))
            keep_ratio = float(row.get("keep_ratio", args.default_keep_ratio))
            sentence_length = max(len(sentence), 1)
            sentence_token_len = max(len(tokenize_mixed(sentence)), 1)

            for span_index, span in enumerate(spans):
                pruned_sentence = compressor.cleanup_sentence(sentence[: span.start] + sentence[span.end :], sentence)
                modified_context = context.replace(sentence, pruned_sentence, 1)
                removal_quality = compute_answer_quality(question, modified_context, answer)
                answer_drop = max(0.0, base_answer_quality - removal_quality)
                answer_overlap = answer_overlap_score(answer, span.text)
                feature_dict = build_span_feature_dict(
                    question=question,
                    span_text=span.text,
                    span_kind=span.kind,
                    span_index=span_index,
                    num_spans=len(spans),
                    sentence_length=sentence_length,
                    sentence_token_length=sentence_token_len,
                    start=span.start,
                    end=span.end,
                    sentence_score=sentence_score,
                    keep_ratio=keep_ratio,
                    question_type=question_type,
                    attention_score=attention_scores[span_index],
                    dac_score=dac_scores[span_index],
                    answer_overlap=answer_overlap,
                    answer_drop=answer_drop,
                    protected=span.protected,
                )
                label, pseudo_keep_score = build_pseudo_label(span, feature_dict, answer_drop)
                span_rows.append(
                    {
                        "text": span.text,
                        "start": span.start,
                        "end": span.end,
                        "kind": span.kind,
                        "protected": span.protected,
                        "answer_overlap": answer_overlap,
                        "answer_drop": answer_drop,
                        "pseudo_keep_score": pseudo_keep_score,
                        "label": label,
                        "features": feature_dict,
                    }
                )

            output_rows.append(
                {
                    "source_id": row.get("id", f"row_{row_idx:06d}"),
                    "question": question,
                    "context": context,
                    "selected_sentence": sentence,
                    "question_type": question_type,
                    "sentence_score": sentence_score,
                    "keep_ratio": keep_ratio,
                    "answer": answer,
                    "base_answer_quality": base_answer_quality,
                    "spans": span_rows,
                    "source_role": "positive" if sentence == row.get("positive_sentence") else "supporting",
                    "selected_sentence_index": sent_idx,
                }
            )

    save_jsonl(Path(args.output_file), output_rows)
    print(f"saved pseudo labels: {args.output_file}")
    print(f"num_sentence_examples: {len(output_rows)}")


if __name__ == "__main__":
    main()
