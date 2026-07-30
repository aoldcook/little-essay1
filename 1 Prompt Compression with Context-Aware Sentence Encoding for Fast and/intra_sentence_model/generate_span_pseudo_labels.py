from __future__ import annotations

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import argparse
import json
import torch
from dataclasses import dataclass
from typing import Iterable, List, Sequence

from context_aware_encoder_model.context_aware_sentence_encoder import (
    ContextAwareEncoderConfig,
    ContextAwareSentenceEncoder,
)
from intra_sentence_model.span_feature_utils import (
    answer_overlap_score,
    build_span_feature_dict,
    detect_question_type,
    normalize_scores,
    query_overlap_score,
    tokenize_mixed,
)
from pipeline.dac_adapter import DacCompressionConfig
from pipeline.task_aware_compression import DynamicSpanCompressor, IntraSentenceCompressionConfig


@dataclass
class TrainingSentence:
    text: str
    role: str
    sentence_score: float


class LightweightTokenizer:
    is_fast = False
    mask_token_id = None
    all_special_ids: List[int] = []

    def tokenize(self, text: str) -> List[str]:
        return tokenize_mixed(text)

    def convert_tokens_to_ids(self, token: str) -> int:
        return abs(hash(token)) % 100000


class LightweightSentenceEncoder:
    def __init__(self, max_length: int = 1024):
        self.tokenizer = LightweightTokenizer()
        self.encoder = None
        self.device = torch.device("cpu")
        self.config = type(
            "LightweightEncoderConfig",
            (),
            {
                "model_name": "lightweight_lexical_fallback",
                "max_length": max_length,
                "cache_dir": "",
                "trust_remote_code": False,
            },
        )()


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


def load_existing_example_ids(path: Path) -> set[str]:
    existing_ids: set[str] = set()
    if not path.exists():
        return existing_ids
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError:
                continue
            example_id = str(row.get("example_id", "")).strip()
            if example_id:
                existing_ids.add(example_id)
    return existing_ids


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


def token_recall(reference: str, candidate: str) -> float:
    ref_toks = set(tokenize_mixed(reference))
    if not ref_toks:
        return 0.0
    cand_toks = set(tokenize_mixed(candidate))
    return len(ref_toks & cand_toks) / max(len(ref_toks), 1)


def average_token_recall(references: Sequence[str], candidate: str) -> float:
    scores = [token_recall(ref, candidate) for ref in references if str(ref).strip()]
    if not scores:
        return 0.0
    return float(sum(scores) / len(scores))


def extractive_demo_answer(question: str, context: str) -> str:
    sentences = split_sentences(context)
    if not sentences:
        return ""
    scored = [(query_overlap_score(question, sent), idx, sent) for idx, sent in enumerate(sentences)]
    scored.sort(key=lambda x: (x[0], -len(x[2])), reverse=True)
    top = [sent for _, _, sent in scored[:2]]
    return " ".join(top)


def reader_answer_quality(prediction: str, gold_answer: str) -> float:
    """Answer F1 of a real reader's output against the gold answer."""
    if not gold_answer.strip() or not prediction.strip():
        return 0.0
    from evaluation.qa_metrics import score_prediction

    return float(score_prediction(prediction, [gold_answer]).get("f1", 0.0))


def build_reader_pseudo_label(answer_drop: float, threshold: float) -> tuple[int, float]:
    """Label a span by what removing it actually does to a downstream reader.

    This is the point of the reader-grounded policy. The heuristic and feedback
    policies both compute the label as a weighted sum over the SAME features the
    span model receives as input, so the model can only ever rediscover that
    formula -- measured at 0.990 group-disjoint dev accuracy, versus 0.576 for a
    majority-class baseline and 0.904 for plain logistic regression. A model that
    reproduces a rule it was handed cannot be reported as learned.

    Here the target is a measured quantity the model cannot see: the drop in the
    reader's answer F1 when this span is deleted. Predicting it from cheap
    features is a real learning problem with a real generalisation gap.
    """
    label = 1 if answer_drop >= threshold else 0
    return label, float(answer_drop)


def compute_answer_quality(question: str, context: str, answer: str) -> float:
    if not answer.strip():
        return 0.0
    pred = extractive_demo_answer(question, context)
    return char_f1(pred, answer)


def load_encoder_from_dir(encoder_dir: str, device: str) -> ContextAwareSentenceEncoder:
    if encoder_dir == "lightweight_lexical_fallback":
        return LightweightSentenceEncoder()

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


def pick_training_sentences(row: dict, include_negative_sentences: bool, max_negative_training_sentences: int) -> List[TrainingSentence]:
    context = row.get("context", "")
    context_sentences = split_sentences(context)
    seen = set()
    out: List[TrainingSentence] = []
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
        out.append(TrainingSentence(sentence, "supporting", 0.72))
    positive = str(row.get("positive_sentence", "")).strip()
    if positive and positive in context_sentences and positive not in seen:
        out.append(TrainingSentence(positive, "positive", 0.78))

    if include_negative_sentences:
        negative_count = 0
        for sentence in row.get("negative_sentences", []) or []:
            sentence = str(sentence).strip()
            if not sentence:
                continue
            if sentence not in context_sentences:
                continue
            key = sentence.strip()
            if key in seen:
                continue
            seen.add(key)
            out.append(TrainingSentence(sentence, "negative", 0.22))
            negative_count += 1
            if negative_count >= max_negative_training_sentences:
                break
    return out


def is_hard_keep_span(question: str, question_type: str, answer: str, span_text: str, feature_dict: dict, qa_feedback_drop: float) -> bool:
    if qa_feedback_drop >= 0.12:
        return True
    if feature_dict["answer_overlap"] >= 0.30:
        return True
    if feature_dict["query_overlap"] >= 0.40:
        return True
    if question_type in {"numeric", "factoid"} and feature_dict["answer_overlap"] > 0.0:
        return True
    if question_type == "numeric" and any(ch.isdigit() for ch in span_text):
        answer_has_number = any(ch.isdigit() for ch in answer)
        question_has_number_hint = any(hint in question.lower() for hint in ("how many", "how much", "number", "percent", "value", "year"))
        if answer_has_number or question_has_number_hint:
            return True
    return False


def build_heuristic_pseudo_label(
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


def build_feedback_pseudo_label(
    question: str,
    question_type: str,
    answer: str,
    role: str,
    span,
    feature_dict: dict,
    answer_drop: float,
    support_drop: float,
    sentence_drop: float,
) -> tuple[int, float, dict]:
    qa_feedback_drop = max(answer_drop, support_drop, sentence_drop)
    keep_score = (
        0.34 * qa_feedback_drop
        + 0.20 * feature_dict["answer_overlap"]
        + 0.16 * feature_dict["query_overlap"]
        + 0.12 * feature_dict["anchor_score"]
        + 0.10 * feature_dict["task_reward"]
        + 0.04 * feature_dict["attention_score"]
        + 0.04 * feature_dict["dac_score"]
    )
    if role == "negative":
        keep_score -= 0.22
    if role == "positive":
        keep_score += 0.05
    elif role == "supporting":
        keep_score += 0.03

    if span.kind == "source_attribution":
        keep_score -= 0.22
    elif span.kind == "background_lead":
        keep_score -= 0.18
    elif span.kind == "parenthetical":
        keep_score -= 0.16
    elif span.kind == "tail":
        keep_score -= 0.10
    elif span.kind == "example" and qa_feedback_drop < 0.08:
        keep_score -= 0.08

    hard_keep = is_hard_keep_span(question, question_type, answer, span.text, feature_dict, qa_feedback_drop)
    weak_redundant = (
        qa_feedback_drop < 0.04
        and feature_dict["answer_overlap"] == 0.0
        and feature_dict["query_overlap"] < 0.20
        and feature_dict["anchor_score"] < 0.45
    )
    if role == "negative" and not hard_keep:
        label = 0
    elif hard_keep:
        label = 1
    elif weak_redundant:
        label = 0
    else:
        label = 1 if keep_score >= 0.34 else 0

    feedback = {
        "qa_feedback_drop": float(qa_feedback_drop),
        "answer_drop": float(answer_drop),
        "support_drop": float(support_drop),
        "sentence_drop": float(sentence_drop),
        "hard_keep": bool(hard_keep),
        "weak_redundant": bool(weak_redundant),
    }
    return label, float(keep_score), feedback


def build_example_id(source_id: str, sent_idx: int) -> str:
    return f"{source_id}::{sent_idx}"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_file", type=str, required=True)
    parser.add_argument("--output_file", type=str, required=True)
    parser.add_argument("--encoder_dir", type=str, required=True)
    parser.add_argument("--device", type=str, default="cpu")
    parser.add_argument("--default_keep_ratio", type=float, default=0.52)
    parser.add_argument("--min_sentence_chars", type=int, default=45)
    parser.add_argument("--include_negative_sentences", action="store_true")
    parser.add_argument("--max_negative_training_sentences", type=int, default=2)
    parser.add_argument("--label_policy", choices=["heuristic", "feedback", "reader"], default="heuristic",
                        help="'reader' labels each span by the measured drop in a real "
                             "reader's answer F1 when the span is removed -- a target the "
                             "span model cannot see. 'heuristic' and 'feedback' both derive "
                             "the label from the model's own input features and are "
                             "therefore only distillations of a hand-written rule.")
    parser.add_argument("--answer_mode", choices=["lexical", "reader"], default="lexical",
                        help="'lexical' uses the extractive_demo_answer stub. 'reader' calls "
                             "a real LLM. Required by --label_policy reader.")
    parser.add_argument("--label_reader_model", type=str, default="qwen-flash",
                        help="Reader used to GENERATE labels. Deliberately defaults to a "
                             "different model from the evaluation reader: labelling and "
                             "scoring with the same LLM would tune Stage 2 to its own judge.")
    parser.add_argument("--reader_concurrency", type=int, default=8)
    parser.add_argument("--reader_cache", type=str, default="",
                        help="JSONL cache of reader answers. Defaults to <output_file>.readercache.jsonl")
    parser.add_argument("--reader_drop_threshold", type=float, default=0.10,
                        help="Answer-F1 drop at or above which a span is labelled keep=1.")
    parser.add_argument("--log_every", type=int, default=500)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--disable_dac", action="store_true",
                        help="Generate features with the DAC salience signal off (ablation arm).")
    parser.add_argument("--dac_salience_backend", type=str, default="causal",
                        choices=["causal", "mlm"])
    parser.add_argument("--dac_salience_model", type=str, default=None)
    args = parser.parse_args()

    if args.label_policy == "reader" and args.answer_mode != "reader":
        raise SystemExit(
            "--label_policy reader requires --answer_mode reader: the label IS the "
            "measured reader drop, so there is nothing to derive it from otherwise."
        )

    batch_reader = None
    if args.answer_mode == "reader":
        from evaluation.cached_reader import CachedBatchReader
        from evaluation.reader_client import QwenReader, ReaderConfig

        reader = QwenReader(ReaderConfig.from_env(model=args.label_reader_model))
        smoke = reader.smoke_test()
        print("label_reader_smoke_test:", json.dumps(smoke, ensure_ascii=False))
        if not smoke["reachable"]:
            raise SystemExit(
                f"Label reader unreachable (model={smoke['model']}): {smoke['error']}"
            )
        cache_path = args.reader_cache or (str(args.output_file) + ".readercache.jsonl")
        batch_reader = CachedBatchReader(
            reader, cache_path=cache_path, concurrency=args.reader_concurrency
        )
        print("label_reader:", json.dumps(batch_reader.provenance(), ensure_ascii=False))

    rows = load_jsonl(Path(args.input_file))
    encoder = load_encoder_from_dir(args.encoder_dir, args.device)
    compressor = DynamicSpanCompressor(
        encoder,
        IntraSentenceCompressionConfig(
            target_keep_ratio=args.default_keep_ratio,
            min_sentence_chars=args.min_sentence_chars,
        ),
        dac_config=DacCompressionConfig(
            salience_backend=args.dac_salience_backend,
            salience_model_name=args.dac_salience_model or "",
        ),
        enable_dac=not args.disable_dac,
    )

    output_path = Path(args.output_file)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    existing_ids = load_existing_example_ids(output_path) if args.resume else set()
    mode = "a" if args.resume and output_path.exists() else "w"
    initial_examples = len(existing_ids)
    written_examples = 0

    with output_path.open(mode, encoding="utf-8") as out_f:
        for row_idx, row in enumerate(rows):
            if args.log_every > 0 and row_idx > 0 and row_idx % args.log_every == 0:
                print(
                    f"processed_rows={row_idx}/{len(rows)} pseudo_sentence_examples={initial_examples + written_examples}",
                    flush=True,
                )

            question = str(row.get("question", "")).strip()
            context = str(row.get("context", "")).strip()
            answer = str(row.get("answer", "")).strip()
            if not question or not context:
                continue

            candidate_sentences = pick_training_sentences(
                row,
                include_negative_sentences=args.include_negative_sentences,
                max_negative_training_sentences=args.max_negative_training_sentences,
            )
            if not candidate_sentences:
                continue

            question_type = str(row.get("question_type", "")).strip() or detect_question_type(question)
            base_answer_quality = compute_answer_quality(question, context, answer)
            source_id = str(row.get("id", f"row_{row_idx:06d}"))

            support_refs = [str(item).strip() for item in (row.get("supporting_sentences", []) or []) if str(item).strip()]
            positive_ref = str(row.get("positive_sentence", "")).strip()
            if positive_ref:
                support_refs.append(positive_ref)
            base_support_recall = average_token_recall(support_refs, context)

            for sent_idx, training_sentence in enumerate(candidate_sentences):
                sentence = training_sentence.text
                example_id = build_example_id(source_id, sent_idx)
                if example_id in existing_ids:
                    continue
                if len(sentence.strip()) < args.min_sentence_chars:
                    continue

                spans = compressor.split_sentence_into_spans(question, sentence, question_type)
                spans = compressor.refine_long_spans(question, spans, question_type)
                spans = compressor.apply_evidence_list_floor(question, spans, question_type)
                if len(spans) < 2:
                    continue

                attention_scores = normalize_scores(compressor.compute_span_attention_scores(question, spans))
                # `spans` carry offsets into `sentence`; it must be passed through or
                # the salience scores get attributed to the wrong tokens (finding D8).
                dac_scores = normalize_scores(
                    compressor.compute_dac_span_scores(question, spans, sentence=sentence)
                )
                if not attention_scores:
                    attention_scores = [0.0 for _ in spans]
                if not dac_scores:
                    dac_scores = [0.0 for _ in spans]

                # Reader-grounded labels: ask a real LLM to answer with the full
                # context and with each span removed, and use the drop in answer
                # F1 as supervision. Batched here because the calls for one
                # sentence are independent; see evaluation/cached_reader.py.
                reader_removal_quality: List[float] = []
                reader_base_quality = None
                if batch_reader is not None:
                    pruned_contexts = []
                    for span in spans:
                        pruned = compressor.cleanup_sentence(
                            sentence[: span.start] + sentence[span.end :], sentence
                        )
                        pruned_contexts.append(context.replace(sentence, pruned, 1))
                    answers = batch_reader.answer_many(
                        [(question, context)] + [(question, c) for c in pruned_contexts]
                    )
                    reader_base_quality = reader_answer_quality(answers[0], answer)
                    reader_removal_quality = [
                        reader_answer_quality(a, answer) for a in answers[1:]
                    ]

                span_rows = []
                sentence_score = float(row.get("sentence_score", training_sentence.sentence_score))
                keep_ratio = float(row.get("keep_ratio", args.default_keep_ratio))
                sentence_length = max(len(sentence), 1)
                sentence_token_len = max(len(tokenize_mixed(sentence)), 1)
                base_sentence_recall = token_recall(sentence, context)

                for span_index, span in enumerate(spans):
                    pruned_sentence = compressor.cleanup_sentence(sentence[: span.start] + sentence[span.end :], sentence)
                    modified_context = context.replace(sentence, pruned_sentence, 1)
                    if batch_reader is not None:
                        removal_quality = reader_removal_quality[span_index]
                        answer_drop = max(0.0, (reader_base_quality or 0.0) - removal_quality)
                    else:
                        removal_quality = compute_answer_quality(question, modified_context, answer)
                        answer_drop = max(0.0, base_answer_quality - removal_quality)
                    answer_recall_drop = max(0.0, token_recall(answer, context) - token_recall(answer, modified_context)) if answer else 0.0
                    support_drop = max(0.0, base_support_recall - average_token_recall(support_refs, modified_context))
                    sentence_drop = max(0.0, base_sentence_recall - token_recall(sentence, modified_context))
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
                        answer_drop=max(answer_drop, answer_recall_drop),
                        protected=span.protected,
                    )
                    heuristic_label, heuristic_score = build_heuristic_pseudo_label(span, feature_dict, answer_drop)
                    if args.label_policy == "reader":
                        label, pseudo_keep_score = build_reader_pseudo_label(
                            answer_drop, args.reader_drop_threshold
                        )
                        feedback = {
                            "qa_feedback_drop": float(answer_drop),
                            "answer_drop": float(answer_drop),
                            "reader_base_quality": float(reader_base_quality or 0.0),
                            "reader_removal_quality": float(removal_quality),
                            "support_drop": float(support_drop),
                            "sentence_drop": float(sentence_drop),
                            "hard_keep": False,
                            "weak_redundant": False,
                        }
                    elif args.label_policy == "feedback":
                        label, pseudo_keep_score, feedback = build_feedback_pseudo_label(
                            question=question,
                            question_type=question_type,
                            answer=answer,
                            role=training_sentence.role,
                            span=span,
                            feature_dict=feature_dict,
                            answer_drop=max(answer_drop, answer_recall_drop),
                            support_drop=support_drop,
                            sentence_drop=sentence_drop,
                        )
                    else:
                        label, pseudo_keep_score = heuristic_label, heuristic_score
                        feedback = {
                            "qa_feedback_drop": float(max(answer_drop, answer_recall_drop, support_drop, sentence_drop)),
                            "answer_drop": float(max(answer_drop, answer_recall_drop)),
                            "support_drop": float(support_drop),
                            "sentence_drop": float(sentence_drop),
                            "hard_keep": False,
                            "weak_redundant": False,
                        }
                    span_rows.append(
                        {
                            "text": span.text,
                            "start": span.start,
                            "end": span.end,
                            "kind": span.kind,
                            "protected": span.protected,
                            "answer_overlap": answer_overlap,
                            "answer_drop": max(answer_drop, answer_recall_drop),
                            "support_drop": support_drop,
                            "sentence_drop": sentence_drop,
                            "qa_feedback_drop": feedback["qa_feedback_drop"],
                            "heuristic_label": heuristic_label,
                            "hard_disagreement": bool(label != heuristic_label),
                            "hard_keep": feedback["hard_keep"],
                            "weak_redundant": feedback["weak_redundant"],
                            "pseudo_keep_score": pseudo_keep_score,
                            # Reader diagnostics. reader_base_quality is what the
                            # reader scored on the FULL context: when it is 0 the
                            # reader could not answer even before compression, so
                            # answer_drop is 0 by construction and this span
                            # carries no supervision. Such rows must be counted,
                            # and excluded from any claim about learned behaviour.
                            "reader_base_quality": (
                                float(reader_base_quality) if reader_base_quality is not None else None
                            ),
                            "reader_removal_quality": (
                                float(reader_removal_quality[span_index])
                                if reader_removal_quality else None
                            ),
                            "label": label,
                            "features": feature_dict,
                        }
                    )

                output_row = {
                    "example_id": example_id,
                    "source_id": source_id,
                    "question": question,
                    "context": context,
                    "selected_sentence": sentence,
                    "question_type": question_type,
                    "sentence_score": sentence_score,
                    "keep_ratio": keep_ratio,
                    "answer": answer,
                    "base_answer_quality": base_answer_quality,
                    "spans": span_rows,
                    "source_role": training_sentence.role,
                    "selected_sentence_index": sent_idx,
                    "label_policy": args.label_policy,
                    # Whether dac_score in `spans[*].features` carries real salience
                    # or a constant 0.0. train_span_model stamps this into the
                    # checkpoint so inference cannot silently change the regime.
                    "dac_active": bool(getattr(compressor.dac_adapter, "available", False)),
                    "dac_salience_model": getattr(
                        compressor.dac_adapter, "salience_model_name", None
                    ),
                    "dac_salience_backend": getattr(
                        compressor.dac_adapter, "backend", None
                    ),
                    # Which supervision produced `label`. A span model trained on
                    # rule-derived labels and one trained on measured reader drops
                    # are different objects and must not be confused downstream.
                    "answer_mode": args.answer_mode,
                    "label_reader_model": (
                        args.label_reader_model if args.answer_mode == "reader" else None
                    ),
                    "reader_drop_threshold": (
                        args.reader_drop_threshold if args.label_policy == "reader" else None
                    ),
                }
                out_f.write(json.dumps(output_row, ensure_ascii=False) + "\n")
                written_examples += 1
                existing_ids.add(example_id)

                if args.log_every > 0 and written_examples % max(args.log_every, 1) == 0:
                    out_f.flush()

        out_f.flush()

    print(f"saved pseudo labels: {args.output_file}")
    print(f"num_sentence_examples: {initial_examples + written_examples}")


if __name__ == "__main__":
    main()
