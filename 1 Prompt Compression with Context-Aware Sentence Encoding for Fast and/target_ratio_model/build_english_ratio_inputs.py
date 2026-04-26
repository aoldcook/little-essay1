from __future__ import annotations

import argparse
import json
import re
from pathlib import Path
from typing import Iterable, List, Optional, Sequence

from context_aware_encoder_model.context_aware_sentence_encoder import default_hf_cache_dir
from pipeline.compression_pipeline import ContextAwareCompressor
from target_ratio_model.budget_features import split_sentences


TOKEN_RE = re.compile(r"[A-Za-z][A-Za-z0-9]*(?:-[A-Za-z0-9]+)*|\d+(?:\.\d+)?%?")
STOPWORDS = {
    "a", "an", "and", "are", "as", "at", "be", "by", "can", "could", "do", "does", "did",
    "for", "from", "how", "in", "is", "it", "of", "on", "or", "should", "that", "the", "to",
    "was", "were", "what", "when", "where", "which", "who", "why", "with",
}
QWEN_REQUIRED_FILES = (
    "config.json",
    "tokenizer_config.json",
    "tokenizer.json",
    "model.safetensors.index.json",
    "model-00001-of-00004.safetensors",
    "model-00002-of-00004.safetensors",
    "model-00003-of-00004.safetensors",
    "model-00004-of-00004.safetensors",
)


def load_jsonl(path: Path) -> List[dict]:
    rows = []
    with path.open("r", encoding="utf-8-sig") as f:
        for line in f:
            line = line.strip()
            if line:
                row = json.loads(line)
                if isinstance(row, dict):
                    rows.append(row)
    return rows


def write_jsonl(path: Path, rows: Iterable[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def first_nonempty(row: dict, keys: Sequence[str]) -> str:
    for key in keys:
        value = row.get(key)
        if isinstance(value, str) and value.strip():
            return value.strip()
        if isinstance(value, list) and value:
            parts = [str(item).strip() for item in value if str(item).strip()]
            if parts:
                return " ".join(parts)
    return ""


def normalize_context(row: dict) -> str:
    context = first_nonempty(row, ("context", "passage", "document", "article", "input_context"))
    if context:
        return context

    for key in ("documents", "paragraphs", "ctxs", "retrieved_docs"):
        value = row.get(key)
        if not isinstance(value, list):
            continue
        parts = []
        for item in value:
            if isinstance(item, str):
                parts.append(item)
            elif isinstance(item, dict):
                parts.append(first_nonempty(item, ("text", "context", "paragraph", "contents", "passage")))
        context = " ".join(part for part in parts if part.strip())
        if context.strip():
            return context.strip()
    return ""


def normalize_answer(row: dict) -> str:
    for key in ("gold_answer", "answer", "answers", "output", "target"):
        value = row.get(key)
        if isinstance(value, str) and value.strip():
            return value.strip()
        if isinstance(value, list) and value:
            parts = []
            for item in value:
                if isinstance(item, str):
                    parts.append(item)
                elif isinstance(item, dict):
                    parts.extend(str(v) for v in item.values() if isinstance(v, str))
                else:
                    parts.append(str(item))
            answer = " | ".join(part.strip() for part in parts if part.strip())
            if answer:
                return answer
    return ""


def normalize_term(term: str) -> str:
    term = term.lower().strip("'\"")
    if len(term) > 4 and term.endswith("ies"):
        return term[:-3] + "y"
    for suffix in ("ing", "ed", "es", "s"):
        if len(term) > len(suffix) + 3 and term.endswith(suffix):
            return term[: -len(suffix)]
    return term


def tokenize(text: str) -> List[str]:
    return [tok for tok in (normalize_term(t) for t in TOKEN_RE.findall(text)) if tok and tok not in STOPWORDS]


def lexical_similarity(question: str, answer: str, sentence: str) -> float:
    q_tokens = set(tokenize(question))
    a_tokens = set(tokenize(answer))
    s_tokens = set(tokenize(sentence))
    if not s_tokens:
        return 0.0
    q_overlap = len(q_tokens & s_tokens) / max(len(q_tokens), 1)
    a_overlap = len(a_tokens & s_tokens) / max(len(a_tokens), 1) if a_tokens else 0.0
    entity_bonus = 0.05 if re.search(r"\b[A-Z][A-Za-z0-9-]{2,}\b|\d+(?:\.\d+)?%?", sentence) else 0.0
    return min(1.0, 0.65 * q_overlap + 0.30 * a_overlap + entity_bonus)


def find_complete_qwen_snapshot(cache_dir: str) -> Optional[Path]:
    root = Path(cache_dir) / "models--Qwen--Qwen3-Embedding-8B" / "snapshots"
    if not root.exists():
        return None
    for snapshot_dir in root.glob("*"):
        if not snapshot_dir.is_dir():
            continue
        try:
            if all((snapshot_dir / name).is_file() and (snapshot_dir / name).stat().st_size > 0 for name in QWEN_REQUIRED_FILES):
                return snapshot_dir
        except OSError:
            continue
    return None


def build_qwen_scorer(cache_dir: str, device: str) -> ContextAwareCompressor:
    snapshot = find_complete_qwen_snapshot(cache_dir)
    encoder_source = str(snapshot) if snapshot is not None else "lightweight_lexical_fallback"
    return ContextAwareCompressor(
        encoder_dir=encoder_source,
        encoder_cache_dir=cache_dir,
        device=device,
        use_attention_probe=False,
        use_task_descriptor=False,
        use_sentence_dynamics=False,
        enable_second_stage=False,
        allow_heuristic_fallback=True,
    )


def row_to_ratio_input(row: dict, scorer: Optional[ContextAwareCompressor], scorer_name: str, min_sentences: int) -> Optional[dict]:
    question = first_nonempty(row, ("question", "query", "input", "prompt"))
    context = normalize_context(row)
    answer = normalize_answer(row)
    if not question or not context:
        return None

    sentences = split_sentences(context)
    if len(sentences) < min_sentences:
        return None

    row_similarities = row.get("similarities")
    if scorer_name == "existing" and isinstance(row_similarities, list) and len(row_similarities) == len(sentences):
        similarities = [float(score) for score in row_similarities]
    elif scorer_name == "qwen":
        if scorer is None:
            raise ValueError("qwen scorer requested but scorer was not initialized")
        _, score_dict, _ = scorer.score_context(question, context)
        similarities = [float(score) for score in score_dict["semantic_similarities"]]
    else:
        similarities = [lexical_similarity(question, answer, sentence) for sentence in sentences]

    source_id = str(row.get("id") or row.get("_id") or row.get("qid") or row.get("source_id") or len(context))
    out = {
        "id": source_id,
        "question": question,
        "context": context,
        "similarities": similarities,
        "metadata": {
            "source_dataset": row.get("dataset") or row.get("source_dataset") or "english_cqr_or_benchmark",
            "similarity_source": scorer_name,
            "sentence_count": len(sentences),
        },
    }
    if answer:
        out["gold_answer"] = answer
    if row.get("positive_sentence"):
        out["positive_sentence"] = row["positive_sentence"]
    if row.get("supporting_sentences"):
        out["supporting_sentences"] = row["supporting_sentences"]
    return out


def main() -> None:
    parser = argparse.ArgumentParser(description="Build English target-ratio pseudo-label inputs from CQR or benchmark JSONL.")
    parser.add_argument("--input_jsonl", action="append", required=True)
    parser.add_argument("--output_jsonl", type=str, required=True)
    parser.add_argument("--similarity_source", choices=["qwen", "lexical", "existing"], default="qwen")
    parser.add_argument("--cache_dir", type=str, default=default_hf_cache_dir())
    parser.add_argument("--device", type=str, default="cpu")
    parser.add_argument("--min_sentences", type=int, default=3)
    parser.add_argument("--limit", type=int, default=0)
    args = parser.parse_args()

    scorer = None
    if args.similarity_source == "qwen":
        scorer = build_qwen_scorer(args.cache_dir, args.device)
        if scorer.encoder_runtime != "transformers":
            print("warning: Qwen snapshot was not available; using lightweight lexical fallback")
            args.similarity_source = "lexical"

    out_rows = []
    for input_path in args.input_jsonl:
        for row in load_jsonl(Path(input_path)):
            out_row = row_to_ratio_input(
                row=row,
                scorer=scorer,
                scorer_name=args.similarity_source,
                min_sentences=args.min_sentences,
            )
            if out_row is not None:
                out_rows.append(out_row)
                if args.limit > 0 and len(out_rows) >= args.limit:
                    break
        if args.limit > 0 and len(out_rows) >= args.limit:
            break

    write_jsonl(Path(args.output_jsonl), out_rows)
    print(f"saved: {args.output_jsonl}")
    print(f"rows: {len(out_rows)}")


if __name__ == "__main__":
    main()

