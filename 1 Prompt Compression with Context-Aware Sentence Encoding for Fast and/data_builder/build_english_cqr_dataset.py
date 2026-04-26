from __future__ import annotations

import argparse
import json
import random
import re
from pathlib import Path
from typing import Iterable, List, Sequence


TOKEN_RE = re.compile(r"[A-Za-z][A-Za-z0-9]*(?:-[A-Za-z0-9]+)*|\d+(?:\.\d+)?%?")
STOPWORDS = {
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
SUPPORTED_DATASETS = {
    "longbench_hotpotqa",
    "longbench_2wikimqa",
    "longbench_musique",
    "longbench_multifieldqa_en",
    "longbench_qasper",
    "longbench_narrativeqa",
    "hotpotqa",
    "2wikimultihopqa",
    "musique",
    "qasper",
    "natural_questions",
    "triviaqa",
}


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


def token_overlap(a: str, b: str) -> float:
    a_tokens = set(tokenize(a))
    b_tokens = set(tokenize(b))
    if not a_tokens or not b_tokens:
        return 0.0
    return len(a_tokens & b_tokens) / max(len(a_tokens), 1)


def token_jaccard(a: str, b: str) -> float:
    a_tokens = set(tokenize(a))
    b_tokens = set(tokenize(b))
    if not a_tokens or not b_tokens:
        return 0.0
    return len(a_tokens & b_tokens) / max(len(a_tokens | b_tokens), 1)


def split_sentences(text: str) -> List[str]:
    stripped = re.sub(r"\s+", " ", text.strip())
    if not stripped:
        return []

    sentences: List[str] = []
    current: List[str] = []
    length = len(stripped)
    abbreviations = {"dr", "mr", "mrs", "ms", "prof", "inc", "ltd", "fig", "e.g", "i.e", "vs", "u.s", "u.k"}

    for idx, ch in enumerate(stripped):
        current.append(ch)
        if ch not in ".!?;":
            continue
        prev_char = stripped[idx - 1] if idx > 0 else ""
        next_char = stripped[idx + 1] if idx + 1 < length else ""
        if ch == ".":
            if prev_char.isdigit() and next_char.isdigit():
                continue
            prefix = "".join(current).strip().split()[-1].rstrip(".").lower() if current else ""
            if prefix in abbreviations or (next_char and next_char.islower()):
                continue
        sentence = "".join(current).strip()
        if sentence:
            sentences.append(sentence)
        current = []

    tail = "".join(current).strip()
    if tail:
        sentences.append(tail)
    return sentences


def read_json_or_jsonl(path: Path) -> List[dict]:
    text = path.read_text(encoding="utf-8-sig")
    stripped = text.lstrip()
    if not stripped:
        return []
    if stripped[0] == "[":
        data = json.loads(text)
        if not isinstance(data, list):
            raise ValueError(f"expected a JSON list in {path}")
        return [row for row in data if isinstance(row, dict)]

    rows = []
    for line in text.splitlines():
        line = line.strip()
        if line:
            row = json.loads(line)
            if isinstance(row, dict):
                rows.append(row)
    return rows


def first_nonempty(row: dict, keys: Sequence[str]) -> str:
    for key in keys:
        value = row.get(key)
        if isinstance(value, str) and value.strip():
            return value.strip()
        if isinstance(value, list) and value:
            joined = " ".join(str(item) for item in value if str(item).strip())
            if joined.strip():
                return joined.strip()
    return ""


def normalize_answer(row: dict) -> str:
    for key in ("answer", "answers", "gold_answer", "output", "target"):
        value = row.get(key)
        if isinstance(value, str) and value.strip():
            return value.strip()
        if isinstance(value, list) and value:
            flattened = []
            for item in value:
                if isinstance(item, str):
                    flattened.append(item)
                elif isinstance(item, dict):
                    flattened.extend(str(v) for v in item.values() if isinstance(v, str))
                else:
                    flattened.append(str(item))
            answer = " | ".join(item.strip() for item in flattened if item.strip())
            if answer:
                return answer
    return ""


def normalize_context(row: dict) -> str:
    context = first_nonempty(row, ("context", "passage", "document", "documents", "article", "input_context"))
    if context:
        return context

    for key in ("paragraphs", "ctxs", "retrieved_docs"):
        value = row.get(key)
        if isinstance(value, list):
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


def normalize_supports(row: dict) -> List[str]:
    supports = []
    for key in ("supporting_sentences", "evidence", "evidences", "supporting_facts", "support"):
        value = row.get(key)
        if isinstance(value, str) and value.strip():
            supports.append(value.strip())
        elif isinstance(value, list):
            for item in value:
                if isinstance(item, str) and item.strip():
                    supports.append(item.strip())
                elif isinstance(item, list) and item:
                    supports.append(" ".join(str(x) for x in item if str(x).strip()))
                elif isinstance(item, dict):
                    support = first_nonempty(item, ("sentence", "text", "fact", "evidence", "paragraph"))
                    if support:
                        supports.append(support)
    return supports


def sentence_matches_support(sentence: str, supports: Sequence[str]) -> bool:
    sentence_norm = re.sub(r"\s+", " ", sentence.strip().lower())
    for support in supports:
        support_norm = re.sub(r"\s+", " ", support.strip().lower())
        if not support_norm:
            continue
        if support_norm in sentence_norm or sentence_norm in support_norm:
            return True
        if token_jaccard(sentence_norm, support_norm) >= 0.55:
            return True
    return False


def score_sentence(question: str, answer: str, sentence: str, supports: Sequence[str]) -> float:
    support_bonus = 0.35 if sentence_matches_support(sentence, supports) else 0.0
    q_overlap = token_overlap(question, sentence)
    answer_overlap = token_overlap(answer, sentence) if answer else 0.0
    numeric_bonus = 0.08 if re.search(r"\d", sentence) and re.search(r"\d", question + " " + answer) else 0.0
    entity_bonus = 0.05 if re.search(r"\b[A-Z][A-Za-z0-9-]{2,}\b", sentence) else 0.0
    return min(1.0, support_bonus + 0.35 * q_overlap + 0.35 * answer_overlap + numeric_bonus + entity_bonus)


def choose_positive_sentences(
    question: str,
    answer: str,
    sentences: Sequence[str],
    supports: Sequence[str],
    max_positives: int,
    min_positive_score: float,
) -> List[str]:
    scored = [(score_sentence(question, answer, sentence, supports), idx, sentence) for idx, sentence in enumerate(sentences)]
    positives = [sentence for score, _, sentence in scored if score >= min_positive_score]
    if not positives and scored:
        positives = [max(scored, key=lambda item: item[0])[2]]
    return positives[:max_positives]


def choose_negative_sentences(
    question: str,
    answer: str,
    sentences: Sequence[str],
    positives: Sequence[str],
    max_negatives: int,
) -> List[str]:
    positive_set = {sentence.strip() for sentence in positives}
    candidates = []
    for idx, sentence in enumerate(sentences):
        if sentence.strip() in positive_set:
            continue
        q_overlap = token_overlap(question, sentence)
        answer_overlap = token_overlap(answer, sentence) if answer else 0.0
        hard_negative_score = q_overlap - 0.25 * answer_overlap
        candidates.append((hard_negative_score, idx, sentence))
    candidates.sort(key=lambda item: (item[0], -len(item[2])), reverse=True)
    return [sentence for _, _, sentence in candidates[:max_negatives]]


def normalize_row(row: dict, dataset_name: str, row_index: int, args: argparse.Namespace) -> List[dict]:
    question = first_nonempty(row, ("question", "query", "input", "prompt"))
    context = normalize_context(row)
    answer = normalize_answer(row)
    if not question or not context:
        return []

    sentences = split_sentences(context)
    if len(sentences) < args.min_sentences:
        return []

    context_token_count = len(TOKEN_RE.findall(context))
    if context_token_count < args.min_context_tokens:
        return []
    if args.max_context_tokens > 0 and context_token_count > args.max_context_tokens:
        kept = []
        token_total = 0
        for sentence in sentences:
            sent_tokens = len(TOKEN_RE.findall(sentence))
            if token_total + sent_tokens > args.max_context_tokens:
                break
            kept.append(sentence)
            token_total += sent_tokens
        sentences = kept
        context = " ".join(sentences)

    supports = normalize_supports(row)
    positives = choose_positive_sentences(
        question=question,
        answer=answer,
        sentences=sentences,
        supports=supports,
        max_positives=args.max_positives,
        min_positive_score=args.min_positive_score,
    )
    if not positives:
        return []

    negatives = choose_negative_sentences(
        question=question,
        answer=answer,
        sentences=sentences,
        positives=positives,
        max_negatives=args.max_negatives,
    )
    if len(negatives) < args.min_negatives:
        return []

    source_id = str(row.get("id") or row.get("_id") or row.get("qid") or f"{dataset_name}_{row_index:06d}")
    out_rows = []
    for pos_idx, positive in enumerate(positives):
        out_rows.append(
            {
                "id": f"{source_id}::pos{pos_idx}",
                "source_id": source_id,
                "dataset": dataset_name,
                "question": question,
                "context": context,
                "answer": answer,
                "positive_sentence": positive,
                "supporting_sentences": positives,
                "negative_sentences": negatives,
                "metadata": {
                    "context_tokens": context_token_count,
                    "source_dataset": dataset_name,
                    "construction": "english_cqr_answer_support_overlap_hard_negative",
                },
            }
        )
    return out_rows


def write_jsonl(path: Path, rows: Iterable[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def split_rows(rows: List[dict], dev_ratio: float, test_ratio: float, seed: int) -> tuple[List[dict], List[dict], List[dict]]:
    rng = random.Random(seed)
    rows = rows[:]
    rng.shuffle(rows)
    n = len(rows)
    test_n = int(round(n * test_ratio))
    dev_n = int(round(n * dev_ratio))
    test = rows[:test_n]
    dev = rows[test_n : test_n + dev_n]
    train = rows[test_n + dev_n :]
    return train, dev, test


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build English CQR-style sentence ranking data for prompt compression.")
    parser.add_argument("--input_file", action="append", required=True, help="Raw JSON/JSONL file. Repeat for multiple files.")
    parser.add_argument("--dataset_name", action="append", required=True, help="Dataset name for each input_file.")
    parser.add_argument("--output_dir", type=str, required=True)
    parser.add_argument("--min_context_tokens", type=int, default=80)
    parser.add_argument("--max_context_tokens", type=int, default=8192)
    parser.add_argument("--min_sentences", type=int, default=3)
    parser.add_argument("--max_positives", type=int, default=3)
    parser.add_argument("--max_negatives", type=int, default=6)
    parser.add_argument("--min_negatives", type=int, default=2)
    parser.add_argument("--min_positive_score", type=float, default=0.18)
    parser.add_argument("--dev_ratio", type=float, default=0.05)
    parser.add_argument("--test_ratio", type=float, default=0.05)
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if len(args.input_file) != len(args.dataset_name):
        raise ValueError("--input_file and --dataset_name must have the same count")

    all_rows: List[dict] = []
    summary = []
    for input_file, dataset_name in zip(args.input_file, args.dataset_name):
        if dataset_name not in SUPPORTED_DATASETS:
            print(f"warning: dataset_name={dataset_name} is not in the curated list; continuing anyway")
        raw_rows = read_json_or_jsonl(Path(input_file))
        converted = []
        for row_index, row in enumerate(raw_rows):
            converted.extend(normalize_row(row, dataset_name, row_index, args))
        all_rows.extend(converted)
        summary.append({"dataset": dataset_name, "input_rows": len(raw_rows), "cqr_rows": len(converted)})

    train, dev, test = split_rows(all_rows, args.dev_ratio, args.test_ratio, args.seed)
    output_dir = Path(args.output_dir)
    write_jsonl(output_dir / "train.jsonl", train)
    write_jsonl(output_dir / "dev.jsonl", dev)
    write_jsonl(output_dir / "test.jsonl", test)
    write_jsonl(output_dir / "all.jsonl", all_rows)

    summary_obj = {
        "total_cqr_rows": len(all_rows),
        "train_rows": len(train),
        "dev_rows": len(dev),
        "test_rows": len(test),
        "datasets": summary,
        "recommended_sources": [
            "LongBench: hotpotqa, 2wikimqa, musique, multifieldqa_en, qasper, narrativeqa",
            "HotpotQA, 2WikiMultiHopQA, MuSiQue, Qasper, Natural Questions, TriviaQA",
        ],
    }
    (output_dir / "build_summary.json").write_text(json.dumps(summary_obj, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(summary_obj, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()


