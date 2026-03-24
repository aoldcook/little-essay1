from __future__ import annotations

import argparse
import glob
import hashlib
import json
import re
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple


SENTENCE_SPLIT_RE = re.compile(r"(?<=[。！？.!?])\s*")
ZH_RE = re.compile(r"[\u4e00-\u9fff]")
LATIN_RE = re.compile(r"[a-z0-9]+")
QUESTION_STOPWORDS = {
    "什么",
    "为什么",
    "为何",
    "如何",
    "哪些",
    "哪个",
    "多少",
    "是否",
    "关于",
    "请问",
    "the",
    "a",
    "an",
    "of",
    "to",
    "for",
    "in",
    "on",
    "with",
    "and",
    "or",
    "is",
    "are",
    "what",
    "why",
    "how",
}
LOW_QUALITY_QUESTION_PREFIXES = (
    "根据上文",
    "根据这段",
    "根据文中",
    "文中提到",
    "上文提到",
    "这段文字",
)
LOW_QUALITY_QUESTION_PATTERNS = (
    "是不是",
    "是否",
    "对吗",
    "吗？",
    "吗?",
)
DEFINITION_HINTS = (
    "是什么",
    "指什么",
    "定义是什么",
    "概念是什么",
)


def split_sentences(text: str) -> List[str]:
    return [s.strip() for s in SENTENCE_SPLIT_RE.split(text.strip()) if s.strip()]


def normalize_text(text: str) -> str:
    text = text.strip().lower()
    text = re.sub(r"\s+", "", text)
    return text


def tokenize_mixed(text: str) -> List[str]:
    text = normalize_text(text)
    zh_chars = ZH_RE.findall(text)
    latin_words = LATIN_RE.findall(text)
    tokens = zh_chars + latin_words
    return [t for t in tokens if t and t not in QUESTION_STOPWORDS]


def overlap_ratio(a: str, b: str) -> float:
    ta = tokenize_mixed(a)
    tb = tokenize_mixed(b)
    if not ta or not tb:
        return 0.0
    ca = Counter(ta)
    cb = Counter(tb)
    overlap = sum(min(count, cb.get(tok, 0)) for tok, count in ca.items())
    return overlap / max(len(ta), 1)


def stable_hash(text: str) -> str:
    return hashlib.md5(normalize_text(text).encode("utf-8")).hexdigest()


def find_exact_sentence(context_sentences: Sequence[str], target: str) -> Optional[str]:
    target_norm = normalize_text(target)
    for sentence in context_sentences:
        if normalize_text(sentence) == target_norm:
            return sentence
    return None


def save_jsonl(path: Path, rows: Iterable[dict]) -> None:
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def load_jsonl(path: Path) -> List[dict]:
    rows: List[dict] = []
    with path.open("r", encoding="utf-8") as f:
        for line_no, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError as exc:
                raise ValueError(f"invalid json in {path} line {line_no}") from exc
    return rows


@dataclass
class CleanerConfig:
    min_sentences: int = 5
    max_sentences: int = 8
    min_negative_count: int = 2
    max_negative_count: int = 3
    min_supporting_count: int = 2
    max_questions_per_context: int = 3
    max_positive_answer_overlap: float = 0.82


class CqrCleaner:
    def __init__(self, config: CleanerConfig):
        self.config = config
        self.stats: Counter[str] = Counter()

    def clean_rows(self, rows: Sequence[dict]) -> List[dict]:
        validated: List[dict] = []
        seen_question_context = set()
        context_usage: Counter[str] = Counter()

        for row in rows:
            cleaned, reason = self.validate_and_normalize(row)
            if cleaned is None:
                self.stats[reason] += 1
                continue

            qctx_key = (
                normalize_text(cleaned["question"]),
                stable_hash(cleaned["context"]),
            )
            if qctx_key in seen_question_context:
                self.stats["drop_duplicate_question_context"] += 1
                continue

            context_hash = stable_hash(cleaned["context"])
            if context_usage[context_hash] >= self.config.max_questions_per_context:
                self.stats["drop_context_quota"] += 1
                continue

            seen_question_context.add(qctx_key)
            context_usage[context_hash] += 1
            validated.append(cleaned)

        self.stats["kept_after_validation"] = len(validated)
        return validated

    def validate_and_normalize(self, row: dict) -> Tuple[Optional[dict], str]:
        required = {"question", "context", "positive_sentence", "negative_sentences"}
        if not required.issubset(row):
            return None, "drop_missing_required_fields"

        question = str(row["question"]).strip()
        context = str(row["context"]).strip()
        positive_sentence = str(row["positive_sentence"]).strip()
        negative_sentences = row.get("negative_sentences") or []
        supporting_sentences = row.get("supporting_sentences") or []
        answer = str(row.get("answer", "")).strip()
        question_type = str(row.get("question_type", "")).strip()

        if not question or not context or not positive_sentence or not isinstance(negative_sentences, list):
            return None, "drop_invalid_required_values"

        if self.looks_low_quality_question(question):
            return None, "drop_low_quality_question"

        context_sentences = split_sentences(context)
        if not (self.config.min_sentences <= len(context_sentences) <= self.config.max_sentences):
            return None, "drop_sentence_count"

        matched_positive = find_exact_sentence(context_sentences, positive_sentence)
        if matched_positive is None:
            return None, "drop_positive_not_in_context"

        matched_supporting: List[str] = []
        for sent in supporting_sentences:
            matched = find_exact_sentence(context_sentences, str(sent))
            if matched is not None:
                matched_supporting.append(matched)
        matched_supporting = self._deduplicate_preserve_order(matched_supporting)

        if supporting_sentences:
            if len(matched_supporting) < self.config.min_supporting_count:
                return None, "drop_supporting_count"
            if normalize_text(matched_positive) not in {normalize_text(s) for s in matched_supporting}:
                return None, "drop_positive_not_in_supporting"

        matched_negatives: List[str] = []
        for sent in negative_sentences:
            matched = find_exact_sentence(context_sentences, str(sent))
            if matched is not None:
                matched_negatives.append(matched)
        matched_negatives = self._deduplicate_preserve_order(matched_negatives)

        if len(matched_negatives) < self.config.min_negative_count:
            return None, "drop_negative_count"

        if supporting_sentences:
            supporting_norm = {normalize_text(s) for s in matched_supporting}
            if any(normalize_text(s) in supporting_norm for s in matched_negatives):
                return None, "drop_negative_supporting_overlap"

        high_overlap_neg = [
            sent for sent in matched_negatives if overlap_ratio(question, sent) >= 0.75
        ]
        if len(high_overlap_neg) >= max(1, len(matched_negatives) - 1):
            return None, "drop_negative_too_related"

        if supporting_sentences:
            if len(set(normalize_text(s) for s in matched_supporting)) < self.config.min_supporting_count:
                return None, "drop_supporting_unique_count"

        if answer and self.is_positive_answer_like(matched_positive, answer):
            return None, "drop_positive_answers_question"

        cleaned = {
            "question": question,
            "context": context,
            "positive_sentence": matched_positive,
            "negative_sentences": matched_negatives[: self.config.max_negative_count],
        }
        if question_type:
            cleaned["question_type"] = question_type
        if answer:
            cleaned["answer"] = answer
        if matched_supporting:
            cleaned["supporting_sentences"] = matched_supporting
        return cleaned, "keep"

    def looks_low_quality_question(self, question: str) -> bool:
        question_norm = normalize_text(question)
        if len(question_norm) < 6:
            return True
        if any(question.startswith(prefix) for prefix in LOW_QUALITY_QUESTION_PREFIXES):
            return True
        if any(pattern in question for pattern in LOW_QUALITY_QUESTION_PATTERNS):
            return True
        if any(hint in question for hint in DEFINITION_HINTS):
            return True
        return False

    def is_positive_answer_like(self, positive_sentence: str, answer: str) -> bool:
        pos_norm = normalize_text(positive_sentence)
        ans_norm = normalize_text(answer)
        if not ans_norm:
            return False
        if ans_norm in pos_norm and len(ans_norm) >= 6:
            return True
        return overlap_ratio(answer, positive_sentence) >= self.config.max_positive_answer_overlap

    @staticmethod
    def _deduplicate_preserve_order(items: Sequence[str]) -> List[str]:
        seen = set()
        out = []
        for item in items:
            key = normalize_text(item)
            if key in seen:
                continue
            seen.add(key)
            out.append(item)
        return out


def split_by_context_hash(
    rows: Sequence[dict],
    train_size: int,
    dev_size: int,
    test_size: int,
) -> Dict[str, List[dict]]:
    target_sizes = {
        "train": train_size,
        "dev": dev_size,
        "test": test_size,
    }
    buckets = {name: [] for name in target_sizes}
    group_map: Dict[str, List[dict]] = defaultdict(list)
    for row in rows:
        group_map[stable_hash(row["context"])].append(row)

    grouped = sorted(group_map.values(), key=lambda group: stable_hash(group[0]["context"]))
    for group in grouped:
        remaining = {
            name: target_sizes[name] - len(buckets[name])
            for name in target_sizes
        }
        candidates = sorted(
            target_sizes,
            key=lambda name: (remaining[name] < len(group), -remaining[name], name),
        )
        chosen = candidates[0]
        buckets[chosen].extend(group)

    return buckets


def export_training_format(rows: Sequence[dict]) -> List[dict]:
    return [
        {
            "question": row["question"],
            "context": row["context"],
            "positive_sentence": row["positive_sentence"],
            "negative_sentences": row["negative_sentences"],
        }
        for row in rows
    ]


def resolve_paths(project_root: Path, input_glob: str) -> List[Path]:
    if Path(input_glob).is_absolute():
        matched = glob.glob(input_glob)
    else:
        matched = glob.glob(str(project_root / input_glob))
    return sorted(Path(path) for path in matched)


def load_all_rows(project_root: Path, input_glob: str) -> List[dict]:
    paths = resolve_paths(project_root, input_glob)
    if not paths:
        raise FileNotFoundError(f"no files matched input_glob={input_glob}")
    rows: List[dict] = []
    for path in paths:
        rows.extend(load_jsonl(path))
    return rows


def build_summary(
    raw_count: int,
    cleaned_rows: Sequence[dict],
    splits: Dict[str, List[dict]],
    stats: Counter[str],
) -> dict:
    question_type_counter = Counter(row.get("question_type", "unknown") for row in cleaned_rows)
    return {
        "raw_count": raw_count,
        "cleaned_count": len(cleaned_rows),
        "split_sizes": {name: len(rows) for name, rows in splits.items()},
        "question_type_distribution": dict(question_type_counter),
        "drop_stats": dict(stats),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Clean and split CQR dataset JSONL files.")
    parser.add_argument(
        "--input_glob",
        type=str,
        default="data_builder/raw_batches/*.jsonl",
        help="Glob pattern relative to project root for raw batch files.",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="data_builder/cleaned",
        help="Directory for cleaned outputs, relative to project root by default.",
    )
    parser.add_argument("--train_size", type=int, default=8000)
    parser.add_argument("--dev_size", type=int, default=1000)
    parser.add_argument("--test_size", type=int, default=1000)
    parser.add_argument("--min_sentences", type=int, default=5)
    parser.add_argument("--max_sentences", type=int, default=8)
    parser.add_argument("--max_questions_per_context", type=int, default=3)
    args = parser.parse_args()

    project_root = Path(__file__).resolve().parent.parent
    rows = load_all_rows(project_root, args.input_glob)
    config = CleanerConfig(
        min_sentences=args.min_sentences,
        max_sentences=args.max_sentences,
        max_questions_per_context=args.max_questions_per_context,
    )
    cleaner = CqrCleaner(config)
    cleaned_rows = cleaner.clean_rows(rows)
    splits = split_by_context_hash(
        cleaned_rows,
        train_size=args.train_size,
        dev_size=args.dev_size,
        test_size=args.test_size,
    )

    output_dir = Path(args.output_dir)
    if not output_dir.is_absolute():
        output_dir = project_root / output_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    save_jsonl(output_dir / "cqr_cleaned_enriched.jsonl", cleaned_rows)
    save_jsonl(output_dir / "cqr_train.jsonl", export_training_format(splits["train"]))
    save_jsonl(output_dir / "cqr_dev.jsonl", export_training_format(splits["dev"]))
    save_jsonl(output_dir / "cqr_test.jsonl", export_training_format(splits["test"]))

    summary = build_summary(len(rows), cleaned_rows, splits, cleaner.stats)
    (output_dir / "cleaning_summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
