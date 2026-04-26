from __future__ import annotations

import argparse
import json
import random
import re
import zipfile
from collections import Counter
from pathlib import Path
from typing import Iterable, List, Sequence

from datasets import load_dataset
from huggingface_hub import hf_hub_download

from data_builder.build_english_cqr_dataset import (
    TOKEN_RE,
    choose_negative_sentences,
    choose_positive_sentences,
    normalize_answer,
    split_sentences,
    token_overlap,
)
from data_builder.cqr_split_utils import dedupe_rows, save_jsonl, split_rows_by_group, summarize_splits


DEFAULT_LONGBENCH_DATASETS = ("hotpotqa", "2wikimqa", "multifieldqa_en", "qasper")
PROJECT_ROOT = Path(__file__).resolve().parents[1]
WORKSPACE_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_CACHE_DIR = WORKSPACE_ROOT / "hf_cache"


def token_count(text: str) -> int:
    return len(TOKEN_RE.findall(text))


def clean_text(text: str) -> str:
    return re.sub(r"\s+", " ", str(text or "").strip())


def with_metadata(row: dict, dataset: str, quality: str, construction: str) -> dict:
    metadata = dict(row.get("metadata") or {})
    metadata.update(
        {
            "source_dataset": dataset,
            "quality": quality,
            "construction": construction,
            "context_tokens": token_count(str(row.get("context") or "")),
        }
    )
    row["metadata"] = metadata
    row["dataset"] = dataset
    return row


def valid_cqr_row(row: dict, min_negatives: int) -> bool:
    return bool(
        str(row.get("question") or "").strip()
        and str(row.get("context") or "").strip()
        and str(row.get("positive_sentence") or "").strip()
        and len(row.get("negative_sentences") or []) >= min_negatives
    )


def sample_rows(rows: Sequence[dict], limit: int, rng: random.Random) -> List[dict]:
    if limit <= 0:
        return []
    rows = list(rows)
    rng.shuffle(rows)
    return rows[:limit]


def cqr_negatives_from_samples(segments: Sequence[str], neg_samples: object, max_negatives: int) -> List[str]:
    negatives: List[str] = []
    if isinstance(neg_samples, list):
        for item in neg_samples:
            if isinstance(item, int) and 0 <= item < len(segments):
                negatives.append(clean_text(segments[item]))
            elif isinstance(item, str):
                if item.isdigit() and int(item) < len(segments):
                    negatives.append(clean_text(segments[int(item)]))
                else:
                    negatives.append(clean_text(item))
            if len(negatives) >= max_negatives:
                break
    return [sentence for sentence in negatives if sentence][:max_negatives]


def build_cqr_rows(limit: int, cache_dir: Path, seed: int, max_negatives: int, min_negatives: int) -> List[dict]:
    if limit <= 0:
        return []
    raw = load_dataset("deadcode99/CQR", split="train", cache_dir=str(cache_dir))
    rng = random.Random(seed)
    indices = list(range(len(raw)))
    rng.shuffle(indices)

    rows: List[dict] = []
    for raw_idx in indices:
        item = raw[int(raw_idx)]
        segments = [clean_text(segment) for segment in item.get("segments", []) if clean_text(segment)]
        pos_idx = int(item.get("pos_sent_idx", -1))
        if not segments or pos_idx < 0 or pos_idx >= len(segments):
            continue

        positive = segments[pos_idx]
        negatives = cqr_negatives_from_samples(segments, item.get("neg_samples"), max_negatives)
        if len(negatives) < min_negatives:
            negatives = choose_negative_sentences(
                question=str(item.get("question") or ""),
                answer=str(item.get("answer") or ""),
                sentences=segments,
                positives=[positive],
                max_negatives=max_negatives,
            )
        row = {
            "id": f"cqr::{raw_idx}",
            "source_id": str(raw_idx),
            "question": clean_text(item.get("question")),
            "context": " ".join(segments),
            "answer": clean_text(item.get("answer")),
            "positive_sentence": positive,
            "supporting_sentences": [positive],
            "negative_sentences": negatives[:max_negatives],
        }
        row = with_metadata(row, "cqr_official", "gold", "cpc_official_cqr")
        if valid_cqr_row(row, min_negatives):
            rows.append(row)
        if len(rows) >= limit:
            break
    return rows


def hotpot_context_sentences(item: dict) -> tuple[str, List[tuple[str, int, str]]]:
    context = item.get("context") or {}
    titles = context.get("title") or []
    sentence_groups = context.get("sentences") or []
    flat: List[tuple[str, int, str]] = []
    parts: List[str] = []
    for title, sentences in zip(titles, sentence_groups):
        clean_title = clean_text(title)
        if clean_title:
            parts.append(f"{clean_title}.")
        for sent_idx, sentence in enumerate(sentences or []):
            clean_sentence = clean_text(sentence)
            if clean_sentence:
                flat.append((clean_title, sent_idx, clean_sentence))
                parts.append(clean_sentence)
    return " ".join(parts), flat


def hotpot_support_sentences(item: dict, flat_sentences: Sequence[tuple[str, int, str]]) -> List[str]:
    facts = item.get("supporting_facts") or {}
    titles = facts.get("title") or []
    sent_ids = facts.get("sent_id") or []
    support_keys = {(clean_text(title), int(sent_id)) for title, sent_id in zip(titles, sent_ids)}
    return [sentence for title, sent_idx, sentence in flat_sentences if (title, sent_idx) in support_keys]


def build_hotpot_rows(limit: int, cache_dir: Path, seed: int, max_negatives: int, min_negatives: int) -> List[dict]:
    if limit <= 0:
        return []
    raw = load_dataset("hotpotqa/hotpot_qa", "distractor", split="train", cache_dir=str(cache_dir))
    rng = random.Random(seed)
    indices = list(range(len(raw)))
    rng.shuffle(indices)

    rows: List[dict] = []
    for raw_idx in indices:
        item = raw[int(raw_idx)]
        context, flat = hotpot_context_sentences(item)
        sentences = [sentence for _, _, sentence in flat]
        supports = hotpot_support_sentences(item, flat)
        if len(sentences) < 3 or not supports:
            continue

        question = clean_text(item.get("question"))
        answer = clean_text(item.get("answer"))
        negatives = choose_negative_sentences(
            question=question,
            answer=answer,
            sentences=sentences,
            positives=supports,
            max_negatives=max_negatives,
        )
        if len(negatives) < min_negatives:
            continue

        for pos_idx, positive in enumerate(supports):
            row = {
                "id": f"hotpotqa::{item.get('id', raw_idx)}::pos{pos_idx}",
                "source_id": str(item.get("id") or raw_idx),
                "question": question,
                "context": context,
                "answer": answer,
                "positive_sentence": positive,
                "supporting_sentences": supports,
                "negative_sentences": negatives[:max_negatives],
            }
            row = with_metadata(row, "hotpotqa_supporting_facts", "gold", "hotpotqa_sentence_supporting_facts")
            if valid_cqr_row(row, min_negatives):
                rows.append(row)
                if len(rows) >= limit:
                    return rows
    return rows


def iter_longbench_rows(cache_dir: Path, dataset_name: str) -> Iterable[dict]:
    zip_path = hf_hub_download(
        repo_id="THUDM/LongBench",
        repo_type="dataset",
        filename="data.zip",
        cache_dir=str(cache_dir),
    )
    member = f"data/{dataset_name}.jsonl"
    with zipfile.ZipFile(zip_path) as zf:
        with zf.open(member) as f:
            for raw_line in f:
                line = raw_line.decode("utf-8").strip()
                if line:
                    yield json.loads(line)


def build_longbench_rows(
    limit: int,
    cache_dir: Path,
    seed: int,
    max_negatives: int,
    min_negatives: int,
    datasets: Sequence[str],
) -> List[dict]:
    if limit <= 0:
        return []
    rng = random.Random(seed)
    per_dataset_limit = max(1, limit // max(len(datasets), 1))
    rows: List[dict] = []

    for dataset_name in datasets:
        raw_rows = list(iter_longbench_rows(cache_dir, dataset_name))
        rng.shuffle(raw_rows)
        dataset_rows: List[dict] = []
        for raw_idx, item in enumerate(raw_rows):
            question = clean_text(item.get("input"))
            context = clean_text(item.get("context"))
            answer = normalize_answer(item)
            sentences = split_sentences(context)
            if not question or len(sentences) < 3:
                continue

            positives = choose_positive_sentences(
                question=question,
                answer=answer,
                sentences=sentences,
                supports=[],
                max_positives=1,
                min_positive_score=0.16,
            )
            negatives = choose_negative_sentences(
                question=question,
                answer=answer,
                sentences=sentences,
                positives=positives,
                max_negatives=max_negatives,
            )
            if not positives or len(negatives) < min_negatives:
                continue

            row = {
                "id": f"longbench_{dataset_name}::{item.get('_id', raw_idx)}",
                "source_id": str(item.get("_id") or raw_idx),
                "question": question,
                "context": context,
                "answer": answer,
                "positive_sentence": positives[0],
                "supporting_sentences": positives,
                "negative_sentences": negatives[:max_negatives],
            }
            row = with_metadata(row, f"longbench_{dataset_name}", "silver", "longbench_answer_overlap_structure_adaptation")
            if valid_cqr_row(row, min_negatives):
                dataset_rows.append(row)
            if len(dataset_rows) >= per_dataset_limit:
                break
        rows.extend(dataset_rows)
        if len(rows) >= limit:
            return rows[:limit]
    return rows[:limit]


def question_from_sentence(sentence: str) -> tuple[str, str] | None:
    sentence = clean_text(sentence)
    if not sentence or token_count(sentence) < 8:
        return None

    definition = re.search(
        r"^([A-Z][A-Za-z0-9 .,'&()/-]{2,80}?)\s+(is|are|was|were|refers to|means)\s+(.+?)[.!?]?$",
        sentence,
    )
    if definition:
        subject = clean_text(definition.group(1)).strip(" ,")
        answer = clean_text(definition.group(3)).strip(" .")
        if 2 <= token_count(subject) <= 10 and token_count(answer) >= 4:
            return f"What is {subject}?", answer

    year = re.search(r"\b(18|19|20)\d{2}\b", sentence)
    entity = re.search(r"\b[A-Z][A-Za-z0-9-]+(?:\s+[A-Z][A-Za-z0-9-]+){0,4}\b", sentence)
    if year and entity:
        entity_text = clean_text(entity.group(0))
        return f"What year is mentioned in relation to {entity_text}?", year.group(0)

    if re.search(r"\b(because|due to|therefore|led to|caused|resulted in|as a result)\b", sentence.lower()):
        if entity:
            return f"What causal information is given about {clean_text(entity.group(0))}?", sentence
        return "What causal information is described?", sentence

    if entity:
        entity_text = clean_text(entity.group(0))
        return f"What information is given about {entity_text}?", sentence

    return None


def build_wikitext_rows(limit: int, cache_dir: Path, seed: int, max_negatives: int, min_negatives: int) -> List[dict]:
    if limit <= 0:
        return []
    raw = load_dataset("Salesforce/wikitext", "wikitext-103-raw-v1", split="train", cache_dir=str(cache_dir))
    rng = random.Random(seed)
    start_indices = list(range(min(len(raw), 80000)))
    rng.shuffle(start_indices)

    rows: List[dict] = []
    context_buffer: List[str] = []
    doc_id = 0
    seen_questions = set()

    def flush_context(buffer: Sequence[str], local_doc_id: int) -> None:
        if len(rows) >= limit:
            return
        context = " ".join(buffer)
        sentences = split_sentences(context)
        if len(sentences) < 4:
            return
        sentence_order = list(range(len(sentences)))
        rng.shuffle(sentence_order)
        for pos_idx in sentence_order:
            positive = sentences[pos_idx]
            generated = question_from_sentence(positive)
            if generated is None:
                continue
            question, answer = generated
            key = (question, positive)
            if key in seen_questions:
                continue
            negatives = choose_negative_sentences(
                question=question,
                answer=answer,
                sentences=sentences,
                positives=[positive],
                max_negatives=max_negatives,
            )
            if len(negatives) < min_negatives:
                continue
            seen_questions.add(key)
            row = {
                "id": f"wikitext_pseudo::{local_doc_id}::pos{pos_idx}",
                "source_id": str(local_doc_id),
                "question": question,
                "context": context,
                "answer": answer,
                "positive_sentence": positive,
                "supporting_sentences": [positive],
                "negative_sentences": negatives[:max_negatives],
            }
            row = with_metadata(row, "wikitext103_pseudo_cqr", "silver", "teacher_free_wikitext_question_positive_generation")
            if valid_cqr_row(row, min_negatives):
                rows.append(row)
                if len(rows) >= limit:
                    return

    for raw_idx in start_indices:
        text = clean_text(raw[int(raw_idx)].get("text"))
        if not text:
            continue
        if text.startswith("=") and context_buffer:
            flush_context(context_buffer, doc_id)
            doc_id += 1
            context_buffer = []
            if len(rows) >= limit:
                break
        if text.startswith("="):
            continue
        context_buffer.append(text)
        if len(context_buffer) >= 8:
            flush_context(context_buffer, doc_id)
            doc_id += 1
            context_buffer = []
            if len(rows) >= limit:
                break

    if context_buffer and len(rows) < limit:
        flush_context(context_buffer, doc_id)
    return rows[:limit]


def fill_shortfall(
    rows: List[dict],
    target_total: int,
    cache_dir: Path,
    seed: int,
    max_negatives: int,
    min_negatives: int,
) -> List[dict]:
    shortfall = target_total - len(rows)
    if shortfall <= 0:
        return rows[:target_total]

    extra = build_hotpot_rows(shortfall, cache_dir, seed + 1009, max_negatives, min_negatives)
    existing_ids = {row["id"] for row in rows}
    for row in extra:
        if row["id"] not in existing_ids:
            rows.append(row)
            existing_ids.add(row["id"])
        if len(rows) >= target_total:
            break
    return rows[:target_total]


def write_examples(path: Path, rows: Sequence[dict], limit: int = 8) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    examples = []
    for row in rows[:limit]:
        examples.append(
            {
                "dataset": row.get("dataset"),
                "quality": (row.get("metadata") or {}).get("quality"),
                "question": row.get("question"),
                "positive_sentence": row.get("positive_sentence"),
                "negative_sentences": row.get("negative_sentences", [])[:2],
            }
        )
    path.write_text(json.dumps(examples, ensure_ascii=False, indent=2), encoding="utf-8")


def summarize_all(rows: Sequence[dict], splits: dict, deduped_removed: int, args: argparse.Namespace) -> dict:
    token_counts = [token_count(str(row.get("context") or "")) for row in rows]
    return {
        "target_total": args.target_total,
        "actual_total": len(rows),
        "deduped_removed": deduped_removed,
        "source_counts": dict(Counter(str(row.get("dataset") or "") for row in rows)),
        "quality_counts": dict(Counter(str((row.get("metadata") or {}).get("quality") or "") for row in rows)),
        "split_summary": summarize_splits(splits),
        "context_tokens": {
            "min": min(token_counts) if token_counts else 0,
            "max": max(token_counts) if token_counts else 0,
            "avg": round(sum(token_counts) / max(len(token_counts), 1), 2),
        },
        "build_args": vars(args),
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build a 5k mixed English CQR training dataset.")
    parser.add_argument("--output_dir", type=str, default=str(PROJECT_ROOT / "data_builder" / "english_cqr_mixed_5k"))
    parser.add_argument("--cache_dir", type=str, default=str(DEFAULT_CACHE_DIR))
    parser.add_argument("--target_total", type=int, default=5000)
    parser.add_argument("--cqr_rows", type=int, default=2000)
    parser.add_argument("--hotpotqa_rows", type=int, default=1800)
    parser.add_argument("--longbench_rows", type=int, default=400)
    parser.add_argument("--wikitext_rows", type=int, default=800)
    parser.add_argument("--longbench_datasets", nargs="*", default=list(DEFAULT_LONGBENCH_DATASETS))
    parser.add_argument("--max_negatives", type=int, default=4)
    parser.add_argument("--min_negatives", type=int, default=2)
    parser.add_argument("--train_ratio", type=float, default=0.90)
    parser.add_argument("--dev_ratio", type=float, default=0.05)
    parser.add_argument("--test_ratio", type=float, default=0.05)
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    cache_dir = Path(args.cache_dir)
    output_dir = Path(args.output_dir)
    rng = random.Random(args.seed)

    source_rows = {
        "cqr_official": build_cqr_rows(args.cqr_rows, cache_dir, args.seed + 1, args.max_negatives, args.min_negatives),
        "hotpotqa_supporting_facts": build_hotpot_rows(
            args.hotpotqa_rows, cache_dir, args.seed + 2, args.max_negatives, args.min_negatives
        ),
        "longbench_structure": build_longbench_rows(
            args.longbench_rows,
            cache_dir,
            args.seed + 3,
            args.max_negatives,
            args.min_negatives,
            args.longbench_datasets,
        ),
        "wikitext103_pseudo_cqr": build_wikitext_rows(
            args.wikitext_rows, cache_dir, args.seed + 4, args.max_negatives, args.min_negatives
        ),
    }

    all_rows: List[dict] = []
    for rows in source_rows.values():
        all_rows.extend(rows)
    rng.shuffle(all_rows)
    all_rows, deduped_removed = dedupe_rows(all_rows)
    all_rows = fill_shortfall(all_rows, args.target_total, cache_dir, args.seed + 5, args.max_negatives, args.min_negatives)
    rng.shuffle(all_rows)

    splits = split_rows_by_group(
        all_rows,
        train_ratio=args.train_ratio,
        dev_ratio=args.dev_ratio,
        test_ratio=args.test_ratio,
        seed=args.seed,
        stratify_by_dataset=True,
        stratify_by_quality=True,
    )

    output_dir.mkdir(parents=True, exist_ok=True)
    save_jsonl(output_dir / "all.jsonl", all_rows)
    for split_name, split_rows in splits.items():
        save_jsonl(output_dir / f"{split_name}.jsonl", split_rows)
    write_examples(output_dir / "examples.json", all_rows)

    summary = summarize_all(all_rows, splits, deduped_removed, args)
    summary["source_build_counts_before_dedupe"] = {key: len(value) for key, value in source_rows.items()}
    (output_dir / "build_summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
