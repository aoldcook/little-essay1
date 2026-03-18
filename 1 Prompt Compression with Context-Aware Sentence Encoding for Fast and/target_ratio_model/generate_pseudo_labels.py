import argparse
import json
import math
import re
import sys
import time
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple
from urllib import request, error

from target_ratio_model.budget_features import split_sentences


DEFAULT_RATIOS = [0.2, 0.3, 0.4, 0.5, 0.6, 0.7]
STOPWORDS = {
    "的", "了", "和", "是", "在", "与", "及", "对", "中", "并", "或", "将", "把", "被", "就", "也", "而", "但", "且",
    "what", "why", "how", "the", "a", "an", "is", "are", "of", "to", "for", "in", "on", "with", "by", "and", "or",
}


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


def parse_ratios(s: str) -> List[float]:
    vals = [float(x.strip()) for x in s.split(",") if x.strip()]
    vals = sorted(set(vals))
    for v in vals:
        if not (0.0 < v <= 1.0):
            raise ValueError(f"invalid ratio: {v}")
    return vals


def compress_with_scores(context: str, sentences: List[str], similarities: List[float], target_ratio: float) -> str:
    scores_and_sentences = list(zip(similarities, sentences))
    scores_and_sentences.sort(key=lambda x: x[0], reverse=True)

    total_len = sum(len(s) for s in sentences)
    target_len = max(1, int(total_len * target_ratio))

    selected = []
    cur_len = 0
    for score, sent in scores_and_sentences:
        sent_len = len(sent)
        if cur_len + sent_len <= target_len:
            selected.append(sent)
            cur_len += sent_len

    if not selected and scores_and_sentences:
        selected = [scores_and_sentences[0][1]]

    selected_set = set(selected)
    ordered = [s for s in sentences if s in selected_set]
    return "".join(ordered)


def normalize_text(text: str) -> str:
    text = text.strip().lower()
    text = re.sub(r"\s+", "", text)
    return text


def tokenize_mixed(text: str) -> List[str]:
    text = normalize_text(text)
    zh_chars = re.findall(r"[\u4e00-\u9fff]", text)
    latin_words = re.findall(r"[a-z0-9]+", text)
    toks = zh_chars + latin_words
    return [t for t in toks if t and t not in STOPWORDS]


def char_f1(pred: str, ref: str) -> float:
    pred_toks = tokenize_mixed(pred)
    ref_toks = tokenize_mixed(ref)
    if not pred_toks or not ref_toks:
        return 0.0
    pred_counts: Dict[str, int] = {}
    ref_counts: Dict[str, int] = {}
    for t in pred_toks:
        pred_counts[t] = pred_counts.get(t, 0) + 1
    for t in ref_toks:
        ref_counts[t] = ref_counts.get(t, 0) + 1

    overlap = 0
    for t, c in pred_counts.items():
        overlap += min(c, ref_counts.get(t, 0))
    if overlap == 0:
        return 0.0
    precision = overlap / max(len(pred_toks), 1)
    recall = overlap / max(len(ref_toks), 1)
    return 2 * precision * recall / max(precision + recall, 1e-12)


def jaccard_sim(a: str, b: str) -> float:
    ta = set(tokenize_mixed(a))
    tb = set(tokenize_mixed(b))
    if not ta or not tb:
        return 0.0
    return len(ta & tb) / len(ta | tb)


def question_overlap_score(question: str, sentence: str) -> float:
    q = set(tokenize_mixed(question))
    s = set(tokenize_mixed(sentence))
    if not q or not s:
        return 0.0
    return len(q & s) / len(q)


def extractive_demo_answer(question: str, context: str) -> str:
    sentences = split_sentences(context)
    if not sentences:
        return ""
    scored = [(question_overlap_score(question, s), i, s) for i, s in enumerate(sentences)]
    scored.sort(key=lambda x: (x[0], -len(x[2])), reverse=True)
    top = [s for _, _, s in scored[:2]]
    return " ".join(top)


class OpenAICompatibleClient:
    def __init__(self, base_url: str, api_key: str, model: str, temperature: float = 0.0, timeout: int = 120):
        self.base_url = base_url.rstrip("/")
        self.api_key = api_key
        self.model = model
        self.temperature = temperature
        self.timeout = timeout

    def chat(self, system_prompt: str, user_prompt: str) -> str:
        url = self.base_url + "/chat/completions"
        payload = {
            "model": self.model,
            "temperature": self.temperature,
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
        }
        req = request.Request(
            url,
            data=json.dumps(payload).encode("utf-8"),
            headers={
                "Content-Type": "application/json",
                "Authorization": f"Bearer {self.api_key}",
            },
            method="POST",
        )
        try:
            with request.urlopen(req, timeout=self.timeout) as resp:
                data = json.loads(resp.read().decode("utf-8"))
        except error.HTTPError as e:
            body = e.read().decode("utf-8", errors="ignore")
            raise RuntimeError(f"HTTPError {e.code}: {body[:500]}") from e
        except Exception as e:
            raise RuntimeError(f"request failed: {e}") from e

        try:
            return data["choices"][0]["message"]["content"].strip()
        except Exception as e:
            raise RuntimeError(f"unexpected API response: {data}") from e


ANSWER_SYSTEM_PROMPT = (
    "你是一个严谨的问答助手。请只依据给定上下文回答问题；"
    "若上下文不足以支持确定答案，请明确回答‘无法从给定上下文确定’。"
)


def build_answer_prompt(question: str, context: str) -> str:
    return (
        f"问题：{question}\n\n"
        f"上下文：\n{context}\n\n"
        "请给出简洁、直接的答案。"
    )


def judge_consistency(pred_answer: str, teacher_answer: str, mode: str) -> float:
    if mode == "char_f1":
        return char_f1(pred_answer, teacher_answer)
    if mode == "jaccard":
        return jaccard_sim(pred_answer, teacher_answer)
    raise ValueError(f"unsupported judge mode: {mode}")


def choose_label_ratio(
    question: str,
    context: str,
    similarities: List[float],
    ratios: List[float],
    threshold: float,
    answer_mode: str,
    judge_mode: str,
    client: Optional[OpenAICompatibleClient] = None,
    gold_answer: Optional[str] = None,
    full_answer: Optional[str] = None,
    sleep_seconds: float = 0.0,
) -> Tuple[float, Dict]:
    sentences = split_sentences(context)
    if len(sentences) != len(similarities):
        raise ValueError(
            f"len(sentences)={len(sentences)} but len(similarities)={len(similarities)}; "
            "请保证 similarities 与 split_sentences(context) 一一对应"
        )

    if full_answer:
        teacher_answer = full_answer
    elif answer_mode == "demo_extractive":
        teacher_answer = extractive_demo_answer(question, context)
    elif answer_mode == "openai_compatible":
        if client is None:
            raise ValueError("openai_compatible mode requires client")
        teacher_answer = client.chat(ANSWER_SYSTEM_PROMPT, build_answer_prompt(question, context))
        if sleep_seconds > 0:
            time.sleep(sleep_seconds)
    else:
        raise ValueError(f"unsupported answer mode: {answer_mode}")

    debug_rows = []
    chosen_ratio = ratios[-1]

    for ratio in ratios:
        compressed_context = compress_with_scores(context, sentences, similarities, ratio)

        if answer_mode == "demo_extractive":
            pred_answer = extractive_demo_answer(question, compressed_context)
        else:
            pred_answer = client.chat(ANSWER_SYSTEM_PROMPT, build_answer_prompt(question, compressed_context))
            if sleep_seconds > 0:
                time.sleep(sleep_seconds)

        if gold_answer:
            quality_score = char_f1(pred_answer, gold_answer)
            quality_ref = "gold_answer"
        else:
            quality_score = judge_consistency(pred_answer, teacher_answer, judge_mode)
            quality_ref = "teacher_answer"

        debug_rows.append({
            "ratio": ratio,
            "compressed_char_len": len(compressed_context),
            "pred_answer": pred_answer,
            "score": quality_score,
            "score_ref": quality_ref,
        })

        if quality_score >= threshold:
            chosen_ratio = ratio
            break

    return chosen_ratio, {
        "teacher_answer": teacher_answer,
        "scan": debug_rows,
    }


def main():
    parser = argparse.ArgumentParser(description="自动生成 label_ratio 伪标签")
    parser.add_argument("--input_jsonl", type=str, required=True)
    parser.add_argument("--output_jsonl", type=str, required=True)
    parser.add_argument("--ratios", type=str, default=",".join(str(x) for x in DEFAULT_RATIOS))
    parser.add_argument("--threshold", type=float, default=0.82)
    parser.add_argument("--answer_mode", type=str, default="demo_extractive", choices=["demo_extractive", "openai_compatible"])
    parser.add_argument("--judge_mode", type=str, default="char_f1", choices=["char_f1", "jaccard"])
    parser.add_argument("--base_url", type=str, default="")
    parser.add_argument("--api_key", type=str, default="")
    parser.add_argument("--model", type=str, default="")
    parser.add_argument("--sleep_seconds", type=float, default=0.0)
    parser.add_argument("--keep_debug", action="store_true")
    args = parser.parse_args()

    ratios = parse_ratios(args.ratios)
    rows = load_jsonl(Path(args.input_jsonl))

    client = None
    if args.answer_mode == "openai_compatible":
        missing = [name for name, val in [("base_url", args.base_url), ("api_key", args.api_key), ("model", args.model)] if not val]
        if missing:
            raise ValueError(f"missing required args for openai_compatible: {missing}")
        client = OpenAICompatibleClient(
            base_url=args.base_url,
            api_key=args.api_key,
            model=args.model,
            temperature=0.0,
        )

    out_rows = []
    for idx, row in enumerate(rows, start=1):
        required = ["question", "context", "similarities"]
        missing = [k for k in required if k not in row]
        if missing:
            raise ValueError(f"row {idx} missing keys: {missing}")

        label_ratio, debug = choose_label_ratio(
            question=row["question"],
            context=row["context"],
            similarities=row["similarities"],
            ratios=ratios,
            threshold=args.threshold,
            answer_mode=args.answer_mode,
            judge_mode=args.judge_mode,
            client=client,
            gold_answer=row.get("gold_answer"),
            full_answer=row.get("full_answer"),
            sleep_seconds=args.sleep_seconds,
        )

        out_row = dict(row)
        out_row["label_ratio"] = label_ratio
        out_row["teacher_answer"] = debug["teacher_answer"]
        if args.keep_debug:
            out_row["debug_scan"] = debug["scan"]
        out_rows.append(out_row)

        chosen_debug = next((item for item in debug["scan"] if float(item["ratio"]) == float(label_ratio)), None)
        chosen_score = chosen_debug["score"] if chosen_debug else 0.0
        print(f"[{idx}/{len(rows)}] id={row.get('id', idx)} label_ratio={label_ratio:.2f} chosen_score={chosen_score:.4f}")
        sys.stdout.flush()

    save_jsonl(Path(args.output_jsonl), out_rows)
    print(f"saved: {args.output_jsonl}")


if __name__ == "__main__":
    main()
