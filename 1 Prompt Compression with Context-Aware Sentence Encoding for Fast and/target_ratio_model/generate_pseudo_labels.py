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


DEFAULT_RATIOS = [0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]
STOPWORDS = {
    "a", "an", "and", "are", "as", "at", "be", "by", "can", "could", "do", "does", "did",
    "for", "from", "how", "in", "is", "it", "of", "on", "or", "should", "that", "the", "to",
    "was", "were", "what", "when", "where", "which", "who", "why", "with",
}
TOKEN_RE = re.compile(r"[A-Za-z][A-Za-z0-9]*(?:-[A-Za-z0-9]+)*|\d+(?:\.\d+)?%?")


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


def budget_token_len(text: str) -> int:
    return max(1, len(TOKEN_RE.findall(text)))


def compress_with_scores(context: str, sentences: List[str], similarities: List[float], target_ratio: float) -> str:
    scores_and_sentences = list(zip(similarities, sentences))
    scores_and_sentences.sort(key=lambda x: x[0], reverse=True)

    total_len = sum(budget_token_len(s) for s in sentences)
    target_len = max(1, int(total_len * target_ratio))

    selected = []
    cur_len = 0
    for score, sent in scores_and_sentences:
        sent_len = budget_token_len(sent)
        if cur_len + sent_len <= target_len:
            selected.append(sent)
            cur_len += sent_len

    if not selected and scores_and_sentences:
        selected = [scores_and_sentences[0][1]]

    selected_set = set(selected)
    ordered = [s for s in sentences if s in selected_set]
    return " ".join(ordered)


def normalize_text(text: str) -> str:
    text = text.strip().lower()
    return re.sub(r"\s+", " ", text)


def normalize_term(term: str) -> str:
    term = term.lower().strip("'\"")
    if len(term) > 4 and term.endswith("ies"):
        return term[:-3] + "y"
    for suffix in ("ing", "ed", "es", "s"):
        if len(term) > len(suffix) + 3 and term.endswith(suffix):
            return term[: -len(suffix)]
    return term


def tokenize_mixed(text: str) -> List[str]:
    text = normalize_text(text)
    toks = [normalize_term(token) for token in TOKEN_RE.findall(text)]
    return [token for token in toks if token and token not in STOPWORDS]


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


def evidence_coverage_score(compressed_context: str, evidence_texts: List[str]) -> float:
    refs = [ref for ref in evidence_texts if isinstance(ref, str) and ref.strip()]
    if not refs:
        return 0.0

    comp_tokens = tokenize_mixed(compressed_context)
    comp_counts: Dict[str, int] = {}
    for token in comp_tokens:
        comp_counts[token] = comp_counts.get(token, 0) + 1

    scores = []
    for ref in refs:
        ref_tokens = tokenize_mixed(ref)
        if not ref_tokens:
            continue
        ref_counts: Dict[str, int] = {}
        for token in ref_tokens:
            ref_counts[token] = ref_counts.get(token, 0) + 1
        overlap = sum(min(count, comp_counts.get(token, 0)) for token, count in ref_counts.items())
        recall = overlap / max(len(ref_tokens), 1)
        scores.append(max(recall, char_f1(compressed_context, ref)))

    return sum(scores) / len(scores) if scores else 0.0


def collect_evidence_texts(row: dict) -> List[str]:
    texts: List[str] = []
    for key in ("positive_sentence", "supporting_sentences", "evidence", "evidences", "support"):
        value = row.get(key)
        if isinstance(value, str) and value.strip():
            texts.append(value.strip())
        elif isinstance(value, list):
            for item in value:
                if isinstance(item, str) and item.strip():
                    texts.append(item.strip())
                elif isinstance(item, dict):
                    for nested_key in ("sentence", "text", "fact", "evidence", "paragraph"):
                        nested = item.get(nested_key)
                        if isinstance(nested, str) and nested.strip():
                            texts.append(nested.strip())
                            break
    if not texts and isinstance(row.get("gold_answer"), str) and row["gold_answer"].strip():
        texts.append(row["gold_answer"].strip())
    return texts


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


class LLMClient:
    def __init__(
        self,
        base_url: str,
        api_key: str,
        model: str,
        api_style: str = "auto",
        temperature: float = 0.0,
        timeout: int = 120,
        max_retries: int = 8,
    ):
        self.base_url = base_url.rstrip("/")
        self.api_key = api_key
        self.model = model
        self.api_style = self._resolve_api_style(api_style, self.base_url)
        self.temperature = temperature
        self.timeout = timeout
        self.max_retries = max_retries

    @staticmethod
    def _resolve_api_style(api_style: str, base_url: str) -> str:
        if api_style != "auto":
            return api_style
        lowered = base_url.lower()
        if lowered.endswith("/responses") or "ark.cn-beijing.volces.com/api/v3" in lowered:
            return "ark_responses"
        return "openai_chat"

    def _build_request(self, system_prompt: str, user_prompt: str) -> Tuple[str, dict]:
        if self.api_style == "ark_responses":
            url = self.base_url if self.base_url.endswith("/responses") else self.base_url + "/responses"
            payload = {
                "model": self.model,
                "stream": False,
                "input": [
                    {
                        "role": "system",
                        "content": [{"type": "input_text", "text": system_prompt}],
                    },
                    {
                        "role": "user",
                        "content": [{"type": "input_text", "text": user_prompt}],
                    },
                ],
            }
            return url, payload

        url = self.base_url + "/chat/completions"
        payload = {
            "model": self.model,
            "temperature": self.temperature,
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
        }
        return url, payload

    def _extract_content(self, data: dict) -> str:
        if isinstance(data, dict) and data.get("error"):
            err = data.get("error") or {}
            err_code = err.get("code")
            err_text = json.dumps(err, ensure_ascii=False)
            raise RuntimeError(f"ProviderError {err_code}: {err_text[:500]}")

        if self.api_style == "ark_responses":
            output_text = data.get("output_text")
            if isinstance(output_text, str) and output_text.strip():
                return output_text.strip()
            for item in reversed(data.get("output", [])):
                if item.get("type") != "message":
                    continue
                for block in item.get("content", []):
                    text = block.get("text") or block.get("output_text") or block.get("content")
                    if isinstance(text, str) and text.strip():
                        return text.strip()
            raise RuntimeError(f"No text found in Ark responses payload: {json.dumps(data, ensure_ascii=False)[:500]}")

        return data["choices"][0]["message"]["content"].strip()

    def chat(self, system_prompt: str, user_prompt: str) -> str:
        url, payload = self._build_request(system_prompt, user_prompt)
        body = json.dumps(payload).encode("utf-8")
        for attempt in range(self.max_retries):
            req = request.Request(
                url,
                data=body,
                headers={
                    "Content-Type": "application/json",
                    "Authorization": f"Bearer {self.api_key}",
                },
                method="POST",
            )
            try:
                with request.urlopen(req, timeout=self.timeout) as resp:
                    data = json.loads(resp.read().decode("utf-8"))
                try:
                    return self._extract_content(data)
                except RuntimeError as exc:
                    message = str(exc)
                    retriable = any(token in message for token in ("429", "500", "502", "503", "504"))
                    if retriable and attempt + 1 < self.max_retries:
                        time.sleep(min(10 * (attempt + 1), 60))
                        continue
                    raise
            except error.HTTPError as e:
                body_text = e.read().decode("utf-8", errors="ignore")
                retriable = e.code in {429, 500, 502, 503, 504}
                if retriable and attempt + 1 < self.max_retries:
                    time.sleep(min(10 * (attempt + 1), 60))
                    continue
                raise RuntimeError(f"HTTPError {e.code}: {body_text[:500]}") from e
            except Exception as e:
                if attempt + 1 < self.max_retries:
                    time.sleep(min(10 * (attempt + 1), 60))
                    continue
                raise RuntimeError(f"request failed: {e}") from e

        raise RuntimeError("LLM request failed after retries")


ANSWER_SYSTEM_PROMPT = (
    "You are a careful question-answering assistant. Answer only from the given context; "
    "if the context is insufficient, state that the answer cannot be determined from the context."
)


def build_answer_prompt(question: str, context: str) -> str:
    return (
        f"Question: {question}\n\n"
        f"Context:\n{context}\n\n"
        "Give a concise, direct answer."
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
    client: Optional[LLMClient] = None,
    gold_answer: Optional[str] = None,
    full_answer: Optional[str] = None,
    evidence_texts: Optional[List[str]] = None,
    sleep_seconds: float = 0.0,
) -> Tuple[float, Dict]:
    sentences = split_sentences(context)
    if len(sentences) != len(similarities):
        raise ValueError(
            f"len(sentences)={len(sentences)} but len(similarities)={len(similarities)}; "
            "make sure similarities aligns with split_sentences(context)"
        )

    evidence_texts = evidence_texts or []
    if full_answer:
        teacher_answer = full_answer
    elif answer_mode == "evidence_coverage":
        teacher_answer = " | ".join(evidence_texts) if evidence_texts else extractive_demo_answer(question, context)
    elif answer_mode == "demo_extractive":
        teacher_answer = extractive_demo_answer(question, context)
    elif answer_mode in {"openai_compatible", "llm"}:
        if client is None:
            raise ValueError("llm mode requires client")
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
        elif answer_mode == "evidence_coverage":
            pred_answer = compressed_context
        else:
            pred_answer = client.chat(ANSWER_SYSTEM_PROMPT, build_answer_prompt(question, compressed_context))
            if sleep_seconds > 0:
                time.sleep(sleep_seconds)

        if answer_mode == "evidence_coverage":
            refs = evidence_texts or ([gold_answer] if gold_answer else [])
            quality_score = evidence_coverage_score(compressed_context, refs)
            quality_ref = "evidence_coverage"
        elif gold_answer:
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
    parser = argparse.ArgumentParser(description="Generate English label_ratio pseudo labels")
    parser.add_argument("--input_jsonl", type=str, required=True)
    parser.add_argument("--output_jsonl", type=str, required=True)
    parser.add_argument("--ratios", type=str, default=",".join(str(x) for x in DEFAULT_RATIOS))
    parser.add_argument("--threshold", type=float, default=0.82)
    parser.add_argument("--answer_mode", type=str, default="demo_extractive", choices=["demo_extractive", "evidence_coverage", "llm", "openai_compatible"])
    parser.add_argument("--judge_mode", type=str, default="char_f1", choices=["char_f1", "jaccard"])
    parser.add_argument("--base_url", type=str, default="")
    parser.add_argument("--api_key", type=str, default="")
    parser.add_argument("--model", type=str, default="")
    parser.add_argument("--api_style", type=str, default="auto", choices=["auto", "openai_chat", "ark_responses"])
    parser.add_argument("--sleep_seconds", type=float, default=0.0)
    parser.add_argument("--keep_debug", action="store_true")
    args = parser.parse_args()

    ratios = parse_ratios(args.ratios)
    rows = load_jsonl(Path(args.input_jsonl))

    client = None
    if args.answer_mode in {"openai_compatible", "llm"}:
        missing = [name for name, val in [("base_url", args.base_url), ("api_key", args.api_key), ("model", args.model)] if not val]
        if missing:
            raise ValueError(f"missing required args for llm: {missing}")
        client = LLMClient(
            base_url=args.base_url,
            api_key=args.api_key,
            model=args.model,
            api_style=args.api_style,
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
            evidence_texts=collect_evidence_texts(row),
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




