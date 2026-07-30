"""Disk-cached, concurrent reader calls for label generation.

Reader-grounded span labels need one call per (question, candidate context).
For the 5k pool that is ~13k calls, so two things are mandatory:

  * A disk cache. Label generation gets interrupted, re-run with a different
    threshold, or resumed; re-paying for identical calls is pure waste. The
    cache key includes the model id, so switching readers never reads stale
    answers from a different model.
  * Concurrency. Serially at ~0.6 s/call this is over two hours; at 8 workers
    it is under twenty minutes.

Determinism is preserved: the reader runs at temperature 0, and a cache hit
returns exactly what that model returned before, so a re-run reproduces the
same labels.
"""

from __future__ import annotations

import hashlib
import json
import threading
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

from evaluation.reader_client import QwenReader


def request_key(model: str, question: str, context: str) -> str:
    """Stable cache key. Model id is part of the key on purpose."""
    # "\x1f" (unit separator) cannot occur in the normalised fields, so distinct
    # (model, question, context) triples can never concatenate to the same key.
    payload = "\x1f".join(
        [model, " ".join(question.split()), " ".join(context.split())]
    )
    return hashlib.sha1(payload.encode("utf-8")).hexdigest()


@dataclass
class CacheStats:
    hits: int = 0
    misses: int = 0
    failures: int = 0
    lock: threading.Lock = field(default_factory=threading.Lock, repr=False)

    def to_dict(self) -> Dict[str, object]:
        total = self.hits + self.misses
        return {
            "cache_hits": self.hits,
            "cache_misses": self.misses,
            "failures": self.failures,
            "hit_rate": round(self.hits / total, 4) if total else None,
        }


class CachedBatchReader:
    """Answer many (question, context) pairs, caching to disk and running in parallel."""

    def __init__(
        self,
        reader: QwenReader,
        cache_path: Path | str,
        concurrency: int = 8,
    ):
        self.reader = reader
        self.model = reader.config.model
        self.cache_path = Path(cache_path)
        self.concurrency = max(1, int(concurrency))
        self.stats = CacheStats()
        self._cache: Dict[str, str] = {}
        self._write_lock = threading.Lock()
        self._load_cache()

    def _load_cache(self) -> None:
        if not self.cache_path.is_file():
            return
        with self.cache_path.open("r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    record = json.loads(line)
                except json.JSONDecodeError:
                    continue  # tolerate a truncated final line from a hard kill
                if record.get("key") and record.get("model") == self.model:
                    self._cache[record["key"]] = record.get("answer", "")

    def _persist(self, key: str, answer: str) -> None:
        with self._write_lock:
            self.cache_path.parent.mkdir(parents=True, exist_ok=True)
            with self.cache_path.open("a", encoding="utf-8") as f:
                f.write(
                    json.dumps(
                        {"key": key, "model": self.model, "answer": answer},
                        ensure_ascii=False,
                    )
                    + "\n"
                )

    def answer_many(self, requests: Sequence[Tuple[str, str]]) -> List[str]:
        """Answer each (question, context) pair, in the order given.

        Failed calls yield "" rather than raising: one unreachable span must not
        abort a multi-hour labelling run. Failures are counted in stats, and the
        caller must treat an empty answer as "no signal", never as a wrong answer.
        """
        keys = [request_key(self.model, q, c) for q, c in requests]
        results: List[Optional[str]] = [None] * len(requests)

        pending: List[int] = []
        for idx, key in enumerate(keys):
            cached = self._cache.get(key)
            if cached is not None:
                results[idx] = cached
                with self.stats.lock:
                    self.stats.hits += 1
            else:
                pending.append(idx)

        def run(idx: int) -> None:
            question, context = requests[idx]
            out = self.reader.answer(question=question, context=context)
            answer = str(out.get("answer") or "")
            ok = bool(out.get("ok"))
            results[idx] = answer
            with self.stats.lock:
                self.stats.misses += 1
                if not ok:
                    self.stats.failures += 1
            if ok:
                # Only cache successes: caching a failure would freeze a transient
                # network error into the dataset permanently.
                self._cache[keys[idx]] = answer
                self._persist(keys[idx], answer)

        if pending:
            with ThreadPoolExecutor(max_workers=self.concurrency) as pool:
                list(pool.map(run, pending))

        return [r if r is not None else "" for r in results]

    def provenance(self) -> Dict[str, object]:
        return {
            "label_reader_model": self.model,
            "concurrency": self.concurrency,
            "cache_path": str(self.cache_path),
            **self.stats.to_dict(),
        }
