"""Frozen downstream reader LLM client (Alibaba Bailian / DashScope).

Addresses EVAL_VALIDITY_AUDIT.md finding C2: compression quality must be judged
by whether a real reader can answer, not by token overlap.

Design constraints:
  * The API key is read from the DASHSCOPE_API_KEY environment variable only.
    It is never accepted as a CLI argument, never logged, never written to any
    results file. The manifest records only whether it was present.
  * The reader is FROZEN and deterministic (temperature=0) so that differences
    between runs are attributable to the compressor, not to sampling.
  * Prompting is identical across all compression methods. This is essential:
    a per-method prompt would confound the comparison.
"""

from __future__ import annotations

import os
import random
import time
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence

from evaluation.env_loader import load_env, require_env

DEFAULT_BASE_URL = "https://dashscope.aliyuncs.com/compatible-mode/v1"
DEFAULT_MODEL = "qwen3.6-flash"

# One fixed prompt for every method and every compression ratio.
SYSTEM_PROMPT = (
    "You are a precise extractive question answering system. "
    "Answer using only the provided context. "
    "Reply with the shortest exact answer span - a word, number, or short phrase. "
    "Do not explain. If the context does not contain the answer, reply exactly: unanswerable"
)
USER_TEMPLATE = "Context:\n{context}\n\nQuestion: {question}\nAnswer:"


class ReaderError(RuntimeError):
    """Raised when the reader cannot be reached or configured."""


@dataclass
class ReaderConfig:
    model: str = DEFAULT_MODEL
    base_url: str = DEFAULT_BASE_URL
    temperature: float = 0.0
    max_tokens: int = 64
    timeout: float = 60.0
    max_retries: int = 5
    retry_base_delay: float = 1.5

    @classmethod
    def from_env(cls, **overrides) -> "ReaderConfig":
        load_env()
        config = cls(
            model=os.environ.get("READER_MODEL") or DEFAULT_MODEL,
            base_url=os.environ.get("DASHSCOPE_BASE_URL") or DEFAULT_BASE_URL,
        )
        for key, value in overrides.items():
            if value is not None and hasattr(config, key):
                setattr(config, key, value)
        return config

    def public_dict(self) -> Dict[str, object]:
        """Config safe to embed in results (contains no secrets)."""
        return {
            "model": self.model,
            "base_url": self.base_url,
            "temperature": self.temperature,
            "max_tokens": self.max_tokens,
            "system_prompt": SYSTEM_PROMPT,
        }


@dataclass
class ReaderStats:
    num_calls: int = 0
    num_retries: int = 0
    num_failures: int = 0
    prompt_tokens: int = 0
    completion_tokens: int = 0
    total_latency_s: float = 0.0
    errors: List[str] = field(default_factory=list)

    def to_dict(self) -> Dict[str, object]:
        return {
            "num_calls": self.num_calls,
            "num_retries": self.num_retries,
            "num_failures": self.num_failures,
            "prompt_tokens": self.prompt_tokens,
            "completion_tokens": self.completion_tokens,
            "avg_latency_s": (
                self.total_latency_s / self.num_calls if self.num_calls else None
            ),
            "sample_errors": self.errors[:5],
        }


class QwenReader:
    """Thin, deterministic wrapper over the OpenAI-compatible DashScope endpoint."""

    def __init__(self, config: Optional[ReaderConfig] = None):
        self.config = config or ReaderConfig.from_env()
        self.stats = ReaderStats()

        api_key = require_env(
            "DASHSCOPE_API_KEY",
            hint=(
                "Get a key from https://bailian.console.aliyun.com/ and put it in "
                ".env as DASHSCOPE_API_KEY=... (see .env.example)."
            ),
        )
        try:
            from openai import OpenAI
        except ImportError as exc:  # pragma: no cover
            raise ReaderError(
                "The 'openai' package is required for the downstream reader. "
                "Install it with: pip install -r requirements.txt"
            ) from exc

        self._client = OpenAI(
            api_key=api_key,
            base_url=self.config.base_url,
            timeout=self.config.timeout,
        )

    def build_messages(self, question: str, context: str) -> List[Dict[str, str]]:
        return [
            {"role": "system", "content": SYSTEM_PROMPT},
            {
                "role": "user",
                "content": USER_TEMPLATE.format(context=context.strip(), question=question.strip()),
            },
        ]

    def answer(self, question: str, context: str) -> Dict[str, object]:
        """Return {'answer', 'latency_s', 'ok', 'error'} for one QA instance."""
        messages = self.build_messages(question, context)
        last_error: Optional[str] = None

        for attempt in range(self.config.max_retries):
            start = time.monotonic()
            try:
                response = self._client.chat.completions.create(
                    model=self.config.model,
                    messages=messages,
                    temperature=self.config.temperature,
                    max_tokens=self.config.max_tokens,
                )
                latency = time.monotonic() - start
                text = (response.choices[0].message.content or "").strip()

                self.stats.num_calls += 1
                self.stats.total_latency_s += latency
                usage = getattr(response, "usage", None)
                if usage is not None:
                    self.stats.prompt_tokens += int(getattr(usage, "prompt_tokens", 0) or 0)
                    self.stats.completion_tokens += int(
                        getattr(usage, "completion_tokens", 0) or 0
                    )
                return {
                    "answer": text,
                    "latency_s": latency,
                    "ok": True,
                    "error": None,
                    "prompt_tokens": int(getattr(usage, "prompt_tokens", 0) or 0) if usage else None,
                }
            except Exception as exc:  # noqa: BLE001 - surface provider errors verbatim
                last_error = f"{type(exc).__name__}: {exc}"
                self.stats.num_retries += 1
                if attempt == self.config.max_retries - 1:
                    break
                # Exponential backoff with jitter; DashScope rate-limits under load.
                delay = self.config.retry_base_delay ** (attempt + 1)
                time.sleep(delay + random.uniform(0, 0.5))

        self.stats.num_failures += 1
        if last_error and last_error not in self.stats.errors:
            self.stats.errors.append(last_error)
        return {"answer": "", "latency_s": None, "ok": False, "error": last_error}

    def smoke_test(self) -> Dict[str, object]:
        """Verify credentials and the model id before a long, costly sweep.

        Model ids change over time; running this first turns a 3-hour failure
        into a 3-second one.
        """
        result = self.answer(
            question="What colour is the sky in the context?",
            context="The sky in this example is green.",

        )
        return {
            "reachable": bool(result["ok"]),
            "model": self.config.model,
            "base_url": self.config.base_url,
            "raw_answer": result["answer"],
            "error": result["error"],
        }
