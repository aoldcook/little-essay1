"""Dependency-free .env loading.

Secrets are read from the process environment. This helper only populates the
environment from a local .env file; it never logs, prints, or returns secret
values.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Iterable, Optional

ENV_FILENAME = ".env"


def _candidate_dirs(start: Optional[Path] = None) -> Iterable[Path]:
    here = (start or Path.cwd()).resolve()
    yield here
    yield from here.parents
    project_root = Path(__file__).resolve().parent.parent
    yield project_root
    yield from project_root.parents


def find_env_file(start: Optional[Path] = None) -> Optional[Path]:
    seen = set()
    for directory in _candidate_dirs(start):
        if directory in seen:
            continue
        seen.add(directory)
        candidate = directory / ENV_FILENAME
        if candidate.is_file():
            return candidate
    return None


def load_env(path: Optional[Path] = None, override: bool = False) -> Optional[Path]:
    """Load KEY=VALUE pairs from a .env file into os.environ.

    Existing environment variables win unless `override=True`, so an explicitly
    exported variable always beats a stale file.
    """
    env_path = Path(path) if path else find_env_file()
    if env_path is None or not env_path.is_file():
        return None

    for raw_line in env_path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, _, value = line.partition("=")
        key = key.strip()
        if key.startswith("export "):
            key = key[len("export ") :].strip()
        value = value.strip().strip('"').strip("'")
        if not key:
            continue
        if override or key not in os.environ:
            os.environ[key] = value
    return env_path


def require_env(name: str, hint: str = "") -> str:
    """Fetch a required environment variable or raise with actionable guidance."""
    load_env()
    value = os.environ.get(name, "").strip()
    if not value:
        suffix = f"\n{hint}" if hint else ""
        raise RuntimeError(
            f"Required environment variable {name} is not set.\n"
            f"Create a .env file next to the repository root (see .env.example) "
            f"or export it in your shell.{suffix}"
        )
    return value
