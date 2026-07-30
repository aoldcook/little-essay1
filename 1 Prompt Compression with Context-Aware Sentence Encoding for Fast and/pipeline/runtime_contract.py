"""Encoder checkpoint loading contract and runtime provenance.

Purpose (see EVAL_VALIDITY_AUDIT.md findings C1 / H3):

The pipeline historically resolved a missing or broken encoder checkpoint to a
lexical token-overlap heuristic *silently*. Every runnable entry point therefore
produced plausible numbers that did not come from the trained context-aware
encoder at all. This module makes the backend an explicit, recorded, opt-in
decision so that no result can be misattributed to a model that never ran.

Three guarantees:

1. Requesting the lexical fallback is only possible with an explicit opt-in.
2. A checkpoint that is requested but unusable raises, it never degrades.
3. Whatever backend actually ran is recorded (with a checkpoint fingerprint)
   and is meant to be embedded in every results file.
"""

from __future__ import annotations

import hashlib
import platform
import sys
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Dict, List, Optional

# Canonical identifier for the non-neural lexical backend.
LEXICAL_FALLBACK_ID = "lexical://fallback"

# Legacy magic string kept so existing configs/CLI invocations remain parseable.
# It resolves to the same backend but is still gated by the opt-in flag.
LEGACY_LEXICAL_IDS = frozenset({"lightweight_lexical_fallback", LEXICAL_FALLBACK_ID})

# A directory is a usable context-aware encoder checkpoint only if it carries a
# config plus at least one recognised weight artifact.
ENCODER_CONFIG_FILENAME = "encoder_config.json"
WEIGHT_SUFFIXES = (".safetensors", ".bin", ".pt")

# Files above this size are fingerprinted by (name, size, head bytes) instead of
# a full digest, because a full pass over multi-GB shards is not worth the wall
# clock on every evaluation run. The mode used is always recorded.
FULL_HASH_MAX_BYTES = 256 * 1024 * 1024
_QUICK_HASH_HEAD_BYTES = 8 * 1024 * 1024


class EncoderContractError(RuntimeError):
    """Raised when the requested encoder backend cannot be honoured as-asked.

    This is deliberately fatal. Silently substituting a weaker backend is the
    exact failure mode that made previous results uninterpretable.
    """


@dataclass
class ResolvedEncoder:
    """The backend that will actually be used, after validation."""

    kind: str  # "context_aware_encoder" | "lexical_fallback"
    requested: str
    path: Optional[str] = None
    reason: str = ""

    @property
    def is_lexical(self) -> bool:
        return self.kind == "lexical_fallback"


@dataclass
class RuntimeProvenance:
    """Everything needed to attribute a number to the system that produced it."""

    encoder_kind: str
    encoder_requested: str
    encoder_path: Optional[str] = None
    encoder_runtime: str = ""
    lexical_fallback_used: bool = False
    fallback_reason: str = ""
    encoder_load_error: Optional[str] = None
    checkpoint_fingerprint: Dict[str, object] = field(default_factory=dict)
    span_model_dir: Optional[str] = None
    span_model_active: Optional[bool] = None
    budget_model_dir: Optional[str] = None
    python_version: str = field(default_factory=lambda: sys.version.split()[0])
    platform: str = field(default_factory=platform.platform)

    def to_dict(self) -> Dict[str, object]:
        return asdict(self)

    def system_label(self) -> str:
        """Short label safe to use as a results key.

        Any lexical-backed run is named so that it cannot be mistaken for the
        neural system in a table or a plot.
        """
        if self.lexical_fallback_used or self.encoder_kind == "lexical_fallback":
            return "LEXICAL-FALLBACK(not-the-neural-system)"
        return "context-aware-encoder"

    def assert_neural(self, context: str = "evaluation") -> None:
        """Fail loudly if this provenance does not describe the neural system."""
        if self.lexical_fallback_used or self.encoder_kind == "lexical_fallback":
            raise EncoderContractError(
                f"Refusing to report {context} results: the active backend is the "
                f"lexical fallback, not the trained context-aware encoder.\n"
                f"  requested       : {self.encoder_requested}\n"
                f"  fallback reason : {self.fallback_reason or 'explicitly requested'}\n"
                f"  load error      : {self.encoder_load_error or 'n/a'}\n\n"
                "Either supply a valid --encoder_dir checkpoint, or pass "
                "--allow_lexical_fallback to record this explicitly as a "
                "non-neural ablation baseline."
            )


def _iter_weight_files(path: Path) -> List[Path]:
    return sorted(
        p for p in path.rglob("*") if p.is_file() and p.suffix in WEIGHT_SUFFIXES
    )


def _digest_file(path: Path) -> Dict[str, object]:
    size = path.stat().st_size
    sha = hashlib.sha256()
    if size <= FULL_HASH_MAX_BYTES:
        with path.open("rb") as f:
            for chunk in iter(lambda: f.read(1024 * 1024), b""):
                sha.update(chunk)
        mode = "sha256_full"
    else:
        with path.open("rb") as f:
            sha.update(f.read(_QUICK_HASH_HEAD_BYTES))
        sha.update(str(size).encode("utf-8"))
        mode = "sha256_head8mb_plus_size"
    return {"name": path.name, "bytes": size, "digest": sha.hexdigest()[:32], "mode": mode}


def checkpoint_fingerprint(source: Optional[str]) -> Dict[str, object]:
    """Fingerprint a checkpoint directory so a result can be tied to weights."""
    if not source or source in LEGACY_LEXICAL_IDS:
        return {"kind": "lexical_fallback", "files": []}

    path = Path(source)
    if not path.exists():
        return {"kind": "missing", "path": str(path), "files": []}

    if path.is_file():
        return {"kind": "file", "path": str(path), "files": [_digest_file(path)]}

    files = [_digest_file(p) for p in _iter_weight_files(path)]
    combined = hashlib.sha256(
        "".join(f"{f['name']}:{f['digest']}" for f in files).encode("utf-8")
    ).hexdigest()[:32]
    return {
        "kind": "directory",
        "path": str(path),
        "num_weight_files": len(files),
        "combined_digest": combined,
        "files": files,
    }


def describe_checkpoint_problem(path: Path) -> Optional[str]:
    """Return a human-readable reason the path is not a usable checkpoint."""
    if not path.exists():
        return f"path does not exist: {path}"
    if not path.is_dir():
        return f"path is not a directory: {path}"
    if not (path / ENCODER_CONFIG_FILENAME).is_file():
        # A bare HuggingFace snapshot (config.json + weights) is also acceptable.
        if (path / "config.json").is_file() and _iter_weight_files(path):
            return None
        return (
            f"missing {ENCODER_CONFIG_FILENAME} (and no HuggingFace config.json"
            f" + weights) in: {path}"
        )
    if not _iter_weight_files(path):
        return f"no weight files ({', '.join(WEIGHT_SUFFIXES)}) found under: {path}"
    return None


def resolve_encoder_source(
    requested: Optional[str],
    allow_lexical_fallback: bool = False,
) -> ResolvedEncoder:
    """Validate the requested encoder backend before any model is constructed.

    Raises EncoderContractError unless the request can be honoured exactly, or
    the lexical fallback was explicitly opted into.
    """
    if requested is None or str(requested).strip() == "":
        if not allow_lexical_fallback:
            raise EncoderContractError(
                "No encoder checkpoint was specified.\n"
                "Pass --encoder_dir /path/to/checkpoint to evaluate the trained "
                "context-aware encoder, or --allow_lexical_fallback to run the "
                "non-neural lexical baseline on purpose."
            )
        return ResolvedEncoder(
            kind="lexical_fallback",
            requested=LEXICAL_FALLBACK_ID,
            reason="no checkpoint specified; lexical fallback explicitly allowed",
        )

    requested = str(requested).strip()

    if requested in LEGACY_LEXICAL_IDS:
        if not allow_lexical_fallback:
            raise EncoderContractError(
                f"The lexical fallback backend ({requested!r}) was requested, but it "
                "is not the trained context-aware encoder and must be opted into "
                "explicitly.\n"
                "Pass --allow_lexical_fallback to record this run as a non-neural "
                "ablation baseline, or supply a real --encoder_dir checkpoint."
            )
        return ResolvedEncoder(
            kind="lexical_fallback",
            requested=requested,
            reason="lexical fallback explicitly requested",
        )

    problem = describe_checkpoint_problem(Path(requested))
    if problem is not None:
        raise EncoderContractError(
            f"Requested encoder checkpoint is not usable: {problem}\n\n"
            "This is fatal by design: falling back to the lexical heuristic here "
            "would silently produce numbers that do not come from the trained "
            "encoder. Train/download a checkpoint, or pass "
            "--allow_lexical_fallback to run the non-neural baseline knowingly."
        )

    return ResolvedEncoder(
        kind="context_aware_encoder",
        requested=requested,
        path=requested,
        reason="validated checkpoint",
    )
