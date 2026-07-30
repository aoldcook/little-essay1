"""Reproducibility manifest (EVAL_VALIDITY_AUDIT.md finding M2).

Every evaluation run should emit one of these alongside its results so that a
number in a table can be traced back to: the exact commit, the exact installed
library versions, the exact dataset files (by content hash), the seeds, and the
resolved configuration.

Usage as a library:
    from repro.manifest import build_manifest, write_manifest
    manifest = build_manifest(seeds={"seed": 42}, datasets=[path], config=vars(args))
    write_manifest(out_dir / "manifest.json", manifest)

Usage as a CLI (prints the current environment manifest):
    python -m repro.manifest
"""

from __future__ import annotations

import hashlib
import json
import os
import platform
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence

# Packages whose versions materially change results.
TRACKED_PACKAGES = (
    "torch",
    "transformers",
    "numpy",
    "scikit-learn",
    "openai",
)

_HASH_CHUNK = 1024 * 1024
_HASH_MAX_BYTES = 512 * 1024 * 1024


def _run_git(args: Sequence[str], cwd: Path) -> Optional[str]:
    try:
        out = subprocess.run(
            ["git", *args],
            cwd=str(cwd),
            capture_output=True,
            text=True,
            timeout=15,
            check=False,
        )
    except (OSError, subprocess.SubprocessError):
        return None
    if out.returncode != 0:
        return None
    return out.stdout.strip()


def git_state(repo_root: Optional[Path] = None) -> Dict[str, object]:
    """Capture commit, branch and whether the tree was dirty at run time.

    A dirty tree is recorded explicitly: results produced from uncommitted code
    are not reproducible and should be labelled as such.
    """
    root = repo_root or Path(__file__).resolve().parent.parent
    commit = _run_git(["rev-parse", "HEAD"], root)
    status = _run_git(["status", "--porcelain"], root)
    return {
        "commit": commit,
        "branch": _run_git(["rev-parse", "--abbrev-ref", "HEAD"], root),
        "dirty": bool(status) if status is not None else None,
        "dirty_files": [line[3:] for line in (status or "").splitlines()][:50],
    }


def package_versions(packages: Iterable[str] = TRACKED_PACKAGES) -> Dict[str, Optional[str]]:
    try:
        from importlib.metadata import PackageNotFoundError, version
    except ImportError:  # pragma: no cover - Python < 3.8
        return {name: None for name in packages}

    versions: Dict[str, Optional[str]] = {}
    for name in packages:
        try:
            versions[name] = version(name)
        except PackageNotFoundError:
            versions[name] = None
    return versions


def file_fingerprint(path: Path) -> Dict[str, object]:
    """Content hash of a dataset file, so splits can be verified later."""
    path = Path(path)
    if not path.exists():
        return {"path": str(path), "exists": False}

    size = path.stat().st_size
    sha = hashlib.sha256()
    truncated = False
    read = 0
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(_HASH_CHUNK), b""):
            if read >= _HASH_MAX_BYTES:
                truncated = True
                break
            sha.update(chunk)
            read += len(chunk)

    info: Dict[str, object] = {
        "path": str(path),
        "exists": True,
        "bytes": size,
        "sha256": sha.hexdigest(),
        "hash_truncated": truncated,
    }
    if path.suffix == ".jsonl":
        with path.open("r", encoding="utf-8", errors="replace") as f:
            info["num_lines"] = sum(1 for line in f if line.strip())
    return info


def hardware_state() -> Dict[str, object]:
    info: Dict[str, object] = {
        "platform": platform.platform(),
        "processor": platform.processor(),
        "python": sys.version.split()[0],
    }
    try:
        import torch

        info["torch_cuda_available"] = bool(torch.cuda.is_available())
        if torch.cuda.is_available():
            info["cuda_version"] = torch.version.cuda
            info["gpus"] = [
                torch.cuda.get_device_name(i) for i in range(torch.cuda.device_count())
            ]
    except Exception:
        info["torch_cuda_available"] = None
    return info


def build_manifest(
    seeds: Optional[Dict[str, int]] = None,
    datasets: Optional[Sequence[Path]] = None,
    config: Optional[Dict[str, object]] = None,
    provenance: Optional[Dict[str, object]] = None,
    extra: Optional[Dict[str, object]] = None,
    repo_root: Optional[Path] = None,
) -> Dict[str, object]:
    """Assemble a full run manifest.

    `provenance` is expected to be RuntimeProvenance.to_dict() so that the
    backend that actually ran is part of the permanent record.
    """
    manifest: Dict[str, object] = {
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "git": git_state(repo_root),
        "packages": package_versions(),
        "hardware": hardware_state(),
        "seeds": dict(seeds or {}),
        "datasets": [file_fingerprint(Path(p)) for p in (datasets or [])],
        "config": _jsonable(config or {}),
        "runtime_provenance": provenance or {},
        "env": {
            # Never record secret values, only whether they were present.
            "DASHSCOPE_API_KEY_present": bool(os.environ.get("DASHSCOPE_API_KEY")),
            "READER_MODEL": os.environ.get("READER_MODEL"),
        },
    }
    if extra:
        manifest.update(_jsonable(extra))
    return manifest


def _jsonable(value: object) -> object:
    """Best-effort conversion so a manifest never fails to serialise."""
    if isinstance(value, dict):
        return {str(k): _jsonable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(v) for v in value]
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    if isinstance(value, Path):
        return str(value)
    return repr(value)


def write_manifest(path: Path, manifest: Dict[str, object]) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    return path


def main() -> None:
    print(json.dumps(build_manifest(), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
