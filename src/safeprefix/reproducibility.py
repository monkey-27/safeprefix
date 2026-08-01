"""Atomic artifacts, deterministic RNG, provenance, and resumability helpers."""

from __future__ import annotations

import hashlib
import json
import os
import platform
import random
import subprocess
import tempfile
from contextlib import contextmanager
from datetime import datetime, timezone
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path
from typing import Any, Iterable, Iterator, Mapping

import numpy as np


def artifact_path_reference(path: str | Path, *, repository_root: str | Path) -> str:
    """Serialize repo artifacts relatively and external-volume artifacts absolutely."""
    artifact = Path(path)
    root = Path(repository_root)
    try:
        return str(artifact.relative_to(root))
    except ValueError:
        if not artifact.is_absolute():
            raise
        return str(artifact)


def now_iso() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat()


def stable_hash(value: Any) -> str:
    payload = json.dumps(value, sort_keys=True, ensure_ascii=False, separators=(",", ":"), default=str)
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def stable_seed(*parts: Any) -> int:
    return int(stable_hash(parts)[:16], 16) % (2**31)


def git_commit(root: Path | None = None) -> str:
    root = root or Path(__file__).resolve().parents[2]
    try:
        return subprocess.run(
            ["git", "rev-parse", "HEAD"], cwd=root, check=True,
            capture_output=True, text=True,
        ).stdout.strip()
    except Exception:
        return "unknown"


def package_versions() -> dict[str, str]:
    packages = [
        "accelerate", "datasets", "numpy", "pandas", "pyarrow", "PyYAML",
        "scikit-learn", "scipy", "torch", "transformers", "typer",
    ]
    result: dict[str, str] = {}
    for package in packages:
        try:
            result[package] = version(package)
        except PackageNotFoundError:
            result[package] = "unavailable"
    return result


def set_seed(seed: int, deterministic: bool = True) -> None:
    random.seed(seed)
    np.random.seed(seed)
    try:
        import torch

        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
        if deterministic:
            torch.use_deterministic_algorithms(True, warn_only=True)
    except ImportError:
        pass


def provenance(config: Mapping[str, Any], command: str | None = None) -> dict[str, Any]:
    configured_models = {
        str(name): {
            "hf_model_id": entry.get("hf_model_id"),
            "model_revision": entry.get("revision"),
            "tokenizer_id": entry.get("tokenizer_id"),
            "tokenizer_revision": entry.get("tokenizer_revision"),
        }
        for name, entry in config.get("models", {}).items()
        if isinstance(entry, Mapping)
    }
    configured_datasets = {
        str(name): {
            "hf_dataset_id": entry.get("hf_dataset_id"),
            "dataset_revision": entry.get("revision"),
        }
        for name, entry in config.get("datasets", {}).items()
        if isinstance(entry, Mapping)
    }
    return {
        "created_at": now_iso(),
        "git_commit": git_commit(),
        "python": platform.python_version(),
        "platform": platform.platform(),
        "command": command,
        "seed": config.get("seed"),
        "config_sha256": stable_hash(config),
        "configured_model_revisions": configured_models,
        "configured_dataset_revisions": configured_datasets,
        "package_versions": package_versions(),
    }


def atomic_text(path: str | Path, text: str) -> None:
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile("w", encoding="utf-8", dir=destination.parent, delete=False) as handle:
        temporary = Path(handle.name)
        handle.write(text)
        handle.flush()
        os.fsync(handle.fileno())
    temporary.replace(destination)


def atomic_json(path: str | Path, value: Any) -> None:
    atomic_text(path, json.dumps(value, indent=2, sort_keys=True, ensure_ascii=False, default=str) + "\n")


def atomic_jsonl(path: str | Path, rows: Iterable[Mapping[str, Any]]) -> None:
    atomic_text(path, "".join(json.dumps(dict(row), sort_keys=True, ensure_ascii=False, default=str) + "\n" for row in rows))


def atomic_parquet(path: str | Path, frame: Any) -> None:
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(dir=destination.parent, suffix=".parquet", delete=False) as handle:
        temporary = Path(handle.name)
    try:
        frame.to_parquet(temporary, index=False)
        temporary.replace(destination)
    finally:
        temporary.unlink(missing_ok=True)


def artifact_complete(path: str | Path, required_columns: Iterable[str] = ()) -> bool:
    candidate = Path(path)
    if not candidate.is_file() or candidate.stat().st_size == 0:
        return False
    try:
        if candidate.suffix == ".json":
            json.loads(candidate.read_text(encoding="utf-8"))
        elif candidate.suffix == ".jsonl":
            for line in candidate.read_text(encoding="utf-8").splitlines():
                if line.strip():
                    json.loads(line)
        elif candidate.suffix == ".parquet":
            import pandas as pd

            columns = set(pd.read_parquet(candidate).columns)
            if not set(required_columns).issubset(columns):
                return False
    except Exception:
        return False
    return True


def refuse_overwrite(path: str | Path, *, resume: bool, overwrite: bool) -> bool:
    """Return True when a valid existing artifact should be skipped."""
    candidate = Path(path)
    if artifact_complete(candidate):
        if overwrite:
            return False
        if resume:
            return True
        raise FileExistsError(f"refusing to overwrite existing artifact: {candidate}")
    if candidate.exists() and not overwrite:
        raise RuntimeError(f"existing artifact is incomplete or invalid: {candidate}")
    return False


@contextmanager
def stage_manifest(
    output_dir: str | Path,
    stage: str,
    config: Mapping[str, Any],
    command: str | None = None,
) -> Iterator[dict[str, Any]]:
    root = Path(output_dir)
    root.mkdir(parents=True, exist_ok=True)
    state = {**provenance(config, command), "stage": stage, "status": "running"}
    atomic_json(root / f"{stage}.manifest.json", state)
    try:
        yield state
    except Exception as exc:
        state.update(status="failed", error=f"{type(exc).__name__}: {exc}", finished_at=now_iso())
        atomic_json(root / f"{stage}.manifest.json", state)
        raise
    else:
        state.update(status="complete", finished_at=now_iso())
        atomic_json(root / f"{stage}.manifest.json", state)
