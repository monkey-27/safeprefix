#!/usr/bin/env python3
"""CPU-only Modal checksum verification for geometry input publication."""

from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
import time
from typing import Any

import modal


LOCAL_ROOT = Path(__file__).resolve().parents[1]
MANIFEST_ENV = "SAFEPREFIX_GEOMETRY_INPUT_MANIFEST"
embedded_manifest = Path("/workspace/input_manifest.json")


def _manifest_path() -> Path | None:
    configured = os.environ.get(MANIFEST_ENV)
    if configured and Path(configured).is_file():
        return Path(configured)
    if embedded_manifest.is_file():
        return embedded_manifest
    return None

COMPLETION_VOLUME = "safeprefix-recoverability-geometry-input-completion-v2"
BOUNDARY_VOLUME = "safeprefix-recoverability-geometry-input-boundary-v2"
GEOMETRY_VOLUME = "safeprefix-recoverability-geometry-output-v2"
AUTHORIZED_PROFILES = {
    "collaborator_arjun",
    "scopedupdate_6b",
    "workspace_reauth",
    "workspace_reauth_2",
}

app = modal.App("safeprefix-geometry-input-verification")
completion_volume = modal.Volume.from_name(
    COMPLETION_VOLUME, create_if_missing=False, version=2
)
boundary_volume = modal.Volume.from_name(
    BOUNDARY_VOLUME, create_if_missing=False, version=2
)
geometry_volume = modal.Volume.from_name(
    GEOMETRY_VOLUME, create_if_missing=True, version=2
)

image = modal.Image.debian_slim(python_version="3.11")
manifest_local = _manifest_path()
if manifest_local is not None and manifest_local != embedded_manifest:
    image = image.add_local_file(
        manifest_local, str(embedded_manifest), copy=True
    )


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _manifest_digest(manifest: dict[str, Any]) -> str:
    payload = dict(manifest)
    payload.pop("manifest_sha256", None)
    return hashlib.sha256(
        json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
    ).hexdigest()


def verify_rows(manifest: dict[str, Any]) -> dict[str, Any]:
    if manifest.get("manifest_sha256") != _manifest_digest(manifest):
        raise RuntimeError("publication manifest digest differs on the remote worker")
    if manifest.get("native_artifacts_included") is not False:
        raise RuntimeError("remote publication manifest includes native artifacts")
    roots = {
        COMPLETION_VOLUME: Path("/completion"),
        BOUNDARY_VOLUME: Path("/boundary"),
    }
    failures: list[dict[str, Any]] = []
    checked_bytes = 0
    for row in manifest["files"]:
        root = roots.get(row["volume_name"])
        if root is None:
            failures.append({"remote_path": row["remote_path"], "reason": "unknown_volume"})
            continue
        path = root / str(row["remote_path"]).lstrip("/")
        if not path.is_file():
            failures.append({"remote_path": row["remote_path"], "reason": "missing"})
            continue
        size = path.stat().st_size
        if size != int(row["bytes"]):
            failures.append({
                "remote_path": row["remote_path"],
                "reason": "size_mismatch",
                "expected": int(row["bytes"]),
                "actual": size,
            })
            continue
        actual = _sha256(path)
        if actual != row["sha256"]:
            failures.append({
                "remote_path": row["remote_path"],
                "reason": "sha256_mismatch",
                "expected": row["sha256"],
                "actual": actual,
            })
            continue
        checked_bytes += size
    return {
        "passed": not failures,
        "expected_files": len(manifest["files"]),
        "verified_files": len(manifest["files"]) - len(failures),
        "verified_bytes": checked_bytes,
        "failures": failures,
    }


@app.function(
    image=image,
    cpu=4,
    memory=4096,
    timeout=60 * 60,
    volumes={
        "/completion": completion_volume.with_mount_options(read_only=True),
        "/boundary": boundary_volume.with_mount_options(read_only=True),
        "/geometry": geometry_volume,
    },
)
def verify(profile: str) -> dict[str, Any]:
    if profile not in AUTHORIZED_PROFILES:
        raise RuntimeError(f"profile is not authorized: {profile}")
    manifest = json.loads(Path("/workspace/input_manifest.json").read_text())
    result = verify_rows(manifest)
    result.update({
        "profile": profile,
        "manifest_sha256": manifest["manifest_sha256"],
        "verified_unix": time.time(),
        "gpu_count": 0,
    })
    destination = Path("/geometry/input_publication") / profile / "verification.json"
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_suffix(".json.tmp")
    temporary.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    temporary.replace(destination)
    geometry_volume.commit()
    if not result["passed"]:
        raise RuntimeError(
            f"remote publication verification failed for {len(result['failures'])} files"
        )
    return result


@app.local_entrypoint()
def main(profile: str) -> None:
    if _manifest_path() is None:
        raise RuntimeError(f"{MANIFEST_ENV} must name the local publication manifest")
    if profile not in AUTHORIZED_PROFILES:
        raise RuntimeError(f"profile is not authorized: {profile}")
    active = os.environ.get("MODAL_PROFILE")
    if active != profile:
        raise RuntimeError(f"MODAL_PROFILE={active!r} does not match --profile={profile!r}")
    print(json.dumps(verify.remote(profile), indent=2, sort_keys=True))
