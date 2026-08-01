#!/usr/bin/env python3
"""Publish checksummed teacher-forced inputs to authorized Modal workspaces.

This utility uploads data only.  It cannot launch a GPU job, and it refuses
profiles outside the fixed four-workspace allowlist.
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path, PurePosixPath
import sys
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

from safeprefix.recoverability_geometry_publication import (
    AUTHORIZED_PROFILES,
    BOUNDARY_VOLUME,
    COMPLETION_VOLUME,
    build_publication_manifest,
    default_sources,
    publication_commands,
    validate_manifest,
    validate_profile,
)
from safeprefix.reproducibility import atomic_json, atomic_text


def _load_manifest(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise RuntimeError("publication manifest must contain a JSON object")
    return value


def _require_active_profile(profile: str) -> str:
    selected = validate_profile(profile)
    active = os.environ.get("MODAL_PROFILE")
    if active != selected:
        raise RuntimeError(
            f"MODAL_PROFILE must exactly match --profile ({selected}); got {active!r}"
        )
    return selected


def _publication_marker_path(manifest_sha256: str) -> str:
    return f"/_geometry_input_publication/{manifest_sha256}.json"


def _published_manifest_sha(volume: Any, marker_path: str) -> str | None:
    """Return the committed publication digest, or None for an absent marker."""

    import modal

    try:
        payload = b"".join(volume.read_file(marker_path))
    except (FileNotFoundError, modal.exception.NotFoundError):
        return None
    try:
        marker = json.loads(payload)
    except (json.JSONDecodeError, UnicodeDecodeError):
        return None
    value = marker.get("manifest_sha256")
    return str(value) if value else None


def _publish(manifest_path: Path, profile: str, *, force: bool = False) -> dict[str, Any]:
    selected = _require_active_profile(profile)
    manifest = _load_manifest(manifest_path)
    validation = validate_manifest(manifest, rehash=True)

    import modal

    volumes = {
        COMPLETION_VOLUME: modal.Volume.from_name(
            COMPLETION_VOLUME, create_if_missing=True, version=2
        ),
        BOUNDARY_VOLUME: modal.Volume.from_name(
            BOUNDARY_VOLUME, create_if_missing=True, version=2
        ),
    }
    uploaded_sources: list[dict[str, Any]] = []
    skipped_volumes: list[str] = []
    marker_path = _publication_marker_path(manifest["manifest_sha256"])
    for volume_name, volume in volumes.items():
        if not force and _published_manifest_sha(volume, marker_path) == manifest["manifest_sha256"]:
            skipped_volumes.append(volume_name)
            continue
        volume_sources = [
            row for row in manifest["sources"] if row["volume_name"] == volume_name
        ]
        with volume.batch_upload(force=True) as batch:
            for source in volume_sources:
                batch.put_directory(
                    Path(source["local_root"]),
                    PurePosixPath(source["remote_root"]),
                )
                uploaded_sources.append(
                    {
                        "role": source["role"],
                        "volume_name": volume_name,
                        "remote_root": source["remote_root"],
                        "file_count": source["file_count"],
                        "total_bytes": source["total_bytes"],
                    }
                )
            batch.put_file(
                manifest_path,
                PurePosixPath(marker_path),
            )

    result = {
        "status": "PUBLISHED_PENDING_REMOTE_CHECKSUM_VERIFICATION",
        "profile": selected,
        "manifest_path": str(manifest_path),
        "manifest_sha256": manifest["manifest_sha256"],
        "total_files": validation["total_files"],
        "total_bytes": validation["total_bytes"],
        "sources": uploaded_sources,
        "volumes": sorted(volumes),
        "skipped_already_committed_volumes": sorted(skipped_volumes),
        "force": bool(force),
        "gpu_jobs_submitted": 0,
    }
    local_result = (
        manifest_path.parent
        / f"publication_result_{selected}_{manifest['manifest_sha256'][:12]}.json"
    )
    atomic_json(local_result, result)
    return {**result, "local_result": str(local_result)}


def _write_report(manifest_path: Path, report_path: Path) -> None:
    manifest = _load_manifest(manifest_path)
    validation = validate_manifest(manifest, rehash=False)
    lines = [
        "# Distributed input publication plan",
        "",
        f"Manifest SHA256: `{manifest['manifest_sha256']}`",
        f"Files: {validation['total_files']:,}",
        f"Bytes: {validation['total_bytes']:,}",
        "",
        "Only the four explicitly authorized profiles are emitted. No GPU job is launched.",
        "",
        "## Sources",
        "",
    ]
    for source in manifest["sources"]:
        lines.append(
            f"- `{source['role']}`: {source['file_count']:,} files, "
            f"{source['total_bytes']:,} bytes -> "
            f"`{source['volume_name']}:{source['remote_root']}`"
        )
    lines.extend(["", "## Commands", ""])
    for command in publication_commands(
        manifest_path=manifest_path, repo_root=REPO_ROOT
    ):
        lines.extend(["```bash", command, "```", ""])
    atomic_text(report_path, "\n".join(lines).rstrip() + "\n")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)

    manifest_parser = subparsers.add_parser("manifest")
    manifest_parser.add_argument("--output", type=Path, required=True)
    manifest_parser.add_argument("--report", type=Path)

    commands_parser = subparsers.add_parser("commands")
    commands_parser.add_argument("--manifest", type=Path, required=True)

    publish_parser = subparsers.add_parser("publish")
    publish_parser.add_argument("--manifest", type=Path, required=True)
    publish_parser.add_argument("--profile", required=True, choices=AUTHORIZED_PROFILES)
    publish_parser.add_argument("--force", action="store_true")

    args = parser.parse_args()
    if args.command == "manifest":
        manifest = build_publication_manifest(repo_root=REPO_ROOT)
        atomic_json(args.output, manifest)
        if args.report:
            _write_report(args.output, args.report)
        print(json.dumps({
            "manifest": str(args.output),
            "manifest_sha256": manifest["manifest_sha256"],
            "total_files": manifest["total_files"],
            "total_bytes": manifest["total_bytes"],
        }, indent=2, sort_keys=True))
    elif args.command == "commands":
        manifest = _load_manifest(args.manifest)
        validate_manifest(manifest, rehash=False)
        print("\n".join(publication_commands(
            manifest_path=args.manifest, repo_root=REPO_ROOT
        )))
    elif args.command == "publish":
        print(json.dumps(
            _publish(args.manifest, args.profile, force=args.force),
            indent=2,
            sort_keys=True,
        ))
    else:  # pragma: no cover
        raise AssertionError(args.command)


if __name__ == "__main__":
    main()
