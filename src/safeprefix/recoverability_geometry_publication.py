"""Safe publication plan for distributed teacher-forced geometry inputs.

The helper is intentionally restricted to four explicitly authorized Modal
profiles.  Native artifacts and the excluded workspaces are rejected before a
manifest or command is produced.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
import hashlib
import json
from pathlib import Path, PurePosixPath
from typing import Any, Iterable, Mapping, Sequence


AUTHORIZED_PROFILES = (
    "collaborator_arjun",
    "scopedupdate_6b",
    "workspace_reauth",
    "workspace_reauth_2",
)
EXCLUDED_PROFILES = ("meskmmy", "marketingdeals")
# Dedicated v2 volumes avoid both cross-experiment mutation and accidental
# attachment to legacy v1 volumes that happen to use the older generic names.
COMPLETION_VOLUME = "safeprefix-recoverability-geometry-input-completion-v2"
BOUNDARY_VOLUME = "safeprefix-recoverability-geometry-input-boundary-v2"
GEOMETRY_VOLUME = "safeprefix-recoverability-geometry-output-v2"
ORIGINAL_RUN_ID = "safeprefix_full_teacher_forced_20260726_r3"
COMPLETION_RUN_ID = "safeprefix_teacher_forced_completion_20260727_r5"
BOUNDARY_RUN_ID = "safeprefix_boundary_model_v1_20260728_r2"
FORBIDDEN_PATH_TERMS = ("native", "native_eval", "native-eval", "final_test")


@dataclass(frozen=True)
class PublicationSource:
    role: str
    local_root: Path
    volume_name: str
    remote_root: str


def default_sources(repo_root: Path) -> tuple[PublicationSource, ...]:
    return (
        PublicationSource(
            role="original_teacher_forced_run",
            local_root=Path(
                "/private/tmp/safeprefix_final_download/"
                f"{ORIGINAL_RUN_ID}"
            ),
            volume_name=COMPLETION_VOLUME,
            remote_root=f"/{ORIGINAL_RUN_ID}",
        ),
        PublicationSource(
            role="completion_manifests",
            local_root=Path("/private/tmp/safeprefix-cross-seed")
            / COMPLETION_RUN_ID,
            volume_name=COMPLETION_VOLUME,
            remote_root=f"/{COMPLETION_RUN_ID}",
        ),
        PublicationSource(
            role="completion_extension_repairability",
            local_root=Path("/private/tmp/safeprefix-boundary-r5-20260728")
            / "repairability",
            volume_name=COMPLETION_VOLUME,
            remote_root=(
                f"/{COMPLETION_RUN_ID}/artifacts/teacher_forced_completion/"
                "repairability"
            ),
        ),
        PublicationSource(
            role="boundary_model_v1",
            local_root=repo_root
            / "artifacts/boundary_model_v1_remote/boundary_model_v1",
            volume_name=BOUNDARY_VOLUME,
            remote_root=f"/{BOUNDARY_RUN_ID}/artifacts/boundary_model_v1",
        ),
    )


def validate_profile(profile: str) -> str:
    value = str(profile)
    if value in EXCLUDED_PROFILES:
        raise RuntimeError(f"Modal profile is explicitly excluded: {value}")
    if value not in AUTHORIZED_PROFILES:
        raise RuntimeError(f"Modal profile is not in the fixed allowlist: {value}")
    return value


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def manifest_digest(manifest: Mapping[str, Any]) -> str:
    """Hash a publication manifest without its self-referential digest field."""

    payload = dict(manifest)
    payload.pop("manifest_sha256", None)
    return hashlib.sha256(
        json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
    ).hexdigest()


def _remote_path(root: str, relative: Path) -> str:
    path = PurePosixPath(root) / PurePosixPath(relative.as_posix())
    text = "/" + str(path).lstrip("/")
    if ".." in PurePosixPath(text).parts:
        raise RuntimeError(f"unsafe remote publication path: {text}")
    return text


def _required_paths(sources: Mapping[str, PublicationSource]) -> list[Path]:
    models = (
        "family_a_small",
        "family_a_large",
        "family_b_small",
        "family_b_large",
    )
    original = sources["original_teacher_forced_run"].local_root
    completion = sources["completion_manifests"].local_root
    extension = sources["completion_extension_repairability"].local_root
    boundary = sources["boundary_model_v1"].local_root
    required = [
        original / "artifacts/full_teacher_forced_suite/immutable_manifests/immutable_protocol_manifest.json",
        completion / "artifacts/teacher_forced_completion/frozen_execution_manifest.json",
        completion
        / "artifacts/teacher_forced_completion/immutable_manifests/repairability/common_trace_manifest.jsonl",
        extension / "aggregated_checkpoint_outcomes",
        boundary / "data/canonical_checkpoint_manifest.parquet",
        boundary / "integrity/final_integrity.json",
    ]
    for model in models:
        required.extend(
            [
                original
                / f"artifacts/full_teacher_forced_suite/raw_rollout_shards/{model}",
                completion
                / (
                    "artifacts/teacher_forced_completion/immutable_manifests/"
                    f"repairability/per_model/{model}/trace_manifest.jsonl"
                ),
                extension / f"raw_rollout_shards/{model}",
                boundary / f"data/features/{model}.pt",
                boundary
                / f"training/{model}/linear_probe/lr_1e-03/seed_0/best.pt",
                boundary / f"calibration/{model}/seed_0/calibrator.json",
            ]
        )
    return required


def build_publication_manifest(
    *,
    repo_root: Path,
    sources: Sequence[PublicationSource] | None = None,
) -> dict[str, Any]:
    selected = tuple(sources or default_sources(repo_root))
    by_role = {source.role: source for source in selected}
    if len(by_role) != len(selected):
        raise RuntimeError("publication roles must be unique")
    missing_roles = {
        "original_teacher_forced_run",
        "completion_manifests",
        "completion_extension_repairability",
        "boundary_model_v1",
    } - set(by_role)
    if missing_roles:
        raise RuntimeError(f"publication sources lack roles: {sorted(missing_roles)}")
    for source in selected:
        lowered = source.local_root.as_posix().casefold()
        if any(term in lowered for term in FORBIDDEN_PATH_TERMS):
            raise RuntimeError(f"native/final-test source is forbidden: {source.local_root}")
        if not source.local_root.is_dir():
            raise FileNotFoundError(source.local_root)
        if source.volume_name not in {COMPLETION_VOLUME, BOUNDARY_VOLUME}:
            raise RuntimeError(f"unexpected publication volume: {source.volume_name}")
    missing = [str(path) for path in _required_paths(by_role) if not path.exists()]
    if missing:
        raise RuntimeError(f"required publication inputs are missing: {missing}")

    remote_seen: set[tuple[str, str]] = set()
    files: list[dict[str, Any]] = []
    source_summaries: list[dict[str, Any]] = []
    for source in selected:
        source_files = sorted(
            path for path in source.local_root.rglob("*") if path.is_file()
        )
        if not source_files:
            raise RuntimeError(f"publication source is empty: {source.local_root}")
        source_rows = []
        for path in source_files:
            if path.is_symlink():
                raise RuntimeError(f"symlink input is prohibited: {path}")
            relative = path.relative_to(source.local_root)
            remote_path = _remote_path(source.remote_root, relative)
            key = (source.volume_name, remote_path)
            if key in remote_seen:
                raise RuntimeError(f"remote publication collision: {key}")
            remote_seen.add(key)
            row = {
                "role": source.role,
                "local_path": str(path),
                "relative_path": relative.as_posix(),
                "volume_name": source.volume_name,
                "remote_path": remote_path,
                "bytes": int(path.stat().st_size),
                "sha256": sha256_file(path),
            }
            files.append(row)
            source_rows.append(row)
        tree_digest = hashlib.sha256(
            json.dumps(
                [
                    [row["relative_path"], row["bytes"], row["sha256"]]
                    for row in source_rows
                ],
                separators=(",", ":"),
            ).encode("utf-8")
        ).hexdigest()
        source_summaries.append(
            {
                **{key: str(value) if isinstance(value, Path) else value for key, value in asdict(source).items()},
                "file_count": len(source_rows),
                "total_bytes": sum(row["bytes"] for row in source_rows),
                "tree_sha256": tree_digest,
            }
        )
    manifest = {
        "schema_version": 1,
        "status": "READY_FOR_AUTHORIZED_PUBLICATION",
        "authorized_profiles": list(AUTHORIZED_PROFILES),
        "excluded_profiles": list(EXCLUDED_PROFILES),
        "volume_names": {
            "completion": COMPLETION_VOLUME,
            "boundary": BOUNDARY_VOLUME,
            "geometry_output": GEOMETRY_VOLUME,
        },
        "sources": source_summaries,
        "files": files,
        "total_files": len(files),
        "total_bytes": sum(row["bytes"] for row in files),
        "native_artifacts_included": False,
    }
    manifest["manifest_sha256"] = manifest_digest(manifest)
    return manifest


def validate_manifest(manifest: Mapping[str, Any], *, rehash: bool = True) -> dict[str, Any]:
    if manifest.get("manifest_sha256") != manifest_digest(manifest):
        raise RuntimeError("publication manifest digest differs")
    if manifest.get("native_artifacts_included") is not False:
        raise RuntimeError("publication manifest does not exclude native artifacts")
    if tuple(manifest.get("authorized_profiles", ())) != AUTHORIZED_PROFILES:
        raise RuntimeError("publication profile allowlist differs")
    if set(manifest.get("excluded_profiles", ())) != set(EXCLUDED_PROFILES):
        raise RuntimeError("publication profile exclusions differ")
    rows = list(manifest.get("files", []))
    if len(rows) != int(manifest.get("total_files", -1)):
        raise RuntimeError("publication file count differs")
    identities = [(row["volume_name"], row["remote_path"]) for row in rows]
    if len(identities) != len(set(identities)):
        raise RuntimeError("publication manifest contains remote collisions")
    actual_bytes = 0
    for row in rows:
        path = Path(row["local_path"])
        if not path.is_file():
            raise RuntimeError(f"publication input disappeared: {path}")
        if path.stat().st_size != int(row["bytes"]):
            raise RuntimeError(f"publication input size changed: {path}")
        if rehash and sha256_file(path) != row["sha256"]:
            raise RuntimeError(f"publication input checksum changed: {path}")
        actual_bytes += int(row["bytes"])
    if actual_bytes != int(manifest.get("total_bytes", -1)):
        raise RuntimeError("publication byte total differs")
    return {
        "passed": True,
        "total_files": len(rows),
        "total_bytes": actual_bytes,
        "manifest_sha256": manifest["manifest_sha256"],
    }


def publication_commands(*, manifest_path: Path, repo_root: Path) -> list[str]:
    commands: list[str] = []
    for profile in AUTHORIZED_PROFILES:
        commands.append(
            f"MODAL_PROFILE={profile} python3 scripts/publish_recoverability_geometry_inputs.py "
            f"publish --manifest {manifest_path} --profile {profile}"
        )
        commands.append(
            f"MODAL_PROFILE={profile} SAFEPREFIX_GEOMETRY_INPUT_MANIFEST={manifest_path} "
            "python3 -m modal run scripts/modal_verify_recoverability_geometry_inputs.py "
            f"--profile {profile}"
        )
    if any(profile in "\n".join(commands) for profile in EXCLUDED_PROFILES):
        raise RuntimeError("excluded profile leaked into publication commands")
    return commands


__all__ = [
    "AUTHORIZED_PROFILES",
    "BOUNDARY_VOLUME",
    "COMPLETION_VOLUME",
    "EXCLUDED_PROFILES",
    "GEOMETRY_VOLUME",
    "PublicationSource",
    "build_publication_manifest",
    "default_sources",
    "manifest_digest",
    "publication_commands",
    "sha256_file",
    "validate_manifest",
    "validate_profile",
]
