from __future__ import annotations

import copy
from pathlib import Path

import pytest

from safeprefix.recoverability_geometry_publication import (
    AUTHORIZED_PROFILES,
    BOUNDARY_VOLUME,
    COMPLETION_VOLUME,
    EXCLUDED_PROFILES,
    PublicationSource,
    build_publication_manifest,
    manifest_digest,
    publication_commands,
    validate_manifest,
    validate_profile,
)


def _fixture_sources(tmp_path: Path) -> tuple[PublicationSource, ...]:
    roles = (
        ("original_teacher_forced_run", COMPLETION_VOLUME, "/original"),
        ("completion_manifests", COMPLETION_VOLUME, "/completion"),
        ("completion_extension_repairability", COMPLETION_VOLUME, "/completion/repairability"),
        ("boundary_model_v1", BOUNDARY_VOLUME, "/boundary"),
    )
    values = []
    for index, (role, volume, remote) in enumerate(roles):
        root = tmp_path / role
        root.mkdir()
        (root / f"file-{index}.txt").write_text(f"payload-{index}")
        values.append(PublicationSource(role, root, volume, remote))
    return tuple(values)


def test_profile_allowlist_rejects_excluded_and_unknown() -> None:
    for profile in AUTHORIZED_PROFILES:
        assert validate_profile(profile) == profile
    for profile in (*EXCLUDED_PROFILES, "unknown"):
        with pytest.raises(RuntimeError):
            validate_profile(profile)


def test_manifest_is_deterministic_and_checksum_validated(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    sources = _fixture_sources(tmp_path)
    monkeypatch.setattr(
        "safeprefix.recoverability_geometry_publication._required_paths",
        lambda _sources: [],
    )
    first = build_publication_manifest(repo_root=tmp_path, sources=sources)
    second = build_publication_manifest(repo_root=tmp_path, sources=sources)
    assert first == second
    assert first["manifest_sha256"] == manifest_digest(first)
    assert validate_manifest(first)["passed"] is True
    assert len({(row["volume_name"], row["remote_path"]) for row in first["files"]}) == 4

    Path(first["files"][0]["local_path"]).write_text("changed")
    with pytest.raises(RuntimeError, match="size changed|checksum changed"):
        validate_manifest(first)


def test_manifest_digest_tampering_is_rejected(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    sources = _fixture_sources(tmp_path)
    monkeypatch.setattr(
        "safeprefix.recoverability_geometry_publication._required_paths",
        lambda _sources: [],
    )
    manifest = build_publication_manifest(repo_root=tmp_path, sources=sources)
    altered = copy.deepcopy(manifest)
    altered["status"] = "tampered"
    with pytest.raises(RuntimeError, match="manifest digest differs"):
        validate_manifest(altered, rehash=False)


def test_remote_collision_is_rejected(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    sources = list(_fixture_sources(tmp_path))
    collision_root = tmp_path / "collision"
    collision_root.mkdir()
    (collision_root / "file-1.txt").write_text("collision")
    sources[2] = PublicationSource(
        "completion_extension_repairability",
        collision_root,
        COMPLETION_VOLUME,
        "/completion",
    )
    monkeypatch.setattr(
        "safeprefix.recoverability_geometry_publication._required_paths",
        lambda _sources: [],
    )
    with pytest.raises(RuntimeError, match="remote publication collision"):
        build_publication_manifest(repo_root=tmp_path, sources=sources)


def test_commands_include_only_authorized_profiles(tmp_path: Path) -> None:
    commands = publication_commands(
        manifest_path=tmp_path / "manifest.json", repo_root=tmp_path
    )
    joined = "\n".join(commands)
    assert len(commands) == 2 * len(AUTHORIZED_PROFILES)
    assert all(profile in joined for profile in AUTHORIZED_PROFILES)
    assert all(profile not in joined for profile in EXCLUDED_PROFILES)
