from pathlib import Path
import json
import tarfile

import pytest

from safeprefix.cross_workspace_completion import (
    WORKSPACE_PLAN,
    build_balanced_partitions,
    build_workspace_partitions,
    validated_safety_archive_members,
)


def _write_safety_archive(path: Path, member_names: list[str]) -> None:
    source = path.parent / "payload"
    source.write_bytes(b"payload")
    with tarfile.open(path, "w:") as archive:
        for name in member_names:
            archive.add(source, arcname=name)


def test_safety_transfer_archive_requires_exact_safe_paths(tmp_path: Path) -> None:
    model = "family_b_small"
    valid = tmp_path / "valid.tar"
    _write_safety_archive(
        valid,
        [
            f"._{model}",
            f"{model}/pack/complete.json",
            f"{model}/pack/features.pt",
            f"{model}/pack/checkpoint_metadata.parquet",
            f"{model}/pack/._features.pt",
        ],
    )
    assert len(validated_safety_archive_members(valid, model_key=model)) == 3

    traversal = tmp_path / "traversal.tar"
    _write_safety_archive(traversal, [f"{model}/../escape/features.pt"])
    with pytest.raises(RuntimeError, match="unsafe archive member"):
        validated_safety_archive_members(traversal, model_key=model)

    unexpected = tmp_path / "unexpected.tar"
    _write_safety_archive(unexpected, [f"{model}/pack/model.safetensors"])
    with pytest.raises(RuntimeError, match="unexpected archive artifact"):
        validated_safety_archive_members(unexpected, model_key=model)


def _write_assignments(root: Path, kind: str, model: str, slots: range) -> set[str]:
    path = root / kind / "per_model" / model
    path.mkdir(parents=True, exist_ok=True)
    payload = {str(slot): [f"{model}-{kind}-{slot}-a", f"{model}-{kind}-{slot}-b"] for slot in slots}
    (path / "worker_assignments.json").write_text(json.dumps(payload))
    return {value for values in payload.values() for value in values}


def test_workspace_partitions_are_disjoint_complete_and_skip_markers(tmp_path: Path) -> None:
    expected = {"repairability": {}, "safety": {}}
    for rows in WORKSPACE_PLAN.values():
        for model, slots in rows:
            for kind in expected:
                if model not in expected[kind]:
                    all_slots = range(20 if model == "family_b_large" else 10 if model == "family_b_small" else 5)
                    expected[kind][model] = _write_assignments(tmp_path, kind, model, all_slots)
    completed = {"repairability": {}, "safety_features": {}}
    for model in expected["repairability"]:
        completed["repairability"][model] = [sorted(expected["repairability"][model])[0]]
        completed["safety_features"][model] = [sorted(expected["safety"][model])[0]]

    partitions, summary = build_workspace_partitions(
        tmp_path, completed, run_id="run", source_commit="abc"
    )

    assert set(partitions) == set(WORKSPACE_PLAN)
    assert sum(row["workers"] for row in summary["workspace_counts"].values()) == 40
    for kind, field in (("repairability", "repair_ids"), ("safety", "safety_ids")):
        owned = [
            pack_id
            for partition in partitions.values()
            for assignment in partition["assignments"]
            for pack_id in assignment[field]
        ]
        assert len(owned) == len(set(owned))
        completed_ids = {
            value
            for values in completed["repairability" if kind == "repairability" else "safety_features"].values()
            for value in values
        }
        assert set(owned) == set().union(*expected[kind].values()) - completed_ids


def test_workspace_partition_hash_is_stable(tmp_path: Path) -> None:
    for rows in WORKSPACE_PLAN.values():
        for model, _ in rows:
            if not (tmp_path / "repairability/per_model" / model).exists():
                slots = range(20 if model == "family_b_large" else 10 if model == "family_b_small" else 5)
                _write_assignments(tmp_path, "repairability", model, slots)
                _write_assignments(tmp_path, "safety", model, slots)
    kwargs = dict(run_id="run", source_commit="abc")
    first, _ = build_workspace_partitions(tmp_path, {}, **kwargs)
    second, _ = build_workspace_partitions(tmp_path, {}, **kwargs)
    assert {key: value["partition_hash"] for key, value in first.items()} == {
        key: value["partition_hash"] for key, value in second.items()
    }


def test_balanced_partitions_cover_remaining_once(tmp_path: Path) -> None:
    for kind, filename in (("repairability", "extension_execution_packs.jsonl"), ("safety", "execution_packs.jsonl")):
        path = tmp_path / kind / "per_model" / "model" / filename
        path.parent.mkdir(parents=True, exist_ok=True)
        rows = [
            {"pack_id": f"{kind}-{index}", "estimated_work": 100 - index}
            for index in range(12)
        ]
        path.write_text("".join(json.dumps(row) + "\n" for row in rows))
    snapshots = [{
        "repairability": {"model": ["repairability-0"]},
        "safety_features": {"model": ["safety-0"]},
    }]
    partitions, summary = build_balanced_partitions(
        tmp_path,
        snapshots,
        {"one": {"model": 2}, "two": {"model": 2}},
        run_id="run",
        source_commit="abc",
    )
    for field in ("repair_ids", "safety_ids"):
        values = [
            pack_id
            for partition in partitions.values()
            for assignment in partition["assignments"]
            for pack_id in assignment[field]
        ]
        assert len(values) == 11
        assert len(values) == len(set(values))
    assert summary["balance"]["model"]["load_ratio"] < 1.25
