"""Freeze disjoint cross-workspace ownership of unfinished completion packs."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
import tarfile
from typing import Any, Mapping


WORKSPACE_PLAN: dict[str, list[tuple[str, range]]] = {
    "meskmmy": [
        ("family_a_small", range(0, 5)),
        ("family_a_large", range(0, 5)),
    ],
    "larpmonk": [("family_b_small", range(0, 10))],
    "marketingdeals666": [("family_b_large", range(0, 10))],
    "monkelarp123": [("family_b_large", range(10, 20))],
}


def validated_safety_archive_members(
    archive_path: Path,
    *,
    model_key: str,
) -> list[tarfile.TarInfo]:
    """Return the complete, path-safe file set in a safety transfer archive."""
    allowed_files = {"complete.json", "features.pt", "checkpoint_metadata.parquet"}
    members: list[tarfile.TarInfo] = []
    with tarfile.open(archive_path, "r:") as archive:
        for member in archive.getmembers():
            path = Path(member.name)
            if path.is_absolute() or ".." in path.parts:
                raise RuntimeError(f"unsafe archive member: {member.name}")
            if any(part.startswith("._") for part in path.parts) or (
                path.parts and path.parts[0] == "__MACOSX"
            ):
                # BSD tar may add non-content AppleDouble sidecars. They are
                # never extracted and never count toward artifact coverage.
                continue
            if not path.parts or path.parts[0] != model_key:
                raise RuntimeError(f"archive member is outside {model_key}: {member.name}")
            if member.issym() or member.islnk() or member.isdev():
                raise RuntimeError(f"archive links/devices are prohibited: {member.name}")
            if member.isfile():
                if len(path.parts) != 3 or path.name not in allowed_files:
                    raise RuntimeError(f"unexpected archive artifact: {member.name}")
                members.append(member)
            elif not member.isdir():
                raise RuntimeError(f"unsupported archive member: {member.name}")
    return members


def _load_assignments(root: Path, kind: str, model_key: str) -> dict[int, list[str]]:
    path = root / kind / "per_model" / model_key / "worker_assignments.json"
    payload = json.loads(path.read_text())
    if not isinstance(payload, dict):
        raise TypeError(f"worker assignments must be a mapping: {path}")
    result = {int(slot): list(map(str, values)) for slot, values in payload.items()}
    flattened = [pack_id for values in result.values() for pack_id in values]
    if len(flattened) != len(set(flattened)):
        raise ValueError(f"duplicate pack assignment in {path}")
    return result


def _hash_partition(partition: Mapping[str, Any]) -> str:
    canonical = json.dumps(dict(partition), sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(canonical.encode()).hexdigest()


def build_workspace_partitions(
    manifest_root: Path,
    completed_snapshot: Mapping[str, Any],
    *,
    run_id: str,
    source_commit: str,
) -> tuple[dict[str, dict[str, Any]], dict[str, Any]]:
    """Build immutable workspace partitions preserving frozen worker slots.

    Completed packs are excluded solely from durable ``complete.json`` marker
    IDs supplied in ``completed_snapshot``.  Every unfinished frozen pack is
    assigned exactly once across the four workspaces.
    """
    repair_completed = {
        model: set(map(str, values))
        for model, values in completed_snapshot.get("repairability", {}).items()
    }
    safety_completed = {
        model: set(map(str, values))
        for model, values in completed_snapshot.get("safety_features", {}).items()
    }
    assignments_by_kind: dict[str, dict[str, dict[int, list[str]]]] = {
        "repairability": {},
        "safety": {},
    }
    models = sorted({model for rows in WORKSPACE_PLAN.values() for model, _ in rows})
    for kind in assignments_by_kind:
        for model in models:
            assignments_by_kind[kind][model] = _load_assignments(manifest_root, kind, model)

    expected: dict[str, dict[str, set[str]]] = {"repairability": {}, "safety": {}}
    for kind in expected:
        for model, slots in assignments_by_kind[kind].items():
            expected[kind][model] = {pack_id for values in slots.values() for pack_id in values}

    completed_by_kind = {"repairability": repair_completed, "safety": safety_completed}
    for kind, per_model in completed_by_kind.items():
        for model, completed in per_model.items():
            unknown = completed - expected[kind].get(model, set())
            if unknown:
                raise ValueError(f"completed snapshot contains unknown {kind}/{model} packs: {sorted(unknown)[:3]}")

    partitions: dict[str, dict[str, Any]] = {}
    global_owned: dict[str, set[str]] = {"repairability": set(), "safety": set()}
    for workspace, rows in WORKSPACE_PLAN.items():
        assignments: list[dict[str, Any]] = []
        workspace_slot = 0
        for model, source_slots in rows:
            for source_slot in source_slots:
                repair_ids = [
                    pack_id
                    for pack_id in assignments_by_kind["repairability"][model][source_slot]
                    if pack_id not in repair_completed.get(model, set())
                ]
                safety_ids = [
                    pack_id
                    for pack_id in assignments_by_kind["safety"][model][source_slot]
                    if pack_id not in safety_completed.get(model, set())
                ]
                for kind, values in (("repairability", repair_ids), ("safety", safety_ids)):
                    overlap = global_owned[kind].intersection(values)
                    if overlap:
                        raise ValueError(f"cross-workspace duplicate {kind} ownership: {sorted(overlap)[:3]}")
                    global_owned[kind].update(values)
                assignments.append(
                    {
                        "worker_slot": workspace_slot,
                        "source_worker_slot": source_slot,
                        "model_key": model,
                        "repair_ids": repair_ids,
                        "safety_ids": safety_ids,
                    }
                )
                workspace_slot += 1
        base: dict[str, Any] = {
            "schema_version": 1,
            "run_id": run_id,
            "source_commit": source_commit,
            "workspace": workspace,
            "assignments": assignments,
        }
        partitions[workspace] = {**base, "partition_hash": _hash_partition(base)}

    expected_unfinished = {
        kind: {
            pack_id
            for model, packs in expected[kind].items()
            for pack_id in packs - completed_by_kind[kind].get(model, set())
        }
        for kind in expected
    }
    for kind in expected_unfinished:
        if global_owned[kind] != expected_unfinished[kind]:
            missing = expected_unfinished[kind] - global_owned[kind]
            extra = global_owned[kind] - expected_unfinished[kind]
            raise ValueError(f"invalid {kind} coverage: missing={len(missing)} extra={len(extra)}")

    summary = {
        "schema_version": 1,
        "run_id": run_id,
        "source_commit": source_commit,
        "completed_snapshot": completed_snapshot,
        "expected": {kind: sum(len(values) for values in rows.values()) for kind, rows in expected.items()},
        "completed": {
            "repairability": sum(len(values) for values in repair_completed.values()),
            "safety": sum(len(values) for values in safety_completed.values()),
        },
        "assigned_unfinished": {kind: len(values) for kind, values in global_owned.items()},
        "workspace_counts": {
            workspace: {
                "workers": len(partition["assignments"]),
                "repairability": sum(len(row["repair_ids"]) for row in partition["assignments"]),
                "safety": sum(len(row["safety_ids"]) for row in partition["assignments"]),
            }
            for workspace, partition in partitions.items()
        },
    }
    return partitions, summary


def _read_pack_rows(manifest_root: Path, kind: str, model_key: str) -> list[dict[str, Any]]:
    filename = "extension_execution_packs.jsonl" if kind == "repairability" else "execution_packs.jsonl"
    path = manifest_root / kind / "per_model" / model_key / filename
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def build_balanced_partitions(
    manifest_root: Path,
    completed_snapshots: list[Mapping[str, Any]],
    placement: Mapping[str, Mapping[str, int]],
    *,
    run_id: str,
    source_commit: str,
) -> tuple[dict[str, dict[str, Any]], dict[str, Any]]:
    """LPT-balance immutable unfinished packs over an explicit workspace plan.

    Rebalancing changes only worker ownership. Pack membership, model inputs,
    rollout seeds, parsing, verification, and output identities remain frozen.
    """
    completed: dict[str, dict[str, set[str]]] = {"repairability": {}, "safety": {}}
    for snapshot in completed_snapshots:
        for source_kind, kind in (("repairability", "repairability"), ("safety_features", "safety")):
            for model, values in snapshot.get(source_kind, {}).items():
                completed[kind].setdefault(str(model), set()).update(map(str, values))

    model_workers: dict[str, list[tuple[str, int]]] = {}
    for workspace, models in placement.items():
        if sum(int(count) for count in models.values()) > 10:
            raise ValueError(f"{workspace} requests more than ten workers")
        workspace_slot = 0
        for model, count in models.items():
            for _ in range(int(count)):
                model_workers.setdefault(str(model), []).append((str(workspace), workspace_slot))
                workspace_slot += 1

    assignments: dict[str, dict[int, dict[str, Any]]] = {
        workspace: {} for workspace in placement
    }
    expected_remaining: dict[str, set[str]] = {"repairability": set(), "safety": set()}
    balance: dict[str, Any] = {}
    for model, workers in model_workers.items():
        if not workers:
            raise ValueError(f"no workers allocated for {model}")
        bins = [
            {
                "workspace": workspace,
                "worker_slot": slot,
                "model_key": model,
                "repair_ids": [],
                "safety_ids": [],
                "estimated_work": 0,
            }
            for workspace, slot in workers
        ]
        pack_rows: list[tuple[int, str, str]] = []
        for kind in ("repairability", "safety"):
            rows = _read_pack_rows(manifest_root, kind, model)
            expected_ids = {str(row["pack_id"]) for row in rows}
            unknown = completed[kind].get(model, set()) - expected_ids
            if unknown:
                raise ValueError(f"unknown completed {kind}/{model} packs: {sorted(unknown)[:3]}")
            remaining = expected_ids - completed[kind].get(model, set())
            expected_remaining[kind].update(remaining)
            pack_rows.extend(
                (int(row.get("estimated_work", 1)), kind, str(row["pack_id"]))
                for row in rows
                if str(row["pack_id"]) in remaining
            )
        for work, kind, pack_id in sorted(pack_rows, key=lambda row: (-row[0], row[2])):
            target = min(bins, key=lambda row: (row["estimated_work"], row["workspace"], row["worker_slot"]))
            target["repair_ids" if kind == "repairability" else "safety_ids"].append(pack_id)
            target["estimated_work"] += work
        for row in bins:
            assignments[row["workspace"]][row["worker_slot"]] = row
        loads = [int(row["estimated_work"]) for row in bins]
        balance[model] = {
            "workers": len(bins),
            "remaining_packs": len(pack_rows),
            "minimum_estimated_work": min(loads),
            "maximum_estimated_work": max(loads),
            "load_ratio": max(loads) / max(1, min(loads)),
        }

    partitions: dict[str, dict[str, Any]] = {}
    owned: dict[str, set[str]] = {"repairability": set(), "safety": set()}
    for workspace in placement:
        rows = [assignments[workspace][slot] for slot in sorted(assignments[workspace])]
        for row in rows:
            for kind, field in (("repairability", "repair_ids"), ("safety", "safety_ids")):
                values = set(row[field])
                if owned[kind].intersection(values):
                    raise ValueError(f"duplicate {kind} ownership")
                owned[kind].update(values)
        base: dict[str, Any] = {
            "schema_version": 2,
            "run_id": run_id,
            "source_commit": source_commit,
            "workspace": workspace,
            "assignment_policy": "lpt_estimated_work_v1",
            "assignments": rows,
        }
        partitions[workspace] = {**base, "partition_hash": _hash_partition(base)}
    if owned != expected_remaining:
        raise ValueError("balanced partitions do not exactly cover all unfinished packs")
    summary = {
        "schema_version": 2,
        "run_id": run_id,
        "source_commit": source_commit,
        "assignment_policy": "lpt_estimated_work_v1",
        "completed": {kind: len(set().union(*models.values())) if models else 0 for kind, models in completed.items()},
        "remaining": {kind: len(values) for kind, values in expected_remaining.items()},
        "balance": balance,
        "workspace_counts": {
            workspace: {
                "workers": len(partition["assignments"]),
                "repairability": sum(len(row["repair_ids"]) for row in partition["assignments"]),
                "safety": sum(len(row["safety_ids"]) for row in partition["assignments"]),
                "estimated_work": sum(int(row["estimated_work"]) for row in partition["assignments"]),
            }
            for workspace, partition in partitions.items()
        },
    }
    return partitions, summary
