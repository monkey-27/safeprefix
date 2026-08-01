"""Immutable two-workspace ownership for threshold-generation packs."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Mapping

import pandas as pd

from safeprefix.reproducibility import atomic_json, stable_hash

from .data import MODEL_KEYS, read_jsonl, sha256_file
from .runtime import _valid_pack, production_assignments


SHARD_IDS = ("shard-00", "shard-01")
SHARD_SLOTS: dict[str, dict[str, tuple[int, ...]]] = {
    "shard-00": {
        "family_a_small": (0, 1),
        "family_a_large": (0,),
        "family_b_small": (0, 1, 2),
        "family_b_large": (0, 1, 2, 3),
    },
    "shard-01": {
        "family_a_small": (2,),
        "family_a_large": (1, 2),
        "family_b_small": (3, 4),
        "family_b_large": (4, 5, 6, 7, 8),
    },
}


def _binding_hashes(artifact_root: Path) -> dict[str, str]:
    root = Path(artifact_root)
    paths = {
        "frozen_protocol_sha256": root / "manifests/frozen_protocol.json",
        "execution_packs_sha256": root / "manifests/execution_packs.jsonl",
        "checkpoint_workload_sha256": root
        / "manifests/dense_checkpoint_rollout_manifest.parquet",
        "full_regeneration_workload_sha256": root
        / "manifests/full_regeneration_manifest.parquet",
    }
    missing = [str(path) for path in paths.values() if not path.is_file()]
    if missing:
        raise RuntimeError(f"prepared threshold bundle is incomplete: {missing}")
    return {name: sha256_file(path) for name, path in paths.items()}


def _write_frozen_json(path: Path, payload: Mapping[str, Any]) -> None:
    if path.is_file():
        existing = json.loads(path.read_text())
        if existing != dict(payload):
            raise RuntimeError(f"frozen shard artifact differs: {path}")
        return
    atomic_json(path, dict(payload))


def freeze_execution_shards(
    config: Mapping[str, Any],
    *,
    artifact_root: Path,
    run_id: str,
    source_digest: str,
    source_commit: str,
) -> dict[str, Any]:
    """Freeze two ten-slot shards without changing scientific pack identity."""
    root = Path(artifact_root)
    frozen_protocol = json.loads((root / "manifests/frozen_protocol.json").read_text())
    configuration_hash = stable_hash(config)
    if frozen_protocol.get("configuration_hash") != configuration_hash:
        raise RuntimeError("prepared bundle configuration hash differs")
    assignments = production_assignments(config, root)
    packs = read_jsonl(root / "manifests/execution_packs.jsonl")
    pack_by_id = {str(row["pack_id"]): row for row in packs}
    bindings = _binding_hashes(root)
    shard_payloads: dict[str, dict[str, Any]] = {}
    for shard_id in SHARD_IDS:
        slots = []
        for model_key in MODEL_KEYS:
            for slot in SHARD_SLOTS[shard_id][model_key]:
                pack_ids = list(assignments[model_key][slot])
                slots.append(
                    {
                        "model_key": model_key,
                        "worker_slot": int(slot),
                        "pack_ids": pack_ids,
                        "pack_count": len(pack_ids),
                        "estimated_work": int(
                            sum(int(pack_by_id[pack_id]["estimated_work"]) for pack_id in pack_ids)
                        ),
                    }
                )
        identity: dict[str, Any] = {
            "schema_version": 1,
            "shard_id": shard_id,
            "run_id": str(run_id),
            "source_digest": str(source_digest),
            "source_commit": str(source_commit),
            "configuration_hash": configuration_hash,
            **bindings,
            "slots": slots,
            "slot_count": len(slots),
            "pack_count": sum(int(row["pack_count"]) for row in slots),
            "estimated_work": sum(int(row["estimated_work"]) for row in slots),
        }
        payload = {**identity, "shard_hash": stable_hash(identity)}
        _write_frozen_json(root / f"manifests/execution_shards/{shard_id}.json", payload)
        shard_payloads[shard_id] = payload

    master_identity: dict[str, Any] = {
        "schema_version": 1,
        "run_id": str(run_id),
        "source_digest": str(source_digest),
        "source_commit": str(source_commit),
        "configuration_hash": configuration_hash,
        **bindings,
        "shards": {
            shard_id: {
                "shard_hash": shard_payloads[shard_id]["shard_hash"],
                "slot_count": shard_payloads[shard_id]["slot_count"],
                "pack_count": shard_payloads[shard_id]["pack_count"],
                "estimated_work": shard_payloads[shard_id]["estimated_work"],
            }
            for shard_id in SHARD_IDS
        },
        "total_slots": sum(int(row["slot_count"]) for row in shard_payloads.values()),
        "total_packs": sum(int(row["pack_count"]) for row in shard_payloads.values()),
        "total_estimated_work": sum(
            int(row["estimated_work"]) for row in shard_payloads.values()
        ),
    }
    master = {**master_identity, "master_hash": stable_hash(master_identity)}
    _write_frozen_json(root / "manifests/execution_shards.json", master)
    return validate_execution_shards(
        config,
        artifact_root=root,
        run_id=run_id,
        source_digest=source_digest,
        source_commit=source_commit,
    )


def validate_execution_shards(
    config: Mapping[str, Any],
    *,
    artifact_root: Path,
    run_id: str,
    source_digest: str,
    source_commit: str,
) -> dict[str, Any]:
    """Validate every binding and require exact once-only global pack ownership."""
    root = Path(artifact_root)
    master_path = root / "manifests/execution_shards.json"
    if not master_path.is_file():
        raise RuntimeError("prepared threshold bundle lacks execution shard manifest")
    master = json.loads(master_path.read_text())
    bindings = _binding_hashes(root)
    expected_header = {
        "run_id": str(run_id),
        "source_digest": str(source_digest),
        "source_commit": str(source_commit),
        "configuration_hash": stable_hash(config),
        **bindings,
    }
    for key, expected in expected_header.items():
        if master.get(key) != expected:
            raise RuntimeError(f"execution shard binding differs for {key}")
    master_identity = {key: value for key, value in master.items() if key != "master_hash"}
    if master.get("master_hash") != stable_hash(master_identity):
        raise RuntimeError("execution shard master hash differs")

    assignments = production_assignments(config, root)
    expected_packs = {
        str(row["pack_id"]) for row in read_jsonl(root / "manifests/execution_packs.jsonl")
    }
    owned: list[str] = []
    shard_payloads: dict[str, dict[str, Any]] = {}
    for shard_id in SHARD_IDS:
        path = root / f"manifests/execution_shards/{shard_id}.json"
        payload = json.loads(path.read_text())
        identity = {key: value for key, value in payload.items() if key != "shard_hash"}
        if payload.get("shard_hash") != stable_hash(identity):
            raise RuntimeError(f"{shard_id}: shard hash differs")
        for key, expected in {"shard_id": shard_id, **expected_header}.items():
            if payload.get(key) != expected:
                raise RuntimeError(f"{shard_id}: binding differs for {key}")
        expected_slots = [
            (model_key, slot)
            for model_key in MODEL_KEYS
            for slot in SHARD_SLOTS[shard_id][model_key]
        ]
        observed_slots = [
            (str(row["model_key"]), int(row["worker_slot"])) for row in payload["slots"]
        ]
        if observed_slots != expected_slots or int(payload.get("slot_count", -1)) != 10:
            raise RuntimeError(f"{shard_id}: slot ownership differs")
        for row in payload["slots"]:
            model_key = str(row["model_key"])
            slot = int(row["worker_slot"])
            if list(row["pack_ids"]) != list(assignments[model_key][slot]):
                raise RuntimeError(f"{shard_id}/{model_key}/{slot}: pack assignment differs")
            owned.extend(map(str, row["pack_ids"]))
        shard_payloads[shard_id] = payload
        if master.get("shards", {}).get(shard_id, {}).get("shard_hash") != payload["shard_hash"]:
            raise RuntimeError(f"{shard_id}: master binding differs")
    if len(owned) != len(set(owned)) or set(owned) != expected_packs:
        raise RuntimeError("execution shards do not own every pack exactly once")
    if int(master.get("total_slots", -1)) != 20 or int(master.get("total_packs", -1)) != len(owned):
        raise RuntimeError("execution shard global counts differ")
    return {"master": master, "shards": shard_payloads}


def validate_shard_results(
    config: Mapping[str, Any], artifact_root: Path, shard: Mapping[str, Any]
) -> dict[str, int]:
    """Run the full immutable pack validator for every pack owned by one shard."""
    root = Path(artifact_root)
    packs = {
        str(row["pack_id"]): row for row in read_jsonl(root / "manifests/execution_packs.jsonl")
    }
    checkpoint_manifest = pd.read_parquet(root / "manifests/dense_checkpoint_rollout_manifest.parquet")
    regeneration_manifest = pd.read_parquet(root / "manifests/full_regeneration_manifest.parquet")
    valid = 0
    expected = 0
    for slot in shard["slots"]:
        model_key = str(slot["model_key"])
        for pack_id in map(str, slot["pack_ids"]):
            expected += 1
            pack = packs[pack_id]
            pack_root = root / f"raw_outcomes/production_packs/{model_key}/{pack_id}"
            if _valid_pack(
                pack,
                pack_root,
                added_rollouts=12,
                full_regenerations=16,
                scientific=True,
                checkpoint_manifest=checkpoint_manifest[
                    checkpoint_manifest["pack_id"].astype(str).eq(pack_id)
                ],
                regeneration_manifest=regeneration_manifest[
                    regeneration_manifest["pack_id"].astype(str).eq(pack_id)
                ],
            ):
                valid += 1
    return {"expected_packs": expected, "valid_packs": valid}
