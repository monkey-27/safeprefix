"""Exact resumable K=16 suffix and full-regeneration execution."""

from __future__ import annotations

from dataclasses import dataclass
import json
import os
from pathlib import Path
import time
from typing import Any, Mapping, Sequence

import pandas as pd
import torch

from safeprefix.manifests import decode_reference_answer
from safeprefix.models.cache_checkpoint import CacheCheckpoint, tokenizer_checkpoint_metadata
from safeprefix.models.generation import capture_prompt_checkpoint
from safeprefix.models.teacher_forcing import teacher_force_token_ids_chunked
from safeprefix.parsing.answer_parsers import parse_answer_region
from safeprefix.reproducibility import atomic_json, atomic_parquet, now_iso, stable_hash, stable_seed
from safeprefix.rollout.production_engine import decode_execution_pack
from safeprefix.rollout.verifier import resolve_verifier

from .data import MODEL_KEYS, read_jsonl, row_artifact_hash, sha256_file


@dataclass(frozen=True)
class ThresholdRolloutRequest:
    """Production-engine compatible request with threshold-specific indices."""

    branch_id: str
    rollout_seed: int
    rollout_index: int
    checkpoint: CacheCheckpoint
    metadata: Mapping[str, Any]

    def __post_init__(self) -> None:
        if not self.branch_id:
            raise ValueError("branch_id must be nonempty")
        if int(self.rollout_index) < 0:
            raise ValueError("rollout index must be nonnegative")
        self.checkpoint.validate()


def _commit_output_volume() -> None:
    if not os.environ.get("MODAL_TASK_ID"):
        return
    name = os.environ.get("SAFEPREFIX_THRESHOLD_OUTPUT_VOLUME")
    if not name:
        raise RuntimeError("threshold output volume name is not configured")
    import modal

    modal.Volume.from_name(name).commit()


def _row_hash(row: Mapping[str, Any]) -> str:
    return row_artifact_hash(row)


def _pack_root(artifact_root: Path, model_key: str, pack_id: str, *, smoke: bool) -> Path:
    prefix = "smoke/raw_packs" if smoke else "raw_outcomes/production_packs"
    return Path(artifact_root) / prefix / model_key / pack_id


def _valid_pack(
    pack: Mapping[str, Any],
    root: Path,
    *,
    added_rollouts: int,
    full_regenerations: int,
    scientific: bool,
    checkpoint_manifest: pd.DataFrame,
    regeneration_manifest: pd.DataFrame,
) -> bool:
    marker_path = root / "complete.json"
    dense_path = root / "added_checkpoint_suffixes.parquet"
    regeneration_path = root / "full_regenerations.parquet"
    if not all(path.is_file() for path in (marker_path, dense_path, regeneration_path)):
        return False
    try:
        marker = json.loads(marker_path.read_text())
        if marker.get("status") != "COMPLETE":
            return False
        if marker.get("scientific") is not bool(scientific):
            return False
        if marker.get("pack_hash") != pack["pack_hash"]:
            return False
        if marker.get("dense_sha256") != sha256_file(dense_path):
            return False
        if marker.get("full_regeneration_sha256") != sha256_file(regeneration_path):
            return False
        dense = pd.read_parquet(dense_path)
        regeneration = pd.read_parquet(regeneration_path)
        if len(dense) != int(pack["checkpoint_count"]) * int(added_rollouts):
            return False
        if len(regeneration) != int(pack["trace_count"]) * int(full_regenerations):
            return False
        if not dense["infrastructure_status"].eq("executed").all():
            return False
        if not regeneration["infrastructure_status"].eq("executed").all():
            return False
        if not dense["scientific"].eq(bool(scientific)).all():
            return False
        if not regeneration["scientific"].eq(bool(scientific)).all():
            return False
        if set(dense["pack_id"].astype(str)) != {str(pack["pack_id"])}:
            return False
        if set(regeneration["pack_id"].astype(str)) != {str(pack["pack_id"])}:
            return False
        if dense.duplicated(["trace_id", "checkpoint_ordinal", "rollout_index"]).any():
            return False
        if regeneration.duplicated(["trace_id", "rollout_index"]).any():
            return False
        if dense["rollout_seed"].duplicated().any() or regeneration["rollout_seed"].duplicated().any():
            return False
        if dense["artifact_hash"].duplicated().any() or regeneration["artifact_hash"].duplicated().any():
            return False
        expected_dense = set(
            zip(
                checkpoint_manifest["base_model"].astype(str),
                checkpoint_manifest["trace_id"].astype(str),
                checkpoint_manifest["checkpoint_id"].astype(str),
                checkpoint_manifest["checkpoint_ordinal"].astype(int),
                checkpoint_manifest["rollout_index"].astype(int),
                checkpoint_manifest["rollout_seed"].astype(int),
            )
        )
        observed_dense = set(
            zip(
                dense["model_key"].astype(str),
                dense["trace_id"].astype(str),
                dense["checkpoint_id"].astype(str),
                dense["checkpoint_ordinal"].astype(int),
                dense["rollout_index"].astype(int),
                dense["rollout_seed"].astype(int),
            )
        )
        expected_full = set(
            zip(
                regeneration_manifest["base_model"].astype(str),
                regeneration_manifest["trace_id"].astype(str),
                regeneration_manifest["rollout_index"].astype(int),
                regeneration_manifest["rollout_seed"].astype(int),
            )
        )
        observed_full = set(
            zip(
                regeneration["model_key"].astype(str),
                regeneration["trace_id"].astype(str),
                regeneration["rollout_index"].astype(int),
                regeneration["rollout_seed"].astype(int),
            )
        )
        if observed_dense != expected_dense or observed_full != expected_full:
            return False
        if any(
            row_artifact_hash(row) != str(row["artifact_hash"])
            for row in dense.to_dict("records")
        ):
            return False
        if any(
            row_artifact_hash(row) != str(row["artifact_hash"])
            for row in regeneration.to_dict("records")
        ):
            return False
        return True
    except Exception:
        return False


def _load_context(
    config: Mapping[str, Any],
    *,
    artifact_root: Path,
    model_key: str,
    pack_id: str,
    smoke: bool,
) -> tuple[dict[str, Any], dict[str, dict[str, Any]], pd.DataFrame, pd.DataFrame]:
    if model_key not in MODEL_KEYS:
        raise ValueError(f"unknown model key: {model_key}")
    packs = {row["pack_id"]: row for row in read_jsonl(Path(artifact_root) / "manifests/execution_packs.jsonl")}
    if pack_id not in packs:
        raise KeyError(f"unknown threshold execution pack: {pack_id}")
    pack = dict(packs[pack_id])
    if str(pack.get("model_key")) != model_key:
        raise RuntimeError("threshold execution pack belongs to another model")
    traces = {
        str(row["trace_id"]): row
        for row in read_jsonl(Path(artifact_root) / f"manifests/source_traces/{model_key}.jsonl")
    }
    checkpoint_manifest = pd.read_parquet(
        Path(artifact_root) / "manifests/dense_checkpoint_rollout_manifest.parquet",
        filters=[("base_model", "==", model_key), ("pack_id", "==", pack_id)],
    )
    regeneration_manifest = pd.read_parquet(
        Path(artifact_root) / "manifests/full_regeneration_manifest.parquet",
        filters=[("base_model", "==", model_key), ("pack_id", "==", pack_id)],
    )
    if smoke:
        chosen = list(pack["trace_ids"])[: int(config["smoke"]["traces_per_model"])]
        checkpoint_manifest = checkpoint_manifest[checkpoint_manifest["trace_id"].isin(chosen)].copy()
        checkpoint_manifest = checkpoint_manifest[
            checkpoint_manifest["rollout_index"].isin(list(range(4, 4 + int(config["smoke"]["added_checkpoint_rollouts"]))))
        ].copy()
        regeneration_manifest = regeneration_manifest[regeneration_manifest["trace_id"].isin(chosen)].copy()
        regeneration_manifest = regeneration_manifest[
            regeneration_manifest["rollout_index"].isin(list(range(int(config["smoke"]["full_regenerations"]))))
        ].copy()
        checkpoint_manifest["rollout_seed"] = [
            stable_seed("threshold-smoke-checkpoint-v1", model_key, row.trace_id, int(row.checkpoint_ordinal), int(row.rollout_index))
            for row in checkpoint_manifest.itertuples(index=False)
        ]
        regeneration_manifest["rollout_seed"] = [
            stable_seed("threshold-smoke-full-v1", model_key, row.trace_id, int(row.rollout_index))
            for row in regeneration_manifest.itertuples(index=False)
        ]
        identity = {
            "schema_version": 1,
            "model_key": model_key,
            "trace_ids": chosen,
            "checkpoint_rollout_indices": sorted(checkpoint_manifest["rollout_index"].astype(int).unique()),
            "full_regeneration_indices": sorted(regeneration_manifest["rollout_index"].astype(int).unique()),
            "scientific": False,
        }
        pack = {
            **identity,
            "pack_id": f"smoke-{model_key}",
            "pack_hash": stable_hash(identity),
            "trace_count": len(chosen),
            "checkpoint_count": int(checkpoint_manifest["checkpoint_id"].nunique()),
        }
    expected_traces = set(checkpoint_manifest["trace_id"].astype(str))
    if expected_traces != set(regeneration_manifest["trace_id"].astype(str)):
        raise RuntimeError("checkpoint and full-regeneration trace membership differs")
    subset = {trace_id: traces[trace_id] for trace_id in expected_traces}
    if expected_traces != set(map(str, pack["trace_ids"])):
        raise RuntimeError("pack trace membership differs from its frozen identity")
    return pack, subset, checkpoint_manifest, regeneration_manifest


def execute_threshold_pack(
    config: Mapping[str, Any],
    *,
    artifact_root: Path,
    model_key: str,
    pack_id: str,
    loaded: Any,
    smoke: bool = False,
) -> dict[str, Any]:
    """Execute one immutable pack; incorrect/parser-failed outputs remain valid rows."""
    pack, traces, checkpoint_manifest, regeneration_manifest = _load_context(
        config,
        artifact_root=artifact_root,
        model_key=model_key,
        pack_id=pack_id,
        smoke=smoke,
    )
    added_per_checkpoint = int(
        config["smoke"]["added_checkpoint_rollouts"] if smoke else 12
    )
    full_per_trace = int(config["smoke"]["full_regenerations"] if smoke else 16)
    output_root = _pack_root(artifact_root, model_key, str(pack["pack_id"]), smoke=smoke)
    if _valid_pack(
        pack,
        output_root,
        added_rollouts=added_per_checkpoint,
        full_regenerations=full_per_trace,
        scientific=not smoke,
        checkpoint_manifest=checkpoint_manifest,
        regeneration_manifest=regeneration_manifest,
    ):
        marker = json.loads((output_root / "complete.json").read_text())
        return {**marker, "status": "SKIPPED_VALID", "resumed_from_marker": True}

    started = time.perf_counter()
    model, tokenizer = loaded.model, loaded.tokenizer
    model_config = config["models"][model_key]
    if (
        str(loaded.model_id) != str(model_config["hf_model_id"])
        or str(loaded.model_revision) != str(model_config["revision"])
        or str(loaded.tokenizer_revision) != str(model_config["tokenizer_revision"])
    ):
        raise RuntimeError("loaded model/tokenizer identity differs from the frozen configuration")
    generation = dict(config["generation"])
    configured_context = int(config["models"][model_key]["max_context_length"])
    native_context = int(getattr(model.config, "max_position_embeddings", configured_context))
    if configured_context > native_context:
        raise RuntimeError("configured context exceeds pinned model native context")
    verifier = resolve_verifier(str(generation["verifier"]))
    checkpoint_lookup = {
        (str(row.trace_id), int(row.checkpoint_ordinal), int(row.rollout_index)): row
        for row in checkpoint_manifest.itertuples(index=False)
    }
    regeneration_lookup = {
        (str(row.trace_id), int(row.rollout_index)): row
        for row in regeneration_manifest.itertuples(index=False)
    }
    identity_rows = checkpoint_manifest.drop_duplicates("trace_id").set_index("trace_id")
    requests: list[ThresholdRolloutRequest] = []
    prefill_seconds = 0.0
    for trace_id in sorted(traces):
        source = traces[trace_id]
        prompt_ids = list(map(int, source["prompt_token_ids"]))
        completion_ids = list(map(int, source["completion_token_ids"]))
        offsets = list(map(int, source["eligible_checkpoint_offsets"]))
        prefill_started = time.perf_counter()
        forced = teacher_force_token_ids_chunked(
            model,
            prompt_ids + completion_ids,
            prompt_count=len(prompt_ids),
            chunk_size=int(config["execution"]["prefill_chunk_size"]),
            selected_layers=(),
            selected_token_offsets=(),
            selected_checkpoint_offsets=offsets,
        )
        prefill_seconds += time.perf_counter() - prefill_started
        for checkpoint_ordinal, token_offset in enumerate(offsets):
            checkpoint = forced.cache_checkpoint(
                token_offset,
                model_id=loaded.model_id,
                model_revision=loaded.model_revision,
                tokenizer_id=loaded.tokenizer_id,
                tokenizer_revision=loaded.tokenizer_revision,
                tokenizer_metadata=tokenizer_checkpoint_metadata(tokenizer),
                generation_metadata=generation,
                clone_cache=False,
            )
            keys = sorted(
                [key for key in checkpoint_lookup if key[:2] == (trace_id, checkpoint_ordinal)],
                key=lambda key: key[2],
            )
            if len(keys) != added_per_checkpoint:
                raise RuntimeError("pack checkpoint added-rollout count differs")
            for key in keys:
                logical = checkpoint_lookup[key]
                requests.append(
                    ThresholdRolloutRequest(
                        branch_id=f"threshold:{model_key}:{trace_id}:{checkpoint_ordinal}:{int(logical.rollout_index)}",
                        rollout_seed=int(logical.rollout_seed),
                        rollout_index=int(logical.rollout_index),
                        checkpoint=checkpoint,
                        metadata={
                            "stage": "checkpoint",
                            "trace_id": trace_id,
                            "checkpoint_ordinal": checkpoint_ordinal,
                            "checkpoint_token_offset": token_offset,
                        },
                    )
                )
        # This prefill receives only the original problem/chat prompt.  The
        # failed teacher-forced completion and reference are not arguments.
        prompt_checkpoint, prompt_seconds, _ = capture_prompt_checkpoint(
            model,
            tokenizer,
            torch.tensor([prompt_ids], dtype=torch.long),
            model_id=loaded.model_id,
            model_revision=loaded.model_revision,
            tokenizer_id=loaded.tokenizer_id,
            tokenizer_revision=loaded.tokenizer_revision,
            generation=generation,
            selected_hidden_layers=(),
        )
        prefill_seconds += prompt_seconds
        full_keys = sorted(
            [key for key in regeneration_lookup if key[0] == trace_id], key=lambda key: key[1]
        )
        if len(full_keys) != full_per_trace:
            raise RuntimeError("pack full-regeneration count differs")
        for key in full_keys:
            logical = regeneration_lookup[key]
            metadata = {"stage": "full_regeneration", "trace_id": trace_id}
            if set(metadata) & {"reference_answer", "failed_trace", "original_wrong_answer", "verifier_feedback", "gold_answer"}:
                raise AssertionError("full-regeneration request exposed prohibited information")
            requests.append(
                ThresholdRolloutRequest(
                    branch_id=f"threshold-full:{model_key}:{trace_id}:{int(logical.rollout_index)}",
                    rollout_seed=int(logical.rollout_seed),
                    rollout_index=int(logical.rollout_index),
                    checkpoint=prompt_checkpoint,
                    metadata=metadata,
                )
            )
        del forced

    results, decode_metrics = decode_execution_pack(
        model,
        tokenizer,
        requests,
        generation=generation,
        maximum_batch_size=int(config["execution"]["branch_batch_sizes"][model_key]),
        compaction_quantum=int(config["execution"]["compaction_quantum"]),
        maximum_context_length=configured_context,
        maximum_decode_kv_bytes=int(config["execution"]["maximum_decode_kv_bytes"][model_key]),
    )
    dense_rows: list[dict[str, Any]] = []
    full_rows: list[dict[str, Any]] = []
    for result in results:
        trace_id = str(result.request.metadata["trace_id"])
        source = traces[trace_id]
        identity = identity_rows.loc[trace_id]
        parsed = parse_answer_region(result.text)
        truncation = str(result.stop_reason) == "length"
        verifier_pass = bool(
            parsed.success
            and not truncation
            and verifier(parsed.parsed_answer, decode_reference_answer(source["reference_answer"]), {})
        )
        base = {
            "schema_version": 1,
            "scientific": not smoke,
            "model_key": model_key,
            "model_id": loaded.model_id,
            "model_revision": loaded.model_revision,
            "tokenizer_id": loaded.tokenizer_id,
            "tokenizer_revision": loaded.tokenizer_revision,
            "trace_id": trace_id,
            "common_trace_id": str(identity["common_trace_id"]),
            "source_trace_id": source["source_trace_id"],
            "problem_id": source["problem_id"],
            "problem_group": source["production_problem_group"],
            "dataset": source["source_dataset"],
            "domain": source["source_bucket"],
            "pack_id": pack["pack_id"],
            "pack_hash": pack["pack_hash"],
            "rollout_index": int(result.request.rollout_index),
            "rollout_seed": int(result.request.rollout_seed),
            "raw_suffix": result.text,
            "generated_token_ids": list(map(int, result.token_ids)),
            "generated_token_count": len(result.token_ids),
            "normalized_final_answer": None if not parsed.success else str(parsed.parsed_answer),
            "parser_status": "success" if parsed.success else "failure",
            "parser_method": parsed.method,
            "truncation_status": bool(truncation),
            "stop_reason": str(result.stop_reason),
            "verifier_outcome": verifier_pass,
            "binary_outcome": verifier_pass,
            "infrastructure_status": "executed",
            "latency_seconds": float(result.latency_seconds),
        }
        if str(result.request.metadata["stage"]) == "checkpoint":
            logical = checkpoint_lookup[
                (
                    trace_id,
                    int(result.request.metadata["checkpoint_ordinal"]),
                    int(result.request.rollout_index),
                )
            ]
            base.update(
                {
                    "stage": "added_checkpoint_suffix",
                    "checkpoint_id": str(logical.checkpoint_id),
                    "checkpoint_ordinal": int(result.request.metadata["checkpoint_ordinal"]),
                    "checkpoint_token_offset": int(result.request.metadata["checkpoint_token_offset"]),
                }
            )
            base["artifact_hash"] = _row_hash(base)
            dense_rows.append(base)
        else:
            base.update(
                {
                    "stage": "full_regeneration",
                    "prompt_input_token_count": len(source["prompt_token_ids"]),
                    "checkpoint_id": None,
                    "checkpoint_ordinal": None,
                    "checkpoint_token_offset": len(source["prompt_token_ids"]),
                }
            )
            base["artifact_hash"] = _row_hash(base)
            full_rows.append(base)
    dense = pd.DataFrame(dense_rows).sort_values(
        ["trace_id", "checkpoint_ordinal", "rollout_index"]
    )
    full = pd.DataFrame(full_rows).sort_values(["trace_id", "rollout_index"])
    if len(dense) != int(pack["checkpoint_count"]) * added_per_checkpoint:
        raise RuntimeError("completed added-checkpoint row count differs")
    if len(full) != int(pack["trace_count"]) * full_per_trace:
        raise RuntimeError("completed full-regeneration row count differs")
    expected_dense = set(
        zip(
            checkpoint_manifest["trace_id"].astype(str),
            checkpoint_manifest["checkpoint_ordinal"].astype(int),
            checkpoint_manifest["rollout_index"].astype(int),
            checkpoint_manifest["rollout_seed"].astype(int),
        )
    )
    observed_dense = set(
        zip(
            dense["trace_id"].astype(str), dense["checkpoint_ordinal"].astype(int),
            dense["rollout_index"].astype(int), dense["rollout_seed"].astype(int),
        )
    )
    expected_full = set(
        zip(
            regeneration_manifest["trace_id"].astype(str),
            regeneration_manifest["rollout_index"].astype(int),
            regeneration_manifest["rollout_seed"].astype(int),
        )
    )
    observed_full = set(
        zip(full["trace_id"].astype(str), full["rollout_index"].astype(int), full["rollout_seed"].astype(int))
    )
    if observed_dense != expected_dense or observed_full != expected_full:
        raise RuntimeError("completed logical seed keys differ from frozen manifests")
    output_root.mkdir(parents=True, exist_ok=True)
    dense_path = output_root / "added_checkpoint_suffixes.parquet"
    full_path = output_root / "full_regenerations.parquet"
    atomic_parquet(dense_path, dense)
    atomic_parquet(full_path, full)
    marker = {
        "status": "COMPLETE",
        "completed_at": now_iso(),
        "scientific": not smoke,
        "model_key": model_key,
        "pack_id": pack["pack_id"],
        "pack_hash": pack["pack_hash"],
        "trace_count": int(pack["trace_count"]),
        "checkpoint_count": int(pack["checkpoint_count"]),
        "added_checkpoint_rollouts": len(dense),
        "full_regenerations": len(full),
        "dense_sha256": sha256_file(dense_path),
        "full_regeneration_sha256": sha256_file(full_path),
        "prefill_seconds": prefill_seconds,
        "decode_metrics": decode_metrics.to_dict(),
        "wall_seconds": time.perf_counter() - started,
        "native_evaluation_used": False,
        "teacher_forced_test_used": False,
    }
    atomic_json(output_root / "complete.json", marker)
    _commit_output_volume()
    if not _valid_pack(
        pack,
        output_root,
        added_rollouts=added_per_checkpoint,
        full_regenerations=full_per_trace,
        scientific=not smoke,
        checkpoint_manifest=checkpoint_manifest,
        regeneration_manifest=regeneration_manifest,
    ):
        raise RuntimeError("threshold execution pack failed post-write validation")
    return marker


def production_assignments(config: Mapping[str, Any], artifact_root: Path) -> dict[str, dict[int, list[str]]]:
    """Longest-processing-time assignment to frozen per-model worker slots."""
    packs = read_jsonl(Path(artifact_root) / "manifests/execution_packs.jsonl")
    result: dict[str, dict[int, list[str]]] = {}
    for model_key in MODEL_KEYS:
        worker_count = int(config["execution"]["gpu_workers_by_model"][model_key])
        slots = {index: [] for index in range(worker_count)}
        loads = [0] * worker_count
        for pack in sorted(
            [row for row in packs if row["model_key"] == model_key],
            key=lambda row: (-int(row["estimated_work"]), str(row["pack_id"])),
        ):
            target = min(range(worker_count), key=lambda index: (loads[index], index))
            slots[target].append(str(pack["pack_id"]))
            loads[target] += int(pack["estimated_work"])
        result[model_key] = slots
    return result
