"""Segmentation-independent hidden-state cache for frozen native failures.

The cache stores the final transformer-layer vector at the prompt boundary and
at every generated completion token.  This permits later deterministic
reasoning segmentation without another model forward and without freezing an
unapproved segmenter during data acquisition.  No K/V cache, boundary label,
boundary prediction, or repair continuation is produced here.
"""

from __future__ import annotations

import hashlib
import json
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence

import torch

from safeprefix.models.teacher_forcing import teacher_force_token_ids_chunked
from safeprefix.reproducibility import atomic_json, atomic_jsonl, stable_hash


@dataclass(frozen=True)
class NativeHiddenStatePack:
    pack_id: str
    model_key: str
    trace_ids: tuple[str, ...]
    configuration_hash: str

    def to_dict(self) -> dict[str, Any]:
        return {**asdict(self), "trace_ids": list(self.trace_ids)}


def sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def read_jsonl(path: str | Path) -> list[dict[str, Any]]:
    return [
        json.loads(line)
        for line in Path(path).read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def build_hidden_state_packs(
    cohort: Sequence[Mapping[str, Any]], *, model_key: str,
    pack_size: int, configuration_hash: str,
) -> list[dict[str, Any]]:
    if int(pack_size) < 1:
        raise ValueError("hidden-state pack size must be positive")
    trace_ids = [str(row["trace_id"]) for row in cohort]
    if len(trace_ids) != len(set(trace_ids)):
        raise RuntimeError("frozen native cohort has duplicate trace IDs")
    packs = []
    for start in range(0, len(trace_ids), int(pack_size)):
        members = tuple(trace_ids[start : start + int(pack_size)])
        packs.append(
            NativeHiddenStatePack(
                pack_id=stable_hash(
                    ["native-hidden-state-pack-v1", model_key, configuration_hash, members]
                )[:24],
                model_key=model_key,
                trace_ids=members,
                configuration_hash=configuration_hash,
            ).to_dict()
        )
    if [trace_id for pack in packs for trace_id in pack["trace_ids"]] != trace_ids:
        raise AssertionError("hidden-state packs changed frozen cohort order")
    return packs


def validate_source_cohort(
    cohort: Sequence[Mapping[str, Any]], *, model_key: str,
    model_revision: str, tokenizer_revision: str,
) -> dict[str, Any]:
    if not cohort:
        raise RuntimeError("native hidden-state source cohort is empty")
    trace_ids = [str(row["trace_id"]) for row in cohort]
    if len(trace_ids) != len(set(trace_ids)):
        raise RuntimeError("duplicate trace in native hidden-state source cohort")
    for row in cohort:
        if str(row["model_key"]) != model_key:
            raise RuntimeError("cross-model trace entered native hidden-state cohort")
        if str(row["model_revision"]) != model_revision:
            raise RuntimeError("native trace model revision differs from extraction model")
        if str(row["tokenizer_revision"]) != tokenizer_revision:
            raise RuntimeError("native trace tokenizer revision differs from extraction tokenizer")
        if row.get("initial_status") != "valid_incorrect":
            raise RuntimeError("native hidden-state cohort contains a non-failure")
        if row.get("parser_status") != "parsed" or row.get("verifier_status") != "completed":
            raise RuntimeError("native hidden-state cohort contains an unparsed or unverifiable trace")
        if bool(row.get("truncation_flag")):
            raise RuntimeError("native hidden-state cohort contains a truncated trace")
        if not row.get("prompt_token_ids") or not row.get("completion_token_ids"):
            raise RuntimeError("native hidden-state trace lacks exact saved token IDs")
    return {
        "status": "PASS",
        "model_key": model_key,
        "trace_count": len(cohort),
        "unique_trace_count": len(set(trace_ids)),
        "prompt_tokens": sum(len(row["prompt_token_ids"]) for row in cohort),
        "completion_tokens": sum(len(row["completion_token_ids"]) for row in cohort),
        "model_revision": model_revision,
        "tokenizer_revision": tokenizer_revision,
    }


def extract_trace_hidden_states(
    *, model: Any, trace: Mapping[str, Any], chunk_size: int,
    selected_layer: int = -1,
) -> tuple[dict[str, Any], dict[str, Any]]:
    prompt_ids = list(map(int, trace["prompt_token_ids"]))
    completion_ids = list(map(int, trace["completion_token_ids"]))
    token_ids = prompt_ids + completion_ids
    prompt_count = len(prompt_ids)
    # Row zero is the prompt-only state. Each subsequent row is aligned with
    # one exact completion token, including the answer region. Later code can
    # select any causal prefix boundary without retokenization.
    offsets = [prompt_count - 1, *range(prompt_count, len(token_ids))]
    started = time.perf_counter()
    forced = teacher_force_token_ids_chunked(
        model,
        token_ids,
        prompt_count=prompt_count,
        chunk_size=int(chunk_size),
        selected_layers=[int(selected_layer)],
        selected_token_offsets=offsets,
    )
    layer_values = forced.selected_hidden_states[int(selected_layer)]
    hidden = torch.stack([layer_values[offset] for offset in offsets]).to(torch.float16).contiguous()
    completion_log_probs = forced.token_log_probabilities[
        max(prompt_count - 1, 0) : max(len(token_ids) - 1, 0)
    ].to(torch.float32).contiguous()
    if hidden.shape[0] != len(completion_ids) + 1:
        raise AssertionError("hidden-state rows do not align to prompt root plus completion tokens")
    if completion_log_probs.shape[0] != len(completion_ids):
        raise AssertionError("completion log probabilities do not align to completion tokens")
    if not torch.isfinite(hidden.float()).all():
        raise RuntimeError("non-finite hidden state encountered")
    payload = {
        "schema_version": 1,
        "trace_id": str(trace["trace_id"]),
        "model_key": str(trace["model_key"]),
        "model_id": str(trace["model_id"]),
        "model_revision": str(trace["model_revision"]),
        "tokenizer_id": str(trace["tokenizer_id"]),
        "tokenizer_revision": str(trace["tokenizer_revision"]),
        "extraction_method": "posthoc_teacher_force_exact_saved_native_tokens",
        "selected_layer": int(selected_layer),
        "storage_dtype": "float16",
        "token_scope": "prompt_final_plus_every_completion_token",
        "prompt_token_count": prompt_count,
        "completion_token_count": len(completion_ids),
        "prompt_token_ids_hash": stable_hash(prompt_ids),
        "completion_token_ids_hash": stable_hash(completion_ids),
        "hidden_states": hidden,
        "completion_token_log_probabilities": completion_log_probs,
        "kv_cache_persisted": False,
    }
    metadata = {
        key: value
        for key, value in payload.items()
        if key not in {"hidden_states", "completion_token_log_probabilities"}
    }
    metadata.update(
        hidden_state_shape=list(hidden.shape),
        hidden_state_dtype=str(hidden.dtype),
        log_probability_count=int(completion_log_probs.shape[0]),
        extraction_seconds=time.perf_counter() - started,
    )
    del forced
    return payload, metadata


def _atomic_torch_save(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    torch.save(dict(payload), temporary)
    temporary.replace(path)


def validate_feature_pack(
    pack_root: str | Path, *, pack: Mapping[str, Any],
    cohort_index: Mapping[str, Mapping[str, Any]],
    expected_model_revision: str, expected_tokenizer_revision: str,
) -> dict[str, Any]:
    root = Path(pack_root)
    feature_path = root / "hidden_states.pt"
    metadata_path = root / "trace_metadata.jsonl"
    marker_path = root / "complete.json"
    if not (feature_path.is_file() and metadata_path.is_file() and marker_path.is_file()):
        raise RuntimeError(f"incomplete native hidden-state pack: {root}")
    marker = json.loads(marker_path.read_text(encoding="utf-8"))
    if marker.get("status") != "COMPLETE" or marker.get("pack_id") != pack["pack_id"]:
        raise RuntimeError("invalid native hidden-state pack marker")
    if marker.get("features_sha256") != sha256_file(feature_path):
        raise RuntimeError("native hidden-state feature checksum mismatch")
    if marker.get("metadata_sha256") != sha256_file(metadata_path):
        raise RuntimeError("native hidden-state metadata checksum mismatch")
    values = torch.load(feature_path, map_location="cpu", weights_only=False)
    metadata = read_jsonl(metadata_path)
    expected_ids = list(map(str, pack["trace_ids"]))
    if list(values) != expected_ids or [str(row["trace_id"]) for row in metadata] != expected_ids:
        raise RuntimeError("native hidden-state pack membership or order changed")
    if int(marker.get("trace_count", -1)) != len(expected_ids):
        raise RuntimeError("native hidden-state marker trace count mismatch")
    for trace_id, row in zip(expected_ids, metadata):
        source = cohort_index[trace_id]
        value = values[trace_id]
        hidden = value["hidden_states"]
        log_probs = value["completion_token_log_probabilities"]
        expected_completion = len(source["completion_token_ids"])
        if tuple(hidden.shape[:1]) != (expected_completion + 1,) or hidden.ndim != 2:
            raise RuntimeError("native hidden-state tensor has wrong token alignment")
        if hidden.dtype != torch.float16 or not torch.isfinite(hidden.float()).all():
            raise RuntimeError("native hidden-state tensor has wrong dtype or non-finite values")
        if tuple(log_probs.shape) != (expected_completion,):
            raise RuntimeError("native log-probability tensor has wrong token alignment")
        if str(value["model_revision"]) != expected_model_revision:
            raise RuntimeError("native hidden-state pack model revision mismatch")
        if str(value["tokenizer_revision"]) != expected_tokenizer_revision:
            raise RuntimeError("native hidden-state pack tokenizer revision mismatch")
        if value["prompt_token_ids_hash"] != stable_hash(source["prompt_token_ids"]):
            raise RuntimeError("native prompt token hash mismatch")
        if value["completion_token_ids_hash"] != stable_hash(source["completion_token_ids"]):
            raise RuntimeError("native completion token hash mismatch")
        if bool(value.get("kv_cache_persisted")):
            raise RuntimeError("native hidden-state artifact unexpectedly persisted KV cache")
        if row["hidden_state_shape"] != list(hidden.shape):
            raise RuntimeError("native hidden-state metadata shape mismatch")
    return marker


def execute_feature_pack(
    *, model: Any, pack: Mapping[str, Any], cohort_index: Mapping[str, Mapping[str, Any]],
    output_root: str | Path, chunk_size: int, selected_layer: int,
    model_revision: str, tokenizer_revision: str,
) -> dict[str, Any]:
    root = Path(output_root) / "packs" / str(pack["model_key"]) / str(pack["pack_id"])
    marker_path = root / "complete.json"
    if marker_path.is_file():
        marker = validate_feature_pack(
            root, pack=pack, cohort_index=cohort_index,
            expected_model_revision=model_revision,
            expected_tokenizer_revision=tokenizer_revision,
        )
        return {**marker, "status": "SKIPPED_VALID"}
    payloads: dict[str, Any] = {}
    metadata_rows: list[dict[str, Any]] = []
    started = time.perf_counter()
    for trace_id in map(str, pack["trace_ids"]):
        payload, metadata = extract_trace_hidden_states(
            model=model,
            trace=cohort_index[trace_id],
            chunk_size=int(chunk_size),
            selected_layer=int(selected_layer),
        )
        payloads[trace_id] = payload
        metadata_rows.append(metadata)
    feature_path = root / "hidden_states.pt"
    metadata_path = root / "trace_metadata.jsonl"
    _atomic_torch_save(feature_path, payloads)
    atomic_jsonl(metadata_path, metadata_rows)
    marker = {
        "status": "COMPLETE",
        "pack_id": str(pack["pack_id"]),
        "model_key": str(pack["model_key"]),
        "trace_count": len(payloads),
        "trace_ids": list(payloads),
        "completion_tokens": sum(row["completion_token_count"] for row in metadata_rows),
        "hidden_state_rows": sum(row["hidden_state_shape"][0] for row in metadata_rows),
        "selected_layer": int(selected_layer),
        "model_revision": model_revision,
        "tokenizer_revision": tokenizer_revision,
        "features_sha256": sha256_file(feature_path),
        "metadata_sha256": sha256_file(metadata_path),
        "wall_seconds": time.perf_counter() - started,
        "kv_cache_persisted": False,
        "semantic_segmentation_performed": False,
        "boundary_model_work_performed": False,
    }
    atomic_json(marker_path, marker)
    return marker


def aggregate_feature_cache(
    *, cohort: Sequence[Mapping[str, Any]], packs: Sequence[Mapping[str, Any]],
    output_root: str | Path, model_key: str,
    model_revision: str, tokenizer_revision: str,
) -> dict[str, Any]:
    root = Path(output_root)
    cohort_index = {str(row["trace_id"]): row for row in cohort}
    index_rows: list[dict[str, Any]] = []
    pack_markers: list[dict[str, Any]] = []
    seen: set[str] = set()
    for pack in packs:
        pack_root = root / "packs" / model_key / str(pack["pack_id"])
        marker = validate_feature_pack(
            pack_root, pack=pack, cohort_index=cohort_index,
            expected_model_revision=model_revision,
            expected_tokenizer_revision=tokenizer_revision,
        )
        pack_markers.append(marker)
        for row in read_jsonl(pack_root / "trace_metadata.jsonl"):
            trace_id = str(row["trace_id"])
            if trace_id in seen:
                raise RuntimeError("duplicate trace across native hidden-state packs")
            seen.add(trace_id)
            index_rows.append({
                **row,
                "pack_id": str(pack["pack_id"]),
                "feature_path": str(pack_root / "hidden_states.pt"),
                "feature_sha256": marker["features_sha256"],
            })
    expected = [str(row["trace_id"]) for row in cohort]
    if [str(row["trace_id"]) for row in index_rows] != expected:
        raise RuntimeError("aggregated native hidden-state cache does not exactly cover cohort")
    model_root = root / "final" / model_key
    atomic_jsonl(model_root / "trace_index.jsonl", index_rows)
    integrity = {
        "status": "PASS",
        "model_key": model_key,
        "cohort_traces": len(cohort),
        "cached_traces": len(index_rows),
        "pack_count": len(packs),
        "completed_pack_count": len(pack_markers),
        "completion_tokens": sum(row["completion_token_count"] for row in index_rows),
        "hidden_state_rows": sum(row["hidden_state_shape"][0] for row in index_rows),
        "selected_layers": [-1],
        "storage_dtype": "float16",
        "token_scope": "prompt_final_plus_every_completion_token",
        "exact_cohort_membership": True,
        "exact_token_hashes": True,
        "all_hidden_states_finite": True,
        "kv_caches_persisted": False,
        "semantic_segmentation_performed": False,
        "boundary_model_work_performed": False,
        "repair_generation_performed": False,
    }
    atomic_json(model_root / "integrity_report.json", integrity)
    summary = {
        "status": "COMPLETE",
        "model_key": model_key,
        "model_revision": model_revision,
        "tokenizer_revision": tokenizer_revision,
        "integrity": integrity,
        "summed_pack_wall_seconds": sum(float(row["wall_seconds"]) for row in pack_markers),
        "total_feature_bytes": sum(
            (root / "packs" / model_key / str(pack["pack_id"]) / "hidden_states.pt").stat().st_size
            for pack in packs
        ),
    }
    atomic_json(model_root / "summary.json", summary)
    return summary
