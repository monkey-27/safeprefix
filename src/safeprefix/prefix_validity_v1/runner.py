"""End-to-end orchestration for monotonic prefix-validity probe training.

This module keeps the scientific phase boundary explicit: ProcessBench probes,
calibrators, and gate cutoffs are frozen before the held-out ProcessBench test
is opened. The small functions are shared by local tests and the Modal driver.
"""

from __future__ import annotations

from dataclasses import asdict
import json
import os
from pathlib import Path
import subprocess
import tempfile
from typing import Any, Mapping, Sequence

import numpy as np
import pandas as pd
import torch
import yaml

from safeprefix.boundary_v1.data import sha256_file, stable_hash
from safeprefix.reproducibility import atomic_json, atomic_jsonl, atomic_parquet
from safeprefix.teacher_forced_completion import build_checkpoint_features
from safeprefix.models.teacher_forcing import teacher_force_token_ids_chunked
from .calibration import apply_calibrator, fit_fivefold_oof_calibration
from .data import PrefixValidityData, PrefixValidityExtractionPlan
from .evaluation import evaluate_by_domain
from .monotonic import apply_monotonic_projection, select_gate_cutoff
from .training import (
    PrefixValidityCorpus,
    PrefixValidityTrainingConfig,
    fit_probe_matrix,
    load_frozen_probe,
    predict_probe,
    save_frozen_probe,
    select_learning_rate_and_median_seed,
    training_manifest,
)


class PrefixValidityRunError(RuntimeError):
    """A frozen protocol or artifact invariant was violated."""


EXPECTED_MODELS = (
    "family_a_small", "family_a_large", "family_b_small", "family_b_large"
)


def load_prefix_validity_config(path: str | Path) -> dict[str, Any]:
    payload = yaml.safe_load(Path(path).read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise PrefixValidityRunError("prefix-validity config must be a mapping")
    if tuple(payload.get("selected_models", ())) != EXPECTED_MODELS:
        raise PrefixValidityRunError("prefix-validity experiment requires the frozen four-model matrix")
    protocol = payload.get("protocol", {})
    required_false = (
        "allow_new_recoverability_supervision",
        "allow_recoverability_probe_modification",
        "allow_recoverability_cutoff_modification",
        "allow_native_outcomes_before_selection_freeze",
        "allow_manuscript_edits",
    )
    for name in required_false:
        if protocol.get(name) is not False:
            raise PrefixValidityRunError(f"protocol guard {name} must be false")
    training = payload.get("training", {})
    if list(map(float, training.get("learning_rates", []))) != [1e-3, 3e-4]:
        raise PrefixValidityRunError("learning rates must remain [1e-3, 3e-4]")
    if list(map(int, training.get("optimization_seeds", []))) != [0, 1, 2]:
        raise PrefixValidityRunError("optimization seeds must remain [0, 1, 2]")
    if training.get("class_balanced_loss") is not False:
        raise PrefixValidityRunError("primary correctness loss must remain unweighted")
    if protocol.get("allow_native_application") is not False:
        raise PrefixValidityRunError("this run is probe training only; native application must remain disabled")
    if float(payload.get("calibration", {}).get("maximum_late_boundary_rate", -1)) != 0.05:
        raise PrefixValidityRunError("late-boundary calibration constraint must remain 0.05")
    return payload


def load_boundary_selected_layers(
    path: str | Path, model_key: str
) -> list[int]:
    """Return the exact hidden layers consumed by the frozen boundary probe."""

    payload = yaml.safe_load(Path(path).read_text(encoding="utf-8"))
    try:
        layers = list(
            map(
                int,
                payload["source"]["expected_models"][str(model_key)]["selected_layers"],
            )
        )
    except (KeyError, TypeError, ValueError) as exc:
        raise PrefixValidityRunError(
            f"{model_key}: frozen boundary representation metadata is missing"
        ) from exc
    if not layers or layers[-1] != -1 or len(set(layers)) != len(layers):
        raise PrefixValidityRunError(
            f"{model_key}: invalid frozen boundary selected layers {layers}"
        )
    return layers


def git_commit(repo_root: Path) -> str:
    return subprocess.run(
        ["git", "rev-parse", "HEAD"], cwd=repo_root, check=True,
        capture_output=True, text=True,
    ).stdout.strip()


def freeze_extraction_plan(
    plan: PrefixValidityExtractionPlan, *, output_root: str | Path,
    config: Mapping[str, Any], source_commit: str,
) -> dict[str, Any]:
    """Persist the immutable trace universe before any missing forward pass."""

    root = Path(output_root)
    root.mkdir(parents=True, exist_ok=True)
    manifest_path = root / "manifests/processbench_extraction_plan.parquet"
    proposed_hash = stable_hash(plan.rows.to_dict("records"))
    if manifest_path.is_file():
        existing = pd.read_parquet(manifest_path)
        if stable_hash(existing.to_dict("records")) != proposed_hash:
            raise PrefixValidityRunError("frozen extraction plan would change")
    else:
        atomic_parquet(manifest_path, plan.rows)
    exclusion_path = root / "manifests/processbench_extraction_exclusions.parquet"
    exclusions = (
        plan.exclusions
        if plan.exclusions is not None
        else pd.DataFrame(columns=["model_key", "trace_id", "exclusion_reason"])
    )
    if exclusion_path.is_file():
        existing = pd.read_parquet(exclusion_path)
        if stable_hash(existing.to_dict("records")) != stable_hash(
            exclusions.to_dict("records")
        ):
            raise PrefixValidityRunError("frozen extraction exclusions would change")
    else:
        atomic_parquet(exclusion_path, exclusions)
    freeze = {
        "status": "FROZEN_BEFORE_NEW_TEACHER_FORCING",
        "schema_version": "prefix-validity-extraction-v1",
        "source_commit": source_commit,
        "configuration_hash": stable_hash(config),
        "manifest_sha256": sha256_file(manifest_path),
        "exclusions_sha256": sha256_file(exclusion_path),
        "summary": plan.summary,
        "new_suffix_rollouts": 0,
        "native_outcomes_accessed": False,
    }
    path = root / "manifests/extraction_freeze.json"
    if path.is_file() and json.loads(path.read_text()) != freeze:
        raise PrefixValidityRunError("extraction freeze would change on resume")
    atomic_json(path, freeze)
    return freeze


def build_missing_feature_packs(
    plan: PrefixValidityExtractionPlan, *, traces_per_pack: int,
) -> tuple[list[dict[str, Any]], dict[str, dict[str, Any]]]:
    """Create deterministic exact-once packs for only absent full-step features."""

    if int(traces_per_pack) <= 0:
        raise ValueError("traces_per_pack must be positive")
    packs: list[dict[str, Any]] = []
    index: dict[str, dict[str, Any]] = {}
    for model_key in EXPECTED_MODELS:
        rows = [
            dict(row) for row in plan.traces_by_model[model_key]
            if row["feature_origin"] == "new_full_step_teacher_forcing_required"
        ]
        rows.sort(key=lambda row: str(row["trace_id"]))
        for row in rows:
            trace_id = str(row["trace_id"])
            model_trace_key = f"{model_key}\0{trace_id}"
            if model_trace_key in index:
                raise PrefixValidityRunError(
                    f"duplicate missing model/trace identity: {model_key}/{trace_id}"
                )
            index[model_trace_key] = row
        for offset in range(0, len(rows), int(traces_per_pack)):
            trace_ids = [str(row["trace_id"]) for row in rows[offset:offset + traces_per_pack]]
            identity = ["prefix-validity-full-step-v1", model_key, trace_ids]
            packs.append({
                "pack_id": stable_hash(identity)[:24],
                "model_key": model_key,
                "trace_ids": trace_ids,
                "trace_count": len(trace_ids),
                "pack_hash": stable_hash(identity),
            })
    if len(index) != int(plan.summary["total_new_teacher_forcing_traces"]):
        raise PrefixValidityRunError("missing feature-pack trace count differs from extraction plan")
    return packs, index


def _atomic_torch(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, name = tempfile.mkstemp(dir=path.parent, prefix=f".{path.name}.")
    os.close(fd)
    temporary = Path(name)
    try:
        torch.save(payload, temporary)
        temporary.replace(path)
    finally:
        temporary.unlink(missing_ok=True)


def execute_missing_feature_pack(
    *, loaded: Any, model_key: str, pack: Mapping[str, Any],
    trace_index: Mapping[str, Mapping[str, Any]], output_root: str | Path,
    selected_layers: Sequence[int], chunk_size: int,
) -> dict[str, Any]:
    """Teacher-force a deterministic missing-feature pack, with no KV/output rollout."""

    # Match the completed safety-cache layout so the same strict discovery and
    # checksum validator consumes reused and newly extracted packs.
    root = (
        Path(output_root) / "safety/hidden_state_feature_shards"
        / model_key / str(pack["pack_id"])
    )
    marker_path = root / "complete.json"
    features_path = root / "features.pt"
    metadata_path = root / "checkpoint_metadata.parquet"
    if marker_path.is_file() and features_path.is_file() and metadata_path.is_file():
        marker = json.loads(marker_path.read_text())
        if (
            marker.get("pack_hash") == pack["pack_hash"]
            and marker.get("features_sha256") == sha256_file(features_path)
            and marker.get("metadata_sha256") == sha256_file(metadata_path)
        ):
            return {**marker, "status": "SKIPPED_VALID"}
    payload: dict[str, Any] = {}
    metadata: list[dict[str, Any]] = []
    for trace_id in map(str, pack["trace_ids"]):
        source = dict(trace_index[trace_id])
        ids = list(map(int, source["prompt_token_ids"])) + list(map(int, source["completion_token_ids"]))
        offsets = list(map(int, source["checkpoint_offsets"]))
        token_offsets = [offset - 1 for offset in offsets]
        final_offset = len(ids) - 1
        forced = teacher_force_token_ids_chunked(
            loaded.model, ids, prompt_count=len(source["prompt_token_ids"]),
            chunk_size=int(chunk_size), selected_layers=list(map(int, selected_layers)),
            selected_token_offsets=sorted(set([*token_offsets, final_offset])),
            selected_checkpoint_offsets=(),
        )
        features = build_checkpoint_features(
            forced.selected_hidden_states, token_offsets, final_offset=final_offset,
            prefix_total_tokens=len(ids), nll_by_token=-forced.token_log_probabilities,
        ).to(torch.float16)
        error = int(source["first_error_zero_based"])
        labels = [True, *[step < error for step in range(len(source["reasoning_steps"]))]]
        mask = [False, *([True] * len(source["reasoning_steps"]))]
        payload[trace_id] = {
            "features": features, "checkpoint_offsets": offsets,
            "checkpoint_indices": list(range(len(offsets))),
            "visible_safety_labels": labels, "safety_supervision_mask": mask,
            "first_error_zero_based": error, "selected_layers": list(map(int, selected_layers)),
            "model_revision": str(loaded.model_revision),
            "tokenizer_revision": str(source["tokenizer_revision"]),
            "representation": "three_layer_checkpoint_final_with_difference_and_trace_summary_v1",
            "pooling_rule": "checkpoint_final_token",
            "mean_all_token_nll": float((-forced.token_log_probabilities).mean()),
        }
        for checkpoint_index, checkpoint_offset in enumerate(offsets):
            metadata.append({
                "model_key": model_key, "trace_id": trace_id,
                "source_trace_id": source["source_trace_id"], "problem_id": source["problem_id"],
                "problem_group_hash": source["production_problem_group"],
                "pipeline_split": source["pipeline_split"],
                "checkpoint_index": checkpoint_index,
                "checkpoint_token_offset": checkpoint_offset,
                "visible_safety_label": labels[checkpoint_index],
                "safety_supervision_mask": mask[checkpoint_index],
                "terminal_correct": False, "first_error_zero_based": error,
                "feature_row_index": checkpoint_index,
            })
    _atomic_torch(features_path, payload)
    atomic_parquet(metadata_path, pd.DataFrame(metadata))
    marker = {
        "status": "COMPLETE", "pack_id": pack["pack_id"], "pack_hash": pack["pack_hash"],
        "model_key": model_key, "trace_count": len(payload), "checkpoint_count": len(metadata),
        "features_sha256": sha256_file(features_path),
        "metadata_sha256": sha256_file(metadata_path),
        "rollouts_generated": 0, "kv_caches_persisted": False,
        "recoverability_probe_modified": False,
    }
    atomic_json(marker_path, marker)
    return marker


def _config_training(config: Mapping[str, Any]) -> PrefixValidityTrainingConfig:
    raw = config["training"]
    return PrefixValidityTrainingConfig(
        learning_rates=tuple(map(float, raw["learning_rates"])),
        seeds=tuple(map(int, raw["optimization_seeds"])),
        max_epochs=int(raw["max_epochs"]), patience=int(raw["patience"]),
        batch_size_traces=int(raw["batch_size_traces"]),
        weight_decay=float(raw["weight_decay"]),
        gradient_clip_norm=float(raw["gradient_clip_norm"]),
        hidden_width=int(raw["hidden_width"]), dropout=float(raw["dropout"]),
    )


def _model_split_corpus(
    data: PrefixValidityData,
    *,
    model_key: str,
    splits: Sequence[str],
) -> PrefixValidityCorpus:
    """Materialize only the declared split rows and their aligned features.

    In particular, probe fitting receives no ProcessBench test rows or hidden
    vectors at all.  This is stronger than merely avoiding a test DataLoader:
    test labels cannot be inspected by corpus validation, preprocessing, model
    selection, calibration, or cutoff selection before the global freeze.
    """

    selected_splits = tuple(map(str, splits))
    all_rows = data.rows.loc[data.rows["model_key"].astype(str).eq(model_key)].copy()
    feature_index = (
        "hidden_feature_row_index"
        if "hidden_feature_row_index" in all_rows
        else "feature_row_index"
    )
    all_rows.sort_values(feature_index, inplace=True, kind="mergesort")
    if (
        len(all_rows) != len(data.features[model_key])
        or all_rows[feature_index].astype(int).tolist() != list(range(len(all_rows)))
    ):
        raise PrefixValidityRunError(f"{model_key}: feature row identity is not contiguous")
    scoped = all_rows.loc[all_rows["split"].astype(str).isin(selected_splits)].copy()
    if set(scoped["split"].astype(str)) != set(selected_splits):
        raise PrefixValidityRunError(
            f"{model_key}: declared split scope is incomplete: {selected_splits}"
        )
    source_indices = scoped[feature_index].astype(int).to_numpy()
    scoped["source_hidden_feature_row_index"] = source_indices
    scoped[feature_index] = np.arange(len(scoped), dtype=int)
    scoped.reset_index(drop=True, inplace=True)
    features = data.features[model_key].index_select(
        0, torch.tensor(source_indices, dtype=torch.long)
    )
    return PrefixValidityCorpus(
        scoped,
        features,
        required_splits=selected_splits,
    )


def train_and_freeze_model(
    *, data: PrefixValidityData, model_key: str, config: Mapping[str, Any],
    output_root: str | Path, device: str = "cpu",
) -> dict[str, Any]:
    """Fit both matched probes, OOF-calibrate, select gamma, and evaluate test."""

    root = Path(output_root) / "processbench" / model_key
    corpus = _model_split_corpus(
        data,
        model_key=model_key,
        splits=("train", "architecture_dev", "calibration"),
    )
    training_config = _config_training(config)
    result: dict[str, Any] = {}
    for architecture in ("linear_probe", "position_only"):
        print(
            f"[prefix-validity:fit] {model_key}/{architecture} candidate matrix started",
            flush=True,
        )
        candidates = fit_probe_matrix(corpus, architecture=architecture, config=training_config, device=device)
        print(
            f"[prefix-validity:fit] {model_key}/{architecture} candidate matrix complete: "
            f"{len(candidates)} candidates",
            flush=True,
        )
        selected, selection = select_learning_rate_and_median_seed(candidates)
        probe_path = root / architecture / "frozen_probe.pt"
        save_frozen_probe(selected, probe_path)
        probe = load_frozen_probe(probe_path, device=device)
        predictions = pd.concat(
            [predict_probe(probe, corpus, split=split, device=device) for split in
             ("train", "architecture_dev", "calibration")],
            ignore_index=True,
        )
        calibration_rows = predictions.loc[predictions["split"].eq("calibration")].copy()
        oof = fit_fivefold_oof_calibration(
            calibration_rows, fold_seed=int(config["calibration"]["fold_seed"]),
            max_iterations=int(config["calibration"]["max_iterations"]),
        )
        oof_monotonic = apply_monotonic_projection(oof.predictions)
        cutoff, cutoff_audit = select_gate_cutoff(
            oof_monotonic,
            max_late_rate=float(config["calibration"]["maximum_late_boundary_rate"]),
        )
        print(
            f"[prefix-validity:fit] {model_key}/{architecture} frozen: "
            f"gamma={cutoff.gamma:.8g} validated={cutoff.validated_gate_exists}",
            flush=True,
        )
        atomic_parquet(root / architecture / "all_split_predictions.parquet", predictions)
        atomic_parquet(root / architecture / "calibration_oof_predictions.parquet", oof_monotonic)
        atomic_parquet(root / architecture / "gate_cutoff_audit.parquet", cutoff_audit)
        atomic_json(root / architecture / "training_selection.json", selection)
        atomic_json(root / architecture / "training_candidates.json", {
            "candidates": [candidate.metadata() for candidate in candidates],
            "training_protocol": training_manifest(training_config),
        })
        atomic_json(root / architecture / "calibration_map.json", oof.final_calibrator.to_dict())
        atomic_json(root / architecture / "gate_cutoff.json", asdict(cutoff))
        result[architecture] = {
            "probe_path": str(probe_path), "probe_sha256": sha256_file(probe_path),
            "calibration_path": str(root / architecture / "calibration_map.json"),
            "calibration_sha256": sha256_file(root / architecture / "calibration_map.json"),
            "gate_cutoff_path": str(root / architecture / "gate_cutoff.json"),
            "gate_cutoff_sha256": sha256_file(root / architecture / "gate_cutoff.json"),
            "gamma": float(cutoff.gamma), "validated_gate_exists": bool(cutoff.validated_gate_exists),
            "test_accessed": False,
        }
    atomic_json(root / "MODEL_COMPLETE.json", {"status": "COMPLETE", "model_key": model_key, **result})
    return result


def assert_processbench_bundle(output_root: str | Path) -> dict[str, Any]:
    """Validate the global freeze and every referenced immutable component."""

    path = Path(output_root) / "processbench/PROCESSBENCH_COMPONENTS_FROZEN.json"
    if not path.is_file():
        raise PrefixValidityRunError("ProcessBench test is locked until all four gates are frozen")
    bundle = json.loads(path.read_text())
    observed_hash = bundle.get("freeze_hash")
    unhashed = {key: value for key, value in bundle.items() if key != "freeze_hash"}
    if observed_hash != stable_hash(unhashed):
        raise PrefixValidityRunError("ProcessBench component freeze hash changed")
    if bundle.get("status") != "FROZEN_BEFORE_PROCESSBENCH_TEST_EVALUATION":
        raise PrefixValidityRunError("ProcessBench component freeze has invalid status")
    if set(bundle.get("models", {})) != set(EXPECTED_MODELS):
        raise PrefixValidityRunError("ProcessBench component freeze lacks the four-model matrix")
    if bundle.get("native_outcomes_accessed") is not False:
        raise PrefixValidityRunError("native outcomes entered the ProcessBench component freeze")
    for model_key, model in bundle["models"].items():
        for architecture in ("linear_probe", "position_only"):
            metadata = model.get(architecture, {})
            if metadata.get("test_accessed") is not False:
                raise PrefixValidityRunError("ProcessBench test was accessed before component freeze")
            for path_key, hash_key in (
                ("probe_path", "probe_sha256"),
                ("calibration_path", "calibration_sha256"),
                ("gate_cutoff_path", "gate_cutoff_sha256"),
            ):
                artifact = Path(str(metadata.get(path_key, "")))
                if not artifact.is_file() or sha256_file(artifact) != str(metadata.get(hash_key)):
                    raise PrefixValidityRunError(
                        f"frozen {model_key}/{architecture} {path_key} changed"
                    )
    return bundle


def evaluate_frozen_model_test(
    *, data: PrefixValidityData, model_key: str, frozen_result: Mapping[str, Any],
    output_root: str | Path, device: str = "cpu",
) -> dict[str, Any]:
    """Open ProcessBench test only after the global component freeze exists."""

    root = Path(output_root)
    freeze = assert_processbench_bundle(root)
    if dict(freeze["models"][model_key]) != dict(frozen_result):
        raise PrefixValidityRunError("test evaluation inputs differ from the globally frozen model bundle")
    corpus = _model_split_corpus(
        data,
        model_key=model_key,
        splits=("teacher_forced_test",),
    )
    evaluations: dict[str, Any] = {}
    for architecture in ("linear_probe", "position_only"):
        metadata = frozen_result[architecture]
        probe = load_frozen_probe(metadata["probe_path"], device=device)
        test = predict_probe(probe, corpus, split="teacher_forced_test", device=device)
        calibrator = json.loads(Path(metadata["calibration_path"]).read_text())
        test = apply_monotonic_projection(apply_calibrator(test, calibrator))
        evaluation = evaluate_by_domain(test, gamma=float(metadata["gamma"]))
        destination = root / "processbench" / model_key / architecture
        atomic_parquet(destination / "test_predictions.parquet", test)
        atomic_json(destination / "test_metrics.json", evaluation)
        evaluations[architecture] = evaluation
    atomic_json(
        root / "processbench" / model_key / "TEST_COMPLETE.json",
        {"status": "COMPLETE", "model_key": model_key, "component_freeze_hash": freeze["freeze_hash"], "metrics": evaluations},
    )
    return evaluations


def freeze_processbench_bundle(
    *, output_root: str | Path, model_results: Mapping[str, Mapping[str, Any]],
    config: Mapping[str, Any], data: PrefixValidityData,
) -> dict[str, Any]:
    """Freeze all first-error components before held-out test evaluation."""

    if set(model_results) != set(EXPECTED_MODELS):
        raise PrefixValidityRunError("all four model gates must complete before held-out scoring")
    for model, result in model_results.items():
        for architecture in ("linear_probe", "position_only"):
            if result[architecture].get("test_accessed") is not False:
                raise PrefixValidityRunError(
                    f"{model}/{architecture}: test was accessed before the global freeze"
                )
    bundle = {
        "status": "FROZEN_BEFORE_PROCESSBENCH_TEST_EVALUATION",
        "schema_version": "prefix-validity-processbench-freeze-v2",
        "configuration_hash": stable_hash(config),
        "indexing_convention": data.indexing_convention,
        "data_integrity": data.integrity,
        "models": model_results,
        "validated_gate_by_model": {
            model: bool(result["linear_probe"]["validated_gate_exists"])
            for model, result in model_results.items()
        },
        "monotonic_decoder": "deterministic_unweighted_PAVA_nonincreasing_v1",
        "recoverability_probe_modified": False,
        "recoverability_cutoff_modified": False,
        "native_outcomes_accessed": False,
    }
    # Hash the exact JSON representation that is persisted.  Some nested
    # diagnostics contain NumPy scalar values handled by ``default=str``;
    # hashing the pre-serialization object made an immediately written bundle
    # fail its own round-trip integrity check.
    bundle = json.loads(json.dumps(bundle, sort_keys=True, default=str))
    bundle["freeze_hash"] = stable_hash(bundle)
    path = Path(output_root) / "processbench/PROCESSBENCH_COMPONENTS_FROZEN.json"
    if path.is_file() and json.loads(path.read_text()) != bundle:
        raise PrefixValidityRunError("ProcessBench frozen component bundle would change")
    atomic_json(path, bundle)
    return bundle


def write_final_report(
    *, output_root: str | Path, summary: Mapping[str, Any], integrity: Mapping[str, Any],
) -> Path:
    """Write the report only after terminal integrity state is known."""

    root = Path(output_root)
    status = str(integrity.get("status", "INCOMPLETE"))
    processbench = summary.get("processbench", {})
    macro = (
        summary.get("bootstrap", {})
        .get("overall_equal_model_macro", {})
        .get("macro", {})
    )
    exact_contrast = macro.get(
        "hidden_minus_position_boundary_exact_last_valid_checkpoint_accuracy", {}
    )
    late_contrast = macro.get("hidden_minus_position_boundary_late_boundary_rate", {})
    lines = [
        "# Monotonic Prefix-Validity Gate", "", "## Executive conclusion", "",
        f"**Status: {status}.**", "",
        "No native labels trained the prefix-validity probes. Native application was not run by this training job.",
        "", "## ProcessBench first-error localization", "",
        f"Models completed: {len(processbench)} / 4.", "",
    ]
    for model, value in sorted(processbench.items()):
        hidden = value.get("linear_probe", {}).get("overall", {}).get("boundary", {})
        position = value.get("position_only", {}).get("overall", {}).get("boundary", {})
        lines.extend([
            f"- `{model}`: hidden late-boundary rate {hidden.get('late_boundary_rate', 'NA')}; position-only {position.get('late_boundary_rate', 'NA')}; hidden exact boundary {hidden.get('exact_last_valid_checkpoint_accuracy', 'NA')}.",
        ])
    lines.extend([
        "", "## Hidden state versus position", "",
        f"Equal-model macro hidden-minus-position exact-boundary contrast: {exact_contrast.get('estimate', 'NA')} with 95% interval {exact_contrast.get('ci95', 'NA')}.",
        f"Equal-model macro hidden-minus-position late-boundary-rate contrast: {late_contrast.get('estimate', 'NA')} with 95% interval {late_contrast.get('ci95', 'NA')}.",
        "", "## Scope", "",
        "This run stops after freezing the visible-error probes, affine calibrators, monotonic decoder, and model-specific cutoffs. It does not alter or execute the separate dual-inference policy.",
        "Accordingly, whether the gate repairs native transfer and whether recoverability improves over direct first-error rewind are not evaluated in this run.",
        "", "## Integrity", "", "```json", json.dumps(dict(integrity), indent=2, sort_keys=True), "```", "",
    ])
    path = root / "PREFIX_VALIDITY_REPORT.md"
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(".md.tmp")
    temporary.write_text("\n".join(lines), encoding="utf-8")
    temporary.replace(path)
    return path
