"""Frozen manifests and integrity guards for teacher-forced recoverability geometry.

This module is deliberately data-plane agnostic: it freezes scientific units,
logical rollout identities, and the selected affine probe axes without opening
any protected evaluation artifact or loading a base language model.
"""

from __future__ import annotations

from collections import Counter
import hashlib
import json
from pathlib import Path
from typing import Any, Iterable, Mapping

import pandas as pd
import torch

from safeprefix.reproducibility import (
    atomic_json,
    atomic_jsonl,
    atomic_parquet,
    git_commit,
    package_versions,
    stable_hash,
    stable_seed,
)


PRIMARY_SPLIT = "teacher_forced_test"
EXPECTED_MODELS = (
    "family_a_small",
    "family_a_large",
    "family_b_small",
    "family_b_large",
)
PROHIBITED_SOURCE_TERMS = (
    "native",
    "final_test",
    "final-test",
    "prompt_pilot",
    "cache_debug",
    "dummy",
)


def assert_teacher_forced_path(path: str | Path) -> Path:
    candidate = Path(path)
    lowered = candidate.as_posix().casefold()
    if any(term in lowered for term in PROHIBITED_SOURCE_TERMS):
        raise RuntimeError(f"protected non-teacher-forced path is forbidden: {candidate}")
    return candidate


def read_jsonl(path: str | Path) -> list[dict[str, Any]]:
    source = assert_teacher_forced_path(path)
    return [json.loads(line) for line in source.read_text().splitlines() if line.strip()]


def sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _probe_path(boundary_root: Path, model_key: str, seed: int) -> Path:
    return (
        boundary_root
        / "training"
        / model_key
        / "linear_probe"
        / "lr_1e-03"
        / f"seed_{seed}"
        / "best.pt"
    )


def audit_affine_probe(path: str | Path) -> dict[str, Any]:
    """Expose the exact common feature space in which the probe is affine.

    The selected probe is ``LayerNorm(raw_state) -> Linear``. Learned LayerNorm
    scale and shift differ by training seed, so geometry is performed in the
    common, non-affine-normalized feature ``n=(x-mean(x))/std(x)``. In that
    space the effective direction is ``linear_weight * layernorm_gamma`` and
    the effective intercept absorbs the learned LayerNorm beta.
    """

    source = assert_teacher_forced_path(path)
    payload = torch.load(source, map_location="cpu", weights_only=False)
    state = payload["state_dict"]
    required = {
        "local_model.0.weight",
        "local_model.0.bias",
        "local_model.1.weight",
        "local_model.1.bias",
    }
    if not required.issubset(state):
        raise RuntimeError(f"selected checkpoint is not LayerNorm plus Linear: {source}")
    gamma = state["local_model.0.weight"].detach().float().flatten()
    beta = state["local_model.0.bias"].detach().float().flatten()
    head = state["local_model.1.weight"].detach().float().flatten()
    head_bias = state["local_model.1.bias"].detach().float().flatten()
    if gamma.shape != beta.shape or gamma.shape != head.shape or head_bias.numel() != 1:
        raise RuntimeError(f"selected affine probe tensor shapes differ: {source}")
    direction = head * gamma
    intercept = head_bias[0] + torch.dot(head, beta)
    if not torch.isfinite(direction).all() or not torch.isfinite(intercept):
        raise RuntimeError(f"selected probe has nonfinite affine parameters: {source}")
    resolved = payload.get("resolved_training_config", {})
    if resolved.get("architecture") != "linear_probe" or float(
        resolved.get("learning_rate", -1)
    ) != 0.001:
        raise RuntimeError(f"probe selection differs from frozen protocol: {source}")
    return {
        "checkpoint": str(source),
        "checkpoint_sha256": sha256_file(source),
        "training_seed": int(resolved["seed"]),
        "best_dev_nll": float(
            payload["best_dev_metrics"]["trace_weighted_binomial_nll"]
        ),
        "input_dimension": int(direction.numel()),
        "raw_saved_feature": "final_transformer_layer_checkpoint_final_token_v1",
        "common_affine_feature": "per_checkpoint_layernorm_without_affine",
        "learned_layernorm_folded_into_direction": True,
        "layernorm_eps": 1e-5,
        "direction": direction.tolist(),
        "intercept": float(intercept),
        "direction_norm": float(torch.linalg.vector_norm(direction)),
    }


def build_axis_manifest(
    boundary_root: str | Path,
    model_keys: Iterable[str] = EXPECTED_MODELS,
) -> dict[str, Any]:
    root = assert_teacher_forced_path(boundary_root)
    models: dict[str, Any] = {}
    for model_key in model_keys:
        seeds = [audit_affine_probe(_probe_path(root, model_key, seed)) for seed in range(3)]
        by_nll = sorted(seeds, key=lambda row: (row["best_dev_nll"], row["training_seed"]))
        canonical_seed = int(by_nll[1]["training_seed"])
        directions = {
            int(row["training_seed"]): torch.tensor(row["direction"], dtype=torch.float32)
            for row in seeds
        }
        cosines = []
        for left in range(3):
            for right in range(left + 1, 3):
                cosine = torch.nn.functional.cosine_similarity(
                    directions[left][None], directions[right][None]
                ).item()
                cosines.append({"seed_left": left, "seed_right": right, "cosine": cosine})
        models[model_key] = {
            "canonical_seed": canonical_seed,
            "selection_rule": "median architecture-development NLL among seeds 0, 1, 2",
            "seeds": seeds,
            "pairwise_direction_cosines": cosines,
        }
    return {
        "status": "FROZEN",
        "architecture": "linear_probe",
        "learning_rate": 0.001,
        "primary_probe_retrained": False,
        "test_used_for_axis_selection": False,
        "models": models,
    }


def _load_canonical_manifest(path: Path) -> pd.DataFrame:
    frame = pd.read_parquet(assert_teacher_forced_path(path))
    required = {
        "base_model",
        "model_id",
        "model_revision",
        "tokenizer_revision",
        "trace_id",
        "common_trace_id",
        "problem_id",
        "problem_group",
        "domain",
        "split",
        "checkpoint_id",
        "checkpoint_ordinal",
        "checkpoint_token_offset",
        "prefix_token_count",
        "success_count",
        "num_rollouts",
        "checkpoint_validity_status",
        "feature_row_index",
    }
    missing = required - set(frame.columns)
    if missing:
        raise RuntimeError(f"canonical checkpoint columns are missing: {sorted(missing)}")
    return frame


def _test_checkpoint_frame(
    canonical: pd.DataFrame,
    *,
    configured_models: Mapping[str, Mapping[str, Any]],
) -> pd.DataFrame:
    test = canonical.loc[canonical["split"].eq(PRIMARY_SPLIT)].copy()
    if test.empty or set(test["base_model"]) != set(EXPECTED_MODELS):
        raise RuntimeError("teacher-forced test model matrix is incomplete")
    if not test["num_rollouts"].eq(4).all():
        raise RuntimeError("original teacher-forced test labels are not exactly K=4")
    if not test["checkpoint_validity_status"].eq("included_production").all():
        raise RuntimeError("nonproduction or invalid checkpoint entered geometry test")
    if test.duplicated(["base_model", "checkpoint_id"]).any():
        raise RuntimeError("duplicate model-checkpoint key in teacher-forced test")
    common_sets = {
        model: set(group["common_trace_id"])
        for model, group in test.groupby("base_model", sort=True)
    }
    if len({frozenset(value) for value in common_sets.values()}) != 1:
        raise RuntimeError("teacher-forced test traces are not shared across all models")
    for model_key, group in test.groupby("base_model", sort=True):
        expected = configured_models[model_key]
        if set(group["model_revision"]) != {str(expected["model_revision"])}:
            raise RuntimeError(f"model revision mismatch for {model_key}")
        if set(group["tokenizer_revision"]) != {str(expected["tokenizer_revision"])}:
            raise RuntimeError(f"tokenizer revision mismatch for {model_key}")
        ordinals = group.groupby("trace_id")["checkpoint_ordinal"].apply(list)
        if any(values != list(range(len(values))) for values in ordinals):
            raise RuntimeError(f"checkpoint order is not contiguous for {model_key}")
        if (group["checkpoint_token_offset"].astype(int) < 0).any():
            raise RuntimeError(f"invalid checkpoint token offset for {model_key}")
    return test.sort_values(
        ["base_model", "common_trace_id", "checkpoint_ordinal"]
    ).reset_index(drop=True)


def _source_rows_for_test(
    test: pd.DataFrame,
    source_rows: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    by_trace = {str(row["trace_id"]): row for row in source_rows}
    required = sorted(set(map(str, test["common_trace_id"])))
    missing = sorted(set(required) - set(by_trace))
    if missing:
        raise RuntimeError(f"teacher-forced source rows are missing: {missing[:5]}")
    output = []
    for trace_id in required:
        row = dict(by_trace[trace_id])
        if row.get("pipeline_split") not in {"dev", "heldout"}:
            raise RuntimeError(f"test trace did not originate in frozen heldout: {trace_id}")
        if not row.get("problem_text") or row.get("reference_answer") is None:
            raise RuntimeError(f"test trace lacks problem or reference: {trace_id}")
        output.append(row)
    return output


def _dense_rollout_rows(test: pd.DataFrame, *, base_seed: int) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for checkpoint in test.to_dict("records"):
        for rollout_index in range(4, 32):
            rows.append(
                {
                    "logical_id": stable_hash(
                        [
                            "recoverability-geometry-dense-v1",
                            checkpoint["base_model"],
                            checkpoint["checkpoint_id"],
                            rollout_index,
                        ]
                    )[:32],
                    "base_model": checkpoint["base_model"],
                    "common_trace_id": checkpoint["common_trace_id"],
                    "trace_id": checkpoint["trace_id"],
                    "problem_id": checkpoint["problem_id"],
                    "domain": checkpoint["domain"],
                    "checkpoint_id": checkpoint["checkpoint_id"],
                    "checkpoint_ordinal": int(checkpoint["checkpoint_ordinal"]),
                    "checkpoint_token_offset": int(checkpoint["checkpoint_token_offset"]),
                    "rollout_index": rollout_index,
                    "rollout_seed": stable_seed(
                        base_seed,
                        checkpoint["trace_id"],
                        int(checkpoint["checkpoint_ordinal"]),
                        rollout_index,
                    ),
                    "scientific_split": PRIMARY_SPLIT,
                }
            )
    return rows


def _prompt_rollout_rows(
    source_rows: list[dict[str, Any]],
    *,
    base_seed: int,
) -> list[dict[str, Any]]:
    # Prompt solvability is defined once per underlying problem/model, not once
    # per failed trace variant.  Two frozen test pairs are text-identical
    # variants of the same problem, so use the already-frozen problem-group
    # identifier and deterministically retain one representative trace.
    canonical_by_problem: dict[str, dict[str, Any]] = {}
    for source in source_rows:
        group = str(source["production_problem_group"])
        previous = canonical_by_problem.get(group)
        if previous is not None and (
            str(source["problem_text"]) != str(previous["problem_text"])
            or str(source["reference_answer"]) != str(previous["reference_answer"])
        ):
            raise RuntimeError(
                f"problem group {group} has nonidentical prompt or reference variants"
            )
        if previous is None or str(source["trace_id"]) < str(previous["trace_id"]):
            canonical_by_problem[group] = source
    rows: list[dict[str, Any]] = []
    for model_key in EXPECTED_MODELS:
        for group, source in sorted(canonical_by_problem.items()):
            for rollout_index in range(16):
                rows.append(
                    {
                        "logical_id": stable_hash(
                            [
                                "recoverability-geometry-prompt-v1",
                                model_key,
                                group,
                                rollout_index,
                            ]
                        )[:32],
                        "base_model": model_key,
                        "common_trace_id": source["trace_id"],
                        "problem_id": source["problem_id"],
                        "problem_group": group,
                        "domain": source["source_bucket"],
                        "rollout_index": rollout_index,
                        "rollout_seed": stable_seed(
                            base_seed,
                            "geometry_prompt",
                            model_key,
                            group,
                            rollout_index,
                        ),
                        "scientific_split": PRIMARY_SPLIT,
                    }
                )
    return rows


def prepare_geometry_manifests(
    *,
    config: Mapping[str, Any],
    canonical_manifest_path: str | Path,
    source_manifest_path: str | Path,
    boundary_root: str | Path,
    output_root: str | Path,
) -> dict[str, Any]:
    output = assert_teacher_forced_path(output_root)
    canonical_path = assert_teacher_forced_path(canonical_manifest_path)
    source_path = assert_teacher_forced_path(source_manifest_path)
    boundary = assert_teacher_forced_path(boundary_root)
    canonical = _load_canonical_manifest(canonical_path)
    configured_models = config["source"]["expected_models"] if "expected_models" in config.get("source", {}) else {
        key: {
            "model_revision": config["models"][key]["revision"],
            "tokenizer_revision": config["models"][key]["tokenizer_revision"],
        }
        for key in EXPECTED_MODELS
    }
    test = _test_checkpoint_frame(canonical, configured_models=configured_models)
    sources = _source_rows_for_test(test, read_jsonl(source_path))
    axis_manifest = build_axis_manifest(boundary)
    base_seed = int(config["frozen_generation"]["base_seed"])
    dense_rows = _dense_rollout_rows(test, base_seed=base_seed)
    prompt_rows = _prompt_rollout_rows(sources, base_seed=base_seed)

    expected_dense = len(test) * 28
    unique_problem_count = len(
        {str(row["production_problem_group"]) for row in sources}
    )
    expected_prompt = unique_problem_count * len(EXPECTED_MODELS) * 16
    if len(dense_rows) != expected_dense or len(prompt_rows) != expected_prompt:
        raise AssertionError("prelaunch rollout census arithmetic failed")
    if len({row["logical_id"] for row in dense_rows}) != len(dense_rows):
        raise RuntimeError("dense logical rollout IDs collide")
    if len({row["logical_id"] for row in prompt_rows}) != len(prompt_rows):
        raise RuntimeError("prompt logical rollout IDs collide")

    manifests = output / "manifests"
    atomic_parquet(manifests / "checkpoint_manifest.parquet", test)
    atomic_jsonl(manifests / "eligible_geometry_trace_manifest.jsonl", sources)
    atomic_jsonl(manifests / "dense_rollout_manifest.jsonl", dense_rows)
    atomic_jsonl(manifests / "prompt_solvability_manifest.jsonl", prompt_rows)
    atomic_json(manifests / "canonical_axis_manifest.json", axis_manifest)
    atomic_jsonl(manifests / "exclusion_manifest.jsonl", [])

    source_census = {
        model: {
            "traces": int(group["trace_id"].nunique()),
            "checkpoints": int(len(group)),
            "domains": {
                str(domain): {
                    "traces": int(domain_group["trace_id"].nunique()),
                    "checkpoints": int(len(domain_group)),
                }
                for domain, domain_group in group.groupby("domain", sort=True)
            },
        }
        for model, group in test.groupby("base_model", sort=True)
    }
    summary = {
        "status": "PRELAUNCH_FROZEN",
        "repository_commit": git_commit(),
        "package_versions": package_versions(),
        "configuration_hash": stable_hash(config),
        "canonical_manifest_sha256": sha256_file(canonical_path),
        "source_manifest_sha256": sha256_file(source_path),
        "boundary_integrity_status": json.loads(
            (boundary / "integrity" / "final_integrity.json").read_text()
        ).get("status"),
        "shared_test_traces": len(sources),
        "problem_groups": unique_problem_count,
        "checkpoint_model_pairs": int(len(test)),
        "existing_prediction_rows_three_seeds": int(len(test) * 3),
        "original_k4_rollouts": int(len(test) * 4),
        "new_dense_rollouts": len(dense_rows),
        "final_k32_rollouts": int(len(test) * 32),
        "prompt_generations": len(prompt_rows),
        "maximum_local_branch_prefixes": 160 * 12,
        "maximum_local_child_continuations": 160 * 12 * 3 * 4,
        "maximum_new_terminal_continuations": len(dense_rows)
        + len(prompt_rows)
        + 160 * 12 * 3 * 4,
        "native_artifacts_accessed": False,
        "new_calibrator_fitted": False,
        "operational_tau_selected": False,
        "source_census": source_census,
        "domain_trace_counts": dict(Counter(row["source_bucket"] for row in sources)),
        "canonical_seeds": {
            model: details["canonical_seed"]
            for model, details in axis_manifest["models"].items()
        },
    }
    atomic_json(output / "prelaunch" / "prelaunch_summary.json", summary)
    hashes = {}
    for path in sorted(manifests.iterdir()):
        if path.is_file():
            hashes[str(path.relative_to(output))] = sha256_file(path)
    atomic_json(manifests / "artifact_hashes.json", hashes)
    report = f"""# Recoverability geometry teacher-forced prelaunch

Status: **{summary['status']}**

This census uses only the frozen teacher-forced test split. No protected
evaluation artifact was opened, no selected probe was retrained, and no new
calibrator or operational threshold was selected.

## Exact scientific units

- Shared traces: **{summary['shared_test_traces']}**
- Model/checkpoint pairs: **{summary['checkpoint_model_pairs']}**
- Existing prediction rows (three probe seeds): **{summary['existing_prediction_rows_three_seeds']}**
- Original K=4 attempts: **{summary['original_k4_rollouts']}**
- New dense attempts (+28): **{summary['new_dense_rollouts']}**
- Prompt-level generations: **{summary['prompt_generations']}**
- Maximum local prefixes: **{summary['maximum_local_branch_prefixes']}**
- Maximum local child continuations: **{summary['maximum_local_child_continuations']}**
- Maximum total new terminal continuations: **{summary['maximum_new_terminal_continuations']}**

The 5,724-row existing prediction table is 1,908 unique checkpoint/model pairs
times three retained probe seeds. Inference is scheduled per unique checkpoint,
so the dense workload is 1,908 x 28 = 53,424 rather than 5,724 x 28.

## Canonical axes

Median architecture-development NLL selects: `{json.dumps(summary['canonical_seeds'], sort_keys=True)}`.
The selected probe is affine only after per-checkpoint LayerNorm normalization;
learned LayerNorm scale is folded into the reported direction.
"""
    (output / "PRELAUNCH_CENSUS_AND_INTEGRITY.md").parent.mkdir(parents=True, exist_ok=True)
    (output / "PRELAUNCH_CENSUS_AND_INTEGRITY.md").write_text(report)
    return summary
