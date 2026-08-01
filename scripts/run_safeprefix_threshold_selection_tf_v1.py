#!/usr/bin/env python3
"""CLI for the calibration-only SafePrefix threshold experiment."""

from __future__ import annotations

import json
from pathlib import Path
import sys

import pandas as pd
import typer


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from safeprefix.boundary_v1.evaluation import apply_calibration, fit_positive_affine_calibrator  # noqa: E402
from safeprefix.config import load_config  # noqa: E402
from safeprefix.reproducibility import atomic_json  # noqa: E402
from safeprefix.threshold_selection_tf_v1.analysis import (  # noqa: E402
    analyze_threshold_experiment,
    select_threshold_actions,
)
from safeprefix.threshold_selection_tf_v1.data import prepare_threshold_experiment  # noqa: E402
from safeprefix.threshold_selection_tf_v1.reporting import (  # noqa: E402
    publish_compact_export,
    report_threshold_experiment,
)
from safeprefix.threshold_selection_tf_v1.sharding import (  # noqa: E402
    freeze_execution_shards,
    validate_execution_shards,
    validate_shard_results,
)


app = typer.Typer(add_completion=False)


@app.command("prepare")
def prepare(
    config: Path = typer.Option(...),
    artifact_root: Path = typer.Option(...),
    boundary_root: Path | None = typer.Option(None),
    completion_root: Path | None = typer.Option(None),
    completion_mount: Path = typer.Option(Path("/completion")),
) -> None:
    resolved = load_config(config).data
    result = prepare_threshold_experiment(
        resolved,
        artifact_root=artifact_root,
        boundary_root=boundary_root,
        completion_root=completion_root,
        completion_mount=completion_mount,
    )
    print(json.dumps(result, indent=2, sort_keys=True))


@app.command("smoke-policy")
def smoke_policy(
    config: Path = typer.Option(...),
    artifact_root: Path = typer.Option(...),
) -> None:
    """Exercise one cross-fit fold and outcome-blind policy aggregation only."""
    resolved = load_config(config).data
    raw = pd.read_parquet(artifact_root / "manifests/frozen_raw_logits.parquet")
    folds = pd.read_parquet(artifact_root / "manifests/five_fold_assignment.parquet")
    operational = json.loads((artifact_root / "manifests/operational_seeds.json").read_text())[
        "operational_seeds"
    ]
    rows = []
    policy_rows = []
    for model_key, seed in operational.items():
        part = raw[
            raw["architecture"].eq("linear_probe")
            & raw["base_model"].eq(model_key)
            & raw["training_seed"].eq(int(seed))
        ].merge(
            folds[["common_trace_id", "problem_group", "domain", "fold"]],
            on=["common_trace_id", "problem_group", "domain"],
            validate="many_to_one",
        )
        train = part[part["fold"].ne(0)].copy()
        held = part[part["fold"].eq(0)].copy()
        calibrator = fit_positive_affine_calibrator(
            train, max_iterations=int(resolved["cross_fit"]["max_iterations"])
        )
        smoke_root = artifact_root / "smoke/raw_packs" / model_key / f"smoke-{model_key}"
        added = pd.read_parquet(smoke_root / "added_checkpoint_suffixes.parquet")
        full = pd.read_parquet(smoke_root / "full_regenerations.parquet")
        smoke_trace_ids = set(added["trace_id"].astype(str))
        if (
            len(smoke_trace_ids) != 2
            or smoke_trace_ids != set(full["trace_id"].astype(str))
            or not smoke_trace_ids <= set(held["trace_id"].astype(str))
        ):
            raise RuntimeError("smoke traces are not exactly two fold-zero traces")
        held_smoke = held[held["trace_id"].astype(str).isin(smoke_trace_ids)].copy()
        scored = apply_calibration(held_smoke, calibrator).rename(
            columns={"calibrated_probability": "oof_probability"}
        )
        actions = select_threshold_actions(
            scored[
                [
                    "base_model", "trace_id", "common_trace_id", "problem_id", "problem_group",
                    "domain", "checkpoint_id", "checkpoint_ordinal", "checkpoint_token_offset",
                    "prefix_token_count", "total_trace_token_count", "total_checkpoint_count",
                    "oof_probability",
                ]
            ],
            [0.0, 0.5, 0.95, 1.0],
        )
        if added["scientific"].any() or full["scientific"].any():
            raise RuntimeError("smoke rows were incorrectly marked scientific")
        checkpoint_rates = (
            added.groupby(["trace_id", "checkpoint_id"], as_index=False)
            .agg(smoke_checkpoint_success=("binary_outcome", "mean"))
        )
        full_rates = (
            full.groupby("trace_id", as_index=False)
            .agg(smoke_full_success=("binary_outcome", "mean"))
        )
        scored_actions = actions.merge(
            checkpoint_rates,
            left_on=["trace_id", "selected_checkpoint_id"],
            right_on=["trace_id", "checkpoint_id"],
            how="left",
            validate="many_to_one",
        ).merge(full_rates, on="trace_id", validate="many_to_one")
        if len(scored_actions) != 8 or scored_actions["trace_id"].nunique() != 2:
            raise RuntimeError("smoke policy aggregation lacks two traces by four thresholds")
        checkpoint_selected = ~scored_actions["fallback"].astype(bool)
        if scored_actions.loc[checkpoint_selected, "smoke_checkpoint_success"].isna().any():
            raise RuntimeError("smoke-selected checkpoint lacks a generated outcome")
        scored_actions["smoke_policy_success"] = scored_actions[
            "smoke_checkpoint_success"
        ].where(checkpoint_selected, scored_actions["smoke_full_success"])
        policy_rows.append(scored_actions)
        rows.append(
            {
                "base_model": model_key,
                "training_fold_checkpoints": len(train),
                "held_out_fold_checkpoints": len(held),
                "smoke_held_out_checkpoints": len(held_smoke),
                "held_out_groups_disjoint": not bool(
                    set(train["problem_group"].astype(str)) & set(held["problem_group"].astype(str))
                ),
                "calibrator_a_positive": float(calibrator["a"]) > 0,
                "policy_rows": len(actions),
                "outcome_columns_passed_to_selector": False,
            }
        )
    smoke_packs = list((artifact_root / "smoke/raw_packs").glob("*/smoke-*/complete.json"))
    if len(smoke_packs) != 4:
        raise RuntimeError("expected one completed smoke pack per model")
    resume_checks = list((artifact_root / "smoke/resume_checks").glob("*.json"))
    if len(resume_checks) != 4 or not all(
        json.loads(path.read_text()).get("second_status") == "SKIPPED_VALID"
        for path in resume_checks
    ):
        raise RuntimeError("smoke did not prove exact resume-by-skip behavior")
    smoke_policy_frame = pd.concat(policy_rows, ignore_index=True)
    threshold_aggregate = (
        smoke_policy_frame.groupby("threshold", as_index=False)
        .agg(
            trace_model_rows=("trace_id", "size"),
            checkpoint_coverage=("fallback", lambda values: float((~values).mean())),
            policy_success=("smoke_policy_success", "mean"),
            full_regeneration_success=("smoke_full_success", "mean"),
        )
        .to_dict("records")
    )
    result = {
        "status": "PASS",
        "scientific": False,
        "cross_fit_folds_executed": 1,
        "model_results": rows,
        "completed_smoke_packs": len(smoke_packs),
        "resume_behavior_verified_by_second_skip": True,
        "verifier_outcomes_aggregated": True,
        "threshold_aggregation": threshold_aggregate,
        "native_evaluation_used": False,
    }
    atomic_json(artifact_root / "smoke/SMOKE_COMPLETE.json", result)
    print(json.dumps(result, indent=2, sort_keys=True))


@app.command("freeze-shards")
def freeze_shards(
    config: Path = typer.Option(...),
    artifact_root: Path = typer.Option(...),
    run_id: str = typer.Option(...),
    source_digest: str = typer.Option(...),
    source_commit: str = typer.Option(...),
) -> None:
    result = freeze_execution_shards(
        load_config(config).data,
        artifact_root=artifact_root,
        run_id=run_id,
        source_digest=source_digest,
        source_commit=source_commit,
    )
    print(json.dumps(result["master"], indent=2, sort_keys=True))


@app.command("validate-transfer")
def validate_transfer(
    config: Path = typer.Option(...),
    artifact_root: Path = typer.Option(...),
    required_shard: str | None = typer.Option(None),
) -> None:
    """Validate a downloaded bundle before uploading it to another workspace."""
    master = json.loads((artifact_root / "manifests/execution_shards.json").read_text())
    resolved = load_config(config).data
    contract = validate_execution_shards(
        resolved,
        artifact_root=artifact_root,
        run_id=str(master["run_id"]),
        source_digest=str(master["source_digest"]),
        source_commit=str(master["source_commit"]),
    )
    validations = {}
    if required_shard is not None:
        if required_shard not in contract["shards"]:
            raise typer.BadParameter(f"unknown shard: {required_shard}")
        validation = validate_shard_results(
            resolved, artifact_root, contract["shards"][required_shard]
        )
        if validation["valid_packs"] != validation["expected_packs"]:
            raise RuntimeError(f"{required_shard}: transfer bundle is incomplete or corrupt")
        validations[required_shard] = validation
    print(
        json.dumps(
            {
                "status": "PASS",
                "run_id": master["run_id"],
                "master_hash": master["master_hash"],
                "required_shard_validations": validations,
                "native_data_present": False,
            },
            indent=2,
            sort_keys=True,
        )
    )


@app.command("publish")
def publish(
    artifact_root: Path = typer.Option(...),
    destination: Path = typer.Option(...),
    run_id: str = typer.Option(...),
) -> None:
    result = publish_compact_export(
        artifact_root=artifact_root,
        destination=destination,
        run_id=run_id,
    )
    print(json.dumps(result, indent=2, sort_keys=True))


@app.command("analyze")
def analyze(
    config: Path = typer.Option(...),
    artifact_root: Path = typer.Option(...),
) -> None:
    result = analyze_threshold_experiment(load_config(config).data, artifact_root=artifact_root)
    print(json.dumps(result, indent=2, sort_keys=True))


@app.command("report")
def report(
    config: Path = typer.Option(...),
    artifact_root: Path = typer.Option(...),
    run_id: str = typer.Option(...),
) -> None:
    result = report_threshold_experiment(
        load_config(config).data,
        artifact_root=artifact_root,
        run_id=run_id,
    )
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    app()
