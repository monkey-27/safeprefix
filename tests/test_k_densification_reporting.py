from __future__ import annotations

import json
from pathlib import Path
import sqlite3
from typing import Any

import pandas as pd
import pytest

from safeprefix.k_densification_v1.calibration import MODEL_KEYS, sha256_file
from safeprefix.k_densification_v1.reporting import (
    NESTED_K,
    PART_A,
    PART_B,
    REQUIRED_CAVEATS,
    finalize_k_densification_reporting,
    merge_k32_nested_outcomes,
)
from safeprefix.reproducibility import stable_hash, stable_seed
from safeprefix.threshold_selection_tf_v1.data import row_artifact_hash


DOMAINS = ("crv_arithmetic", "processbench_math", "processbench_olympiadbench", "processbench_omnimath")
ARCHITECTURES = ("position_only", "linear_probe", "local_mlp", "change_aware_mlp", "causal_gru")


def _config() -> dict[str, Any]:
    frozen = {
        model: {
            "model_id": f"fixture/{model}",
            "model_revision": stable_hash([model, "model"]),
            "tokenizer_revision": stable_hash([model, "tokenizer"]),
        }
        for model in MODEL_KEYS
    }
    return {
        "seed": 20260729,
        "experiment": {"id": "safeprefix_k_densification_v1"},
        "source": {"threshold_expected_hashes": {"merged_k16_outcomes_sha256": "pending"}},
        "frozen_models": frozen,
        "cohort": {
            "domains": {domain: domain for domain in DOMAINS},
            "confirmation_per_model_domain": 16,
            "confirmation_per_position_quartile": 4,
        },
        "rollouts": {
            "seed_namespace": "k-densification-rollout-v1",
            "permutation_seed_namespace": "k-densification-permutation-v1",
        },
        "analysis": {"permutation_replicates": 2, "bootstrap_replicates": 4},
    }


def _prepare_inputs(tmp_path: Path) -> tuple[dict[str, Any], Path, Path, Path]:
    config = _config()
    output = tmp_path / "output"
    threshold = tmp_path / "threshold"
    output.mkdir(parents=True)
    (threshold / "raw_outcomes").mkdir(parents=True)
    checkpoint_rows = []
    for model_index, model in enumerate(MODEL_KEYS):
        frozen = config["frozen_models"][model]
        for domain_index, domain in enumerate(DOMAINS):
            for local_index in range(16):
                ordinal = local_index % 4
                offset = 20 + local_index
                trace_id = f"trace-{model_index}-{domain_index}-{local_index:02d}"
                problem_id = f"problem-{model_index}-{domain_index}-{local_index:02d}"
                prefix_hash = stable_hash(["prefix", model, domain, local_index])
                prompt_hash = stable_hash(["prompt", model, domain, local_index])
                policy_hash = stable_hash(["policy"])
                checkpoint_key = stable_hash(
                    [
                        "k-densification-checkpoint-v2",
                        frozen["model_id"],
                        frozen["model_revision"],
                        frozen["tokenizer_revision"],
                        problem_id,
                        trace_id,
                        offset,
                        prefix_hash,
                        prompt_hash,
                        policy_hash,
                        "exact_answer_v1",
                    ]
                )
                checkpoint_rows.append(
                    {
                        "checkpoint_key": checkpoint_key,
                        "model_key": model,
                        **frozen,
                        "trace_id": trace_id,
                        "source_trace_id": f"source-{trace_id}",
                        "problem_id": problem_id,
                        "problem_group": f"group-{model_index}-{domain_index}-{local_index}",
                        "domain": domain,
                        "split": "calibration",
                        "checkpoint_id": f"{trace_id}:{ordinal}",
                        "checkpoint_ordinal": ordinal,
                        "checkpoint_token_offset": offset,
                        "prefix_token_count": offset,
                        "total_trace_token_count": 100,
                        "normalized_checkpoint_position": offset / 100.0,
                        "position_quartile": local_index // 4 + 1,
                        "prefix_token_hash": prefix_hash,
                        "prompt_token_hash": prompt_hash,
                        "continuation_policy_hash": policy_hash,
                        "verifier_version": "exact_answer_v1",
                        "selection_hash": stable_hash(["selection", checkpoint_key]),
                    }
                )
    checkpoint = pd.DataFrame(checkpoint_rows)
    checkpoint.to_parquet(output / "k32_confirmation_checkpoints.parquet", index=False)
    manifest_columns = [
        "checkpoint_key", "model_key", "model_id", "model_revision", "tokenizer_revision",
        "problem_id", "problem_group", "domain", "trace_id", "source_trace_id", "checkpoint_id",
        "checkpoint_ordinal", "checkpoint_token_offset", "prefix_token_count",
        "total_trace_token_count", "normalized_checkpoint_position", "position_quartile",
        "prefix_token_hash", "prompt_token_hash", "continuation_policy_hash", "verifier_version",
        "selection_hash",
    ]
    core = {
        "schema_version": 2,
        "status": "FROZEN_BEFORE_K32_GENERATION",
        "selected_checkpoints": 256,
        "rows": checkpoint[manifest_columns].to_dict("records"),
    }
    (output / "k32_confirmation_manifest.json").write_text(
        json.dumps({**core, "manifest_sha256": stable_hash(core), "frozen_at": "fixture"}) + "\n"
    )

    k16_rows = []
    original_k4_source_rows = []
    added_k12_source_rows = []
    generation_rows = []
    generated_rows = []
    for checkpoint_row in checkpoint.to_dict("records"):
        for slot in range(16):
            row = {
                "model_key": checkpoint_row["model_key"],
                "model_id": checkpoint_row["model_id"],
                "model_revision": checkpoint_row["model_revision"],
                "tokenizer_revision": checkpoint_row["tokenizer_revision"],
                "trace_id": checkpoint_row["trace_id"],
                "source_trace_id": checkpoint_row["source_trace_id"],
                "problem_id": checkpoint_row["problem_id"],
                "checkpoint_ordinal": checkpoint_row["checkpoint_ordinal"],
                "checkpoint_token_offset": checkpoint_row["checkpoint_token_offset"],
                "rollout_index": slot,
                "rollout_seed": stable_seed("frozen-k16", checkpoint_row["checkpoint_key"], slot),
                "verifier_outcome": bool((slot + checkpoint_row["checkpoint_ordinal"]) % 3 == 0),
                "binary_outcome": bool((slot + checkpoint_row["checkpoint_ordinal"]) % 3 == 0),
                "generated_token_count": 8 + slot,
                "infrastructure_status": "executed",
            }
            source_row = {
                **row,
                **(
                    {"outcome_origin": "original_k4"}
                    if slot < 4
                    else {"scientific": True}
                ),
            }
            source_row["artifact_hash"] = row_artifact_hash(source_row)
            if slot < 4:
                original_k4_source_rows.append(source_row)
                merged_row = {key: value for key, value in source_row.items() if key != "outcome_origin"}
            else:
                added_k12_source_rows.append(source_row)
                merged_row = {key: value for key, value in source_row.items() if key != "scientific"}
            k16_rows.append(merged_row)
        for slot in range(16, 32):
            seed = stable_seed(
                config["rollouts"]["seed_namespace"],
                checkpoint_row["model_id"],
                checkpoint_row["problem_id"],
                checkpoint_row["trace_id"],
                checkpoint_row["checkpoint_token_offset"],
                checkpoint_row["prefix_token_hash"],
                slot,
                checkpoint_row["continuation_policy_hash"],
            )
            manifest_row = {
                **{key: checkpoint_row[key] for key in (
                    "checkpoint_key", "model_key", "model_id", "model_revision", "tokenizer_revision",
                    "trace_id", "source_trace_id", "problem_id", "domain", "checkpoint_id",
                    "checkpoint_ordinal", "checkpoint_token_offset", "prefix_token_hash",
                    "prompt_token_hash", "continuation_policy_hash", "verifier_version",
                )},
                "rollout_slot": slot,
                "rollout_seed": seed,
            }
            generation_rows.append(manifest_row)
            block_start = (slot // 4) * 4
            generated = {
                **{key: value for key, value in manifest_row.items() if key != "rollout_slot"},
                "job_id": stable_hash(["k32-four-slot-job-v1", checkpoint_row["checkpoint_key"], block_start]),
                "rollout_index": slot,
                "verifier_outcome": bool((slot + checkpoint_row["checkpoint_ordinal"]) % 4 == 0),
                "binary_outcome": bool((slot + checkpoint_row["checkpoint_ordinal"]) % 4 == 0),
                "infrastructure_status": "executed",
                "generated_token_count": 8 + slot,
            }
            generated["artifact_hash"] = row_artifact_hash(generated)
            generated_rows.append(generated)

    k16_path = threshold / "raw_outcomes/merged_k16_checkpoint_suffixes.parquet"
    pd.DataFrame(k16_rows).to_parquet(k16_path, index=False)
    pd.DataFrame(original_k4_source_rows).to_parquet(
        threshold / "raw_outcomes/original_k4_checkpoint_suffixes.parquet", index=False
    )
    pd.DataFrame(added_k12_source_rows).to_parquet(
        threshold / "raw_outcomes/all_added_checkpoint_suffixes.parquet", index=False
    )
    config["source"]["threshold_expected_hashes"]["merged_k16_outcomes_sha256"] = sha256_file(k16_path)
    generation_path = output / "generation_input/k32_generation_manifest.parquet"
    generation_path.parent.mkdir(parents=True)
    pd.DataFrame(generation_rows).to_parquet(generation_path, index=False)
    generated_path = output / "generated_slots_16_31.parquet"
    pd.DataFrame(generated_rows).to_parquet(generated_path, index=False)
    (output / "calibration_k16_inventory.json").write_text(
        json.dumps({"status": "PASS", "checkpoint_count": 1932}) + "\n"
    )
    (output / "k32_reuse_report.json").write_text(
        json.dumps(
            {
                "status": "COMPLETE",
                "reused_slots_0_15": 4096,
                "projected_new_rollouts": 4096,
                "native_artifacts_loaded": False,
            }
        )
        + "\n"
    )
    registry = sqlite3.connect(output / "k32_rollout_registry.sqlite")
    registry.execute("CREATE TABLE jobs(status TEXT NOT NULL)")
    registry.execute("CREATE TABLE rollout_slots(slot INTEGER NOT NULL)")
    registry.executemany("INSERT INTO jobs(status) VALUES('complete')", [()] * 1024)
    registry.executemany("INSERT INTO rollout_slots(slot) VALUES(?)", [(slot,) for slot in range(8192)])
    registry.commit()
    registry.close()
    (output / "gpu_worker_transitions.json").write_text(
        json.dumps([{"worker_id": f"gpu-worker-{index:02d}"} for index in range(10)]) + "\n"
    )
    (output / "gpu_utilization_summary.json").write_text(
        json.dumps(
            {
                "status": "COMPLETE",
                "worker_count": 10,
                "generated_outcomes": 4096,
                "generated_sha256": sha256_file(generated_path),
            }
        )
        + "\n"
    )
    return config, threshold, output, generated_path


def _write_analysis_outputs(output: Path) -> None:
    internal = output / "internal"
    internal.mkdir(exist_ok=True)
    metric_frames = []
    for analysis, ks, filename in (
        (PART_A, (1, 2, 4, 8, 16), "k16_full_corpus_metrics.csv"),
        (PART_B, NESTED_K, "k32_confirmation_metrics.csv"),
    ):
        reliability = []
        split_sample = []
        stability = []
        metrics = []
        for k in ks:
            reliability.append(
                {
                    "analysis": analysis, "K": k, "dense_reference_K": max(ks),
                    "model_key": "equal_model_macro", "aggregation": "equal_model_macro",
                    "mae": 1 / (k + 1), "rmse": 1 / (k + .5), "spearman": .7 + k / 200,
                    "majority_label_agreement": .8 + k / 200, "checkpoints": 256,
                }
            )
            if 2 * k <= max(ks):
                split_sample.append(
                    {
                        "analysis": analysis,
                        "K": k,
                        "model_key": "equal_model_macro",
                        "aggregation": "equal_model_macro",
                        "replicates": 2,
                        "mae": 1 / (k + 2),
                        "rmse": 1 / (k + 1),
                        "spearman": .6 + k / 200,
                        "majority_label_agreement": .75 + k / 200,
                    }
                )
            for architecture_index, architecture in enumerate(ARCHITECTURES):
                stability.append(
                    {
                        "analysis": analysis, "K": k, "architecture": architecture, "replicates": 2,
                        "architecture_count": 5, "win_fraction": float(architecture_index == 1),
                        "mean_rank": architecture_index + 1, "mean_rank_correlation_with_dense": .9,
                        "mean_best_nonlinear_minus_linear_nll": .01, "q025_best_nonlinear_minus_linear_nll": -.01,
                        "q975_best_nonlinear_minus_linear_nll": .03, "delta_sign_change_fraction": .1,
                    }
                )
                metrics.append(
                    {
                        "analysis": analysis, "K": k, "model_key": "equal_model_macro",
                        "architecture": architecture, "aggregation": "equal_model_macro",
                        "training_seed": "all_0_1_2", "trace_weighted_binomial_nll": .6 + architecture_index / 100 - k / 1000,
                        "brier_score": .2, "roc_auc": .7, "average_precision": .7,
                        "within_trace_concordance": .6, "traces": 256, "checkpoints": 256,
                    }
                )
        pd.DataFrame(reliability).to_csv(internal / f"{analysis}_label_reliability.csv", index=False)
        pd.DataFrame(split_sample).to_csv(internal / f"{analysis}_split_sample.csv", index=False)
        pd.DataFrame(stability).to_csv(internal / f"{analysis}_architecture_stability.csv", index=False)
        bootstrap = {
            "analysis": analysis,
            "dense_reference_K": max(ks),
            "results": {
                f"K{k}": {
                    "comparison": "causal_gru_minus_linear_probe_nll", "estimate": .01,
                    "ci_low": -.01, "ci_high": .03, "replicates": 4,
                    "bootstrap_unit": "complete_trace_within_model", "models_resampled_independently": True,
                    "aggregate": "equal_weight_four_model_macro",
                }
                for k in ks
            },
        }
        (internal / f"{analysis}_bootstrap.json").write_text(json.dumps(bootstrap) + "\n")
        (internal / f"{analysis}_complete.json").write_text(
            json.dumps({"status": "COMPLETE", "analysis": analysis, "dense_reference_K": max(ks)}) + "\n"
        )
        pd.DataFrame(metrics).to_csv(output / filename, index=False)
        metric_frames.append(metrics)


def test_merge_builds_exact_256_by_32_nested_pool(tmp_path: Path) -> None:
    config, threshold, output, generated = _prepare_inputs(tmp_path)
    merged = merge_k32_nested_outcomes(
        config=config,
        threshold_root=threshold,
        output_root=output,
        generated_outcomes_path=generated,
    )
    assert len(merged) == 8192
    assert merged["checkpoint_key"].nunique() == 256
    assert set(merged["outcome_source"]) == {"frozen_threshold_k16", "generated_confirmation"}
    assert all(
        part["rollout_index"].astype(int).tolist() == list(range(32))
        for _, part in merged.groupby("checkpoint_key", sort=True)
    )


def test_merge_rejects_seed_tampering_even_with_rehashed_row(tmp_path: Path) -> None:
    config, threshold, output, generated_path = _prepare_inputs(tmp_path)
    generated = pd.read_parquet(generated_path)
    generated.loc[0, "rollout_seed"] = int(generated.loc[0, "rollout_seed"]) + 1
    row = generated.iloc[0].to_dict()
    generated.loc[0, "artifact_hash"] = row_artifact_hash(row)
    generated.to_parquet(generated_path, index=False)
    utilization_path = output / "gpu_utilization_summary.json"
    utilization = json.loads(utilization_path.read_text())
    utilization["generated_sha256"] = sha256_file(generated_path)
    utilization_path.write_text(json.dumps(utilization) + "\n")
    with pytest.raises(RuntimeError, match="rollout_seed|seed"):
        merge_k32_nested_outcomes(
            config=config,
            threshold_root=threshold,
            output_root=output,
            generated_outcomes_path=generated_path,
        )
    assert not (output / "k32_nested_outcomes.parquet").exists()


def test_finalize_preserves_blocker_requires_caveats_and_writes_complete_last(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    config, threshold, output, generated = _prepare_inputs(tmp_path)
    _write_analysis_outputs(output)
    blocker_bytes = b'{"status":"BLOCKED_BEFORE_ROLLOUT_GENERATION","evidence":"sparse-test"}\n'
    (output / "run_manifest.json").write_bytes(blocker_bytes)
    for name in ("BLOCKED.json", "PREFLIGHT_BLOCKER_REPORT.md", "candidate_problem_census.csv"):
        (output / name).write_text(f"historical {name}\n")
    blocker_evidence = {
        name: (output / name).read_bytes()
        for name in ("BLOCKED.json", "PREFLIGHT_BLOCKER_REPORT.md", "candidate_problem_census.csv")
    }

    import safeprefix.k_densification_v1.reporting as reporting

    original_validate = reporting.validate_required_outputs

    def fail_before_manifest(root: str | Path, *, include_manifest: bool = True) -> dict[str, Any]:
        if not include_manifest:
            raise RuntimeError("injected terminal validation failure")
        return original_validate(root, include_manifest=include_manifest)

    monkeypatch.setattr(reporting, "validate_required_outputs", fail_before_manifest)
    with pytest.raises(RuntimeError, match="injected terminal validation failure"):
        finalize_k_densification_reporting(
            config=config,
            threshold_root=threshold,
            output_root=output,
            generated_outcomes_path=generated,
            run_part_b=False,
        )
    assert not (output / "COMPLETE.json").exists()
    monkeypatch.setattr(reporting, "validate_required_outputs", original_validate)

    terminal = finalize_k_densification_reporting(
        config=config,
        threshold_root=threshold,
        output_root=output,
        generated_outcomes_path=generated,
        run_part_b=False,
    )
    assert terminal["status"] == "COMPLETE"
    assert (output / "internal/teacher_forced_test_blocker_run_manifest.json").read_bytes() == blocker_bytes
    for name, contents in blocker_evidence.items():
        assert (output / name).read_bytes() == contents
    report = (output / "K_DENSIFICATION_REPORT.md").read_text()
    assert all(statement in report for statement in REQUIRED_CAVEATS)
    manifest = json.loads((output / "run_manifest.json").read_text())
    assert manifest["historical_blocker_evidence_is_superseded_not_deleted"] is True
    assert manifest["guards"]["native_outcomes_loaded"] is False
    assert json.loads((output / "COMPLETE.json").read_text())["integrity_status"] == "PASS"
