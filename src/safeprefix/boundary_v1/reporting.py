"""Artifact completeness audit and reproducible Markdown reports."""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any, Iterable

import pandas as pd

from .data import atomic_json, load_config, read_jsonl, sha256_file


def _markdown_table(frame: pd.DataFrame, columns: Iterable[str] | None = None) -> str:
    view = frame[list(columns)].copy() if columns is not None else frame.copy()
    if view.empty:
        return "_No rows._"
    view = view.fillna("")
    headers = [str(column) for column in view.columns]
    lines = [
        "| " + " | ".join(headers) + " |",
        "| " + " | ".join(["---"] * len(headers)) + " |",
    ]
    for row in view.itertuples(index=False, name=None):
        formatted: list[str] = []
        for value in row:
            if isinstance(value, float):
                formatted.append(f"{value:.6f}")
            else:
                formatted.append(str(value).replace("|", "\\|"))
        lines.append("| " + " | ".join(formatted) + " |")
    return "\n".join(lines)


def _write(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(text.rstrip() + "\n")
    temporary.replace(path)


def audit_complete(*, config_path: Path, artifact_root: Path) -> dict[str, Any]:
    config = load_config(config_path)
    models = list(config["source"]["expected_models"])
    seeds = list(map(int, config["experiment"]["training_seeds"]))
    expected_per_model = len(seeds) * (
        1
        + len(config["training"]["architectures"])
        * len(config["training"]["learning_rates"])
    )
    failures: list[str] = []
    ready_path = artifact_root / "data/READY.json"
    if not ready_path.is_file():
        failures.append("data READY marker missing")
    else:
        ready = json.loads(ready_path.read_text())
        if ready.get("native_evaluation_used") is not False:
            failures.append("data marker does not certify native exclusion")
    canonical_path = artifact_root / "data/canonical_checkpoint_manifest.parquet"
    if not canonical_path.is_file():
        failures.append("canonical checkpoint manifest missing")
        canonical = pd.DataFrame()
    else:
        canonical = pd.read_parquet(canonical_path)
        if len(canonical) != 43_952:
            failures.append(f"canonical checkpoint count is {len(canonical)}, expected 43952")
        overlap = canonical.groupby("problem_group")["split"].nunique()
        if (overlap > 1).any():
            failures.append("a problem group crosses splits")
        if set(canonical["num_rollouts"].astype(int)) != {4}:
            failures.append("non-k=4 checkpoints entered the canonical corpus")
    completed_runs = 0
    for model in models:
        matrix_path = artifact_root / f"training/{model}/matrix_summary.json"
        if not matrix_path.is_file():
            failures.append(f"training matrix missing for {model}")
            continue
        matrix = json.loads(matrix_path.read_text())
        completed_runs += int(matrix.get("runs", 0))
        if matrix.get("runs") != expected_per_model or matrix.get("expected_runs") != expected_per_model:
            failures.append(f"incomplete training matrix for {model}")
    selection_path = artifact_root / "selection/selected_model.json"
    if not selection_path.is_file():
        failures.append("frozen architecture selection missing")
        selection: dict[str, Any] = {}
    else:
        selection = json.loads(selection_path.read_text())
        if selection.get("calibration_used_for_selection") is not False:
            failures.append("calibration entered architecture selection")
        if selection.get("test_used_for_selection") is not False:
            failures.append("test entered architecture selection")
    calibrators = list((artifact_root / "calibration").glob("*/seed_*/calibrator.json"))
    if len(calibrators) != len(models) * len(seeds):
        failures.append(f"calibrator count is {len(calibrators)}, expected {len(models)*len(seeds)}")
    for path in calibrators:
        payload = json.loads(path.read_text())
        if payload.get("fit_split") != "calibration" or float(payload.get("a", 0)) <= 0:
            failures.append(f"invalid calibrator: {path}")
    test_complete_path = artifact_root / "test/complete.json"
    if not test_complete_path.is_file():
        failures.append("teacher-forced test completion marker missing")
    else:
        test_complete = json.loads(test_complete_path.read_text())
        if test_complete.get("native_evaluation_used") is not False:
            failures.append("test completion marker does not certify native exclusion")
        if test_complete.get("final_tau_selected") is not False:
            failures.append("a final tau was selected")
    predictions_path = artifact_root / "test/teacher_forced_test_predictions.parquet"
    if predictions_path.is_file() and not canonical.empty:
        predictions = pd.read_parquet(predictions_path)
        expected_predictions = (
            len(canonical.loc[canonical["split"] == "teacher_forced_test"]) * len(seeds)
        )
        if len(predictions) != expected_predictions:
            failures.append(
                f"test predictions count is {len(predictions)}, expected {expected_predictions}"
            )
    else:
        failures.append("teacher-forced test predictions missing")
    result = {
        "status": "PASS" if not failures else "FAIL",
        "failures": failures,
        "expected_training_runs": expected_per_model * len(models),
        "completed_training_runs": completed_runs,
        "expected_calibrators": len(models) * len(seeds),
        "completed_calibrators": len(calibrators),
        "native_evaluation_used": False,
        "final_tau_selected": False,
        "selection": selection,
    }
    atomic_json(artifact_root / "integrity/final_integrity.json", result)
    if failures:
        raise RuntimeError("boundary artifact audit failed: " + "; ".join(failures))
    return result


def generate_reports(*, config_path: Path, artifact_root: Path) -> dict[str, Any]:
    integrity = audit_complete(config_path=config_path, artifact_root=artifact_root)
    config = load_config(config_path)
    reports = artifact_root / "reports"
    census = json.loads((artifact_root / "data/dataset_census.json").read_text())
    hidden_integrity = json.loads(
        (artifact_root / "data/hidden_state_integrity.json").read_text()
    )
    selection = json.loads((artifact_root / "selection/selected_model.json").read_text())
    one_se = pd.read_csv(artifact_root / "selection/one_se_summary.csv")
    comparison = pd.read_csv(artifact_root / "selection/architecture_comparison.csv")
    calibration = pd.read_csv(artifact_root / "calibration/calibration_summary.csv")
    test_summary = pd.read_csv(artifact_root / "test/seed_summary.csv")
    diagnostics = pd.read_csv(artifact_root / "test/architecture_diagnostics.csv")
    stratified = pd.read_csv(artifact_root / "test/stratified_metrics.csv")
    split_rows: list[dict[str, Any]] = []
    for split in ("train", "architecture_dev", "calibration", "teacher_forced_test"):
        rows = read_jsonl(artifact_root / f"data/splits/{split}_problems.jsonl")
        split_rows.append(
            {
                "split": split,
                "problem_groups": len(rows),
                "shared_traces": sum(int(row["trace_count"]) for row in rows),
                "checkpoints_all_models": census["split_checkpoint_counts"].get(split, 0),
            }
        )
    split_frame = pd.DataFrame(split_rows)
    model_census = pd.DataFrame(
        [
            {"base_model": model, **values}
            for model, values in census["models"].items()
        ]
    )

    _write(
        reports / "DATA_AND_SPLIT_REPORT.md",
        f"""# Data and split report

Status: **{census['status']}**. Only completed production teacher-forced artifacts were included.

{_markdown_table(model_census, ['base_model', 'traces', 'unique_problems', 'checkpoints', 'rollouts', 'excluded_nonproduction_pack_count'])}

## Frozen problem-level splits

{_markdown_table(split_frame)}

The frozen production train split was preserved. Its generic held-out pool was divided deterministically at the problem-group level into 40% architecture dev, 30% calibration, and 30% teacher-forced test, stratified by source domain. The split seed is `{config['experiment']['split_seed']}`. Exact hashes are in `data/manifest_hashes.json`.

The canonical representation is the final transformer-layer, final-token state at each existing checkpoint. It is stored as FP16 and converted to FP32 for training. Gold answers, verifier messages, suffix text, rollout outcomes, first-error annotations, future checkpoints, smoke packs, and native data are not features.
""",
    )

    matrix_rows = []
    for model in config["source"]["expected_models"]:
        matrix = json.loads((artifact_root / f"training/{model}/matrix_summary.json").read_text())
        matrix_rows.append(
            {
                "base_model": model,
                "completed_runs": matrix["runs"],
                "expected_runs": matrix["expected_runs"],
            }
        )
    _write(
        reports / "TRAINING_REPORT.md",
        f"""# Training report

All required predictors completed for every base model and seeds 0, 1, and 2.

{_markdown_table(pd.DataFrame(matrix_rows))}

Each hidden-state architecture used learning rates `1e-3` and `3e-4`. Runs used AdamW, weight decay `1e-4`, gradient clipping at `1.0`, maximum 50 epochs, and patience 7 on uncalibrated architecture-dev trace-weighted binomial NLL. Every trace has equal total objective weight.

Detailed histories, resolved configurations, checkpoints, dev predictions, and completion markers are under `training/`.
""",
    )

    _write(
        reports / "ARCHITECTURE_SELECTION_REPORT.md",
        f"""# Architecture selection report

Selected **{selection['selected_architecture']}** at learning rate **{selection['selected_learning_rate']:.0e}**.

The numerical best was `{selection['best_numerical_architecture']}` at `{selection['best_numerical_learning_rate']:.0e}` with macro dev NLL {selection['best_mean_macro_dev_nll']:.6f}. Its standard error was {selection['best_standard_error']:.6f}, giving a one-SE threshold of {selection['one_se_threshold']:.6f}. The selected model was the simplest hidden-state candidate within that threshold. Position-only was never eligible for selection.

{_markdown_table(one_se, ['architecture', 'learning_rate', 'mean_macro_dev_nll', 'std_macro_dev_nll', 'standard_error', 'within_one_standard_error'])}

Selection used only uncalibrated architecture-dev predictions. Calibration and teacher-forced test results were not available to the selector.
""",
    )

    calibration_display = calibration.copy()
    calibration_display["nll_delta_post_minus_pre"] = (
        calibration_display["post_nll"] - calibration_display["pre_nll"]
    )
    domain_calibration = (
        stratified.loc[
            (stratified["stage"] == "post_calibration")
            & (stratified["stratum"] == "domain")
        ]
        .groupby(["base_model", "value"], as_index=False)
        .agg(
            post_calibrated_nll_mean=("trace_weighted_binomial_nll", "mean"),
            post_calibrated_ece_mean=("ece_equal_count_10", "mean"),
            post_calibrated_ece_std=("ece_equal_count_10", "std"),
        )
        .sort_values("post_calibrated_ece_mean", ascending=False)
    )
    worst_domain = domain_calibration.iloc[0]
    test_calibration_delta = test_summary.pivot(
        index="base_model", columns="stage", values="nll_mean"
    )
    test_calibration_delta["post_minus_pre_nll"] = (
        test_calibration_delta["post_calibration"]
        - test_calibration_delta["pre_calibration"]
    )
    worsened_models = test_calibration_delta.loc[
        test_calibration_delta["post_minus_pre_nll"] > 0
    ].index.tolist()
    _write(
        reports / "CALIBRATION_REPORT.md",
        f"""# Calibration report

One positive affine logistic calibrator was fit independently for each base model and training seed using only the frozen calibration split. No domain-specific calibration was added.

{_markdown_table(calibration_display, ['base_model', 'seed', 'a', 'b', 'pre_nll', 'post_nll', 'pre_ece', 'post_ece', 'nll_delta_post_minus_pre'])}

Per-domain results are in `test/stratified_metrics.csv`; reliability tables, plots, and raw calibration predictions are under `calibration/`.

## Cross-domain transfer

{_markdown_table(domain_calibration)}

The largest post-calibration domain ECE is {float(worst_domain['post_calibrated_ece_mean']):.6f} for `{worst_domain['base_model']}` on `{worst_domain['value']}`. This is material residual miscalibration, so the single global per-model calibrator does not transfer uniformly across domains. No domain-specific map was added. On the held-out test, calibration worsened mean NLL for {', '.join(f'`{model}`' for model in worsened_models) if worsened_models else 'no model'}; this exception is retained rather than hidden.
""",
    )

    diagnostic_mean = diagnostics.groupby("architecture", as_index=False).agg(
        test_nll=("test_uncalibrated_nll", "mean")
    )
    diagnostic_by_name = dict(
        zip(diagnostic_mean["architecture"], diagnostic_mean["test_nll"])
    )
    local_gru = diagnostic_by_name.get("local_mlp", float("nan")) - diagnostic_by_name.get(
        "causal_gru", float("nan")
    )
    local_change = diagnostic_by_name.get("local_mlp", float("nan")) - diagnostic_by_name.get(
        "change_aware_mlp", float("nan")
    )
    hidden_position = diagnostic_by_name.get(selection["selected_architecture"], float("nan")) - diagnostic_by_name.get(
        "position_only", float("nan")
    )
    linear_gain = diagnostic_by_name.get("linear_probe", float("nan")) - diagnostic_by_name.get(
        selection["selected_architecture"], float("nan")
    )
    if selection["selected_architecture"] == "local_mlp":
        interpretation = (
            "The one-SE rule found the local MLP sufficient: the current contextual checkpoint "
            "state contains most of the useful signal seen by the GRU. This does not establish "
            "that recoverability is fully encoded in one state."
        )
    elif selection["selected_architecture"] == "causal_gru":
        interpretation = (
            "The GRU gain exceeded dev-seed uncertainty under the one-SE rule, indicating useful "
            "signal in the checkpoint-state trajectory beyond the current state alone."
        )
    else:
        interpretation = (
            "The one-SE rule selected a simpler diagnostic architecture; conclusions are limited "
            "to predictive decodability under this teacher-forced corpus."
        )
    _write(
        reports / "TEACHER_FORCED_TEST_REPORT.md",
        f"""# Teacher-forced test report

Held-out evaluation used the architecture and learning rate frozen from dev. Values below are mean and standard deviation across all three training seeds.

{_markdown_table(test_summary)}

## Frozen post-selection diagnostics

{_markdown_table(diagnostic_mean)}

- Local MLP minus GRU NLL: `{local_gru:.6f}`.
- Local MLP minus change-aware MLP NLL: `{local_change:.6f}`.
- Selected hidden model minus position-only NLL: `{hidden_position:.6f}`; values near zero would mean the representation claim is weak.
- Linear probe minus selected-model NLL: `{linear_gain:.6f}`; strong linear performance indicates recoverability is substantially linearly decodable.

{interpretation}

Threshold sweeps from 0.05 through 0.95 are diagnostic only. Dangerous-late selections mean choices later than the latest checkpoint with any observed rollout success; unnecessary rewind is the retained-prefix gap to that oracle checkpoint. **No final tau was selected.**
""",
    )

    _write(
        reports / "INTEGRITY_REPORT.md",
        f"""# Integrity report

Final status: **{integrity['status']}**.

- Completed training runs: {integrity['completed_training_runs']} / {integrity['expected_training_runs']}.
- Completed calibrators: {integrity['completed_calibrators']} / {integrity['expected_calibrators']}.
- Problem groups crossing splits: {hidden_integrity['problem_split_overlap']}.
- Native evaluation rows used: {hidden_integrity['native_evaluation_rows']}.
- First-error annotations used as targets: {hidden_integrity['first_error_used_as_target']}.
- Future checkpoint features used: {hidden_integrity['future_checkpoint_features_used']}.
- Final tau selected: {integrity['final_tau_selected']}.

All production feature packs passed checksum, revision, dimension, token-offset, and exact-K=4 coverage checks before training. Smoke and validation packs appear only in the exclusion manifest.
""",
    )

    run_id = os.environ.get("SAFEPREFIX_RUN_ID") or (
        artifact_root.parents[1].name
        if artifact_root.parent.name == "artifacts"
        else "RUN_ID"
    )
    _write(
        reports / "README.md",
        f"""# SafePrefix boundary model v1 artifacts

This directory is a completed, resumable teacher-forced boundary-model phase. It does not contain native evaluation results and does not freeze a threshold.

## Local commands

```bash
PYTHONPATH=src python3 scripts/run_boundary_model_v1.py prepare --artifact-root artifacts/boundary_model_v1 --old-root OLD_ROOT --completion-root COMPLETION_ROOT --manifest-root MANIFEST_ROOT
PYTHONPATH=src python3 scripts/run_boundary_model_v1.py train-model --artifact-root artifacts/boundary_model_v1 --model-key family_a_small --device cpu
PYTHONPATH=src python3 scripts/run_boundary_model_v1.py select --artifact-root artifacts/boundary_model_v1
PYTHONPATH=src python3 scripts/run_boundary_model_v1.py finalize --artifact-root artifacts/boundary_model_v1 --device cpu
PYTHONPATH=src python3 scripts/run_boundary_model_v1.py report --artifact-root artifacts/boundary_model_v1
```

## Modal launch, status, resume, and fetch

```bash
MODAL_PROFILE=meskmmy python3 -m modal run scripts/modal_safeprefix_boundary_model_v1.py --action launch --run-id {run_id}
MODAL_PROFILE=meskmmy python3 -m modal run scripts/modal_safeprefix_boundary_model_v1.py --action status --run-id {run_id}
MODAL_PROFILE=meskmmy python3 -m modal run scripts/modal_safeprefix_boundary_model_v1.py --action resume --run-id {run_id}
MODAL_PROFILE=meskmmy python3 -m modal run scripts/modal_safeprefix_boundary_model_v1.py --action report --run-id {run_id}
MODAL_PROFILE=meskmmy python3 -m modal run scripts/modal_safeprefix_boundary_model_v1.py --action fetch --run-id {run_id} --local-root artifacts/boundary_model_v1
```

`resume` is idempotent at completed-run boundaries. Incomplete training runs continue from their atomic per-epoch `resume.pt` state; completed runs are hash-checked and skipped.
""",
    )
    report_files = [
        "DATA_AND_SPLIT_REPORT.md",
        "TRAINING_REPORT.md",
        "ARCHITECTURE_SELECTION_REPORT.md",
        "CALIBRATION_REPORT.md",
        "TEACHER_FORCED_TEST_REPORT.md",
        "INTEGRITY_REPORT.md",
        "README.md",
    ]
    result = {
        "status": "COMPLETE",
        "reports": {
            name: sha256_file(reports / name)
            for name in report_files
        },
        "selected_architecture": selection["selected_architecture"],
        "selected_learning_rate": selection["selected_learning_rate"],
        "native_evaluation_used": False,
        "final_tau_selected": False,
    }
    atomic_json(reports / "report_summary.json", result)
    return result
