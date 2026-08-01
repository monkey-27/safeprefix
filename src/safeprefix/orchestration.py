"""Deterministic command plans for the scalable SafePrefix pipeline."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Iterable


SCALABLE_STAGES = (
    "audit",
    "stage-a-model",
    "aggregate-stage-a",
    "prepare-model",
    "aggregate-preparation",
    "rollouts-model",
    "aggregate-rollouts",
    "boundary",
    "native-model",
    "aggregate-native",
    "dense-model",
    "aggregate-dense",
    "report",
)


def _command(
    script: str,
    config: Path,
    *,
    overrides: Iterable[str] = (),
    mock: bool = False,
    overwrite: bool = False,
    extras: Iterable[str] = (),
) -> list[str]:
    command = ["python3", f"scripts/{script}", "--config", str(config)]
    for value in overrides:
        command.extend(["--set", str(value)])
    command.extend(map(str, extras))
    if mock:
        command.append("--mock")
    if overwrite:
        command.append("--overwrite")
    return command


def build_stage_commands(
    stage: str,
    config: Path,
    config_data: dict[str, Any],
    *,
    model_name: str | None = None,
    overrides: Iterable[str] = (),
    mock: bool = False,
    overwrite: bool = False,
    boundary_device: str = "cpu",
    final_status: str | None = None,
) -> list[list[str]]:
    """Build one resumable stage without altering the resolved config hash."""
    if stage not in SCALABLE_STAGES:
        raise ValueError(f"unknown scalable stage: {stage}")
    configured = ["mock"] if mock else list(config_data["selected_models"])
    model_stages = {"stage-a-model", "prepare-model", "rollouts-model", "native-model", "dense-model"}
    if stage in model_stages:
        if model_name is None:
            raise ValueError(f"{stage} requires --model-name")
        if model_name not in configured:
            raise ValueError(f"model {model_name!r} is not configured")
    elif model_name is not None:
        raise ValueError(f"{stage} does not accept --model-name")

    common = dict(overrides=overrides, mock=mock, overwrite=overwrite)
    if stage == "audit":
        return [_command("00_audit_datasets.py", config, **common)]
    if stage == "stage-a-model":
        return [
            _command(
                "04_prepare_teacher_forced_data.py", config, **common,
                extras=("--model-name", model_name, "--stage-a-only", "--metadata-only"),
            ),
            _command(
                "05_run_rollout_count_pilot.py", config, **common,
                extras=("--model-name", model_name, "--throughput-benchmark"),
            ),
        ]
    if stage == "aggregate-stage-a":
        return [_command("05_run_rollout_count_pilot.py", config, **common, extras=("--throughput-benchmark", "--aggregate-only"))]
    if stage == "prepare-model":
        return [_command("04_prepare_teacher_forced_data.py", config, **common, extras=("--model-name", model_name))]
    if stage == "aggregate-preparation":
        return [_command("04_prepare_teacher_forced_data.py", config, **common, extras=("--aggregate-only",))]
    if stage == "rollouts-model":
        return [_command("05_run_rollout_count_pilot.py", config, **common, extras=("--model-name", model_name))]
    if stage == "aggregate-rollouts":
        return [_command("05_run_rollout_count_pilot.py", config, **common, extras=("--aggregate-only",))]
    if stage == "boundary":
        return [_command("06_train_boundary_models.py", config, overrides=overrides, overwrite=overwrite, extras=("--device", boundary_device))]
    if stage == "native-model":
        return [_command("07_run_native_transfer_pilot.py", config, **common, extras=("--model-name", model_name, "--device", boundary_device))]
    if stage == "aggregate-native":
        return [_command("07_run_native_transfer_pilot.py", config, **common, extras=("--aggregate-only",))]
    if stage == "dense-model":
        return [_command("08_run_dense_native_audit.py", config, **common, extras=("--model-name", model_name))]
    if stage == "aggregate-dense":
        return [_command("08_run_dense_native_audit.py", config, **common, extras=("--aggregate-only",))]
    extras: tuple[str, ...] = () if final_status is None else ("--final-status", final_status)
    return [_command("09_generate_report.py", config, overrides=overrides, extras=extras)]
