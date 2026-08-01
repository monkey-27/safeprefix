"""Fail-closed finalization for the calibration K-densification study.

This module does not generate outcomes or fit predictors.  It validates and
joins the frozen K=16 outcomes with the newly generated slots 16--31, invokes
the already-frozen analysis implementation, and publishes only data-backed
calibration-robustness products.
"""

from __future__ import annotations

import json
import math
from pathlib import Path
import re
import shutil
import sqlite3
import tempfile
from typing import Any, Mapping, Sequence

import numpy as np
import pandas as pd

from safeprefix.reproducibility import (
    atomic_json,
    atomic_parquet,
    atomic_text,
    git_commit,
    now_iso,
    package_versions,
    stable_hash,
    stable_seed,
)
from safeprefix.threshold_selection_tf_v1.data import row_artifact_hash

from .analysis import run_analysis
from .calibration import MODEL_KEYS, sha256_file


PART_A = "full_calibration_k16"
PART_B = "k32_confirmation"
NESTED_K = (1, 2, 4, 8, 16, 32)

REQUIRED_CAVEATS = (
    "This analysis tests robustness to denser EVALUATION labels.",
    "It does not prove that training nonlinear models with denser supervision would produce the same result.",
    "The calibration corpus was used because sparse test traces could not support the originally proposed balanced four-checkpoint design.",
)

REPORT_NAME = "K_DENSIFICATION_REPORT.md"
FIGURE_NAMES = (
    "k_label_reliability.png",
    "k_macro_nll_by_predictor.png",
    "k_linear_vs_best_nonlinear_nll.png",
    "k_architecture_win_frequency.png",
)
REQUIRED_ANALYSIS_FILES = (
    "k16_full_corpus_metrics.csv",
    "k32_confirmation_metrics.csv",
    "k_label_reliability.csv",
    "k_split_sample_reliability.csv",
    "k_architecture_order_stability.csv",
    "k_bootstrap_intervals.json",
)
REQUIRED_PROTOCOL_FILES = (
    "calibration_k16_inventory.json",
    "k32_confirmation_manifest.json",
    "k32_reuse_report.json",
    "k32_rollout_registry.sqlite",
    "generated_slots_16_31.parquet",
    "gpu_worker_transitions.json",
    "gpu_utilization_summary.json",
)

CHECKPOINT_IDENTITY_COLUMNS = (
    "checkpoint_key",
    "model_key",
    "model_id",
    "model_revision",
    "tokenizer_revision",
    "trace_id",
    "source_trace_id",
    "problem_id",
    "domain",
    "checkpoint_id",
    "checkpoint_ordinal",
    "checkpoint_token_offset",
    "prefix_token_hash",
    "prompt_token_hash",
    "continuation_policy_hash",
    "verifier_version",
)
K16_JOIN_COLUMNS = (
    "model_key",
    "trace_id",
    "checkpoint_ordinal",
    "checkpoint_token_offset",
)
HEX64 = re.compile(r"^[0-9a-f]{64}$")


def _required_columns(frame: pd.DataFrame, columns: Sequence[str], name: str) -> None:
    missing = set(columns) - set(frame.columns)
    if missing:
        raise RuntimeError(f"{name} lacks required columns: {sorted(missing)}")


def _assert_non_native_path(path: Path) -> None:
    if "native" in path.as_posix().casefold():
        raise RuntimeError(f"native input is prohibited: {path}")


def _assert_hash_column(frame: pd.DataFrame, column: str, name: str) -> None:
    if not frame[column].astype(str).map(lambda value: bool(HEX64.fullmatch(value))).all():
        raise RuntimeError(f"{name} contains a malformed {column}")


def _frames_equal(left: pd.DataFrame, right: pd.DataFrame) -> bool:
    lhs = left.reset_index(drop=True).astype(object).where(pd.notna(left.reset_index(drop=True)), None)
    rhs = right.reset_index(drop=True).astype(object).where(pd.notna(right.reset_index(drop=True)), None)
    try:
        pd.testing.assert_frame_equal(
            lhs,
            rhs,
            check_dtype=False,
            check_like=False,
        )
    except AssertionError:
        return False
    return True


def _write_parquet_once(path: Path, frame: pd.DataFrame) -> None:
    if path.exists():
        if not _frames_equal(pd.read_parquet(path), frame):
            raise RuntimeError(f"refusing to overwrite conflicting artifact: {path}")
        return
    atomic_parquet(path, frame)


def _write_csv_once(path: Path, frame: pd.DataFrame) -> None:
    if path.exists():
        if not _frames_equal(pd.read_csv(path), frame):
            raise RuntimeError(f"refusing to overwrite conflicting artifact: {path}")
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(
        "w", encoding="utf-8", newline="", dir=path.parent, delete=False
    ) as handle:
        temporary = Path(handle.name)
        frame.to_csv(handle, index=False)
        handle.flush()
    temporary.replace(path)


def _write_json_once(path: Path, payload: Mapping[str, Any]) -> None:
    if path.exists():
        if json.loads(path.read_text(encoding="utf-8")) != dict(payload):
            raise RuntimeError(f"refusing to overwrite conflicting artifact: {path}")
        return
    atomic_json(path, payload)


def _write_text_once(path: Path, text: str) -> None:
    if path.exists():
        if path.read_text(encoding="utf-8") != text:
            raise RuntimeError(f"refusing to overwrite conflicting artifact: {path}")
        return
    atomic_text(path, text)


def _validate_confirmation_inventory(
    config: Mapping[str, Any], output_root: Path
) -> pd.DataFrame:
    checkpoint_path = output_root / "k32_confirmation_checkpoints.parquet"
    manifest_path = output_root / "k32_confirmation_manifest.json"
    checkpoint = pd.read_parquet(checkpoint_path)
    _required_columns(
        checkpoint,
        (
            *CHECKPOINT_IDENTITY_COLUMNS,
            "prefix_token_count",
            "total_trace_token_count",
            "position_quartile",
            "normalized_checkpoint_position",
        ),
        "confirmation checkpoint inventory",
    )
    per_cell = int(config["cohort"]["confirmation_per_model_domain"])
    domains = tuple(map(str, config["cohort"]["domains"]))
    expected_count = len(MODEL_KEYS) * len(domains) * per_cell
    if expected_count != 256 or len(checkpoint) != 256:
        raise RuntimeError(
            f"K32 confirmation must contain exactly 256 checkpoints, got {len(checkpoint)}"
        )
    if checkpoint["checkpoint_key"].astype(str).duplicated().any():
        raise RuntimeError("confirmation checkpoint key is duplicated")
    if set(checkpoint["model_key"].astype(str)) != set(MODEL_KEYS):
        raise RuntimeError("confirmation model set differs")
    if set(checkpoint["domain"].astype(str)) != set(domains):
        raise RuntimeError("confirmation domain set differs")
    if "split" in checkpoint and set(checkpoint["split"].astype(str)) != {"calibration"}:
        raise RuntimeError("non-calibration checkpoint entered K32 confirmation")
    counts = checkpoint.groupby(["model_key", "domain"], sort=True).size()
    if len(counts) != len(MODEL_KEYS) * len(domains) or set(counts.astype(int)) != {per_cell}:
        raise RuntimeError("confirmation model/domain counts differ")
    quartile_count = int(config["cohort"]["confirmation_per_position_quartile"])
    quartiles = checkpoint.groupby(
        ["model_key", "domain", "position_quartile"], sort=True
    ).size()
    if len(quartiles) != len(MODEL_KEYS) * len(domains) * 4 or set(quartiles.astype(int)) != {
        quartile_count
    }:
        raise RuntimeError("confirmation position-quartile counts differ")
    expected_position = checkpoint["prefix_token_count"].to_numpy(float) / np.maximum(
        checkpoint["total_trace_token_count"].to_numpy(float), 1.0
    )
    if not np.allclose(
        checkpoint["normalized_checkpoint_position"].to_numpy(float),
        expected_position,
        rtol=0,
        atol=1e-12,
    ):
        raise RuntimeError("normalized checkpoint position convention differs")
    for model_key in MODEL_KEYS:
        part = checkpoint.loc[checkpoint["model_key"].astype(str).eq(model_key)]
        frozen = config["frozen_models"][model_key]
        for column, key in (
            ("model_id", "model_id"),
            ("model_revision", "model_revision"),
            ("tokenizer_revision", "tokenizer_revision"),
        ):
            if set(part[column].astype(str)) != {str(frozen[key])}:
                raise RuntimeError(f"{model_key}: frozen {column} differs")
    for row in checkpoint.itertuples(index=False):
        expected_key = stable_hash(
            [
                "k-densification-checkpoint-v2",
                row.model_id,
                row.model_revision,
                row.tokenizer_revision,
                row.problem_id,
                row.trace_id,
                int(row.checkpoint_token_offset),
                row.prefix_token_hash,
                row.prompt_token_hash,
                row.continuation_policy_hash,
                row.verifier_version,
            ]
        )
        if str(row.checkpoint_key) != expected_key:
            raise RuntimeError("confirmation checkpoint key hash differs")
    for column in ("checkpoint_key", "prefix_token_hash", "prompt_token_hash", "continuation_policy_hash"):
        _assert_hash_column(checkpoint, column, "confirmation checkpoint inventory")

    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if manifest.get("status") != "FROZEN_BEFORE_K32_GENERATION":
        raise RuntimeError("confirmation manifest is not frozen")
    if int(manifest.get("selected_checkpoints", -1)) != 256:
        raise RuntimeError("confirmation manifest checkpoint count differs")
    core = {key: value for key, value in manifest.items() if key not in {"frozen_at", "manifest_sha256"}}
    if stable_hash(core) != str(manifest.get("manifest_sha256")):
        raise RuntimeError("confirmation manifest content hash differs")
    manifest_rows = pd.DataFrame(manifest["rows"])
    manifest_columns = list(manifest_rows.columns)
    left = checkpoint[manifest_columns].sort_values("checkpoint_key", kind="mergesort").reset_index(drop=True)
    right = manifest_rows.sort_values("checkpoint_key", kind="mergesort").reset_index(drop=True)
    if not _frames_equal(left, right):
        raise RuntimeError("confirmation parquet differs from its frozen JSON manifest")
    return checkpoint.sort_values("checkpoint_key", kind="mergesort").reset_index(drop=True)


def _validate_generation_manifest(
    config: Mapping[str, Any], output_root: Path, checkpoint: pd.DataFrame
) -> pd.DataFrame:
    path = output_root / "generation_input/k32_generation_manifest.parquet"
    generation = pd.read_parquet(path)
    _required_columns(
        generation,
        (*CHECKPOINT_IDENTITY_COLUMNS, "rollout_slot", "rollout_seed"),
        "K32 generation manifest",
    )
    if len(generation) != 256 * 16:
        raise RuntimeError("K32 generation manifest must contain exactly 4096 rows")
    if generation.duplicated(["checkpoint_key", "rollout_slot"]).any():
        raise RuntimeError("K32 generation manifest contains a duplicate slot")
    expected_slots = list(range(16, 32))
    for checkpoint_key, part in generation.groupby("checkpoint_key", sort=True):
        if sorted(part["rollout_slot"].astype(int)) != expected_slots:
            raise RuntimeError(f"{checkpoint_key}: generation manifest slot set differs")
        if part["rollout_seed"].astype(int).nunique() != 16:
            raise RuntimeError(f"{checkpoint_key}: generation manifest seed collision")
    expected = checkpoint[list(CHECKPOINT_IDENTITY_COLUMNS)]
    joined = generation.merge(
        expected,
        on="checkpoint_key",
        how="left",
        validate="many_to_one",
        suffixes=("", "_expected"),
    )
    if joined[[f"{column}_expected" for column in CHECKPOINT_IDENTITY_COLUMNS[1:]]].isna().any().any():
        raise RuntimeError("generation manifest contains an unknown checkpoint")
    for column in CHECKPOINT_IDENTITY_COLUMNS[1:]:
        if not joined[column].astype(str).eq(joined[f"{column}_expected"].astype(str)).all():
            raise RuntimeError(f"generation manifest {column} differs from confirmation inventory")
    for row in generation.itertuples(index=False):
        expected_seed = stable_seed(
            config["rollouts"]["seed_namespace"],
            row.model_id,
            row.problem_id,
            row.trace_id,
            int(row.checkpoint_token_offset),
            row.prefix_token_hash,
            int(row.rollout_slot),
            row.continuation_policy_hash,
        )
        if int(row.rollout_seed) != expected_seed:
            raise RuntimeError("generation manifest rollout seed differs from frozen derivation")
    return generation.sort_values(["checkpoint_key", "rollout_slot"], kind="mergesort").reset_index(drop=True)


def _validate_artifact_rows(frame: pd.DataFrame, name: str) -> None:
    _required_columns(
        frame,
        ("artifact_hash", "verifier_outcome", "binary_outcome", "infrastructure_status"),
        name,
    )
    _assert_hash_column(frame, "artifact_hash", name)
    if not frame["verifier_outcome"].astype(bool).eq(frame["binary_outcome"].astype(bool)).all():
        raise RuntimeError(f"{name} verifier and binary outcomes differ")
    if set(frame["infrastructure_status"].astype(str)) != {"executed"}:
        raise RuntimeError(f"{name} contains a non-executed row")
    if frame["artifact_hash"].astype(str).duplicated().any():
        raise RuntimeError(f"{name} contains a duplicate artifact hash")


def _load_frozen_k16(
    config: Mapping[str, Any], threshold_root: Path, checkpoint: pd.DataFrame
) -> pd.DataFrame:
    _assert_non_native_path(threshold_root)
    path = threshold_root / "raw_outcomes/merged_k16_checkpoint_suffixes.parquet"
    expected_sha = str(config["source"]["threshold_expected_hashes"]["merged_k16_outcomes_sha256"])
    if sha256_file(path) != expected_sha:
        raise RuntimeError("frozen K16 source file hash differs")
    raw = pd.read_parquet(path)
    _required_columns(
        raw,
        (
            *K16_JOIN_COLUMNS,
            "source_trace_id",
            "problem_id",
            "model_id",
            "model_revision",
            "tokenizer_revision",
            "rollout_index",
            "rollout_seed",
            "artifact_hash",
        ),
        "frozen K16 outcomes",
    )
    selected = raw.merge(
        checkpoint[list(K16_JOIN_COLUMNS) + [
            column for column in CHECKPOINT_IDENTITY_COLUMNS if column not in K16_JOIN_COLUMNS
        ]],
        on=list(K16_JOIN_COLUMNS),
        how="inner",
        validate="many_to_one",
        suffixes=("", "_expected"),
    )
    if len(selected) != 256 * 16:
        raise RuntimeError("frozen K16 selection must contain exactly 4096 rows")
    # The frozen merged file contains the intersection of the original-K4 and
    # added-K12 schemas, while each artifact hash was computed on its complete
    # pre-merge source row.  Recompute against those immutable source rows;
    # hashing the lossy merged projection would necessarily differ.
    selected_hashes = set(selected["artifact_hash"].astype(str))
    verified_hashes: list[str] = []
    for source_name in (
        "original_k4_checkpoint_suffixes.parquet",
        "all_added_checkpoint_suffixes.parquet",
    ):
        source_path = threshold_root / "raw_outcomes" / source_name
        source = pd.read_parquet(source_path)
        _required_columns(source, ("artifact_hash",), f"frozen K16 source {source_name}")
        source = source.loc[source["artifact_hash"].astype(str).isin(selected_hashes)]
        if any(
            row_artifact_hash(row) != str(row["artifact_hash"])
            for row in source.to_dict("records")
        ):
            raise RuntimeError(f"frozen K16 source-row hash differs: {source_name}")
        verified_hashes.extend(source["artifact_hash"].astype(str))
    if (
        len(verified_hashes) != len(selected)
        or len(set(verified_hashes)) != len(verified_hashes)
        or set(verified_hashes) != selected_hashes
    ):
        raise RuntimeError("frozen K16 source-row artifact hash differs")
    _validate_artifact_rows(selected, "frozen K16 outcomes")
    for column in CHECKPOINT_IDENTITY_COLUMNS[1:]:
        expected_column = f"{column}_expected"
        if expected_column in selected and not selected[column].astype(str).eq(
            selected[expected_column].astype(str)
        ).all():
            raise RuntimeError(f"frozen K16 {column} differs from confirmation inventory")
    if selected.duplicated(["checkpoint_key", "rollout_index"]).any():
        raise RuntimeError("frozen K16 contains a duplicate checkpoint slot")
    for checkpoint_key, part in selected.groupby("checkpoint_key", sort=True):
        if sorted(part["rollout_index"].astype(int)) != list(range(16)):
            raise RuntimeError(f"{checkpoint_key}: frozen K16 slot set differs")
        if part["rollout_seed"].astype(int).nunique() != 16:
            raise RuntimeError(f"{checkpoint_key}: frozen K16 seed collision")
    selected["outcome_source"] = "frozen_threshold_k16"
    selected["dense_reference_K"] = 32
    return selected


def _load_generated_k32(
    generated_path: Path, generation: pd.DataFrame
) -> pd.DataFrame:
    _assert_non_native_path(generated_path)
    generated = pd.read_parquet(generated_path)
    _required_columns(
        generated,
        (*CHECKPOINT_IDENTITY_COLUMNS, "job_id", "rollout_index", "rollout_seed"),
        "generated K32 outcomes",
    )
    if len(generated) != 256 * 16:
        raise RuntimeError("generated slots 16--31 must contain exactly 4096 rows")
    summary_path = generated_path.parent / "gpu_utilization_summary.json"
    if summary_path.is_file():
        summary = json.loads(summary_path.read_text(encoding="utf-8"))
        if (
            summary.get("status") != "COMPLETE"
            or int(summary.get("generated_outcomes", -1)) != 4096
            or str(summary.get("generated_sha256")) != sha256_file(generated_path)
        ):
            raise RuntimeError("generated K32 utilization summary hash/count differs")
    if any(
        row_artifact_hash(row) != str(row["artifact_hash"])
        for row in generated.to_dict("records")
    ):
        raise RuntimeError("generated K32 row artifact hash differs")
    _validate_artifact_rows(generated, "generated K32 outcomes")
    if generated.duplicated(["checkpoint_key", "rollout_index"]).any():
        raise RuntimeError("generated K32 outcomes contain a duplicate checkpoint slot")
    expected = generation.rename(columns={"rollout_slot": "rollout_index"})
    joined = generated.merge(
        expected,
        on=["checkpoint_key", "rollout_index"],
        how="outer",
        validate="one_to_one",
        suffixes=("", "_expected"),
        indicator=True,
    )
    if set(joined["_merge"].astype(str)) != {"both"}:
        raise RuntimeError("generated K32 population differs from frozen generation manifest")
    for column in (*CHECKPOINT_IDENTITY_COLUMNS[1:], "rollout_seed"):
        if not joined[column].astype(str).eq(joined[f"{column}_expected"].astype(str)).all():
            raise RuntimeError(f"generated K32 {column} differs from frozen generation manifest")
    for row in generated.itertuples(index=False):
        block_start = (int(row.rollout_index) // 4) * 4
        expected_job = stable_hash(["k32-four-slot-job-v1", row.checkpoint_key, block_start])
        if str(row.job_id) != expected_job:
            raise RuntimeError("generated K32 job identity differs")
    for checkpoint_key, part in generated.groupby("checkpoint_key", sort=True):
        if sorted(part["rollout_index"].astype(int)) != list(range(16, 32)):
            raise RuntimeError(f"{checkpoint_key}: generated K32 slot set differs")
        if part["rollout_seed"].astype(int).nunique() != 16:
            raise RuntimeError(f"{checkpoint_key}: generated K32 seed collision")
    generated["outcome_source"] = "generated_confirmation"
    generated["dense_reference_K"] = 32
    return generated


def merge_k32_nested_outcomes(
    *,
    config: Mapping[str, Any],
    threshold_root: str | Path,
    output_root: str | Path,
    generated_outcomes_path: str | Path | None = None,
) -> pd.DataFrame:
    """Validate and merge the exact frozen 256-by-32 nested outcome pool."""

    output = Path(output_root)
    checkpoint = _validate_confirmation_inventory(config, output)
    generation = _validate_generation_manifest(config, output, checkpoint)
    k16 = _load_frozen_k16(config, Path(threshold_root), checkpoint)
    generated_path = Path(generated_outcomes_path or output / "generated_slots_16_31.parquet")
    generated = _load_generated_k32(generated_path, generation)
    common = sorted(set(k16.columns) | set(generated.columns))
    combined = pd.concat(
        [k16.reindex(columns=common), generated.reindex(columns=common)],
        ignore_index=True,
    ).sort_values(["checkpoint_key", "rollout_index"], kind="mergesort").reset_index(drop=True)
    if len(combined) != 256 * 32 or combined["checkpoint_key"].nunique() != 256:
        raise RuntimeError("combined nested pool is not exactly 256 by 32")
    if combined.duplicated(["checkpoint_key", "rollout_index"]).any():
        raise RuntimeError("combined nested pool contains a duplicate checkpoint slot")
    for checkpoint_key, part in combined.groupby("checkpoint_key", sort=True):
        if part["rollout_index"].astype(int).tolist() != list(range(32)):
            raise RuntimeError(f"{checkpoint_key}: combined nested slot order differs")
        if part["rollout_seed"].astype(int).nunique() != 32:
            raise RuntimeError(f"{checkpoint_key}: combined nested seed collision")
        # Every nested label is an exact prefix of this one immutable order.
        for k in NESTED_K:
            if part.iloc[:k]["rollout_index"].astype(int).tolist() != list(range(k)):
                raise RuntimeError(f"{checkpoint_key}: K={k} is not a strict nested prefix")
    _write_parquet_once(output / "k32_nested_outcomes.parquet", combined)
    return combined


def run_part_b_analysis(
    *, config: Mapping[str, Any], threshold_root: str | Path, output_root: str | Path
) -> dict[str, Any]:
    """Run the frozen analysis implementation on the validated K=32 subset."""

    output = Path(output_root)
    result = run_analysis(
        config=config,
        threshold_root=threshold_root,
        output_root=output,
        analysis_name=PART_B,
        checkpoint_path=output / "k32_confirmation_checkpoints.parquet",
        outcomes_path=output / "k32_nested_outcomes.parquet",
        ks=NESTED_K,
        dense_k=32,
        required_metrics_filename="k32_confirmation_metrics.csv",
    )
    _write_json_once(output / f"internal/{PART_B}_complete.json", result)
    return result


def _validate_analysis_frame(
    frame: pd.DataFrame, *, analysis: str, expected_k: Sequence[int], name: str
) -> None:
    _required_columns(frame, ("analysis", "K"), name)
    if set(frame["analysis"].astype(str)) != {analysis}:
        raise RuntimeError(f"{name} analysis identity differs")
    if set(frame["K"].astype(int)) != set(map(int, expected_k)):
        raise RuntimeError(f"{name} nested K set differs")


def collate_analysis_outputs(output_root: str | Path) -> dict[str, Any]:
    """Combine the completed full-K16 and selected-K32 analysis products."""

    output = Path(output_root)
    part_specs = {
        PART_A: (1, 2, 4, 8, 16),
        PART_B: NESTED_K,
    }
    reliability_frames = []
    split_frames = []
    stability_frames = []
    bootstrap: dict[str, Any] = {}
    for analysis, ks in part_specs.items():
        completion_path = output / f"internal/{analysis}_complete.json"
        completion = json.loads(completion_path.read_text(encoding="utf-8"))
        if completion.get("status") != "COMPLETE" or completion.get("analysis") != analysis:
            raise RuntimeError(f"{analysis} completion marker differs")
        reliability = pd.read_csv(output / f"internal/{analysis}_label_reliability.csv")
        split = pd.read_csv(output / f"internal/{analysis}_split_sample.csv")
        stability = pd.read_csv(output / f"internal/{analysis}_architecture_stability.csv")
        payload = json.loads((output / f"internal/{analysis}_bootstrap.json").read_text())
        _validate_analysis_frame(
            reliability, analysis=analysis, expected_k=ks, name=f"{analysis} reliability"
        )
        _validate_analysis_frame(
            stability, analysis=analysis, expected_k=ks, name=f"{analysis} stability"
        )
        split_ks = tuple(k for k in ks if 2 * int(k) <= max(ks))
        _validate_analysis_frame(
            split, analysis=analysis, expected_k=split_ks, name=f"{analysis} split sample"
        )
        if payload.get("analysis") != analysis or int(payload.get("dense_reference_K", -1)) != max(ks):
            raise RuntimeError(f"{analysis} bootstrap identity differs")
        if set(payload.get("results", {})) != {f"K{k}" for k in ks}:
            raise RuntimeError(f"{analysis} bootstrap nested K set differs")
        reliability_frames.append(reliability)
        split_frames.append(split)
        stability_frames.append(stability)
        bootstrap[analysis] = payload
    reliability = pd.concat(reliability_frames, ignore_index=True).sort_values(
        ["analysis", "K", "aggregation", "model_key"], kind="mergesort"
    ).reset_index(drop=True)
    stability = pd.concat(stability_frames, ignore_index=True).sort_values(
        ["analysis", "K", "architecture"], kind="mergesort"
    ).reset_index(drop=True)
    split = pd.concat(split_frames, ignore_index=True).sort_values(
        ["analysis", "K", "aggregation", "model_key"], kind="mergesort"
    ).reset_index(drop=True)
    payload = {
        "schema_version": 1,
        "analyses": bootstrap,
        "bootstrap_unit": "complete_trace_within_model",
        "models_resampled_independently": True,
        "aggregate": "equal_weight_four_model_macro",
    }
    _write_csv_once(output / "k_label_reliability.csv", reliability)
    _write_csv_once(output / "k_split_sample_reliability.csv", split)
    _write_csv_once(output / "k_architecture_order_stability.csv", stability)
    _write_json_once(output / "k_bootstrap_intervals.json", payload)
    return {
        "reliability": reliability,
        "split_sample": split,
        "stability": stability,
        "bootstrap": payload,
    }


def _save_figure_once(path: Path, draw: Any) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    path.parent.mkdir(parents=True, exist_ok=True)
    figure = draw(plt)
    figure.tight_layout()
    with tempfile.NamedTemporaryFile(dir=path.parent, suffix=".png", delete=False) as handle:
        temporary = Path(handle.name)
    try:
        figure.savefig(temporary, dpi=240, bbox_inches="tight")
        plt.close(figure)
        if path.exists():
            if sha256_file(path) != sha256_file(temporary):
                raise RuntimeError(f"refusing to overwrite conflicting figure: {path}")
        else:
            temporary.replace(path)
    finally:
        temporary.unlink(missing_ok=True)


def generate_publication_figures(output_root: str | Path) -> dict[str, str]:
    output = Path(output_root)
    figures = output / "figures"
    reliability = pd.read_csv(output / "k_label_reliability.csv")
    split = pd.read_csv(output / "k_split_sample_reliability.csv")
    stability = pd.read_csv(output / "k_architecture_order_stability.csv")
    metrics = pd.concat(
        [
            pd.read_csv(output / "k16_full_corpus_metrics.csv"),
            pd.read_csv(output / "k32_confirmation_metrics.csv"),
        ],
        ignore_index=True,
    )
    bootstrap = json.loads((output / "k_bootstrap_intervals.json").read_text())

    macro_reliability = reliability.loc[reliability["aggregation"].eq("equal_model_macro")]
    macro_split = split.loc[split["aggregation"].eq("equal_model_macro")]

    def label_figure(plt: Any) -> Any:
        figure, axes = plt.subplots(1, 2, figsize=(10.5, 4.2))
        for analysis, part in macro_reliability.groupby("analysis", sort=True):
            axes[0].plot(part["K"], part["mae"], marker="o", label=analysis)
        for analysis, part in macro_split.groupby("analysis", sort=True):
            axes[1].plot(part["K"], part["mae"], marker="o", label=analysis)
        axes[0].set(xscale="log", xlabel="Nested K", ylabel="MAE to dense label", title="Label error")
        axes[1].set(xscale="log", xlabel="Nested K", ylabel="MAE between disjoint estimates", title="Split-sample reliability")
        for axis in axes:
            axis.set_xticks(NESTED_K, [str(k) for k in NESTED_K]); axis.grid(alpha=.25); axis.legend(fontsize=8)
        return figure

    macro_metrics = metrics.loc[metrics["aggregation"].eq("equal_model_macro")]

    architecture_rows = []
    nonlinear = {"local_mlp", "change_aware_mlp", "causal_gru"}
    for (analysis, k), part in macro_metrics.groupby(["analysis", "K"], sort=True):
        ordered = part.sort_values(
            ["trace_weighted_binomial_nll", "architecture"], kind="mergesort"
        )
        linear = ordered.loc[ordered["architecture"].eq("linear_probe")].iloc[0]
        best = ordered.loc[ordered["architecture"].isin(nonlinear)].iloc[0]
        architecture_rows.append(
            {
                "analysis": analysis,
                "K": int(k),
                "best_nonlinear": best["architecture"],
                "best_nonlinear_minus_linear_nll": float(
                    best["trace_weighted_binomial_nll"]
                    - linear["trace_weighted_binomial_nll"]
                ),
            }
        )
    architecture_summary = pd.DataFrame(architecture_rows)

    def nll_figure(plt: Any) -> Any:
        figure, axes = plt.subplots(1, 2, figsize=(11.5, 4.3), sharey=True)
        for axis, analysis in zip(axes, (PART_A, PART_B), strict=True):
            part = macro_metrics.loc[macro_metrics["analysis"].eq(analysis)]
            for architecture, values in part.groupby("architecture", sort=True):
                axis.plot(values["K"], values["trace_weighted_binomial_nll"], marker="o", label=architecture)
            axis.set(xscale="log", xlabel="Nested K", title=analysis)
            axis.set_xticks(sorted(part["K"].unique()), [str(int(k)) for k in sorted(part["K"].unique())]); axis.grid(alpha=.25)
        axes[0].set_ylabel("Trace-weighted binomial NLL")
        axes[1].legend(fontsize=7)
        return figure

    def delta_figure(plt: Any) -> Any:
        figure, axes = plt.subplots(1, 2, figsize=(11.5, 4.3), sharey=True)
        for axis, analysis in zip(axes, (PART_A, PART_B), strict=True):
            part = architecture_summary.loc[architecture_summary["analysis"].eq(analysis)]
            axis.plot(part["K"], part["best_nonlinear_minus_linear_nll"], marker="o")
            axis.axhline(0, color="black", linewidth=.8)
            axis.set(xscale="log", xlabel="Nested K", title=analysis)
            axis.xaxis.set_label_coords(0.5, -0.10)
            axis.set_xticks(sorted(part["K"].unique()), [str(int(k)) for k in sorted(part["K"].unique())]); axis.grid(alpha=.25)
        axes[0].set_ylabel("NLL(best nonlinear) - NLL(linear)")
        return figure

    def win_figure(plt: Any) -> Any:
        figure, axes = plt.subplots(1, 2, figsize=(11.5, 4.3), sharey=True)
        for axis, analysis in zip(axes, (PART_A, PART_B), strict=True):
            part = stability.loc[stability["analysis"].eq(analysis)]
            for architecture, values in part.groupby("architecture", sort=True):
                axis.plot(values["K"], values["win_fraction"], marker="o", label=architecture)
            axis.set(xscale="log", xlabel="Nested K", title=analysis, ylim=(-.02, 1.02))
            axis.set_xticks(sorted(part["K"].unique()), [str(int(k)) for k in sorted(part["K"].unique())]); axis.grid(alpha=.25)
        axes[0].set_ylabel("Permutation win fraction")
        axes[1].legend(fontsize=7)
        return figure

    drawers = (label_figure, nll_figure, delta_figure, win_figure)
    paths = {}
    for name, draw in zip(FIGURE_NAMES, drawers, strict=True):
        path = figures / name
        _save_figure_once(path, draw)
        paths[name] = path.relative_to(output).as_posix()
    return paths


def _markdown_table(frame: pd.DataFrame, columns: Sequence[str], digits: int = 5) -> str:
    if frame.empty:
        raise RuntimeError("refusing to render an empty scientific table")
    lines = [
        "| " + " | ".join(columns) + " |",
        "| " + " | ".join("---" for _ in columns) + " |",
    ]
    for row in frame[list(columns)].itertuples(index=False, name=None):
        values = []
        for value in row:
            if isinstance(value, (float, np.floating)):
                values.append("NA" if not math.isfinite(float(value)) else f"{float(value):.{digits}f}")
            else:
                values.append(str(value))
        lines.append("| " + " | ".join(values) + " |")
    return "\n".join(lines)


def _preserve_historical_blocker(output: Path) -> dict[str, Any]:
    source = output / "run_manifest.json"
    destination = output / "internal/teacher_forced_test_blocker_run_manifest.json"
    if not source.is_file():
        return {"present": False, "preserved": False}
    payload = json.loads(source.read_text(encoding="utf-8"))
    if payload.get("status") != "BLOCKED_BEFORE_ROLLOUT_GENERATION":
        raise RuntimeError("existing run_manifest.json is not the historical sparse-TEST blocker")
    if destination.exists():
        if destination.read_bytes() != source.read_bytes():
            raise RuntimeError("preserved sparse-TEST blocker manifest differs")
    else:
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(source, destination)
    return {
        "present": True,
        "preserved": True,
        "path": destination.relative_to(output).as_posix(),
        "sha256": sha256_file(destination),
        "status": payload["status"],
        "superseded_by": "post_hoc_calibration_k_densification",
        "untouched_evidence": [
            "BLOCKED.json",
            "PREFLIGHT_BLOCKER_REPORT.md",
            "candidate_problem_census.csv",
        ],
    }


def render_report(output_root: str | Path, historical: Mapping[str, Any]) -> Path:
    output = Path(output_root)
    reliability = pd.read_csv(output / "k_label_reliability.csv")
    split = pd.read_csv(output / "k_split_sample_reliability.csv")
    stability = pd.read_csv(output / "k_architecture_order_stability.csv")
    metrics = pd.concat(
        [
            pd.read_csv(output / "k16_full_corpus_metrics.csv"),
            pd.read_csv(output / "k32_confirmation_metrics.csv"),
        ],
        ignore_index=True,
    )
    bootstrap = json.loads((output / "k_bootstrap_intervals.json").read_text())
    macro_reliability = reliability.loc[reliability["aggregation"].eq("equal_model_macro")]
    macro_split = split.loc[split["aggregation"].eq("equal_model_macro")]
    macro_metrics = metrics.loc[metrics["aggregation"].eq("equal_model_macro")]
    nonlinear = {"local_mlp", "change_aware_mlp", "causal_gru"}
    architecture_rows = []
    for (analysis, k), part in macro_metrics.groupby(["analysis", "K"], sort=True):
        ordered = part.sort_values(
            ["trace_weighted_binomial_nll", "architecture"], kind="mergesort"
        )
        linear = ordered.loc[ordered["architecture"].eq("linear_probe")].iloc[0]
        best = ordered.loc[ordered["architecture"].isin(nonlinear)].iloc[0]
        architecture_rows.append(
            {
                "analysis": analysis,
                "K": int(k),
                "linear_nll": float(linear["trace_weighted_binomial_nll"]),
                "best_nonlinear": best["architecture"],
                "best_nonlinear_nll": float(best["trace_weighted_binomial_nll"]),
                "nonlinear_minus_linear_nll": float(
                    best["trace_weighted_binomial_nll"]
                    - linear["trace_weighted_binomial_nll"]
                ),
                "linear_concordance": float(linear["within_trace_concordance"]),
                "best_nonlinear_concordance": float(best["within_trace_concordance"]),
                "nll_order": " < ".join(ordered["architecture"].astype(str)),
            }
        )
    architecture_summary = pd.DataFrame(architecture_rows)
    permutation_winners = (
        stability.sort_values(
            ["analysis", "K", "win_fraction", "architecture"],
            ascending=[True, True, False, True],
            kind="mergesort",
        )
        .groupby(["analysis", "K"], sort=True, as_index=False)
        .first()[
            [
                "analysis",
                "K",
                "architecture",
                "win_fraction",
                "mean_rank_correlation_with_dense",
                "delta_sign_change_fraction",
            ]
        ]
        .rename(columns={"architecture": "permutation_winner"})
    )
    bootstrap_rows = []
    for analysis, payload in bootstrap["analyses"].items():
        for key, value in payload["results"].items():
            bootstrap_rows.append(
                {
                    "analysis": analysis,
                    "K": int(key[1:]),
                    "comparison": value["comparison"],
                    "estimate": value["estimate"],
                    "ci_low": value["ci_low"],
                    "ci_high": value["ci_high"],
                }
            )
    bootstrap_summary = pd.DataFrame(bootstrap_rows).sort_values(["analysis", "K"])
    historical_text = (
        f"The original sparse teacher-forced TEST design remains preserved at "
        f"`{historical['path']}`. `BLOCKED.json`, `PREFLIGHT_BLOCKER_REPORT.md`, and "
        "`candidate_problem_census.csv` are retained untouched as superseded historical evidence."
        if historical.get("present")
        else "No historical sparse-TEST blocker manifest was present in this run directory."
    )
    text = f"""# SafePrefix calibration K-densification robustness report

## Scope and required caveats

{REQUIRED_CAVEATS[0]}

{REQUIRED_CAVEATS[1]}

{REQUIRED_CAVEATS[2]}

Part A uses every valid calibration checkpoint with the immutable K=16 pool. Part B uses the outcome-blind, position-stratified 256-checkpoint confirmation subset with the same K=16 slots plus newly generated slots 16--31. These populations answer complementary questions and their raw values should not be treated as a paired full-corpus comparison. Nested K values reuse prefixes of one immutable rollout order and are therefore statistically dependent.

No predictor was retrained, no calibrator was refit, no threshold was selected, and no native, teacher-forced TEST, architecture-development, training, or geometry-child outcome was loaded.

## Label reliability

{_markdown_table(macro_reliability, ['analysis','K','dense_reference_K','mae','rmse','spearman','majority_label_agreement','checkpoints'])}

The complete model-specific and equal-model-macro results are in `k_label_reliability.csv`.

## Independent split-sample reliability

{_markdown_table(macro_split, ['analysis','K','replicates','mae','rmse','spearman','majority_label_agreement'])}

## Frozen architecture comparison

{_markdown_table(architecture_summary, ['analysis','K','linear_nll','best_nonlinear','best_nonlinear_nll','nonlinear_minus_linear_nll','linear_concordance','best_nonlinear_concordance','nll_order'])}

Negative `nonlinear_minus_linear_nll` values favor the named frozen nonlinear predictor; positive values favor the frozen linear probe. This is an evaluation-label robustness comparison, not a claim about retraining with denser labels.

## Permutation architecture-order stability

{_markdown_table(permutation_winners, ['analysis','K','permutation_winner','win_fraction','mean_rank_correlation_with_dense','delta_sign_change_fraction'])}

The full five-architecture win frequencies and ranks are in `k_architecture_order_stability.csv`.

## Paired trace-bootstrap intervals

{_markdown_table(bootstrap_summary, ['analysis','K','comparison','estimate','ci_low','ci_high'])}

Each bootstrap independently resamples complete traces within each model and then forms an equal-weight four-model macro average. The full machine-readable intervals are in `k_bootstrap_intervals.json`.

## Publication figures

""" + "\n".join(f"- `{(Path('figures') / name).as_posix()}`" for name in FIGURE_NAMES) + f"""

## Historical design record

{historical_text}

## Integrity boundary

`k32_nested_outcomes.parquet` contains exactly 256 checkpoints by 32 ordered slots. Slots 0--15 retain the frozen threshold-run rows; slots 16--31 match the frozen generation manifest, seed derivation, checkpoint identity, job identity, and per-row artifact hashes. `COMPLETE.json` is written only after all required machine-readable products, figures, this report, and `run_manifest.json` pass terminal validation.
"""
    path = output / REPORT_NAME
    _write_text_once(path, text)
    return path


def _required_paths(output: Path, *, include_manifest: bool) -> list[Path]:
    paths = [
        *(output / name for name in REQUIRED_PROTOCOL_FILES),
        output / "k32_nested_outcomes.parquet",
        *(output / name for name in REQUIRED_ANALYSIS_FILES),
        *(output / "figures" / name for name in FIGURE_NAMES),
        output / REPORT_NAME,
    ]
    if include_manifest:
        paths.append(output / "run_manifest.json")
    return paths


def validate_required_outputs(
    output_root: str | Path, *, include_manifest: bool = True
) -> dict[str, Any]:
    output = Path(output_root)
    paths = _required_paths(output, include_manifest=include_manifest)
    missing = [path.relative_to(output).as_posix() for path in paths if not path.is_file() or path.stat().st_size == 0]
    if missing:
        raise RuntimeError(f"required K-densification outputs are missing: {missing}")
    outcomes = pd.read_parquet(output / "k32_nested_outcomes.parquet")
    if len(outcomes) != 8192 or outcomes["checkpoint_key"].nunique() != 256:
        raise RuntimeError("terminal K32 outcome dimensions differ")
    counts = outcomes.groupby("checkpoint_key")["rollout_index"].agg(["count", "nunique", "min", "max"])
    if not (
        counts["count"].eq(32).all()
        and counts["nunique"].eq(32).all()
        and counts["min"].eq(0).all()
        and counts["max"].eq(31).all()
    ):
        raise RuntimeError("terminal K32 nested coverage differs")
    reliability = pd.read_csv(output / "k_label_reliability.csv")
    stability = pd.read_csv(output / "k_architecture_order_stability.csv")
    for frame, name in ((reliability, "reliability"), (stability, "stability")):
        if set(frame["analysis"].astype(str)) != {PART_A, PART_B}:
            raise RuntimeError(f"terminal {name} analysis set differs")
    bootstrap = json.loads((output / "k_bootstrap_intervals.json").read_text())
    if set(bootstrap.get("analyses", {})) != {PART_A, PART_B}:
        raise RuntimeError("terminal bootstrap analysis set differs")
    inventory = json.loads((output / "calibration_k16_inventory.json").read_text())
    inventory_count = inventory.get("checkpoint_rows", inventory.get("checkpoint_count", -1))
    if (
        inventory.get("status") not in {"PASS", "COMPLETE_K16_REUSE_ONLY"}
        or int(inventory_count) != 1932
        or bool(inventory.get("native_outcomes_loaded"))
        or bool(inventory.get("teacher_forced_test_loaded"))
        or int(inventory.get("new_generation_for_part_a", 0)) != 0
    ):
        raise RuntimeError("terminal calibration K16 inventory differs")
    reuse = json.loads((output / "k32_reuse_report.json").read_text())
    if (
        reuse.get("status") != "COMPLETE"
        or int(reuse.get("reused_slots_0_15", -1)) != 4096
        or int(reuse.get("projected_new_rollouts", -1)) != 4096
        or bool(reuse.get("native_artifacts_loaded"))
    ):
        raise RuntimeError("terminal K32 reuse report differs")
    utilization = json.loads((output / "gpu_utilization_summary.json").read_text())
    if (
        utilization.get("status") != "COMPLETE"
        or int(utilization.get("worker_count", -1)) != 10
        or int(utilization.get("generated_outcomes", -1)) != 4096
        or str(utilization.get("generated_sha256"))
        != sha256_file(output / "generated_slots_16_31.parquet")
    ):
        raise RuntimeError("terminal GPU utilization summary differs")
    transitions = json.loads((output / "gpu_worker_transitions.json").read_text())
    if not isinstance(transitions, list) or len(transitions) < 10:
        raise RuntimeError("terminal GPU worker transition log differs")
    registry_uri = f"file:{(output / 'k32_rollout_registry.sqlite').resolve()}?mode=ro"
    with sqlite3.connect(registry_uri, uri=True) as connection:
        integrity = connection.execute("PRAGMA integrity_check").fetchone()[0]
        job_status = dict(
            connection.execute("SELECT status,COUNT(*) FROM jobs GROUP BY status").fetchall()
        )
        registry_slots = int(connection.execute("SELECT COUNT(*) FROM rollout_slots").fetchone()[0])
    if integrity != "ok" or job_status != {"complete": 1024} or registry_slots != 8192:
        raise RuntimeError("terminal rollout registry differs")
    for path, analysis, ks in (
        (output / "k16_full_corpus_metrics.csv", PART_A, (1, 2, 4, 8, 16)),
        (output / "k32_confirmation_metrics.csv", PART_B, NESTED_K),
    ):
        _validate_analysis_frame(
            pd.read_csv(path), analysis=analysis, expected_k=ks, name=path.name
        )
    report = (output / REPORT_NAME).read_text(encoding="utf-8")
    missing_caveats = [statement for statement in REQUIRED_CAVEATS if statement not in report]
    if missing_caveats:
        raise RuntimeError(f"required report caveats are missing: {missing_caveats}")
    for name in FIGURE_NAMES:
        if not (output / "figures" / name).read_bytes().startswith(b"\x89PNG\r\n\x1a\n"):
            raise RuntimeError(f"publication figure is not a PNG: {name}")
    if include_manifest:
        manifest = json.loads((output / "run_manifest.json").read_text())
        if manifest.get("status") != "VALIDATED_AWAITING_COMPLETE_MARKER":
            raise RuntimeError("terminal run manifest status differs")
        expected_artifacts = {
            path.relative_to(output).as_posix()
            for path in _required_paths(output, include_manifest=False)
        }
        artifact_rows = manifest.get("artifacts", [])
        observed_artifacts = {str(row.get("path")): row for row in artifact_rows}
        if set(observed_artifacts) != expected_artifacts:
            raise RuntimeError("terminal run manifest artifact set differs")
        for relative, row in observed_artifacts.items():
            path = output / relative
            if (
                int(row.get("size", -1)) != path.stat().st_size
                or str(row.get("sha256")) != sha256_file(path)
            ):
                raise RuntimeError(f"terminal run manifest artifact hash differs: {relative}")
        guards = manifest.get("guards", {})
        if any(
            guards.get(key) is not False
            for key in (
                "native_outcomes_loaded",
                "teacher_forced_test_loaded",
                "predictor_retrained",
                "calibrator_refit",
                "threshold_reselected",
            )
        ):
            raise RuntimeError("terminal run manifest guard differs")
    return {
        "status": "PASS",
        "required_output_count": len(paths),
        "outcome_rows": len(outcomes),
        "checkpoints": int(outcomes["checkpoint_key"].nunique()),
        "slots_per_checkpoint": 32,
        "required_caveats_present": True,
    }


def _artifact_records(output: Path) -> list[dict[str, Any]]:
    records = []
    for path in _required_paths(output, include_manifest=False):
        records.append(
            {
                "path": path.relative_to(output).as_posix(),
                "size": path.stat().st_size,
                "sha256": sha256_file(path),
            }
        )
    return records


def finalize_k_densification_reporting(
    *,
    config: Mapping[str, Any],
    threshold_root: str | Path,
    output_root: str | Path,
    generated_outcomes_path: str | Path | None = None,
    run_part_b: bool = True,
) -> dict[str, Any]:
    """Produce terminal calibration K-densification artifacts and COMPLETE.json."""

    output = Path(output_root)
    output.mkdir(parents=True, exist_ok=True)
    complete_path = output / "COMPLETE.json"
    if complete_path.exists():
        raise RuntimeError("refusing to modify a completed K-densification run")
    historical = _preserve_historical_blocker(output)
    merged = merge_k32_nested_outcomes(
        config=config,
        threshold_root=threshold_root,
        output_root=output,
        generated_outcomes_path=generated_outcomes_path,
    )
    if run_part_b:
        part_b = run_part_b_analysis(
            config=config, threshold_root=threshold_root, output_root=output
        )
    else:
        completion_path = output / f"internal/{PART_B}_complete.json"
        if not completion_path.is_file():
            raise RuntimeError("existing Part B outputs lack a completion marker")
        part_b = json.loads(completion_path.read_text(encoding="utf-8"))
        if part_b.get("status") != "COMPLETE" or part_b.get("analysis") != PART_B:
            raise RuntimeError("existing Part B completion marker differs")
    collate_analysis_outputs(output)
    figures = generate_publication_figures(output)
    report = render_report(output, historical)
    pre_manifest_validation = validate_required_outputs(output, include_manifest=False)
    manifest = {
        "schema_version": 1,
        "status": "VALIDATED_AWAITING_COMPLETE_MARKER",
        "experiment_id": str(config["experiment"]["id"]),
        "created_at": now_iso(),
        "git_commit": git_commit(),
        "package_versions": package_versions(),
        "cohort_role": "post_hoc_calibration_robustness_not_final_test",
        "nested_K": list(NESTED_K),
        "dense_reference_K": 32,
        "checkpoint_count": 256,
        "outcome_rows": len(merged),
        "part_a": {
            "analysis": PART_A,
            "population": "all_valid_calibration_checkpoints",
            "dense_reference_K": 16,
            "new_generation": 0,
        },
        "part_b": {
            "analysis": PART_B,
            "population": "outcome_blind_position_stratified_confirmation_subset",
            "dense_reference_K": 32,
            "new_generation": 4096,
            "analysis_result": part_b,
        },
        "historical_sparse_test_blocker": historical,
        "historical_blocker_evidence_is_superseded_not_deleted": bool(historical.get("present")),
        "required_caveats": list(REQUIRED_CAVEATS),
        "guards": {
            "native_outcomes_loaded": False,
            "teacher_forced_test_loaded": False,
            "architecture_dev_outcomes_loaded": False,
            "training_outcomes_loaded": False,
            "geometry_child_outcomes_loaded": False,
            "predictor_retrained": False,
            "calibrator_refit": False,
            "threshold_reselected": False,
            "k16_regenerated": False,
        },
        "source_sha256": {
            "frozen_k16": sha256_file(
                Path(threshold_root) / "raw_outcomes/merged_k16_checkpoint_suffixes.parquet"
            ),
            "confirmation_manifest": sha256_file(output / "k32_confirmation_manifest.json"),
            "generation_manifest": sha256_file(
                output / "generation_input/k32_generation_manifest.parquet"
            ),
            "generated_slots_16_31": sha256_file(
                Path(generated_outcomes_path or output / "generated_slots_16_31.parquet")
            ),
            "k32_nested_outcomes": sha256_file(output / "k32_nested_outcomes.parquet"),
        },
        "publication_figures": figures,
        "generation_provenance": {
            "orchestration_state": (
                json.loads((output / "orchestration_state.json").read_text())
                if (output / "orchestration_state.json").is_file()
                else None
            ),
            "gpu_utilization_summary_sha256": sha256_file(
                output / "gpu_utilization_summary.json"
            ),
            "worker_transitions_sha256": sha256_file(
                output / "gpu_worker_transitions.json"
            ),
        },
        "report": report.relative_to(output).as_posix(),
        "artifacts": _artifact_records(output),
        "pre_manifest_validation": pre_manifest_validation,
    }
    # Replacing run_manifest.json is intentional only for the preserved sparse-
    # TEST blocker.  Any other pre-existing manifest was rejected above.
    atomic_json(output / "run_manifest.json", manifest)
    validation = validate_required_outputs(output, include_manifest=True)
    terminal = {
        "schema_version": 1,
        "status": "COMPLETE",
        "completed_at": now_iso(),
        "experiment_id": str(config["experiment"]["id"]),
        "cohort_role": "post_hoc_calibration_robustness_not_final_test",
        "checkpoint_count": 256,
        "outcome_rows": 8192,
        "slots_per_checkpoint": 32,
        "integrity_status": validation["status"],
        "run_manifest_sha256": sha256_file(output / "run_manifest.json"),
        "native_evaluation_used": False,
        "teacher_forced_test_used": False,
        "predictor_retrained": False,
        "calibrator_refit": False,
        "threshold_reselected": False,
        "historical_sparse_test_blocker_preserved": bool(historical.get("preserved")),
        "required_output_count": validation["required_output_count"],
    }
    atomic_json(complete_path, terminal)
    return terminal
