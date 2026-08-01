from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest

from safeprefix.prefix_validity_v1.data import PrefixValidityData, PrefixValidityExtractionPlan
from safeprefix.prefix_validity_v1.runner import (
    PrefixValidityRunError,
    assert_processbench_bundle,
    build_missing_feature_packs,
    load_boundary_selected_layers,
    load_prefix_validity_config,
    train_and_freeze_model,
    write_final_report,
)
from safeprefix.boundary_v1.data import sha256_file, stable_hash
import numpy as np
import torch


ROOT = Path(__file__).resolve().parents[1]


def test_frozen_config_is_probe_training_only() -> None:
    config = load_prefix_validity_config(ROOT / "configs/prefix_validity_v1.yaml")
    assert config["protocol"]["allow_native_application"] is False
    assert config["protocol"]["allow_new_suffix_rollouts_after_selection_freeze"] is False
    assert config["protocol"]["allow_recoverability_probe_modification"] is False
    assert config["protocol"]["allow_recoverability_cutoff_modification"] is False
    assert config["execution"]["reuse_completed_safety_features"] is False
    assert load_boundary_selected_layers(
        ROOT / "configs/boundary_model_v1.yaml", "family_a_small"
    ) == [18, 27, -1]


def test_config_rejects_native_application(tmp_path: Path) -> None:
    path = tmp_path / "bad.yaml"
    source = (ROOT / "configs/prefix_validity_v1.yaml").read_text()
    path.write_text(source.replace("allow_native_application: false", "allow_native_application: true"))
    with pytest.raises(PrefixValidityRunError, match="probe training only"):
        load_prefix_validity_config(path)


def test_missing_feature_packs_are_deterministic_and_cover_only_missing() -> None:
    models = ["family_a_small", "family_a_large", "family_b_small", "family_b_large"]
    traces = {}
    rows = []
    for model in models:
        traces[model] = []
        for index, origin in enumerate(["reused_teacher_forced_completion_r5", "new_full_step_teacher_forcing_required", "new_full_step_teacher_forcing_required"]):
            row = {"trace_id": f"{model}-{index}", "feature_origin": origin}
            traces[model].append(row)
            rows.append({"model_key": model, **row})
    plan = PrefixValidityExtractionPlan(
        rows=pd.DataFrame(rows),
        summary={"total_new_teacher_forcing_traces": 8},
        traces_by_model=traces,
    )
    packs, index = build_missing_feature_packs(plan, traces_per_pack=1)
    assert len(packs) == 8
    assert len(index) == 8
    assert all(pack["trace_count"] == 1 for pack in packs)
    again, _ = build_missing_feature_packs(plan, traces_per_pack=1)
    assert again == packs


def test_report_is_written_from_terminal_integrity(tmp_path: Path) -> None:
    path = write_final_report(
        output_root=tmp_path,
        summary={"processbench": {}},
        integrity={"status": "PASS", "native_application_run": False},
    )
    text = path.read_text()
    assert "Status: PASS" in text
    assert "Native application was not run" in text


def test_training_does_not_open_processbench_test_before_global_freeze(tmp_path: Path) -> None:
    rows = []
    for split in ("train", "architecture_dev", "calibration", "teacher_forced_test"):
        for trace_number in range(5):
            trace_id = f"{split}-{trace_number}"
            for checkpoint in (1, 2, 3):
                rows.append({
                    "model_key": "family_a_small", "base_model": "family_a_small",
                    "domain": "processbench_math", "split": split, "trace_id": trace_id,
                    "checkpoint_id": f"{trace_id}:{checkpoint}",
                    "checkpoint_index": checkpoint, "checkpoint_ordinal": checkpoint,
                    "checkpoint_token_offset": checkpoint * 10, "prefix_token_count": checkpoint * 10,
                    "total_checkpoint_count": 3, "full_trace_token_count": 40,
                    "total_trace_token_count": 40, "prefix_valid": int(checkpoint < 3),
                    "true_last_valid_checkpoint": 2, "feature_row_index": len(rows),
                })
    frame = pd.DataFrame(rows)
    data = PrefixValidityData(
        rows=frame, features={"family_a_small": torch.tensor(np.random.default_rng(3).normal(size=(len(frame), 4)), dtype=torch.float16)},
        exclusions=pd.DataFrame(), inventory=pd.DataFrame(), pack_inventory=pd.DataFrame(),
        indexing_convention={}, integrity={},
    )
    config = load_prefix_validity_config(ROOT / "configs/prefix_validity_v1.yaml")
    config["training"].update(max_epochs=1, patience=1, batch_size_traces=16)
    train_and_freeze_model(
        data=data, model_key="family_a_small", config=config, output_root=tmp_path, device="cpu",
    )
    assert not list(tmp_path.glob("**/test_predictions.parquet"))
    assert not list(tmp_path.glob("**/test_metrics.json"))


def test_training_corpus_physically_excludes_test_rows(tmp_path: Path) -> None:
    rows = []
    for split in ("train", "architecture_dev", "calibration", "teacher_forced_test"):
        for trace_number in range(5):
            trace_id = f"{split}-{trace_number}"
            # Deliberately make the held-out trace labels non-monotonic.  The
            # fitting function must not even construct a corpus containing
            # these rows before the global freeze.
            labels = (0, 1, 0) if split == "teacher_forced_test" else (1, 1, 0)
            for checkpoint, label in zip((1, 2, 3), labels):
                rows.append({
                    "model_key": "family_a_small", "base_model": "family_a_small",
                    "domain": "processbench_math", "split": split, "trace_id": trace_id,
                    "checkpoint_id": f"{trace_id}:{checkpoint}",
                    "checkpoint_index": checkpoint, "checkpoint_ordinal": checkpoint,
                    "checkpoint_token_offset": checkpoint * 10,
                    "prefix_token_count": checkpoint * 10,
                    "total_checkpoint_count": 3, "full_trace_token_count": 40,
                    "total_trace_token_count": 40, "prefix_valid": label,
                    "true_last_valid_checkpoint": 2, "feature_row_index": len(rows),
                })
    frame = pd.DataFrame(rows)
    data = PrefixValidityData(
        rows=frame,
        features={"family_a_small": torch.tensor(
            np.random.default_rng(8).normal(size=(len(frame), 4)), dtype=torch.float16
        )},
        exclusions=pd.DataFrame(), inventory=pd.DataFrame(), pack_inventory=pd.DataFrame(),
        indexing_convention={}, integrity={},
    )
    config = load_prefix_validity_config(ROOT / "configs/prefix_validity_v1.yaml")
    config["training"].update(max_epochs=1, patience=1, batch_size_traces=16)
    train_and_freeze_model(
        data=data,
        model_key="family_a_small",
        config=config,
        output_root=tmp_path,
        device="cpu",
    )


def test_processbench_bundle_rejects_artifact_mutation(tmp_path: Path) -> None:
    models = {}
    for model in ("family_a_small", "family_a_large", "family_b_small", "family_b_large"):
        models[model] = {}
        for architecture in ("linear_probe", "position_only"):
            directory = tmp_path / model / architecture
            directory.mkdir(parents=True)
            paths = {
                "probe": directory / "probe.pt",
                "calibration": directory / "calibration.json",
                "gate_cutoff": directory / "gate.json",
            }
            for name, path in paths.items():
                path.write_text(f"{model}-{architecture}-{name}")
            models[model][architecture] = {
                "probe_path": str(paths["probe"]),
                "probe_sha256": sha256_file(paths["probe"]),
                "calibration_path": str(paths["calibration"]),
                "calibration_sha256": sha256_file(paths["calibration"]),
                "gate_cutoff_path": str(paths["gate_cutoff"]),
                "gate_cutoff_sha256": sha256_file(paths["gate_cutoff"]),
                "test_accessed": False,
            }
    payload = {
        "status": "FROZEN_BEFORE_PROCESSBENCH_TEST_EVALUATION",
        "models": models,
        "native_outcomes_accessed": False,
    }
    payload["freeze_hash"] = stable_hash(payload)
    path = tmp_path / "processbench/PROCESSBENCH_COMPONENTS_FROZEN.json"
    path.parent.mkdir(parents=True)
    path.write_text(__import__("json").dumps(payload))
    assert assert_processbench_bundle(tmp_path)["freeze_hash"] == payload["freeze_hash"]
    Path(models["family_a_small"]["linear_probe"]["probe_path"]).write_text("tampered")
    with pytest.raises(PrefixValidityRunError, match="changed"):
        assert_processbench_bundle(tmp_path)
