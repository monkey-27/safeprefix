from __future__ import annotations

import importlib.util
from pathlib import Path
from typing import Any

import pytest


ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location(
    "safeprefix_full_tf_runner",
    ROOT / "scripts" / "25_run_full_teacher_forced_rollouts.py",
)
assert SPEC is not None and SPEC.loader is not None
RUNNER = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(RUNNER)


GATES = {
    "minimum_gpu_memory_headroom_fraction": 0.05,
    "minimum_gpu_utilization_samples": 1,
    "minimum_mean_gpu_utilization_percent": 20.0,
    "maximum_forwarded_to_useful_ratio": 1.35,
}


def _candidate(
    candidate_id: str,
    *,
    useful: int = 8_000,
    decode_seconds: float = 10.0,
    wall_seconds: float = 12.0,
    forwarded: int = 8_400,
    utilization_samples: int = 10,
    utilization: float = 70.0,
    reserved: int = 60,
    total_memory: int = 100,
    batch: int = 32,
    quantum: int = 128,
) -> dict[str, Any]:
    return {
        "status": "COMPLETE",
        "candidate_id": candidate_id,
        "settings": {
            "branch_batch_size": batch,
            "compaction_quantum": quantum,
            "prefill_chunk_size": 256,
            "traces_per_decode_group": 12,
            "maximum_decode_kv_bytes": 1_000_000,
        },
        "truncation_count": 2,
        "packs": [
            {
                "row_count": 40,
                "total_wall_seconds": wall_seconds,
                "gpu_utilization_sample_count": utilization_samples,
                "gpu_utilization_mean_percent": utilization,
                "gpu_total_memory_bytes": total_memory,
                "decode_metrics": {
                    "useful_output_tokens": useful,
                    "forwarded_row_steps": forwarded,
                    "decode_wall_seconds": decode_seconds,
                    "maximum_reserved_bytes": reserved,
                    "preallocated_cache_waves": 1,
                    "heterogeneous_prefix_waves": 1,
                },
            }
        ],
    }


def test_benchmark_selects_fastest_candidate_that_passes_all_gates() -> None:
    slow = _candidate("slow", useful=8_000, decode_seconds=20.0, quantum=64)
    fast = _candidate("fast", useful=8_000, decode_seconds=10.0, quantum=128)
    too_wasteful = _candidate(
        "wasteful", useful=8_000, decode_seconds=5.0, forwarded=12_000, quantum=256
    )

    selected, evaluated = RUNNER._select_benchmark_candidate(
        [slow, fast, too_wasteful],
        expected_rollouts=800,
        worker_count=8,
        benchmark_cfg=GATES,
    )

    assert selected["candidate_id"] == "fast"
    by_id = {item["candidate_id"]: item for item in evaluated}
    assert by_id["fast"]["eligible"] is True
    assert by_id["wasteful"]["eligible"] is False
    assert "forwarded_work_waste" in by_id["wasteful"]["gate_failures"]
    # 800 / 40 times twelve measured worker-seconds, divided over eight H100s.
    assert selected["projected_eight_h100_seconds"] == pytest.approx(30.0)


def test_benchmark_rejects_missing_utilization_samples() -> None:
    missing = _candidate("missing", utilization_samples=0, utilization=0.0)
    evaluated = RUNNER._evaluate_benchmark_candidate(
        missing,
        expected_rollouts=800,
        worker_count=8,
        benchmark_cfg=GATES,
    )
    assert evaluated["eligible"] is False
    assert "gpu_utilization_samples" in evaluated["gate_failures"]
    assert "gpu_utilization" in evaluated["gate_failures"]


def test_benchmark_grid_changes_only_frozen_performance_fields() -> None:
    config = {
        "production_execution": {
            "prefill_chunk_size": 256,
            "benchmark_tunable": {
                "branch_batch_sizes": {"model": 16},
                "compaction_quantum": {"model": 64},
                "traces_per_decode_group": {"model": 12},
                "maximum_decode_kv_bytes": {"model": 1234},
                "candidate_settings": {
                    "model": [
                        {"branch_batch_size": 16, "compaction_quantum": 64},
                        {"branch_batch_size": 24, "compaction_quantum": 128},
                    ]
                },
            },
        }
    }
    candidates = RUNNER._benchmark_candidate_settings(config, "model")
    assert [item["branch_batch_size"] for item in candidates] == [16, 24]
    assert [item["compaction_quantum"] for item in candidates] == [64, 128]
    assert all(item["traces_per_decode_group"] == 12 for item in candidates)

    bad = config.copy()
    bad = {
        **config,
        "production_execution": {
            **config["production_execution"],
            "benchmark_tunable": {
                **config["production_execution"]["benchmark_tunable"],
                "candidate_settings": {
                    "model": [
                        {
                            "branch_batch_size": 16,
                            "compaction_quantum": 64,
                            "temperature": 0.8,
                        }
                    ]
                },
            },
        },
    }
    with pytest.raises(ValueError, match="only branch_batch_size"):
        RUNNER._benchmark_candidate_settings(bad, "model")
