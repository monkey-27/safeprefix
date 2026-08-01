"""CPU orchestration and reporting for native failure acquisition.

The GPU worker writes immutable pack shards.  This module is the only place
that may freeze a cohort or aggregate regeneration outcomes.  In particular,
cohort selection never reads a regeneration artifact.
"""

from __future__ import annotations

import json
import time
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Mapping, Sequence

from safeprefix.native_failed_trace_acquisition import (
    STRATA,
    acquisition_metrics,
    attach_cohort_status,
    build_attempt_packs,
    build_regeneration_packs,
    bootstrap_trace_metrics,
    file_sha256,
    read_jsonl,
    regeneration_metrics,
    select_frozen_cohort,
    validate_final_integrity,
    validate_partial_regeneration_integrity,
)
from safeprefix.native_failed_trace_runtime import acquisition_root
from safeprefix.reproducibility import atomic_json, atomic_jsonl, atomic_text, stable_hash


def _write_immutable_jsonl(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    values = [dict(row) for row in rows]
    if path.is_file():
        if stable_hash(read_jsonl(path)) != stable_hash(values):
            raise RuntimeError(f"immutable manifest changed on resume: {path}")
        return
    atomic_jsonl(path, values)


def _read_complete_pack(
    pack_root: Path, *, filename: str, pack_id: str, expected_rows: int,
) -> list[dict[str, Any]]:
    marker_path = pack_root / "complete.json"
    data_path = pack_root / filename
    if not marker_path.is_file() or not data_path.is_file():
        return []
    marker = json.loads(marker_path.read_text(encoding="utf-8"))
    if (
        marker.get("status") != "COMPLETE"
        or marker.get("pack_id") != pack_id
        or int(marker.get("row_count", -1)) != int(expected_rows)
        or marker.get("data_sha256") != file_sha256(data_path)
    ):
        raise RuntimeError(f"invalid completed pack marker: {pack_root}")
    rows = read_jsonl(data_path)
    if len(rows) != int(expected_rows):
        raise RuntimeError(f"completed pack row count drift: {pack_root}")
    return rows


def attempt_pack_manifests(
    config: Mapping[str, Any], source_rows: Sequence[Mapping[str, Any]], model_key: str,
) -> dict[str, list[dict[str, Any]]]:
    digest = stable_hash(config)
    return {
        stratum: [
            pack.to_dict()
            for pack in build_attempt_packs(
                source_rows,
                model_key=model_key,
                stratum=stratum,
                pack_size=int(config["acquisition"]["attempt_pack_size"]),
                configuration_hash=digest,
            )
        ]
        for stratum in STRATA
    }


def write_attempt_pack_manifests(
    config: Mapping[str, Any], source_rows: Sequence[Mapping[str, Any]],
    *, runs_root: str | Path, run_id: str, model_key: str,
) -> dict[str, list[dict[str, Any]]]:
    root = acquisition_root(runs_root, run_id)
    manifests = attempt_pack_manifests(config, source_rows, model_key)
    for stratum, packs in manifests.items():
        _write_immutable_jsonl(
            root / "execution_packs/attempts" / model_key / f"{stratum}.jsonl", packs
        )
    return manifests


def collect_attempt_rows(
    *, runs_root: str | Path, run_id: str, model_key: str,
    manifests: Mapping[str, Sequence[Mapping[str, Any]]],
) -> tuple[list[dict[str, Any]], dict[str, list[str]]]:
    root = acquisition_root(runs_root, run_id)
    rows: list[dict[str, Any]] = []
    completed: dict[str, list[str]] = {stratum: [] for stratum in STRATA}
    for stratum in STRATA:
        for pack in manifests[stratum]:
            pack_id = str(pack["pack_id"])
            pack_rows = _read_complete_pack(
                root / "attempts/raw" / model_key / stratum / pack_id,
                filename="attempts.jsonl",
                pack_id=pack_id,
                expected_rows=len(pack["source_ids"]),
            )
            if not pack_rows:
                continue
            rows.extend(pack_rows)
            completed[stratum].append(pack_id)
    identities = [str(row["attempt_key"]) for row in rows]
    if len(identities) != len(set(identities)):
        raise RuntimeError(f"duplicate attempt rows for {model_key}")
    return rows, completed


def target_for_stratum(
    config: Mapping[str, Any], attempts: Sequence[Mapping[str, Any]], stratum: str,
    *, prior_strata_exhausted: Mapping[str, bool],
) -> int:
    """Return the frozen sequential target, including only earned deficits."""

    base = {key: int(value) for key, value in config["acquisition"]["target_by_stratum"].items()}
    valid = Counter(
        str(row["stratum"])
        for row in attempts
        if str(row.get("initial_status")) == "valid_incorrect"
    )
    if stratum == "gsm1k":
        return base[stratum]
    gsm_deficit = 0
    if prior_strata_exhausted.get("gsm1k", False):
        gsm_deficit = max(0, base["gsm1k"] - valid["gsm1k"])
    if stratum == "math_level_3":
        return base[stratum] + gsm_deficit
    math3_target = base["math_level_3"] + gsm_deficit
    math3_deficit = 0
    if prior_strata_exhausted.get("math_level_3", False):
        math3_deficit = max(0, math3_target - valid["math_level_3"])
    return base["math_level_4"] + math3_deficit


def next_attempt_wave(
    config: Mapping[str, Any], manifests: Mapping[str, Sequence[Mapping[str, Any]]],
    attempts: Sequence[Mapping[str, Any]], completed: Mapping[str, Sequence[str]],
    *, stratum: str, prior_strata_exhausted: Mapping[str, bool],
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    target = target_for_stratum(
        config, attempts, stratum, prior_strata_exhausted=prior_strata_exhausted,
    )
    ordered_packs = [dict(pack) for pack in manifests[stratum]]
    completed_ids = set(map(str, completed.get(stratum, [])))
    completed_prefix = 0
    for pack in ordered_packs:
        if str(pack["pack_id"]) not in completed_ids:
            break
        completed_prefix += 1
    prefix_ids = {str(pack["pack_id"]) for pack in ordered_packs[:completed_prefix]}
    found = sum(
        row["stratum"] == stratum
        and str(row.get("pack_id")) in prefix_ids
        and row.get("initial_status") == "valid_incorrect"
        for row in attempts
    )
    workers = int(config["acquisition"]["workers_per_workspace"])
    next_slice = ordered_packs[completed_prefix : completed_prefix + workers]
    pending = [pack for pack in next_slice if str(pack["pack_id"]) not in completed_ids]
    state = {
        "stratum": stratum,
        "target": target,
        "valid_failures_found": found,
        "completed_packs": len(completed_ids),
        "completed_contiguous_prefix_packs": completed_prefix,
        "total_packs": len(manifests[stratum]),
        "source_exhausted": completed_prefix == len(ordered_packs),
        "quota_reached": found >= target,
    }
    if found >= target or state["source_exhausted"]:
        return [], state
    if not pending:
        raise RuntimeError("source-order gap could not be scheduled")
    return pending[:workers], state


def freeze_model_cohort(
    config: Mapping[str, Any], attempts: Sequence[Mapping[str, Any]],
    *, runs_root: str | Path, run_id: str, model_key: str,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    root = acquisition_root(runs_root, run_id)
    cohort, summary = select_frozen_cohort(attempts, config["acquisition"]["target_by_stratum"])
    if any(str(row["model_key"]) != model_key for row in cohort):
        raise RuntimeError("cross-model row entered cohort freeze")
    cohort_path = root / "cohorts" / f"{model_key}.jsonl"
    marker_path = root / "cohorts" / f"{model_key}.immutable.json"
    payload_hash = stable_hash(cohort)
    if marker_path.is_file():
        marker = json.loads(marker_path.read_text(encoding="utf-8"))
        if not cohort_path.is_file() or marker.get("cohort_file_sha256") != file_sha256(cohort_path):
            raise RuntimeError("immutable cohort file is missing or corrupted")
        if marker.get("cohort_sha256") != payload_hash:
            raise RuntimeError("immutable cohort would change on resume")
        return read_jsonl(cohort_path), marker["summary"]
    atomic_jsonl(cohort_path, cohort)
    marker = {
        "status": "IMMUTABLE_COHORT_FROZEN_BEFORE_REGENERATION",
        "model_key": model_key,
        "configuration_hash": stable_hash(config),
        "cohort_sha256": payload_hash,
        "cohort_file_sha256": file_sha256(cohort_path),
        "summary": summary,
    }
    atomic_json(marker_path, marker)
    return cohort, summary


def write_regeneration_pack_manifest(
    config: Mapping[str, Any], cohort: Sequence[Mapping[str, Any]],
    *, runs_root: str | Path, run_id: str, model_key: str,
) -> list[dict[str, Any]]:
    root = acquisition_root(runs_root, run_id)
    packs = [
        pack.to_dict()
        for pack in build_regeneration_packs(
            cohort,
            model_key=model_key,
            pack_size=int(config["acquisition"]["regeneration_trace_pack_size"]),
            configuration_hash=stable_hash(config),
        )
    ]
    _write_immutable_jsonl(
        root / "execution_packs/regenerations" / f"{model_key}.jsonl", packs
    )
    return packs


def collect_regeneration_rows(
    *, runs_root: str | Path, run_id: str, model_key: str,
    packs: Sequence[Mapping[str, Any]],
) -> tuple[list[dict[str, Any]], list[str]]:
    root = acquisition_root(runs_root, run_id)
    rows: list[dict[str, Any]] = []
    complete: list[str] = []
    for pack in packs:
        pack_id = str(pack["pack_id"])
        values = _read_complete_pack(
            root / "regenerations/raw" / model_key / pack_id,
            filename="regenerations.jsonl", pack_id=pack_id,
            expected_rows=4 * len(pack["trace_ids"]),
        )
        if values:
            rows.extend(values)
            complete.append(pack_id)
    identities = [str(row["rollout_key"]) for row in rows]
    if len(identities) != len(set(identities)):
        raise RuntimeError(f"duplicate regeneration rows for {model_key}")
    return rows, complete


def _write_model_report(
    path: Path, *, model_key: str, cohort_summary: Mapping[str, Any],
    acquisition: Mapping[str, Any], regeneration: Mapping[str, Any],
    integrity: Mapping[str, Any], compute: Mapping[str, Any],
) -> None:
    overall = regeneration["metrics"]["overall"]
    lines = [
        f"# Native failed-trace acquisition: {model_key}", "",
        f"Integrity: **{integrity['status']}**", "",
        f"- Frozen failures: `{cohort_summary['selected_total']}`",
        f"- Composition: `{json.dumps(cohort_summary['selected_by_stratum'], sort_keys=True)}`",
        f"- Source-exhaustion shortfall: `{cohort_summary['source_exhaustion_shortfall']}`",
        f"- FR@1: `{overall['fr_at_1']}`",
        f"- FR@4: `{overall['fr_at_4']}`",
        f"- PF@4: `{overall['pf_at_4']}`", "",
        f"- Generated tokens: `{compute['generated_tokens']}`",
        f"- Summed pack GPU wall seconds: `{compute['summed_pack_wall_seconds']:.3f}`",
        f"- Decode useful tokens/second over summed pack decode time: `{compute['decode_useful_tokens_per_second']:.3f}`",
        f"- Peak reserved GPU bytes: `{compute['peak_reserved_bytes']}`", "",
        "The cohort was frozen from initial attempts before regeneration artifacts were read. "
        "No semantic segmentation, checkpointing, hidden-state extraction, KV persistence, "
        "boundary inference, or suffix repair was run.", "",
        "## Acquisition by stratum", "",
        "```json", json.dumps(acquisition, indent=2, sort_keys=True), "```", "",
        "## Four-regeneration metrics", "", "```json",
        json.dumps(regeneration["metrics"], indent=2, sort_keys=True), "```", "",
    ]
    atomic_text(path, "\n".join(lines))


def _compute_accounting(
    root: Path, *, model_key: str,
    manifests: Mapping[str, Sequence[Mapping[str, Any]]],
    regeneration_packs: Sequence[Mapping[str, Any]],
    attempts: Sequence[Mapping[str, Any]], rollouts: Sequence[Mapping[str, Any]],
    workspace_started_unix: float | None = None,
) -> dict[str, Any]:
    markers: list[dict[str, Any]] = []
    for stratum, packs in manifests.items():
        for pack in packs:
            path = root / "attempts/raw" / model_key / stratum / str(pack["pack_id"]) / "complete.json"
            if path.is_file():
                markers.append(json.loads(path.read_text(encoding="utf-8")))
    for pack in regeneration_packs:
        path = root / "regenerations/raw" / model_key / str(pack["pack_id"]) / "complete.json"
        if path.is_file():
            markers.append(json.loads(path.read_text(encoding="utf-8")))
    useful = sum(int(row.get("decode_metrics", {}).get("useful_output_tokens", 0)) for row in markers)
    decode_seconds = sum(float(row.get("decode_metrics", {}).get("decode_wall_seconds", 0.0)) for row in markers)
    return {
        "completed_pack_count": len(markers),
        "generated_tokens": sum(int(row.get("generated_token_count", 0)) for row in [*attempts, *rollouts]),
        "summed_pack_wall_seconds": sum(float(row.get("wall_seconds", 0.0)) for row in markers),
        "summed_prefill_seconds": sum(float(row.get("prefill_seconds", 0.0)) for row in markers),
        "summed_decode_seconds": decode_seconds,
        "decode_useful_output_tokens": useful,
        "decode_useful_tokens_per_second": useful / max(decode_seconds, 1e-12),
        "peak_allocated_bytes": max(
            [int(row.get("decode_metrics", {}).get("maximum_allocated_bytes", 0)) for row in markers] or [0]
        ),
        "peak_reserved_bytes": max(
            [int(row.get("decode_metrics", {}).get("maximum_reserved_bytes", 0)) for row in markers] or [0]
        ),
        "infrastructure_retries": "recorded_in_modal_event_history",
        "workspace_wall_seconds_to_finalization": (
            None if workspace_started_unix is None else max(0.0, time.time() - workspace_started_unix)
        ),
    }


def finalize_model(
    config: Mapping[str, Any], source_rows: Sequence[Mapping[str, Any]],
    *, runs_root: str | Path, run_id: str, model_key: str,
    manifests: Mapping[str, Sequence[Mapping[str, Any]]],
    regeneration_packs: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    root = acquisition_root(runs_root, run_id)
    attempts, _completed = collect_attempt_rows(
        runs_root=runs_root, run_id=run_id, model_key=model_key, manifests=manifests,
    )
    cohort_path = root / "cohorts" / f"{model_key}.jsonl"
    marker_path = root / "cohorts" / f"{model_key}.immutable.json"
    if not marker_path.is_file():
        raise RuntimeError("cohort must be frozen before finalization")
    cohort = read_jsonl(cohort_path)
    cohort_summary = json.loads(marker_path.read_text(encoding="utf-8"))["summary"]
    rollouts, complete = collect_regeneration_rows(
        runs_root=runs_root, run_id=run_id, model_key=model_key, packs=regeneration_packs,
    )
    if len(complete) != len(regeneration_packs):
        raise RuntimeError("cannot finalize with incomplete regeneration packs")
    tagged_attempts = attach_cohort_status(attempts, cohort)
    integrity = validate_final_integrity(
        config=config, source_rows=source_rows, attempts=tagged_attempts,
        cohort=cohort, rollouts=rollouts,
    )
    acquisition = acquisition_metrics(tagged_attempts, cohort)
    regeneration = regeneration_metrics(cohort, rollouts, config["statistics"])
    state_path = Path(runs_root) / run_id / "remote_run_state" / f"{model_key}.json"
    state = json.loads(state_path.read_text(encoding="utf-8")) if state_path.is_file() else {}
    compute = _compute_accounting(
        root, model_key=model_key, manifests=manifests,
        regeneration_packs=regeneration_packs, attempts=tagged_attempts,
        rollouts=rollouts,
        workspace_started_unix=(
            float(state["started_unix"]) if state.get("started_unix") is not None else None
        ),
    )
    model_root = root / "final" / model_key
    atomic_jsonl(model_root / "attempted_problems.jsonl", tagged_attempts)
    atomic_jsonl(model_root / "raw_full_regeneration_outcomes.jsonl", rollouts)
    atomic_jsonl(model_root / "trace_success_counts.jsonl", regeneration["trace_records"])
    atomic_json(model_root / "acquisition_metrics.json", acquisition)
    atomic_json(model_root / "full_regeneration_metrics.json", regeneration["metrics"])
    atomic_json(model_root / "integrity_report.json", integrity)
    atomic_json(model_root / "compute_report.json", compute)
    summary = {
        "status": "COMPLETE" if integrity["status"] == "PASS" else "FAILED",
        "model_key": model_key,
        "cohort": cohort_summary,
        "acquisition": acquisition,
        "regeneration_metrics": regeneration["metrics"],
        "integrity": integrity,
        "compute": compute,
    }
    atomic_json(model_root / "summary.json", summary)
    _write_model_report(
        model_root / "FINAL_REPORT.md", model_key=model_key,
        cohort_summary=cohort_summary, acquisition=acquisition,
        regeneration=regeneration, integrity=integrity, compute=compute,
    )
    return summary


def finalize_partial_model(
    config: Mapping[str, Any], source_rows: Sequence[Mapping[str, Any]],
    *, runs_root: str | Path, run_id: str, model_key: str,
    manifests: Mapping[str, Sequence[Mapping[str, Any]]],
    regeneration_packs: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    """Aggregate only complete immutable packs after an explicit user stop.

    This function writes to ``final_partial`` and never changes the immutable
    600-trace cohort or writes a normal ``COMPLETE`` summary.  It is therefore
    safe to retain for descriptive analysis while keeping the original full-run
    completion criterion visibly failed.
    """

    root = acquisition_root(runs_root, run_id)
    attempts, _completed = collect_attempt_rows(
        runs_root=runs_root, run_id=run_id, model_key=model_key, manifests=manifests,
    )
    cohort_path = root / "cohorts" / f"{model_key}.jsonl"
    marker_path = root / "cohorts" / f"{model_key}.immutable.json"
    if not marker_path.is_file():
        raise RuntimeError("cohort must be frozen before partial finalization")
    frozen_cohort = read_jsonl(cohort_path)
    frozen_summary = json.loads(marker_path.read_text(encoding="utf-8"))["summary"]
    rollouts, complete_pack_ids = collect_regeneration_rows(
        runs_root=runs_root, run_id=run_id, model_key=model_key, packs=regeneration_packs,
    )
    if not complete_pack_ids:
        raise RuntimeError("cannot partially finalize with zero completed regeneration packs")
    if len(complete_pack_ids) == len(regeneration_packs):
        raise RuntimeError("all packs are complete; use the ordinary finalizer")

    complete_set = set(complete_pack_ids)
    completed_trace_ids = [
        str(trace_id)
        for pack in regeneration_packs
        if str(pack["pack_id"]) in complete_set
        for trace_id in pack["trace_ids"]
    ]
    completed_set = set(completed_trace_ids)
    completed_cohort = [
        row for row in frozen_cohort if str(row["trace_id"]) in completed_set
    ]
    if [str(row["trace_id"]) for row in completed_cohort] != completed_trace_ids:
        raise RuntimeError("completed packs do not preserve frozen cohort membership order")

    tagged_attempts = attach_cohort_status(attempts, frozen_cohort)
    integrity = validate_partial_regeneration_integrity(
        config=config,
        source_rows=source_rows,
        attempts=tagged_attempts,
        frozen_cohort=frozen_cohort,
        completed_cohort=completed_cohort,
        rollouts=rollouts,
        expected_pack_count=len(regeneration_packs),
        completed_pack_count=len(complete_pack_ids),
    )
    acquisition = acquisition_metrics(tagged_attempts, frozen_cohort)
    regeneration = regeneration_metrics(completed_cohort, rollouts, config["statistics"])
    state_path = Path(runs_root) / run_id / "remote_run_state" / f"{model_key}.json"
    state = json.loads(state_path.read_text(encoding="utf-8")) if state_path.is_file() else {}
    compute = _compute_accounting(
        root, model_key=model_key, manifests=manifests,
        regeneration_packs=[
            pack for pack in regeneration_packs if str(pack["pack_id"]) in complete_set
        ],
        attempts=tagged_attempts, rollouts=rollouts,
        workspace_started_unix=(
            float(state["started_unix"]) if state.get("started_unix") is not None else None
        ),
    )
    partial_by_stratum = Counter(str(row["stratum"]) for row in completed_cohort)
    partial_summary = {
        "selected_total": len(completed_cohort),
        "selected_by_stratum": {stratum: partial_by_stratum[stratum] for stratum in STRATA},
        "source_exhaustion_shortfall": frozen_summary["source_exhaustion_shortfall"],
        "full_frozen_cohort_total": len(frozen_cohort),
        "missing_frozen_cohort_rows": len(frozen_cohort) - len(completed_cohort),
        "selection_basis": "complete immutable execution packs available at user-directed stop",
    }
    model_root = root / "final_partial" / model_key
    atomic_jsonl(model_root / "attempted_problems.jsonl", tagged_attempts)
    atomic_jsonl(model_root / "completed_pack_cohort.jsonl", completed_cohort)
    atomic_jsonl(model_root / "raw_full_regeneration_outcomes.jsonl", rollouts)
    atomic_jsonl(model_root / "trace_success_counts.jsonl", regeneration["trace_records"])
    atomic_json(model_root / "acquisition_metrics.json", acquisition)
    atomic_json(model_root / "full_regeneration_metrics_partial.json", regeneration["metrics"])
    atomic_json(model_root / "integrity_report.json", integrity)
    atomic_json(model_root / "compute_report.json", compute)
    atomic_json(model_root / "missing_pack_ids.json", {
        "missing_pack_ids": [
            str(pack["pack_id"]) for pack in regeneration_packs
            if str(pack["pack_id"]) not in complete_set
        ]
    })
    summary = {
        "status": "PARTIAL_COMPLETE_USER_DIRECTED",
        "model_key": model_key,
        "cohort": frozen_summary,
        "completed_pack_cohort": partial_summary,
        "acquisition": acquisition,
        "regeneration_metrics_partial": regeneration["metrics"],
        "integrity": integrity,
        "compute": compute,
        "scientific_caveat": (
            "Metrics describe only traces in completed immutable execution packs. "
            "Pack completion can depend on runtime and output length, so these metrics "
            "must not be substituted for the preregistered full-cohort result."
        ),
    }
    atomic_json(model_root / "summary.json", summary)
    overall = regeneration["metrics"]["overall"]
    atomic_text(model_root / "FINAL_PARTIAL_REPORT.md", "\n".join([
        f"# Native failed-trace partial aggregation: {model_key}", "",
        "Status: **PARTIAL_PASS for completed immutable packs only**", "",
        f"- Original frozen cohort: `{len(frozen_cohort)}` traces",
        f"- Completed trace packs represented: `{len(completed_cohort)}` traces",
        f"- Completed packs: `{len(complete_pack_ids)} / {len(regeneration_packs)}`",
        f"- Complete rollout rows: `{len(rollouts)}`",
        f"- FR@1 on completed packs: `{overall['fr_at_1']}`",
        f"- FR@4 on completed packs: `{overall['fr_at_4']}`",
        f"- PF@4 on completed packs: `{overall['pf_at_4']}`", "",
        "Every included trace has exactly four valid identity-seeded regeneration records. ",
        "The original cohort and pack manifest were not modified. The missing packs remain ",
        "explicitly incomplete. Because pack completion may correlate with runtime or output ",
        "length, these descriptive metrics are not a replacement for the full 600-trace result.", "",
        "No semantic segmentation, hidden-state or KV extraction, boundary-model work, or suffix repair ran.", "",
    ]))
    return summary


def merge_workspace_summaries(
    model_summaries: Sequence[Mapping[str, Any]], *, output_dir: str | Path,
) -> dict[str, Any]:
    """Create the global report after one completed model is downloaded per workspace."""

    output = Path(output_dir)
    expected = {"family_a_small", "family_a_large", "family_b_small", "family_b_large"}
    by_model = {str(row["model_key"]): dict(row) for row in model_summaries}
    if set(by_model) != expected:
        raise RuntimeError(f"global merge requires all four model summaries; got {sorted(by_model)}")
    if any(row.get("status") != "COMPLETE" for row in by_model.values()):
        raise RuntimeError("global merge refuses an incomplete model run")
    success_counts_by_stratum: dict[str, list[int]] = {
        label: [] for label in ("overall", "gsm1k", "math_level_3", "math_level_4")
    }
    total_cohort = 0
    for row in by_model.values():
        total_cohort += int(row["cohort"]["selected_total"])
        for label in success_counts_by_stratum:
            histogram = row["regeneration_metrics"][label]["success_count_counts"]
            for value in range(5):
                success_counts_by_stratum[label].extend(
                    [value] * int(histogram[str(value)])
                )
    pooled_metrics = {
        label: bootstrap_trace_metrics(
            counts, replicates=10000, seed=2701, confidence_level=0.95,
        )
        for label, counts in success_counts_by_stratum.items()
    }
    payload = {
        "status": "COMPLETE",
        "model_count": 4,
        "total_frozen_failed_traces": total_cohort,
        "total_full_regenerations": 4 * total_cohort,
        "pooled_trace_level_metrics": pooled_metrics,
        "models": by_model,
        "boundary_training_run": False,
        "native_checkpoint_or_hidden_state_work_run": False,
    }
    atomic_json(output / "global_summary.json", payload)
    atomic_text(
        output / "GLOBAL_FINAL_REPORT.md",
        "# SafePrefix native failure acquisition and full regeneration\n\n"
        f"Status: **COMPLETE**\n\n- Models: `4`\n- Frozen failures: `{total_cohort}`\n"
        f"- Fresh prompt-root regenerations: `{4 * total_cohort}`\n\n"
        "## Pooled trace-level metrics\n\n```json\n"
        + json.dumps(pooled_metrics, indent=2, sort_keys=True)
        + "\n```\n\nNo boundary inference, semantic segmentation, hidden-state extraction, KV rewind, or suffix repair was run.\n",
    )
    return payload
