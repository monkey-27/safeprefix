"""Final artifact aggregation and immutable go/no-go gate evaluation."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Mapping

from safeprefix.reproducibility import atomic_json, atomic_text


TERMINAL_RUN_STATUSES = {"PASS", "FAIL", "INFRASTRUCTURE_FAILURE", "STOPPED_AT_GATE", "NOT_RUN"}


def assert_final_run_state(remote_state: Mapping[str, Any]) -> None:
    """Refuse to write a final Markdown report from an in-flight run state."""
    models = remote_state.get("models", {})
    if not isinstance(models, Mapping):
        raise ValueError("remote run state models must be a mapping")
    unfinished = {
        str(name): str(state.get("status"))
        for name, state in models.items()
        if not isinstance(state, Mapping) or state.get("status") not in TERMINAL_RUN_STATUSES
    }
    if unfinished:
        raise RuntimeError(f"final report requested before terminal run status: {unfinished}")


def evaluate_gates(summary: Mapping[str, Any]) -> dict[str, Any]:
    def value(path: str) -> Any:
        current: Any = summary
        for part in path.split("."):
            if not isinstance(current, Mapping) or part not in current:
                return None
            current = current[part]
        return current

    mock_only = bool(value("data_audit.mock"))
    adaptive_selected = value("rollout_count.adaptive_selected")
    adaptive_observed = True if adaptive_selected is False else value("rollout_count.adaptive_average")
    rules = [
        ("answer_parser_success", value("prompt.answer_parser_success"), lambda x: x >= 0.98),
        ("valid_segmentation", value("segmentation.valid_segmentation"), lambda x: x >= 0.95),
        ("cache_restore_correctness", value("cache.passed"), bool),
        ("rollout_k_le_6_noninferior", value("rollout_count.passed"), bool),
        ("adaptive_average_le_4_if_selected", adaptive_observed, lambda x: x is True or x <= 4.0),
        ("hidden_beats_text_and_latest", value("boundary.hidden_beats_baselines"), bool),
        ("native_transfer_three_of_four", value("native.models_working"), lambda x: x >= 3),
        ("accuracy_or_recomputation_gain", value("native.target_met"), bool),
    ]
    checks = []
    for name, observed, predicate in rules:
        if mock_only:
            checks.append({"name": name, "status": "NOT_RUN", "observed": None, "reason": "mock artifacts are plumbing evidence only"})
        elif observed is None:
            checks.append({"name": name, "status": "NOT_RUN", "observed": None})
        else:
            checks.append({"name": name, "status": "PASS" if predicate(observed) else "FAIL", "observed": observed})
    status = "GO" if checks and all(item["status"] == "PASS" for item in checks) else "NO_GO"
    return {"decision": status, "checks": checks}


def generate_report(summary: dict[str, Any], output_root: str | Path) -> dict[str, Any]:
    root = Path(output_root)
    final = root / "final"
    (final / "tables").mkdir(parents=True, exist_ok=True)
    (final / "figures").mkdir(parents=True, exist_ok=True)
    gates = evaluate_gates(summary)
    combined = {**summary, "go_no_go": gates}
    sections = [
        "data audit", "cache correctness", "prompt comparison", "segmentation quality",
        "teacher-forcing likelihood shift", "rollout-count analysis", "boundary-model comparisons",
        "native transfer", "one-shot repair", "budget-matched repair", "post-error checkpoint audit",
        "runtime and memory",
    ]
    lines = ["# SafePrefix pilot report", "", f"**Automatic decision: {gates['decision']}**", "", "No missing stage is interpreted as a positive result.", ""]
    for section in sections:
        key = section.replace("-", "_").replace(" ", "_")
        lines.extend([f"## {section.title()}", "", f"```json\n{_pretty(_compact(summary.get(key, {'status': 'NOT_RUN'})))}\n```", ""])
    lines.extend(["## Go/no-go checklist", ""])
    lines.extend(f"- {item['name']}: **{item['status']}** (observed: `{item['observed']}`)" for item in gates["checks"])
    atomic_text(final / "pilot_report.md", "\n".join(lines) + "\n")
    atomic_json(final / "summary.json", combined)
    return combined


def _pretty(value: Any) -> str:
    import json

    return json.dumps(value, indent=2, sort_keys=True, default=str)


def _compact(value: Any) -> Any:
    if isinstance(value, list):
        if len(value) > 20:
            return {"count": len(value), "first_five": [_compact(item) for item in value[:5]], "truncated_in_markdown_only": True}
        return [_compact(item) for item in value]
    if isinstance(value, Mapping):
        return {key: _compact(item) for key, item in value.items()}
    return value
