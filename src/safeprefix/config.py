"""Validated YAML configuration with deterministic include and override handling."""

from __future__ import annotations

import copy
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping

import yaml


class ConfigError(ValueError):
    """Raised when an experiment configuration is incomplete or inconsistent."""


REPO_ROOT = Path(__file__).resolve().parents[2]


def _merge(left: Mapping[str, Any], right: Mapping[str, Any]) -> dict[str, Any]:
    result = copy.deepcopy(dict(left))
    for key, value in right.items():
        if isinstance(value, Mapping) and isinstance(result.get(key), Mapping):
            result[key] = _merge(result[key], value)
        else:
            result[key] = copy.deepcopy(value)
    return result


def _read_yaml(path: Path) -> dict[str, Any]:
    value = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ConfigError(f"configuration must be a mapping: {path}")
    return value


def _read_yaml_tree(path: Path, stack: tuple[Path, ...] = ()) -> dict[str, Any]:
    """Resolve nested includes so full configs can override the pilot safely."""
    resolved = path.resolve()
    if resolved in stack:
        chain = " -> ".join(str(item) for item in (*stack, resolved))
        raise ConfigError(f"cyclic configuration include: {chain}")
    root = _read_yaml(resolved)
    includes = root.pop("include", [])
    if isinstance(includes, str):
        includes = [includes]
    merged: dict[str, Any] = {}
    for item in includes:
        candidate = Path(item)
        if not candidate.is_absolute():
            candidate = (resolved.parent / candidate).resolve()
        merged = _merge(merged, _read_yaml_tree(candidate, (*stack, resolved)))
    return _merge(merged, root)


def _set_dotted(config: dict[str, Any], dotted: str, value: Any) -> None:
    target = config
    parts = dotted.split(".")
    for part in parts[:-1]:
        existing = target.setdefault(part, {})
        if not isinstance(existing, dict):
            raise ConfigError(f"override traverses a scalar: {dotted}")
        target = existing
    target[parts[-1]] = value


@dataclass(frozen=True)
class ResolvedConfig:
    data: dict[str, Any]
    source: Path

    @property
    def digest(self) -> str:
        from .reproducibility import stable_hash

        return stable_hash(self.data)


def validate_config(config: Mapping[str, Any]) -> None:
    required = {"experiment", "seed", "artifacts_root"}
    missing = required - set(config)
    if missing:
        raise ConfigError(f"missing top-level configuration keys: {sorted(missing)}")
    if not isinstance(config["seed"], int):
        raise ConfigError("seed must be an integer")
    experiment = config["experiment"]
    if not isinstance(experiment, Mapping) or not experiment.get("id"):
        raise ConfigError("experiment.id is required")
    models = config.get("models", {})
    selected = config.get("selected_models", [])
    unknown = sorted(set(selected) - set(models))
    if unknown:
        raise ConfigError(f"unknown selected_models entries: {unknown}")
    for name in selected:
        entry = models[name]
        for field in ("hf_model_id", "tokenizer_id", "dtype", "max_context_length"):
            if field not in entry:
                raise ConfigError(f"models.{name}.{field} is required")
        production = entry.get("production")
        if production is not None:
            if str(entry.get("attn_implementation", "")).casefold() != "sdpa":
                raise ConfigError(f"models.{name} production attention must be sdpa")
            if str(entry.get("dtype", "")).casefold() not in {"bf16", "bfloat16"}:
                raise ConfigError(f"models.{name} production dtype must be bf16")
            if int(production.get("tensor_parallel_degree", 0)) < 1:
                raise ConfigError(f"models.{name}.production.tensor_parallel_degree must be positive")
            if bool(production.get("torch_compile", False)):
                raise ConfigError(f"models.{name} production may not use torch.compile")
    cache_v3 = config.get("production_cache_validation", {})
    if cache_v3:
        if str(cache_v3.get("attn_implementation", "")).casefold() != "sdpa":
            raise ConfigError("production_cache_validation.attn_implementation must be sdpa")
        if str(cache_v3.get("dtype", "")).casefold() not in {"bf16", "bfloat16"}:
            raise ConfigError("production_cache_validation.dtype must be bf16")
        sizes = [int(value) for value in cache_v3.get("branch_batch_sizes", [])]
        if sizes != sorted(set(sizes)) or not sizes or sizes[0] != 1:
            raise ConfigError("branch_batch_sizes must be sorted unique positive values beginning with 1")
    rollout = config.get("rollout", {})
    for key in ("train_per_checkpoint", "dev_per_checkpoint", "dense_per_checkpoint"):
        if key in rollout and int(rollout[key]) < 1:
            raise ConfigError(f"rollout.{key} must be positive")
    scalable = config.get("scalable")
    if scalable is not None:
        execution = config.get("production_execution", {})
        if execution.get("dtype") != "bf16" or execution.get("attention_backend") != "sdpa":
            raise ConfigError("scalable SafePrefix production requires BF16 SDPA")
        if execution.get("microbatch_policy") != "fixed_padded":
            raise ConfigError("scalable SafePrefix requires fixed_padded microbatch policy")
        if execution.get("cache_protocol") != "complete_cache_saved_next_logits_no_replay":
            raise ConfigError("scalable SafePrefix requires complete-cache no-replay continuation")
        counts = config.get("sample_counts", {})
        required_counts = {
            "max_train_failures_per_model",
            "max_dev_failures_per_model",
            "target_native_eval_failures_per_model",
            "dense_teacher_forced_failures_per_model",
            "dense_native_failures_per_model",
        }
        missing_counts = required_counts - set(counts)
        if missing_counts:
            raise ConfigError(f"missing scalable sample counts: {sorted(missing_counts)}")
        if any(int(counts[key]) < 1 for key in required_counts):
            raise ConfigError("all scalable sample counts must be positive")
        composition = config.get("teacher_forced", {}).get("failed_composition", {})
        train_total = sum(int(value) for value in composition.get("train", {}).values())
        dev_total = sum(int(value) for value in composition.get("dev", {}).values())
        if train_total != int(counts["max_train_failures_per_model"]):
            raise ConfigError("teacher_forced train composition must equal max_train_failures_per_model")
        if dev_total != int(counts["max_dev_failures_per_model"]):
            raise ConfigError("teacher_forced dev composition must equal max_dev_failures_per_model")
        allocation = config.get("rollout_allocation", {})
        if int(allocation.get("initial_rollouts_per_checkpoint", 0)) != 4:
            raise ConfigError("scalable nested rollout collection must begin with exactly four outcomes")
        if int(allocation.get("k6_total_rollouts_per_checkpoint", 0)) != 6:
            raise ConfigError("k6 allocation must contain six total outcomes")
        if int(allocation.get("dense_total_rollouts_per_checkpoint", 0)) < 12:
            raise ConfigError("dense teacher-forced audit requires at least 12 outcomes per checkpoint")
    selection = config.get("configuration_selection")
    if selection is not None:
        if len(selected) != 4:
            raise ConfigError("configuration-selection pilot requires the complete four-model matrix")
        execution = config.get("production_execution", {})
        if execution.get("dtype") != "bf16" or execution.get("attention_backend") != "sdpa":
            raise ConfigError("configuration selection requires BF16 SDPA")
        if execution.get("microbatch_policy") != "dynamic_unpadded":
            raise ConfigError("configuration selection requires dynamic_unpadded microbatching")
        if execution.get("cache_protocol") != "complete_cache_saved_next_logits_no_replay":
            raise ConfigError("configuration selection requires complete-cache no-replay continuation")
        guard = config.get("final_test_guard", {})
        if not bool(selection.get("final_test_locked", False)):
            raise ConfigError("configuration selection must lock the final test")
        if any(bool(guard.get(key, False)) for key in ("allow_prompt_loading", "allow_generation", "allow_verifier_outputs")):
            raise ConfigError("configuration-selection config may not unlock final-test access")
        split = config.get("immutable_splits", {})
        for role in ("teacher_forced_train", "teacher_forced_dev"):
            entry = split.get(role, {})
            if sum(int(value) for value in entry.get("composition", {}).values()) != int(entry.get("total", -1)):
                raise ConfigError(f"immutable_splits.{role} composition does not match total")
            floors = entry.get("minimum_failed_composition", {})
            if any(int(floors.get(bucket, 0)) > int(count) for bucket, count in entry.get("composition", {}).items()):
                raise ConfigError(f"immutable_splits.{role} failed floor exceeds composition")
            if any("gsm8k" in str(bucket).casefold() for bucket in entry.get("composition", {})):
                raise ConfigError("GSM8K is prohibited from teacher-forced train and development")
        phase_a = config.get("phase_a", {})
        if sum(int(value) for value in phase_a.get("composition", {}).values()) != int(phase_a.get("problems_per_model", -1)):
            raise ConfigError("Phase A composition must equal problems_per_model")
        phase_c = config.get("phase_c", {})
        if sum(int(value) for value in phase_c.get("composition", {}).values()) != int(phase_c.get("failed_training_traces_per_model", -1)):
            raise ConfigError("Phase C composition must equal failed_training_traces_per_model")
        train_floor = split.get("teacher_forced_train", {}).get("minimum_failed_composition", {})
        if any(int(train_floor.get(bucket, 0)) < int(count) for bucket, count in phase_c.get("composition", {}).items()):
            raise ConfigError("teacher-forced train failed floor must cover the Phase C cohort")
        if int(phase_c.get("rollout_outcomes_per_checkpoint", 0)) != 6:
            raise ConfigError("Phase C must collect one nested six-outcome stream")
        if list(map(int, phase_c.get("nested_k", []))) != [2, 4, 6]:
            raise ConfigError("Phase C nested rollout conditions must be [2, 4, 6]")
        phase_d = config.get("phase_d", {})
        if list(map(int, phase_d.get("sample_sizes", []))) != [250, 500, 750, 1000]:
            raise ConfigError("Phase D sample sizes must be [250, 500, 750, 1000]")
        if list(map(int, phase_d.get("optimization_seeds", []))) != list(map(int, config.get("boundary", {}).get("optimization_seeds", []))):
            raise ConfigError("Phase D and boundary optimization seeds must match")


def load_config(
    path: str | Path,
    overrides: Mapping[str, Any] | None = None,
) -> ResolvedConfig:
    source = Path(path).resolve()
    merged = _read_yaml_tree(source)
    for dotted, value in (overrides or {}).items():
        _set_dotted(merged, dotted, value)
    validate_config(merged)
    return ResolvedConfig(data=merged, source=source)


def parse_overrides(items: list[str]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for item in items:
        if "=" not in item:
            raise ConfigError(f"override must be key=value: {item}")
        key, raw = item.split("=", 1)
        result[key] = yaml.safe_load(raw)
    return result


def dump_resolved(config: Mapping[str, Any]) -> str:
    return yaml.safe_dump(json.loads(json.dumps(config)), sort_keys=True)
