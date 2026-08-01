"""Assertions and immutable signatures for the SafePrefix production backend."""

from __future__ import annotations

import hashlib
import importlib.metadata
from pathlib import Path
from typing import Any, Mapping

import torch


PRODUCTION_DTYPE = "bf16"
PRODUCTION_ATTENTION = "sdpa"
GENERATION_IMPLEMENTATION = "complete_cache_saved_logits_branch_stream_v3"


def _version(name: str) -> str:
    try:
        return importlib.metadata.version(name)
    except importlib.metadata.PackageNotFoundError:
        return "unavailable"


def generation_implementation_revision() -> str:
    path = Path(__file__).with_name("generation.py")
    return hashlib.sha256(path.read_bytes()).hexdigest()


def configured_production(entry: Mapping[str, Any]) -> Mapping[str, Any]:
    value = entry.get("production")
    if not isinstance(value, Mapping) or not value.get("enabled", False):
        raise ValueError("model configuration is not enabled for SafePrefix production")
    if str(entry.get("dtype", "")).casefold() not in {"bf16", "bfloat16"}:
        raise ValueError("SafePrefix production dtype must be BF16")
    if str(entry.get("attn_implementation", "")).casefold() != PRODUCTION_ATTENTION:
        raise ValueError("SafePrefix production attention backend must be SDPA")
    if str(value.get("dtype", "")).casefold() not in {"bf16", "bfloat16"}:
        raise ValueError("production.dtype must be BF16")
    if str(value.get("attn_implementation", "")).casefold() != PRODUCTION_ATTENTION:
        raise ValueError("production.attn_implementation must be SDPA")
    if bool(value.get("torch_compile", False)):
        raise ValueError("torch.compile is not validated for SafePrefix production")
    if int(value.get("tensor_parallel_degree", 0)) < 1:
        raise ValueError("production.tensor_parallel_degree must be positive")
    if int(value.get("branch_batch_size", 0)) < 1:
        raise ValueError("production.branch_batch_size must be positive")
    return value


def resolved_attention_backend(model: Any) -> str | None:
    config = model.config
    for name in ("_attn_implementation", "_attn_implementation_internal", "attn_implementation"):
        value = getattr(config, name, None)
        if value:
            return str(value).casefold()
    return None


def _parameter_dtypes(model: Any) -> list[str]:
    return sorted({str(parameter.dtype).removeprefix("torch.") for parameter in model.parameters() if parameter.is_floating_point()})


def _device_map(model: Any) -> dict[str, str]:
    mapping = getattr(model, "hf_device_map", None)
    if isinstance(mapping, Mapping):
        return {str(key): str(value) for key, value in mapping.items()}
    try:
        return {"": str(next(model.parameters()).device)}
    except StopIteration:
        return {"": "unknown"}


def _runtime_cuda_devices() -> list[dict[str, Any]]:
    if not torch.cuda.is_available():
        return []
    return [
        {
            "index": index,
            "name": torch.cuda.get_device_name(index),
            "capability": list(torch.cuda.get_device_capability(index)),
            "total_memory_bytes": int(torch.cuda.get_device_properties(index).total_memory),
        }
        for index in range(torch.cuda.device_count())
    ]


def assert_production_backend(
    model: Any,
    tokenizer: Any,
    entry: Mapping[str, Any],
    *,
    model_key: str,
) -> dict[str, Any]:
    """Fail closed unless the loaded deployment is the preregistered backend."""
    production = configured_production(entry)
    resolved_attention = resolved_attention_backend(model)
    if resolved_attention != PRODUCTION_ATTENTION:
        raise RuntimeError(
            f"requested SDPA but loaded model resolved attention backend is {resolved_attention!r}"
        )
    if model.training:
        raise RuntimeError("production model must be in eval mode")
    if hasattr(model, "_orig_mod"):
        raise RuntimeError("torch.compile-wrapped models are not validated")
    training_dropouts = [name for name, module in model.named_modules() if isinstance(module, torch.nn.Dropout) and module.training]
    if training_dropouts:
        raise RuntimeError(f"dropout remains active: {training_dropouts[:5]}")
    dtypes = _parameter_dtypes(model)
    if dtypes != ["bfloat16"]:
        raise RuntimeError(f"loaded floating parameter dtypes are not exclusively BF16: {dtypes}")
    tensor_parallel_degree = int(production["tensor_parallel_degree"])
    if tensor_parallel_degree != 1:
        raise RuntimeError(
            "the current Transformers backend implements only tensor_parallel_degree=1; "
            "a multi-GPU deployment needs its own validated loader"
        )
    model_revision = getattr(model.config, "_commit_hash", None) or entry.get("revision")
    tokenizer_revision = getattr(tokenizer, "init_kwargs", {}).get("_commit_hash") or entry.get("tokenizer_revision")
    if not model_revision or not tokenizer_revision or str(model_revision) == "main" or str(tokenizer_revision) == "main":
        raise RuntimeError("runtime model and tokenizer revisions must resolve to concrete values")
    parameter_count = sum(parameter.numel() for parameter in model.parameters())
    return {
        "model_key": model_key,
        "hf_model_id": str(entry["hf_model_id"]),
        "model_revision": str(model_revision),
        "tokenizer_id": str(entry.get("tokenizer_id") or entry["hf_model_id"]),
        "tokenizer_revision": str(tokenizer_revision),
        "parameter_count": int(parameter_count),
        "requested_attention_backend": PRODUCTION_ATTENTION,
        "resolved_attention_backend": resolved_attention,
        "dtype": PRODUCTION_DTYPE,
        "parameter_dtypes": dtypes,
        "device_map": _device_map(model),
        "tensor_parallel_degree": tensor_parallel_degree,
        "gpu_type": str(production.get("gpu_type", "unspecified")),
        "gpu_count": int(production.get("gpu_count", 1)),
        "branch_batch_size": int(production["branch_batch_size"]),
        "cache_topology": str(production.get("cache_topology", "unspecified")),
        "torch_compile": False,
        "generation_implementation": GENERATION_IMPLEMENTATION,
        "generation_implementation_revision": generation_implementation_revision(),
        "transformers_version": _version("transformers"),
        "pytorch_version": torch.__version__,
        "cuda_version": torch.version.cuda,
        "cuda_devices": _runtime_cuda_devices(),
        "flash_attn_version": _version("flash-attn"),
        "accelerate_version": _version("accelerate"),
        "model_eval": True,
        "dropout_disabled": True,
    }


def validate_batching_fairness(
    method_batch_sizes: Mapping[str, int],
    *,
    validated_invariant_sizes: set[int],
    fixed_padded_batch_size: int | None = None,
) -> dict[str, Any]:
    """Enforce the preregistered no-method-specific-batch-shape rule."""
    sizes = {str(name): int(value) for name, value in method_batch_sizes.items()}
    if any(value < 1 for value in sizes.values()):
        raise ValueError("method batch sizes must be positive")
    natural_sizes = set(sizes.values())
    if natural_sizes.issubset(validated_invariant_sizes):
        return {
            "passed": True,
            "policy": "validated_shape_invariance",
            "method_batch_sizes": sizes,
            "validated_invariant_sizes": sorted(validated_invariant_sizes),
        }
    if fixed_padded_batch_size is not None and set(sizes.values()) == {int(fixed_padded_batch_size)}:
        return {
            "passed": True,
            "policy": "fixed_padded_batch",
            "method_batch_sizes": sizes,
            "fixed_padded_batch_size": int(fixed_padded_batch_size),
        }
    missing = sorted(natural_sizes - validated_invariant_sizes)
    raise RuntimeError(
        "batching fairness is unvalidated for method batch sizes "
        f"{missing}; use one validated fixed padded batch shape"
    )
