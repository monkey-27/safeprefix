"""Config-driven Hugging Face causal-model loading."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping


@dataclass(frozen=True)
class LoadedModel:
    model: Any
    tokenizer: Any
    model_id: str
    tokenizer_id: str
    model_revision: str | None
    tokenizer_revision: str | None
    production_signature: Mapping[str, Any] | None = None


@dataclass(frozen=True)
class LoadedTokenizer:
    tokenizer: Any
    tokenizer_id: str
    tokenizer_revision: str | None


def _dtype(name: str) -> Any:
    import torch

    aliases = {
        "bf16": torch.bfloat16,
        "bfloat16": torch.bfloat16,
        "fp16": torch.float16,
        "float16": torch.float16,
        "fp32": torch.float32,
        "float32": torch.float32,
    }
    if name.casefold() not in aliases:
        raise ValueError(f"unsupported dtype: {name}")
    return aliases[name.casefold()]


def load_tokenizer(entry: Mapping[str, Any]) -> LoadedTokenizer:
    """Load only the configured tokenizer for metadata-only preparation."""
    from transformers import AutoTokenizer

    model_id = str(entry["hf_model_id"])
    tokenizer_id = str(entry.get("tokenizer_id") or model_id)
    revision = entry.get("revision")
    tokenizer_revision = entry.get("tokenizer_revision") or revision
    tokenizer = AutoTokenizer.from_pretrained(
        tokenizer_id,
        revision=tokenizer_revision,
        trust_remote_code=bool(entry.get("trust_remote_code", False)),
        use_fast=True,
    )
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    return LoadedTokenizer(
        tokenizer=tokenizer,
        tokenizer_id=tokenizer_id,
        tokenizer_revision=tokenizer.init_kwargs.get("_commit_hash") or tokenizer_revision,
    )


def load_model(entry: Mapping[str, Any]) -> LoadedModel:
    from transformers import AutoModelForCausalLM

    model_id = str(entry["hf_model_id"])
    revision = entry.get("revision")
    loaded_tokenizer = load_tokenizer(entry)
    tokenizer = loaded_tokenizer.tokenizer
    tokenizer_id = loaded_tokenizer.tokenizer_id
    tokenizer_revision = loaded_tokenizer.tokenizer_revision
    devices = entry.get("device", "auto")
    if isinstance(devices, list):
        device_map: Any = "balanced" if len(devices) > 1 else {"": int(devices[0])}
    elif devices in {"auto", "balanced", "balanced_low_0", "sequential"}:
        device_map = devices
    elif str(devices).startswith("cuda"):
        device_map = {"": str(devices)}
    else:
        device_map = None
    kwargs: dict[str, Any] = {
        "revision": revision,
        "trust_remote_code": bool(entry.get("trust_remote_code", False)),
        "torch_dtype": _dtype(str(entry.get("dtype", "bf16"))),
        "attn_implementation": entry.get("attn_implementation", "sdpa"),
        "low_cpu_mem_usage": True,
    }
    if device_map is not None:
        kwargs["device_map"] = device_map
    model = AutoModelForCausalLM.from_pretrained(model_id, **kwargs)
    if device_map is None and str(devices) != "cpu":
        model.to(str(devices))
    model.eval()
    production_signature = None
    if isinstance(entry.get("production"), Mapping) and entry["production"].get("enabled", False):
        from .production_backend import assert_production_backend

        production_signature = assert_production_backend(
            model,
            tokenizer,
            entry,
            model_key=str(entry.get("model_key", model_id)),
        )
    return LoadedModel(
        model=model,
        tokenizer=tokenizer,
        model_id=model_id,
        tokenizer_id=tokenizer_id,
        model_revision=getattr(model.config, "_commit_hash", None) or revision,
        tokenizer_revision=tokenizer_revision,
        production_signature=production_signature,
    )


def primary_device(model: Any) -> Any:
    try:
        return next(model.parameters()).device
    except StopIteration:
        import torch

        return torch.device("cpu")
