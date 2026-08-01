"""Model-native chat rendering with an explicit, auditable fallback."""

from __future__ import annotations

from typing import Any, Mapping


def render_chat_prompt(
    tokenizer: Any,
    user_text: str,
    *,
    system_text: str | None = None,
    override: str | None = None,
    template_kwargs: Mapping[str, Any] | None = None,
) -> str:
    messages = []
    if system_text:
        messages.append({"role": "system", "content": system_text})
    messages.append({"role": "user", "content": user_text})
    if override:
        roles = {"system": system_text or "", "user": user_text}
        return override.format(**roles)
    template = getattr(tokenizer, "chat_template", None)
    if template and hasattr(tokenizer, "apply_chat_template"):
        return tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
            **dict(template_kwargs or {}),
        )
    prefix = f"System: {system_text}\n\n" if system_text else ""
    return f"{prefix}User: {user_text}\n\nAssistant:"


def tokenize_chat_prefix(tokenizer: Any, rendered_prompt: str, device: Any = None) -> dict[str, Any]:
    encoded = tokenizer(rendered_prompt, return_tensors="pt", add_special_tokens=False)
    if device is not None:
        encoded = {key: value.to(device) for key, value in encoded.items()}
    return encoded
