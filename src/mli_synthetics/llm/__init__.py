"""LLM integration."""
from __future__ import annotations

import os
from typing import Any

from mli_synthetics.llm.knowledge import build_knowledge_context
from mli_synthetics.llm.ollama_client import OllamaClient


def _flag(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    return str(value).strip().lower() in {"1", "true", "yes"}


def _hf_enabled() -> bool:
    return _flag(os.environ.get("USE_HF_CLIENT", ""))


def get_llm_client(settings: Any | None = None):
    """Return the LLM client implied by the current configuration.

    Priority: Anthropic > Groq > HFClient (env flag) > OllamaClient (default).

    - `settings.use_anthropic=True` or `USE_ANTHROPIC=true` -> AnthropicClient
    - `settings.use_groq=True` or `USE_GROQ=true` -> GroqClient
    - `USE_HF_CLIENT=true` -> HFClient (HuggingFace, GPU)
    - otherwise -> OllamaClient
    """
    if settings is None:
        from mli_synthetics.settings import get_settings

        settings = get_settings()

    use_anthropic = _flag(getattr(settings, "use_anthropic", False)) or _flag(
        os.environ.get("USE_ANTHROPIC", "")
    )
    use_groq = _flag(getattr(settings, "use_groq", False)) or _flag(
        os.environ.get("USE_GROQ", "")
    )

    if use_anthropic:
        from mli_synthetics.llm.anthropic_client import AnthropicClient

        return AnthropicClient(settings)
    if use_groq:
        from mli_synthetics.llm.groq_client import GroqClient

        return GroqClient(settings)
    if _hf_enabled():
        from mli_synthetics.llm.hf_client import HFClient

        return HFClient(timeout=settings.ollama_timeout_seconds)
    return OllamaClient(
        base_url=settings.ollama_base_url,
        timeout=settings.ollama_timeout_seconds,
    )


# Backward-compat alias for any existing imports.
get_default_client = get_llm_client


__all__ = [
    "OllamaClient",
    "build_knowledge_context",
    "get_llm_client",
    "get_default_client",
]
