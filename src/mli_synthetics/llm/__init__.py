"""LLM integration."""
from __future__ import annotations

import os
from typing import Any

from mli_synthetics.llm.knowledge import build_knowledge_context
from mli_synthetics.llm.ollama_client import OllamaClient


def _hf_enabled() -> bool:
    return os.environ.get("USE_HF_CLIENT", "").lower() in {"1", "true", "yes"}


def get_default_client(settings: Any | None = None):
    """Return the LLM client implied by the current environment.

    `USE_HF_CLIENT=true` -> HFClient (HuggingFace transformers, GPU-aware,
    Kaggle-friendly). Otherwise the default OllamaClient is returned.
    """
    if settings is None:
        from mli_synthetics.settings import get_settings

        settings = get_settings()
    if _hf_enabled():
        from mli_synthetics.llm.hf_client import HFClient

        return HFClient(timeout=settings.ollama_timeout_seconds)
    return OllamaClient(
        base_url=settings.ollama_base_url,
        timeout=settings.ollama_timeout_seconds,
    )


__all__ = [
    "OllamaClient",
    "build_knowledge_context",
    "get_default_client",
]
