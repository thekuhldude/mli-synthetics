from __future__ import annotations

"""llama-cpp-python based HFClient for Mistral-Nemo GGUF (class singleton).

Drop-in replacement for OllamaClient. Loads
`bartowski/Mistral-Nemo-Instruct-2407-GGUF` (Q4_K_M variant) onto the GPU
via llama-cpp-python with all layers offloaded and a 16K context window.

Class-level singleton: `_instance` and `_llm` live on the class object
itself, so every `HFClient()` call returns the same instance and the
GGUF is loaded into VRAM exactly once per process even when analyzer +
designer + warmup all construct a client.

Activate by setting `USE_HF_CLIENT=true`. The factory in
`mli_synthetics.llm.get_default_client` then returns an `HFClient`.
"""

import asyncio
import json
import re
from pathlib import Path
from typing import Any

from mli_synthetics.errors import (
    OllamaConnectionError,
    OllamaError,
    OllamaInvalidJSONError,
    OllamaModelNotFoundError,
)
from mli_synthetics.logging_config import get_logger

logger = get_logger()

DEFAULT_REPO = "bartowski/Mistral-Nemo-Instruct-2407-GGUF"
DEFAULT_FILE = "Mistral-Nemo-Instruct-2407-Q4_K_M.gguf"


class HFClient:
    """llama-cpp-python wrapper exposing the OllamaClient interface.

    Class-level singleton: model loads once, lives for the whole process.
    """

    _instance: "HFClient | None" = None
    _llm: Any = None  # class attribute, not instance attribute

    def __new__(cls, *args, **kwargs):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance

    def __init__(self, base_url: str = "", timeout: int = 900, **kwargs: Any):
        # Singleton: already constructed - nothing to do.
        # Model loads lazily on first generate() via _load().
        pass

    # ------------------------------------------------------------------
    @classmethod
    def _load(cls) -> None:
        """Load the GGUF model onto the class once."""
        if cls._llm is not None:
            return
        try:
            from llama_cpp import Llama
        except ImportError as exc:
            raise OllamaConnectionError(
                "llama-cpp-python required. Install with: "
                "pip install llama-cpp-python"
            ) from exc

        logger.info("HFClient: loading {} ({})", DEFAULT_REPO, DEFAULT_FILE)
        try:
            cls._llm = Llama.from_pretrained(
                repo_id=DEFAULT_REPO,
                filename=DEFAULT_FILE,
                n_gpu_layers=-1,
                n_ctx=16384,
                verbose=False,
            )
        except Exception as exc:  # noqa: BLE001
            msg = str(exc)
            if "404" in msg or "not found" in msg.lower():
                raise OllamaModelNotFoundError(
                    f"GGUF file '{DEFAULT_FILE}' not found in {DEFAULT_REPO}"
                ) from exc
            raise OllamaConnectionError(f"GGUF model load failed: {exc}") from exc

    # ------------------------------------------------------------------
    async def health_check(self) -> bool:
        try:
            import llama_cpp  # noqa: F401
            return True
        except ImportError:
            return False

    async def list_models(self) -> list[str]:
        return [f"{DEFAULT_REPO}/{DEFAULT_FILE}"]

    @property
    def model_id(self) -> str:
        return f"{DEFAULT_REPO}/{DEFAULT_FILE}"

    @property
    def timeout(self) -> float:
        # Compat shim: callers (designer / analyzer) may read this.
        return 900.0

    # ------------------------------------------------------------------
    async def generate(
        self,
        model: str | None = None,
        prompt: str = "",
        messages: list[dict] | None = None,
        system: str | None = None,
        temperature: float = 0.7,
        max_tokens: int = 1500,
        json_mode: bool = False,
        **kwargs: Any,
    ) -> str:
        del model  # bound model is fixed at class level

        # Ensure the class-level llm is loaded BEFORE running in executor
        # so cancellation can't strand a half-loaded model.
        HFClient._load()

        sys_text = system if system is not None else kwargs.get("system", "")
        sys_text = (sys_text or "").rstrip()
        if json_mode:
            sys_text = (
                sys_text
                + "\n\nRespond with ONLY valid JSON, no markdown, no commentary."
            ).strip()

        user_text = prompt
        if not user_text and messages:
            for m in messages:
                if m.get("role") == "user":
                    user_text = m.get("content", "")
                    break

        formatted = f"[INST] <<SYS>>\n{sys_text}\n<</SYS>>\n\n{user_text} [/INST]"

        timeout = float(kwargs.get("timeout", self.timeout))
        loop = asyncio.get_event_loop()
        try:
            raw = await asyncio.wait_for(
                loop.run_in_executor(
                    None, lambda: self._run(formatted, temperature, max_tokens)
                ),
                timeout=timeout,
            )
        except asyncio.TimeoutError as exc:
            raise OllamaConnectionError(
                "LLM took too long. Try a smaller model or split the input."
            ) from exc

        cleaned = raw.strip()
        if json_mode:
            cleaned = _extract_json(cleaned)
            try:
                json.loads(cleaned)
            except json.JSONDecodeError:
                logger.warning(
                    "HFClient: JSON parse failed, retrying once at lower temp"
                )
                try:
                    raw2 = await asyncio.wait_for(
                        loop.run_in_executor(
                            None,
                            lambda: self._run(
                                formatted, max(0.1, temperature - 0.3), max_tokens
                            ),
                        ),
                        timeout=timeout,
                    )
                except asyncio.TimeoutError as exc2:
                    raise OllamaConnectionError(
                        "LLM took too long. Try a smaller model or split the input."
                    ) from exc2
                cleaned = _extract_json(raw2.strip())
                try:
                    json.loads(cleaned)
                except json.JSONDecodeError as exc3:
                    raise OllamaInvalidJSONError(
                        f"GGUF model did not return valid JSON after retry: {exc3}"
                    ) from exc3
        return cleaned

    # ------------------------------------------------------------------
    def _run(self, formatted_prompt: str, temperature: float, max_tokens: int) -> str:
        cls = type(self)
        cls._load()
        llm = cls._llm
        assert llm is not None
        out = llm(
            formatted_prompt,
            max_tokens=max_tokens,
            temperature=temperature,
            stop=["[/INST]", "</s>"],
            echo=False,
        )
        if isinstance(out, dict):
            choices = out.get("choices", [])
            if choices:
                return str(choices[0].get("text", ""))
        return str(out)


# ---------------------------------------------------------------------------
def _extract_json(text: str) -> str:
    fence = re.search(r"```(?:json)?\s*(.*?)```", text, re.DOTALL)
    if fence:
        text = fence.group(1).strip()
    first = text.find("{")
    last = text.rfind("}")
    if first != -1 and last != -1 and last > first:
        return text[first : last + 1]
    return text


__all__ = [
    "HFClient",
    "DEFAULT_REPO",
    "DEFAULT_FILE",
    "OllamaConnectionError",
    "OllamaError",
    "OllamaInvalidJSONError",
    "OllamaModelNotFoundError",
]
