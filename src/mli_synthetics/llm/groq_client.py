from __future__ import annotations

"""Groq API client - drop-in replacement for OllamaClient / HFClient.

Uses the official `groq` SDK (OpenAI-compatible interface). Activate by
setting `USE_GROQ=true` (or `use_groq=true` in settings).
"""

import json
import os
import re
from datetime import datetime
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

DEFAULT_GROQ_MODEL = "llama-3.1-70b-versatile"


class GroqClient:
    """Mirrors the OllamaClient.generate interface over the Groq API."""

    def __init__(self, settings: Any | None = None, **kwargs: Any):
        if settings is None:
            from mli_synthetics.settings import get_settings

            settings = get_settings()
        self.settings = settings
        self.api_key = (
            getattr(settings, "groq_api_key", "")
            or os.environ.get("GROQ_API_KEY", "")
        )
        self.model = getattr(settings, "groq_model", "") or DEFAULT_GROQ_MODEL
        self.timeout = float(getattr(settings, "ollama_timeout_seconds", 600))
        self.log_dir = Path(settings.outputs_dir) / "llm_logs"
        self.log_dir.mkdir(parents=True, exist_ok=True)
        self._client: Any = None

    # ------------------------------------------------------------------
    def _get_client(self) -> Any:
        if self._client is not None:
            return self._client
        try:
            from groq import AsyncGroq
        except ImportError as exc:
            raise OllamaConnectionError(
                "groq SDK required. Install with: pip install groq"
            ) from exc
        if not self.api_key:
            raise OllamaConnectionError(
                "groq_api_key not set. Put GROQ_API_KEY in your .env or "
                "pass groq_api_key in settings."
            )
        self._client = AsyncGroq(api_key=self.api_key, timeout=self.timeout)
        return self._client

    # ------------------------------------------------------------------
    async def generate(
        self,
        model: str | None = None,
        prompt: str = "",
        system: str | None = None,
        messages: list[dict] | None = None,
        temperature: float = 0.7,
        max_tokens: int = 1500,
        json_mode: bool = False,
        **kwargs: Any,
    ) -> str:
        client = self._get_client()
        sys_text = system or kwargs.get("system", "") or ""
        if json_mode:
            sys_text = (
                sys_text + "\n\nRespond with ONLY valid JSON, no markdown."
            ).strip()

        user_text = prompt
        if not user_text and messages:
            for m in messages:
                if m.get("role") == "user":
                    user_text = m.get("content", "")
                    break

        msg_list: list[dict[str, str]] = []
        if sys_text:
            msg_list.append({"role": "system", "content": sys_text})
        msg_list.append({"role": "user", "content": user_text})

        try:
            response = await client.chat.completions.create(
                model=model or self.model,
                messages=msg_list,
                temperature=temperature,
                max_tokens=max_tokens,
            )
        except Exception as exc:  # noqa: BLE001
            msg = str(exc)
            low = msg.lower()
            if "404" in msg or "not_found" in low or "does not exist" in low:
                raise OllamaModelNotFoundError(
                    f"Groq model '{model or self.model}' not found: {exc}"
                ) from exc
            raise OllamaConnectionError(f"Groq API error: {exc}") from exc

        raw = response.choices[0].message.content
        raw = (raw or "").strip()

        if json_mode:
            raw = _extract_json(raw)
            try:
                json.loads(raw)
            except json.JSONDecodeError as exc:
                raise OllamaInvalidJSONError(
                    f"Groq model {model or self.model} returned invalid JSON: {exc}"
                ) from exc

        self._log_exchange(msg_list, model or self.model, raw, temperature, max_tokens)
        return raw

    # ------------------------------------------------------------------
    async def health_check(self) -> bool:
        try:
            await self.generate(prompt="hi", max_tokens=5)
            return True
        except Exception:  # noqa: BLE001
            return False

    async def list_models(self) -> list[str]:
        return [self.model]

    # ------------------------------------------------------------------
    def _log_exchange(
        self,
        msg_list: list[dict[str, str]],
        model: str,
        response: str,
        temperature: float,
        max_tokens: int,
    ) -> None:
        try:
            stamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
            safe = re.sub(r"[^A-Za-z0-9_.-]", "_", model)
            path = self.log_dir / f"{stamp}_groq_{safe}.json"
            payload = {
                "timestamp": datetime.now().isoformat(),
                "backend": "groq",
                "model": model,
                "request": {
                    "system": next(
                        (m["content"] for m in msg_list if m["role"] == "system"),
                        "",
                    )[:500],
                    "user": next(
                        (m["content"] for m in msg_list if m["role"] == "user"),
                        "",
                    )[:500],
                    "temperature": temperature,
                    "max_tokens": max_tokens,
                },
                "response_preview": response[:2000],
                "response_length_chars": len(response),
            }
            path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
        except Exception as exc:  # noqa: BLE001
            logger.debug("GroqClient: failed to log exchange: {}", exc)


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
    "GroqClient",
    "DEFAULT_GROQ_MODEL",
    "OllamaConnectionError",
    "OllamaError",
    "OllamaInvalidJSONError",
    "OllamaModelNotFoundError",
]
