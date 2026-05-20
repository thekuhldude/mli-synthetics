from __future__ import annotations

"""Anthropic API client - drop-in replacement for OllamaClient / HFClient.

Uses the official `anthropic` SDK with the AsyncAnthropic client. The
designer's system prompt (knowledge base + fixture library, ~30 KB) is
identical across every chunk call, so we set `cache_control` on the
system block to cut ~90 % off the cached portion's cost after the first
request.

Activate by setting `USE_ANTHROPIC=true` (or `use_anthropic=true` in
settings). The factory in `mli_synthetics.llm.get_llm_client` reads
that flag and returns an `AnthropicClient` instead of OllamaClient /
HFClient / GroqClient.
"""

import json
import os
import re
from datetime import datetime
from pathlib import Path
from typing import Any

from tenacity import (
    AsyncRetrying,
    RetryError,
    retry_if_exception_type,
    stop_after_attempt,
    wait_exponential,
)

from mli_synthetics.errors import (
    OllamaConnectionError,
    OllamaError,
    OllamaInvalidJSONError,
    OllamaModelNotFoundError,
)
from mli_synthetics.logging_config import get_logger

logger = get_logger()

DEFAULT_ANTHROPIC_MODEL = "claude-haiku-4-5-20251001"
# Caching breakpoint: only apply cache_control when the system prompt is
# above the minimum cacheable prefix for the model. Haiku 4.5 minimum is
# 4096 tokens (~16K chars at ~4 chars/token).
_CACHE_MIN_CHARS = 4096 * 4


class AnthropicClient:
    """Mirrors the OllamaClient.generate interface over the Anthropic API."""

    def __init__(self, settings: Any | None = None, **kwargs: Any):
        if settings is None:
            from mli_synthetics.settings import get_settings

            settings = get_settings()
        self.settings = settings
        self.api_key = (
            getattr(settings, "anthropic_api_key", "")
            or os.environ.get("ANTHROPIC_API_KEY", "")
        )
        self.model = (
            getattr(settings, "anthropic_model", "") or DEFAULT_ANTHROPIC_MODEL
        )
        self.timeout = float(getattr(settings, "ollama_timeout_seconds", 600))
        self.log_dir = Path(settings.outputs_dir) / "llm_logs"
        self.log_dir.mkdir(parents=True, exist_ok=True)
        self._client: Any = None

    # ------------------------------------------------------------------
    def _get_client(self) -> Any:
        if self._client is not None:
            return self._client
        try:
            from anthropic import AsyncAnthropic
        except ImportError as exc:
            raise OllamaConnectionError(
                "anthropic SDK required. Install with: pip install anthropic"
            ) from exc
        if not self.api_key:
            raise OllamaConnectionError(
                "anthropic_api_key not set. Put ANTHROPIC_API_KEY in your "
                ".env or pass anthropic_api_key in settings."
            )
        # Disable the SDK's built-in retries; tenacity handles them here.
        self._client = AsyncAnthropic(
            api_key=self.api_key,
            timeout=self.timeout,
            max_retries=0,
        )
        return self._client

    # ------------------------------------------------------------------
    async def health_check(self) -> bool:
        try:
            import anthropic  # noqa: F401
            return bool(self.api_key)
        except ImportError:
            return False

    async def list_models(self) -> list[str]:
        return [self.model]

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
        model_id = model or self.model

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

        # Prompt caching: apply cache_control to the system block when it
        # is long enough to clear the model's minimum cacheable prefix.
        # The designer's KB is ~30 KB, well above this threshold.
        request: dict[str, Any] = {
            "model": model_id,
            "max_tokens": max_tokens,
            "messages": [{"role": "user", "content": user_text}],
            "temperature": temperature,
        }
        if sys_text:
            if len(sys_text) >= _CACHE_MIN_CHARS:
                request["system"] = [
                    {
                        "type": "text",
                        "text": sys_text,
                        "cache_control": {"type": "ephemeral"},
                    }
                ]
            else:
                request["system"] = sys_text

        raw = await self._call_with_retry(client, request)

        cleaned = raw.strip()
        if json_mode:
            cleaned = _extract_json(cleaned)
            try:
                json.loads(cleaned)
            except json.JSONDecodeError as exc:
                # Retry once at lower temperature for a stricter JSON pass.
                logger.warning(
                    "AnthropicClient: JSON parse failed, retrying once at lower temp"
                )
                request["temperature"] = max(0.1, temperature - 0.3)
                raw2 = await self._call_with_retry(client, request)
                cleaned = _extract_json(raw2.strip())
                try:
                    json.loads(cleaned)
                except json.JSONDecodeError as exc2:
                    raise OllamaInvalidJSONError(
                        f"Anthropic model {model_id} did not return valid JSON "
                        f"after retry: {exc2}"
                    ) from exc

        self._log_exchange(request, cleaned)
        return cleaned

    # ------------------------------------------------------------------
    async def _call_with_retry(self, client: Any, request: dict[str, Any]) -> str:
        try:
            import anthropic
        except ImportError as exc:  # pragma: no cover
            raise OllamaConnectionError(
                "anthropic SDK required. Install with: pip install anthropic"
            ) from exc

        try:
            async for attempt in AsyncRetrying(
                stop=stop_after_attempt(3),
                wait=wait_exponential(multiplier=1, min=1, max=8),
                retry=retry_if_exception_type(
                    (anthropic.APIConnectionError, anthropic.RateLimitError)
                ),
                reraise=True,
            ):
                with attempt:
                    response = await client.messages.create(**request)
                    # Surface cache stats for verification (per skill guidance)
                    usage = getattr(response, "usage", None)
                    if usage is not None:
                        cache_read = getattr(usage, "cache_read_input_tokens", 0) or 0
                        cache_write = (
                            getattr(usage, "cache_creation_input_tokens", 0) or 0
                        )
                        if cache_read or cache_write:
                            logger.debug(
                                "AnthropicClient: cache_read={} cache_write={} input={}",
                                cache_read,
                                cache_write,
                                getattr(usage, "input_tokens", 0),
                            )
                    # response.content is a list of content blocks; pull text
                    for block in response.content:
                        if getattr(block, "type", None) == "text":
                            return str(block.text)
                    return ""
        except anthropic.NotFoundError as exc:
            raise OllamaModelNotFoundError(
                f"Anthropic model '{request['model']}' not found. "
                "Check the model id or your API key permissions."
            ) from exc
        except anthropic.AuthenticationError as exc:
            raise OllamaConnectionError(
                f"Anthropic authentication failed: {exc}"
            ) from exc
        except (anthropic.APIConnectionError, anthropic.RateLimitError) as exc:
            raise OllamaConnectionError(
                f"Anthropic API connection failed after retries: {exc}"
            ) from exc
        except anthropic.APIStatusError as exc:
            raise OllamaConnectionError(
                f"Anthropic API error {exc.status_code}: {exc.message}"
            ) from exc
        except RetryError as exc:
            raise OllamaConnectionError(
                f"Anthropic API request failed after retries: {exc}"
            ) from exc
        raise OllamaError("Unreachable")  # pragma: no cover

    # ------------------------------------------------------------------
    def _log_exchange(self, request: dict[str, Any], response: str) -> None:
        try:
            stamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
            safe = re.sub(r"[^A-Za-z0-9_.-]", "_", str(request.get("model", "model")))
            path = self.log_dir / f"{stamp}_anthropic_{safe}.json"
            sys_value = request.get("system", "")
            if isinstance(sys_value, list) and sys_value:
                sys_preview = sys_value[0].get("text", "")[:500]
            else:
                sys_preview = str(sys_value)[:500]
            payload = {
                "timestamp": datetime.now().isoformat(),
                "backend": "anthropic",
                "model": request.get("model"),
                "request": {
                    "system": sys_preview,
                    "user": (
                        request["messages"][0]["content"][:500]
                        if request.get("messages")
                        else ""
                    ),
                    "temperature": request.get("temperature"),
                    "max_tokens": request.get("max_tokens"),
                },
                "response_preview": response[:2000],
                "response_length_chars": len(response),
            }
            path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
        except Exception as exc:  # noqa: BLE001
            logger.debug("AnthropicClient: failed to log exchange: {}", exc)


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
    "AnthropicClient",
    "DEFAULT_ANTHROPIC_MODEL",
    "OllamaConnectionError",
    "OllamaError",
    "OllamaInvalidJSONError",
    "OllamaModelNotFoundError",
]
