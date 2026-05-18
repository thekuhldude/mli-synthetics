"""Async Ollama HTTP client."""
from __future__ import annotations

import json
import re
from datetime import datetime
from pathlib import Path
from typing import Any

import httpx
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


class OllamaClient:
    def __init__(
        self,
        base_url: str = "http://localhost:11434",
        timeout: int = 300,
        log_dir: Path | None = None,
    ):
        self.base_url = base_url.rstrip("/")
        self.timeout = timeout
        if log_dir is None:
            from mli_synthetics.settings import get_settings

            log_dir = get_settings().outputs_dir / "llm_logs"
        self.log_dir = log_dir
        self.log_dir.mkdir(parents=True, exist_ok=True)

    # ------------------------------------------------------------------
    async def health_check(self) -> bool:
        try:
            async with httpx.AsyncClient(timeout=10) as client:
                resp = await client.get(f"{self.base_url}/api/tags")
                return resp.status_code == 200
        except httpx.HTTPError as exc:
            logger.warning("Ollama health check failed: {}", exc)
            return False

    async def list_models(self) -> list[str]:
        try:
            async with httpx.AsyncClient(timeout=10) as client:
                resp = await client.get(f"{self.base_url}/api/tags")
                resp.raise_for_status()
                data = resp.json()
                return [m["name"] for m in data.get("models", [])]
        except httpx.HTTPError as exc:
            raise OllamaConnectionError(f"Cannot reach Ollama at {self.base_url}: {exc}") from exc

    # ------------------------------------------------------------------
    async def generate(
        self,
        model: str,
        prompt: str,
        system: str | None = None,
        temperature: float = 0.7,
        max_tokens: int = 2000,
        json_mode: bool = False,
    ) -> str:
        if json_mode and system is not None:
            system = (
                system.rstrip()
                + "\n\nRespond with ONLY valid JSON, no markdown, no commentary."
            )
        body: dict[str, Any] = {
            "model": model,
            "prompt": prompt,
            "stream": False,
            "options": {
                "temperature": temperature,
                "num_predict": max_tokens,
            },
        }
        if system:
            body["system"] = system
        if json_mode:
            body["format"] = "json"

        response_text = await self._call_with_retry(body)
        cleaned = response_text.strip()

        if json_mode:
            cleaned = _extract_json(cleaned)
            try:
                json.loads(cleaned)
            except json.JSONDecodeError:
                logger.warning("First JSON parse failed; retrying once")
                body["options"]["temperature"] = max(0.1, temperature - 0.3)
                response_text = await self._call_with_retry(body)
                cleaned = _extract_json(response_text.strip())
                try:
                    json.loads(cleaned)
                except json.JSONDecodeError as exc:
                    raise OllamaInvalidJSONError(
                        f"Model {model} did not return valid JSON after retry: {exc}"
                    ) from exc

        self._log_exchange(body, cleaned)
        return cleaned

    # ------------------------------------------------------------------
    async def _call_with_retry(self, body: dict[str, Any]) -> str:
        timeout = float(self.timeout)
        try:
            async for attempt in AsyncRetrying(
                stop=stop_after_attempt(3),
                wait=wait_exponential(multiplier=1, min=1, max=8),
                retry=retry_if_exception_type(httpx.TransportError),
                reraise=True,
            ):
                with attempt:
                    try:
                        async with httpx.AsyncClient(timeout=timeout) as client:
                            resp = await client.post(
                                f"{self.base_url}/api/generate", json=body
                            )
                    except httpx.ReadTimeout as exc:
                        raise OllamaConnectionError(
                            "LLM took too long. Try a smaller model or split the input."
                        ) from exc
                    if resp.status_code == 404:
                        raise OllamaModelNotFoundError(
                            f"Model '{body['model']}' not found. "
                            f"Run: ollama pull {body['model']}"
                        )
                    resp.raise_for_status()
                    data = resp.json()
                    return data.get("response", "")
        except OllamaModelNotFoundError:
            raise
        except OllamaConnectionError:
            raise
        except (httpx.HTTPError, RetryError) as exc:
            raise OllamaConnectionError(
                f"Ollama request failed: {exc}"
            ) from exc
        raise OllamaError("Unreachable")  # pragma: no cover

    # ------------------------------------------------------------------
    def _log_exchange(self, request_body: dict[str, Any], response: str) -> None:
        try:
            stamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
            path = self.log_dir / f"{stamp}_{request_body.get('model', 'unknown')}.json"
            payload = {
                "timestamp": datetime.now().isoformat(),
                "request": {
                    "model": request_body.get("model"),
                    "system": request_body.get("system", "")[:500],
                    "prompt": request_body.get("prompt", "")[:500],
                    "options": request_body.get("options", {}),
                    "format": request_body.get("format"),
                },
                "response_preview": response[:2000],
                "response_length_chars": len(response),
            }
            path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
        except Exception as exc:  # noqa: BLE001
            logger.debug("Failed to log LLM exchange: {}", exc)


# ---------------------------------------------------------------------------
def _extract_json(text: str) -> str:
    """Strip markdown fences and surrounding prose around a JSON object."""
    # Remove ```json ... ``` fences
    fence = re.search(r"```(?:json)?\s*(.*?)```", text, re.DOTALL)
    if fence:
        text = fence.group(1).strip()
    # Trim to first '{' and last '}'
    first = text.find("{")
    last = text.rfind("}")
    if first != -1 and last != -1 and last > first:
        return text[first : last + 1]
    return text
