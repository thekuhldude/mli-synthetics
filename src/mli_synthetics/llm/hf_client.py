"""HuggingFace text-generation client - drop-in replacement for OllamaClient.

Same async interface (`generate`, `health_check`, `list_models`) so the
designer / analyzer code can use either backend transparently. Loading
transformers/torch is deferred until the first `generate` call so that
unit tests and CI without a GPU can still import this module.

Activate by setting `USE_HF_CLIENT=true` in the environment. The
`get_default_client` factory in `mli_synthetics.llm` reads that env
var and returns either an `HFClient` or an `OllamaClient`.

Default model: `mistralai/Mistral-Nemo-Instruct-2407`. On a 16 GB GPU
(Kaggle T4) you almost certainly need 4-bit quantization - set
`HF_LOAD_IN_4BIT=true` (default) and have `bitsandbytes` installed.
"""
from __future__ import annotations

import asyncio
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

DEFAULT_HF_MODEL = "mistralai/Mistral-Nemo-Instruct-2407"


# Module-level singleton. The factory in mli_synthetics.llm builds a
# new HFClient on every call, which was loading the 12B model into VRAM
# multiple times on Kaggle and OOM-killing the kernel. The __new__ hook
# below makes every HFClient() call return the same instance.
_GLOBAL_CLIENT: "HFClient | None" = None


class HFClient:
    """Mirrors `OllamaClient.generate` over a local transformers pipeline.

    Singleton: every constructor call returns the same instance, so the
    model is loaded into GPU memory exactly once per process.
    """

    def __new__(cls, *args, **kwargs):
        global _GLOBAL_CLIENT
        if _GLOBAL_CLIENT is None:
            _GLOBAL_CLIENT = super().__new__(cls)
            _GLOBAL_CLIENT._initialized = False
        return _GLOBAL_CLIENT

    def __init__(
        self,
        base_url: str = "",  # accepted for compat with OllamaClient signature
        timeout: int = 600,
        model_id: str | None = None,
        log_dir: Path | None = None,
        **kwargs: Any,  # swallow any extra keyword args
    ):
        if getattr(self, "_initialized", False):
            return
        self._initialized = True
        self.base_url = base_url
        self.timeout = timeout
        self.model_id = model_id or os.environ.get("HF_MODEL_ID", DEFAULT_HF_MODEL)
        if log_dir is None:
            from mli_synthetics.settings import get_settings

            log_dir = get_settings().outputs_dir / "llm_logs"
        self.log_dir = Path(log_dir)
        self.log_dir.mkdir(parents=True, exist_ok=True)
        self._pipeline = None
        self._tokenizer = None

    # ------------------------------------------------------------------
    async def health_check(self) -> bool:
        try:
            import torch  # noqa: F401
            import transformers  # noqa: F401

            return True
        except ImportError as exc:
            logger.warning("HFClient health_check: missing deps ({})", exc)
            return False

    async def list_models(self) -> list[str]:
        return [self.model_id]

    # ------------------------------------------------------------------
    def _load(self) -> None:
        if self._pipeline is not None:
            return
        try:
            import torch
            from transformers import (
                AutoModelForCausalLM,
                AutoTokenizer,
                pipeline,
            )
        except ImportError as exc:
            raise OllamaConnectionError(
                "HFClient requires `transformers` and `torch`. "
                "Install with: pip install transformers torch accelerate"
            ) from exc

        load_4bit = os.environ.get("HF_LOAD_IN_4BIT", "true").lower() in {
            "1", "true", "yes",
        }
        gpu = torch.cuda.is_available()

        model_kwargs: dict[str, Any] = {}
        if gpu:
            model_kwargs["device_map"] = "auto"
            if load_4bit:
                try:
                    from transformers import BitsAndBytesConfig

                    model_kwargs["quantization_config"] = BitsAndBytesConfig(
                        load_in_4bit=True,
                        bnb_4bit_compute_dtype=torch.bfloat16,
                        bnb_4bit_quant_type="nf4",
                        bnb_4bit_use_double_quant=True,
                    )
                    logger.info("HFClient: loading {} in 4-bit on GPU", self.model_id)
                except ImportError:
                    logger.warning(
                        "bitsandbytes not installed; falling back to bfloat16. "
                        "On T4 (16GB) Mistral-Nemo will OOM - install bitsandbytes."
                    )
                    model_kwargs["torch_dtype"] = torch.bfloat16
            else:
                model_kwargs["torch_dtype"] = torch.bfloat16
                logger.info("HFClient: loading {} in bfloat16 on GPU", self.model_id)
        else:
            logger.warning(
                "HFClient: no GPU detected, falling back to CPU (will be very slow)"
            )

        try:
            self._tokenizer = AutoTokenizer.from_pretrained(self.model_id)
            model = AutoModelForCausalLM.from_pretrained(self.model_id, **model_kwargs)
        except Exception as exc:  # noqa: BLE001
            msg = str(exc)
            if "404" in msg or "not found" in msg.lower():
                raise OllamaModelNotFoundError(
                    f"HF model '{self.model_id}' not found or not accessible. "
                    "Check the model id and your HuggingFace token."
                ) from exc
            raise OllamaConnectionError(f"HF model load failed: {exc}") from exc

        self._pipeline = pipeline(
            "text-generation",
            model=model,
            tokenizer=self._tokenizer,
        )

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
        # `model` kwarg from the call sites refers to the Ollama tag; we
        # ignore it - the HFClient is bound to a single model_id at init.
        del model

        if json_mode and system is not None:
            system = (
                system.rstrip()
                + "\n\nRespond with ONLY valid JSON, no markdown, no commentary."
            )
        elif json_mode:
            system = "Respond with ONLY valid JSON, no markdown, no commentary."

        messages: list[dict[str, str]] = []
        if system:
            messages.append({"role": "system", "content": system})
        messages.append({"role": "user", "content": prompt})

        loop = asyncio.get_event_loop()
        try:
            raw = await asyncio.wait_for(
                loop.run_in_executor(
                    None,
                    lambda: self._generate_sync(
                        messages, temperature, max_tokens
                    ),
                ),
                timeout=float(self.timeout),
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
            except json.JSONDecodeError as exc:
                # Single retry at lower temperature
                logger.warning("HFClient: JSON parse failed, retrying once")
                try:
                    raw_retry = await asyncio.wait_for(
                        loop.run_in_executor(
                            None,
                            lambda: self._generate_sync(
                                messages,
                                max(0.1, temperature - 0.3),
                                max_tokens,
                            ),
                        ),
                        timeout=float(self.timeout),
                    )
                except asyncio.TimeoutError as exc2:
                    raise OllamaConnectionError(
                        "LLM took too long. Try a smaller model or split the input."
                    ) from exc2
                cleaned = _extract_json(raw_retry.strip())
                try:
                    json.loads(cleaned)
                except json.JSONDecodeError as exc3:
                    raise OllamaInvalidJSONError(
                        f"HF model {self.model_id} did not return valid JSON "
                        f"after retry: {exc3}"
                    ) from exc

        self._log_exchange(messages, cleaned, temperature, max_tokens)
        return cleaned

    # ------------------------------------------------------------------
    def _generate_sync(
        self, messages: list[dict[str, str]], temperature: float, max_tokens: int
    ) -> str:
        self._load()
        assert self._tokenizer is not None and self._pipeline is not None

        text = self._tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        kwargs: dict[str, Any] = {
            "max_new_tokens": max_tokens,
            "return_full_text": False,
            "pad_token_id": self._tokenizer.eos_token_id,
        }
        if temperature > 0:
            kwargs["temperature"] = temperature
            kwargs["do_sample"] = True
        else:
            kwargs["do_sample"] = False

        result = self._pipeline(text, **kwargs)
        if isinstance(result, list) and result:
            return str(result[0].get("generated_text", ""))
        return str(result)

    # ------------------------------------------------------------------
    def _log_exchange(
        self,
        messages: list[dict[str, str]],
        response: str,
        temperature: float,
        max_tokens: int,
    ) -> None:
        try:
            stamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
            safe = re.sub(r"[^A-Za-z0-9_.-]", "_", self.model_id)
            path = self.log_dir / f"{stamp}_{safe}.json"
            payload = {
                "timestamp": datetime.now().isoformat(),
                "backend": "hf",
                "model": self.model_id,
                "request": {
                    "system": next(
                        (m["content"] for m in messages if m["role"] == "system"),
                        "",
                    )[:500],
                    "user": next(
                        (m["content"] for m in messages if m["role"] == "user"),
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
            logger.debug("HFClient: failed to log exchange: {}", exc)


# ---------------------------------------------------------------------------
def _extract_json(text: str) -> str:
    """Strip markdown fences and prose surrounding a JSON object."""
    fence = re.search(r"```(?:json)?\s*(.*?)```", text, re.DOTALL)
    if fence:
        text = fence.group(1).strip()
    first = text.find("{")
    last = text.rfind("}")
    if first != -1 and last != -1 and last > first:
        return text[first : last + 1]
    return text


# Re-export errors so import-side code stays clean
__all__ = [
    "HFClient",
    "DEFAULT_HF_MODEL",
    "OllamaConnectionError",
    "OllamaError",
    "OllamaInvalidJSONError",
    "OllamaModelNotFoundError",
]
