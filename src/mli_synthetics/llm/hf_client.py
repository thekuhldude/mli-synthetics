from __future__ import annotations

"""HuggingFace transformers client - Mistral-Nemo 4-bit on GPU.

Class-level singleton: the 12B model loads into VRAM exactly once per
process even if multiple call sites (analyzer + designer + warmup)
instantiate `HFClient()`. All model state lives on the class object
(`_instance`, `_pipeline_obj`, `_tokenizer`) so it survives repeated
constructor calls.

Activate by setting `USE_HF_CLIENT=true`. The factory in
`mli_synthetics.llm.get_default_client` then returns an `HFClient`
instead of the default `OllamaClient`.
"""

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


class HFClient:
    """transformers-based singleton client matching `OllamaClient`'s interface."""

    _instance: "HFClient | None" = None
    _pipeline_obj: Any = None  # class attribute - the transformers pipeline
    _tokenizer: Any = None
    _model_id: str | None = None

    def __new__(cls, *args, **kwargs):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance

    def __init__(self, *args, **kwargs):
        # Lazy: the model loads on first generate() call via _load().
        # Doing nothing here makes the singleton trivially safe to
        # re-construct from any call site.
        pass

    # ------------------------------------------------------------------
    @classmethod
    def _load(cls) -> None:
        """Load the model + tokenizer once into class attributes."""
        if cls._pipeline_obj is not None:
            return
        try:
            import torch
            from transformers import (
                AutoModelForCausalLM,
                AutoTokenizer,
                BitsAndBytesConfig,
                pipeline,
            )
        except ImportError as exc:
            raise OllamaConnectionError(
                "HFClient requires transformers + torch + bitsandbytes. "
                "Install with: pip install transformers torch accelerate bitsandbytes"
            ) from exc

        model_id = os.environ.get("HF_MODEL_ID", DEFAULT_HF_MODEL)
        cls._model_id = model_id

        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.float16,
            bnb_4bit_use_double_quant=True,
        )

        logger.info("HFClient: loading {} (4-bit nf4, fp16 compute)", model_id)
        try:
            cls._tokenizer = AutoTokenizer.from_pretrained(model_id)
            model = AutoModelForCausalLM.from_pretrained(
                model_id,
                quantization_config=bnb_config,
                device_map="auto",
                torch_dtype=torch.float16,
            )
        except Exception as exc:  # noqa: BLE001
            msg = str(exc)
            if "404" in msg or "not found" in msg.lower():
                raise OllamaModelNotFoundError(
                    f"HF model '{model_id}' not accessible. "
                    "Check the model id and your HuggingFace token."
                ) from exc
            raise OllamaConnectionError(f"HF model load failed: {exc}") from exc

        cls._pipeline_obj = pipeline(
            "text-generation",
            model=model,
            tokenizer=cls._tokenizer,
        )

    # ------------------------------------------------------------------
    async def health_check(self) -> bool:
        try:
            import torch  # noqa: F401
            import transformers  # noqa: F401

            return True
        except ImportError:
            return False

    async def list_models(self) -> list[str]:
        return [type(self)._model_id or DEFAULT_HF_MODEL]

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
        del model  # bound model is fixed at class level

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

        chat_messages: list[dict[str, str]] = []
        if sys_text:
            chat_messages.append({"role": "system", "content": sys_text})
        chat_messages.append({"role": "user", "content": user_text})

        timeout = float(kwargs.get("timeout", 900))
        loop = asyncio.get_event_loop()
        # Trigger the load up front so the executor task doesn't get
        # cancelled mid-load by `asyncio.wait_for`.
        HFClient._load()
        try:
            raw = await asyncio.wait_for(
                loop.run_in_executor(
                    None,
                    lambda: self._generate_sync(
                        chat_messages, temperature, max_tokens
                    ),
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
                            lambda: self._generate_sync(
                                chat_messages,
                                max(0.1, temperature - 0.3),
                                max_tokens,
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
                        f"Model did not return valid JSON after retry: {exc3}"
                    ) from exc3
        self._log_exchange(chat_messages, cleaned, temperature, max_tokens)
        return cleaned

    # ------------------------------------------------------------------
    def _generate_sync(
        self,
        messages: list[dict],
        temperature: float,
        max_tokens: int,
    ) -> str:
        cls = type(self)
        tokenizer = cls._tokenizer
        pipe = cls._pipeline_obj
        assert tokenizer is not None and pipe is not None

        text = tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        kwargs: dict[str, Any] = {
            "max_new_tokens": max_tokens,
            "return_full_text": False,
            "pad_token_id": tokenizer.eos_token_id,
        }
        if temperature > 0:
            kwargs["temperature"] = temperature
            kwargs["do_sample"] = True
        else:
            kwargs["do_sample"] = False

        result = pipe(text, **kwargs)
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
            from mli_synthetics.settings import get_settings

            log_dir = get_settings().outputs_dir / "llm_logs"
            log_dir.mkdir(parents=True, exist_ok=True)
            stamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
            safe = re.sub(r"[^A-Za-z0-9_.-]", "_", type(self)._model_id or "model")
            path = log_dir / f"{stamp}_{safe}.json"
            payload = {
                "timestamp": datetime.now().isoformat(),
                "backend": "hf",
                "model": type(self)._model_id,
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
    "DEFAULT_HF_MODEL",
    "OllamaConnectionError",
    "OllamaError",
    "OllamaInvalidJSONError",
    "OllamaModelNotFoundError",
]
