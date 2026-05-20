"""Two-stage song analyzer: numerical features + LLM interpretation."""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from pydantic import BaseModel

from mli_synthetics.audio.features import NumericalFeatures, extract_numerical_features
from mli_synthetics.audio.structure import SongAnalysis
from mli_synthetics.errors import AudioAnalysisError
from mli_synthetics.logging_config import get_logger

logger = get_logger()


# ---------------------------------------------------------------------------
# Enum coercion - small LLMs invent label variants. Map them to valid values
# before Pydantic validates; unknown values fall back to a safe default.
# ---------------------------------------------------------------------------
VALID_LABELS = {
    "intro", "verse", "pre_chorus", "chorus", "drop",
    "build_up", "breakdown", "outro", "instrumental",
}
VALID_MOMENTS = {"drop", "build_peak", "vocal_entry", "instrumental_solo"}

LABEL_ALIASES: dict[str, str] = {
    "verse_bridge": "verse",
    "bridge": "verse",
    "pre-chorus": "pre_chorus",
    "prechorus": "pre_chorus",
    "drop_section": "drop",
    "buildup": "build_up",
    "build-up": "build_up",
    "break": "breakdown",
    "break_down": "breakdown",
    "outro_fade": "outro",
}

MOMENT_ALIASES: dict[str, str] = {
    "chorus": "build_peak",
    "verse": "vocal_entry",
    "solo": "instrumental_solo",
    "instrumental": "instrumental_solo",
    "peak": "build_peak",
    "climax": "build_peak",
}


def coerce_labels(parsed_dict: dict) -> dict:
    """Coerce LLM-invented enum values to valid SongAnalysis enums in place.

    Unknown labels fall back to ``instrumental``; unknown moment types fall
    back to ``build_peak``. Each coercion is logged at warning level.
    """
    for seg in parsed_dict.get("structure", []) or []:
        raw = str(seg.get("label", "")).strip().lower().replace("-", "_")
        if raw in VALID_LABELS:
            seg["label"] = raw
            continue
        if raw in LABEL_ALIASES:
            mapped = LABEL_ALIASES[raw]
            logger.warning("coerce_labels: section label '{}' -> '{}'", raw, mapped)
            seg["label"] = mapped
        else:
            logger.warning(
                "coerce_labels: unknown section label '{}' -> 'instrumental'", raw
            )
            seg["label"] = "instrumental"

    for moment in parsed_dict.get("key_moments", []) or []:
        raw = str(moment.get("type", "")).strip().lower().replace("-", "_")
        if raw in VALID_MOMENTS:
            moment["type"] = raw
            continue
        if raw in MOMENT_ALIASES:
            mapped = MOMENT_ALIASES[raw]
            logger.warning("coerce_labels: key_moment type '{}' -> '{}'", raw, mapped)
            moment["type"] = mapped
        else:
            logger.warning(
                "coerce_labels: unknown key_moment type '{}' -> 'build_peak'", raw
            )
            moment["type"] = "build_peak"

    return parsed_dict


class SongAnalysisResult(BaseModel):
    numerical: NumericalFeatures
    interpreted: SongAnalysis | None = None


class SongAnalyzer:
    """Top-level entry point for song analysis.

    The LLM stage is optional and lazily imported so that numerical
    analysis still works when Ollama is unavailable.
    """

    def __init__(self, ollama_client: Any | None = None, settings: Any | None = None):
        self._ollama = ollama_client
        if settings is None:
            from mli_synthetics.settings import get_settings

            settings = get_settings()
        self.settings = settings

    async def analyze(
        self,
        audio_path: Path,
        skip_llm: bool = False,
    ) -> SongAnalysisResult:
        numerical = extract_numerical_features(audio_path)
        if skip_llm:
            return SongAnalysisResult(numerical=numerical)

        try:
            interpreted = await self._llm_interpret(numerical)
        except Exception as exc:  # noqa: BLE001
            logger.warning("LLM interpretation failed; falling back to numerical only: {}", exc)
            interpreted = None

        return SongAnalysisResult(numerical=numerical, interpreted=interpreted)

    # ------------------------------------------------------------------
    async def _llm_interpret(self, numerical: NumericalFeatures) -> SongAnalysis:
        from mli_synthetics.llm import get_llm_client
        from mli_synthetics.llm.prompts import (
            ANALYZER_SYSTEM_PROMPT,
            ANALYZER_USER_PROMPT_TEMPLATE,
        )

        ollama = self._ollama or get_llm_client(self.settings)
        features_summary = _format_features_summary(numerical)
        user_prompt = ANALYZER_USER_PROMPT_TEMPLATE.format(
            features_summary=features_summary
        )

        raw = await ollama.generate(
            model=self.settings.ollama_model_analyzer,
            prompt=user_prompt,
            system=ANALYZER_SYSTEM_PROMPT,
            temperature=self.settings.analyzer_temperature,
            max_tokens=self.settings.max_tokens_analyzer,
            json_mode=True,
        )
        try:
            data = json.loads(raw)
        except json.JSONDecodeError as exc:
            raise AudioAnalysisError(f"LLM returned invalid JSON: {exc}") from exc
        data = coerce_labels(data)
        analysis = SongAnalysis.model_validate(data)
        analysis.duration_s = numerical.duration_s
        analysis.tempo_bpm = numerical.tempo_bpm
        analysis.key = numerical.key
        return analysis


def _format_features_summary(n: NumericalFeatures) -> str:
    s = n.summary()
    energy_snippet = _summarize_curve(n.rms_curve)
    drops = ", ".join(f"{a:.1f}-{b:.1f}s" for a, b in s["drop_candidates"]) or "none"
    segments = "; ".join(
        f"{seg['start_s']:.1f}-{seg['end_s']:.1f}s (cluster {seg['label']})"
        for seg in s["structural_segments"]
    )
    return (
        f"- Duration: {s['duration_s']}s\n"
        f"- Tempo: {s['tempo_bpm']} BPM\n"
        f"- Key: {s['key']}\n"
        f"- Beats detected: {s['n_beats']}\n"
        f"- Onset density: {s['onset_density_per_s']} per second\n"
        f"- Structural segments: {segments}\n"
        f"- Energy curve (RMS, 1Hz, summary): {energy_snippet}\n"
        f"- Drop candidates: {drops}\n"
        f"- High-energy peak count: {s['n_energy_peaks']}\n"
    )


def _summarize_curve(curve: list[float]) -> str:
    if not curve:
        return "empty"
    import numpy as np

    arr = np.array(curve)
    n = len(arr)
    # Split into 8 segments and report mean of each, normalized
    buckets = np.array_split(arr, min(8, n))
    means = [float(b.mean()) for b in buckets if len(b) > 0]
    if not means:
        return "empty"
    peak = max(means) or 1.0
    normalized = [m / peak for m in means]
    return "[" + ", ".join(f"{v:.2f}" for v in normalized) + "]  (segments low->high)"
