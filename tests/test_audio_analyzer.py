"""Tests for the audio analyzer."""
from __future__ import annotations

import wave
from pathlib import Path

import numpy as np
import pytest

from mli_synthetics.audio.analyzer import (
    SongAnalyzer,
    coerce_labels,
)
from mli_synthetics.audio.features import extract_numerical_features
from mli_synthetics.errors import AudioAnalysisError


def _write_synthetic_wav(path: Path, duration_s: float = 6.0, sr: int = 22050) -> None:
    """Write a simple sine + click track for deterministic tests."""
    t = np.linspace(0, duration_s, int(sr * duration_s), endpoint=False)
    # Sine carrier
    audio = 0.3 * np.sin(2 * np.pi * 220 * t)
    # Clicks every 0.5s to give librosa beats to find
    click_interval = int(sr * 0.5)
    for i in range(0, len(audio), click_interval):
        end = min(i + 100, len(audio))
        audio[i:end] += 0.6
    # Loud middle section for energy peaks / drop candidate
    mid_start = int(sr * 2.0)
    mid_end = int(sr * 5.0)
    audio[mid_start:mid_end] *= 2.5
    audio = np.clip(audio, -1.0, 1.0)
    pcm = (audio * 32767).astype(np.int16)
    with wave.open(str(path), "wb") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(sr)
        wf.writeframes(pcm.tobytes())


@pytest.fixture
def synthetic_wav(tmp_path: Path) -> Path:
    p = tmp_path / "test.wav"
    _write_synthetic_wav(p)
    return p


def test_numerical_features_extract(synthetic_wav):
    features = extract_numerical_features(synthetic_wav)
    assert features.duration_s > 5.0
    assert features.tempo_bpm > 0
    assert len(features.beat_times) > 0
    assert len(features.onset_times) > 0
    assert len(features.rms_curve) > 0
    assert features.key in {"C", "C#", "D", "D#", "E", "F", "F#", "G", "G#", "A", "A#", "B"}
    assert len(features.structural_segments) >= 1


def test_missing_file_raises(tmp_path):
    with pytest.raises(AudioAnalysisError):
        extract_numerical_features(tmp_path / "does_not_exist.wav")


def test_summary_serializable(synthetic_wav):
    import json

    features = extract_numerical_features(synthetic_wav)
    summary = features.summary()
    json.dumps(summary)  # should not raise


@pytest.mark.asyncio
async def test_skip_llm_returns_only_numerical(synthetic_wav):
    analyzer = SongAnalyzer()
    result = await analyzer.analyze(synthetic_wav, skip_llm=True)
    assert result.numerical is not None
    assert result.interpreted is None


def test_coerce_labels_maps_aliases():
    data = {
        "structure": [
            {"label": "pre-chorus", "start_s": 0, "end_s": 4, "energy": 0.5},
            {"label": "buildup", "start_s": 4, "end_s": 8, "energy": 0.6},
            {"label": "Verse_Bridge", "start_s": 8, "end_s": 12, "energy": 0.5},
        ],
        "key_moments": [
            {"time_s": 1.0, "type": "climax"},
            {"time_s": 2.0, "type": "solo"},
        ],
    }
    out = coerce_labels(data)
    assert out["structure"][0]["label"] == "pre_chorus"
    assert out["structure"][1]["label"] == "build_up"
    assert out["structure"][2]["label"] == "verse"
    assert out["key_moments"][0]["type"] == "build_peak"
    assert out["key_moments"][1]["type"] == "instrumental_solo"


def test_coerce_labels_falls_back_for_unknown():
    data = {
        "structure": [{"label": "freestyle_thing", "start_s": 0, "end_s": 4, "energy": 0.5}],
        "key_moments": [{"time_s": 0.0, "type": "explosion"}],
    }
    out = coerce_labels(data)
    assert out["structure"][0]["label"] == "instrumental"
    assert out["key_moments"][0]["type"] == "build_peak"


def test_coerce_labels_passes_through_valid_values():
    data = {
        "structure": [{"label": "chorus", "start_s": 0, "end_s": 4, "energy": 0.5}],
        "key_moments": [{"time_s": 0.0, "type": "drop"}],
    }
    out = coerce_labels(data)
    assert out["structure"][0]["label"] == "chorus"
    assert out["key_moments"][0]["type"] == "drop"


def test_coerce_labels_handles_missing_keys():
    # Empty / missing keys should not crash
    coerce_labels({})
    coerce_labels({"structure": None, "key_moments": None})


@pytest.mark.asyncio
async def test_llm_failure_degrades_gracefully(synthetic_wav, monkeypatch):
    """If the LLM call raises, analyzer returns numerical-only."""

    class BrokenOllama:
        async def generate(self, *args, **kwargs):
            raise RuntimeError("ollama down")

    analyzer = SongAnalyzer(ollama_client=BrokenOllama())
    result = await analyzer.analyze(synthetic_wav, skip_llm=False)
    assert result.numerical is not None
    assert result.interpreted is None
