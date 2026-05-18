"""Numerical feature extraction via librosa."""
from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
from pydantic import BaseModel

from mli_synthetics.errors import AudioAnalysisError
from mli_synthetics.logging_config import get_logger

logger = get_logger()


class StructuralSegment(BaseModel):
    start_s: float
    end_s: float
    label: int


class NumericalFeatures(BaseModel):
    audio_path: str
    duration_s: float
    sample_rate: int
    tempo_bpm: float
    beat_times: list[float]
    onset_times: list[float]
    onset_density_per_s: float
    rms_curve: list[float]  # 1Hz resolution
    rms_times: list[float]
    spectral_centroid_curve: list[float]
    structural_segments: list[StructuralSegment]
    key: str
    energy_peaks: list[float]
    drop_candidates: list[tuple[float, float]]

    def summary(self) -> dict[str, Any]:
        return {
            "duration_s": round(self.duration_s, 2),
            "tempo_bpm": round(self.tempo_bpm, 1),
            "key": self.key,
            "n_beats": len(self.beat_times),
            "n_onsets": len(self.onset_times),
            "onset_density_per_s": round(self.onset_density_per_s, 2),
            "n_segments": len(self.structural_segments),
            "n_energy_peaks": len(self.energy_peaks),
            "drop_candidates": [
                (round(s, 2), round(e, 2)) for s, e in self.drop_candidates
            ],
            "structural_segments": [
                {"start_s": round(s.start_s, 2), "end_s": round(s.end_s, 2), "label": s.label}
                for s in self.structural_segments
            ],
        }


_KEY_NAMES = ["C", "C#", "D", "D#", "E", "F", "F#", "G", "G#", "A", "A#", "B"]


def extract_numerical_features(audio_path: Path) -> NumericalFeatures:
    """Extract numerical features from an audio file using librosa."""
    try:
        import librosa
    except ImportError as exc:  # pragma: no cover
        raise AudioAnalysisError("librosa is required") from exc

    if not audio_path.exists():
        raise AudioAnalysisError(f"Audio file not found: {audio_path}")

    logger.info("Loading audio: {}", audio_path.name)
    try:
        y, sr = librosa.load(str(audio_path), sr=22050, mono=True)
    except Exception as exc:  # noqa: BLE001
        raise AudioAnalysisError(f"Failed to load audio: {exc}") from exc

    duration_s = float(librosa.get_duration(y=y, sr=sr))
    if duration_s < 1.0:
        raise AudioAnalysisError(f"Audio too short: {duration_s:.2f}s")

    # Tempo and beats
    tempo, beat_frames = librosa.beat.beat_track(y=y, sr=sr)
    tempo_bpm = float(np.atleast_1d(tempo)[0])
    beat_times = librosa.frames_to_time(beat_frames, sr=sr).tolist()

    # Onsets
    onset_frames = librosa.onset.onset_detect(y=y, sr=sr)
    onset_times = librosa.frames_to_time(onset_frames, sr=sr).tolist()
    onset_density = len(onset_times) / max(duration_s, 1e-6)

    # RMS curve at ~1 Hz
    hop = 512
    rms_frames = librosa.feature.rms(y=y, hop_length=hop)[0]
    rms_frame_times = librosa.frames_to_time(np.arange(len(rms_frames)), sr=sr, hop_length=hop)
    rms_curve, rms_times = _downsample_curve(rms_frames, rms_frame_times, target_hz=1.0)

    # Spectral centroid (same downsample)
    centroid = librosa.feature.spectral_centroid(y=y, sr=sr, hop_length=hop)[0]
    centroid_curve, _ = _downsample_curve(centroid, rms_frame_times, target_hz=1.0)

    # Structural segments via agglomerative clustering on chroma
    chroma = librosa.feature.chroma_cqt(y=y, sr=sr)
    n_segments = max(4, min(10, int(duration_s // 20)))
    try:
        boundary_frames = librosa.segment.agglomerative(chroma, n_segments)
        boundary_times = librosa.frames_to_time(boundary_frames, sr=sr)
        boundaries = sorted(set([0.0, *boundary_times.tolist(), duration_s]))
        segments: list[StructuralSegment] = []
        # Label = simple rotating index
        for i in range(len(boundaries) - 1):
            segments.append(
                StructuralSegment(
                    start_s=float(boundaries[i]),
                    end_s=float(boundaries[i + 1]),
                    label=i,
                )
            )
    except Exception as exc:  # noqa: BLE001
        logger.warning("Segment detection failed: {}", exc)
        segments = [StructuralSegment(start_s=0.0, end_s=duration_s, label=0)]

    # Key estimation: dominant chroma
    chroma_mean = chroma.mean(axis=1)
    key_idx = int(np.argmax(chroma_mean))
    key = _KEY_NAMES[key_idx]

    # Energy peaks (> 80th pct of RMS)
    rms_arr = np.array(rms_curve)
    if len(rms_arr) > 0:
        threshold = float(np.percentile(rms_arr, 80))
        peak_idx = np.where(rms_arr >= threshold)[0]
        energy_peaks = [float(rms_times[i]) for i in peak_idx]
    else:
        energy_peaks = []

    # Drop candidates: high-energy AND high-onset regions ≥ 4s
    drops = _find_drop_candidates(rms_curve, rms_times, onset_times, duration_s)

    return NumericalFeatures(
        audio_path=str(audio_path),
        duration_s=duration_s,
        sample_rate=sr,
        tempo_bpm=tempo_bpm,
        beat_times=beat_times,
        onset_times=onset_times,
        onset_density_per_s=onset_density,
        rms_curve=[float(v) for v in rms_curve],
        rms_times=[float(t) for t in rms_times],
        spectral_centroid_curve=[float(v) for v in centroid_curve],
        structural_segments=segments,
        key=key,
        energy_peaks=energy_peaks,
        drop_candidates=drops,
    )


def _downsample_curve(
    values: np.ndarray, times: np.ndarray, target_hz: float = 1.0
) -> tuple[list[float], list[float]]:
    if len(values) == 0:
        return [], []
    duration = float(times[-1]) if len(times) > 0 else 0.0
    n_buckets = max(1, int(duration * target_hz))
    bucket_edges = np.linspace(0, duration, n_buckets + 1)
    out_values: list[float] = []
    out_times: list[float] = []
    for i in range(n_buckets):
        mask = (times >= bucket_edges[i]) & (times < bucket_edges[i + 1])
        if not mask.any():
            continue
        out_values.append(float(values[mask].mean()))
        out_times.append(float((bucket_edges[i] + bucket_edges[i + 1]) / 2))
    return out_values, out_times


def _find_drop_candidates(
    rms_curve: list[float],
    rms_times: list[float],
    onset_times: list[float],
    duration_s: float,
    min_duration_s: float = 4.0,
) -> list[tuple[float, float]]:
    if not rms_curve:
        return []
    rms = np.array(rms_curve)
    times = np.array(rms_times)
    energy_thr = float(np.percentile(rms, 75))

    # Onset rate per 1-second window
    onset_rate = np.zeros_like(rms)
    onsets = np.array(onset_times)
    for i, t in enumerate(times):
        onset_rate[i] = ((onsets >= t - 0.5) & (onsets < t + 0.5)).sum()
    onset_thr = float(np.percentile(onset_rate, 75)) if onset_rate.size else 0.0

    mask = (rms >= energy_thr) & (onset_rate >= max(1.0, onset_thr))
    drops: list[tuple[float, float]] = []
    start: float | None = None
    for i, m in enumerate(mask):
        if m and start is None:
            start = float(times[i])
        elif not m and start is not None:
            end = float(times[i])
            if end - start >= min_duration_s:
                drops.append((start, end))
            start = None
    if start is not None and duration_s - start >= min_duration_s:
        drops.append((start, duration_s))
    return drops
