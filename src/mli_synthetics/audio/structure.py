"""Structured song analysis schema (LLM output)."""
from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field


class SongStructureSection(BaseModel):
    start_s: float
    end_s: float
    label: Literal[
        "intro",
        "verse",
        "pre_chorus",
        "chorus",
        "drop",
        "build_up",
        "breakdown",
        "outro",
        "instrumental",
    ]
    energy: float = Field(ge=0.0, le=1.0)
    description: str = ""


class KeyMoment(BaseModel):
    time_s: float
    type: Literal["drop", "build_peak", "vocal_entry", "instrumental_solo"]


class LightingHints(BaseModel):
    color_palette: Literal["warm", "cool", "neutral", "contrast"]
    movement_intensity: Literal["static", "subtle", "moderate", "high", "chaotic"]
    strobe_appropriate: bool
    beam_appropriate: bool


class SongAnalysis(BaseModel):
    genre_estimate: Literal[
        "edm", "rock", "pop", "acoustic", "hip_hop", "metal", "other"
    ]
    energy_profile: Literal["low", "medium", "high", "variable"]
    mood: Literal[
        "energetic", "melancholic", "aggressive", "uplifting", "dark", "euphoric"
    ]
    structure: list[SongStructureSection]
    key_moments: list[KeyMoment] = Field(default_factory=list)
    lighting_hints: LightingHints

    # Optional pass-through metadata so downstream code has both numerical
    # and interpreted views in one object.
    duration_s: float | None = None
    tempo_bpm: float | None = None
    key: str | None = None
