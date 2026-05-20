"""Lighting designer LLM orchestrator (chunk-based)."""
from __future__ import annotations

import asyncio
import json
import time
from pathlib import Path
from typing import Any

from pydantic import BaseModel, Field, ValidationError

from mli_synthetics.audio.structure import SongAnalysis
from mli_synthetics.errors import DesignerError
from mli_synthetics.llm.ollama_client import OllamaClient
from mli_synthetics.llm.prompts import (
    DESIGNER_CHUNK_USER_PROMPT_TEMPLATE,
    DESIGNER_SYSTEM_PROMPT,
)
from mli_synthetics.logging_config import get_logger
from mli_synthetics.stage.fixtures import (
    Fixture,
    FixtureCategory,
    Position,
    StageLayout,
)

logger = get_logger()


# ---------------------------------------------------------------------------
# Cue list models
# ---------------------------------------------------------------------------
class Movement(BaseModel):
    # NOTE: pan/tilt accept any float on input. Out-of-range values are
    # interpreted as degrees and normalized to [-1.0, 1.0] in
    # post-processing (see `_normalize_axis`).
    pan: float = 0.0
    tilt: float = 0.5
    movement_type: str = "static"
    speed: str = "slow"


class FixtureState(BaseModel):
    fixture_id: str
    intensity: float = Field(ge=0.0, le=1.0)
    color: tuple[int, int, int]
    movement: Movement = Field(default_factory=Movement)
    effect: str = "none"
    effect_speed_hz: float = 0.0


class Cue(BaseModel):
    time_s: float = Field(ge=0.0)
    duration_s: float = Field(gt=0.0)
    section_label: str
    description: str = ""
    fixture_states: list[FixtureState]


class CueListMetadata(BaseModel):
    designer_notes: str = ""
    color_palette: list[str] = Field(default_factory=list)
    intensity_arc: str = ""


class CueList(BaseModel):
    metadata: CueListMetadata
    cues: list[Cue]


class _ChunkCues(BaseModel):
    """Minimal schema the LLM returns for a single chunk."""

    cues: list[Cue]


# ---------------------------------------------------------------------------
MAX_CHUNK_S = 30.0
CONTEXT_PAD_S = 2.0
# Soft timeout for the first per-chunk LLM call. If it exceeds this, the
# designer falls back to two parallel sub-calls each with half the zones.
FIRST_CALL_SOFT_TIMEOUT_S = 120.0


# Defaults inserted for any missing required field on a fixture_state.
# Applied BEFORE Pydantic validation so small models that omit fields
# don't crash the parser.
DEFAULT_FIXTURE_STATE: dict[str, Any] = {
    "intensity": 0.0,
    "color": [0, 0, 0],
    "movement": {
        "pan": 0.0,
        "tilt": 0.0,
        "movement_type": "static",
        "speed": "slow",
    },
    "effect": "none",
    "effect_speed_hz": 0.0,
}


def fill_missing_cue_fields(parsed_dict: dict) -> dict:
    """Fill missing required fields on each cue with safe defaults.

    Some small models drop `section_label`, `description`, or `duration_s`
    on individual cues. Pydantic would reject the cue; this helper repairs
    it in place before validation.
    """
    for i, cue in enumerate(parsed_dict.get("cues", []) or []):
        if not isinstance(cue, dict):
            continue
        if "section_label" not in cue or not cue["section_label"]:
            cue["section_label"] = "instrumental"
        if "description" not in cue or cue["description"] is None:
            cue["description"] = f"Cue {i}"
        if "duration_s" not in cue or cue["duration_s"] is None:
            cue["duration_s"] = 4.0
    return parsed_dict


# ---------------------------------------------------------------------------
# Six-zone grouping (Fix 1).
#
# Stages over `max_fixtures_for_llm` (default 12) are grouped into AT MOST
# 6 fixed zones. Empty zones are dropped, so the LLM sees between 1 and 6
# entries regardless of stage size.
# ---------------------------------------------------------------------------
SIX_ZONE_ORDER: tuple[str, ...] = (
    "zone_back",
    "zone_mid",
    "zone_front",
    "zone_floor",
    "zone_sides",
    "zone_special",
)

_SPECIAL_CATEGORIES: set[FixtureCategory] = {
    FixtureCategory.HAZER,
    FixtureCategory.CO2_JET,
    FixtureCategory.PYRO,
    FixtureCategory.LASER,
}


def _zone_for_fixture(f: Fixture) -> str:
    """Map a fixture to one of the six fixed zones."""
    if f.category in _SPECIAL_CATEGORIES:
        return "zone_special"
    # Blinder always rides with front, regardless of physical position
    if f.category == FixtureCategory.BLINDER:
        return "zone_front"
    pos = f.position
    if pos in (Position.BACK_TRUSS, Position.BACKLINE):
        return "zone_back"
    if pos == Position.MID_TRUSS:
        return "zone_mid"
    if pos in (Position.FRONT_TRUSS, Position.AUDIENCE_FACING):
        return "zone_front"
    if pos == Position.STAGE_FLOOR:
        return "zone_floor"
    if pos in (Position.SIDE_LEFT, Position.SIDE_RIGHT):
        return "zone_sides"
    return "zone_floor"


def build_six_zone_view(
    layout: StageLayout, max_fixtures: int = 12
) -> tuple[dict, dict[str, list[str]]]:
    """Pass-through view for ≤max_fixtures stages; otherwise group into
    at most 6 fixed zones (zone_back/mid/front/floor/sides/special).

    Returns `(stage_view, zone_to_fixtures)` exactly like the original
    `build_llm_view`. The zone_map is empty when no grouping was needed.
    """
    base: dict[str, Any] = {
        "width_m": layout.width_m,
        "depth_m": layout.depth_m,
        "height_m": layout.height_m,
        "venue_size": layout.venue_size.value,
        "target_genre": layout.target_genre,
    }

    if len(layout.fixtures) <= max_fixtures:
        base["fixtures"] = [
            {
                "fixture_id": f.fixture_id,
                "category": f.category.value,
                "position": f.position.value,
                "x": f.x,
                "y": f.y,
                "z": f.z,
            }
            for f in layout.fixtures
        ]
        return base, {}

    groups: dict[str, list[Fixture]] = {z: [] for z in SIX_ZONE_ORDER}
    for f in layout.fixtures:
        groups[_zone_for_fixture(f)].append(f)

    zone_entries: list[dict] = []
    zone_to_fixtures: dict[str, list[str]] = {}
    for zone_id in SIX_ZONE_ORDER:
        items = groups[zone_id]
        if not items:
            continue
        categories = sorted({i.category.value for i in items})
        zone_entries.append(
            {
                "fixture_id": zone_id,
                "categories": categories,
                "count": len(items),
                "note": f"This is a {zone_id.replace('zone_', '')} zone "
                f"containing {len(items)} fixtures of types {', '.join(categories)}; "
                f"control as a group.",
            }
        )
        zone_to_fixtures[zone_id] = [f.fixture_id for f in items]

    base["fixtures"] = zone_entries
    base["zoning_applied"] = True
    base["zone_count"] = len(zone_entries)
    return base, zone_to_fixtures


def fill_missing_fixture_fields(parsed_dict: dict) -> dict:
    """Fill missing required fields on every fixture_state with defaults.

    Mutates in place AND returns the same dict for chaining. Also coerces
    `color` from a string (LLMs occasionally write `"blue"`) to a safe
    white RGB array.
    """
    for cue in parsed_dict.get("cues", []) or []:
        for state in cue.get("fixture_states", []) or []:
            if not isinstance(state, dict):
                continue
            for key, default in DEFAULT_FIXTURE_STATE.items():
                if key not in state:
                    state[key] = default if not isinstance(default, dict) else dict(default)
            # Color: tolerate strings/null/wrong shapes
            color = state.get("color")
            if isinstance(color, str) or color is None or not _is_rgb_triple(color):
                state["color"] = [255, 255, 255]
            # Movement: tolerate it being something weird
            mv = state.get("movement")
            if not isinstance(mv, dict):
                state["movement"] = dict(DEFAULT_FIXTURE_STATE["movement"])
            else:
                for mk, mv_default in DEFAULT_FIXTURE_STATE["movement"].items():
                    if mk not in mv:
                        mv[mk] = mv_default
    return parsed_dict


def _is_rgb_triple(value: Any) -> bool:
    if not isinstance(value, (list, tuple)) or len(value) != 3:
        return False
    return all(isinstance(v, (int, float)) for v in value)


class LightingDesignerLLM:
    def __init__(
        self,
        settings: Any | None = None,
        ollama: OllamaClient | None = None,
        knowledge_context: str | None = None,
    ):
        if settings is None:
            from mli_synthetics.settings import get_settings

            settings = get_settings()
        self.settings = settings
        if ollama is None:
            from mli_synthetics.llm import get_llm_client

            ollama = get_llm_client(settings)
        self.ollama = ollama
        self._knowledge_context = knowledge_context

    # ------------------------------------------------------------------
    async def design_show(
        self,
        stage: StageLayout,
        song_analysis: SongAnalysis,
        max_retries: int = 1,
    ) -> CueList:
        """Generate a validated cue list via chunked LLM calls."""
        if self._knowledge_context is None:
            self._knowledge_context = self._load_reduced_knowledge()

        system_prompt = DESIGNER_SYSTEM_PROMPT.format(
            knowledge_context=self._knowledge_context
        )

        chunks = self._compute_chunks(song_analysis, MAX_CHUNK_S)
        if not chunks:
            raise DesignerError("Song analysis has no usable duration for chunking")
        logger.info(
            "Designer: splitting song ({:.1f}s) into {} chunks",
            song_analysis.duration_s or 0.0,
            len(chunks),
        )

        # Bug 3: cap fixture count exposed to the LLM by grouping into zones.
        max_for_llm = getattr(self.settings, "max_fixtures_for_llm", 12)
        stage_view, zone_map = build_six_zone_view(stage, max_fixtures=max_for_llm)
        if zone_map:
            logger.info(
                "Stage has {} fixtures > {}; grouping into {} zones for LLM",
                len(stage.fixtures),
                max_for_llm,
                len(zone_map),
            )

        all_cues: list[Cue] = []
        failed: list[int] = []
        previous_last_cue: Cue | None = None

        for i, (start_s, end_s) in enumerate(chunks, start=1):
            t0 = time.perf_counter()
            try:
                chunk_cues = await self._design_chunk(
                    stage=stage,
                    stage_view=stage_view,
                    zone_map=zone_map,
                    song_analysis=song_analysis,
                    chunk_start_s=start_s,
                    chunk_end_s=end_s,
                    previous_last_cue=previous_last_cue,
                    system_prompt=system_prompt,
                    max_retries=max_retries,
                )
                # Shift relative times to absolute
                for c in chunk_cues:
                    c.time_s = round(c.time_s + start_s, 3)
                all_cues.extend(chunk_cues)
                if chunk_cues:
                    previous_last_cue = chunk_cues[-1]
                elapsed = time.perf_counter() - t0
                logger.info(
                    "Chunk {}/{} done ({:.1f}s, {} cues)",
                    i,
                    len(chunks),
                    elapsed,
                    len(chunk_cues),
                )
            except Exception as exc:  # noqa: BLE001
                elapsed = time.perf_counter() - t0
                logger.error(
                    "Chunk {}/{} failed after {:.1f}s: {}",
                    i,
                    len(chunks),
                    elapsed,
                    exc,
                )
                failed.append(i)

        if failed:
            logger.warning(
                "Designer finished with {} failed chunk(s): {}",
                len(failed),
                failed,
            )

        if not all_cues:
            raise DesignerError(
                f"All {len(chunks)} chunks failed; no cues produced"
            )

        # Re-sort defensively (chunks already in order, but be safe)
        all_cues.sort(key=lambda c: c.time_s)

        metadata = CueListMetadata(
            designer_notes=(
                f"Generated via chunked pipeline: {len(chunks)} chunks, "
                f"{len(failed)} failed. Mood={song_analysis.mood}, "
                f"genre={song_analysis.genre_estimate}."
            ),
            color_palette=[song_analysis.lighting_hints.color_palette],
            intensity_arc=f"Energy profile: {song_analysis.energy_profile}",
        )
        return CueList(metadata=metadata, cues=all_cues)

    # ------------------------------------------------------------------
    async def _design_chunk(
        self,
        stage: StageLayout,
        stage_view: dict | None = None,
        zone_map: dict[str, list[str]] | None = None,
        *,
        song_analysis: SongAnalysis,
        chunk_start_s: float,
        chunk_end_s: float,
        previous_last_cue: Cue | None,
        system_prompt: str,
        max_retries: int,
    ) -> list[Cue]:
        if stage_view is None:
            max_for_llm = getattr(self.settings, "max_fixtures_for_llm", 12)
            stage_view, zone_map = build_six_zone_view(stage, max_fixtures=max_for_llm)
        zone_map = zone_map or {}
        chunk_duration = chunk_end_s - chunk_start_s
        # +/- context window (informational only, used for prompt narrative)
        ctx_start = max(0.0, chunk_start_s - CONTEXT_PAD_S)
        ctx_end = chunk_end_s + CONTEXT_PAD_S

        # Sections overlapping the *context* window
        relevant = [
            {
                "start_s": round(s.start_s, 2),
                "end_s": round(s.end_s, 2),
                "label": s.label,
                "energy": s.energy,
                "description": s.description,
            }
            for s in song_analysis.structure
            if s.end_s > ctx_start and s.start_s < ctx_end
        ]

        previous_state_json = (
            json.dumps(previous_last_cue.model_dump(), indent=2)
            if previous_last_cue is not None
            else "null  (this is the FIRST chunk - start from blackout)"
        )

        min_cues = max(1, int(chunk_duration // 2))
        max_cues = max(min_cues + 1, int(chunk_duration // 1))

        user_prompt = DESIGNER_CHUNK_USER_PROMPT_TEMPLATE.format(
            stage_json=json.dumps(stage_view, indent=2),
            song_analysis_json=json.dumps(song_analysis.model_dump(), indent=2),
            chunk_start_s=round(chunk_start_s, 2),
            chunk_end_s=round(chunk_end_s, 2),
            chunk_duration_s=round(chunk_duration, 2),
            relevant_sections_json=json.dumps(relevant, indent=2),
            previous_state_json=previous_state_json,
            min_cues=min_cues,
            max_cues=max_cues,
        )

        last_error: str | None = None
        attempts = max_retries + 1
        for attempt in range(1, attempts + 1):
            prompt = user_prompt
            if last_error:
                prompt += (
                    f"\n\nPrevious attempt failed because: {last_error}\n"
                    "Fix this and produce a valid chunk."
                )
            try:
                # Fix 3: cap the first call at 120s; on timeout, fall back to
                # two parallel sub-calls each handling half the zones.
                try:
                    raw = await asyncio.wait_for(
                        self.ollama.generate(
                            model=self.settings.ollama_model_designer,
                            prompt=prompt,
                            system=system_prompt,
                            temperature=self.settings.designer_temperature,
                            max_tokens=self.settings.max_tokens_designer,
                            json_mode=True,
                        ),
                        timeout=FIRST_CALL_SOFT_TIMEOUT_S,
                    )
                except asyncio.TimeoutError:
                    logger.warning(
                        "Chunk @ {:.1f}s attempt {}: first call > {}s; "
                        "falling back to split-zone calls",
                        chunk_start_s,
                        attempt,
                        FIRST_CALL_SOFT_TIMEOUT_S,
                    )
                    processed = await self._design_chunk_split_zones(
                        stage=stage,
                        stage_view=stage_view,
                        zone_map=zone_map,
                        song_analysis=song_analysis,
                        chunk_start_s=chunk_start_s,
                        chunk_end_s=chunk_end_s,
                        chunk_duration=chunk_duration,
                        previous_last_cue=previous_last_cue,
                        system_prompt=system_prompt,
                        relevant=relevant,
                    )
                    self._validate_chunk_lenient(processed, chunk_duration)
                    return processed

                data = json.loads(raw)
                # Fill missing required fields BEFORE Pydantic validation -
                # small models routinely drop both fixture and cue fields.
                data = fill_missing_fixture_fields(data)
                data = fill_missing_cue_fields(data)
                chunk = _ChunkCues.model_validate(data)
                # Post-process: expand zones, normalize pan/tilt, auto-fill
                # missing fixtures, drop unknown IDs. All operations are
                # forgiving (warnings only) so small models don't fail.
                processed = [
                    self._postprocess_cue(c, stage, zone_map, chunk_start_s)
                    for c in chunk.cues
                ]
                # Lenient structural validation (monotonic, durations, time window)
                self._validate_chunk_lenient(processed, chunk_duration)
                return processed
            except (json.JSONDecodeError, ValidationError, DesignerError) as exc:
                last_error = str(exc)[:500]
                logger.warning(
                    "Chunk attempt {}/{} validation failed: {}",
                    attempt,
                    attempts,
                    last_error,
                )
        raise DesignerError(f"Chunk failed after {attempts} attempts: {last_error}")

    # ------------------------------------------------------------------
    async def _design_chunk_split_zones(
        self,
        *,
        stage: StageLayout,
        stage_view: dict,
        zone_map: dict[str, list[str]],
        song_analysis: SongAnalysis,
        chunk_start_s: float,
        chunk_end_s: float,
        chunk_duration: float,
        previous_last_cue: Cue | None,
        system_prompt: str,
        relevant: list,
    ) -> list[Cue]:
        """Split-zone fallback: two parallel LLM calls, each on half the zones.

        Each sub-call produces partial cues (only its zones populated).
        Cues from both halves are merged by time window so the resulting
        timeline carries full state per cue; the missing fixtures are
        autofilled at the end.
        """
        zone_ids = list(zone_map.keys())
        if len(zone_ids) < 2:
            raise DesignerError(
                "Cannot split a single-zone chunk; the model is probably hung"
            )
        half = (len(zone_ids) + 1) // 2
        halves = [zone_ids[:half], zone_ids[half:]]
        logger.warning(
            "Chunk @ {:.1f}s: splitting {} zones into 2 sub-calls of sizes {}",
            chunk_start_s,
            len(zone_ids),
            [len(h) for h in halves],
        )

        sub_calls = [
            self._call_zone_subset(
                zone_subset=h,
                full_view=stage_view,
                full_zone_map=zone_map,
                stage=stage,
                song_analysis=song_analysis,
                chunk_start_s=chunk_start_s,
                chunk_end_s=chunk_end_s,
                chunk_duration=chunk_duration,
                previous_last_cue=previous_last_cue,
                system_prompt=system_prompt,
                relevant=relevant,
            )
            for h in halves
            if h
        ]
        sub_results = await asyncio.gather(*sub_calls, return_exceptions=True)

        partial_cues: list[Cue] = []
        for r in sub_results:
            if isinstance(r, Exception):
                logger.error("Zone-split sub-call failed: {}", r)
                continue
            partial_cues.extend(r)
        if not partial_cues:
            raise DesignerError("All sub-calls failed in zone-split fallback")

        merged = _merge_cues_by_time(partial_cues, tolerance_s=0.5)

        # Final autofill so every cue covers every fixture
        stage_ids = {f.fixture_id for f in stage.fixtures}
        for cue in merged:
            present = {s.fixture_id for s in cue.fixture_states}
            missing = stage_ids - present
            if missing:
                for fid in sorted(missing):
                    cue.fixture_states.append(_default_state(fid))
        return merged

    # ------------------------------------------------------------------
    async def _call_zone_subset(
        self,
        *,
        zone_subset: list[str],
        full_view: dict,
        full_zone_map: dict[str, list[str]],
        stage: StageLayout,
        song_analysis: SongAnalysis,
        chunk_start_s: float,
        chunk_end_s: float,
        chunk_duration: float,
        previous_last_cue: Cue | None,
        system_prompt: str,
        relevant: list,
    ) -> list[Cue]:
        """One LLM call covering only a subset of zones. Returns
        partially-populated cues (zones expanded, pan/tilt normalized,
        unknowns dropped, NO autofill)."""
        sub_view = {k: v for k, v in full_view.items() if k != "fixtures"}
        sub_view["fixtures"] = [
            e for e in full_view["fixtures"] if e["fixture_id"] in zone_subset
        ]
        sub_view["zoning_applied"] = True
        sub_view["zone_count"] = len(sub_view["fixtures"])
        sub_zone_map = {k: full_zone_map[k] for k in zone_subset}

        prev_state_json = (
            json.dumps(previous_last_cue.model_dump(), indent=2)
            if previous_last_cue is not None
            else "null  (this is the FIRST chunk - start from blackout)"
        )
        min_cues = max(1, int(chunk_duration // 2))
        max_cues = max(min_cues + 1, int(chunk_duration // 1))

        user_prompt = DESIGNER_CHUNK_USER_PROMPT_TEMPLATE.format(
            stage_json=json.dumps(sub_view, indent=2),
            song_analysis_json=json.dumps(song_analysis.model_dump(), indent=2),
            chunk_start_s=round(chunk_start_s, 2),
            chunk_end_s=round(chunk_end_s, 2),
            chunk_duration_s=round(chunk_duration, 2),
            relevant_sections_json=json.dumps(relevant, indent=2),
            previous_state_json=prev_state_json,
            min_cues=min_cues,
            max_cues=max_cues,
        )

        raw = await self.ollama.generate(
            model=self.settings.ollama_model_designer,
            prompt=user_prompt,
            system=system_prompt,
            temperature=self.settings.designer_temperature,
            max_tokens=self.settings.max_tokens_designer,
            json_mode=True,
        )
        data = json.loads(raw)
        data = fill_missing_fixture_fields(data)
        data = fill_missing_cue_fields(data)
        chunk = _ChunkCues.model_validate(data)
        # Partial post-process: expand sub-zones + normalize pan/tilt;
        # NO autofill so we can merge with the other half first.
        return [
            _expand_and_normalize_cue(c, stage, sub_zone_map)
            for c in chunk.cues
        ]

    # ------------------------------------------------------------------
    @staticmethod
    def _validate_chunk_lenient(cues: list[Cue], chunk_duration_s: float) -> None:
        """Lenient validation: only structural checks.

        Missing/unknown fixtures are handled by `_postprocess_cue` and
        no longer raise here.
        """
        if not cues:
            raise DesignerError("Chunk cue list is empty")
        prev_t = -1.0
        for i, cue in enumerate(cues):
            if cue.time_s < prev_t:
                raise DesignerError(
                    f"Chunk cue {i} time_s={cue.time_s} not monotonic (prev={prev_t})"
                )
            if cue.duration_s <= 0:
                raise DesignerError(f"Chunk cue {i} has non-positive duration")
            if cue.time_s > chunk_duration_s + 1.0:
                raise DesignerError(
                    f"Chunk cue {i} time_s={cue.time_s} exceeds chunk duration "
                    f"{chunk_duration_s} (chunk-relative times required)"
                )
            prev_t = cue.time_s

    # ------------------------------------------------------------------
    @staticmethod
    def _postprocess_cue(
        cue: Cue,
        stage: StageLayout,
        zone_map: dict[str, list[str]],
        chunk_start_s: float,
    ) -> Cue:
        """Expand zones, normalize pan/tilt, drop unknowns, auto-fill missing.

        All operations log warnings; none raise. This is the heart of the
        small-model-robustness work.
        """
        stage_ids = {f.fixture_id for f in stage.fixtures}

        # 1. Expand zone-level states onto the underlying fixtures
        expanded: list[FixtureState] = []
        for s in cue.fixture_states:
            targets = zone_map.get(s.fixture_id)
            if targets is None:
                expanded.append(s)
            else:
                for fid in targets:
                    expanded.append(s.model_copy(update={"fixture_id": fid}))

        # 2. Normalize pan/tilt on each state
        for s in expanded:
            s.movement.pan = _normalize_axis(s.movement.pan)
            s.movement.tilt = _normalize_axis(s.movement.tilt)

        # 3. Drop unknown fixture IDs (silently warn); keep first occurrence
        kept: list[FixtureState] = []
        seen: set[str] = set()
        unknown: list[str] = []
        for s in expanded:
            if s.fixture_id not in stage_ids:
                unknown.append(s.fixture_id)
                continue
            if s.fixture_id in seen:
                continue
            kept.append(s)
            seen.add(s.fixture_id)
        if unknown:
            sample = list(dict.fromkeys(unknown))[:5]
            logger.warning(
                "Cue @ {:.2f}s (chunk-relative): dropped {} unknown fixture id(s) e.g. {}",
                cue.time_s,
                len(unknown),
                sample,
            )

        # 4. Auto-fill missing fixtures with safe defaults
        missing = stage_ids - seen
        if missing:
            logger.warning(
                "Cue @ {:.2f}s (chunk-relative, abs ~{:.2f}s): "
                "auto-filling {}/{} missing fixtures with defaults (off)",
                cue.time_s,
                cue.time_s + chunk_start_s,
                len(missing),
                len(stage_ids),
            )
            for fid in sorted(missing):
                kept.append(_default_state(fid))

        cue.fixture_states = kept
        return cue

    # ------------------------------------------------------------------
    @staticmethod
    def _validate_constraints(
        cue_list: CueList,
        stage: StageLayout,
        song_analysis: SongAnalysis,
    ) -> None:
        """Validate a complete (already-assembled) cue list.

        Lenient: missing/unknown fixtures only emit warnings (those are
        auto-handled upstream in `_postprocess_cue`). Hard failures are
        reserved for true structural problems (empty list, non-monotonic
        time, non-positive durations, gross coverage gaps).
        """
        if not cue_list.cues:
            raise DesignerError("Cue list is empty")
        stage_ids = {f.fixture_id for f in stage.fixtures}
        duration = song_analysis.duration_s or (
            cue_list.cues[-1].time_s + cue_list.cues[-1].duration_s
        )
        prev_time = -1.0
        for i, cue in enumerate(cue_list.cues):
            if cue.time_s < prev_time:
                raise DesignerError(
                    f"Cue {i} time_s={cue.time_s} not monotonic (prev={prev_time})"
                )
            if cue.duration_s <= 0:
                raise DesignerError(f"Cue {i} has non-positive duration")
            prev_time = cue.time_s
            cue_ids = {s.fixture_id for s in cue.fixture_states}
            missing = stage_ids - cue_ids
            if missing:
                logger.warning(
                    "Cue {} missing {}/{} fixtures (would have been auto-filled)",
                    i,
                    len(missing),
                    len(stage_ids),
                )
            unknown = cue_ids - stage_ids
            if unknown:
                logger.warning(
                    "Cue {} references {} unknown fixture id(s)",
                    i,
                    len(unknown),
                )
        first_t = cue_list.cues[0].time_s
        last = cue_list.cues[-1]
        last_t = last.time_s + last.duration_s
        if first_t > 2.0:
            raise DesignerError(f"First cue starts at {first_t}s; should be near 0")
        if duration and last_t < duration - 5.0:
            raise DesignerError(
                f"Cue list ends at {last_t:.1f}s but song duration is {duration:.1f}s"
            )

    # ------------------------------------------------------------------
    @staticmethod
    def _compute_chunks(
        analysis: SongAnalysis, max_chunk_s: float
    ) -> list[tuple[float, float]]:
        """Split [0, duration] into chunks aligned to structural segments.

        Greedy: extend the current chunk through segments until adding
        the next would exceed `max_chunk_s`; then close at the segment
        boundary. Segments longer than `max_chunk_s` are hard-split.
        """
        segments = sorted(analysis.structure, key=lambda s: s.start_s)
        duration = analysis.duration_s or (segments[-1].end_s if segments else 0.0)
        if duration <= 0:
            return []

        if not segments:
            # Fallback: uniform chunks
            chunks: list[tuple[float, float]] = []
            t = 0.0
            while t < duration:
                end = min(t + max_chunk_s, duration)
                chunks.append((round(t, 3), round(end, 3)))
                t = end
            return chunks

        chunks = []
        current_start = 0.0
        for seg in segments:
            # If this segment's end pushes us past the budget, close at its start
            if seg.end_s - current_start > max_chunk_s:
                if seg.start_s > current_start + 0.1:
                    chunks.append((round(current_start, 3), round(seg.start_s, 3)))
                    current_start = seg.start_s
                # If the segment itself is longer than max, hard-split
                while seg.end_s - current_start > max_chunk_s:
                    cut = current_start + max_chunk_s
                    chunks.append((round(current_start, 3), round(cut, 3)))
                    current_start = cut
        if current_start < duration - 0.1:
            chunks.append((round(current_start, 3), round(duration, 3)))
        return chunks

    # ------------------------------------------------------------------
    def _load_reduced_knowledge(self) -> str:
        """Reduced KB: only fixture_library.md, no PDFs.

        Falls back to a tiny string if the file is missing.
        """
        path: Path = self.settings.knowledge_base_dir / "fixture_library.md"
        if path.exists():
            try:
                text = path.read_text(encoding="utf-8")
                logger.info(
                    "Designer using reduced KB (fixture_library.md only, {} chars)",
                    len(text),
                )
                return f"====== SOURCE: fixture_library.md ======\n{text}"
            except Exception as exc:  # noqa: BLE001
                logger.warning("Failed reading fixture_library.md: {}", exc)
        logger.warning("fixture_library.md missing; using minimal fallback context")
        return (
            "Use general professional knowledge of stage lighting. "
            "Common fixture categories: moving beams, spots, washes, LED pars, "
            "blinders, strobes, pixel bars, followspots, lasers, hazers, CO2, pyro."
        )


# ---------------------------------------------------------------------------
# Module-level helpers
# ---------------------------------------------------------------------------
def _normalize_axis(value: float) -> float:
    """Normalize a pan/tilt value to [-1.0, 1.0].

    If `value` is already in [-1, 1], return as-is. Otherwise interpret
    as a degree value over a typical 270° pan range and apply:
        normalized = (degrees / 270) * 2 - 1
    Finally clamp to [-1, 1].
    """
    v = float(value)
    if -1.0 <= v <= 1.0:
        return v
    normalized = (v / 270.0) * 2.0 - 1.0
    if normalized > 1.0:
        return 1.0
    if normalized < -1.0:
        return -1.0
    return normalized


def _expand_and_normalize_cue(
    cue: Cue,
    stage: StageLayout,
    zone_map: dict[str, list[str]],
) -> Cue:
    """Like `_postprocess_cue` but WITHOUT the auto-fill step.

    Used by the zone-split fallback so cues from both halves can be
    merged before missing-fixture autofill runs at the end.
    """
    stage_ids = {f.fixture_id for f in stage.fixtures}
    expanded: list[FixtureState] = []
    for s in cue.fixture_states:
        targets = zone_map.get(s.fixture_id)
        if targets is None:
            expanded.append(s)
        else:
            for fid in targets:
                expanded.append(s.model_copy(update={"fixture_id": fid}))
    for s in expanded:
        s.movement.pan = _normalize_axis(s.movement.pan)
        s.movement.tilt = _normalize_axis(s.movement.tilt)
    kept: list[FixtureState] = []
    seen: set[str] = set()
    for s in expanded:
        if s.fixture_id not in stage_ids or s.fixture_id in seen:
            continue
        kept.append(s)
        seen.add(s.fixture_id)
    cue.fixture_states = kept
    return cue


def _merge_cues_by_time(cues: list[Cue], tolerance_s: float = 0.5) -> list[Cue]:
    """Merge cues whose `time_s` are within `tolerance_s` of each other.

    For merged cues, fixture_states are combined keyed by `fixture_id`
    (later state wins for collisions). duration_s is the max of the two.
    Used by the zone-split fallback to reconcile two parallel sub-call
    timelines into a single coherent cue stream.
    """
    if not cues:
        return []
    sorted_cues = sorted(cues, key=lambda c: c.time_s)
    merged: list[Cue] = [sorted_cues[0]]
    for c in sorted_cues[1:]:
        last = merged[-1]
        if abs(c.time_s - last.time_s) <= tolerance_s:
            existing = {s.fixture_id: s for s in last.fixture_states}
            for s in c.fixture_states:
                existing[s.fixture_id] = s
            last.fixture_states = list(existing.values())
            last.duration_s = max(last.duration_s, c.duration_s)
            if not last.description and c.description:
                last.description = c.description
        else:
            merged.append(c)
    return merged


def _default_state(fixture_id: str) -> FixtureState:
    """Safe-default fixture state used for auto-fill (fixture is off)."""
    return FixtureState(
        fixture_id=fixture_id,
        intensity=0.0,
        color=(0, 0, 0),
        movement=Movement(pan=0.0, tilt=0.0, movement_type="static", speed="slow"),
        effect="none",
        effect_speed_hz=0.0,
    )
