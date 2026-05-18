"""Tests for the designer LLM and cue-list validation."""
from __future__ import annotations

import asyncio
import json
from typing import Any

import pytest

from mli_synthetics.audio.structure import (
    LightingHints,
    SongAnalysis,
    SongStructureSection,
)
from mli_synthetics.errors import DesignerError
from mli_synthetics.llm.designer import (
    Cue,
    CueList,
    CueListMetadata,
    FixtureState,
    LightingDesignerLLM,
    Movement,
)
from mli_synthetics.stage.fixtures import (
    Fixture,
    FixtureCategory,
    Position,
    StageLayout,
    VenueSize,
)


@pytest.fixture
def small_stage() -> StageLayout:
    return StageLayout(
        width_m=8.0,
        depth_m=5.0,
        height_m=4.0,
        venue_size=VenueSize.SMALL_CLUB,
        target_genre="pop",
        fixtures=[
            Fixture(
                fixture_id="a",
                category=FixtureCategory.LED_PAR,
                position=Position.STAGE_FLOOR,
                x=0.0, y=2.5, z=0.1,
            ),
            Fixture(
                fixture_id="b",
                category=FixtureCategory.MOVING_SPOT,
                position=Position.FRONT_TRUSS,
                x=-2.0, y=0.2, z=3.5,
            ),
        ],
    )


@pytest.fixture
def song_analysis() -> SongAnalysis:
    return SongAnalysis(
        genre_estimate="pop",
        energy_profile="medium",
        mood="uplifting",
        structure=[
            SongStructureSection(start_s=0.0, end_s=60.0, label="verse", energy=0.4),
        ],
        key_moments=[],
        lighting_hints=LightingHints(
            color_palette="warm",
            movement_intensity="subtle",
            strobe_appropriate=False,
            beam_appropriate=False,
        ),
        duration_s=60.0,
    )


@pytest.fixture
def short_song_analysis() -> SongAnalysis:
    """Short enough to produce a single chunk (≤ MAX_CHUNK_S)."""
    return SongAnalysis(
        genre_estimate="pop",
        energy_profile="medium",
        mood="uplifting",
        structure=[
            SongStructureSection(start_s=0.0, end_s=20.0, label="verse", energy=0.4),
        ],
        key_moments=[],
        lighting_hints=LightingHints(
            color_palette="warm",
            movement_intensity="subtle",
            strobe_appropriate=False,
            beam_appropriate=False,
        ),
        duration_s=20.0,
    )


def _state(fid: str) -> FixtureState:
    return FixtureState(
        fixture_id=fid,
        intensity=0.5,
        color=(255, 200, 100),
        movement=Movement(),
        effect="none",
        effect_speed_hz=0.0,
    )


def _valid_cuelist(fixture_ids: list[str], duration_s: float = 60.0) -> CueList:
    cues: list[Cue] = []
    t = 0.0
    while t < duration_s:
        cues.append(
            Cue(
                time_s=t,
                duration_s=4.0,
                section_label="verse",
                description="x",
                fixture_states=[_state(f) for f in fixture_ids],
            )
        )
        t += 4.0
    return CueList(metadata=CueListMetadata(designer_notes="test"), cues=cues)


def test_valid_cuelist_passes(small_stage, song_analysis):
    cl = _valid_cuelist([f.fixture_id for f in small_stage.fixtures])
    # Should not raise
    LightingDesignerLLM._validate_constraints(cl, small_stage, song_analysis)


def test_missing_fixture_does_not_raise(small_stage, song_analysis):
    """Bug 1: missing fixtures are auto-filled upstream; validator only warns."""
    cl = _valid_cuelist(["a"])  # missing 'b'
    # Should NOT raise - validator is now lenient on missing fixtures
    LightingDesignerLLM._validate_constraints(cl, small_stage, song_analysis)


def test_non_monotonic_time_raises(small_stage, song_analysis):
    ids = [f.fixture_id for f in small_stage.fixtures]
    cl = CueList(
        metadata=CueListMetadata(),
        cues=[
            Cue(time_s=10.0, duration_s=4.0, section_label="v", description="",
                fixture_states=[_state(i) for i in ids]),
            Cue(time_s=5.0, duration_s=4.0, section_label="v", description="",
                fixture_states=[_state(i) for i in ids]),
        ],
    )
    with pytest.raises(DesignerError, match="monotonic"):
        LightingDesignerLLM._validate_constraints(cl, small_stage, song_analysis)


def test_empty_cuelist_raises(small_stage, song_analysis):
    cl = CueList(metadata=CueListMetadata(), cues=[])
    with pytest.raises(DesignerError, match="empty"):
        LightingDesignerLLM._validate_constraints(cl, small_stage, song_analysis)


def test_unknown_fixture_does_not_raise(small_stage, song_analysis):
    """Bug 1: unknown IDs are dropped upstream; validator only warns."""
    ids = [f.fixture_id for f in small_stage.fixtures] + ["ghost"]
    cl = _valid_cuelist(ids)
    # Should NOT raise - validator is now lenient on unknown fixtures
    LightingDesignerLLM._validate_constraints(cl, small_stage, song_analysis)


def test_short_coverage_raises(small_stage, song_analysis):
    ids = [f.fixture_id for f in small_stage.fixtures]
    cl = _valid_cuelist(ids, duration_s=20.0)  # song is 60s
    with pytest.raises(DesignerError, match="ends at"):
        LightingDesignerLLM._validate_constraints(cl, small_stage, song_analysis)


# ---------------------------------------------------------------------------
class _StubOllama:
    """Stub Ollama client whose `generate` returns canned JSON.

    Tracks attempt count so retry logic can be tested.
    """

    def __init__(self, responses: list[str]):
        self.responses = responses
        self.calls = 0

    async def generate(self, **kwargs):
        idx = min(self.calls, len(self.responses) - 1)
        self.calls += 1
        return self.responses[idx]


@pytest.mark.asyncio
async def test_designer_retries_on_invalid_then_succeeds(small_stage, short_song_analysis):
    ids = [f.fixture_id for f in small_stage.fixtures]
    # Single chunk (20s ≤ MAX_CHUNK_S=30). Chunk schema is {"cues": [...]}.
    good = {"cues": _valid_cuelist(ids, duration_s=20.0).model_dump()["cues"]}
    bad = {"cues": []}
    stub = _StubOllama(responses=[json.dumps(bad), json.dumps(good)])
    designer = LightingDesignerLLM(ollama=stub, knowledge_context="(stub)")
    result = await designer.design_show(small_stage, short_song_analysis, max_retries=1)
    assert stub.calls == 2
    assert len(result.cues) > 0


@pytest.mark.asyncio
async def test_designer_fails_after_max_retries(small_stage, short_song_analysis):
    bad = {"cues": []}
    stub = _StubOllama(responses=[json.dumps(bad), json.dumps(bad)])
    designer = LightingDesignerLLM(ollama=stub, knowledge_context="(stub)")
    with pytest.raises(DesignerError):
        await designer.design_show(small_stage, short_song_analysis, max_retries=1)
    # Single chunk × (1 + 1) retries = 2 calls
    assert stub.calls == 2


@pytest.mark.asyncio
async def test_designer_chunks_long_song(small_stage, song_analysis):
    """A 60s song should split into 2 chunks of 30s each."""
    ids = [f.fixture_id for f in small_stage.fixtures]
    # Cues within a 30s chunk: time_s 0..28 step 4
    chunk_cues = [
        {
            "time_s": float(t),
            "duration_s": 4.0,
            "section_label": "verse",
            "description": "x",
            "fixture_states": [_state(i).model_dump() for i in ids],
        }
        for t in range(0, 30, 4)
    ]
    stub = _StubOllama(responses=[json.dumps({"cues": chunk_cues})])
    designer = LightingDesignerLLM(ollama=stub, knowledge_context="(stub)")
    result = await designer.design_show(small_stage, song_analysis, max_retries=0)
    # 2 chunks × 1 call each = 2 Ollama calls
    assert stub.calls == 2
    # All cues from both chunks combined
    assert len(result.cues) == len(chunk_cues) * 2
    # Times in second chunk should be shifted by 30s
    second_chunk_starts = [c.time_s for c in result.cues[len(chunk_cues):]]
    assert min(second_chunk_starts) >= 30.0


def test_compute_chunks_aligns_to_segments():
    """Chunks should not split mid-segment when possible."""
    analysis = SongAnalysis(
        genre_estimate="pop",
        energy_profile="medium",
        mood="uplifting",
        structure=[
            SongStructureSection(start_s=0.0, end_s=20.0, label="intro", energy=0.3),
            SongStructureSection(start_s=20.0, end_s=45.0, label="verse", energy=0.5),
            SongStructureSection(start_s=45.0, end_s=80.0, label="chorus", energy=0.8),
            SongStructureSection(start_s=80.0, end_s=100.0, label="outro", energy=0.4),
        ],
        key_moments=[],
        lighting_hints=LightingHints(
            color_palette="warm", movement_intensity="moderate",
            strobe_appropriate=False, beam_appropriate=True,
        ),
        duration_s=100.0,
    )
    chunks = LightingDesignerLLM._compute_chunks(analysis, max_chunk_s=30.0)
    # Every chunk should respect the 30s ceiling
    for start, end in chunks:
        assert end - start <= 30.01
    # Chunks are contiguous and cover [0, 100]
    assert chunks[0][0] == 0.0
    assert chunks[-1][1] == pytest.approx(100.0)
    for (_, prev_end), (next_start, _) in zip(chunks, chunks[1:]):
        assert next_start == pytest.approx(prev_end)
    # At least some chunk boundary should align with a real segment edge
    # (proving we prefer segment boundaries over uniform cuts)
    segment_edges = {0.0, 20.0, 45.0, 80.0, 100.0}
    boundary_set = {round(e, 2) for _, e in chunks}
    aligned = boundary_set & segment_edges
    assert len(aligned) >= 2, f"Expected segment-aligned boundaries, got {boundary_set}"


def test_compute_chunks_hard_splits_long_segment():
    """A single segment > max_chunk_s must be hard-split."""
    analysis = SongAnalysis(
        genre_estimate="other", energy_profile="medium", mood="energetic",
        structure=[
            SongStructureSection(start_s=0.0, end_s=75.0, label="instrumental", energy=0.5),
        ],
        key_moments=[],
        lighting_hints=LightingHints(
            color_palette="cool", movement_intensity="moderate",
            strobe_appropriate=False, beam_appropriate=True,
        ),
        duration_s=75.0,
    )
    chunks = LightingDesignerLLM._compute_chunks(analysis, max_chunk_s=30.0)
    assert len(chunks) == 3
    for start, end in chunks:
        assert end - start <= 30.01


# ---------------------------------------------------------------------------
# Bug fixes: auto-fill, pan/tilt normalization, zoning
# ---------------------------------------------------------------------------
from mli_synthetics.llm.designer import (
    DEFAULT_FIXTURE_STATE,
    SIX_ZONE_ORDER,
    _default_state,
    _merge_cues_by_time,
    _normalize_axis,
    build_six_zone_view,
    fill_missing_cue_fields,
    fill_missing_fixture_fields,
)
from mli_synthetics.stage.generator import build_llm_view


def test_normalize_axis_in_range_pass_through():
    assert _normalize_axis(0.0) == 0.0
    assert _normalize_axis(-1.0) == -1.0
    assert _normalize_axis(1.0) == 1.0
    assert _normalize_axis(0.42) == 0.42


def test_normalize_axis_treats_out_of_range_as_degrees():
    # 0° -> -1, 135° -> 0, 270° -> +1 per the user-provided formula
    assert _normalize_axis(135.0) == pytest.approx(0.0)
    assert _normalize_axis(270.0) == pytest.approx(1.0)
    assert _normalize_axis(2.0) == pytest.approx((2.0 / 270.0) * 2.0 - 1.0)


def test_normalize_axis_clamps_extremes():
    # Values way beyond 270° clamp to the boundaries
    assert _normalize_axis(540.0) == 1.0
    assert _normalize_axis(-540.0) == -1.0


def test_default_state_is_off():
    s = _default_state("foo")
    assert s.fixture_id == "foo"
    assert s.intensity == 0.0
    assert s.color == (0, 0, 0)
    assert s.movement.pan == 0.0
    assert s.movement.tilt == 0.0
    assert s.movement.movement_type == "static"
    assert s.effect == "none"


def test_movement_accepts_raw_degrees_without_validation_error():
    """Bug 2: Movement no longer enforces [-1, 1] - degrees are accepted."""
    m = Movement(pan=180.0, tilt=540.0)
    assert m.pan == 180.0  # raw, will be normalized later in postprocess


def test_build_llm_view_passthrough_when_under_cap():
    layout = StageLayout(
        width_m=8.0, depth_m=5.0, height_m=4.0, venue_size=VenueSize.SMALL_CLUB,
        fixtures=[
            Fixture(fixture_id=f"f{i}", category=FixtureCategory.LED_PAR,
                    position=Position.STAGE_FLOOR, x=float(i), y=2.0, z=0.1)
            for i in range(5)
        ],
    )
    view, zone_map = build_llm_view(layout, max_fixtures=12)
    assert zone_map == {}
    assert len(view["fixtures"]) == 5
    assert view["fixtures"][0]["fixture_id"] == "f0"


def test_build_llm_view_zones_when_over_cap():
    fixtures = []
    # 16 LED pars on floor + 8 beams on back truss = 24 fixtures, 2 zones
    for i in range(16):
        fixtures.append(Fixture(
            fixture_id=f"par_{i}", category=FixtureCategory.LED_PAR,
            position=Position.STAGE_FLOOR, x=float(i), y=2.0, z=0.1,
        ))
    for i in range(8):
        fixtures.append(Fixture(
            fixture_id=f"beam_{i}", category=FixtureCategory.MOVING_BEAM,
            position=Position.BACK_TRUSS, x=float(i), y=4.0, z=3.5,
        ))
    layout = StageLayout(
        width_m=10.0, depth_m=5.0, height_m=4.0,
        venue_size=VenueSize.MEDIUM_CLUB, fixtures=fixtures,
    )
    view, zone_map = build_llm_view(layout, max_fixtures=12)
    assert view["zoning_applied"] is True
    assert len(view["fixtures"]) == 2  # one per (position, category) group
    assert set(zone_map.keys()) == {
        "zone_stage_floor_led_par",
        "zone_back_truss_moving_beam",
    }
    assert len(zone_map["zone_stage_floor_led_par"]) == 16
    assert len(zone_map["zone_back_truss_moving_beam"]) == 8


@pytest.mark.asyncio
async def test_design_show_expands_zones_to_all_fixtures():
    """Bug 3 end-to-end: LLM emits zone-level cues → expanded to all fixtures."""
    fixtures = [
        Fixture(fixture_id=f"par_{i}", category=FixtureCategory.LED_PAR,
                position=Position.STAGE_FLOOR, x=float(i), y=2.0, z=0.1)
        for i in range(16)
    ]
    layout = StageLayout(
        width_m=10.0, depth_m=5.0, height_m=4.0,
        venue_size=VenueSize.SMALL_CLUB, fixtures=fixtures,
    )
    analysis = SongAnalysis(
        genre_estimate="pop", energy_profile="medium", mood="uplifting",
        structure=[SongStructureSection(start_s=0.0, end_s=10.0, label="verse", energy=0.5)],
        key_moments=[],
        lighting_hints=LightingHints(
            color_palette="warm", movement_intensity="subtle",
            strobe_appropriate=False, beam_appropriate=False,
        ),
        duration_s=10.0,
    )
    # LLM returns ONE state per cue, addressing the zone. Post-process
    # should expand it onto all 16 fixtures.
    zone_cue = {
        "time_s": 0.0,
        "duration_s": 5.0,
        "section_label": "verse",
        "description": "wash on",
        "fixture_states": [{
            "fixture_id": "zone_stage_floor_led_par",
            "intensity": 0.6,
            "color": [255, 180, 80],
            "movement": {"pan": 0.0, "tilt": 0.5,
                         "movement_type": "static", "speed": "slow"},
            "effect": "none",
            "effect_speed_hz": 0.0,
        }],
    }
    stub = _StubOllama(responses=[json.dumps({"cues": [zone_cue, {**zone_cue, "time_s": 5.0}]})])
    designer = LightingDesignerLLM(ollama=stub, knowledge_context="(stub)")
    result = await designer.design_show(layout, analysis, max_retries=0)
    # Every cue should now have all 16 fixtures (expanded from the zone)
    for cue in result.cues:
        ids = {s.fixture_id for s in cue.fixture_states}
        assert ids == {f.fixture_id for f in fixtures}


@pytest.mark.asyncio
async def test_design_show_autofills_missing_fixtures(small_stage, short_song_analysis):
    """Bug 1 end-to-end: LLM omits a fixture; post-process fills it with defaults."""
    ids = [f.fixture_id for f in small_stage.fixtures]
    # Only emit a state for the first fixture; the second is missing
    partial_cue = {
        "time_s": 0.0,
        "duration_s": 5.0,
        "section_label": "verse",
        "description": "partial",
        "fixture_states": [_state(ids[0]).model_dump()],
    }
    stub = _StubOllama(responses=[json.dumps({"cues": [partial_cue, {**partial_cue, "time_s": 5.0}]})])
    designer = LightingDesignerLLM(ollama=stub, knowledge_context="(stub)")
    result = await designer.design_show(small_stage, short_song_analysis, max_retries=0)
    for cue in result.cues:
        present = {s.fixture_id for s in cue.fixture_states}
        assert present == set(ids)
        # The auto-filled one should be off
        autofilled = next(s for s in cue.fixture_states if s.fixture_id == ids[1])
        assert autofilled.intensity == 0.0
        assert autofilled.color == (0, 0, 0)


@pytest.mark.asyncio
async def test_design_show_normalizes_pan_tilt(small_stage, short_song_analysis):
    """Bug 2 end-to-end: LLM emits degrees; post-process normalizes."""
    ids = [f.fixture_id for f in small_stage.fixtures]
    cue_with_degrees = {
        "time_s": 0.0,
        "duration_s": 5.0,
        "section_label": "verse",
        "description": "degrees",
        "fixture_states": [{
            "fixture_id": fid,
            "intensity": 0.5,
            "color": [255, 255, 255],
            "movement": {"pan": 270.0, "tilt": 135.0,
                         "movement_type": "static", "speed": "slow"},
            "effect": "none",
            "effect_speed_hz": 0.0,
        } for fid in ids],
    }
    stub = _StubOllama(responses=[json.dumps({"cues": [cue_with_degrees,
                                                       {**cue_with_degrees, "time_s": 5.0}]})])
    designer = LightingDesignerLLM(ollama=stub, knowledge_context="(stub)")
    result = await designer.design_show(small_stage, short_song_analysis, max_retries=0)
    for cue in result.cues:
        for s in cue.fixture_states:
            assert -1.0 <= s.movement.pan <= 1.0
            assert -1.0 <= s.movement.tilt <= 1.0


# ---------------------------------------------------------------------------
# Structural-fix: fill_missing_fixture_fields
# ---------------------------------------------------------------------------
def test_fill_missing_fixture_fields_adds_all_defaults():
    """A fixture_state with only fixture_id gets every default field filled."""
    data = {"cues": [{
        "time_s": 0.0, "duration_s": 4.0, "section_label": "verse",
        "description": "x",
        "fixture_states": [{"fixture_id": "a"}],
    }]}
    fill_missing_fixture_fields(data)
    s = data["cues"][0]["fixture_states"][0]
    assert s["intensity"] == 0.0
    assert s["color"] == [255, 255, 255] or s["color"] == [0, 0, 0]
    assert s["effect"] == "none"
    assert s["effect_speed_hz"] == 0.0
    assert s["movement"]["pan"] == 0.0
    assert s["movement"]["movement_type"] == "static"


def test_fill_missing_fixture_fields_coerces_string_color():
    """Bug: LLM writes color as a string ('blue'). Replace with safe RGB."""
    data = {"cues": [{
        "time_s": 0.0, "duration_s": 4.0, "section_label": "v", "description": "",
        "fixture_states": [{
            "fixture_id": "a", "intensity": 0.5, "color": "blue",
            "movement": {"pan": 0.0, "tilt": 0.0,
                         "movement_type": "static", "speed": "slow"},
            "effect": "none", "effect_speed_hz": 0.0,
        }],
    }]}
    fill_missing_fixture_fields(data)
    color = data["cues"][0]["fixture_states"][0]["color"]
    assert isinstance(color, list) and len(color) == 3


def test_fill_missing_fixture_fields_repairs_partial_movement():
    """If movement is a dict missing some keys, the rest get defaults."""
    data = {"cues": [{
        "time_s": 0.0, "duration_s": 4.0, "section_label": "v", "description": "",
        "fixture_states": [{
            "fixture_id": "a", "intensity": 0.5, "color": [10, 20, 30],
            "movement": {"pan": 0.4},  # tilt/movement_type/speed missing
            "effect": "none", "effect_speed_hz": 0.0,
        }],
    }]}
    fill_missing_fixture_fields(data)
    mv = data["cues"][0]["fixture_states"][0]["movement"]
    assert mv["pan"] == 0.4
    assert mv["tilt"] == 0.0
    assert mv["movement_type"] == "static"
    assert mv["speed"] == "slow"


def test_fill_missing_fixture_fields_handles_empty_or_missing():
    """Empty cue list or missing key should not raise."""
    assert fill_missing_fixture_fields({}) == {}
    assert fill_missing_fixture_fields({"cues": []}) == {"cues": []}
    # Cue with no fixture_states key
    data = {"cues": [{"time_s": 0.0, "duration_s": 4.0,
                      "section_label": "v", "description": ""}]}
    fill_missing_fixture_fields(data)  # should not raise


# ---------------------------------------------------------------------------
# Six-zone grouping (Fix 1)
# ---------------------------------------------------------------------------
def test_six_zone_view_passthrough_when_under_cap():
    layout = StageLayout(
        width_m=8.0, depth_m=5.0, height_m=4.0,
        venue_size=VenueSize.SMALL_CLUB,
        fixtures=[
            Fixture(fixture_id=f"f{i}", category=FixtureCategory.LED_PAR,
                    position=Position.STAGE_FLOOR, x=float(i), y=2.0, z=0.1)
            for i in range(5)
        ],
    )
    view, zone_map = build_six_zone_view(layout, max_fixtures=12)
    assert zone_map == {}
    assert len(view["fixtures"]) == 5


def test_six_zone_view_groups_into_at_most_six_zones():
    """A large stage should be reduced to AT MOST 6 zones, regardless of variety."""
    fixtures = []
    # Plenty of fixtures across every position and several categories
    layouts_to_test = [
        (Position.BACK_TRUSS, FixtureCategory.MOVING_BEAM, 8),
        (Position.MID_TRUSS, FixtureCategory.MOVING_WASH, 6),
        (Position.FRONT_TRUSS, FixtureCategory.MOVING_SPOT, 6),
        (Position.STAGE_FLOOR, FixtureCategory.LED_PAR, 10),
        (Position.SIDE_LEFT, FixtureCategory.STROBE, 3),
        (Position.SIDE_RIGHT, FixtureCategory.STROBE, 3),
        (Position.STAGE_FLOOR, FixtureCategory.HAZER, 2),
        (Position.BACK_TRUSS, FixtureCategory.LASER, 2),
        (Position.FRONT_TRUSS, FixtureCategory.BLINDER, 4),
        (Position.AUDIENCE_FACING, FixtureCategory.BLINDER, 2),
    ]
    i = 0
    for pos, cat, n in layouts_to_test:
        for _ in range(n):
            fixtures.append(Fixture(
                fixture_id=f"f{i}", category=cat, position=pos,
                x=0.0, y=2.0, z=2.0,
            ))
            i += 1
    layout = StageLayout(
        width_m=15.0, depth_m=8.0, height_m=6.0,
        venue_size=VenueSize.LARGE_VENUE, fixtures=fixtures,
    )
    view, zone_map = build_six_zone_view(layout, max_fixtures=12)
    assert len(view["fixtures"]) <= 6
    assert len(zone_map) <= 6
    # All zone ids come from the fixed catalog
    for zid in zone_map:
        assert zid in SIX_ZONE_ORDER


def test_six_zone_routing_rules():
    """Verify each category/position maps to the correct fixed zone."""
    cases = [
        # (category, position, expected zone_id)
        (FixtureCategory.MOVING_BEAM, Position.BACK_TRUSS, "zone_back"),
        (FixtureCategory.LED_PAR, Position.BACKLINE, "zone_back"),
        (FixtureCategory.MOVING_WASH, Position.MID_TRUSS, "zone_mid"),
        (FixtureCategory.MOVING_SPOT, Position.FRONT_TRUSS, "zone_front"),
        (FixtureCategory.MOVING_BEAM, Position.AUDIENCE_FACING, "zone_front"),
        (FixtureCategory.BLINDER, Position.STAGE_FLOOR, "zone_front"),  # blinder→front
        (FixtureCategory.BLINDER, Position.MID_TRUSS, "zone_front"),
        (FixtureCategory.LED_PAR, Position.STAGE_FLOOR, "zone_floor"),
        (FixtureCategory.STROBE, Position.SIDE_LEFT, "zone_sides"),
        (FixtureCategory.LED_PAR, Position.SIDE_RIGHT, "zone_sides"),
        (FixtureCategory.HAZER, Position.STAGE_FLOOR, "zone_special"),
        (FixtureCategory.CO2_JET, Position.BACK_TRUSS, "zone_special"),
        (FixtureCategory.PYRO, Position.STAGE_FLOOR, "zone_special"),
        (FixtureCategory.LASER, Position.MID_TRUSS, "zone_special"),
    ]
    fixtures = [
        Fixture(fixture_id=f"f{i}", category=cat, position=pos,
                x=0.0, y=2.0, z=2.0)
        for i, (cat, pos, _) in enumerate(cases)
    ]
    # Add filler so we cross the 12-fixture cap and zoning activates
    fixtures.extend([
        Fixture(fixture_id=f"filler{i}", category=FixtureCategory.LED_PAR,
                position=Position.STAGE_FLOOR, x=0.0, y=2.0, z=0.1)
        for i in range(15)
    ])
    layout = StageLayout(
        width_m=15.0, depth_m=8.0, height_m=6.0,
        venue_size=VenueSize.LARGE_VENUE, fixtures=fixtures,
    )
    _, zone_map = build_six_zone_view(layout, max_fixtures=12)
    for i, (_, _, expected_zone) in enumerate(cases):
        fid = f"f{i}"
        assert fid in zone_map[expected_zone], (
            f"f{i} expected in {expected_zone}, got {[z for z, ids in zone_map.items() if fid in ids]}"
        )


def test_six_zone_view_drops_empty_zones():
    """Only zones with fixtures should appear in the result."""
    # Many fixtures, all in stage floor → only zone_floor populated
    fixtures = [
        Fixture(fixture_id=f"f{i}", category=FixtureCategory.LED_PAR,
                position=Position.STAGE_FLOOR, x=float(i), y=2.0, z=0.1)
        for i in range(20)
    ]
    layout = StageLayout(
        width_m=10.0, depth_m=5.0, height_m=4.0,
        venue_size=VenueSize.MEDIUM_CLUB, fixtures=fixtures,
    )
    view, zone_map = build_six_zone_view(layout, max_fixtures=12)
    assert set(zone_map.keys()) == {"zone_floor"}
    assert len(view["fixtures"]) == 1


# ---------------------------------------------------------------------------
# fill_missing_cue_fields (Fix 2)
# ---------------------------------------------------------------------------
def test_fill_missing_cue_fields_adds_defaults():
    data = {"cues": [
        {"time_s": 0.0, "fixture_states": []},
        {"time_s": 4.0, "fixture_states": [], "section_label": ""},
    ]}
    fill_missing_cue_fields(data)
    for i, cue in enumerate(data["cues"]):
        assert cue["section_label"] == "instrumental"
        assert cue["description"] == f"Cue {i}"
        assert cue["duration_s"] == 4.0


def test_fill_missing_cue_fields_preserves_existing():
    data = {"cues": [{
        "time_s": 0.0,
        "duration_s": 8.0,
        "section_label": "chorus",
        "description": "punchy",
        "fixture_states": [],
    }]}
    fill_missing_cue_fields(data)
    c = data["cues"][0]
    assert c["section_label"] == "chorus"
    assert c["description"] == "punchy"
    assert c["duration_s"] == 8.0


def test_fill_missing_cue_fields_handles_missing_cues_key():
    assert fill_missing_cue_fields({}) == {}
    assert fill_missing_cue_fields({"cues": None}) == {"cues": None}


@pytest.mark.asyncio
async def test_design_show_recovers_from_missing_cue_fields(small_stage, short_song_analysis):
    """End-to-end: LLM emits cues missing section_label/description/duration_s."""
    ids = [f.fixture_id for f in small_stage.fixtures]
    # Skinniest possible cue dict — only time_s + fixture_states
    cue_minimal = {
        "time_s": 0.0,
        "fixture_states": [{"fixture_id": fid} for fid in ids],
    }
    stub = _StubOllama(responses=[json.dumps({"cues": [cue_minimal,
                                                       {**cue_minimal, "time_s": 5.0}]})])
    designer = LightingDesignerLLM(ollama=stub, knowledge_context="(stub)")
    result = await designer.design_show(small_stage, short_song_analysis, max_retries=0)
    assert len(result.cues) == 2
    for cue in result.cues:
        assert cue.section_label  # filled
        assert cue.description     # filled
        assert cue.duration_s > 0  # filled


# ---------------------------------------------------------------------------
# Split-zone fallback (Fix 3): merge utility
# ---------------------------------------------------------------------------
def test_merge_cues_by_time_combines_close_cues():
    """Two cues within tolerance get fixture_states merged."""
    cue_a = Cue(time_s=0.0, duration_s=4.0, section_label="v", description="a",
                fixture_states=[_state("x")])
    cue_b = Cue(time_s=0.2, duration_s=4.0, section_label="v", description="b",
                fixture_states=[_state("y")])
    out = _merge_cues_by_time([cue_a, cue_b], tolerance_s=0.5)
    assert len(out) == 1
    ids = {s.fixture_id for s in out[0].fixture_states}
    assert ids == {"x", "y"}


def test_merge_cues_by_time_keeps_distant_cues_separate():
    cue_a = Cue(time_s=0.0, duration_s=4.0, section_label="v", description="a",
                fixture_states=[_state("x")])
    cue_b = Cue(time_s=4.0, duration_s=4.0, section_label="v", description="b",
                fixture_states=[_state("y")])
    out = _merge_cues_by_time([cue_a, cue_b], tolerance_s=0.5)
    assert len(out) == 2


def test_merge_cues_by_time_later_state_wins_on_collision():
    cue_a = Cue(time_s=0.0, duration_s=4.0, section_label="v", description="a",
                fixture_states=[FixtureState(
                    fixture_id="x", intensity=0.2, color=(0, 0, 0),
                    movement=Movement(), effect="none", effect_speed_hz=0.0)])
    cue_b = Cue(time_s=0.1, duration_s=4.0, section_label="v", description="b",
                fixture_states=[FixtureState(
                    fixture_id="x", intensity=0.9, color=(255, 0, 0),
                    movement=Movement(), effect="none", effect_speed_hz=0.0)])
    out = _merge_cues_by_time([cue_a, cue_b], tolerance_s=0.5)
    assert len(out) == 1
    # Later state wins
    s = out[0].fixture_states[0]
    assert s.intensity == 0.9


@pytest.mark.asyncio
async def test_design_show_falls_back_on_timeout(monkeypatch):
    """Fix 3 end-to-end: when the first call exceeds 120s, the split-zone
    fallback fires - two parallel sub-calls, results merged.
    """
    # Stage with enough fixtures to trigger zoning (>12)
    fixtures = []
    for pos, cat, n in [
        (Position.BACK_TRUSS, FixtureCategory.MOVING_BEAM, 4),
        (Position.MID_TRUSS, FixtureCategory.MOVING_WASH, 4),
        (Position.FRONT_TRUSS, FixtureCategory.MOVING_SPOT, 4),
        (Position.STAGE_FLOOR, FixtureCategory.LED_PAR, 6),
    ]:
        for j in range(n):
            fixtures.append(Fixture(
                fixture_id=f"{cat.value}_{pos.value}_{j}",
                category=cat, position=pos, x=0.0, y=2.0, z=2.0,
            ))
    layout = StageLayout(
        width_m=12.0, depth_m=8.0, height_m=6.0,
        venue_size=VenueSize.MEDIUM_CLUB, fixtures=fixtures,
    )
    analysis = SongAnalysis(
        genre_estimate="edm", energy_profile="high", mood="energetic",
        structure=[SongStructureSection(start_s=0.0, end_s=10.0, label="drop", energy=0.9)],
        key_moments=[],
        lighting_hints=LightingHints(
            color_palette="contrast", movement_intensity="high",
            strobe_appropriate=True, beam_appropriate=True,
        ),
        duration_s=10.0,
    )

    # First call: hangs (asyncio.sleep > 120). Sub-calls: respond immediately.
    call_count = {"n": 0}

    class FlakyOllama:
        async def generate(self, **kwargs):
            call_count["n"] += 1
            if call_count["n"] == 1:
                # Simulate a long-running call
                await asyncio.sleep(5.0)
                return json.dumps({"cues": []})
            # Sub-calls: return a single cue covering some zones
            return json.dumps({"cues": [{
                "time_s": 0.0, "duration_s": 5.0,
                "section_label": "drop", "description": "x",
                "fixture_states": [{
                    "fixture_id": "zone_back",
                    "intensity": 1.0, "color": [255, 0, 0],
                    "movement": {"pan": 0.0, "tilt": 0.0,
                                 "movement_type": "static", "speed": "fast"},
                    "effect": "none", "effect_speed_hz": 0.0,
                }],
            }]})

    # Shrink timeout to 0.1s so the test runs fast
    monkeypatch.setattr("mli_synthetics.llm.designer.FIRST_CALL_SOFT_TIMEOUT_S", 0.1)

    designer = LightingDesignerLLM(ollama=FlakyOllama(), knowledge_context="(stub)")
    result = await designer.design_show(layout, analysis, max_retries=0)
    # First call timed out, then 2 sub-calls fired
    assert call_count["n"] >= 3
    assert len(result.cues) >= 1
    # Every cue should cover every real fixture (autofill kicked in)
    fixture_ids = {f.fixture_id for f in layout.fixtures}
    for cue in result.cues:
        present = {s.fixture_id for s in cue.fixture_states}
        assert present == fixture_ids


@pytest.mark.asyncio
async def test_design_show_recovers_from_incomplete_fixture_states(
    small_stage, short_song_analysis
):
    """End-to-end: LLM emits cue with fixture_states missing required fields.

    Without fill_missing_fixture_fields, Pydantic would reject these.
    With it, validation passes and missing fields get safe defaults.
    """
    ids = [f.fixture_id for f in small_stage.fixtures]
    skinny_cue = {
        "time_s": 0.0,
        "duration_s": 5.0,
        "section_label": "verse",
        "description": "skinny",
        "fixture_states": [{"fixture_id": fid} for fid in ids],
    }
    stub = _StubOllama(responses=[json.dumps({"cues": [skinny_cue,
                                                       {**skinny_cue, "time_s": 5.0}]})])
    designer = LightingDesignerLLM(ollama=stub, knowledge_context="(stub)")
    result = await designer.design_show(small_stage, short_song_analysis, max_retries=0)
    # No exception, valid cue list
    assert len(result.cues) == 2
    for cue in result.cues:
        for s in cue.fixture_states:
            assert 0.0 <= s.intensity <= 1.0
            assert isinstance(s.color, tuple) and len(s.color) == 3
