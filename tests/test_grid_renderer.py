"""Tests for the Phase 2 grid renderer."""
from __future__ import annotations

import shutil
import subprocess
import wave
from pathlib import Path

import numpy as np
import pytest

from mli_synthetics.llm.designer import (
    Cue,
    CueList,
    CueListMetadata,
    FixtureState,
    Movement,
)
from mli_synthetics.renderer.grid_renderer import GridRenderer
from mli_synthetics.stage.fixtures import (
    Fixture,
    FixtureCategory,
    Position,
    StageLayout,
    VenueSize,
)


def _state(fid: str, intensity: float, rgb: tuple[int, int, int]) -> FixtureState:
    return FixtureState(
        fixture_id=fid,
        intensity=intensity,
        color=rgb,
        movement=Movement(),
        effect="none",
        effect_speed_hz=0.0,
    )


@pytest.fixture
def four_corner_stage() -> StageLayout:
    """Stage with fixtures placed at the four extremes of (x, z)."""
    return StageLayout(
        width_m=8.0, depth_m=5.0, height_m=4.5,
        venue_size=VenueSize.SMALL_CLUB,
        fixtures=[
            # Low x / low z -> grid (0,0)
            Fixture(fixture_id="a", category=FixtureCategory.LED_PAR,
                    position=Position.STAGE_FLOOR, x=-2.0, y=2.0, z=0.5),
            # High x / low z -> grid (0,11)
            Fixture(fixture_id="b", category=FixtureCategory.LED_PAR,
                    position=Position.STAGE_FLOOR, x=2.0, y=2.0, z=0.5),
            # Low x / high z -> grid (7,0)
            Fixture(fixture_id="c", category=FixtureCategory.MOVING_BEAM,
                    position=Position.BACK_TRUSS, x=-2.0, y=4.0, z=4.0),
            # High x / high z -> grid (7,11)
            Fixture(fixture_id="d", category=FixtureCategory.MOVING_BEAM,
                    position=Position.BACK_TRUSS, x=2.0, y=4.0, z=4.0),
        ],
    )


@pytest.fixture
def three_cue_list() -> CueList:
    return CueList(
        metadata=CueListMetadata(designer_notes="test"),
        cues=[
            Cue(time_s=0.0, duration_s=1.0, section_label="intro", description="",
                fixture_states=[
                    _state("a", 1.0, (255, 0, 0)),
                    _state("b", 1.0, (255, 0, 0)),
                    _state("c", 0.0, (0, 0, 0)),
                    _state("d", 0.0, (0, 0, 0)),
                ]),
            Cue(time_s=1.0, duration_s=1.0, section_label="build", description="",
                fixture_states=[
                    _state("a", 0.5, (0, 0, 255)),
                    _state("b", 0.5, (0, 0, 255)),
                    _state("c", 1.0, (0, 255, 0)),
                    _state("d", 1.0, (0, 255, 0)),
                ]),
            Cue(time_s=2.0, duration_s=0.5, section_label="drop", description="",
                fixture_states=[
                    _state(fid, 1.0, (255, 255, 255)) for fid in "abcd"
                ]),
        ],
    )


# ---------------------------------------------------------------------------
def test_grid_map_lands_at_extremes(four_corner_stage, three_cue_list):
    r = GridRenderer(four_corner_stage, three_cue_list)
    assert r.fixture_to_cell == {
        "a": (0, 0),
        "b": (0, 11),
        "c": (7, 0),
        "d": (7, 11),
    }


def test_grid_map_handles_collinear_fixtures():
    """If every fixture has the same x (or z), the span is degenerate;
    the renderer must not divide by zero."""
    layout = StageLayout(
        width_m=8.0, depth_m=5.0, height_m=4.0,
        venue_size=VenueSize.SMALL_CLUB,
        fixtures=[
            Fixture(fixture_id=f"f{i}", category=FixtureCategory.LED_PAR,
                    position=Position.STAGE_FLOOR, x=0.0, y=2.0, z=0.1)
            for i in range(3)
        ],
    )
    cues = CueList(metadata=CueListMetadata(), cues=[
        Cue(time_s=0.0, duration_s=1.0, section_label="a", description="",
            fixture_states=[_state(f"f{i}", 1.0, (255, 0, 0)) for i in range(3)])
    ])
    r = GridRenderer(layout, cues)
    # All fixtures should land in the same cell (no NaN, no exception)
    cells = set(r.fixture_to_cell.values())
    assert len(cells) == 1


def test_render_frame_interpolation_at_midpoint(four_corner_stage, three_cue_list):
    """At t = 0.5s (midway between cue 0 and cue 1) we expect the floor
    cells (a, b) to be roughly half-way between (255,0,0)*1.0 and
    (0,0,255)*0.5: red component falls, blue rises but at lower intensity."""
    r = GridRenderer(four_corner_stage, three_cue_list)
    frame_bgr = r._render_frame_bgr(0.5)
    # Cell (0, 0) is fixture "a"
    # OpenCV is BGR; sample one pixel from the top-left 32x32 block
    pixel_bgr = frame_bgr[16, 16]
    # Red channel (BGR -> R is index 2) should be partway between full red
    # and zero; blue (index 0) partway from zero to mid-blue.
    b, g, _r = int(pixel_bgr[0]), int(pixel_bgr[1]), int(pixel_bgr[2])
    # Math: intensity = 0.5*1.0 + 0.5*0.5 = 0.75
    #       color = 0.5*(255,0,0) + 0.5*(0,0,255) = (127.5, 0, 127.5)
    #       rgb = color * intensity = (95.6, 0, 95.6) -> R=95, B=95
    assert g == 0
    assert 80 <= _r <= 110, f"expected red mid-fade ~95, got R={_r}"
    assert 80 <= b <= 110, f"expected partial blue rise ~95, got B={b}"


def test_additive_blending_clamps_to_255():
    """Two fixtures in the same cell with bright reds should clamp to 255."""
    layout = StageLayout(
        width_m=8.0, depth_m=5.0, height_m=4.0,
        venue_size=VenueSize.SMALL_CLUB,
        fixtures=[
            Fixture(fixture_id="x", category=FixtureCategory.LED_PAR,
                    position=Position.STAGE_FLOOR, x=0.0, y=2.0, z=0.5),
            Fixture(fixture_id="y", category=FixtureCategory.LED_PAR,
                    position=Position.STAGE_FLOOR, x=0.0, y=2.0, z=0.5),
        ],
    )
    cues = CueList(metadata=CueListMetadata(), cues=[
        Cue(time_s=0.0, duration_s=1.0, section_label="a", description="",
            fixture_states=[
                _state("x", 1.0, (200, 0, 0)),
                _state("y", 1.0, (200, 0, 0)),
            ]),
    ])
    r = GridRenderer(layout, cues)
    frame = r._render_frame_bgr(0.0)
    # BGR: R channel is index 2; 200+200 = 400, clamped to 255
    assert int(frame[16, 16, 2]) == 255


@pytest.mark.skipif(shutil.which("ffmpeg") is None, reason="ffmpeg not on PATH")
def test_render_to_video_produces_h264_mp4(four_corner_stage, three_cue_list, tmp_path):
    out = tmp_path / "preview.mp4"
    path = GridRenderer(four_corner_stage, three_cue_list).render_to_video(
        output_video=out, duration_s=2.5, audio_path=None,
    )
    assert path.exists() and path.stat().st_size > 0

    probe = subprocess.run(
        ["ffprobe", "-v", "error", "-show_streams",
         "-of", "default=noprint_wrappers=1", str(path)],
        capture_output=True, text=True, check=True,
    )
    out_text = probe.stdout
    assert "codec_name=h264" in out_text
    assert "width=384" in out_text
    assert "height=256" in out_text


@pytest.mark.skipif(shutil.which("ffmpeg") is None, reason="ffmpeg not on PATH")
def test_render_with_audio_mux(four_corner_stage, three_cue_list, tmp_path):
    # Make a 2.5s mono WAV
    sr = 22050
    dur = 2.5
    t = np.linspace(0, dur, int(sr * dur), endpoint=False)
    pcm = (np.sin(2 * np.pi * 440 * t) * 0.3 * 32767).astype(np.int16)
    wav = tmp_path / "test.wav"
    with wave.open(str(wav), "wb") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(sr)
        wf.writeframes(pcm.tobytes())

    out = tmp_path / "preview.mp4"
    GridRenderer(four_corner_stage, three_cue_list).render_to_video(
        output_video=out, duration_s=2.5, audio_path=wav,
    )
    probe = subprocess.run(
        ["ffprobe", "-v", "error", "-show_streams",
         "-of", "default=noprint_wrappers=1", str(out)],
        capture_output=True, text=True, check=True,
    )
    # Both video and audio streams present
    assert "codec_name=h264" in probe.stdout
    assert "codec_name=aac" in probe.stdout


def test_render_from_run_dir(four_corner_stage, three_cue_list, tmp_path):
    """End-to-end: write the two JSON files into a dir, then call the
    convenience loader as the CLI script does."""
    import json
    (tmp_path / "stage_layout.json").write_text(
        json.dumps(four_corner_stage.model_dump(mode="json")), encoding="utf-8"
    )
    (tmp_path / "cue_list.json").write_text(
        json.dumps(three_cue_list.model_dump()), encoding="utf-8"
    )
    if shutil.which("ffmpeg") is None:
        pytest.skip("ffmpeg not on PATH")
    from mli_synthetics.renderer.grid_renderer import render_from_run_dir

    out = render_from_run_dir(tmp_path)
    assert out == tmp_path / "preview.mp4"
    assert out.exists() and out.stat().st_size > 0


def test_duration_zero_raises(four_corner_stage, three_cue_list, tmp_path):
    with pytest.raises(ValueError, match="duration_s"):
        GridRenderer(four_corner_stage, three_cue_list).render_to_video(
            output_video=tmp_path / "x.mp4", duration_s=0.0,
        )
