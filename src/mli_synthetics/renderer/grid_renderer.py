"""MLI 12x8 grid renderer.

Maps the StageLayout's (x, z) coordinates onto a 12-column / 8-row grid
of 32x32-pixel cells (final image 384x256), then renders one frame per
1/FPS using linear interpolation between consecutive cues. Multiple
fixtures sharing a cell are additively blended and clamped to 255.

The video is first written via OpenCV's VideoWriter (mp4v fourcc) and
then re-encoded to H.264 + AAC by ffmpeg, which also muxes in the
source WAV. Requires ffmpeg on the PATH.
"""
from __future__ import annotations

import bisect
import json
import subprocess
from pathlib import Path
from typing import Any

import numpy as np

from mli_synthetics.logging_config import get_logger

logger = get_logger()


class GridRenderer:
    GRID_COLS = 12
    GRID_ROWS = 8
    CELL_PX = 32
    DEFAULT_FPS = 30

    def __init__(
        self,
        stage_layout: Any,
        cue_list: Any,
        fps: int = DEFAULT_FPS,
    ):
        self.stage = stage_layout
        # Sort cues by start time defensively
        self.cues = sorted(cue_list.cues, key=lambda c: c.time_s)
        self.fps = fps
        self.fixture_to_cell: dict[str, tuple[int, int]] = {}
        self._build_fixture_grid_map()
        # Precompute cue_times for bisect lookups
        self._cue_times = [c.time_s for c in self.cues]

    # ------------------------------------------------------------------
    def _build_fixture_grid_map(self) -> None:
        """Compute (row, col) grid coordinates for every fixture."""
        fixtures = self.stage.fixtures
        if not fixtures:
            return
        xs = [f.x for f in fixtures]
        zs = [f.z for f in fixtures]
        x_min, x_max = min(xs), max(xs)
        z_min, z_max = min(zs), max(zs)
        x_span = max(x_max - x_min, 1e-6)
        z_span = max(z_max - z_min, 1e-6)
        for f in fixtures:
            col = int((f.x - x_min) / x_span * (self.GRID_COLS - 1))
            row = int((f.z - z_min) / z_span * (self.GRID_ROWS - 1))
            col = max(0, min(self.GRID_COLS - 1, col))
            row = max(0, min(self.GRID_ROWS - 1, row))
            self.fixture_to_cell[f.fixture_id] = (row, col)
        logger.info(
            "GridRenderer: mapped {} fixtures onto {}x{} grid",
            len(self.fixture_to_cell),
            self.GRID_COLS,
            self.GRID_ROWS,
        )

    # ------------------------------------------------------------------
    def render_to_video(
        self,
        output_video: Path,
        duration_s: float,
        audio_path: Path | None = None,
    ) -> Path:
        """Render frames and (optionally) mux audio via ffmpeg.

        Returns the final video path. Raises RuntimeError on encoder /
        ffmpeg failure.
        """
        try:
            import cv2
        except ImportError as exc:  # pragma: no cover
            raise RuntimeError(
                "opencv-python required. Install with: pip install opencv-python"
            ) from exc

        if duration_s <= 0:
            raise ValueError(f"duration_s must be > 0, got {duration_s}")

        output_video = Path(output_video)
        output_video.parent.mkdir(parents=True, exist_ok=True)
        tmp_video = output_video.with_name(output_video.stem + ".noaudio.mp4")

        h = self.GRID_ROWS * self.CELL_PX
        w = self.GRID_COLS * self.CELL_PX
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        writer = cv2.VideoWriter(str(tmp_video), fourcc, self.fps, (w, h))
        if not writer.isOpened():
            raise RuntimeError(
                f"cv2.VideoWriter failed to open {tmp_video} "
                f"(fourcc=mp4v, size={w}x{h}, fps={self.fps})"
            )

        n_frames = int(round(duration_s * self.fps))
        logger.info(
            "GridRenderer: rendering {} frames ({:.2f}s @ {} fps) -> {}",
            n_frames,
            duration_s,
            self.fps,
            tmp_video.name,
        )
        try:
            for frame_idx in range(n_frames):
                t = frame_idx / self.fps
                frame_bgr = self._render_frame_bgr(t)
                writer.write(frame_bgr)
        finally:
            writer.release()

        if audio_path is not None:
            self._mux_audio_h264(tmp_video, Path(audio_path), output_video)
            try:
                tmp_video.unlink()
            except OSError:
                pass
        else:
            # No audio - still re-encode to H.264 for browser compat
            self._reencode_h264(tmp_video, output_video)
            try:
                tmp_video.unlink()
            except OSError:
                pass

        logger.info("GridRenderer: wrote {}", output_video)
        return output_video

    # ------------------------------------------------------------------
    def _render_frame_bgr(self, t: float) -> np.ndarray:
        """Render one frame at time `t` (seconds). Returns BGR uint8."""
        cell_grid = np.zeros((self.GRID_ROWS, self.GRID_COLS, 3), dtype=np.float32)

        if not self.cues:
            return _upscale_bgr(cell_grid, self.CELL_PX)

        # idx = last cue whose time_s <= t
        idx = bisect.bisect_right(self._cue_times, t) - 1
        if idx < 0:
            return _upscale_bgr(cell_grid, self.CELL_PX)

        cue_a = self.cues[idx]
        cue_b = self.cues[idx + 1] if idx + 1 < len(self.cues) else None

        if cue_b is not None:
            span = cue_b.time_s - cue_a.time_s
            alpha = (t - cue_a.time_s) / span if span > 1e-9 else 0.0
            alpha = max(0.0, min(1.0, alpha))
        else:
            alpha = 0.0

        states_a = {s.fixture_id: s for s in cue_a.fixture_states}
        states_b = {s.fixture_id: s for s in cue_b.fixture_states} if cue_b else {}

        for fid, (row, col) in self.fixture_to_cell.items():
            sa = states_a.get(fid)
            sb = states_b.get(fid)
            if sa is None and sb is None:
                continue
            if sa is not None and sb is not None:
                intensity = (1.0 - alpha) * sa.intensity + alpha * sb.intensity
                color_a = np.array(sa.color, dtype=np.float32)
                color_b = np.array(sb.color, dtype=np.float32)
                color = (1.0 - alpha) * color_a + alpha * color_b
            elif sa is not None:
                intensity = sa.intensity
                color = np.array(sa.color, dtype=np.float32)
            else:  # sa is None, sb is not None
                intensity = sb.intensity
                color = np.array(sb.color, dtype=np.float32)
            rgb = color * intensity  # 0..255 in float
            cell_grid[row, col] += rgb  # additive blending across fixtures

        np.clip(cell_grid, 0.0, 255.0, out=cell_grid)
        return _upscale_bgr(cell_grid, self.CELL_PX)

    # ------------------------------------------------------------------
    @staticmethod
    def _mux_audio_h264(
        video_in: Path, audio_in: Path, video_out: Path
    ) -> None:
        """Re-encode video to H.264 + AAC and mux source audio."""
        if not audio_in.exists():
            raise FileNotFoundError(f"Audio file not found: {audio_in}")
        cmd = [
            "ffmpeg",
            "-y",
            "-loglevel",
            "error",
            "-i",
            str(video_in),
            "-i",
            str(audio_in),
            "-c:v",
            "libx264",
            "-pix_fmt",
            "yuv420p",
            "-c:a",
            "aac",
            "-b:a",
            "128k",
            "-shortest",
            "-movflags",
            "+faststart",
            str(video_out),
        ]
        result = subprocess.run(cmd, capture_output=True, text=True)
        if result.returncode != 0:
            raise RuntimeError(
                f"ffmpeg mux failed (rc={result.returncode}): "
                f"{(result.stderr or '')[-1000:]}"
            )

    @staticmethod
    def _reencode_h264(video_in: Path, video_out: Path) -> None:
        """Re-encode the silent video to H.264 (browser-compatible)."""
        cmd = [
            "ffmpeg",
            "-y",
            "-loglevel",
            "error",
            "-i",
            str(video_in),
            "-c:v",
            "libx264",
            "-pix_fmt",
            "yuv420p",
            "-movflags",
            "+faststart",
            str(video_out),
        ]
        result = subprocess.run(cmd, capture_output=True, text=True)
        if result.returncode != 0:
            raise RuntimeError(
                f"ffmpeg re-encode failed (rc={result.returncode}): "
                f"{(result.stderr or '')[-1000:]}"
            )


# ---------------------------------------------------------------------------
def _upscale_bgr(cell_grid: np.ndarray, cell_px: int) -> np.ndarray:
    """Tile each grid cell to cell_px x cell_px, return BGR uint8 array."""
    # cell_grid is HxWx3 in RGB float. Nearest-neighbor upscale.
    rgb_full = np.repeat(
        np.repeat(cell_grid, cell_px, axis=0), cell_px, axis=1
    ).astype(np.uint8)
    # OpenCV expects BGR
    return rgb_full[:, :, ::-1].copy()


# ---------------------------------------------------------------------------
def render_from_run_dir(
    output_dir: Path,
    audio_path: Path | None = None,
    output_video: Path | None = None,
    fps: int = GridRenderer.DEFAULT_FPS,
) -> Path:
    """Convenience: load a run directory, render to `preview.mp4`.

    Expects `stage_layout.json` and `cue_list.json` in `output_dir`.
    """
    from mli_synthetics.llm.designer import CueList
    from mli_synthetics.stage.fixtures import StageLayout

    output_dir = Path(output_dir)
    stage_path = output_dir / "stage_layout.json"
    cue_path = output_dir / "cue_list.json"
    if not stage_path.exists() or not cue_path.exists():
        raise FileNotFoundError(
            f"Missing stage_layout.json or cue_list.json under {output_dir}"
        )
    stage = StageLayout.model_validate(json.loads(stage_path.read_text(encoding="utf-8")))
    cue_list = CueList.model_validate(json.loads(cue_path.read_text(encoding="utf-8")))

    if cue_list.cues:
        last = cue_list.cues[-1]
        duration_s = last.time_s + last.duration_s
    else:
        raise ValueError("Cue list is empty")

    if output_video is None:
        output_video = output_dir / "preview.mp4"

    renderer = GridRenderer(stage, cue_list, fps=fps)
    return renderer.render_to_video(
        output_video, duration_s=duration_s, audio_path=audio_path
    )
