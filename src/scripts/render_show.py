"""CLI: render an MLI 12x8 preview video from a Phase 1 output directory."""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import typer
from rich.console import Console

from mli_synthetics.logging_config import configure_logging
from mli_synthetics.renderer.grid_renderer import render_from_run_dir

app = typer.Typer(add_completion=False, help="Render an MLI 12x8 grid preview video.")
console = Console()


@app.command()
def main(
    output_dir: Path = typer.Option(
        ...,
        "--output-dir",
        exists=True,
        file_okay=False,
        readable=True,
        help="Phase 1 run directory containing stage_layout.json + cue_list.json",
    ),
    audio: Path = typer.Option(
        None,
        "--audio",
        exists=True,
        readable=True,
        help="Source audio file (WAV/MP3) to mux into the MP4. Optional.",
    ),
    output: Path = typer.Option(
        None,
        "--output",
        help="Output MP4 path. Defaults to <output_dir>/preview.mp4.",
    ),
    fps: int = typer.Option(30, "--fps", help="Output frame rate."),
):
    configure_logging()
    video_path = render_from_run_dir(
        output_dir=output_dir,
        audio_path=audio,
        output_video=output,
        fps=fps,
    )
    console.print(f"[green]Wrote[/green] {video_path}")


if __name__ == "__main__":
    app()
