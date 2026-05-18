"""CLI: design a lighting cue list given stage + analysis JSON."""
from __future__ import annotations

import asyncio
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import typer
from rich.console import Console

from mli_synthetics.audio.structure import SongAnalysis
from mli_synthetics.llm.designer import LightingDesignerLLM
from mli_synthetics.logging_config import configure_logging
from mli_synthetics.stage.fixtures import StageLayout

app = typer.Typer(add_completion=False, help="Design a lighting cue list.")
console = Console()


@app.command()
def main(
    stage: Path = typer.Option(..., "--stage", exists=True, readable=True),
    analysis: Path = typer.Option(..., "--analysis", exists=True, readable=True),
    output: Path = typer.Option(..., "--output", help="Output cue list JSON"),
):
    configure_logging()
    stage_data = json.loads(stage.read_text(encoding="utf-8"))
    layout = StageLayout.model_validate(stage_data)

    analysis_data = json.loads(analysis.read_text(encoding="utf-8"))
    # analysis JSON could be top-level SongAnalysis or wrapped under "interpreted"
    if "interpreted" in analysis_data and analysis_data["interpreted"]:
        analysis_data = analysis_data["interpreted"]
    song_analysis = SongAnalysis.model_validate(analysis_data)

    designer = LightingDesignerLLM()
    cue_list = asyncio.run(designer.design_show(layout, song_analysis))

    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(cue_list.model_dump(), indent=2), encoding="utf-8")
    console.print(f"[green]Wrote[/green] {output} ({len(cue_list.cues)} cues)")
    console.print(f"Designer notes: {cue_list.metadata.designer_notes}")
    console.print(f"Color palette: {', '.join(cue_list.metadata.color_palette)}")


if __name__ == "__main__":
    app()
