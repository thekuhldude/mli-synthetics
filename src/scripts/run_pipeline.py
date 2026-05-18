"""CLI: run the full Phase 1 pipeline on an audio file."""
from __future__ import annotations

import asyncio
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import typer
from rich.console import Console

from mli_synthetics.logging_config import configure_logging
from mli_synthetics.pipeline.orchestrator import Phase1Pipeline

app = typer.Typer(add_completion=False, help="Run the full Phase 1 pipeline.")
console = Console()


@app.command()
def main(
    audio_file: Path = typer.Argument(..., exists=True, readable=True),
    output_dir: Path = typer.Option(None, "--output-dir"),
    force_genre: str = typer.Option(None, "--force-genre"),
    force_venue: str = typer.Option(None, "--force-venue"),
):
    configure_logging()
    pipeline = Phase1Pipeline()
    summary = asyncio.run(
        pipeline.generate_show(
            audio_path=audio_file,
            output_dir=output_dir,
            force_genre=force_genre,
            force_venue_size=force_venue,
        )
    )
    console.print_json(json.dumps(summary))
    console.print(f"\n[green]Done.[/green] Outputs in {summary['output_dir']}")


if __name__ == "__main__":
    app()
