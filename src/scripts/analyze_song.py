"""CLI: analyze a song into numerical features (+ optional LLM)."""
from __future__ import annotations

import asyncio
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import typer
from rich.console import Console

from mli_synthetics.audio.analyzer import SongAnalyzer
from mli_synthetics.logging_config import configure_logging

app = typer.Typer(add_completion=False, help="Analyze a song.")
console = Console()


@app.command()
def main(
    audio_file: Path = typer.Argument(..., exists=True, readable=True),
    output: Path = typer.Option(None, "--output", help="Where to save analysis JSON"),
    skip_llm: bool = typer.Option(False, "--skip-llm", help="Only numerical features"),
):
    configure_logging()
    analyzer = SongAnalyzer()
    result = asyncio.run(analyzer.analyze(audio_file, skip_llm=skip_llm))

    payload = {
        "numerical_summary": result.numerical.summary(),
        "interpreted": result.interpreted.model_dump() if result.interpreted else None,
    }
    text = json.dumps(payload, indent=2)

    if output:
        output.parent.mkdir(parents=True, exist_ok=True)
        # Write the FULL features alongside, but keep top-level summary
        full = {
            "numerical": result.numerical.model_dump(),
            "interpreted": result.interpreted.model_dump() if result.interpreted else None,
        }
        output.write_text(json.dumps(full, indent=2), encoding="utf-8")
        console.print(f"[green]Wrote[/green] {output}")
    console.print_json(text)


if __name__ == "__main__":
    app()
