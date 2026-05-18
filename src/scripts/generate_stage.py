"""CLI: generate a random stage layout."""
from __future__ import annotations

import json
import sys
from pathlib import Path

# Allow running as a standalone script
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import typer
from rich.console import Console

from mli_synthetics.logging_config import configure_logging
from mli_synthetics.stage.fixtures import VenueSize
from mli_synthetics.stage.generator import StageGenerator, render_ascii

app = typer.Typer(add_completion=False, help="Generate a random stage layout.")
console = Console()


@app.command()
def main(
    genre: str = typer.Option(None, help="edm | rock | pop | acoustic | theater | hip_hop | metal"),
    venue_size: str = typer.Option(None, "--venue-size", help="small_club | medium_club | large_venue | festival"),
    random_full: bool = typer.Option(False, "--random", help="Fully random layout"),
    output: Path = typer.Option(None, "--output", help="JSON output path (default stdout)"),
    visualize: bool = typer.Option(True, "--visualize/--no-visualize", help="Print ASCII visualization"),
    seed: int = typer.Option(None, help="Random seed"),
):
    configure_logging()
    g = StageGenerator(seed=seed)

    if random_full:
        genre = None
        venue_size = None

    venue: VenueSize | None = None
    if venue_size:
        try:
            venue = VenueSize(venue_size)
        except ValueError:
            console.print(f"[red]Unknown venue_size '{venue_size}'[/red]")
            raise typer.Exit(1)

    layout = g.generate(target_genre=genre, venue_size=venue)
    payload = layout.model_dump(mode="json")
    text = json.dumps(payload, indent=2)

    if output:
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(text, encoding="utf-8")
        console.print(f"[green]Wrote[/green] {output}")
    else:
        console.print_json(text)

    if visualize:
        console.print()
        console.print(render_ascii(layout))
        console.print()
        counts = layout.fixture_count_by_category
        total = sum(counts.values())
        console.print(f"[bold]Total fixtures:[/bold] {total}")
        for cat, n in sorted(counts.items(), key=lambda kv: -kv[1]):
            console.print(f"  {cat:<18} {n}")


if __name__ == "__main__":
    app()
