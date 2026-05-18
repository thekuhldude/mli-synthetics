"""End-to-end Phase 1 pipeline."""
from __future__ import annotations

import json
import time
from datetime import datetime
from pathlib import Path
from typing import Any

from mli_synthetics.audio.analyzer import SongAnalyzer
from mli_synthetics.llm.designer import LightingDesignerLLM
from mli_synthetics.logging_config import get_logger
from mli_synthetics.stage.fixtures import VenueSize
from mli_synthetics.stage.generator import StageGenerator, render_ascii

logger = get_logger()


class Phase1Pipeline:
    def __init__(self, settings: Any | None = None):
        if settings is None:
            from mli_synthetics.settings import get_settings

            settings = get_settings()
        self.settings = settings

    async def generate_show(
        self,
        audio_path: Path,
        output_dir: Path | None = None,
        force_genre: str | None = None,
        force_venue_size: str | None = None,
    ) -> dict:
        if output_dir is None:
            stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            output_dir = self.settings.outputs_dir / f"run_{stamp}"
        output_dir.mkdir(parents=True, exist_ok=True)
        logger.info("Pipeline output dir: {}", output_dir)

        timings: dict[str, float] = {}

        # 1. Analyze song
        t0 = time.perf_counter()
        analyzer = SongAnalyzer(settings=self.settings)
        analysis_result = await analyzer.analyze(audio_path, skip_llm=False)
        timings["analyze"] = time.perf_counter() - t0
        (output_dir / "song_analysis.json").write_text(
            json.dumps(
                {
                    "numerical": analysis_result.numerical.model_dump(),
                    "interpreted": (
                        analysis_result.interpreted.model_dump()
                        if analysis_result.interpreted
                        else None
                    ),
                },
                indent=2,
            ),
            encoding="utf-8",
        )

        if analysis_result.interpreted is None:
            raise RuntimeError(
                "LLM song analysis failed; check Ollama and the model is pulled."
            )

        # 2. Determine genre + venue
        genre = force_genre or analysis_result.interpreted.genre_estimate
        venue: VenueSize | None = None
        if force_venue_size:
            venue = VenueSize(force_venue_size)

        # 3. Generate stage
        t0 = time.perf_counter()
        gen = StageGenerator(
            min_fixtures=self.settings.min_fixtures,
            max_fixtures=self.settings.max_fixtures,
        )
        layout = gen.generate(target_genre=genre, venue_size=venue)
        timings["stage"] = time.perf_counter() - t0
        (output_dir / "stage_layout.json").write_text(
            json.dumps(layout.model_dump(mode="json"), indent=2), encoding="utf-8"
        )
        (output_dir / "stage_layout.txt").write_text(
            render_ascii(layout), encoding="utf-8"
        )

        # 4. Design cue list
        t0 = time.perf_counter()
        designer = LightingDesignerLLM(settings=self.settings)
        cue_list = await designer.design_show(layout, analysis_result.interpreted)
        timings["design"] = time.perf_counter() - t0
        (output_dir / "cue_list.json").write_text(
            json.dumps(cue_list.model_dump(), indent=2), encoding="utf-8"
        )

        # 5. Summary
        summary = {
            "audio": str(audio_path),
            "output_dir": str(output_dir),
            "song": {
                "duration_s": round(analysis_result.numerical.duration_s, 2),
                "tempo_bpm": round(analysis_result.numerical.tempo_bpm, 1),
                "key": analysis_result.numerical.key,
                "genre": analysis_result.interpreted.genre_estimate,
                "mood": analysis_result.interpreted.mood,
                "energy_profile": analysis_result.interpreted.energy_profile,
                "n_sections": len(analysis_result.interpreted.structure),
            },
            "stage": {
                "venue_size": layout.venue_size.value,
                "dimensions_m": [layout.width_m, layout.depth_m, layout.height_m],
                "total_fixtures": len(layout.fixtures),
                "by_category": layout.fixture_count_by_category,
            },
            "design": {
                "n_cues": len(cue_list.cues),
                "designer_notes": cue_list.metadata.designer_notes,
                "color_palette": cue_list.metadata.color_palette,
                "intensity_arc": cue_list.metadata.intensity_arc,
            },
            "timings_s": {k: round(v, 2) for k, v in timings.items()},
        }
        (output_dir / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
        (output_dir / "summary.md").write_text(_render_summary_md(summary), encoding="utf-8")
        return summary


def _render_summary_md(s: dict) -> str:
    lines = [
        f"# Phase 1 Run Summary",
        "",
        f"**Audio:** `{s['audio']}`",
        f"**Output:** `{s['output_dir']}`",
        "",
        "## Song",
        f"- Duration: {s['song']['duration_s']}s",
        f"- Tempo: {s['song']['tempo_bpm']} BPM",
        f"- Key: {s['song']['key']}",
        f"- Genre: {s['song']['genre']}",
        f"- Mood: {s['song']['mood']}",
        f"- Energy profile: {s['song']['energy_profile']}",
        f"- Sections: {s['song']['n_sections']}",
        "",
        "## Stage",
        f"- Venue: {s['stage']['venue_size']}",
        f"- Dimensions: {s['stage']['dimensions_m']} m",
        f"- Total fixtures: {s['stage']['total_fixtures']}",
        "",
        "| Category | Count |",
        "|----------|-------|",
    ]
    for cat, n in sorted(s["stage"]["by_category"].items(), key=lambda kv: -kv[1]):
        lines.append(f"| {cat} | {n} |")
    lines.extend(
        [
            "",
            "## Design",
            f"- Cues: {s['design']['n_cues']}",
            f"- Color palette: {', '.join(s['design']['color_palette'])}",
            f"- Designer notes: {s['design']['designer_notes']}",
            f"- Intensity arc: {s['design']['intensity_arc']}",
            "",
            "## Timings (seconds)",
        ]
    )
    for k, v in s["timings_s"].items():
        lines.append(f"- {k}: {v}")
    return "\n".join(lines)
