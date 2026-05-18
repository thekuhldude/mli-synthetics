"""Parse fixture_library.md into structured data.

This is intentionally tolerant: if the MD file is missing or malformed,
we fall back to a hardcoded minimal library so generation still works.
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field
from pathlib import Path

from mli_synthetics.logging_config import get_logger
from mli_synthetics.stage.fixtures import FixtureCategory, Position

logger = get_logger()


# Mapping from MD heading names to FixtureCategory
_HEADING_TO_CATEGORY: dict[str, FixtureCategory] = {
    "MOVING HEAD BEAM": FixtureCategory.MOVING_BEAM,
    "MOVING HEAD SPOT": FixtureCategory.MOVING_SPOT,
    "MOVING HEAD WASH": FixtureCategory.MOVING_WASH,
    "LED PAR": FixtureCategory.LED_PAR,
    "BLINDER": FixtureCategory.BLINDER,
    "STROBE": FixtureCategory.STROBE,
    "PIXEL BAR": FixtureCategory.PIXEL_BAR,
    "FOLLOWSPOT": FixtureCategory.FOLLOWSPOT,
    "FOLLOW SPOT": FixtureCategory.FOLLOWSPOT,
    "LASER": FixtureCategory.LASER,
    "HAZER": FixtureCategory.HAZER,
    "HAZE": FixtureCategory.HAZER,
    "CO2": FixtureCategory.CO2_JET,
    "CO2 JET": FixtureCategory.CO2_JET,
    "PYRO": FixtureCategory.PYRO,
}


@dataclass
class FixtureSpec:
    category: FixtureCategory
    description: str = ""
    capabilities: str = ""
    typical_positions: list[Position] = field(default_factory=list)
    music_pairing: str = ""
    behavior_patterns: str = ""
    raw_text: str = ""


# Default genre distributions (category -> ratio). Tuned from the
# fixture_library.md guidance. Each row sums approximately to 1.0.
_DEFAULT_GENRE_DISTRIBUTIONS: dict[str, dict[FixtureCategory, float]] = {
    "edm": {
        FixtureCategory.MOVING_BEAM: 0.40,
        FixtureCategory.MOVING_WASH: 0.25,
        FixtureCategory.PIXEL_BAR: 0.15,
        FixtureCategory.STROBE: 0.10,
        FixtureCategory.BLINDER: 0.05,
        FixtureCategory.LASER: 0.05,
    },
    "rock": {
        FixtureCategory.MOVING_SPOT: 0.30,
        FixtureCategory.MOVING_WASH: 0.25,
        FixtureCategory.LED_PAR: 0.20,
        FixtureCategory.BLINDER: 0.10,
        FixtureCategory.STROBE: 0.10,
        FixtureCategory.MOVING_BEAM: 0.05,
    },
    "pop": {
        FixtureCategory.MOVING_SPOT: 0.25,
        FixtureCategory.MOVING_WASH: 0.25,
        FixtureCategory.MOVING_BEAM: 0.20,
        FixtureCategory.LED_PAR: 0.15,
        FixtureCategory.PIXEL_BAR: 0.10,
        FixtureCategory.BLINDER: 0.05,
    },
    "acoustic": {
        FixtureCategory.LED_PAR: 0.40,
        FixtureCategory.MOVING_SPOT: 0.30,
        FixtureCategory.MOVING_WASH: 0.20,
        FixtureCategory.FOLLOWSPOT: 0.10,
    },
    "theater": {
        FixtureCategory.MOVING_SPOT: 0.35,
        FixtureCategory.LED_PAR: 0.25,
        FixtureCategory.FOLLOWSPOT: 0.20,
        FixtureCategory.MOVING_WASH: 0.15,
        FixtureCategory.MOVING_BEAM: 0.05,
    },
    "hip_hop": {
        FixtureCategory.MOVING_BEAM: 0.30,
        FixtureCategory.MOVING_WASH: 0.25,
        FixtureCategory.STROBE: 0.15,
        FixtureCategory.BLINDER: 0.10,
        FixtureCategory.PIXEL_BAR: 0.10,
        FixtureCategory.LED_PAR: 0.10,
    },
    "metal": {
        FixtureCategory.MOVING_SPOT: 0.25,
        FixtureCategory.MOVING_BEAM: 0.20,
        FixtureCategory.STROBE: 0.20,
        FixtureCategory.BLINDER: 0.15,
        FixtureCategory.MOVING_WASH: 0.15,
        FixtureCategory.LED_PAR: 0.05,
    },
    "other": {
        FixtureCategory.MOVING_SPOT: 0.25,
        FixtureCategory.MOVING_WASH: 0.25,
        FixtureCategory.LED_PAR: 0.20,
        FixtureCategory.MOVING_BEAM: 0.15,
        FixtureCategory.BLINDER: 0.10,
        FixtureCategory.STROBE: 0.05,
    },
}


_POSITION_KEYWORDS: dict[str, Position] = {
    "front truss": Position.FRONT_TRUSS,
    "foh": Position.FRONT_TRUSS,
    "mid truss": Position.MID_TRUSS,
    "mid-stage truss": Position.MID_TRUSS,
    "back truss": Position.BACK_TRUSS,
    "backline": Position.BACKLINE,
    "floor": Position.STAGE_FLOOR,
    "side": Position.SIDE_LEFT,  # we'll mirror later
    "audience-facing": Position.AUDIENCE_FACING,
    "facing audience": Position.AUDIENCE_FACING,
}


class FixtureLibrary:
    """In-memory representation of the fixture knowledge base."""

    def __init__(
        self,
        specs: dict[FixtureCategory, FixtureSpec],
        raw_markdown: str = "",
    ):
        self.specs = specs
        self.raw_markdown = raw_markdown

    # ------------------------------------------------------------------
    # Loading
    # ------------------------------------------------------------------
    @classmethod
    def load(cls, path: Path | None = None) -> FixtureLibrary:
        from mli_synthetics.settings import get_settings

        if path is None:
            path = get_settings().knowledge_base_dir / "fixture_library.md"

        if not path.exists():
            logger.warning(
                "Fixture library not found at {}; using hardcoded minimal library",
                path,
            )
            return cls._minimal_fallback()

        try:
            text = path.read_text(encoding="utf-8")
        except Exception as exc:  # noqa: BLE001
            logger.warning("Failed to read fixture library: {}", exc)
            return cls._minimal_fallback()

        specs = cls._parse_markdown(text)
        if not specs:
            logger.warning(
                "Fixture library parsed empty; using hardcoded minimal library"
            )
            return cls._minimal_fallback()
        logger.info("Loaded {} fixture categories from {}", len(specs), path.name)
        return cls(specs=specs, raw_markdown=text)

    @classmethod
    def _minimal_fallback(cls) -> FixtureLibrary:
        specs = {cat: FixtureSpec(category=cat) for cat in FixtureCategory}
        return cls(specs=specs, raw_markdown="")

    @staticmethod
    def _parse_markdown(text: str) -> dict[FixtureCategory, FixtureSpec]:
        # Sections begin with "## N. NAME" headings.
        section_re = re.compile(r"^##\s+\d+\.\s+(.+?)\s*$", re.MULTILINE)
        matches = list(section_re.finditer(text))
        specs: dict[FixtureCategory, FixtureSpec] = {}
        for i, m in enumerate(matches):
            heading = m.group(1).strip().upper()
            category: FixtureCategory | None = None
            for key, cat in _HEADING_TO_CATEGORY.items():
                if key in heading:
                    category = cat
                    break
            if category is None:
                continue
            start = m.end()
            end = matches[i + 1].start() if i + 1 < len(matches) else len(text)
            body = text[start:end].strip()
            spec = FixtureSpec(category=category, raw_text=body)
            spec.description = _extract_field(body, "Description")
            spec.capabilities = _extract_field(body, "Capabilities")
            spec.music_pairing = _extract_field(body, "Music Pairing")
            spec.behavior_patterns = _extract_field(body, "Behavior Patterns")
            typical = _extract_field(body, "Typical Position")
            spec.typical_positions = _infer_positions(typical)
            specs[category] = spec
        return specs

    # ------------------------------------------------------------------
    # Lookups
    # ------------------------------------------------------------------
    def get_category(self, category: str | FixtureCategory) -> FixtureSpec | None:
        if isinstance(category, str):
            try:
                category = FixtureCategory(category)
            except ValueError:
                return None
        return self.specs.get(category)

    def genre_distribution(self, genre: str) -> dict[FixtureCategory, float]:
        key = genre.lower().strip()
        return dict(_DEFAULT_GENRE_DISTRIBUTIONS.get(key, _DEFAULT_GENRE_DISTRIBUTIONS["other"]))

    def position_rules(self, category: FixtureCategory) -> list[Position]:
        spec = self.specs.get(category)
        if spec and spec.typical_positions:
            return list(spec.typical_positions)
        return _DEFAULT_POSITION_RULES.get(category, [Position.MID_TRUSS])


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def _extract_field(body: str, field_name: str) -> str:
    """Find a `**Field:** value` line within an MD block."""
    pattern = rf"\*\*{re.escape(field_name)}:?\*\*\s*(.+?)(?=\n\*\*|\n## |\Z)"
    m = re.search(pattern, body, re.DOTALL)
    return m.group(1).strip() if m else ""


def _infer_positions(text: str) -> list[Position]:
    if not text:
        return []
    lower = text.lower()
    found: list[Position] = []
    for kw, pos in _POSITION_KEYWORDS.items():
        if kw in lower and pos not in found:
            found.append(pos)
    return found


_DEFAULT_POSITION_RULES: dict[FixtureCategory, list[Position]] = {
    FixtureCategory.MOVING_BEAM: [Position.BACK_TRUSS, Position.MID_TRUSS, Position.STAGE_FLOOR],
    FixtureCategory.MOVING_SPOT: [Position.FRONT_TRUSS, Position.MID_TRUSS, Position.SIDE_LEFT, Position.SIDE_RIGHT],
    FixtureCategory.MOVING_WASH: [Position.MID_TRUSS, Position.BACK_TRUSS, Position.FRONT_TRUSS],
    FixtureCategory.LED_PAR: [Position.STAGE_FLOOR, Position.BACKLINE, Position.SIDE_LEFT, Position.SIDE_RIGHT],
    FixtureCategory.BLINDER: [Position.FRONT_TRUSS, Position.AUDIENCE_FACING],
    FixtureCategory.STROBE: [Position.MID_TRUSS, Position.BACK_TRUSS, Position.SIDE_LEFT, Position.SIDE_RIGHT],
    FixtureCategory.PIXEL_BAR: [Position.BACK_TRUSS, Position.MID_TRUSS, Position.STAGE_FLOOR],
    FixtureCategory.FOLLOWSPOT: [Position.FRONT_TRUSS, Position.AUDIENCE_FACING],
    FixtureCategory.LASER: [Position.BACK_TRUSS, Position.MID_TRUSS],
    FixtureCategory.HAZER: [Position.STAGE_FLOOR, Position.BACKLINE],
    FixtureCategory.CO2_JET: [Position.STAGE_FLOOR, Position.BACKLINE, Position.FRONT_TRUSS],
    FixtureCategory.PYRO: [Position.STAGE_FLOOR, Position.BACKLINE],
}
