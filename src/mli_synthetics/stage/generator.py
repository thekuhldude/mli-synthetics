"""Rule-based, randomized stage layout generator.

The generator picks venue dimensions, distributes fixtures across
categories per genre, and places them within position zones with
mirrored pairs + small jitter to avoid grid artifacts.
"""
from __future__ import annotations

import random
from dataclasses import dataclass

from mli_synthetics.errors import StageGenerationError
from mli_synthetics.logging_config import get_logger
from mli_synthetics.stage.fixtures import (
    Fixture,
    FixtureCategory,
    Position,
    StageLayout,
    VenueSize,
)
from mli_synthetics.stage.library import FixtureLibrary

logger = get_logger()


@dataclass
class _VenueSpec:
    width_range: tuple[float, float]
    depth_range: tuple[float, float]
    height_range: tuple[float, float]
    fixture_range: tuple[int, int]


_VENUE_SPECS: dict[VenueSize, _VenueSpec] = {
    VenueSize.SMALL_CLUB: _VenueSpec((4, 8), (3, 5), (3, 4), (8, 20)),
    VenueSize.MEDIUM_CLUB: _VenueSpec((8, 15), (5, 8), (4, 6), (20, 50)),
    VenueSize.LARGE_VENUE: _VenueSpec((15, 25), (8, 15), (6, 12), (50, 150)),
    VenueSize.FESTIVAL: _VenueSpec((25, 40), (15, 25), (10, 18), (100, 400)),
}

_VENUE_WEIGHTS = {
    VenueSize.SMALL_CLUB: 0.35,
    VenueSize.MEDIUM_CLUB: 0.35,
    VenueSize.LARGE_VENUE: 0.20,
    VenueSize.FESTIVAL: 0.10,
}

_GENRE_WEIGHTS = {
    "edm": 0.20,
    "rock": 0.20,
    "pop": 0.20,
    "acoustic": 0.10,
    "theater": 0.05,
    "hip_hop": 0.10,
    "metal": 0.10,
    "other": 0.05,
}


class StageGenerator:
    def __init__(
        self,
        library: FixtureLibrary | None = None,
        seed: int | None = None,
        min_fixtures: int = 8,
        max_fixtures: int = 80,
        max_fixtures_total: int = 20,
    ):
        self.library = library or FixtureLibrary.load()
        self.rng = random.Random(seed)
        self.min_fixtures = min_fixtures
        self.max_fixtures = max_fixtures
        # Hard ceiling regardless of venue size; allocation scales down.
        self.max_fixtures_total = max_fixtures_total

    # ------------------------------------------------------------------
    def generate(
        self,
        target_genre: str | None = None,
        venue_size: VenueSize | str | None = None,
    ) -> StageLayout:
        genre = target_genre or self._weighted_choice(_GENRE_WEIGHTS)
        venue = self._resolve_venue(venue_size)

        spec = _VENUE_SPECS[venue]
        width = self.rng.uniform(*spec.width_range)
        depth = self.rng.uniform(*spec.depth_range)
        height = self.rng.uniform(*spec.height_range)

        fixture_target = self.rng.randint(*spec.fixture_range)
        fixture_target = max(self.min_fixtures, min(self.max_fixtures, fixture_target))
        # Hard cap: never exceed max_fixtures_total. Allocation ratios in
        # _allocate_counts scale down proportionally to fit this budget.
        # Reserve one slot for a potential auto-injected hazer.
        hazer_reserve = 1 if fixture_target >= self.max_fixtures_total else 0
        fixture_target = min(fixture_target, self.max_fixtures_total - hazer_reserve)
        fixture_target = max(self.min_fixtures, fixture_target)

        distribution = self.library.genre_distribution(genre)
        counts = self._allocate_counts(distribution, fixture_target)

        fixtures: list[Fixture] = []
        for category, count in counts.items():
            if count <= 0:
                continue
            fixtures.extend(
                self._place_fixtures(category, count, width, depth, height)
            )

        # Ensure hazer if beams/lasers/spots present
        beam_like = {FixtureCategory.MOVING_BEAM, FixtureCategory.LASER, FixtureCategory.MOVING_SPOT}
        has_beam_like = any(f.category in beam_like for f in fixtures)
        has_hazer = any(f.category == FixtureCategory.HAZER for f in fixtures)
        if has_beam_like and not has_hazer:
            # If we're already at the cap, drop the largest category by one
            # to make room for the hazer.
            if len(fixtures) >= self.max_fixtures_total:
                counts: dict[FixtureCategory, int] = {}
                for f in fixtures:
                    counts[f.category] = counts.get(f.category, 0) + 1
                if counts:
                    largest_cat = max(counts, key=lambda c: counts[c])
                    for i in range(len(fixtures) - 1, -1, -1):
                        if fixtures[i].category == largest_cat:
                            fixtures.pop(i)
                            break
            fixtures.extend(
                self._place_fixtures(FixtureCategory.HAZER, 1, width, depth, height)
            )

        # Hard ceiling safety net - never exceed max_fixtures_total.
        if len(fixtures) > self.max_fixtures_total:
            logger.warning(
                "Trimming stage from {} to max_fixtures_total={}",
                len(fixtures),
                self.max_fixtures_total,
            )
            fixtures = fixtures[: self.max_fixtures_total]

        if not fixtures:
            raise StageGenerationError("Generated stage has zero fixtures")

        layout = StageLayout(
            width_m=round(width, 2),
            depth_m=round(depth, 2),
            height_m=round(height, 2),
            venue_size=venue,
            target_genre=genre,
            fixtures=fixtures,
        )
        self._validate(layout)
        return layout

    # ------------------------------------------------------------------
    def _resolve_venue(self, venue_size: VenueSize | str | None) -> VenueSize:
        if venue_size is None:
            return self._weighted_choice(_VENUE_WEIGHTS)
        if isinstance(venue_size, VenueSize):
            return venue_size
        try:
            return VenueSize(venue_size)
        except ValueError as exc:
            raise StageGenerationError(f"Unknown venue size: {venue_size}") from exc

    def _weighted_choice(self, weights: dict):
        items = list(weights.items())
        keys = [k for k, _ in items]
        wts = [w for _, w in items]
        return self.rng.choices(keys, weights=wts, k=1)[0]

    def _allocate_counts(
        self,
        distribution: dict[FixtureCategory, float],
        total: int,
    ) -> dict[FixtureCategory, int]:
        if not distribution:
            return {}
        s = sum(distribution.values()) or 1.0
        normalized = {k: v / s for k, v in distribution.items()}
        # Initial floor allocation
        raw = {cat: ratio * total for cat, ratio in normalized.items()}
        counts = {cat: int(v) for cat, v in raw.items()}
        # Distribute remainder by fractional part, highest first
        remainder = total - sum(counts.values())
        if remainder > 0:
            fractions = sorted(
                ((cat, raw[cat] - counts[cat]) for cat in raw),
                key=lambda kv: kv[1],
                reverse=True,
            )
            for cat, _ in fractions[:remainder]:
                counts[cat] += 1
        # Drop zero categories
        return {c: n for c, n in counts.items() if n > 0}

    # ------------------------------------------------------------------
    def _place_fixtures(
        self,
        category: FixtureCategory,
        count: int,
        width: float,
        depth: float,
        height: float,
    ) -> list[Fixture]:
        positions = self.library.position_rules(category)
        if not positions:
            positions = [Position.MID_TRUSS]
        # Mirror SIDE_LEFT into both sides if only one was inferred
        if Position.SIDE_LEFT in positions and Position.SIDE_RIGHT not in positions:
            idx = positions.index(Position.SIDE_LEFT)
            positions = positions[:idx] + [Position.SIDE_LEFT, Position.SIDE_RIGHT] + positions[idx + 1 :]

        placed: list[Fixture] = []
        # Distribute count across allowed positions evenly
        per_position = max(1, count // len(positions))
        remaining = count
        for pos in positions:
            if remaining <= 0:
                break
            n_here = min(per_position, remaining)
            placed.extend(
                self._place_in_zone(category, pos, n_here, width, depth, height, len(placed))
            )
            remaining -= n_here
        # Remainder: spread back onto first position
        if remaining > 0:
            placed.extend(
                self._place_in_zone(
                    category, positions[0], remaining, width, depth, height, len(placed)
                )
            )
        return placed

    def _place_in_zone(
        self,
        category: FixtureCategory,
        position: Position,
        count: int,
        width: float,
        depth: float,
        height: float,
        id_offset: int,
    ) -> list[Fixture]:
        if count <= 0:
            return []

        # Define zone bounds (x in [-width/2, width/2], y in [0, depth])
        half_w = width / 2.0
        zone_x: tuple[float, float]
        zone_y: tuple[float, float]
        zone_z: tuple[float, float]
        if position == Position.FRONT_TRUSS:
            zone_x, zone_y, zone_z = (-half_w, half_w), (0.0, 0.5), (height - 0.5, height)
        elif position == Position.MID_TRUSS:
            zone_x, zone_y, zone_z = (-half_w, half_w), (depth * 0.4, depth * 0.6), (height - 0.5, height)
        elif position == Position.BACK_TRUSS:
            zone_x, zone_y, zone_z = (-half_w, half_w), (depth - 0.5, depth), (height - 0.5, height)
        elif position == Position.STAGE_FLOOR:
            zone_x, zone_y, zone_z = (-half_w * 0.9, half_w * 0.9), (0.5, depth - 0.5), (0.0, 0.3)
        elif position == Position.SIDE_LEFT:
            zone_x, zone_y, zone_z = (-half_w, -half_w * 0.9), (0.0, depth), (1.0, height - 1.0)
        elif position == Position.SIDE_RIGHT:
            zone_x, zone_y, zone_z = (half_w * 0.9, half_w), (0.0, depth), (1.0, height - 1.0)
        elif position == Position.BACKLINE:
            zone_x, zone_y, zone_z = (-half_w * 0.9, half_w * 0.9), (depth - 0.5, depth), (0.0, 0.5)
        elif position == Position.AUDIENCE_FACING:
            zone_x, zone_y, zone_z = (-half_w, half_w), (-0.5, 0.0), (height * 0.7, height)
        else:
            zone_x, zone_y, zone_z = (-half_w, half_w), (0.0, depth), (height - 0.5, height)

        fixtures: list[Fixture] = []
        symmetric = position in {
            Position.FRONT_TRUSS,
            Position.MID_TRUSS,
            Position.BACK_TRUSS,
            Position.BACKLINE,
            Position.AUDIENCE_FACING,
        }

        if symmetric and count >= 2:
            # Mirrored pairs along x-axis
            pairs = count // 2
            x_step = (zone_x[1] - zone_x[0]) / (pairs * 2 + 1)
            for i in range(pairs):
                offset = x_step * (i + 1)
                x_left = zone_x[0] + offset
                x_right = zone_x[1] - offset
                for x in (x_left, x_right):
                    fixtures.append(
                        self._make_fixture(category, position, x, zone_y, zone_z, id_offset + len(fixtures))
                    )
            # Odd one in center
            if count % 2 == 1:
                fixtures.append(
                    self._make_fixture(category, position, 0.0, zone_y, zone_z, id_offset + len(fixtures))
                )
        else:
            # Even distribution along x
            for i in range(count):
                if count == 1:
                    x = (zone_x[0] + zone_x[1]) / 2.0
                else:
                    x = zone_x[0] + (zone_x[1] - zone_x[0]) * i / (count - 1)
                fixtures.append(
                    self._make_fixture(category, position, x, zone_y, zone_z, id_offset + len(fixtures))
                )

        return fixtures

    def _make_fixture(
        self,
        category: FixtureCategory,
        position: Position,
        x: float,
        zone_y: tuple[float, float],
        zone_z: tuple[float, float],
        index: int,
    ) -> Fixture:
        # small jitter
        jitter_x = self.rng.uniform(-0.05, 0.05)
        y = self.rng.uniform(*zone_y)
        z = self.rng.uniform(*zone_z)
        rotation = 0.0
        if position == Position.AUDIENCE_FACING:
            rotation = 180.0
        elif position == Position.SIDE_LEFT:
            rotation = 90.0
        elif position == Position.SIDE_RIGHT:
            rotation = -90.0
        fixture_id = f"{category.value}_{position.value}_{index + 1}"
        return Fixture(
            fixture_id=fixture_id,
            category=category,
            position=position,
            x=round(x + jitter_x, 3),
            y=round(y, 3),
            z=round(max(0.0, z), 3),
            rotation_y=rotation,
        )

    # ------------------------------------------------------------------
    def _validate(self, layout: StageLayout) -> None:
        half_w = layout.width_m / 2.0
        for f in layout.fixtures:
            if f.z < 0:
                raise StageGenerationError(f"Fixture {f.fixture_id} below floor")
            if abs(f.x) > half_w + 0.5:
                raise StageGenerationError(f"Fixture {f.fixture_id} outside width")
            if f.y < -1.0 or f.y > layout.depth_m + 0.5:
                raise StageGenerationError(f"Fixture {f.fixture_id} outside depth")


# ---------------------------------------------------------------------------
# ASCII visualization
# ---------------------------------------------------------------------------
_LEGEND_SYMBOLS: dict[FixtureCategory, str] = {
    FixtureCategory.MOVING_BEAM: "B",
    FixtureCategory.MOVING_SPOT: "S",
    FixtureCategory.MOVING_WASH: "W",
    FixtureCategory.LED_PAR: "P",
    FixtureCategory.BLINDER: "L",
    FixtureCategory.STROBE: "X",
    FixtureCategory.PIXEL_BAR: "I",
    FixtureCategory.FOLLOWSPOT: "F",
    FixtureCategory.LASER: "R",
    FixtureCategory.HAZER: "H",
    FixtureCategory.CO2_JET: "C",
    FixtureCategory.PYRO: "Y",
}

_LEGEND_NAMES: dict[FixtureCategory, str] = {
    FixtureCategory.MOVING_BEAM: "Beam",
    FixtureCategory.MOVING_SPOT: "Spot",
    FixtureCategory.MOVING_WASH: "Wash",
    FixtureCategory.LED_PAR: "LED Par",
    FixtureCategory.BLINDER: "Blinder",
    FixtureCategory.STROBE: "Strobe",
    FixtureCategory.PIXEL_BAR: "Pixel Bar",
    FixtureCategory.FOLLOWSPOT: "Followspot",
    FixtureCategory.LASER: "Laser",
    FixtureCategory.HAZER: "Hazer",
    FixtureCategory.CO2_JET: "CO2",
    FixtureCategory.PYRO: "Pyro",
}


def build_llm_view(
    layout: StageLayout, max_fixtures: int = 12
) -> tuple[dict, dict[str, list[str]]]:
    """Build the stage representation passed to the designer LLM.

    If the layout has more than `max_fixtures` individual fixtures, group
    them into zones keyed by (position, category) so the LLM only sees a
    handful of entities. Otherwise pass fixtures through individually.

    Returns:
        (stage_view, zone_to_fixtures)
        - stage_view: dict ready to JSON-serialize into the prompt; its
          `fixtures` list contains either real fixture entries or zone
          entries with `fixture_id` set to `zone_<position>_<category>`.
        - zone_to_fixtures: mapping zone_id -> list of real fixture_ids,
          empty if no zoning was applied.
    """
    base = {
        "width_m": layout.width_m,
        "depth_m": layout.depth_m,
        "height_m": layout.height_m,
        "venue_size": layout.venue_size.value,
        "target_genre": layout.target_genre,
    }
    if len(layout.fixtures) <= max_fixtures:
        base["fixtures"] = [
            {
                "fixture_id": f.fixture_id,
                "category": f.category.value,
                "position": f.position.value,
                "x": f.x,
                "y": f.y,
                "z": f.z,
            }
            for f in layout.fixtures
        ]
        return base, {}

    groups: dict[str, list[Fixture]] = {}
    for f in layout.fixtures:
        zone_id = f"zone_{f.position.value}_{f.category.value}"
        groups.setdefault(zone_id, []).append(f)

    zone_entries: list[dict] = []
    zone_to_fixtures: dict[str, list[str]] = {}
    for zone_id, items in groups.items():
        sample = items[0]
        zone_entries.append(
            {
                "fixture_id": zone_id,
                "category": sample.category.value,
                "position": sample.position.value,
                "count": len(items),
                "note": f"This is a zone of {len(items)} {sample.category.value} fixtures; "
                f"control as a group.",
            }
        )
        zone_to_fixtures[zone_id] = [f.fixture_id for f in items]
    base["fixtures"] = zone_entries
    base["zoning_applied"] = True
    return base, zone_to_fixtures


def render_ascii(layout: StageLayout, cols: int = 60) -> str:
    """Top-down ASCII view of the stage grouped by position row."""
    groups: dict[Position, list[Fixture]] = {}
    for f in layout.fixtures:
        groups.setdefault(f.position, []).append(f)

    order = [
        Position.BACK_TRUSS,
        Position.BACKLINE,
        Position.MID_TRUSS,
        Position.STAGE_FLOOR,
        Position.SIDE_LEFT,
        Position.SIDE_RIGHT,
        Position.FRONT_TRUSS,
        Position.AUDIENCE_FACING,
    ]

    bar = "=" * cols
    lines = [bar, f" Stage {layout.width_m:.1f}m x {layout.depth_m:.1f}m x {layout.height_m:.1f}m"
             f"  | venue={layout.venue_size.value} | genre={layout.target_genre or 'n/a'}", bar]
    for pos in order:
        items = groups.get(pos, [])
        if not items:
            continue
        items_sorted = sorted(items, key=lambda f: f.x)
        symbols = " ".join(f"[{_LEGEND_SYMBOLS[f.category]}]" for f in items_sorted)
        lines.append(f" {pos.value:<16}: {symbols}")
    lines.append(bar)
    lines.append(" AUDIENCE")
    lines.append(bar)
    used = {f.category for f in layout.fixtures}
    legend = ", ".join(f"{_LEGEND_SYMBOLS[c]}={_LEGEND_NAMES[c]}" for c in used)
    lines.append(f" Legend: {legend}")
    return "\n".join(lines)
