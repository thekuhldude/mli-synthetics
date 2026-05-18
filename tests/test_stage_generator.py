"""Tests for the stage generator."""
from __future__ import annotations

import pytest

from mli_synthetics.stage.fixtures import FixtureCategory, VenueSize
from mli_synthetics.stage.generator import StageGenerator
from mli_synthetics.stage.library import FixtureLibrary


@pytest.fixture
def library():
    return FixtureLibrary.load()


@pytest.fixture
def gen(library):
    return StageGenerator(library=library, seed=123, min_fixtures=8, max_fixtures=80)


def test_fixture_count_within_bounds(gen):
    for _ in range(10):
        layout = gen.generate(target_genre="edm", venue_size=VenueSize.MEDIUM_CLUB)
        assert 8 <= len(layout.fixtures) <= 80


def test_all_fixtures_in_stage_bounds(gen):
    for _ in range(10):
        layout = gen.generate(target_genre="rock")
        half_w = layout.width_m / 2.0
        for f in layout.fixtures:
            assert f.z >= 0
            assert abs(f.x) <= half_w + 0.5
            assert -1.0 <= f.y <= layout.depth_m + 0.5


def test_hazer_present_when_beams_present(gen):
    # Force a genre that produces beams (edm)
    for _ in range(5):
        layout = gen.generate(target_genre="edm")
        has_beam = any(f.category == FixtureCategory.MOVING_BEAM for f in layout.fixtures)
        if has_beam:
            has_hazer = any(f.category == FixtureCategory.HAZER for f in layout.fixtures)
            assert has_hazer, "Hazer should be present when beams present"


def test_genre_distribution_dominant_categories(gen):
    # EDM should be beam-heavy on average
    counts_total = {}
    for _ in range(8):
        layout = gen.generate(target_genre="edm", venue_size=VenueSize.LARGE_VENUE)
        for cat, n in layout.fixture_count_by_category.items():
            counts_total[cat] = counts_total.get(cat, 0) + n
    total = sum(counts_total.values())
    beam_ratio = counts_total.get("moving_beam", 0) / total
    # EDM target ratio for beams is 0.40; allow generous tolerance
    assert beam_ratio > 0.20, f"EDM beam ratio too low: {beam_ratio}"


def test_unknown_venue_raises(gen):
    from mli_synthetics.errors import StageGenerationError

    with pytest.raises(StageGenerationError):
        gen.generate(venue_size="bogus")


def test_acoustic_distribution_par_heavy(gen):
    counts_total = {}
    for _ in range(8):
        layout = gen.generate(target_genre="acoustic")
        for cat, n in layout.fixture_count_by_category.items():
            counts_total[cat] = counts_total.get(cat, 0) + n
    total = sum(counts_total.values())
    par_ratio = counts_total.get("led_par", 0) / total
    assert par_ratio > 0.20
