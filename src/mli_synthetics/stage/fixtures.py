"""Pydantic models for stage fixtures."""
from enum import Enum

from pydantic import BaseModel, Field


class FixtureCategory(str, Enum):
    MOVING_BEAM = "moving_beam"
    MOVING_SPOT = "moving_spot"
    MOVING_WASH = "moving_wash"
    LED_PAR = "led_par"
    BLINDER = "blinder"
    STROBE = "strobe"
    PIXEL_BAR = "pixel_bar"
    FOLLOWSPOT = "followspot"
    LASER = "laser"
    HAZER = "hazer"
    CO2_JET = "co2_jet"
    PYRO = "pyro"


class Position(str, Enum):
    FRONT_TRUSS = "front_truss"
    MID_TRUSS = "mid_truss"
    BACK_TRUSS = "back_truss"
    STAGE_FLOOR = "stage_floor"
    SIDE_LEFT = "side_left"
    SIDE_RIGHT = "side_right"
    BACKLINE = "backline"
    AUDIENCE_FACING = "audience_facing"


class VenueSize(str, Enum):
    SMALL_CLUB = "small_club"
    MEDIUM_CLUB = "medium_club"
    LARGE_VENUE = "large_venue"
    FESTIVAL = "festival"


class Fixture(BaseModel):
    fixture_id: str
    category: FixtureCategory
    position: Position
    x: float = Field(description="meters from stage center")
    y: float = Field(description="meters depth")
    z: float = Field(description="meters height")
    rotation_y: float = 0.0


class StageLayout(BaseModel):
    width_m: float
    depth_m: float
    height_m: float
    venue_size: VenueSize
    target_genre: str | None = None
    fixtures: list[Fixture]

    @property
    def fixture_count_by_category(self) -> dict[str, int]:
        counts: dict[str, int] = {}
        for f in self.fixtures:
            counts[f.category.value] = counts.get(f.category.value, 0) + 1
        return counts
