"""The two measurements that decide what the runtime is allowed to charge for.

``signal_legibility`` says which pedestrian lights an agent can actually read.
``obstacle_visibility`` says which obstacles it can actually see. Both gate a
penalty, so both are load-bearing, and both replace an earlier metric that was
wrong in the same way: an absolute cutoff sitting in the middle of the
distribution it was measuring.

What is tested here is the property that failure had -- that the answer must not
be a function of the cutoff. Synthetic frames are used deliberately: a lamp of a
known size in a known place is the only way to assert what the answer *should*
be, and the real albums are then measured with the same code.
"""

from __future__ import annotations

import json
import tempfile
import os
from pathlib import Path

import numpy as np
import pytest
from PIL import Image

from embodiedbench.compiler.obstacle_visibility import (
    measure_album as measure_obstacle_album,
    measure_frame,
)
from embodiedbench.compiler.signal_legibility import (
    MIN_MARGIN,
    ApproachLegibility,
    PhaseLamp,
    chroma,
    margin_sweep,
    measure_album,
    measure_arrays,
    threshold_sweep,
)

W, H = 320, 240


def frame(fill=(120, 120, 120)) -> np.ndarray:
    return np.full((H, W, 3), fill, dtype=np.int16)


def patch(image: np.ndarray, colour, x=100, y=100, size=10) -> np.ndarray:
    out = image.copy()
    out[y:y + size, x:x + size] = colour
    return out


def save(path: Path, array: np.ndarray) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(array.astype("uint8")).save(path)
    return path


class TestChromaIsSelfNormalising:
    def test_it_is_the_sign_that_carries_the_colour(self):
        bright = chroma(np.array([[[220, 40, 40]]], dtype=np.int16))[0, 0]
        dim = chroma(np.array([[[70, 12, 12]]], dtype=np.int16))[0, 0]
        assert bright > 0 and dim > 0
        # A lamp in shade must not read as a different colour from one in sun.
        assert abs(bright - dim) < 0.15

    def test_grey_sits_at_zero_whatever_the_exposure(self):
        for level in (10, 90, 200):
            value = chroma(np.array([[[level, level, level]]], dtype=np.int16))[0, 0]
            assert abs(value) < 0.02


class TestLegibility:
    def test_a_lamp_that_flips_colour_is_legible(self):
        red = patch(frame(), (230, 30, 30))
        green = patch(frame(), (30, 230, 90))
        row = measure_arrays("a|b", red, green)
        assert row.decisive() and row.blob_px == 100
        assert row.c_red > 0 > row.c_green

    def test_something_red_in_both_phases_is_not_a_lamp(self):
        """The awning case. It is red, it is big, and it says nothing."""
        red = patch(frame(), (230, 30, 30))
        green = patch(frame(), (230, 30, 30))
        row = measure_arrays("a|b", red, green)
        assert not row.decisive()
        assert row.changed_px == 0

    def test_a_grey_flicker_is_not_a_lamp(self):
        """The foliage case, which is what the old metric let through.

        Leaves move a little between renders, so the pixels change; what they do
        not do is go red in one phase and green in the other. One such clump --
        19 px, margin 0.08 -- was visually confirmed as a tree and was being
        counted as a readable signal.
        """
        # A leaf catching the light: brightness swings 60 levels, colour does not.
        red = patch(frame(), (180, 175, 170))
        green = patch(frame(), (120, 124, 128))
        row = measure_arrays("a|b", red, green)
        assert row.changed_px > 0, "the frames really do differ"
        assert not row.decisive(), "a grey flicker was read as a signal"
        assert row.margin < MIN_MARGIN

    def test_scattered_single_pixels_do_not_add_up_to_a_lamp(self):
        red, green = frame(), frame()
        rng = np.random.default_rng(0)
        for _ in range(200):
            y, x = int(rng.integers(0, H)), int(rng.integers(0, W))
            red[y, x] = (230, 30, 30)
            green[y, x] = (30, 230, 30)
        row = measure_arrays("a|b", red, green)
        assert row.flipped_px >= 150, "the scatter is there"
        assert row.blob_px <= 4, "a scatter was fused into one lamp"
        assert not row.legible()

    def test_a_lamp_too_small_to_survive_the_resize_is_not_legible(self):
        """Decisive is not the same as readable, and the difference is size."""
        red = patch(frame(), (230, 30, 30), size=1)
        green = patch(frame(), (30, 230, 30), size=1)
        row = measure_arrays("a|b", red, green)
        assert row.decisive()
        assert not row.legible()

    def test_the_answer_does_not_move_with_the_pixel_threshold(self):
        """The property the metric it replaces did not have."""
        red = patch(frame(), (230, 30, 30))
        green = patch(frame(), (30, 230, 30))
        sizes = {measure_arrays("a|b", red, green, t).blob_px
                 for t in (8.0, 16.0, 24.0, 48.0, 96.0)}
        assert sizes == {100}

    def test_the_answer_does_not_move_with_the_margin_cutoff(self):
        row = ApproachLegibility(
            key="a|b", blob_px=100, c_red=0.7, c_green=-0.7, width=W, height=H,
            red_lamp=PhaseLamp(px=100, level=210.0),
            green_lamp=PhaseLamp(px=100, level=200.0),
        )
        assert all(row.legible(4.0, m) for m in (0.05, 0.1, 0.125, 0.15, 0.3, 0.6))


class TestThreeAspectHead:
    """The defect this measurement was rewritten for.

    A vehicle head puts red at the top and green at the bottom, so no pixel is
    ever red in one phase and green in the other. The flip test then finds the
    green lens alone -- green when green, an unlit grey lens when red -- and
    reporting that component's colour in the red bake reports the unlit lens and
    concludes the red does not work. It does; the question was wrong.
    """

    def head(self, top, bottom) -> np.ndarray:
        image = patch(frame(), top, x=100, y=60, size=12)
        return patch(image, bottom, x=100, y=100, size=12)

    OFF = (38, 36, 36)          # an unlit lens: dark, and very slightly warm

    def test_the_flip_test_only_finds_one_of_the_two_lenses(self):
        red = self.head((235, 40, 30), self.OFF)
        green = self.head(self.OFF, (40, 225, 110))
        row = measure_arrays("a|b", red, green)
        # It finds a lens, and calls it dark, because it is looking at the green
        # lens during the red phase.
        assert row.blob_px == 144, "one lens, not two"
        assert row.c_red > 0 > row.c_green, "so the old metric called it decisive"

    def test_each_phase_is_measured_in_its_own_bake(self):
        red = self.head((235, 40, 30), self.OFF)
        green = self.head(self.OFF, (40, 225, 110))
        row = measure_arrays("a|b", red, green)
        assert row.red_lamp.px == 144 and row.green_lamp.px == 144
        assert row.red_lamp.level > 200, "the red lamp is lit and says so"
        assert row.green_lamp.level > 200
        assert row.both_lamps() and row.legible()
        # The two lenses are in different places -- that is the whole point.
        assert row.red_lamp.box[1] < row.green_lamp.box[1]

    def test_a_head_whose_red_never_lights_is_not_legible(self):
        """Green alone does not tell a courier what red looks like here."""
        red = self.head(self.OFF, self.OFF)
        green = self.head(self.OFF, (40, 225, 110))
        row = measure_arrays("a|b", red, green)
        assert row.green_lamp.lit()
        assert not row.red_lamp.lit()
        assert not row.legible()

    def test_a_dim_tinted_lens_is_not_a_lit_lamp(self):
        red = self.head((70, 30, 28), self.OFF)
        green = self.head(self.OFF, (40, 225, 110))
        row = measure_arrays("a|b", red, green)
        assert row.red_lamp.px > 0, "something was found"
        assert not row.red_lamp.lit(), "but it is not lit"
        assert not row.legible()

    def test_the_answer_does_not_move_with_the_level_cutoff(self):
        red = self.head((235, 40, 30), self.OFF)
        green = self.head(self.OFF, (40, 225, 110))
        row = measure_arrays("a|b", red, green)
        assert all(row.legible(4.0, MIN_MARGIN, level)
                   for level in (60.0, 80.0, 100.0, 120.0, 140.0, 180.0))


class TestStaticRedIsCountedNotRemoved:
    def test_a_no_entry_disc_is_reported_beside_the_lamp(self):
        """It is red in both bakes, so it is not a lamp -- but it is still there."""
        red = patch(patch(frame(), (230, 30, 30)), (210, 25, 25), x=200, y=50, size=30)
        green = patch(patch(frame(), (30, 230, 30)), (210, 25, 25), x=200, y=50, size=30)
        row = measure_arrays("a|b", red, green)
        assert row.legible(), "the lamp is still readable"
        assert row.confusers == 1
        assert row.biggest_confuser_px == 900
        assert row.biggest_confuser_px > row.red_lamp.px, "and it outranks the lamp"

    def test_a_frame_with_nothing_static_reports_none(self):
        row = measure_arrays("a|b", patch(frame(), (230, 30, 30)),
                             patch(frame(), (30, 230, 30)))
        assert row.confusers == 0 and row.biggest_confuser_px == 0

    def test_an_album_measures_and_writes_a_sidecar(self):
        with tempfile.TemporaryDirectory() as tmp:
            album = Path(tmp)
            save(album / "images" / "n1" / "toward_n2_red.png", patch(frame(), (230, 30, 30)))
            save(album / "images" / "n1" / "toward_n2_green.png", patch(frame(), (30, 230, 30)))
            save(album / "images" / "n1" / "toward_n3_red.png", frame())
            save(album / "images" / "n1" / "toward_n3_green.png", frame())
            measured = measure_album(album)
            assert measured.approaches == 2
            assert measured.legible_keys() == ["n1|n2"]
            sidecar = measured.visibility_sidecar("test")
            assert sidecar["legible_count"] == 1
            assert sidecar["method"]
            assert threshold_sweep(album)[0]["legible"] == 1
            assert margin_sweep(measured)[0]["legible"] == 1


class TestObstacleVisibility:
    def test_a_prop_that_shows_up_is_visible(self):
        with tempfile.TemporaryDirectory() as tmp:
            clear = save(Path(tmp) / "clear.png", frame())
            blocked = save(Path(tmp) / "blocked.png", patch(frame(), (40, 40, 40), size=40))
            row = measure_frame("a|b", "road_block", blocked, clear)
            assert row.blob_px == 1600 and row.visible()

    def test_a_prop_that_does_not_show_up_is_not(self):
        """A barrier behind a wall renders a file and shows nothing."""
        with tempfile.TemporaryDirectory() as tmp:
            clear = save(Path(tmp) / "clear.png", frame())
            blocked = save(Path(tmp) / "blocked.png", frame())
            row = measure_frame("a|b", "road_block", blocked, clear)
            assert row.changed_px == 0 and not row.visible()

    def test_an_approach_needs_both_kinds_to_be_listed(self):
        """Which kind is live is chosen per episode, after the gate is read.

        Listing an approach whose barrier shows but whose congestion does not
        would charge for a frame that shows nothing half the time.
        """
        with tempfile.TemporaryDirectory() as tmp:
            album, clear = Path(tmp) / "obs", Path(tmp) / "clear"
            save(clear / "images" / "n1" / "toward_n2.png", frame())
            save(clear / "images" / "n1" / "toward_n3.png", frame())
            save(album / "images" / "n1" / "toward_n2_road_block.png",
                 patch(frame(), (40, 40, 40), size=40))
            save(album / "images" / "n1" / "toward_n2_slow_pedestrian.png",
                 patch(frame(), (40, 40, 40), size=40))
            save(album / "images" / "n1" / "toward_n3_road_block.png",
                 patch(frame(), (40, 40, 40), size=40))
            save(album / "images" / "n1" / "toward_n3_slow_pedestrian.png", frame())
            measured = measure_obstacle_album(album, clear)
            assert measured.visible_keys() == ["n1|n2"]

    def test_a_missing_clear_frame_is_an_error_not_a_pass(self):
        with tempfile.TemporaryDirectory() as tmp:
            blocked = save(Path(tmp) / "blocked.png", frame())
            row = measure_frame("a|b", "road_block", blocked, Path(tmp) / "nope.png")
            assert row.error and not row.visible()


REAL_SIGNALS = Path(os.environ.get("ALBUMS_DIR", "/data/albums")) / "paris_lamps_real/citycore-paris"


@pytest.mark.skipif(not (REAL_SIGNALS / "signal_visibility.json").exists(),
                    reason="the Paris lamp album is not present here")
class TestTheShippedSidecarWasMeasuredByThisCode:
    def test_the_sidecar_names_the_method_it_used(self):
        data = json.loads((REAL_SIGNALS / "signal_visibility.json").read_text())
        # The wording that has to be there is the *shape* of the measurement
        # (`bake_real_lamps.verified_sidecar`): lamps read from the scene,
        # then every listed crossing rendered in both phases and measured at
        # the size the harness serves. A sidecar written by the older flip
        # test, or by geometry alone, cannot pass as one written by this code.
        assert data["from_scene"] is True
        assert data["verified_against_renders"] is True
        assert "rendered in both phases" in data["method"]
        assert "320 px" in data["method"]
        assert 0 < data["legible_count"] == len(data["legible"])
        # One entry per approach, so the bound is lamps, not junctions.
        assert data["legible_count"] <= data["lamps_in_scene"]
        assert not set(data["rejected"]) & set(data["legible"])
        # The lit-area table arrived after the shipped album was measured;
        # when a sidecar carries it, it must cover exactly the legible set.
        if "lamp_px" in data:
            assert set(data["lamp_px"]) == set(data["legible"])
