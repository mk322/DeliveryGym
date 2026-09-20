"""The phone's map: what it draws, and what it must never be able to draw.

Two kinds of test here and they are guarding different things. The geometric
ones pin the arithmetic -- a map with a sign error in it turns every left turn
into a right one and looks like a policy failure for a week. The rest pin the
line between the map and the eyes: the map is a survey, it knows geometry and
names, and if it could ever show a light or a barrier then ``report_blocked``
would have nothing to do and the photographs would be decoration.
"""

from __future__ import annotations

import math
from pathlib import Path

import pytest

from embodiedbench.compiler.road_network import bearing_deg, build_road_network
from embodiedbench.runtime.city.courier_env import (
    CourierEnv,
    Difficulty,
    Stride,
    compass_of,
)
from embodiedbench.runtime.city.map_image import (
    MapView,
    frame_view,
    render_map,
)


class _Empty:
    """A network with no streets and no buildings, for testing the furniture."""

    streets: list = []
    buildings: list = []


_EMPTY = _Empty()

# Resolved from this file, the way every other test module here does it. A
# relative path only works while the runner's working directory happens to be
# the repository root, and "happens to be" is not a property a test suite has.
PARIS = (Path(__file__).resolve().parents[1] / "vendor" / "vagen" / "vagen"
         / "envs" / "deliverybench" / "maps" / "citycore-paris")
needs_map = pytest.mark.skipif(not PARIS.exists(), reason="Paris map not present")


@pytest.fixture(scope="module")
def paris():
    return build_road_network(PARIS, map_name="citycore-paris")


class TestTheProjection:
    def test_north_is_up_and_east_is_right(self):
        """The one that matters, and the one the first version of this test got
        wrong in exactly the way the code did.

        It asserted that the map's +y is north, because that is what anyone
        would assume -- and this map's north is +x. Written from the same
        assumption as the projection it was checking, it passed over a drawing
        rotated 90 degrees under a compass rose pointing the wrong way. So the
        truth is taken from ``bearing_deg`` and ``compass_of``, which is what the
        rest of the environment speaks, rather than restated here.
        """
        view = MapView(-1000.0, -1000.0, 1000.0, 1000.0, 400, 400)
        centre = view.to_px((0.0, 0.0))
        far = 500.0
        for axis, step in (("x", (far, 0.0)), ("y", (0.0, far)),
                           ("-x", (-far, 0.0)), ("-y", (0.0, -far))):
            where = compass_of(bearing_deg((0.0, 0.0), step))
            x, y = view.to_px(step)
            dx, dy = x - centre[0], y - centre[1]
            if where == "north":
                assert dy < 0 and abs(dx) < 1e-6, f"{axis} is {where} and did not draw up"
            elif where == "south":
                assert dy > 0 and abs(dx) < 1e-6, f"{axis} is {where} and did not draw down"
            elif where == "east":
                assert dx > 0 and abs(dy) < 1e-6, f"{axis} is {where} and did not draw right"
            elif where == "west":
                assert dx < 0 and abs(dy) < 1e-6, f"{axis} is {where} and did not draw left"
            else:  # pragma: no cover - the axes are cardinal by construction
                raise AssertionError(f"{axis} came out as {where}")

    def test_the_facing_arrow_points_where_the_compass_says(self):
        """The marker is a glyph drawn pointing right, rotated by a bearing.

        Same failure mode as the projection: a quarter turn out and the arrow
        contradicts both the map and the route, while looking entirely plausible.
        """
        import re

        drawing = render_map(_EMPTY, here=(0.0, 0.0),
                             facing_deg=bearing_deg((0.0, 0.0), (1.0, 0.0)))
        turn = float(re.search(r"rotate\((-?\d+(?:\.\d+)?)\)", drawing.svg).group(1))
        assert turn % 360 == 270.0, "facing north did not draw the arrow upward"

    def test_the_centre_of_the_window_is_the_centre_of_the_drawing(self):
        view = MapView(-500.0, -500.0, 500.0, 500.0, 400, 300)
        x, y = view.to_px((0.0, 0.0))
        assert (x, y) == pytest.approx((200.0, 150.0))

    def test_the_city_is_not_stretched(self):
        """One scale for both axes: a stretched map turns a right angle into
        something the courier cannot match to the corner it is standing on."""
        view = frame_view([(0.0, 0.0), (1000.0, 40000.0)], width_px=400, height_px=300)
        a, b = view.to_px((0.0, 0.0)), view.to_px((1000.0, 0.0))
        c, d = view.to_px((0.0, 0.0)), view.to_px((0.0, 1000.0))
        assert math.dist(a, b) == pytest.approx(math.dist(c, d), rel=1e-6)

    def test_a_long_route_is_still_a_route_and_a_short_one_still_has_context(self):
        wide = frame_view([(0.0, 0.0), (500000.0, 0.0)])
        tight = frame_view([(0.0, 0.0), (200.0, 0.0)])
        assert wide.span_cm <= 90000.0, "zoomed out until the line was a hair"
        assert tight.span_cm >= 12000.0, "zoomed in until there was no context"

    def test_an_empty_request_still_produces_a_window(self):
        view = frame_view([])
        assert view.span_cm > 0

    def test_the_banner_can_preview_a_nearby_turn_without_changing_the_route(self):
        route = [(0.0, 0.0), (240.0, -180.0), (-600.0, -600.0)]

        drawing = render_map(
            _EMPTY,
            here=route[0],
            route=route,
            next_street="Rue Oberkampf",
            next_heading="north-west",
            next_maneuver_heading="south-west",
            next_maneuver_distance_cm=241.8,
        )

        assert "head north-west on" in drawing.svg
        assert "then south-west in 2 m" in drawing.svg
        assert drawing.route_metres == pytest.approx(
            sum(math.dist(a, b) for a, b in zip(route, route[1:])) / 100.0)


@needs_map
class TestWhatTheMapDraws:
    def env(self, paris, **kw):
        env = CourierEnv(paris, seed=0, difficulty=Difficulty.PAIR, **kw)
        env.reset()
        return env

    def test_it_draws_the_streets_the_blocks_and_the_route(self, paris):
        env = self.env(paris)
        drawing = env.map_drawing(env.target_address())
        assert drawing.streets_drawn > 0
        assert drawing.route_metres > 0
        assert "<svg" in drawing.svg and drawing.svg.rstrip().endswith("</svg>")
        assert "rect class=\"blk\"" in drawing.svg, "no buildings: a map is mostly blocks"

    def test_every_street_it_names_is_a_street_on_the_map(self, paris):
        env = self.env(paris)
        drawing = env.map_drawing(env.target_address())
        real = {street.name for street in paris.streets}
        assert drawing.labels, "an unlabelled map is a diagram"
        assert set(drawing.labels) <= real

    def test_it_works_with_no_route_and_no_destination(self, paris):
        """A deliberately cleared phone screen is still a usable map.

        Reset now lights the active job automatically, so ``map_drawing(None)``
        immediately after reset correctly contains that route.  Clear the
        stored target here to exercise the renderer's actual no-route branch.
        """
        env = self.env(paris)
        env.screen_target = None
        env.screen_route = []
        env._screen_route_owner = env.SCREEN_ROUTE_NONE
        drawing = env.map_drawing(None)
        assert drawing.streets_drawn > 0 and drawing.route_metres == 0.0

    def test_the_route_it_draws_is_the_route_it_speaks(self, paris):
        env = self.env(paris)
        target = env.target_address()
        spoken = sum(leg["metres"] for leg
                     in env.route_legs(env.node_id, target.kerb_node))
        drawn = env.map_drawing(target).route_metres
        assert drawn == pytest.approx(spoken, rel=0.01)

    def test_nothing_the_courier_sees_can_ever_reach_the_map(self, paris):
        """There is no channel from the courier's eyes to the phone.

        The environment used to have one -- ``report_blocked`` -- and it was
        wrong: a rider who meets a skip takes the next street, they do not file
        a report with the map app. With it gone the phone is permanently blind,
        every route it gives will keep naming the shut street, and going round
        is something the courier works out from the photographs.
        """
        env = self.env(paris)
        assert not hasattr(env, "report_blocked")
        walked_into_one = False
        for _ in range(40):
            rows = env.candidates()
            if not rows:
                break
            outcome = env.walk_to(*env.street_at(rows[0]["k"]))
            if not outcome.ok and outcome.code == "way_blocked":
                walked_into_one = True
                break
        svg = env.map_drawing(env.target_address()).svg
        assert 'class="shut"' not in svg
        if walked_into_one:
            # Even having stood at the barrier, the map is unchanged.
            assert 'class="shut"' not in env.map_drawing().svg

    def test_the_map_cannot_show_a_light_or_a_barrier(self, paris):
        """The line the whole tool set rests on, asserted rather than trusted.

        Everything the map is drawn from is geometry and names. If a lamp or a
        skip could reach it, the photographs would stop being the only place
        those live and the courier would have no reason to look at anything.
        """
        env = CourierEnv(paris, seed=2, difficulty=Difficulty.TRIPLE,
                         obstacle_album_root=None)
        env.reset()
        svg = env.map_drawing(env.target_address()).svg.lower()
        for word in ("red", "green", "light", "signal", "lamp", "barrier",
                     "road_block", "obstacle", "pedestrian"):
            assert word not in svg, f"the map leaked {word!r}"

    def test_the_scale_bar_says_a_round_number_of_metres(self, paris):
        env = self.env(paris)
        svg = env.map_drawing(env.target_address()).svg
        assert any(f">{n} m<" in svg for n in (10, 20, 25, 50, 100, 200, 250, 500, 1000))


@needs_map
class TestTheMapReachesTheCourier:
    def session(self, paris, **kw):
        from embodiedbench.agent.courier.session import CourierSession

        env = CourierEnv(paris, seed=0, difficulty=Difficulty.PAIR, stride=Stride.BLOCK, **kw)
        env.reset()
        return CourierSession(env)

    def test_phone_starts_lit_and_navigate_replaces_its_route(self, paris):
        session = self.session(paris)
        assert len([f for f in session.observe().frames if f.kind == "map"]) == 1
        session.step("THOUGHT: t\n```\nnavigate()\n```")
        frames = session.observe().frames
        maps = [f for f in frames if f.kind == "map"]
        assert len(maps) == 1
        assert session.env._screen_route_owner == session.env.SCREEN_ROUTE_EXPLICIT
        assert maps[0].svg.startswith("<svg")
        assert maps[0].path == "", "a drawing is not a file on disk"

    def test_it_is_the_last_picture_in_the_list(self, paris):
        """Captions are read in order and the map is the one picture that did
        not come from the courier's eyes."""
        session = self.session(paris)
        session.step("THOUGHT: t\n```\nnavigate()\n```")
        frames = session.observe().frames
        assert frames[-1].kind == "map"
        assert all(f.kind == "photograph" for f in frames[:-1])

    def test_the_caption_says_it_is_a_drawing_that_cannot_see(self, paris):
        session = self.session(paris)
        session.step("THOUGHT: t\n```\nnavigate()\n```")
        text = session.observe().text
        assert "[map]" in text
        assert "drawing, not a photograph" in text

    def test_the_map_stays_up_and_the_dot_keeps_moving(self, paris):
        """A map app does not switch off when the phone goes in a pocket.

        Showing it for one turn made asking for it a tax: an episode played by
        hand spent 45% of its turns on ``navigate()``, 37 lookups at 15 s each.
        The route is frozen where it was drawn and the position is live, so
        walking is free to watch while a *better* route still costs a call.
        """
        session = self.session(paris)
        session.step("THOUGHT: t\n```\nnavigate()\n```")
        first = next(f for f in session.observe().frames if f.kind == "map")
        drawings = []
        for _ in range(3):
            street, heading = session.env.street_at(1)
            session.step(f'THOUGHT: t\n```\nwalk_to("{street}", "{heading}")\n```')
            frames = [f for f in session.observe().frames if f.kind == "map"]
            assert len(frames) == 1, "the map went out while the courier walked"
            drawings.append(frames[0].svg)
        assert any(d != first.svg for d in drawings), "the dot never moved"

    def test_no_map_where_there_is_no_phone(self, paris):
        from embodiedbench.runtime.city.courier_env import Condition

        env = CourierEnv(paris, seed=0, difficulty=Difficulty.PAIR,
                         condition=Condition.NO_PHONE)
        env.reset()
        assert "navigate" not in env.allowed_tool_names()


class TestTheCourierIsAlwaysOnTheScreen:
    """The one thing this picture must always show is where you are.

    Framing on the route alone put the marker above the top edge whenever the
    route ran that way, so on 13 frames in 100 the courier was off screen or
    hidden under the instruction banner. The first fix moved it the wrong way:
    screen y runs opposite to map x, so pushing the marker down the screen
    raises the window's x centre rather than lowering it.
    """

    def test_the_marker_clears_the_banner_and_the_edges(self):
        from pathlib import Path

        from embodiedbench.compiler.road_network import build_road_network
        from embodiedbench.runtime.city.courier_env import CourierEnv
        from embodiedbench.runtime.city.map_image import (
            BAR_H, HEIGHT_PX, WIDTH_PX)

        maps = (Path(__file__).resolve().parents[1] / "vendor" / "vagen"
                / "vagen" / "envs" / "deliverybench" / "maps" / "citycore-paris")
        if not maps.exists():
            pytest.skip("maps not mounted")
        network = build_road_network(maps, map_name="citycore-paris")
        for seed in range(6):
            env = CourierEnv(network, seed=seed, order_count=1,
                             difficulty="solo", stride="block")
            env.reset()
            order = next((o for o in env.orders if o.live), None)
            if order is None:
                continue
            env.navigate(order.pickup.text)
            for _ in range(4):
                view = env.map_drawing().view
                x, y = view.to_px(env.position())
                assert 40 < x < WIDTH_PX - 40, f"seed {seed}: marker off the side"
                assert BAR_H + 30 < y < HEIGHT_PX - 60, (
                    f"seed {seed}: marker at y={y:.0f} is under the banner or "
                    f"off the foot"
                )
                rows = env.candidates()
                if not rows:
                    break
                env.walk_to(rows[0]["street"],
                            rows[0]["heading"])
