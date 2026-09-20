"""
Focused tests for visual-only NAVIGATE mode.

This mode keeps the normal delivery workflow and lets the agent call NAVIGATE,
but hides the textual next_move hint from policy prompt/observation.

Run:
    PYTHONPATH=. python3 -m vagen.envs.deliverybench.test_visual_route_following
"""

import asyncio
import sys
from dataclasses import replace
from pathlib import Path

from PIL import Image


def _assert_no_next_move_leak(text: str) -> None:
    lowered = text.lower()
    for token in ("next_move", "oracle_next_move", "oracle_next_action"):
        assert token not in lowered, f"policy text leaked {token!r}:\n{text}"


def _cfg(**overrides):
    from vagen.envs.deliverybench import VISUAL_ROUTE_FOLLOWING_CONFIG

    cfg = replace(VISUAL_ROUTE_FOLLOWING_CONFIG, max_steps=5, gmaps_out_scale=0.2)
    return replace(cfg, **overrides)


class _Pos:
    def __init__(self, x, y):
        self.x = x
        self.y = y

    def distance(self, other):
        return ((self.x - other.x) ** 2 + (self.y - other.y) ** 2) ** 0.5


class _Node:
    def __init__(self, waypoint_id, x, y, name=None):
        self.waypoint_id = waypoint_id
        self.waypoint_name = name or waypoint_id
        self.position = _Pos(x, y)


class _Graph:
    def __init__(self, start, target):
        self.start = start
        self.target = target
        self.adjacency_list = {start: [target], target: []}

    def shortest_path_nodes(self, start, target):
        if start is self.start and target is self.target:
            return [self.start, self.target], 1.0
        return [], 0.0


class _CityMap:
    def __init__(self, start, target):
        self.start = start
        self.target = target
        self.waypoint_graph = _Graph(start, target)

    def nearest_waypoint(self, _x, _y):
        return self.start

    def adjacents(self, node):
        if node is not self.start:
            return []
        return [{"node": self.target, "bearing_deg": 0.0, "dist_m": 10.0}]


class _DM:
    def __init__(self):
        self.start = _Node("START", 0, 0, "START")
        self.target = _Node("GOAL", 0, 2000, "GOAL")
        self.x = 0.0
        self.y = 0.0
        self.facing_deg = 0.0
        self.cfg = {}
        self.city_map = _CityMap(self.start, self.target)
        self._nav_target_node = self.target
        self._nav_route_path = [self.start, self.target]
        self._nav_route_color = (26, 115, 232, 255)

    def build_state_observation(self):
        return "\n".join([
            "### agent_state",
            "position: (0, 0)",
            "### ephemeral_context",
            "[navigation]",
            "mode: walk",
            "from: START",
            "to: GOAL",
            "distance_m: 10.0",
            "estimated_time: ~5s",
            "next_move: move forward",
            "### available_actions",
            "MOVE(direction=\"forward\")",
            "NAVIGATE(target=\"restaurant 1\")",
        ])


class _EnvBox:
    def __init__(self, dm):
        self.dms = [dm]


def test_system_prompt_keeps_navigate_but_hides_next_move():
    from vagen.envs.deliverybench import DeliveryBench

    env = DeliveryBench(_cfg())
    prompt = asyncio.run(env.system_prompt())["obs_str"]
    _assert_no_next_move_leak(prompt)
    for expected in ("NAVIGATE", "VIEW_ORDERS", "ACCEPT_ORDER", "PICKUP", "DROP_OFF", "MOVE"):
        assert expected in prompt, f"expected delivery action missing: {expected}"
    commands = prompt[prompt.index("COMMANDS (UPPERCASE):"):prompt.index("You can refer")]
    assert 'NAVIGATE(target="<address or waypoint>"' in commands
    assert 'NAVIGATE(target="146 Church Ave")' in prompt
    assert 'NAVIGATE(target="restaurant 1")' not in prompt
    assert "highlighted route" in prompt or "route remains highlighted" in prompt
    assert "[pickup_hint]" in prompt
    assert "[dropoff_hint]" in prompt
    assert "your next action must be NAVIGATE with the exact Pickup address" in prompt
    assert "do not repeat NAVIGATE for the same target" in prompt
    assert "`planned_route_*` fields are estimates from the last NAVIGATE route plan" in prompt
    assert "Do not use them to judge current progress or arrival" in prompt
    assert "do not call PICKUP just because the address" in prompt
    assert "you have not arrived; your next action must be MOVE" in prompt
    assert "Do not infer arrival or route direction from street names" in prompt
    print("PASS test_system_prompt_keeps_navigate_but_hides_next_move")


async def test_policy_observation_strips_next_move_but_keeps_navigation_block():
    from vagen.envs.deliverybench import DeliveryBench

    env = DeliveryBench(_cfg(enable_map_images=True))
    env._env = _EnvBox(_DM())

    async def _fake_map_images():
        return [Image.new("RGB", (32, 32), (255, 255, 255))]

    env._get_map_images = _fake_map_images
    obs = await env._build_observation({}, {}, init_obs=True)
    _assert_no_next_move_leak(obs["obs_str"])
    assert "[navigation]" in obs["obs_str"]
    assert "from: START" not in obs["obs_str"]
    assert "to: GOAL" in obs["obs_str"]
    assert "planned_route_distance_m: 10.0" in obs["obs_str"]
    assert "planned_route_estimated_time: ~5s" in obs["obs_str"]
    assert "\ndistance_m:" not in obs["obs_str"]
    assert "\nestimated_time:" not in obs["obs_str"]
    assert "NAVIGATE" in obs["obs_str"]
    assert "multi_modal_input" in obs
    print("PASS test_policy_observation_strips_next_move_but_keeps_navigation_block")


def test_oracle_next_move_info_only():
    from vagen.envs.deliverybench import DeliveryBench

    env = DeliveryBench(_cfg())
    env._env = _EnvBox(_DM())
    info = env._visual_route_following_info()
    assert info["oracle_next_move"] == "move forward"
    assert info["oracle_next_action"] == 'MOVE(direction="forward")'
    text = env._get_text_observation()
    _assert_no_next_move_leak(text)
    print("PASS test_oracle_next_move_info_only")


def test_route_renderer_draws_waypoint_labels():
    from vagen.envs.deliverybench.tools.render_gmaps import View, compose_frame

    bg = Image.new("RGBA", (320, 320), (255, 255, 255, 255))
    view = View(xmin=0, xmax=3000, ymin=0, ymax=3000, out_w=320, out_h=320, margin=20)
    img = compose_frame(
        bg,
        view,
        agent_xy=(0, 0),
        order_markers=[
            {"x": 3000, "y": 3000, "kind": "pickup", "label": "#7↑"},
        ],
        nav_route=[
            _Node("wp_start", 0, 0),
            _Node("wp_mid", 1500, 1500),
            _Node("wp_goal", 3000, 3000),
        ],
        crop=False,
        decorate=False,
        route_waypoint_labels=True,
    )
    assert img.size == (320, 320)
    assert img.getbbox() is not None
    print("PASS test_route_renderer_draws_waypoint_labels")


_SYNC = [
    test_system_prompt_keeps_navigate_but_hides_next_move,
    test_oracle_next_move_info_only,
    test_route_renderer_draws_waypoint_labels,
]
_ASYNC = [test_policy_observation_strips_next_move_but_keeps_navigation_block]


if __name__ == "__main__":
    sys.path.insert(0, str(Path(__file__).parents[4]))
    failed = []
    for test in _SYNC:
        try:
            test()
        except Exception as exc:
            print(f"FAIL {test.__name__}: {exc}")
            failed.append(test.__name__)
    for test in _ASYNC:
        try:
            asyncio.run(test())
        except Exception as exc:
            print(f"FAIL {test.__name__}: {exc}")
            failed.append(test.__name__)
    if failed:
        print(f"\n{len(failed)} failed: {failed}")
        sys.exit(1)
    print(f"\nAll {len(_SYNC) + len(_ASYNC)} tests passed.")
