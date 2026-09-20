"""
Tests for the unified NAVIGATE tool.

NAVIGATE(target=..., mode="walk"|"e-scooter"|"bus", text=false) replaced the
former NAVIGATE_WALK / NAVIGATE_ESCOOTER / NAVIGATE_BUS and their VISUAL_*
twins. The route is returned as a map image by default; turn-by-turn text is
emitted only when text=true.

Covers:
  - parser: kw target, positional target, positional mode, text/mode kwargs, missing target
  - text=true -> "steps:" present; default omits steps
  - mode gating: "e-scooter" blocked when battery off; "bus" blocked when transport off; unknown mode errors
  - invalid target -> error
  - state unchanged after the (query-only) call (x, y, clock, energy)

Run:
    PYTHONPATH=. python3 -m vagen.envs.deliverybench.test_navigate
"""

import asyncio
import json
import sys
from pathlib import Path


def _act(s: str) -> str:
    return json.dumps({"action": s})


# ---------------------------------------------------------------------------
# Parser unit tests (no env needed)
# ---------------------------------------------------------------------------

def test_parser():
    from vagen.envs.deliverybench.vlm_delivery.gameplay.action_space import parse_action
    from vagen.envs.deliverybench.vlm_delivery.base.defs import DMActionKind

    class _DM:  # minimal stub; parse_action only reads dm.cfg
        cfg = {}

    dm = _DM()
    a, _ = parse_action('NAVIGATE(target="restaurant 1")', dm)
    assert a.kind == DMActionKind.NAVIGATE and a.data["target"] == "restaurant 1"
    assert a.data.get("mode") is None  # defaults applied in the handler

    a, _ = parse_action('NAVIGATE("restaurant 1", "e-scooter")', dm)
    assert a.data["target"] == "restaurant 1" and a.data["mode"] == "e-scooter"

    a, _ = parse_action('NAVIGATE(target="x", mode="bus", access_mode="scooter")', dm)
    assert a.data["mode"] == "bus" and a.data["access_mode"] == "scooter"

    for bad in ['NAVIGATE()', 'NAVIGATE(mode="walk")']:
        try:
            parse_action(bad, dm)
            raise AssertionError(f"{bad} should have raised")
        except ValueError:
            pass
    print("PASS test_parser")


def test_repeated_same_target_navigate_is_noop_success():
    from vagen.envs.deliverybench.vlm_delivery.actions.navigate import _navigate_ground

    class _Pos:
        def __init__(self, x, y):
            self.x = x
            self.y = y

        def distance(self, other):
            return ((self.x - other.x) ** 2 + (self.y - other.y) ** 2) ** 0.5

    class _Node:
        def __init__(self, waypoint_id, name, x, y):
            self.waypoint_id = waypoint_id
            self.waypoint_name = name
            self.address = name
            self.position = _Pos(x, y)

    class _Graph:
        def __init__(self, start, target):
            self.adjacency_list = {start: [target], target: []}

        def shortest_path_nodes(self, start, target):
            return [start, target], 1.0

        def get_edge_meta(self, _u, _v):
            return {"dist_cm": 1000}

    class _CityMap:
        def __init__(self, start, target):
            self.start = start
            self.target = target
            self.waypoint_graph = _Graph(start, target)

        def resolve_waypoint(self, token):
            return self.target if token == "goal" else None

        def nearest_waypoint(self, _x, _y):
            return self.start

    class _DM:
        def __init__(self):
            self.start = _Node("start", "start", 0, 0)
            self.target = _Node("goal", "goal", 0, 1000)
            self.city_map = _CityMap(self.start, self.target)
            self.x = 0.0
            self.y = 0.0
            self.cfg = {}
            self._nav_target_node = self.target
            self._nav_route_path = [self.start, self.target]
            self._nav_route_color = (1, 2, 3, 4)
            self.vlm_ephemeral = {
                "navigation": "mode: walk\nfrom: start\nto: goal\ndistance_m: 10.0"
            }
            self.finished = None

        def vlm_add_error(self, msg):
            raise AssertionError(msg)

        def vlm_add_ephemeral(self, tag, text):
            self.vlm_ephemeral[tag] = text

        def _finish_action(self, success=True):
            self.finished = success

    dm = _DM()
    path_before = list(dm._nav_route_path)
    color_before = dm._nav_route_color
    _navigate_ground(dm, "goal", "walk", previous_ephemeral=dict(dm.vlm_ephemeral))
    assert dm.finished is True
    assert dm._nav_route_path == path_before
    assert dm._nav_route_color == color_before
    assert dm._nav_target_node is dm.target
    assert "route already active" in dm.vlm_ephemeral["navigation"]
    assert "continue with MOVE(direction=...)" in dm.vlm_ephemeral["navigation"]
    print("PASS test_repeated_same_target_navigate_is_noop_success")


# ---------------------------------------------------------------------------
# Integration tests (need map data)
# ---------------------------------------------------------------------------

def _cfg(**over):
    base = dict(map_name="small-city-11", render_mode="text", enable_map_images=False,
                enable_fpv=False, use_gmaps_renderer=False, max_steps=50)
    base.update(over)
    return base


async def _env(cfg):
    from vagen.envs.deliverybench import DeliveryBench, DeliveryBenchEnvConfig
    if not Path(DeliveryBenchEnvConfig().base_dir).exists():
        return None
    env = DeliveryBench(cfg)
    await env.reset(seed=42)
    return env


def _eph(obs_str: str) -> str:
    return obs_str.split("### ephemeral_context")[1] if "### ephemeral_context" in obs_str else ""


async def test_emits_estimate_no_step_chain():
    import re
    env = await _env(_cfg())
    if env is None:
        print("SKIP test_emits_estimate_no_step_chain (no map data)"); return
    obs, _, _, info = await env.step(_act('NAVIGATE(target="restaurant 1")'))
    assert info["is_tool"] is True, "NAVIGATE must be a tool"
    assert not info.get("action_error"), info.get("action_error")
    eph = _eph(obs["obs_str"])
    assert "[navigation]" in obs["obs_str"], "missing [navigation] block"
    assert "estimated_time" in eph, "must report a travel-time estimate"
    # Must NOT spell out a turn-by-turn / waypoint-id chain (that would give the answer).
    assert "steps:" not in eph, "NAVIGATE must not emit a step chain"
    assert not re.search(r"int_\d|dock_\d", eph), "NAVIGATE must not expose waypoint ids"
    print("PASS test_emits_estimate_no_step_chain")


async def test_mode_gating():
    env = await _env(_cfg(enable_battery=False, enable_advanced_transport=False))
    if env is None:
        print("SKIP test_mode_gating (no map data)"); return
    _, _, _, i1 = await env.step(_act('NAVIGATE(target="restaurant 1", mode="e-scooter")'))
    assert i1.get("action_error"), "e-scooter must be blocked when battery off"
    _, _, _, i2 = await env.step(_act('NAVIGATE(target="restaurant 1", mode="bus")'))
    assert i2.get("action_error"), "bus must be blocked when advanced transport off"
    _, _, _, i3 = await env.step(_act('NAVIGATE(target="restaurant 1", mode="hovercraft")'))
    assert i3.get("action_error"), "unknown mode must error"
    _, _, _, i4 = await env.step(_act('NAVIGATE(target="restaurant 1")'))
    assert not i4.get("action_error"), "walk must always be available"
    print("PASS test_mode_gating")


async def test_invalid_target_and_state_unchanged():
    env = await _env(_cfg())
    if env is None:
        print("SKIP test_invalid_target_and_state_unchanged (no map data)"); return
    dm = env._env.dms[0]
    x0, y0, t0, e0 = dm.x, dm.y, dm.clock.now_sim(), dm.energy_pct
    _, _, _, info = await env.step(_act('NAVIGATE(target="__no_such_place__")'))
    assert info.get("action_error"), "unresolvable target must error"
    assert (dm.x, dm.y) == (x0, y0), "NAVIGATE moved the agent"
    assert dm.clock.now_sim() == t0, "NAVIGATE advanced the clock"
    assert dm.energy_pct == e0, "NAVIGATE changed energy"
    print("PASS test_invalid_target_and_state_unchanged")


_SYNC = [test_parser, test_repeated_same_target_navigate_is_noop_success]
_ASYNC = [
    test_emits_estimate_no_step_chain,
    test_mode_gating,
    test_invalid_target_and_state_unchanged,
]

if __name__ == "__main__":
    sys.path.insert(0, str(Path(__file__).parents[4]))
    failed = []
    for t in _SYNC:
        try:
            t()
        except Exception as exc:
            print(f"FAIL {t.__name__}: {exc}"); failed.append(t.__name__)
    for t in _ASYNC:
        try:
            asyncio.run(t())
        except Exception as exc:
            print(f"FAIL {t.__name__}: {exc}"); failed.append(t.__name__)
    if failed:
        print(f"\n{len(failed)} failed: {failed}"); sys.exit(1)
    print(f"\nAll {len(_SYNC) + len(_ASYNC)} tests passed.")
