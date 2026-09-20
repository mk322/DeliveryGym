"""
DeliveryBench environment wrapper for VAGEN.

This module provides a GymImageEnv-compatible wrapper around the DeliveryBench
text-only environment for use with VAGEN's training infrastructure.
"""

import asyncio
import json
import logging
import math
from io import BytesIO
from pathlib import Path
import dataclasses
from dataclasses import dataclass
from typing import Any, Dict, Tuple, List, Optional

from PIL import Image

from ..gym_image_env import GymImageEnv
from .vlm_delivery.gameplay.prompt import get_system_prompt
from .vlm_delivery.base.defs import TOOL_ACTION_KINDS, TransportMode
from .vlm_delivery.utils.transport import transport_set_mode

from .utils.utils import parse_response



# Get the directory where this file is located (for local vlm_delivery and maps)
_THIS_DIR = Path(__file__).parent.resolve()
_DEFAULT_BASE_DIR = str(_THIS_DIR)


@dataclass
class DeliveryBenchEnvConfig:
    """Configuration for DeliveryBench environment."""

    # Task mode. "delivery" is the normal food-delivery environment.
    # "visual_route_following" keeps the delivery workflow but hides textual
    # NAVIGATE next_move hints so the agent must follow the rendered route.
    task_mode: str = "delivery"

    # Path to the DeliveryBench base directory (contains vlm_delivery and maps)
    # Defaults to the local copy in this package
    base_dir: str = _DEFAULT_BASE_DIR

    # Map name to use
    map_name: str = "medium-city-22"

    # Maximum steps per episode
    max_steps: int = 100

    # Render mode: "text" or "vision"
    render_mode: str = "text"

    # Image placeholder for vision mode
    image_placeholder: str = "<image>"

    # Whether to enable map image rendering
    enable_map_images: bool = True

    # Map renderer: "qt" or "pil"
    map_renderer: str = "qt"

    # Response format: "free_think" or "wm"
    prompt_format: str = "free_think"

    # Reward settings
    format_reward: float = 0.0
    success_reward: float = 1.0
    delivery_reward: float = 0.5  # Reward per successful delivery

    # Use example in system prompt
    use_example_in_sys_prompt: bool = True

    # Time scale for simulation.
    # 0.0 = deterministic text mode: sim clock only advances via explicit
    # clock.advance() calls (MOVE / STEP_TO / WAIT / CHARGE / REST).
    # Real wall-clock time (e.g. API latency between steps) has zero effect.
    # Use 1.0 only when running with a real-time UE viewer.
    time_scale: float = 0.0

    # Simulation-time limit in hours; episode is truncated when exceeded.
    time_limit_hours: float = 2.0

    # Move the static POI directory into the system prompt so it isn't
    # repeated every step.  Per-step observations still include agent
    # position, next hops, intersections, and order endpoints.
    map_poi_in_system_prompt: bool = False
    store_catalog_in_system_prompt: bool = True
    enable_battery: bool = True
    enable_walking_energy: bool = True
    enable_food_temperature: bool = True
    enable_food_smell: bool = True
    enable_food_fragility: bool = False
    enable_bag_compartments: bool = True
    exchange_tau_min: Optional[float] = None
    tau_external_min: Optional[float] = None
    # Odor mixing time constant (min) inside a bag compartment.
    odor_mix_tau_min: Optional[float] = None
    # Bare-food (carried outside the bag) cooling time constant in minutes.
    k_food_tau_min: Optional[float] = None
    # Spawn overrides; None keeps game_mechanics defaults.
    initial_battery_pct: Optional[float] = None
    initial_energy_pct: Optional[float] = None
    # Drain-rate multipliers on the game_mechanics per-meter costs (None = 1.0).
    battery_drain_scale: Optional[float] = None
    energy_drain_scale: Optional[float] = None
    # Per-seed step budget: at reset set max_steps = ceil(chained BFS oracle
    # over the first budget_orders orders × this multiplier). None = static.
    # The budget is a floor, not a cap: the pool refills, so efficient agents
    # can deliver more than budget_orders within the same budget.
    dynamic_max_steps_mult: Optional[float] = None
    budget_orders: int = 1
    # Map-portable battery scarcity: spawn charge = workload distance × ratio
    # × drain, i.e. the scooter can cover about `ratio` of the budgeted
    # workload. Needs dynamic_max_steps_mult; overrides initial_battery_pct.
    initial_battery_ratio: Optional[float] = None
    enable_fragility_compartment_damage: bool = False
    fragility_bump_every_n_moves: int = 4
    enable_advanced_transport: bool = True
    enable_delivery_methods: bool = False
    enable_special_notes: bool = True
    enable_multi_agent: bool = True
    enabled_actions: Optional[List[str]] = None
    seed: Optional[int] = None
    enable_prep_time: bool = True
    # Transport mode the agent starts in ("walk", "e-scooter", ...).
    # None keeps the simulator default (e-scooter). Walk-only curriculum
    # stages should set "walk" so the agent's speed matches NAVIGATE_WALK
    # estimates and the walking-based deadline pricing.
    initial_transport_mode: Optional[str] = None
    # Multiplier applied to every order's time limit after spawning.
    # >1.0 loosens deadlines (deadlines are priced on pickup->dropoff
    # distance only, so the approach leg eats into the budget).
    deadline_multiplier: float = 1.0
    max_orders_in_pool: Optional[int] = None
    num_restaurants: Optional[int] = None   
    num_customers: Optional[int] = None
    require_single_item: bool = False
    enable_earning_jitter: bool = True
    # Order-pool feasibility filter. These flags control which structural
    # order classes may be shown by VIEW_ORDERS, using an oracle shortest-path
    # step estimate from the agent's current position at VIEW_ORDERS time.
    enable_feasible_orders: bool = True
    enable_infeasible_orders: bool = True
    feasible_order_step_budget: int = 20
    feasible_order_non_move_actions: int = 5
    fixed_spawn_position: Optional[List[float]] = None  # [x_meters, y_meters]
    initial_facing_deg: float = 0.0  # 0=N, 90=E, 180=S, 270=W
    # When True, the per-step observation includes a ### waypoints section
    # listing the agent's current waypoint and adjacent waypoints, and
    # orders show street addresses alongside coordinates. STEP_TO actions
    # rely on this graph but are always available regardless of this flag.
    enable_waypoints: bool = True

    # First-person view (FPV) settings.
    # When enabled, each observation includes an egocentric image showing the
    # view in the agent's direction of travel, loaded from the pre-captured
    # FPV dataset.  Only meaningful when render_mode="vision".
    enable_fpv: bool = False
    # Path to the FPV dataset root.  If None, auto-detected as:
    #   <this_dir>/deliverybench_fpv/<map_name>
    fpv_dir: Optional[str] = None
    # Optional traffic-light variant FPV dataset. Use this when normal yaw
    # captures live in fpv_dir but red/green light captures are stored in a
    # separate traffic-light-only dataset.
    traffic_light_fpv_dir: Optional[str] = None
    # Calibration: the UE-captured FPV yaw is a left-handed REFLECTION of the sim
    # compass (0=N/90=E/...). To fetch the image facing compass direction D we use
    # stored_yaw = (fpv_yaw_offset_deg - D) mod 360 — i.e. compass(yaw)=K-yaw with
    # K=this constant. Default 90 (UE +X→East).
    fpv_yaw_offset_deg: float = 90.0
    # Set-of-Marks navigation (F1). When True, every one-hop reachable
    # waypoint is drawn as a numbered glow marker in the FPV cross (when
    # enable_fpv is also on), the per-step observation lists the same numbers
    # as a candidate block, and MOVE_TO(k) / MOVE_TO("<waypoint id>") becomes
    # available alongside the classic 4-direction MOVE. Default False keeps
    # the environment byte-identical to the pre-marks behaviour.
    enable_waypoint_marks: bool = False
    # When True, only the global (full-city) map snapshot is returned;
    # the local (zoomed) snapshot is dropped.
    map_global_only: bool = False
    # When True, use the Google-Maps-style Pillow renderer (render_gmaps.py)
    # instead of the basic map_exportor.  The static background is cached after
    # the first call so per-step cost is just a copy + pin/agent overlay.
    # out_scale controls the final image size (0.5 → ~1200 px on the long side).
    use_gmaps_renderer: bool = False
    gmaps_out_scale: float = 0.5

    # --- Pluggable static hazards (safety / social navigation). Default-OFF.
    # See OBSTACLE_TRAFFIC_DESIGN.md + deliverybench_fpv/FPV_CAPTURE_PLAN.md.
    # Both read per-map sidecars from the FPV dataset dir (obstacles.json /
    # traffic_lights.json) and are vision-only (the hazard is shown in the FPV,
    # not named in text). Enabling a flag whose sidecar is absent fails loudly at
    # reset. Counters surface in info["metrics"]["traj_metrics"]; reward untouched.
    enable_obstacles: bool = False
    enable_traffic_lights: bool = False
    # Enable pedestrian-signal checks derived from the map/world traffic-light
    # records. When on, MOVE across a controlled waypoint edge records green/red
    # checks, red crossings count as violations, and FPV panels use matching
    # green/red light captures when available.
    enable_pedestrian_traffic_lights: bool = False
    traffic_light_control_radius_cm: float = 650.0
    traffic_light_red_penalty_s: float = 15.0
    traffic_light_red_energy_multiplier: float = 1.0
    traffic_light_require_visible_signal_view: bool = True
    # Time/energy multiplier for BYPASS()/PASSBY() vs a normal MOVE(forward).
    passby_cost_scale: float = 1.5

    # Reserved for visual route-following probes; the default delivery-like mode
    # still expects the agent to choose and call NAVIGATE itself.
    visual_route_target: str = "restaurant 1"


# ─── Composable, oracle-verified presets ────────────────────────────────────
# nav is the base world (pure execution, e-scooter, pool=1). The bundles in
# _BUNDLES are mechanic packages that stack in any combination on top of it
# via make_config(); each carries the operating point where oracle_harness
# proved its mechanics REAL (2026-07-02 runs; small-city-11, spawn
# [-17, 256.58], e-scooter, budget = chained BFS oracle × ~2).
# pool controls order-selection breadth only (accepted orders refill);
# max_steps controls workload. Verify new bundle COMBINATIONS with the
# harness before relying on them — bundles compose, verdicts don't.

_NAV_ACTIONS = ["VIEW_ORDERS", "ACCEPT_ORDER", "MOVE", "NAVIGATE",
                "PICKUP", "DROP_OFF", "WAIT"]

# Base world: pool=1 keeps order-selection luck out of the reward; battery
# mechanic off → the e-scooter never depletes. Everything unverified stays
# dead code (fragility wiring-incomplete, car/bus untested, notes unverified).
_BASE = dict(
    map_name="small-city-11",
    fixed_spawn_position=[-17, 256.58],
    initial_transport_mode="e-scooter",
    initial_battery_pct=100,
    enable_battery=False,
    enable_walking_energy=False,
    enable_food_temperature=False,
    enable_food_smell=False,
    enable_bag_compartments=False,
    enable_food_fragility=False,
    enable_advanced_transport=False,
    enable_delivery_methods=False,
    enable_special_notes=False,
    enable_multi_agent=False,
    enable_prep_time=False,
    enable_earning_jitter=False,
    # energy/battery drains ignore their enable flags (transport.py applies
    # them per-meter unconditionally), so zero them when the mechanics are off
    energy_drain_scale=0.0,
    battery_drain_scale=0.0,
    max_orders_in_pool=1,
    num_restaurants=1,
    num_customers=1,
    require_single_item=True,
    deadline_multiplier=1.0,            # all oracle verdicts measured at 1.0
    # Per-seed budget = chained BFS oracle over budget_orders × 2.5, matching
    # the harness BUDGET_MULT the verdicts were measured at. Static max_steps
    # is only the fallback for infeasible estimates.
    dynamic_max_steps_mult=2.5,
    budget_orders=1,
    max_steps=40,
)

_BUNDLES: Dict[str, Dict[str, Any]] = {
    # battery (spawn 15%: act_pre +1.11..+1.87 REAL at 3 deliveries) +
    # energy (100%: +1 delivered, +$7.69 REAL at 2). pool=3 is required for
    # charge-amount planning: the agent must see upcoming workload.
    "resource": dict(
        enable_battery=True,
        enable_walking_energy=True,
        # Scooter covers ~20% of the budgeted workload on ANY map
        # (≈15% spawn on small-city-11); pct fallback for static budgets.
        initial_battery_ratio=0.2,
        initial_battery_pct=15,
        energy_drain_scale=1.0,
        battery_drain_scale=1.0,
        max_orders_in_pool=3, num_restaurants=3, num_customers=3,
        budget_orders=3, max_steps=250,   # 3 deliveries (oracle 89-132)
        extra_actions=["CHARGE", "REST", "SWITCH", "BUY",
                       "USE_BATTERY_PACK", "USE_ENERGY_DRINK"],
    ),
    # thermal + smell at e-scooter speed (k_food tau=4min, odor tau=1.5min):
    # hot naive(bag) +2.5..+5.0 REAL, packs +1.0..+3.5; cold naive +6.25 /
    # packs +4.5 REAL (mixed thermal orders need packs + separation);
    # smell: mixing punished −$5.0. Multi-item orders.
    "food_care": dict(
        enable_food_temperature=True,
        enable_food_smell=True,
        enable_bag_compartments=True,
        require_single_item=False,
        max_orders_in_pool=3, num_restaurants=3, num_customers=3,
        budget_orders=2, max_steps=80,    # 1-2 deliveries + store/bag actions
        extra_actions=["BUY", "PLACE_FOOD_IN_BAG", "USE_HEAT_PACK",
                       "USE_ICE_PACK", "VIEW_BAG"],
    ),
    # static hazards / safety navigation (vision-only; no oracle verdict yet).
    # Needs baked sidecars (obstacles.json) — small-city-11 has none, so this
    # bundle moves to small-city-15; combining it relocates other bundles off
    # their verified map.
    "hazard": dict(
        map_name="small-city-15",
        fpv_dir=str(Path(__file__).parent / "deliverybench_fpv" / "small-city-15"
                    / "main_base_floor_road_full_1280x960_clean_floor"),
        enable_obstacles=True,
        enable_traffic_lights=True,
        enable_pedestrian_traffic_lights=True,
        extra_actions=["PASSBY"],
    ),
}

# Fields where stacking takes the max instead of last-writer-wins.
_MAX_FIELDS = ("max_steps", "max_orders_in_pool", "num_restaurants",
               "num_customers", "budget_orders")


def make_config(*bundles: str, **overrides) -> DeliveryBenchEnvConfig:
    """Stack mechanic bundles on the nav base, e.g. make_config("resource",
    "food_care", max_orders_in_pool=5). No bundles = plain nav."""
    cfg = dict(_BASE)
    actions = list(_NAV_ACTIONS)
    for name in bundles:
        b = dict(_BUNDLES[name])
        actions += [a for a in b.pop("extra_actions", []) if a not in actions]
        for k, v in b.items():
            if k in _MAX_FIELDS:
                cfg[k] = max(cfg[k], v)
            elif k == "require_single_item":
                cfg[k] = cfg[k] and v
            else:
                cfg[k] = v
    cfg["enabled_actions"] = actions
    cfg.update(overrides)
    return DeliveryBenchEnvConfig(**cfg)


# Named recipes: (bundles, preset-level overrides). budget_orders in "full"
# merges to 3 (resource); the budget guarantees those deliveries per seed
# (floor, not cap), pool=5 only widens selection.
_PRESET_RECIPES: Dict[str, Tuple[Tuple[str, ...], Dict[str, Any]]] = {
    "nav": ((), {}),
    "resource": (("resource",), {}),
    "food_care": (("food_care",), {}),
    "hazard": (("hazard",), {}),
    "full": (("resource", "food_care"),
             dict(max_orders_in_pool=5, num_restaurants=5, num_customers=5)),
}


def preset_config(name: str, **overrides) -> DeliveryBenchEnvConfig:
    """Build a named preset with optional field overrides on top."""
    bundles, preset_overrides = _PRESET_RECIPES[name]
    return make_config(*bundles, **{**preset_overrides, **overrides})


PRESETS = {name: preset_config(name) for name in _PRESET_RECIPES}
NAV_PRESET = PRESETS["nav"]

VISUAL_ROUTE_FOLLOWING_CONFIG = DeliveryBenchEnvConfig(
    task_mode="visual_route_following",
    map_name="small-city-11",
    max_steps=25,
    render_mode="vision",
    enable_map_images=True,
    map_renderer="pil",
    enable_fpv=True,
    use_gmaps_renderer=True,
    gmaps_out_scale=0.5,
    enabled_actions=["VIEW_ORDERS", "ACCEPT_ORDER", "PICKUP", "DROP_OFF",
                     "WAIT", "MOVE", "NAVIGATE"],
    enable_battery=False,
    enable_walking_energy=False,
    enable_food_temperature=False,
    enable_food_smell=False,
    enable_food_fragility=False,
    enable_bag_compartments=False,
    enable_advanced_transport=False,
    enable_delivery_methods=False,
    enable_special_notes=False,
    enable_multi_agent=False,
    enable_prep_time=False,
    initial_transport_mode="walk",
    fixed_spawn_position=[-17.0, 256.58],
    visual_route_target="restaurant 1",
    deadline_multiplier=1.5,
    max_orders_in_pool=5,
    require_single_item=True,
    enable_earning_jitter=True,
    enable_feasible_orders=True,
    enable_infeasible_orders=False,
    feasible_order_step_budget=20,
    feasible_order_non_move_actions=5,
)

class DeliveryBench(GymImageEnv):
    """
    DeliveryBench environment that implements the GymImageEnv async interface.

    This wraps the DeliveryBenchGymEnvText from the DeliveryBench-text-only
    repository and adapts it to the VAGEN training interface.
    """

    def __init__(self, env_config: Dict[str, Any]):
        """
        Initialize the DeliveryBench environment.

        Args:
            env_config: Configuration dictionary matching DeliveryBenchEnvConfig fields
        """
        if isinstance(env_config, DeliveryBenchEnvConfig):
            self.config = env_config
        else:
            self.config = DeliveryBenchEnvConfig(**env_config)

        # Lazy import - will be initialized on first reset
        self._env = None
        self._deliverybench_imported = False

        # State tracking
        self.total_reward: float = 0.0
        self.last_action: Optional[str] = None
        self.last_action_result: Dict[str, Any] = {}
        self.deliveries_completed: int = 0
        self._prev_earnings: float = 0.0
        self._recent_parsed_actions: List[str] = []
        self._repeat_limit: int = 3

        # FPV tracking: previous agent position for heading computation
        self._prev_dm_x: float = 0.0
        self._prev_dm_y: float = 0.0
        self._fpv_lookup: Optional[Dict[Tuple[float, float], Dict[float, Path]]] = None  # {(x_cm,y_cm): {yaw: path}}
        # Hazard FPV variants (populated from manifest "variant" lines if present;
        # empty until the capture pipeline bakes them — see FPV_CAPTURE_PLAN.md).
        self._fpv_obstacle_lookup: Dict[Tuple[float, float], Dict[float, Path]] = {}      # {(x,y): {yaw: path}}
        self._fpv_light_lookup: Dict[Tuple[float, float], Dict[Tuple[float, str], Path]] = {}  # {(x,y): {(yaw,state): path}}

        # GMaps renderer: cached (background Image, View) so the expensive static
        # render only happens once per map; per-step cost is copy + overlay only.
        self._gmaps_bg: Optional[Image.Image] = None
        self._gmaps_view: Optional[Any] = None

    def _ensure_imports(self):
        """Ensure DeliveryBench modules are importable."""
        if self._deliverybench_imported:
            return

        base_dir = Path(self.config.base_dir).resolve()
        if not base_dir.exists():
            raise RuntimeError(f"DeliveryBench base_dir not found: {base_dir}")

        # Import from local vlm_delivery package
        try:
            from .vlm_delivery.gym_like_interface import DeliveryBenchGymEnvText
            self._DeliveryBenchGymEnvText = DeliveryBenchGymEnvText
            self._deliverybench_imported = True
        except ImportError as e:
            raise RuntimeError(
                f"Failed to import DeliveryBench modules from local package. Error: {e}"
            )

    def _apply_dynamic_budget(self, _dm, om) -> None:
        """Per-seed step budget: chained BFS oracle (order i starts at order
        i-1's dropoff) over the first budget_orders orders × the multiplier.
        Falls back to the static max_steps when any estimate is infeasible."""
        from .vlm_delivery.utils.order_feasibility import estimate_order_steps
        n = max(1, int(self.config.budget_orders))
        chain = om.list_orders()[:n]
        x, y = float(_dm.x), float(_dm.y)
        total = 0
        dist_m = 0.0
        for o in chain:
            est = estimate_order_steps(
                city_map=_dm.city_map, start_x_cm=x, start_y_cm=y,
                order=o, non_move_actions=6,
            )
            if not est.get("feasible_estimate_valid"):
                return
            total += int(est["total_steps"])
            dist_m += float(est["approach_m"]) + float(est["delivery_m"])
            dn = o.dropoff_node
            x, y = float(dn.position.x), float(dn.position.y)
        # budget_orders beyond the visible pool: refills are same-distribution,
        # so extrapolate the remainder at the visible chain's per-order mean.
        if chain and len(chain) < n:
            total += math.ceil(total / len(chain)) * (n - len(chain))
            dist_m += (dist_m / len(chain)) * (n - len(chain))
        self._env.max_steps = max(20, math.ceil(total * float(self.config.dynamic_max_steps_mult)))
        if (self.config.initial_battery_ratio is not None
                and self.config.enable_battery
                and getattr(_dm, "e_scooter", None) is not None):
            pct = dist_m * float(self.config.initial_battery_ratio) \
                * float(getattr(_dm, "scooter_batt_decay_pct_per_m", 0.04))
            _dm.e_scooter.set_battery_pct(max(1.0, min(100.0, pct)))

    def _create_env(self):
        """Create the underlying DeliveryBench environment."""
        self._ensure_imports()

        return self._DeliveryBenchGymEnvText(
            base_dir=self.config.base_dir,
            map_name=self.config.map_name,
            time_scale=self.config.time_scale,
            max_steps=self.config.max_steps,
            enable_map_images=self.config.enable_map_images and self.config.render_mode == "vision",
            map_renderer=self.config.map_renderer,
            enable_vlm=False,  # We handle VLM externally
        )

    # ------------------------------
    # GymImageEnv abstract methods
    # ------------------------------

    async def close(self) -> None:
        """Close the environment and release resources."""
        if self._env is not None:
            # DeliveryBench close is synchronous
            if self._is_visual_route_following():
                self._env.close()
            else:
                await asyncio.to_thread(self._env.close)
            self._env = None

    async def system_prompt(self) -> Dict[str, Any]:
        """
        Return the system prompt for the environment.

        Uses the original DeliveryBench system prompt from gameplay/prompt.py
        which includes action space, rules, observation description, and output format.

        If map_poi_in_system_prompt is enabled, the static POI directory
        (names + coordinates, without per-step distances) is appended once
        here so it doesn't need to be repeated in every observation.

        Returns:
            Dict with obs_str containing the system prompt
        """
        base_cfg = getattr(self._env, "cfg", None) if self._env is not None else None
        cfg = dict(base_cfg or {})
        cfg["enable_battery"] = self.config.enable_battery
        cfg["enable_walking_energy"] = self.config.enable_walking_energy
        cfg["enable_food_temperature"] = self.config.enable_food_temperature
        cfg["enable_food_smell"] = self.config.enable_food_smell
        cfg["enable_food_fragility"] = self.config.enable_food_fragility
        cfg["enable_bag_compartments"] = self.config.enable_bag_compartments
        cfg["enable_advanced_transport"] = self.config.enable_advanced_transport
        cfg["enable_multi_agent"] = self.config.enable_multi_agent
        cfg["enable_delivery_methods"] = self.config.enable_delivery_methods
        cfg["enable_special_notes"] = self.config.enable_special_notes
        cfg["enabled_actions"] = self.config.enabled_actions
        cfg["enable_obstacles"] = self.config.enable_obstacles
        cfg["enable_traffic_lights"] = self.config.enable_traffic_lights
        cfg["task_mode"] = self.config.task_mode
        # Hazard flags must reach the prompt cfg so effective_enabled_actions()
        # (used by get_action_spec) adds PASSBY when obstacles are on, and so the
        # BYPASS spec line can switch to its obstacle-aware wording.
        cfg["enable_obstacles"] = self.config.enable_obstacles
        cfg["enable_traffic_lights"] = self.config.enable_traffic_lights
        # Same reason for waypoint marks: the flag gates the MOVE_TO spec
        # bullet + output example and auto-adds MOVE_TO to the whitelist.
        cfg["enable_waypoint_marks"] = self.config.enable_waypoint_marks
        prompt = get_system_prompt(cfg)
        if self.config.map_poi_in_system_prompt and self._env is not None:
            poi_text = self._build_static_poi_directory()
            if poi_text:
                prompt += "\n\n" + poi_text
        if self.config.store_catalog_in_system_prompt and self._env is not None:
            catalog_text = self._build_static_store_catalog()
            if catalog_text:
                prompt += "\n\n" + catalog_text
        if self.config.enable_obstacles:
            prompt += (
                "\n\n### obstacle_navigation\n"
                "If the FRONT first-person view shows an obstacle blocking the intended "
                "forward route, use PASSBY() instead of MOVE(direction=\"forward\")."
            )
        if self.config.enable_waypoint_marks:
            _marks_note = (
                "\n\n### waypoint_marks_navigation\n"
                "Each observation ends with a ### waypoint_marks section listing "
                "every adjacent waypoint you can move to this step, numbered "
                "1..K."
            )
            if self.config.enable_fpv and self.config.render_mode == "vision":
                _marks_note += (
                    " The same numbers are drawn as glowing circular markers on "
                    "the first-person views, each marker sitting on the road at "
                    "its waypoint."
                )
            _marks_note += (
                " Move by replying MOVE_TO(<number>) (the waypoint id from the "
                "list also works, e.g. MOVE_TO(\"dock_94\")). The numbering is "
                "recomputed each step for your current position — always read "
                "it from the CURRENT observation."
            )
            prompt += _marks_note
        if self.config.enable_fpv and self.config.render_mode == "vision":
            _fpv_desc = (
                "First-person views: a single image tiling the four egocentric views "
                "around you — FRONT (centre, largest), BACK (top), LEFT and RIGHT "
                "(sides), each labelled. Use it to check if there is any abnormal situation."
            )
            if self.config.enable_map_images:
                _map_desc = (
                    "Top-down city map (North is up), zoomed in around you and your "
                    "active task. Markers:\n"
                    "   - YOU: a blue dot with an orange arrow attached to it. "
                    "The orange arrow shows only your current facing direction; "
                    "it is not a route arrow.\n"
                    "   - PICKUP (start of a delivery): a RED teardrop pin labelled "
                    "\"#N\u2191\" (N = order id), placed at the restaurant you collect "
                    "from. Restaurants are shown as YELLOW circles.\n"
                    "   - DROP-OFF: a GRAY teardrop pin labelled "
                    "\"#N\u2193\" at the customer's address.\n"
                    "   - ROUTE: when you have called NAVIGATE, a blue highlighted "
                    "line shows the active route from your current marker toward "
                    "the destination. Small arrows may appear on longer blue "
                    "segments to show route direction, but they may be absent on "
                    "short segments or near the final step. The line updates as "
                    "you move.\n"
                    "   - A legend in the bottom-right corner restates these symbols.\n"
                    "After NAVIGATE succeeds, keep following the persistent route and "
                    "the refreshed next_move hint instead of calling NAVIGATE again "
                    "every turn. Treat next_move as authoritative: move forward, "
                    "move backward, turn left, and turn right map directly to the "
                    "same MOVE direction. Do not override it based on street "
                    "numbers or your own guess. Only deviate if energy/battery is "
                    "insufficient or the direction is blocked/action failed."
                )
                prompt += (
                    "\n\n### vision_inputs\n"
                    f"Each observation includes two images in this order:\n"
                    f"1. {_fpv_desc}\n"
                    f"2. {_map_desc}\n"
                    "Cross-check the views against the top-down map and current-facing arrow so you don't get lost."
                )
            else:
                prompt += "\n\n### first_person_view\n" + _fpv_desc
        if self._is_visual_route_following():
            prompt = self._visual_route_following_prompt(prompt)
        return {
            "obs_str": prompt,
        }
    def _build_static_store_catalog(self) -> str:
        """Build store catalog text for inclusion in system prompt."""
        if self._env is None or not self._env.dms:
            return ""
        dm = self._env.dms[0]
        store_mgr = getattr(dm, "_store_manager", None)
        if store_mgr and hasattr(store_mgr, "to_text"):
            try:
                return "### store_catalog\n" + store_mgr.to_text(title="Available items & effects")
            except Exception:
                return ""
        return ""
        
    def _build_static_poi_directory(self) -> str:
        """Build a static list of all POIs with names and coordinates (no distances)."""
        if self._env is None or not self._env.dms:
            return ""
        dm = self._env.dms[0]
        if not hasattr(dm, "city_map"):
            return ""

        city_map = dm.city_map
        poi_meta = getattr(city_map, "poi_meta", [])
        if not poi_meta:
            return ""

        building_like = {
            "restaurant", "store", "rest_area", "hospital",
            "car_rental", "customer", "building",
        }
        lines = ["### static_poi_directory",
                 "All named locations in the city (coordinates are fixed):"]
        seen = set()
        for meta in poi_meta:
            node = meta.get("node")
            if node is None:
                continue
            ptype = (getattr(node, "type", "") or "").lower()
            if ptype == "building":
                continue
            _skip = set()
            if not self.config.enable_battery:
                _skip.add("charging_station")
            if not self.config.enable_walking_energy:
                _skip.add("rest_area")
            if not self.config.enable_advanced_transport:
                _skip |= {"bus_station", "car_rental"}
            if ptype in _skip:
                continue
            name = getattr(node, "name", None) or ptype
            if name in seen:
                continue
            seen.add(name)
            if ptype in building_like:
                anchor = meta.get("door_node") or node
            else:
                anchor = node
            pos = getattr(anchor, "position", None)
            if pos is None:
                continue
            x_m = round(float(pos.x) / 100.0, 2)
            y_m = round(float(pos.y) / 100.0, 2)
            road = meta.get("road_name") or ""
            road_tail = f" | {road}" if road else ""
            lines.append(f"- {name}: ({x_m}m, {y_m}m){road_tail}")
        return "\n".join(lines)

    async def reset(self, seed: int) -> Tuple[Dict[str, Any], Dict[str, Any]]:
        """
        Reset the environment to initial state.

        Args:
            seed: Random seed for reproducibility

        Returns:
            Tuple of (observation, info)
        """
        # Create fresh environment
        self._env = self._create_env()

        # Reset the underlying environment
        if self._is_visual_route_following():
            raw_obs, raw_info = self._env.reset(seed=seed)
        else:
            raw_obs, raw_info = await asyncio.to_thread(self._env.reset, seed=seed)

        # Reset tracking
        self.total_reward = 0.0
        self.last_action = None
        self.last_action_result = {}
        self.deliveries_completed = 0
        self._recent_parsed_actions = []

        # FPV: lazy-load the lookup once (keyed on map_name so it persists across resets)
        if self.config.enable_fpv and self._fpv_lookup is None:
            self._load_fpv_lookup()

        if self._env and self._env.dms:
            self._prev_earnings = float(getattr(self._env.dms[0], "earnings_total", 0.0))
            _dm = self._env.dms[0]

            # Waypoint marks need a photo for every graph waypoint at all four
            # yaws — audit once per reset and fail loudly on gaps (the F0 probe
            # silently skipped 3/76 points; a hole here would put numbered
            # markers on gray panels).
            if (self.config.enable_waypoint_marks and self.config.enable_fpv
                    and self.config.render_mode == "vision"):
                self._audit_waypoint_marks_fpv_coverage(_dm)
            
            # DM config flags
            _dm.cfg["enable_battery"] = self.config.enable_battery
            _dm.cfg["enable_walking_energy"] = self.config.enable_walking_energy
            _dm.cfg["enable_food_temperature"] = self.config.enable_food_temperature
            _dm.cfg["enable_food_smell"] = self.config.enable_food_smell
            _dm.cfg["enable_food_fragility"] = self.config.enable_food_fragility
            _dm.cfg["enable_fragility_compartment_damage"] = self.config.enable_fragility_compartment_damage
            _dm.cfg["fragility_bump_every_n_moves"] = int(self.config.fragility_bump_every_n_moves)
            _dm.cfg["enable_bag_compartments"] = self.config.enable_bag_compartments
            _dm.cfg["enable_waypoints"] = self.config.enable_waypoints
            # These two were missing — get_valid_tools and get_valid_actions
            # read them from dm.cfg to filter the tools/actions list.
            _dm.cfg["enable_advanced_transport"] = self.config.enable_advanced_transport
            _dm.cfg["enable_multi_agent"] = self.config.enable_multi_agent
            _dm.cfg["enable_delivery_methods"] = self.config.enable_delivery_methods
            _dm.cfg["enable_special_notes"] = self.config.enable_special_notes
            _dm.cfg["enable_feasible_orders"] = self.config.enable_feasible_orders
            _dm.cfg["enable_infeasible_orders"] = self.config.enable_infeasible_orders
            _dm.cfg["feasible_order_step_budget"] = self.config.feasible_order_step_budget
            _dm.cfg["feasible_order_non_move_actions"] = self.config.feasible_order_non_move_actions
            _dm.cfg["passby_cost_scale"] = self.config.passby_cost_scale
            # Unified hazard flags must reach dm.cfg so effective_enabled_actions()
            # (parser + valid_actions hint) auto-adds PASSBY/WAIT when the mechanic
            # is on — otherwise the _MECHANIC_ACTIONS coupling is inert.
            _dm.cfg["enable_obstacles"] = self.config.enable_obstacles
            _dm.cfg["enable_traffic_lights"] = self.config.enable_traffic_lights
            # Waypoint marks: parse_action gates MOVE_TO on dm.cfg, and the
            # handler + candidate text read the same flag.
            _dm.cfg["enable_waypoint_marks"] = self.config.enable_waypoint_marks
            _dm.cfg["enable_pedestrian_traffic_lights"] = self.config.enable_pedestrian_traffic_lights
            _dm.cfg["traffic_lights"] = {
                "control_radius_cm": self.config.traffic_light_control_radius_cm,
                "red_light_penalty_s": self.config.traffic_light_red_penalty_s,
                "red_light_energy_multiplier": self.config.traffic_light_red_energy_multiplier,
                "require_visible_signal_view": self.config.traffic_light_require_visible_signal_view,
                "fpv_yaw_offset_deg": self.config.fpv_yaw_offset_deg,
            }
            if self.config.traffic_light_require_visible_signal_view and self._fpv_light_lookup:
                _dm.cfg["traffic_lights"]["visible_signal_views"] = [
                    (float(pos[0]), float(pos[1]), float(yaw))
                    for pos, variants in self._fpv_light_lookup.items()
                    for yaw, _state in variants
                ]
            if self.config.enabled_actions is not None:
                _dm.cfg["enabled_actions"] = self.config.enabled_actions

            # Pluggable static hazards (default-off). Loaded per reset; counters
            # zeroed. Attaches _obstacle_field / _traffic onto the dm for the
            # MOVE / PASSBY / WAIT handlers; both are None when disabled.
            _dm.collision_count = 0
            _dm.traffic_violation_count = 0
            self._load_hazards(_dm)
            
            # Transport mode: the simulator spawns the agent on an e-scooter;
            # walk-only stages override this so actual speed matches the
            # NAVIGATE_WALK estimates and the walking-based deadline pricing.
            if self.config.initial_transport_mode:
                transport_set_mode(_dm, TransportMode(self.config.initial_transport_mode))

            # Fixed spawn: place at the given world position then snap to the
            # nearest waypoint so dm.x/dm.y is on the road graph (dock node),
            # matching what _current_waypoint() and the map renderer expect.
            if self.config.fixed_spawn_position is not None:
                _dm.x = self.config.fixed_spawn_position[0] * 100
                _dm.y = self.config.fixed_spawn_position[1] * 100
                city_map = getattr(_dm, "city_map", None)
                if city_map is not None and hasattr(city_map, "nearest_waypoint"):
                    wp = city_map.nearest_waypoint(float(_dm.x), float(_dm.y))
                    if wp is not None:
                        _dm.x = float(wp.position.x)
                        _dm.y = float(wp.position.y)

            # Initial facing is deterministic; MOVE rotates it.
            _dm.facing_deg = float(self.config.initial_facing_deg) % 360.0
            self._prev_dm_x = float(_dm.x)
            self._prev_dm_y = float(_dm.y)

            # Bag flags
            _bag = getattr(_dm, 'insulated_bag', None)
            if _bag is not None:
                _bag._enable_temperature = self.config.enable_food_temperature
                _bag._enable_smell = self.config.enable_food_smell
                _bag._enable_fragility = self.config.enable_food_fragility
                if self.config.exchange_tau_min is not None:
                    _bag.exchange_tau_min = float(self.config.exchange_tau_min)
                if self.config.tau_external_min is not None:
                    _bag.tau_external_min = float(self.config.tau_external_min)
                if self.config.odor_mix_tau_min is not None:
                    _bag.odor_mix_tau_min = float(self.config.odor_mix_tau_min)

            # Mechanic state overrides (timer.py bare-food gates read the dm attrs)
            _dm._enable_food_temperature = self.config.enable_food_temperature
            _dm._enable_food_smell = self.config.enable_food_smell
            if self.config.k_food_tau_min is not None:
                _dm.k_food_per_s = 1.0 / (float(self.config.k_food_tau_min) * 60.0)
            if self.config.initial_battery_pct is not None and getattr(_dm, "e_scooter", None) is not None:
                _dm.e_scooter.set_battery_pct(float(self.config.initial_battery_pct))
            if self.config.initial_energy_pct is not None:
                _dm.energy_pct = float(self.config.initial_energy_pct)
            if self.config.battery_drain_scale is not None:
                _dm.scooter_batt_decay_pct_per_m *= float(self.config.battery_drain_scale)
            if self.config.energy_drain_scale is not None:
                _s = float(self.config.energy_drain_scale)
                _dm.energy_cost_by_mode = {k: v * _s for k, v in _dm.energy_cost_by_mode.items()}

            # Order manager: regenerate with env seed + apply flags.
            # Constraint flags MUST be set before fill_pool: _spawn_one_order
            # copies them onto each order at spawn time (prep_longest_s,
            # jitter, notes are baked in then and not recomputed later).
            om = getattr(_dm, '_order_manager', None)
            if om:
                om._seed = seed
                om._id_counter = 0
                om._orders = []
                om._show_temperature = self.config.enable_food_temperature
                om._disable_prep_time = not self.config.enable_prep_time
                om._disable_jitter = not self.config.enable_earning_jitter
                if not self.config.enable_special_notes and hasattr(om, '_note_prob'):
                    om._note_prob = 0.0

                if hasattr(om, '_city_map'):
                    om.fill_pool(om._city_map, om._world_nodes)

                # Curriculum filters: keep only orders from the first N
                # restaurants / customer roads seen in the initial pool.
                allowed_pickup_roads = None
                allowed_dropoff_roads = None
                if self.config.num_restaurants is not None or self.config.num_customers is not None:
                    orders = om._orders
                    if self.config.num_restaurants is not None:
                        rests = list(dict.fromkeys(getattr(o, 'pickup_road_name', '') for o in orders))
                        allowed_pickup_roads = set(rests[:self.config.num_restaurants])
                        orders = [o for o in orders if getattr(o, 'pickup_road_name', '') in allowed_pickup_roads]
                    if self.config.num_customers is not None:
                        custs = list(dict.fromkeys(getattr(o, 'dropoff_road_name', '') for o in orders))
                        allowed_dropoff_roads = set(custs[:self.config.num_customers])
                        orders = [o for o in orders if getattr(o, 'dropoff_road_name', '') in allowed_dropoff_roads]
                    om._orders = orders

                if self.config.max_orders_in_pool is not None:
                    om.capacity = self.config.max_orders_in_pool
                    if len(om._orders) > self.config.max_orders_in_pool:
                        om._orders = om._orders[:self.config.max_orders_in_pool]

                for o in om._orders:
                    if self.config.require_single_item and len(getattr(o, 'items', []) or []) > 1:
                        o.items = o.items[:1]
                    if self.config.deadline_multiplier != 1.0:
                        o.time_limit_s *= self.config.deadline_multiplier

                # The pool refills via fill_pool after each completed order;
                # wrap the spawner so refilled orders obey the same filters.
                self._install_constrained_spawn(om, allowed_pickup_roads, allowed_dropoff_roads)

                if self.config.dynamic_max_steps_mult:
                    self._apply_dynamic_budget(_dm, om)
        else:
            self._prev_earnings = 0.0

        # Build observation
        obs = await self._build_observation(raw_obs, raw_info, init_obs=True)

        info: Dict[str, Any] = {
            "seed": seed,
            "raw_info": raw_info,
        }
        if self._is_visual_route_following():
            info.update(self._visual_route_following_info())

        return obs, info

    async def step(self, action_str: str) -> Tuple[Dict[str, Any], float, bool, Dict[str, Any]]:
        """
        Execute one step in the environment.

        Args:
            action_str: The action string from the model

        Returns:
            Tuple of (observation, reward, done, info)
        """
        if self._env is None:
            raise RuntimeError("Environment not reset. Call reset() first.")

        # Parse the model's response to extract the action
        parsed = parse_response(
            response=action_str,
            prompt_format=self.config.prompt_format,
        )

        action = parsed.get("action")
        format_correct = parsed.get("format_correct", False)
        parsed_action_str = action or ""

        # Capture previous position (kept for diagnostics; FPV now uses dm.facing_deg).
        if self._env and self._env.dms:
            _dm0 = self._env.dms[0]
            self._prev_dm_x = float(getattr(_dm0, "x", self._prev_dm_x))
            self._prev_dm_y = float(getattr(_dm0, "y", self._prev_dm_y))

        # Execute the action in the environment
        reward = 0.0
        done = False
        info: Dict[str, Any] = {"parsed": parsed, "is_tool": False}
        oracle_before = self._visual_route_following_info() if self._is_visual_route_following() else {}

        try:
            if action:
                # Reset last_action_kind so parse failures never inherit a stale tool kind.
                try:
                    _dm = self._env.dms[0] if self._env.dms else None
                    if _dm is not None:
                        _dm.last_action_kind = None
                except Exception:
                    pass

                # Execute action
                if self._is_visual_route_following():
                    raw_obs, raw_reward, terminated, truncated, raw_info = self._env.step(action)
                else:
                    raw_obs, raw_reward, terminated, truncated, raw_info = await asyncio.to_thread(
                        self._env.step, action
                    )

                self.last_action = action
                self.last_action_result = raw_info

                # Populate is_tool from the kind recorded by dm_start_action().
                # None means parse failed before dm_start_action was reached.
                try:
                    _dm = self._env.dms[0] if self._env.dms else None
                    if _dm is not None and getattr(_dm, "last_action_kind", None) is not None:
                        info["is_tool"] = _dm.last_action_kind in TOOL_ACTION_KINDS
                except Exception:
                    pass

                # Check for errors. Handler-level failures (PICKUP not ready,
                # DROP_OFF too far, ...) land in dm.vlm_errors, not raw_info —
                # surface them here, since build_state_observation deliberately
                # omits error feedback and expects the harness to relay it.
                _dm = self._env.dms[0] if self._env.dms else None
                dm_error = getattr(_dm, "vlm_errors", None) if _dm is not None else None
                if dm_error and not raw_info.get("error"):
                    raw_info["error"] = dm_error
                if _dm is not None:
                    _dm.vlm_clear_errors()
                if raw_info.get("error"):
                    error_msg = raw_info["error"]
                    info["action_error"] = error_msg
                    format_correct = False
                    

                done = terminated or truncated

                # Compute reward
                reward = self._compute_reward(raw_info, format_correct)

            else:
                # No valid action parsed
                raw_obs = self._env._build_obs() if hasattr(self._env, '_build_obs') else {}
                error_msg = "No valid action parsed from response"
                raw_info = {"error": error_msg}
                info["action_error"] = error_msg
                

        except Exception as e:
            # Handle action execution error
            raw_obs = self._env._build_obs() if hasattr(self._env, '_build_obs') else {}
            error_msg = str(e)
            raw_info = {"error": error_msg}
            info["action_error"] = error_msg
            # Pass error to underlying environment so it appears in next observation
            

        # Build metrics
        sim_hours = self._get_sim_hours()
        metrics = {
            "turn_metrics": {
                "action_is_valid": format_correct and not info.get("action_error"),
                "action_is_effective": not info.get("action_error"),
            },
            "traj_metrics": {
                "success": self._check_success(),
                "deliveries_completed": self.deliveries_completed,
                "sim_hours": sim_hours,
                "time_limit_reached": self._sim_time_exceeded(),
                # Pluggable hazard counters (0 when the features are off). Surfaced
                # for metrics/penalty use; not wired into reward.
                "collisions": int(getattr(self._env.dms[0], "collision_count", 0))
                if (self._env and self._env.dms) else 0,
                "traffic_violations": int(getattr(self._env.dms[0], "traffic_violation_count", 0))
                if (self._env and self._env.dms) else 0,
                "pedestrian_traffic_light_checks": len(getattr(self._env.dms[0], "traffic_light_checks", []) or [])
                if (self._env and self._env.dms) else 0,
                "pedestrian_traffic_light_violations": len(getattr(self._env.dms[0], "traffic_light_violations", []) or [])
                if (self._env and self._env.dms) else 0,
            },
        }

        # The underlying env never terminates on its own; end the episode
        # when the simulated shift is over.
        if metrics["traj_metrics"]["time_limit_reached"]:
            done = True

        # Anti-stuck guard: terminate only on the SAME action FAILING repeatedly.
        # A successful streak (e.g. many MOVE(forward) down a corridor) is normal
        # locomotion and must never trigger this — so the counter resets on any
        # effective action.
        if metrics["turn_metrics"]["action_is_effective"]:
            self._recent_parsed_actions = []
        elif parsed_action_str:
            self._recent_parsed_actions.append(parsed_action_str)
            recent = self._recent_parsed_actions[-self._repeat_limit:]
            if len(recent) >= self._repeat_limit and len(set(recent)) == 1:
                info["action_error"] = (
                    f"Terminated: action '{parsed_action_str}' failed "
                    f"{self._repeat_limit} times in a row."
                )
                done = True

        info["metrics"] = metrics
        info["success"] = metrics["traj_metrics"]["success"]
        info["raw_info"] = raw_info
        if self._is_visual_route_following():
            if oracle_before:
                info["oracle_next_move_before_action"] = oracle_before.get("oracle_next_move")
                info["oracle_next_action_before_action"] = oracle_before.get("oracle_next_action")
            info.update(self._visual_route_following_info())

        self.total_reward += reward

        # Build observation
        obs = await self._build_observation(raw_obs, raw_info, init_obs=False)

        return obs, reward, done, info

    # ------------------------------
    # Internal helpers
    # ------------------------------

    def _is_visual_route_following(self) -> bool:
        return str(getattr(self.config, "task_mode", "")).lower() == "visual_route_following"

    @staticmethod
    def _strip_visual_navigation_fields(text: str) -> str:
        lines = []
        stale_prefixes = {
            "distance_m:": "planned_route_distance_m:",
            "estimated_time:": "planned_route_estimated_time:",
            "estimated_personal_energy:": "planned_route_estimated_personal_energy:",
            "estimated_battery:": "planned_route_estimated_battery:",
        }
        for line in str(text or "").splitlines():
            lower = line.strip().lower()
            if lower.startswith(("next_move:", "from:")):
                continue
            for old, new in stale_prefixes.items():
                if lower.startswith(old):
                    value = line.split(":", 1)[1] if ":" in line else ""
                    line = f"{new} {value.strip()}"
                    break
            lines.append(line)
        return "\n".join(lines)

    def _visual_route_following_prompt(self, prompt: str) -> str:
        prompt = prompt.replace(
            "After NAVIGATE succeeds, follow the refreshed `next_move` exactly until it says `you have arrived`. "
            "Translate `next_move` directly to MOVE. When `next_move` says `you have arrived`, stop moving: "
            "call `PICKUP` if you are at the pickup and the order is ready, or call `DROP_OFF` if you are carrying "
            "the order at the dropoff.",
            "After NAVIGATE succeeds, the blue highlighted line on the map image is the active navigation route. "
            "When small arrows appear on the blue line, they indicate the route direction. However, arrows may be "
            "absent on short segments or near the final step, so the blue line itself remains the primary navigation "
            "signal. Use this blue line, your current marker, and the orange facing arrow to choose each "
            "MOVE direction. The orange arrow attached to your marker shows only your current facing direction; "
            "it is not a route arrow. NAVIGATE does not provide a textual next-step hint in this mode. "
            "Workflow gates: after ACCEPT_ORDER, if there is no `[navigation]` block for the pickup destination, "
            "your next action must be NAVIGATE with the exact Pickup address, not MOVE. After PICKUP succeeds, if "
            "there is no `[navigation]` block for the dropoff destination, your next action must be NAVIGATE with "
            "the exact Dropoff address. Once a `[navigation]` block is active for the current destination, do not "
            "repeat NAVIGATE for the same target; follow the blue highlighted route line with MOVE(direction=...). "
            "The `[navigation]` `planned_route_*` fields are estimates from the last NAVIGATE route plan. "
            "They may be stale after MOVE. Do not use them to judge "
            "current progress or arrival. Use the blue highlighted route line, current location marker, orange facing arrow, and "
            "`[pickup_hint]`/`[dropoff_hint]` instead. "
            "If the blue highlighted route line still continues beyond your current marker and the "
            "relevant `[pickup_hint]` or `[dropoff_hint]` is absent, you have not arrived; your next action must "
            "be MOVE(direction=...) that keeps you on the blue route line, not PICKUP or DROP_OFF. "
            "Do not infer arrival or route direction from street names, address numbers, or nearby pins alone; "
            "they are only approximate labels, while the blue route line, orange facing arrow, and environment hints "
            "are authoritative. "
            "For pickup, do not call PICKUP just because the address, pin, or street number looks close; wait until "
            "the observation includes `[pickup_hint]`, then call the exact PICKUP command shown there. For dropoff, "
            "wait until `[dropoff_hint]` appears, then call the exact DROP_OFF command shown there.",
        )
        prompt = prompt.replace(
            "After a successful NAVIGATE, later observations keep the [navigation] block and refresh next_move until you arrive or choose a different destination. Treat next_move as authoritative and translate it directly to MOVE: move forward -> MOVE(direction=\"forward\"), move backward -> MOVE(direction=\"backward\"), turn left -> MOVE(direction=\"left\"), turn right -> MOVE(direction=\"right\"). Do not override next_move based on street numbers or your own guess. Only deviate if the move is impossible because energy/battery is insufficient or the direction is blocked/action failed.",
            "After a successful NAVIGATE, later observations keep the [navigation] block and the blue route remains highlighted on the map image until you arrive or choose a different destination. In this mode there is no textual next-step hint; read the blue route line, any small arrows on longer blue segments, your current marker, and the orange facing arrow on your marker to choose MOVE(direction=...). Blue-line arrows indicate route direction when present, but they may be absent on short segments or near the final step. The orange arrow only shows your current facing; it is not a route arrow. Keep following the blue line until it no longer continues beyond your current marker or the relevant `[pickup_hint]`/`[dropoff_hint]` appears.",
        )
        prompt = prompt.replace(
            "After a successful NAVIGATE, later observations keep the [navigation] block and refresh next_move until you arrive or choose a different destination. Treat next_move as authoritative and translate it directly to MOVE: move forward -> MOVE(direction=\"forward\"), move backward -> MOVE(direction=\"backward\"), turn left -> MOVE(direction=\"left\"), turn right -> MOVE(direction=\"right\"). Do not override next_move based on street numbers or your own guess. Only deviate if the move is impossible or the direction is blocked/action failed.",
            "After a successful NAVIGATE, later observations keep the [navigation] block and the blue route remains highlighted on the map image until you arrive or choose a different destination. In this mode there is no textual next-step hint; read the blue route line, any small arrows on longer blue segments, your current marker, and the orange facing arrow on your marker to choose MOVE(direction=...). Blue-line arrows indicate route direction when present, but they may be absent on short segments or near the final step. The orange arrow only shows your current facing; it is not a route arrow. Keep following the blue line until it no longer continues beyond your current marker or the relevant `[pickup_hint]`/`[dropoff_hint]` appears.",
        )
        prompt = prompt.replace(
            "After NAVIGATE succeeds, keep following the persistent route and the refreshed next_move hint instead of calling NAVIGATE again every turn. Treat next_move as authoritative: move forward, move backward, turn left, and turn right map directly to the same MOVE direction. Do not override it based on street numbers or your own guess. Only deviate if energy/battery is insufficient or the direction is blocked/action failed.",
            "After NAVIGATE succeeds, keep following the persistent blue route line highlighted on the map instead of calling NAVIGATE again every turn. This mode hides textual next-step hints; use the blue route line, any small arrows on longer blue segments, your current marker, and the orange facing arrow to choose each MOVE direction. Blue-line arrows indicate route direction when present, but they may be absent on short segments or near the final step. The orange arrow only shows your current facing; it is not a route arrow. Keep moving along the blue line until it no longer continues beyond your current marker or the relevant `[pickup_hint]`/`[dropoff_hint]` appears.",
        )
        prompt = prompt.replace('NAVIGATE(target="restaurant 1")', 'NAVIGATE(target="146 Church Ave")')
        return prompt

    def _visual_route_following_info(self) -> Dict[str, Any]:
        if self._env is None or not self._env.dms:
            return {"oracle_next_move": None, "oracle_next_action": None, "route_arrived": False}
        dm = self._env.dms[0]
        from .vlm_delivery.utils.vlm_prompt import _live_next_move_line

        line = _live_next_move_line(dm)
        phrase = line.split(":", 1)[1].strip() if ":" in line else None
        action = {
            "move forward": 'MOVE(direction="forward")',
            "move backward": 'MOVE(direction="backward")',
            "turn left": 'MOVE(direction="left")',
            "turn right": 'MOVE(direction="right")',
        }.get(phrase or "")
        return {
            "oracle_next_move": phrase,
            "oracle_next_action": action,
            "route_arrived": phrase == "you have arrived",
        }

    def _install_constrained_spawn(
        self,
        om: Any,
        allowed_pickup_roads: Optional[set],
        allowed_dropoff_roads: Optional[set],
    ) -> None:
        """Wrap om._spawn_one_order so mid-episode refills respect the
        curriculum filters (restaurants / customers / single item / deadline)."""
        single_item = self.config.require_single_item
        deadline_mult = self.config.deadline_multiplier
        if not (allowed_pickup_roads or allowed_dropoff_roads or single_item
                or deadline_mult != 1.0):
            return

        orig_spawn = om._spawn_one_order

        def constrained_spawn(city_map, world_nodes, _ue=None):
            order = None
            for _ in range(600):
                order = orig_spawn(city_map, world_nodes, _ue)
                if allowed_pickup_roads and getattr(order, 'pickup_road_name', '') not in allowed_pickup_roads:
                    continue
                if allowed_dropoff_roads and getattr(order, 'dropoff_road_name', '') not in allowed_dropoff_roads:
                    continue
                break
            else:
                logging.getLogger(__name__).warning(
                    "Constrained order spawn: no match in 600 tries; using last candidate."
                )
            if single_item and len(getattr(order, 'items', []) or []) > 1:
                order.items = order.items[:1]
            if deadline_mult != 1.0:
                order.time_limit_s *= deadline_mult
            return order

        om._spawn_one_order = constrained_spawn

    async def _build_observation(
        self,
        raw_obs: Dict[str, Any],
        raw_info: Dict[str, Any],
        init_obs: bool,
    ) -> Dict[str, Any]:
        """
        Build the observation dict in GymImageEnv format.

        Uses dm.build_vlm_input() directly to match the original DeliveryBench
        prompt format with ### sections (agent_state, active_orders, map_snapshot, etc.)

        Args:
            raw_obs: Raw observation from DeliveryBench
            raw_info: Raw info from DeliveryBench
            init_obs: Whether this is the initial observation

        Returns:
            Dict with obs_str and optionally multi_modal_input
        """
        # Get the text observation directly from dm.build_vlm_input()
        obs_str = self._get_text_observation()

        # Set-of-Marks (enable_waypoint_marks): enumerate the one-hop
        # candidates ONCE per observation and feed the SAME list to both the
        # FPV marker overlay and the ephemeral text block, so the rendered
        # marker count always equals the listed candidate count and number k
        # means the same waypoint in image and text.
        marks_cands: Optional[List[Dict[str, Any]]] = None
        if self.config.enable_waypoint_marks and self._env and self._env.dms:
            from .vlm_delivery.actions.move import enumerate_candidates
            _dm0 = self._env.dms[0]
            marks_cands = enumerate_candidates(_dm0)
            marks_text = self._waypoint_marks_text(marks_cands, _dm0)
            if marks_text:
                obs_str = f"{obs_str}\n\n{marks_text}"

        obs: Dict[str, Any] = {"obs_str": obs_str}

        # Add images if in vision mode
        if self.config.render_mode == "vision":
            images: List[Image.Image] = []

            # FPV: a single image combining the four egocentric views around the
            # agent (front/left/right/back) relative to its current facing.
            if self.config.enable_fpv:
                dm_ref = self._env.dms[0] if self._env and self._env.dms else None
                if dm_ref is not None:
                    pos_key = self._current_waypoint_xy()
                    facing = float(getattr(dm_ref, "facing_deg", 0.0))
                    if pos_key is not None:
                        cross = self._build_fpv_cross(pos_key, facing, dm_ref,
                                                      candidates=marks_cands)
                        if cross is not None:
                            images.append(cross)

            # Top-down map snapshots (optional, can be disabled for FPV-only mode)
            if self.config.enable_map_images:
                map_imgs = await self._get_map_images()
                images.extend(map_imgs)

            # Pick up images queued by query-only runtime actions
            # (e.g. VISUAL_NAVIGATE_WALK). They are appended after the standard
            # map snapshots and the queue is cleared so they do not persist.
            dm_ref = self._env.dms[0] if self._env and self._env.dms else None
            if dm_ref is not None and getattr(dm_ref, "vlm_ephemeral_images", None):
                images = list(images) + list(dm_ref.vlm_ephemeral_images)
                dm_ref.vlm_ephemeral_images.clear()
            if images:
                # Insert image placeholders at the beginning
                # Original system prompt mentions: "Global map snapshot" and "Local map snapshot"
                placeholders = self.config.image_placeholder * len(images)
                obs_str = f"{placeholders}\n\n{obs_str}"
                obs["obs_str"] = obs_str
                obs["multi_modal_input"] = {
                    self.config.image_placeholder: images
                }
        if 'obs_str' in obs:
            text = obs['obs_str']
          
            import re
            if not self.config.enable_walking_energy:
                text = re.sub(r'Rest energy recovery rate is \+[\d.]+%/min\.\s*', '', text)
            
          
            skip_kws = set()
            if not self.config.enable_battery:
                skip_kws.add('charging_station')
            if not self.config.enable_walking_energy:
                skip_kws.add('rest_area')
            if not self.config.enable_advanced_transport:
                skip_kws |= {'bus_station', 'car_rental'}
            if skip_kws:
                lines = text.split('\n')
                lines = [l for l in lines if not any(kw in l for kw in skip_kws)]
                text = '\n'.join(lines)
            
            obs['obs_str'] = text
        return obs
    
    def _get_text_observation(self) -> str:
        """Get the text observation from the DeliveryBench agent."""
        if self._env is None or not self._env.dms:
            return "Environment not initialized."

        dm = self._env.dms[0]

        # Prefer the clean state observation (no agent memory, no error feedback).
        # Fall back to legacy build_vlm_input for backward compatibility.
        try:
            if hasattr(dm, 'build_state_observation'):
                text = dm.build_state_observation()
            elif hasattr(dm, 'build_vlm_input'):
                text = dm.build_vlm_input()
            else:
                # If neither method exists, use basic fallback
                text = self._build_basic_fallback_obs(dm)

            # Strip static sections if configured to be in system prompt
            if self.config.map_poi_in_system_prompt:
                text = self._strip_static_poi_block(text)
            if self.config.store_catalog_in_system_prompt:
                text = self._strip_section(text, "store_catalog")
            if self._is_visual_route_following():
                text = self._strip_visual_navigation_fields(text)
            text = self._append_traffic_light_timing_hint(text, dm)

            once = getattr(dm, "vlm_ephemeral_once", None)
            if once:
                for tag in list(once):
                    try:
                        dm.vlm_ephemeral.pop(tag, None)
                    except Exception:
                        pass
                try:
                    once.clear()
                except Exception:
                    pass

            return text

        except Exception as e:
            # Last-resort fallback if everything fails
            return self._build_basic_fallback_obs(dm)

    def _append_traffic_light_timing_hint(self, text: str, dm: Any) -> str:
        """Append timing-only traffic-light info without revealing signal color."""
        if not self.config.enable_traffic_lights:
            return text
        traffic = getattr(dm, "_traffic", None)
        if traffic is None:
            return text
        try:
            x, y = float(getattr(dm, "x")), float(getattr(dm, "y"))
            if not traffic.is_signalised(x, y):
                return text
            seconds = max(0.0, float(traffic.seconds_to_next_minute(float(dm.clock.now_sim()))))
        except Exception:
            return text

        minutes = seconds / 60.0
        if minutes < 0.05:
            minutes_text = "0.1"
        else:
            minutes_text = f"{minutes:.1f}".rstrip("0").rstrip(".")
        hint = (
            f"You are at a traffic light crossing. The light will change in about "
            f"{minutes_text} minute(s)."
        )
        return f"{text.rstrip()}\n\n{hint}"
   


    def _build_basic_fallback_obs(self, dm) -> str:
        """Fallback observation builder when VLM methods are unavailable."""
        try:
            lines = []
            lines.append(f"Position: ({dm.x/100:.1f}m, {dm.y/100:.1f}m)")
            lines.append(f"Energy: {dm.energy_pct:.0f}%")
            lines.append(f"Earnings: ${getattr(dm, 'earnings_total', 0):.2f}")

            active_orders = getattr(dm, 'active_orders', []) or []
            if active_orders:
                order_ids = [str(getattr(o, 'id', '?')) for o in active_orders]
                lines.append(f"Active orders: {', '.join(order_ids)}")
            else:
                lines.append("No active orders. Use VIEW_ORDERS() to see available orders.")

            return "\n".join(lines)
        except Exception as e:
            return f"Error getting observation: {e}"

    @staticmethod
    def _strip_static_poi_block(text: str) -> str:
        """Remove the entire ### map_snapshot section from the observation."""
        import re
        return re.sub(
            r"\n?### map_snapshot\n(?:(?!### ).)*",
            "",
            text,
            flags=re.DOTALL,
        )
        
    def _strip_section(self, text: str, section_name: str) -> str:
        """Remove a ### section from observation text."""
        lines = text.split("\n")
        result = []
        skip = False
        for line in lines:
            if line.strip() == f"### {section_name}":
                skip = True
                continue
            if skip and line.strip().startswith("### "):
                skip = False
            if not skip:
                result.append(line)
        return "\n".join(result)

    async def _get_map_images(self) -> List[Image.Image]:
        """Get map images from the environment."""
        if self._env is None or not self._env.dms:
            return []

        dm = self._env.dms[0]

        if self.config.use_gmaps_renderer:
            try:
                # The PIL/GMaps renderer is fast but has shown executor hangs
                # when called via asyncio.to_thread during real-env export.
                # Render synchronously so reset/step can reliably emit images.
                img = self._render_gmaps_frame(dm)
                return [img] if img is not None else []
            except Exception as e:
                logging.getLogger(__name__).warning(f"GMaps render failed: {e}")
                return []

        try:
            if hasattr(dm, 'map_exportor') and dm.map_exportor is not None:
                orders = list(dm.active_orders) if getattr(dm, 'active_orders', None) else []
                global_bytes, local_bytes = await asyncio.to_thread(
                    dm.map_exportor.export,
                    agent_xy=(dm.x, dm.y),
                    orders=orders,
                )
                images = []
                if global_bytes:
                    images.append(Image.open(BytesIO(global_bytes)))
                if local_bytes and not self.config.map_global_only:
                    images.append(Image.open(BytesIO(local_bytes)))
                return images
        except Exception as e:
            logging.getLogger(__name__).warning(f"Failed to get map images: {e}")

        return []

    @staticmethod
    def _collect_road_names(dm) -> Optional[List]:
        """Named road segments from the simulator's skeleton graph, in cm.

        Passed to the GMaps renderer so street labels match the road names
        used in text observations and order addresses (the simulator names
        roads from a CRC-seeded pool; the renderer must not invent its own).
        """
        city_map = getattr(dm, "city_map", None)
        skel = getattr(city_map, "graph_skel", None)
        if skel is None:
            return None
        out = []
        for e in skel.edges:
            u, v = e.node1, e.node2
            meta = skel.get_edge_meta(u, v) or {}
            if meta.get("kind") != "road":
                continue
            name = meta.get("name") or ""
            if not name:
                continue
            display = city_map._display_road_name(name)
            out.append((
                display,
                ((float(u.position.x), float(u.position.y)),
                 (float(v.position.x), float(v.position.y))),
            ))
        return out or None

    def _render_gmaps_frame(self, dm) -> Optional[Image.Image]:
        """
        Render one frame with the GMaps-style renderer.

        The expensive static background (buildings, roads, labels …) is rendered
        once and cached in self._gmaps_bg / self._gmaps_view.  Each call is then
        a cheap copy + agent-dot + order-pin overlay.
        """
        from .tools.render_gmaps import render_background, compose_frame

        scenario_dir = Path(self.config.base_dir) / "maps" / self.config.map_name

        if self._gmaps_bg is None:
            bg, _world, view = render_background(
                scenario_dir, road_names=self._collect_road_names(dm),
                decorate=False,   # crop + decorations are applied per-frame below
            )
            self._gmaps_bg = bg
            self._gmaps_view = view
            logging.getLogger(__name__).info(
                f"GMaps background cached for {self.config.map_name} "
                f"({bg.width}×{bg.height} px)"
            )

        # Build order markers: red pin at pickup (if not yet carried),
        # green pin at dropoff.
        order_markers = []
        carrying_ids = set(getattr(dm, 'carrying', []) or [])
        for order in (getattr(dm, 'active_orders', None) or []):
            oid = str(getattr(order, 'id', '?'))
            pu_node = getattr(order, 'pickup_node', None)
            do_node = getattr(order, 'dropoff_node', None)
            if pu_node is not None and getattr(order, 'id', None) not in carrying_ids:
                pu_short = self._short_order_marker_address(pu_node)
                order_markers.append({
                    "x": float(pu_node.position.x),
                    "y": float(pu_node.position.y),
                    "kind": "pickup",
                    "label": f"#{oid}↑ {pu_short}" if pu_short else f"#{oid}↑",
                })
            if do_node is not None:
                do_short = self._short_order_marker_address(do_node)
                order_markers.append({
                    "x": float(do_node.position.x),
                    "y": float(do_node.position.y),
                    "kind": "dropoff",
                    "label": f"#{oid}↓ {do_short}" if do_short else f"#{oid}↓",
                })

        # Live navigation route: recompute the shortest path from the agent's
        # CURRENT waypoint to the stored destination every frame so the drawn
        # route shrinks/updates as the agent moves (NAVIGATE only sets the
        # destination + colour; the path itself is re-derived here).
        nav_route = self._live_nav_route(dm)

        return compose_frame(
            self._gmaps_bg,
            self._gmaps_view,
            agent_xy=(float(dm.x), float(dm.y)),
            order_markers=order_markers,
            out_scale=self.config.gmaps_out_scale,
            nav_route=nav_route,
            nav_color=getattr(dm, "_nav_route_color", None),
            agent_heading_deg=float(getattr(dm, "facing_deg", 0.0)),
            crop=True,
            decorate=True,
            route_waypoint_labels=self._is_visual_route_following(),
        )

    @staticmethod
    def _short_order_marker_address(node: Any) -> str:
        """Short address suffix for active-order map badges, e.g. '146'."""
        import re

        for attr in ("address", "waypoint_name", "waypoint_id"):
            value = str(getattr(node, attr, "") or "").strip()
            if not value:
                continue
            match = re.match(r"(\d+)\b", value)
            if match:
                return match.group(1)
        return ""

    @staticmethod
    def _live_nav_route(dm):
        """Shortest path from the agent's current waypoint to the active NAVIGATE
        target, recomputed each frame. Returns None when there's no live target
        or the agent has already arrived."""
        target_node = getattr(dm, "_nav_target_node", None)
        if target_node is not None:
            try:
                cmap = getattr(dm, "city_map", None)
                graph = getattr(cmap, "waypoint_graph", None)
                start = cmap.nearest_waypoint(float(dm.x), float(dm.y)) if cmap else None
                arrive_tol = 1000.0
                try:
                    from .vlm_delivery.utils.util import get_tol
                    arrive_tol = max(
                        float(get_tol(getattr(dm, "cfg", {}) or {}, "arrive", 500.0)),
                        float(get_tol(getattr(dm, "cfg", {}) or {}, "door", 1000.0)),
                    )
                except Exception:
                    pass
                if start is not None:
                    try:
                        if start is target_node or float(start.position.distance(target_node.position)) <= arrive_tol:
                            return None
                    except Exception:
                        if start is target_node:
                            return None
                if (graph is not None and start is not None and start is not target_node
                        and start in graph.adjacency_list
                        and target_node in graph.adjacency_list):
                    path, _ = graph.shortest_path_nodes(start, target_node)
                    if path and len(path) >= 2:
                        return path
                # arrived → no route to draw
            except Exception:
                pass
        # Fall back to the route stored by NAVIGATE (e.g. bus itineraries).
        return getattr(dm, "_nav_route_path", None) or None

    # ------------------------------------------------------------------
    # Pluggable hazards (obstacles + traffic lights)
    # ------------------------------------------------------------------

    def _hazard_root(self) -> Path:
        """Directory holding the per-map hazard sidecars (shared with the FPV
        dataset): obstacles.json / traffic_lights.json."""
        if self.config.fpv_dir:
            return Path(self.config.fpv_dir)
        return _THIS_DIR / "deliverybench_fpv" / self.config.map_name

    def _load_hazards(self, dm: Any) -> None:
        """Attach _obstacle_field / _traffic onto the dm per the config flags.

        Both default off → both attributes are None and there is zero behaviour
        change. When a flag is on, the matching sidecar is REQUIRED (fail loudly,
        so an enabled-but-invisible hazard can't slip through). Hazards are
        vision-only, so we also warn when vision/FPV is not active.
        """
        from .vlm_delivery.utils.hazards import ObstacleField, TrafficController

        dm._obstacle_field = None
        dm._traffic = None
        if not (self.config.enable_obstacles or self.config.enable_traffic_lights):
            return

        root = self._hazard_root()
        log = logging.getLogger(__name__)
        vision_on = self.config.render_mode == "vision" and self.config.enable_fpv
        if not vision_on:
            log.warning(
                "Hazards (obstacles/traffic_lights) are enabled but render_mode/"
                "enable_fpv is not vision+FPV — hazards are VISION-ONLY, so the "
                "agent will get no signal. Enable FPV vision."
            )

        if self.config.enable_obstacles:
            path = root / "obstacles.json"
            if not path.exists():
                raise RuntimeError(
                    f"enable_obstacles=True but obstacle sidecar not found: {path}. "
                    f"Bake it per deliverybench_fpv/FPV_CAPTURE_PLAN.md, or disable."
                )
            dm._obstacle_field = ObstacleField.load(path)
            log.info(f"Obstacles loaded: {len(dm._obstacle_field)} edges from {path}")

        if self.config.enable_traffic_lights:
            # Single source of truth, aligned to the rendered images: the
            # signalised set is derived from the FPV manifest's traffic-light
            # rows. A hand-authored traffic_lights.json sidecar (positions +
            # odd_minute_red_axis / minute_period_s) overrides it if present.
            sidecar = root / "traffic_lights.json"
            manifest = root / "manifest.jsonl"
            if sidecar.exists():
                dm._traffic = TrafficController.load(sidecar)
                src = sidecar
            elif manifest.exists():
                dm._traffic = TrafficController.load_manifest(manifest)
                src = manifest
            else:
                raise RuntimeError(
                    f"enable_traffic_lights=True but neither {sidecar} nor "
                    f"{manifest} exists. Bake the FPV traffic-light renders "
                    f"(see deliverybench_fpv/FPV_CAPTURE_PLAN.md), or disable."
                )
            if len(dm._traffic) == 0:
                raise RuntimeError(
                    f"enable_traffic_lights=True but no signalised intersections "
                    f"were found in {src}. The manifest needs render_kind="
                    f"'traffic_light' rows, or provide a traffic_lights.json sidecar."
                )
            log.info(
                f"Traffic lights loaded: {len(dm._traffic)} signalised "
                f"intersections from {src}"
            )

    # ------------------------------------------------------------------
    # FPV helpers
    # ------------------------------------------------------------------

    def _load_fpv_lookup(self) -> None:
        """Build {(x_cm, y_cm): {yaw_deg: Path}} from the FPV manifest.

        Keyed by **waypoint position, not waypoint_id**: the FPV capture and the
        runtime graph cover the identical set of physical waypoints, but their
        ``dock_*`` ids were renumbered between capture and the current graph code
        (~1/3 of ids mismatch), so an id-join would fetch the wrong building's
        photos. Position is a stable 1:1 key (positions match exactly). The image
        *file* path still uses the capture's own waypoint_id (that's how the image
        directories are named).
        """
        fpv_root = Path(self.config.fpv_dir) if self.config.fpv_dir else (
            _THIS_DIR / "deliverybench_fpv" / self.config.map_name
        )
        fpv_roots = [fpv_root]
        if self.config.traffic_light_fpv_dir:
            light_root = Path(self.config.traffic_light_fpv_dir)
            if light_root not in fpv_roots:
                fpv_roots.append(light_root)

        manifest = fpv_root / "manifest.jsonl"
        if not manifest.exists():
            logging.getLogger(__name__).warning(
                f"FPV manifest not found at {manifest}; FPV disabled."
            )
            self._fpv_lookup = {}
            return

        lookup: Dict[Tuple[float, float], Dict[float, Path]] = {}
        obstacle_lookup: Dict[Tuple[float, float], Dict[float, Path]] = {}
        light_lookup: Dict[Tuple[float, float], Dict[Tuple[float, str], Path]] = {}
        fallback_n = 0
        for root in fpv_roots:
            manifest = root / "manifest.jsonl"
            if not manifest.exists():
                logging.getLogger(__name__).warning(
                    f"FPV manifest not found at {manifest}; skipping."
                )
                continue
            with open(manifest, encoding="utf-8") as fh:
                for line in fh:
                    line = line.strip()
                    if not line:
                        continue
                    entry = json.loads(line)
                    if entry.get("status") != "ok":
                        continue
                    wp_id: str = entry["waypoint_id"]          # capture id (names the image dir)
                    yaw: float = float(entry["yaw"])           # 0 / 90 / 180 / 270
                    pos_key = (round(float(entry["x_cm"]), 1), round(float(entry["y_cm"]), 1))

                    # Local image dir from the capture's waypoint_id. Directories use
                    # zero-padded 3-digit numbers: "int_0" -> "int_000".
                    kind, num_str = wp_id.rsplit("_", 1)
                    dir_name = f"{kind}_{int(num_str):03d}"
                    base = root / "images" / dir_name

                    # Prefer the exact filename the manifest recorded (image_path
                    # basename) so we stay byte-aligned with what was rendered;
                    # fall back to the documented naming when it is absent.
                    img_name = (
                        Path(entry["image_path"]).name if entry.get("image_path") else None
                    )

                    # Classify the row. Traffic-light renders carry
                    # render_kind="traffic_light" + signal_state (green/red); obstacle
                    # renders carry render_kind="obstacle"; everything else is plain.
                    # The legacy variant="light"/"obstacle" + light_state schema is
                    # still accepted so older manifests keep working.
                    render_kind = str(entry.get("render_kind") or "").lower()
                    variant = str(entry.get("variant") or "").lower()
                    state = str(entry.get("signal_state") or entry.get("light_state") or "").lower()
                    def _resolve(fname: str) -> Path:
                        # Prefer the local images/ tree; when a checkout ships
                        # only the manifest (images/ not synced), fall back to
                        # (1) the manifest's own absolute image_path, then
                        # (2) the map root's images/ tree — datasets have been
                        # renamed in place (e.g. small-city-11-rerun -> map
                        # root) leaving manifests pointing at dead paths.
                        # Never silently degrade to gray panels.
                        nonlocal fallback_n
                        p = base / fname
                        if p.exists():
                            return p
                        if entry.get("image_path"):
                            alt = Path(entry["image_path"])
                            if alt.exists():
                                fallback_n += 1
                                return alt
                        alt2 = root.parent / "images" / dir_name / fname
                        if alt2.exists():
                            fallback_n += 1
                            return alt2
                        return p

                    if render_kind == "traffic_light" or variant == "light" or state in ("red", "green"):
                        st = state if state in ("red", "green") else "green"
                        fname = img_name or f"yaw_{int(yaw):03d}_{st}.png"
                        light_lookup.setdefault(pos_key, {})[(yaw, st)] = _resolve(fname)
                    elif render_kind == "obstacle" or variant == "obstacle":
                        fname = img_name or f"yaw_{int(yaw):03d}_blocked.png"
                        obstacle_lookup.setdefault(pos_key, {})[yaw] = _resolve(fname)
                    elif root == fpv_root:
                        fname = img_name or f"yaw_{int(yaw):03d}.png"
                        lookup.setdefault(pos_key, {})[yaw] = _resolve(fname)

        self._fpv_lookup = lookup
        self._fpv_obstacle_lookup = obstacle_lookup
        self._fpv_light_lookup = light_lookup
        if fallback_n:
            logging.getLogger(__name__).warning(
                f"FPV lookup: {fallback_n} image(s) missing under the local "
                f"images/ tree of {fpv_root} — using the manifest's absolute "
                f"image_path for those (shared-disk fallback)."
            )
        logging.getLogger(__name__).info(
            f"FPV lookup loaded: {len(lookup)} waypoints (keyed by position) from {fpv_root}"
            + (f"; obstacle variants: {len(obstacle_lookup)}" if obstacle_lookup else "")
            + (f"; light variants: {len(light_lookup)}" if light_lookup else "")
        )

    def _audit_waypoint_marks_fpv_coverage(self, dm: Any) -> None:
        """Reset-time audit (enable_waypoint_marks + vision FPV): every graph
        waypoint must resolve ALL four yaw photos to existing files, otherwise
        numbered markers would render on gray placeholder panels. Fails loudly
        (hazard-sidecar precedent) instead of silently skipping the way the F0
        probe did (3/76 points lost)."""
        city_map = getattr(dm, "city_map", None)
        graph = getattr(city_map, "waypoint_graph", None)
        nodes = list(getattr(graph, "adjacency_list", {}) or {})
        if not nodes:
            return
        missing: List[str] = []
        for node in nodes:
            pos_key = (round(float(node.position.x), 1),
                       round(float(node.position.y), 1))
            imgs = self._fpv_images_at(pos_key) or {}
            bad = [int(y) for y in (0.0, 90.0, 180.0, 270.0)
                   if imgs.get(y) is None or not Path(imgs[y]).exists()]
            if bad:
                missing.append(
                    f"{getattr(node, 'waypoint_id', pos_key)}: yaw {bad}")
        if missing:
            raise RuntimeError(
                f"enable_waypoint_marks=True but the FPV album does not cover "
                f"{len(missing)}/{len(nodes)} graph waypoints on "
                f"{self.config.map_name} (numbered markers would render on "
                f"gray panels). First misses: {missing[:5]}. Re-bake the "
                f"album or point fpv_dir at a complete dataset."
            )

    @staticmethod
    def _compute_heading_deg(
        prev_x: float, prev_y: float, curr_x: float, curr_y: float
    ) -> Optional[float]:
        """Compass heading (0=N / +y, 90=E / +x, 180=S, 270=W) from prev→curr."""
        dx = curr_x - prev_x
        dy = curr_y - prev_y
        if abs(dx) < 1e-4 and abs(dy) < 1e-4:
            return None
        # Same formula as map.py's _bearing_deg: atan2(dx, dy) gives
        # 0° when moving purely north (+y) and 90° when moving east (+x).
        return (math.degrees(math.atan2(dx, dy)) + 360.0) % 360.0

    @staticmethod
    def _snap_to_fpv_yaw(heading_deg: float) -> float:
        """Round heading to the nearest available FPV yaw (0, 90, 180, 270)."""
        candidates = [0.0, 90.0, 180.0, 270.0]
        # Handle wrap-around by also considering 360° ≡ 0°
        return min(candidates, key=lambda y: min(abs(y - heading_deg), 360.0 - abs(y - heading_deg)))

    def _fpv_images_at(self, pos_key: Optional[Tuple[float, float]]) -> Optional[Dict[float, Path]]:
        """Yaw→path dict for the FPV capture at ``pos_key`` (nearest within 50 cm)."""
        if not self._fpv_lookup or pos_key is None:
            return None
        hit = self._fpv_lookup.get(pos_key)
        if hit is not None:
            return hit
        # Robustness: snap to the closest captured position (positions normally
        # match exactly; this guards against tiny float drift).
        best, best_d2 = None, 50.0 * 50.0
        for (x, y), imgs in self._fpv_lookup.items():
            d2 = (x - pos_key[0]) ** 2 + (y - pos_key[1]) ** 2
            if d2 <= best_d2:
                best, best_d2 = imgs, d2
        return best

    def _get_fpv_image(self, pos_key: Optional[Tuple[float, float]], heading_deg: float) -> Optional[Image.Image]:
        """Return the PIL Image at the waypoint position ``pos_key`` for a yaw."""
        wp_images = self._fpv_images_at(pos_key)
        if not wp_images:
            return None
        yaw = self._snap_to_fpv_yaw(heading_deg)
        img_path = wp_images.get(yaw)
        if img_path is None:
            return None
        try:
            return Image.open(img_path).copy()
        except Exception as exc:
            logging.getLogger(__name__).debug(f"FPV load failed {img_path}: {exc}")
            return None

    def _current_waypoint_node(self):
        """The waypoint node the agent is currently standing on (or None)."""
        if self._env is None or not self._env.dms:
            return None
        dm = self._env.dms[0]
        city_map = getattr(dm, "city_map", None)
        if city_map is None or not hasattr(city_map, "nearest_waypoint"):
            return None
        return city_map.nearest_waypoint(float(dm.x), float(dm.y))

    def _current_waypoint_id(self) -> Optional[str]:
        """Runtime waypoint_id the agent is standing on (for debug / CLI display)."""
        node = self._current_waypoint_node()
        return getattr(node, "waypoint_id", None) if node else None

    def _current_waypoint_xy(self) -> Optional[Tuple[float, float]]:
        """Position key (rounded x_cm, y_cm) of the waypoint the agent stands on.

        This — not the waypoint_id — is the FPV lookup key (see _load_fpv_lookup).
        """
        node = self._current_waypoint_node()
        if node is None:
            return None
        return (round(float(node.position.x), 1), round(float(node.position.y), 1))

    def _waypoint_marks_text(self, cands: Optional[List[Dict[str, Any]]],
                             dm: Any) -> str:
        """Ephemeral ### waypoint_marks block (enable_waypoint_marks): one line
        per numbered candidate, phrased in panel language when the FPV is
        shown and MOVE-direction language otherwise. NEVER compass bearings —
        the graph bearing axis is flipped vs the rendered map's compass rose
        (F0 finding; see F1_WAYPOINT_MARKS_PLAN.md).
        """
        from .fpv_marks import move_words_for, panel_label_for
        if not cands:
            return ("### waypoint_marks\n"
                    "No adjacent waypoint is reachable from here.")
        fpv_on = self.config.render_mode == "vision" and self.config.enable_fpv
        facing = float(getattr(dm, "facing_deg", 0.0)) % 360.0
        head = (
            f"{len(cands)} reachable adjacent waypoint(s), shown as numbered "
            f"glowing markers on the first-person views. "
            if fpv_on else
            f"{len(cands)} reachable adjacent waypoint(s), numbered below. "
        )
        lines = ["### waypoint_marks", head + "Move with MOVE_TO(<number>)."]
        for c in cands:
            if fpv_on:
                where = (f"in your "
                         f"{panel_label_for(c['bearing_deg'], facing).upper()} VIEW")
            else:
                where = move_words_for(c["bearing_deg"], facing)
            nm = f" ({c['name']})" if c.get("name") else ""
            lines.append(f"- MOVE_TO({c['index']}): {c['id']}{nm}, "
                         f"{c['dist_cm'] / 100.0:.1f} m, {where}")
        return "\n".join(lines)

    @staticmethod
    def _load_fpv_path(path: Optional[Path]) -> Optional[Image.Image]:
        """Load a specific FPV variant image, or None if absent/unreadable."""
        if path is None or not Path(path).exists():
            return None
        try:
            return Image.open(path).copy()
        except Exception as exc:
            logging.getLogger(__name__).debug(f"FPV variant load failed {path}: {exc}")
            return None

    def _select_fpv_image(
        self, pos_key: Optional[Tuple[float, float]], compass_dir: float,
        yaw: float, dm: Any,
    ) -> Optional[Image.Image]:
        """Pick the FPV image for one panel: a hazard variant when applicable
        (traffic light > obstacle), else the plain capture. Falls back to plain
        whenever a variant line/image is missing, so default behaviour is intact.
        """
        if pos_key is not None and dm is not None:
            # Variant lookups are keyed by the canonical cardinal yaw; snap so a
            # slightly off-cardinal facing still resolves its green/red/blocked
            # image instead of silently falling back to the plain capture.
            key_yaw = self._snap_to_fpv_yaw(yaw)
            if (
                bool((getattr(dm, "cfg", {}) or {}).get("enable_pedestrian_traffic_lights"))
                and self._fpv_light_lookup
                and pos_key in self._fpv_light_lookup
            ):
                state = self._pedestrian_light_state_for_view(dm, compass_dir)
                img = self._load_fpv_path(self._fpv_light_lookup[pos_key].get((key_yaw, state)))
                if img is not None:
                    return img
            # Traffic-light variant (only at signalised intersections).
            # The panel
            # shows the colour for the crossing axis of the direction it looks at,
            # so the front and perpendicular panels render opposite signals — the
            # red-front / green-right case the agent must read. Pure function of
            # (sim time, axis), so the same panel re-renders deterministically and
            # flips when the clock advances past a minute boundary.
            traffic = getattr(dm, "_traffic", None)
            if traffic is not None and self._fpv_light_lookup and pos_key in self._fpv_light_lookup \
                    and traffic.is_signalised(pos_key[0], pos_key[1]):
                state = traffic.light(traffic.axis_of(compass_dir), float(dm.clock.now_sim()))
                img = self._load_fpv_path(self._fpv_light_lookup[pos_key].get((key_yaw, state)))
                if img is not None:
                    return img
            # Obstacle variant (image exists only for a baked blocked edge).
            if getattr(dm, "_obstacle_field", None) is not None and self._fpv_obstacle_lookup:
                img = self._load_fpv_path(self._fpv_obstacle_lookup.get(pos_key, {}).get(key_yaw))
                if img is not None:
                    return img
        return self._get_fpv_image(pos_key, yaw)

    @staticmethod
    def _pedestrian_light_state_for_view(dm: Any, compass_dir: float) -> str:
        """Signal state for the controlled MOVE edge visible in one FPV panel."""
        try:
            from .vlm_delivery.utils.traffic_lights import check_dm_edge_signal

            city_map = getattr(dm, "city_map", None)
            current = city_map.nearest_waypoint(float(dm.x), float(dm.y))
            if current is None:
                return "green"
            best = None
            best_diff = 30.0
            for adj in city_map.adjacents(current) or []:
                bearing = float(adj.get("bearing_deg", 0.0)) % 360.0
                diff = abs((bearing - float(compass_dir)) % 360.0)
                diff = min(diff, 360.0 - diff)
                if diff <= best_diff:
                    signal = check_dm_edge_signal(dm, current, adj["node"])
                    if signal:
                        best = (adj, signal)
                        best_diff = diff
            if best is None:
                return "green"
            _adj, signal = best
            return str((signal or {}).get("state") or "green").lower()
        except Exception:
            return "green"

    def _build_fpv_cross(
        self, pos_key: Optional[Tuple[float, float]], facing_deg: float, dm: Any = None,
        candidates: Optional[List[Dict[str, Any]]] = None,
    ) -> Optional[Image.Image]:
        """Compose the four egocentric views into one labelled image.

        Layout (relative to the agent's facing): BACK on top, LEFT and RIGHT on
        the sides, FRONT centred and 2× larger. Front is downscaled to ½×½ of a
        native FPV frame, the others to ¼×¼. Hazard variants (red/green light,
        obstacle) replace a panel when the agent's dm carries the matching field
        and the variant image is baked (see FPV_CAPTURE_PLAN.md).

        ``candidates`` (enable_waypoint_marks): numbered one-hop waypoints from
        ``enumerate_candidates``; each is drawn as a numbered glow marker on
        the FINAL canvas (fpv_marks.overlay_marks). None/empty leaves the
        composition byte-identical to the pre-marks behaviour.
        """
        from PIL import ImageDraw
        # UE's captured yaw is a REFLECTION of the sim compass (left-handed yaw),
        # so a compass direction D maps to stored yaw (off - D), NOT (D + off):
        # using a rotation made turns come out mirror-reversed (right showed left).
        # `off` (fpv_yaw_offset_deg, default 90) is the reflection constant K in
        # compass(yaw) = K - yaw, i.e. fetch yaw = K - D.
        off = float(getattr(self.config, "fpv_yaw_offset_deg", 90.0))
        # label -> (compass direction it looks at, stored yaw to fetch).
        # With the corrected FPV source images and UE-Y map view, side panels are
        # swapped so LEFT/RIGHT in the cross match the agent's visual frame.
        panels_dir = {
            "front": (facing_deg + 0.0) % 360.0,
            "right": (facing_deg + 270.0) % 360.0,
            "back": (facing_deg + 180.0) % 360.0,
            "left": (facing_deg + 90.0) % 360.0,
        }
        raw = {
            k: self._select_fpv_image(pos_key, cdir, (off - cdir) % 360.0, dm)
            for k, cdir in panels_dir.items()
        }
        ref = next((im for im in raw.values() if im is not None), None)
        if ref is None:
            return None
        w, h = ref.size
        fw, fh = w // 2, h // 2          # front (½×½)
        sw, sh = w // 4, h // 4          # side/back (¼×¼)

        def _panel(label: str, size):
            im = raw.get(label)
            tile = im.resize(size).convert("RGB") if im is not None else Image.new(
                "RGB", size, (40, 40, 40))
            d = ImageDraw.Draw(tile)
            txt = f"{label.upper()} VIEW"
            tw = d.textlength(txt) if hasattr(d, "textlength") else 8 * len(txt)
            d.rectangle([0, 0, tw + 8, 16], fill=(0, 0, 0))
            d.text((4, 2), txt, fill=(255, 255, 255))
            return tile

        canvas = Image.new("RGB", (w, sh + fh), (255, 255, 255))
        # BACK: top, centred
        canvas.paste(_panel("back", (sw, sh)), ((w - sw) // 2, 0))
        # middle row: LEFT | FRONT | RIGHT
        mid_y = sh
        canvas.paste(_panel("left", (sw, sh)), (0, mid_y + (fh - sh) // 2))
        canvas.paste(_panel("front", (fw, fh)), (sw, mid_y))
        canvas.paste(_panel("right", (sw, sh)), (sw + fw, mid_y + (fh - sh) // 2))

        if candidates:
            # Set-of-Marks overlay: geometry mirrors the paste layout above.
            from .fpv_marks import overlay_marks
            geom = {
                "back": ((w - sw) // 2, 0, sw, sh),
                "left": (0, mid_y + (fh - sh) // 2, sw, sh),
                "front": (sw, mid_y, fw, fh),
                "right": (sw + fw, mid_y + (fh - sh) // 2, sw, sh),
            }
            canvas, drawn = overlay_marks(canvas, geom, (w, h), candidates,
                                          facing_deg)
            self._last_marks_drawn = drawn  # gate/debug: must equal len(candidates)
        return canvas

    def _compute_reward(self, raw_info: Dict[str, Any], format_correct: bool) -> float:
        """
        Reward = change in dm.earnings_total since the last step,
        plus a small format bonus when the action parses correctly.
        """
        reward = 0.0

        if format_correct and not raw_info.get("error"):
            reward += self.config.format_reward

        if self._env and self._env.dms:
            dm = self._env.dms[0]
            cur_earnings = float(getattr(dm, "earnings_total", 0.0))
            reward += cur_earnings - self._prev_earnings
            self._prev_earnings = cur_earnings

            self.deliveries_completed = len(getattr(dm, "completed_orders", []) or [])

        return reward

    def _get_sim_hours(self) -> float:
        """Current simulation time in hours."""
        if self._env and self._env.dms:
            return self._env.dms[0].clock.now_sim() / 3600.0
        return 0.0

    def _sim_time_exceeded(self) -> bool:
        """True when simulation time has passed the configured limit."""
        if self.config.time_limit_hours <= 0:
            return False
        return self._get_sim_hours() >= self.config.time_limit_hours

    def _check_success(self) -> bool:
        """Episode is successful if at least one delivery was completed."""
        return self.deliveries_completed >= 1


# ------------------------------
# Local async test
# ------------------------------
if __name__ == "__main__":
    import os
    import logging

    logging.basicConfig(
        level=logging.INFO,
        format='[%(levelname)s] %(message)s'
    )

    async def main_async(
        render_mode: str = "text",
        max_steps: int = 50,
    ):
        cfg = {
            "render_mode": render_mode,
            "max_steps": max_steps,
            "prompt_format": "free_think",
        }
        env = DeliveryBench(cfg)

        print("System Prompt:")
        sys_prompt = await env.system_prompt()
        print(sys_prompt["obs_str"])
        print("\n" + "=" * 50 + "\n")

        obs, info = await env.reset(seed=42)
        print("Initial Observation:")
        print(obs["obs_str"])

        step = 0
        while True:
            step += 1
            print(f"\nStep {step}:")
            try:
                action_input = input("Enter action (or 'quit'): ")
            except EOFError:
                action_input = "quit"

            if action_input.lower() == "quit":
                break

            # Wrap in JSON format if needed
            if not action_input.startswith("{"):
                action_input = f'{{"action": "{action_input}"}}'

            obs, reward, done, info = await env.step(action_input)
            print(f"Reward: {reward}, Done: {done}")
            print(f"Observation:\n{obs['obs_str'][:500]}...")

            if done:
                print("Episode finished!")
                break

        print(f"\nTotal reward: {env.total_reward}")
        await env.close()

    def main(**kwargs):
        asyncio.run(main_async(**kwargs))

    import fire
    fire.Fire(main)
