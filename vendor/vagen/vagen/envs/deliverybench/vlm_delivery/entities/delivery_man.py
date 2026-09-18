# entities/delivery_man.py
# -*- coding: utf-8 -*-

import time
import os
import logging
from dataclasses import dataclass, field
from typing import Any, Dict, Optional, List, Callable, Deque, Set
from collections import deque
from concurrent.futures import Future, Executor

from ..base.timer import VirtualClock
from ..base.defs import *

from .escooter import EScooter
from .car import Car
from .store import StoreManager
from .insulated_bag import InsulatedBag
from .bus_manager import BusManager

from ..gameplay.comms import get_comms
from ..gameplay.prompt import get_system_prompt
from ..gameplay.action_space import action_to_text
from ..gameplay.run_recorder import RunRecorder

from ..utils.util import *
from ..utils.global_logger import get_agent_logger
from ..utils.trajectory_recorder import make_run_folder
from ..utils.vlm_runtime import *
from ..utils.timer import agent_timers_pause, agent_timers_resume, handle_poll_time_events as timer_poll_time_events
from ..utils.transport import *
from ..utils.vlm_prompt import *
from ..utils.action_runtime import *
from ..utils.ui import *
from ..utils.viewer import viewer_bind_viewer

from ..actions.say import handle_say as action_handle_say
from ..actions.move import handle_move as action_handle_move
from ..actions.move import handle_move_to as action_handle_move_to
from ..actions.passby import handle_passby as action_handle_passby
from ..actions.accept_orders import handle_accept_order as action_handle_accept_order
from ..actions.view_orders import handle_view_orders as action_handle_view_orders
from ..actions.view_help_board import handle_view_help_board as action_handle_view_help_board
from ..actions.view_bag import handle_view_bag as action_handle_view_bag
from ..actions.pick_up_food import handle_pickup_food as action_handle_pickup_food
from ..actions.drop_off import (
    handle_drop_off as action_handle_drop_off,
    auto_try_dropoff as action_auto_try_dropoff,
)
from ..actions.charge_escooter import (
    handle_charge_escooter as action_handle_charge_escooter,
    advance_charge_to_now as action_advance_charge_to_now,
)
from ..actions.wait import handle_wait as action_handle_wait
from ..actions.rest import handle_rest as action_handle_rest
from ..actions.buy import handle_buy as action_handle_buy
from ..actions.use_battery_pack import handle_use_battery_pack as action_handle_use_battery_pack
from ..actions.use_energy_drink import handle_use_energy_drink as action_handle_use_energy_drink
from ..actions.use_ice_pack import handle_use_ice_pack as action_handle_use_ice_pack
from ..actions.use_heat_pack import handle_use_heat_pack as action_handle_use_heat_pack
from ..actions.post_help_request import handle_post_help_request as action_handle_post_help_request
from ..actions.accept_help_request import handle_accept_help_request as action_handle_accept_help_request
from ..actions.edit_help_request import handle_edit_help_request as action_handle_edit_help_request
from ..actions.switch_transport import handle_switch_transport as action_handle_switch_transport
from ..actions.rent_car import handle_rent_car as action_handle_rent_car
from ..actions.return_car import handle_return_car as action_handle_return_car
from ..actions.place_temp_box import handle_place_temp_box as action_handle_place_temp_box
from ..actions.take_from_temp_box import handle_take_from_temp_box as action_handle_take_from_temp_box
from ..actions.place_food_in_bag import handle_place_food_in_bag as action_handle_place_food_in_bag
from ..actions.report_help_finished import handle_report_help_finished as action_handle_report_help_finished
from ..actions.board_bus import handle_board_bus as action_handle_board_bus
from ..actions.view_bus_schedule import handle_view_bus_schedule as action_handle_view_bus_schedule
from ..actions.navigate import handle_navigate as action_handle_navigate

from typing import TYPE_CHECKING
if TYPE_CHECKING:
    from ..vlm.base_model import BaseModel

@dataclass
class DeliveryMan:
    # === Core identity & world ===
    agent_id: str
    city_map: Any
    world_nodes: List[Dict[str, Any]]
    x: float
    y: float

    # === Basic motion / config ===
    mode: TransportMode = TransportMode.WALK
    clock: VirtualClock = field(default_factory=lambda: VirtualClock())
    cfg: Dict[str, Any] = field(default_factory=dict)

    # Runtime speed / pace
    speed_cm_s: float = field(init=False)
    pace_state: str = "normal"  # "accel" / "normal" / "decel"
    pace_scales: Dict[str, float] = field(init=False, repr=False)

    # Energy & resting
    energy_pct: float = field(init=False, default=100.0)
    rest_rate_pct_per_min: float = field(default=8.0)
    low_energy_threshold_pct: float = field(default=30.0)

    # Earnings & rescue
    earnings_total: float = field(default=100.0)
    is_rescued: bool = field(default=False)
    hospital_rescue_fee: float = 0.0

    # === Orders & carrying ===
    active_orders: List[Any] = field(default_factory=list)
    carrying: List[int] = field(default_factory=list)
    completed_orders: List[Dict[str, Any]] = field(default_factory=list)

    # --- HELP DELIVERY tracking ---
    helping_order_ids: Set[int] = field(default_factory=set)              # orders I am currently helping deliver (as helper)
    _help_delivery_req_by_oid: Dict[int, int] = field(default_factory=dict, repr=False)  # order_id -> req_id
    help_completed_order_ids: Set[int] = field(default_factory=set)       # as publisher: orders already completed via Comms and settled
    _helping_wait_ack_oids: Set[int] = field(default_factory=set, repr=False)            # as helper: orders reported finished, waiting for publisher settlement

    help_orders: Dict[int, Any] = field(default_factory=dict)          # as helper: active helper orders (oid -> order_obj)
    help_orders_completed: Set[int] = field(default_factory=set)       # as helper: orders I have delivered and reported (bounty may or may not be received)
    accepted_help: Dict[int, Any] = field(default_factory=dict)        # req_id -> HelpRequest (I accepted as helper)
    completed_help: Dict[int, Any] = field(default_factory=dict)       # req_id -> HelpRequest (help already completed)

    # === Equipment / items ===
    insulated_bag: Optional[InsulatedBag] = None
    e_scooter: Optional[EScooter] = None
    assist_scooter: Optional[EScooter] = None  # someone else's scooter (can drag/charge but not ride)
    car: Optional[Car] = None
    inventory: Dict[str, int] = field(default_factory=dict)
    charge_price_per_pct: float = field(default=0.1)
    towing_scooter: bool = False

    # === Viewer / UE bindings ===
    name: str = "DM"
    _viewer: Optional[Any] = field(default=None, repr=False)
    _viewer_agent_id: Optional[str] = field(default=None, repr=False)
    _ue: Optional[Any] = field(default=None, repr=False)

    # === Managers ===
    _order_manager: Optional[Any] = field(default=None, repr=False)
    _store_manager: Optional[StoreManager] = field(default=None, repr=False)
    _bus_manager: Optional[BusManager] = field(default=None, repr=False)

    # === Scheduling / current action ===
    _queue: List[DMAction] = field(default_factory=list, repr=False)
    _current: Optional[DMAction] = field(default=None, repr=False)
    _previous_language_plan: Optional[str] = field(default=None, repr=False)
    _action_handlers: Dict[DMActionKind, Callable[['DeliveryMan', DMAction, bool], None]] = field(init=False, repr=False)

    # === Lifecycle / metrics ===
    _recorder: Optional[RunRecorder] = field(default=None, repr=False)
    _sim_active_elapsed_s: float = 0.0
    _lifecycle_done: bool = False

    # Realtime lifecycle tracking
    _realtime_start_ts: Optional[float] = field(default=None, repr=False)
    _realtime_stop_hours: float = 0.0

    # Charging / resting / waiting / hospital contexts (simulation time)
    _charge_ctx: Optional[Dict[str, Any]] = field(default=None, repr=False)
    _rest_ctx: Optional[Dict[str, float]] = field(default=None, repr=False)
    _wait_ctx: Optional[Dict[str, float]] = field(default=None, repr=False)
    _hospital_ctx: Optional[Dict[str, float]] = field(default=None, repr=False)

    # Movement / bus / rental contexts
    _move_ctx: Optional[Dict[str, float]] = field(default=None, repr=False)      # {"tx": float, "ty": float, "tol": float, "blocked": 0/1}
    _rental_ctx: Optional[Dict[str, float]] = field(default=None, repr=False)    # {"last_tick_sim": float, "rate_per_min": float}
    _bus_ctx: Optional[Dict[str, float]] = field(default=None, repr=False)       # {"bus_id": str, "boarding_stop": str, "target_stop": str}

    # Traffic-light rollout state. ``last_traffic_light_check`` is overwritten
    # on each waypoint move; red crossings are also appended for episode export.
    last_traffic_light_check: Optional[Dict[str, Any]] = field(default=None, repr=False)
    traffic_light_checks: List[Dict[str, Any]] = field(default_factory=list, repr=False)
    traffic_light_violations: List[Dict[str, Any]] = field(default_factory=list, repr=False)

    # Pluggable-hazard runtime state (default-off; see OBSTACLE_TRAFFIC_DESIGN.md).
    # ``_traffic`` / ``_obstacle_field`` are attached by the env at reset; the
    # counters are surfaced in info["metrics"]["traj_metrics"] and reward is
    # untouched (red lights and obstacles are counted, not penalised, for now).
    _traffic: Optional[Any] = field(default=None, repr=False)
    _obstacle_field: Optional[Any] = field(default=None, repr=False)
    traffic_violation_count: int = field(default=0, repr=False)
    collision_count: int = field(default=0, repr=False)

    # Movement interruption
    _interrupt_move_flag: bool = field(default=False, repr=False)
    _interrupt_reason: Optional[str] = field(default=None, repr=False)

    # Food placement / forced bag placement
    _pending_food_by_order: Dict[int, List[Any]] = field(default_factory=dict, repr=False)
    _force_place_food_now: bool = False

    # Timers / temperature updates
    _timers_paused: bool = field(default=False, repr=False)
    _orders_last_tick_sim: Optional[float] = field(default=None, repr=False)
    _life_last_tick_sim: Optional[float] = field(default=None, repr=False)
    _last_bag_tick_sim: Optional[float] = None

    # === VLM state ===
    vlm_prompt: str = get_system_prompt()
    vlm_past_memory: List[str] = field(default_factory=list)
    vlm_ephemeral: Dict[str, str] = field(default_factory=dict)
    vlm_ephemeral_once: set = field(default_factory=set, repr=False)
    # Images queued by query-only runtime actions (e.g. VISUAL_NAVIGATE_WALK).
    # Picked up and cleared by the env's observation builder in vision mode.
    vlm_ephemeral_images: List[Any] = field(default_factory=list, repr=False)
    # Persistent nav route drawn on every map frame until overwritten by a new
    # VISUAL_NAVIGATE call.  Stored as (path_nodes, rgba_color) so compose_frame
    # can redraw the polyline on top of each fresh background copy.
    _nav_route_path: List[Any] = field(default_factory=list, repr=False)
    _nav_route_color: Optional[tuple] = field(default=None, repr=False)
    # The NAVIGATE destination node; the route overlay persists on the map until
    # another NAVIGATE overwrites it or the agent reaches this node (MOVE clears).
    _nav_target_node: Optional[Any] = field(default=None, repr=False)
    # Persistent compass facing (0=N, 90=E, 180=S, 270=W). Updated by MOVE:
    # forward keeps it, right +90, backward +180, left +270 (mod 360). Read by
    # the wrapper for the first-person view and the map facing-arrow.
    facing_deg: float = 0.0
    vlm_errors: Optional[str] = None
    vlm_last_actions: Deque[str] = field(default_factory=lambda: deque(maxlen=5), repr=False)
    vlm_last_compiled_input: Optional[str] = None

    # VLM async channel
    _vlm_executor: Optional[Executor] = field(default=None, repr=False)
    _vlm_future: Optional[Future] = field(default=None, repr=False)
    _vlm_results_q: Deque[Dict[str, Any]] = field(default_factory=deque, repr=False)
    _vlm_inflight_token: Optional[int] = field(default=None, repr=False)
    _vlm_token_ctr: int = field(default=0, repr=False)
    _waiting_vlm: bool = field(default=False, repr=False)

    _vlm_client: Optional["BaseModel"] = field(default=None, repr=False)
    _vlm_retry_count: int = field(default=0, repr=False)
    _vlm_retry_max: int = field(default=5, repr=False)
    _vlm_last_bad_output: Optional[str] = field(default=None, repr=False)

    # === Debug / logging / exports ===
    map_exportor: Optional[Any] = field(default=None, repr=False)
    current_step: int = field(default=0, repr=False)
    save_dir: str = field(default="debug_snaps", repr=False)

    logger: logging.Logger = field(init=False, repr=False)
    run_dir: str = field(default="", repr=False, compare=False)
    
    


    def __post_init__(self):
        """
        Post-initialization logic for DeliveryMan.
        Groups related initialization steps and adds clear English comments.
        """

        # ----------------------------------------------------------------------
        # 1. Logger
        # ----------------------------------------------------------------------
        self.logger = get_agent_logger(f"DeliveryMan{self.agent_id}")

        # ----------------------------------------------------------------------
        # 2. Speed & Energy Configurations
        # ----------------------------------------------------------------------
        # Average travel speeds by transport mode (cm/s)
        self.avg_speed_by_mode = {
            TransportMode.WALK:        self.cfg["avg_speed_cm_s"]["walk"],
            TransportMode.SCOOTER:     self.cfg["avg_speed_cm_s"]["e-scooter"],
            TransportMode.DRAG_SCOOTER:self.cfg["avg_speed_cm_s"]["drag_scooter"],
            TransportMode.CAR:         self.cfg["avg_speed_cm_s"]["car"],
            TransportMode.BUS:         self.cfg["avg_speed_cm_s"]["bus"],
        }

        # Energy decay (percentage per meter) by transport mode
        self.energy_cost_by_mode = {
            TransportMode.WALK:        self.cfg["energy_pct_decay_per_m_by_mode"]["walk"],
            TransportMode.DRAG_SCOOTER:self.cfg["energy_pct_decay_per_m_by_mode"]["drag_scooter"],
            TransportMode.SCOOTER:     self.cfg["energy_pct_decay_per_m_by_mode"]["e-scooter"],
            TransportMode.CAR:         self.cfg["energy_pct_decay_per_m_by_mode"]["car"],
            TransportMode.BUS:         self.cfg["energy_pct_decay_per_m_by_mode"]["bus"],
        }

        # Walking pace multipliers (acceleration/normal/deceleration)
        self.pace_scales = {
            "accel":  self.cfg["pace_scales"]["accel"],
            "normal": self.cfg["pace_scales"]["normal"],
            "decel":  self.cfg["pace_scales"]["decel"],
        }

        # Scooter-specific decay and hospital settings
        self.scooter_batt_decay_pct_per_m = self.cfg["scooter_batt_decay_pct_per_m"]
        self.hospital_duration_s = self.cfg["hospital_duration_s"]
        self.hospital_rescue_fee = float(self.cfg.get("hospital_rescue_fee", 0.0))

        # Costs and thresholds
        self.charge_price_per_pct = self.cfg["charge_price_per_percent"]
        self.rest_rate_pct_per_min = self.cfg["rest_rate_pct_per_min"]
        self.low_energy_threshold_pct = self.cfg["low_energy_threshold_pct"]

        # Initial agent status
        self.energy_pct = self.cfg["energy_pct_max"]
        self.earnings_total = self.cfg["initial_earnings"]

        # Ambient and food temperature decay parameters
        self.ambient_temp_c = float(self.cfg.get("ambient_temp_c", 22.0))
        self.k_food_per_s = float(self.cfg.get("k_food_per_s", 1.0/1800.0))

        # ----------------------------------------------------------------------
        # 3. E-scooter Initialization
        # ----------------------------------------------------------------------
        es_cfg = self.cfg["escooter_defaults"]

        if self.e_scooter is None:
            self.e_scooter = EScooter()
            self.e_scooter.set_battery_pct(es_cfg["initial_battery_pct"])
            self.e_scooter.charge_rate_pct_per_min = es_cfg["charge_rate_pct_per_min"]
            self.e_scooter.avg_speed_cm_s = self.cfg["avg_speed_cm_s"]["e-scooter"]

        setattr(self.e_scooter, "owner_id", str(self.agent_id))

        # Compatibility patch: ensure e-scooter has attribute "with_owner"
        if not hasattr(self.e_scooter, "with_owner"):
            setattr(self.e_scooter, "with_owner", True)

        # ----------------------------------------------------------------------
        # 4. Insulated Bag Initialization
        # ----------------------------------------------------------------------
        if self.insulated_bag is None:
            self.insulated_bag = InsulatedBag()
            self.insulated_bag.ambient_temp_c = self.ambient_temp_c

        # ----------------------------------------------------------------------
        # 5. Register Devices to Communication Layer
        # ----------------------------------------------------------------------
        comms = get_comms()
        if comms:
            self.e_scooter = comms.register_scooter(str(self.agent_id), self.e_scooter)

        # Validate transport mode against actual scooter state (battery, park, etc.)
        # so the agent doesn't get e-scooter speed with a dead/parked scooter.
        from ..utils.transport import transport_set_mode
        transport_set_mode(self, self.mode)
        self.speed_cm_s = self.avg_speed_by_mode[self.mode]

        # ----------------------------------------------------------------------
        # 6. Action Handler Routing Table
        # ----------------------------------------------------------------------
        # Maps action kinds to corresponding handler methods
        self._action_handlers = {
            DMActionKind.MOVE:                self._handle_move,
            DMActionKind.MOVE_TO:             self._handle_move_to,
            DMActionKind.PASSBY:              self._handle_passby,
            DMActionKind.ACCEPT_ORDER:        self._handle_accept_order,
            DMActionKind.VIEW_ORDERS:         self._handle_view_orders,
            DMActionKind.VIEW_BAG:            self._handle_view_bag,
            DMActionKind.PICKUP:              self._handle_pickup_food,
            DMActionKind.PLACE_FOOD_IN_BAG:   self._handle_place_food_in_bag,
            DMActionKind.CHARGE_ESCOOTER:     self._handle_charge_escooter,
            DMActionKind.WAIT:                self._handle_wait,
            DMActionKind.REST:                self._handle_rest,
            DMActionKind.BUY:                 self._handle_buy,
            DMActionKind.USE_BATTERY_PACK:    self._handle_use_battery_pack,
            DMActionKind.USE_ENERGY_DRINK:    self._handle_use_energy_drink,
            DMActionKind.USE_ICE_PACK:        self._handle_use_ice_pack,
            DMActionKind.USE_HEAT_PACK:       self._handle_use_heat_pack,
            DMActionKind.VIEW_HELP_BOARD:     self._handle_view_help_board,
            DMActionKind.POST_HELP_REQUEST:   self._handle_post_help_request,
            DMActionKind.ACCEPT_HELP_REQUEST: self._handle_accept_help_request,
            DMActionKind.EDIT_HELP_REQUEST:   self._handle_edit_help_request,
            DMActionKind.SWITCH_TRANSPORT:    self._handle_switch_transport,
            DMActionKind.RENT_CAR:            self._handle_rent_car,
            DMActionKind.RETURN_CAR:          self._handle_return_car,
            DMActionKind.PLACE_TEMP_BOX:      self._handle_place_temp_box,
            DMActionKind.TAKE_FROM_TEMP_BOX:  self._handle_take_from_temp_box,
            DMActionKind.REPORT_HELP_FINISHED:self._handle_report_help_finished,
            DMActionKind.DROP_OFF:            self._handle_drop_off,
            DMActionKind.SAY:                 self._handle_say,
            DMActionKind.BOARD_BUS:           self._handle_board_bus,
            DMActionKind.VIEW_BUS_SCHEDULE:   self._handle_view_bus_schedule,
            DMActionKind.NAVIGATE:            self._handle_navigate,
        }

        # Update towing-related parameters (e.g., dragging scooter)
        self._recalc_towing()

        # Rebuild VLM system prompt with actual config values so transport
        # stats (battery drain, speeds, costs) are always accurate.
        self.vlm_prompt = get_system_prompt(self.cfg)

        # VLM retry limit override
        self._vlm_retry_max = int(self.cfg.get("vlm", {}).get("retry_max", self._vlm_retry_max))

        # ----------------------------------------------------------------------
        # 7. Lifecycle / Recorder Initialization
        # ----------------------------------------------------------------------
        life_cfg = dict(self.cfg.get("lifecycle", {}) or {})
        life_hours = float(life_cfg.get("duration_hours", 0.0))
        life_s = life_hours * 3600.0

        export_path = os.path.join(
            str(life_cfg.get("export_path", ".")),
            f"run_report_agent{self.agent_id}.json"
        )

        # Realtime lifecycle settings
        self._realtime_start_ts = time.time()
        self._realtime_stop_hours = float(life_cfg.get("realtime_stop_hours", 0.0))

        # VLM call count limit
        vlm_call_limit = int(life_cfg.get("vlm_call_limit", 0))

        # Recorder object
        self._recorder = RunRecorder(
            agent_id=str(self.agent_id),
            lifecycle_s=float(life_s if life_s > 0 else 0.0),
            export_path=export_path,
            initial_balance=float(self.earnings_total),
            realtime_stop_hours=self._realtime_stop_hours,
            realtime_start_ts=self._realtime_start_ts,
            vlm_call_limit=vlm_call_limit,
        )

        self._recorder.start(self.clock.now_sim(), self._realtime_start_ts)


    # ======================================================================
    # Basic transport helpers
    # ======================================================================
    def _has_scooter(self) -> bool:
        return self.e_scooter is not None

    def _has_any_scooter(self) -> bool:
        return (self.e_scooter is not None) or (self.assist_scooter is not None)

    def _pace_scale(self) -> float:
        return float(self.pace_scales.get(self.pace_state, 1.0))

    # ======================================================================
    # Wiring: managers, backends, UE, viewer, VLM clients
    # ======================================================================
    # Thin setters used to wire external managers (orders, stores, bus, UE).
    def set_order_manager(self, om: Any):
        self._order_manager = om

    def set_store_manager(self, store_mgr: Any):
        self._store_manager = store_mgr

    def set_bus_manager(self, bus_mgr: Any):
        self._bus_manager = bus_mgr

    def set_ue(self, ue: Any):
        self._ue = ue

    # Register this agent to the global comms hub.
    def register_to_comms(self):
        comms = get_comms()
        if comms:
            comms.register_agent(self)

    # Viewer binding for UI / visualization.
    def bind_viewer(self, viewer: Any):
        viewer_bind_viewer(self, viewer)

    # Spawn the delivery man into the SimWorld / UE backend (if supported).
    def bind_simworld(self):
        if self._ue and hasattr(self._ue, "spawn_delivery_man"):
            self._ue.spawn_delivery_man(self.agent_id, self.x, self.y)

    # Attach a VLM client (e.g., OpenAI / local model) and record the model name.
    def set_vlm_client(self, client: "BaseModel"):
        self._vlm_client = client
        self._recorder.model = str(self._vlm_client.model)

    # Attach an executor for running VLM calls asynchronously (thread pool).
    def set_vlm_executor(self, executor: Executor):
        self._vlm_executor = executor

    # ======================================================================
    # VLM async wrapper (network in thread pool; image capture on main thread)
    # ======================================================================
    def request_vlm_async(self, prompt: str) -> bool:
        return vlm_request_async(self, prompt)

    def pump_vlm_results(self) -> bool:
        return vlm_pump_results(self)

    # ======================================================================
    # Timers: pause / resume (for simulation timeouts or debugging)
    # ======================================================================
    def timers_pause(self):
        agent_timers_pause(self)

    def timers_resume(self):
        agent_timers_resume(self)

    # ======================================================================
    # Logging & VLM memory / error buffers
    # ======================================================================
    def _log(self, text: str):
        dm_log(self, text)

    def vlm_add_memory(self, text: str):
        self.vlm_past_memory.append(str(text))

    def vlm_clear_memory(self):
        self.vlm_past_memory.clear()

    def vlm_add_ephemeral(self, tag: str, text: str):
        self.vlm_ephemeral[str(tag)] = str(text)

    def vlm_add_ephemeral_once(self, tag: str, text: str):
        tag = str(tag)
        self.vlm_ephemeral[tag] = str(text)
        self.vlm_ephemeral_once.add(tag)

    def vlm_add_image(self, img: Any) -> None:
        """Queue a PIL image for the next observation (vision mode only).

        Query-only: this writes to the observation channel, not to any
        simulation state. The env's observation builder consumes and clears
        the queue when assembling ``multi_modal_input``.
        """
        self.vlm_ephemeral_images.append(img)

    def vlm_clear_ephemeral(self):
        self.vlm_ephemeral.clear()
        self.vlm_ephemeral_once.clear()
        self.vlm_ephemeral_images.clear()

    def vlm_add_error(self, msg: str):
        # action_to_text() already ends with a period and the error message
        # often does too; strip them so we don't produce "Move right., ..."
        # or a stray period after the quoted message.
        act = action_to_text(self._current).strip().rstrip(".").strip()
        detail = (msg or "").strip()
        text = f"You just tried {act}, but it failed. Error: {detail}"
        print(f"[Agent {self.agent_id}] {text}")
        self.vlm_errors = text

    def vlm_clear_errors(self):
        self.vlm_errors = None

    def _register_success(self, note: str):
        self.vlm_last_actions.append(note)
        self.vlm_clear_errors()

    # ======================================================================
    # State / speed / transport hooks
    # ======================================================================
    # Whether the agent is currently busy (executing an action or animating in viewer).
    def is_busy(self) -> bool:
        if self._viewer and self._viewer_agent_id and hasattr(self._viewer, "is_busy"):
            return bool(self._viewer.is_busy(self._viewer_agent_id))
        return self._current is not None

    # Recalculate towing state (e.g., dragging scooter while walking).
    def _recalc_towing(self) -> None:
        transport_recalc_towing(self)

    # Speed used by the front-end viewer for visualization.
    def get_current_speed_for_viewer(self) -> float:
        return transport_get_current_speed_for_viewer(self)

    # Switch transport mode (walk, scooter, car, bus, etc.).
    def set_mode(self, mode: TransportMode, *, override_speed_cm_s: Optional[float] = None):
        transport_set_mode(self, mode, override_speed_cm_s=override_speed_cm_s)

    # Called when a move of distance_cm has been executed.
    def on_move_consumed(self, distance_cm: float):
        if (not self.cfg.get("enable_battery", True)
            and not self.cfg.get("enable_walking_energy", True)):
            return
        transport_on_move_consumed(self, distance_cm)

    # Internal energy consumption per distance.
    def _consume_by_distance(self, distance_cm: float):
        transport_consume_by_distance(self, distance_cm)

    # Reset after hospital rescue or other full-recovery events.
    def rescue(self):
        self.energy_pct = float(self.cfg.get("energy_pct_max", 100.0))
        self.is_rescued = False

    # ======================================================================
    # VLM state / text builders (for prompts and hints)
    # ======================================================================
    def _agent_state_text(self) -> str:
        return vlm_agent_state_text(self)

    def _build_bag_place_hint(self) -> str:
        return vlm_build_bag_place_hint(self)

    def _build_pickup_arrival_hint(self, ready_orders, waiting_pairs) -> str:
        return vlm_build_pickup_arrival_hint(self, ready_orders, waiting_pairs)

    def _refresh_pickup_hint_nearby(self) -> None:
        vlm_refresh_pickup_hint_nearby(self)

    def _refresh_poi_hints_nearby(self) -> None:
        vlm_refresh_poi_hints_nearby(self)

    def build_vlm_input(self) -> str:
        return vlm_build_input(self)

    def build_state_observation(self) -> str:
        """Clean state-only observation for RL training. See vlm_build_state_obs."""
        return vlm_build_state_obs(self)
    
    def _map_brief(self) -> str:
        return vlm_map_brief(self)

    # ======================================================================
    # VLM decider (default policy when no scripted controller is used)
    # ======================================================================
    def _default_decider(self) -> Optional[DMAction]:
        return dm_default_decider(self)

    # ======================================================================
    # Loop / scheduling: action queue, starting, finishing
    # ======================================================================
    def kickstart(self):
        dm_kickstart(self)

    def enqueue_action(self, action: DMAction, *, allow_interrupt: bool = False):
        dm_enqueue_action(self, action, allow_interrupt=allow_interrupt)

    def clear_queue(self):
        dm_clear_queue(self)

    def _start_next_if_idle(self):
        dm_start_next_if_idle(self)

    def _start_action(self, act: DMAction, allow_interrupt: bool = True):
        dm_start_action(self, act, allow_interrupt=allow_interrupt)

    def _finish_action(self, *, success: bool):
        dm_finish_action(self, success=success)

    def register_action(
        self,
        kind: DMActionKind,
        handler: Callable[["DeliveryMan", DMAction, bool], None],
    ):
        dm_register_action(self, kind, handler)

    # ======================================================================
    # Action handlers: core delivery actions (say, move, orders, bag, help board)
    # ======================================================================
    def _handle_say(self, act: DMAction, _allow_interrupt: bool):
        action_handle_say(self, act, _allow_interrupt)

    def _handle_move(self, act: DMAction, allow_interrupt: bool):
        action_handle_move(self, act, allow_interrupt)

    def _handle_move_to(self, act: DMAction, allow_interrupt: bool):
        action_handle_move_to(self, act, allow_interrupt)

    def _handle_passby(self, act: DMAction, _allow_interrupt: bool):
        action_handle_passby(self, act, _allow_interrupt)

    def _handle_accept_order(self, act: DMAction, _allow_interrupt: bool):
        action_handle_accept_order(self, act, _allow_interrupt)

    def _handle_view_orders(self, act: DMAction, _allow_interrupt: bool):
        action_handle_view_orders(self, act, _allow_interrupt)

    def _handle_view_help_board(self, act: DMAction, _allow_interrupt: bool):
        action_handle_view_help_board(self, act, _allow_interrupt)

    def _handle_view_bag(self, act: DMAction, _allow_interrupt: bool):
        action_handle_view_bag(self, act, _allow_interrupt)

    def _handle_pickup_food(self, act: DMAction, _allow_interrupt: bool):
        action_handle_pickup_food(self, act, _allow_interrupt)

    def _handle_drop_off(self, act: DMAction, _allow_interrupt: bool):
        action_handle_drop_off(self, act, _allow_interrupt)

    # ======================================================================
    # Charging (E-scooter)
    # ======================================================================
    def _advance_charge_to_now(self) -> None:
        action_advance_charge_to_now(self)

    def _handle_charge_escooter(self, act: DMAction, _allow_interrupt: bool) -> None:
        action_handle_charge_escooter(self, act, _allow_interrupt)

    # ======================================================================
    # WAIT / REST
    # ======================================================================
    def _handle_wait(self, act: DMAction, _allow_interrupt: bool) -> None:
        action_handle_wait(self, act, _allow_interrupt)

    def _handle_rest(self, act: DMAction, _allow_interrupt: bool) -> None:
        action_handle_rest(self, act, _allow_interrupt)

    # ======================================================================
    # Store / consumables (buying and using items)
    # ======================================================================
    def _handle_buy(self, act: DMAction, _allow_interrupt: bool) -> None:
        action_handle_buy(self, act, _allow_interrupt)

    def _handle_use_battery_pack(self, act: DMAction, _allow_interrupt: bool) -> None:
        action_handle_use_battery_pack(self, act, _allow_interrupt)

    def _handle_use_energy_drink(self, act: DMAction, _allow_interrupt: bool) -> None:
        action_handle_use_energy_drink(self, act, _allow_interrupt)

    def _handle_use_ice_pack(self, act: DMAction, _allow_interrupt: bool) -> None:
        action_handle_use_ice_pack(self, act, _allow_interrupt)

    def _handle_use_heat_pack(self, act: DMAction, _allow_interrupt: bool) -> None:
        action_handle_use_heat_pack(self, act, _allow_interrupt)

    # ======================================================================
    # Comms-based help system (requesting / accepting / editing help)
    # ======================================================================
    def _handle_post_help_request(self, act: DMAction, _allow_interrupt: bool) -> None:
        action_handle_post_help_request(self, act, _allow_interrupt)

    def _handle_accept_help_request(self, act: DMAction, _allow_interrupt: bool) -> None:
        action_handle_accept_help_request(self, act, _allow_interrupt)

    def _handle_edit_help_request(self, act: DMAction, _allow_interrupt: bool) -> None:
        action_handle_edit_help_request(self, act, _allow_interrupt)

    # ======================================================================
    # Transport switching / car rental / temp box / collaboration
    # ======================================================================
    def _handle_switch_transport(self, act: DMAction, _allow_interrupt: bool) -> None:
        action_handle_switch_transport(self, act, _allow_interrupt)

    def _handle_rent_car(self, act: DMAction, _allow_interrupt: bool) -> None:
        action_handle_rent_car(self, act, _allow_interrupt)

    def _handle_return_car(self, act: DMAction, _allow_interrupt: bool) -> None:
        action_handle_return_car(self, act, _allow_interrupt)

    def _handle_place_temp_box(self, act: DMAction, _allow_interrupt: bool) -> None:
        action_handle_place_temp_box(self, act, _allow_interrupt)

    def _handle_take_from_temp_box(self, act: DMAction, _allow_interrupt: bool) -> None:
        action_handle_take_from_temp_box(self, act, _allow_interrupt)

    def _handle_place_food_in_bag(self, act: DMAction, _allow_interrupt: bool) -> None:
        action_handle_place_food_in_bag(self, act, _allow_interrupt)

    def _handle_report_help_finished(self, act: DMAction, _allow_interrupt: bool) -> None:
        action_handle_report_help_finished(self, act, _allow_interrupt)

    # ======================================================================
    # Auto drop-off helper (used during movement near drop-off locations)
    # ======================================================================
    def _auto_try_dropoff(self):
        action_auto_try_dropoff(self)

    # ======================================================================
    # Misc: earnings & textual summary
    # ======================================================================
    def add_earnings(self, amount: float):
        self.earnings_total += float(amount)

    def to_text(self) -> str:
        return dm_to_text(self)

    # ======================================================================
    # Progress for UI (charging / resting / hospital rescue)
    # ======================================================================
    def charging_progress(self) -> Optional[Dict[str, Any]]:
        return ui_charging_progress(self)

    def resting_progress(self) -> Optional[Dict[str, Any]]:
        return ui_resting_progress(self)

    def rescue_progress(self) -> Optional[Dict[str, Any]]:
        return ui_rescue_progress(self)

    # ======================================================================
    # Hospital triggers
    # ======================================================================
    def _trigger_hospital_if_needed(self):
        trigger_hospital_if_needed(self)

    # ======================================================================
    # Tick / time events / interruption
    # ======================================================================
    def _interrupt_and_stop(self, reason: str, hint: Optional[str] = None):
        interrupt_and_stop_agent(self, reason, hint)

    def poll_time_events(self):
        timer_poll_time_events(self)

    # ======================================================================
    # Bus-related handlers (boarding, riding updates, schedules)
    # ======================================================================
    def _handle_board_bus(self, act: DMAction, _allow_interrupt: bool) -> None:
        action_handle_board_bus(self, act, _allow_interrupt)

    def _update_bus_riding(self, now: float):
        update_bus_riding(self, now)

    def _handle_view_bus_schedule(self, act: DMAction, _allow_interrupt: bool) -> None:
        action_handle_view_bus_schedule(self, act, _allow_interrupt)

    def _handle_navigate(self, act: DMAction, _allow_interrupt: bool) -> None:
        action_handle_navigate(self, act, _allow_interrupt)
