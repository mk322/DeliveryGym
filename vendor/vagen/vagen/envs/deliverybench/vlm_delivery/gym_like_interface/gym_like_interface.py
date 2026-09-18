# gym_like_interface/gym_like_interface.py
# -*- coding: utf-8 -*-

"""
DeliveryBenchGymEnvQtRouteA — Qt-thread-only mutations + Script-safe driving (Windows/PyQt5)

Key rules enforced:
- Qt GUI thread is the ONLY thread allowed to touch Qt objects / viewer / timers / DM methods that might touch Qt.
- Invoker executes functions on Qt GUI thread using BlockingQueuedConnection (robust on Windows).
- For script: call env.run_qt_loop() on main thread.
- For notebook: do NOT call app.exec_(); use %gui qt (recommended) or env.run_qt_loop_jupyter() in main thread.
"""

from __future__ import annotations

import os
import time
import json
import copy
import random
import threading
import traceback
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

import numpy as np
from concurrent.futures import ThreadPoolExecutor


# -----------------------------------------------------------------------------
# small utils
# -----------------------------------------------------------------------------
def _load_json(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f) or {}


def _deep_merge_dicts(base: dict, override: dict) -> dict:
    merged = dict(base or {})
    for key, value in (override or {}).items():
        if isinstance(value, dict) and isinstance(merged.get(key), dict):
            merged[key] = _deep_merge_dicts(merged[key], value)
        else:
            merged[key] = value
    return merged


def _load_world_nodes(world_json: Path):
    return _load_json(world_json).get("nodes", [])


def _safe_len(x) -> int:
    try:
        return len(x)
    except Exception:
        return 0


def _get_agent_model_config(agent_id: str, models_config: dict) -> dict:
    agents = models_config.get("agents", {}) or {}
    default = models_config.get("default", {}) or {}
    agent_cfg = agents.get(str(agent_id), {}) or {}
    cfg = dict(default)
    cfg.update(agent_cfg)
    return cfg


@dataclass
class EnvPaths:
    base_dir: Path
    simworld_dir: Path
    vlm_delivery_dir: Path
    roads_json: Path
    world_json: Path
    store_items_json: Path
    food_json: Path
    experiment_config_json: Path
    game_mechanics_config_json: Path
    special_notes_json: Path
    models_json: Path


# -----------------------------------------------------------------------------
# Qt main-thread invoker (robust BlockingQueuedConnection)
# -----------------------------------------------------------------------------
class _MainThreadInvoker:
    """
    Execute a Python callable on the Qt GUI thread.

    Implementation:
    - Use a QObject living on GUI thread.
    - Use BlockingQueuedConnection so call() returns only after fn finished.
    """

    def __init__(self, app):
        from PyQt5.QtCore import QObject, pyqtSignal, pyqtSlot, Qt

        class _Invoker(QObject):
            invoke = pyqtSignal(object, object)  # fn, box

            def __init__(self):
                super().__init__()
                # BlockingQueuedConnection: caller thread will block until slot returns
                self.invoke.connect(self._on_invoke, type=Qt.BlockingQueuedConnection)

            @pyqtSlot(object, object)
            def _on_invoke(self, fn, box):
                # We are now running on the receiver's thread (GUI thread)
                try:
                    box["result"] = fn()
                    box["ok"] = True
                except Exception:
                    box["ok"] = False
                    box["exc"] = traceback.format_exc()

        self._app = app
        self._qobj = _Invoker()
        try:
            self._qobj.moveToThread(app.thread())
        except Exception:
            # If moveToThread fails (rare), still usable in many cases but less strict
            pass

    def call(self, fn, timeout_s: float = 5.0) -> dict:
        """
        Run fn on GUI thread and return {'ok': bool, 'result': any, 'exc': str|None}.

        Note:
        - BlockingQueuedConnection itself does not timeout; timeout_s kept for compatibility.
        """
        from PyQt5.QtCore import QThread

        # QApplication is in QtWidgets (not QtCore)
        try:
            from PyQt5.QtWidgets import QApplication
            app = QApplication.instance()
            gui_thread = app.thread() if app is not None else None
        except Exception:
            # Fallback if QtWidgets not available for some reason
            try:
                from PyQt5.QtGui import QGuiApplication
                app = QGuiApplication.instance()
                gui_thread = app.thread() if app is not None else None
            except Exception:
                app = None
                gui_thread = None

        # Fast-path: already on GUI thread
        if gui_thread is not None and QThread.currentThread() == gui_thread:
            box: Dict[str, Any] = {"ok": False, "result": None, "exc": None}
            try:
                box["result"] = fn()
                box["ok"] = True
            except Exception:
                box["ok"] = False
                box["exc"] = traceback.format_exc()
            return box

        # Otherwise: block until done (slot runs on GUI thread)
        box: Dict[str, Any] = {"ok": False, "result": None, "exc": None}
        try:
            self._qobj.invoke.emit(fn, box)
        except Exception:
            return {"ok": False, "result": None, "exc": traceback.format_exc()}
        return box


# -----------------------------------------------------------------------------
# Env
# -----------------------------------------------------------------------------
class DeliveryBenchGymEnvQtRouteA:
    def __init__(
        self,
        base_dir: str,
        map_name: str = "medium-city-22roads",
        ue_ip: str = "127.0.0.1",
        ue_port: int = 9000,
        resolution: Tuple[int, int] = (640, 480),
        time_scale: float = 3.0,
        max_steps: int = 2000,
        ready_timeout_s: float = 15.0,
        sim_tick_ms: int = 100,
        vlm_pump_ms: int = 200,
        enable_viewer: bool = True,
        executor_workers: int = 6,
        idle_timeout_s: float = 10.0,
        vlm_timeout_s: float = 60.0,
        action_timeout_s: float = 120.0,
        dispatch_timeout_s: float = 10.0,
    ):
        self.paths = self._make_paths(Path(base_dir), map_name=map_name)

        self.ue_ip = ue_ip
        self.ue_port = int(ue_port)
        self.resolution = resolution
        self.time_scale = float(time_scale)
        self.max_steps = int(max_steps)
        self.ready_timeout_s = float(ready_timeout_s)
        self.sim_tick_ms = int(sim_tick_ms)
        self.vlm_pump_ms = int(vlm_pump_ms)
        self.enable_viewer = bool(enable_viewer)
        self.executor_workers = int(executor_workers)

        self.idle_timeout_s = float(idle_timeout_s)
        self.vlm_timeout_s = float(vlm_timeout_s)
        self.action_timeout_s = float(action_timeout_s)
        self.dispatch_timeout_s = float(dispatch_timeout_s)

        # runtime handles
        self.cfg: Optional[dict] = None
        self.map = None
        self.nodes = None
        self.clock = None
        self.comms = None
        self.om = None
        self.sm = None
        self.bus_manager = None
        self.ue = None
        self.dms = []
        self.map_exportor = None

        self.elapsed_steps = 0

        # Qt handles
        self._app = None
        self._viewer = None
        self._sim_timer = None
        self._vlm_timer = None

        # sync
        self._cv = threading.Condition()

        # executor
        self._executor: Optional[ThreadPoolExecutor] = None

        # bookkeeping
        self._run_dir: Optional[str] = None

        # invoker
        self._invoker: Optional[_MainThreadInvoker] = None
        self._qt_bootstrapped = False

        # jupyter loop control
        self._jupyter_loop_running = False

    # -------------------------------------------------------------------------
    # Qt bootstrap (must be called on Python main thread)
    # -------------------------------------------------------------------------
    def bootstrap_qt(self):
        if threading.current_thread() is not threading.main_thread():
            raise RuntimeError("bootstrap_qt() must be called on Python MAIN thread.")
        self._ensure_qt_app()
        if self._invoker is None:
            self._invoker = _MainThreadInvoker(self._app)
        self._qt_bootstrapped = True

    # -------------------------------------------------------------------------
    # Gym-like API
    # -------------------------------------------------------------------------
    def reset(self, seed: Optional[int] = None, options: Optional[dict] = None):
        if not self._qt_bootstrapped or self._invoker is None:
            raise RuntimeError("Call env.bootstrap_qt() on Python main thread before env.reset().")

        self.elapsed_steps = 0
        if seed is not None:
            random.seed(seed)
            np.random.seed(seed)

        os.chdir(str(self.paths.base_dir))

        game_cfg = _load_json(self.paths.game_mechanics_config_json)
        experiment_cfg = _load_json(self.paths.experiment_config_json)
        cfg = _deep_merge_dicts(game_cfg, experiment_cfg)
        models_config = _load_json(self.paths.models_json)
        self.cfg = cfg

        os.environ["DELIVERYBENCH_MULTI_AGENT"] = "0"

        (
            Map,
            OrderManager,
            DeliveryMan,
            TransportMode,
            StoreManager,
            BusManager,
            VirtualClock,
            init_comms,
            reset_comms,
            Communicator,
            MapObserver,
            MapExportor,
            make_run_folder,
            BaseModel,
        ) = self._lazy_imports()

        # Tear down stale comms so scooter/help state doesn't leak across episodes.
        reset_comms()

        # map/world
        m = Map(cfg.get("map", {}))
        m.import_roads(str(self.paths.roads_json))
        m.import_pois(str(self.paths.world_json))
        nodes = _load_world_nodes(self.paths.world_json)

        # menu + notes
        food_data = _load_json(self.paths.food_json)
        menu_items = food_data.get("items", [])
        special_notes_data = _load_json(self.paths.special_notes_json)

        # clock + comms
        clock = VirtualClock(time_scale=self.time_scale)
        comms = init_comms(
            clock=clock,
            ambient_temp_c=cfg.get("ambient_temp_c", 22.0),
            k_food_per_s=cfg.get("k_food_per_s", 1.0 / 1200.0),
        )

        # orders
        om = OrderManager(
            capacity=cfg.get("order_pool_capacity", 10),
            menu=menu_items,
            clock=clock,
            special_notes_map=special_notes_data,
            note_prob=cfg.get("special_note_prob", 0.4),
        )
        om.fill_pool(m, nodes)

        # store
        sm = StoreManager()
        sm.load_items(str(self.paths.store_items_json))

        # bus
        bus_cfg = cfg.get("bus", {}) or {}
        world_data = _load_json(self.paths.world_json)
        bus_manager = BusManager(
            clock=clock,
            waiting_time_s=bus_cfg.get("waiting_time_s", 180.0),
            speed_cm_s=bus_cfg.get("speed_cm_s", 1200.0),
        )
        bus_manager.init_bus_system(world_data)

        # UE
        ue = Communicator(
            port=self.ue_port,
            ip=self.ue_ip,
            resolution=self.resolution,
            cfg=cfg,
        )

        # viewer must be created on Qt thread
        if self.enable_viewer:
            box = self._invoker.call(lambda: self._ensure_viewer(MapObserver, clock, m, om, comms, bus_manager))
            if not box.get("ok", False):
                raise RuntimeError(f"ensure_viewer failed:\n{box.get('exc')}")

        # exportor: if MapExportor touches Qt (QImage/QPainter), keep it on Qt thread to avoid thread-parent errors
        def _create_exportor():
            ex = MapExportor(
                map_obj=m,
                world_json_path=str(self.paths.world_json),
                show_road_names=True,
            )
            ex.prepare_base()
            return ex

        box = self._invoker.call(_create_exportor)
        if not box.get("ok", False):
            raise RuntimeError("create MapExportor failed:\n" + str(box.get("exc")))
        map_exportor = box["result"]

        # executor
        if self._executor is None:
            self._executor = ThreadPoolExecutor(max_workers=self.executor_workers)

        # run_dir
        root = (cfg.get("trajectory_output_dir") or "outputs/trajectories") or "outputs/trajectories"
        run_name = datetime.now().strftime("run_%Y%m%d_%H%M%S")
        run_dir = make_run_folder(root, run_name)
        self._run_dir = str(run_dir)

        # spawn agent
        dm = DeliveryMan("1", m, nodes, 0.0, 0.0, mode=TransportMode.SCOOTER, clock=clock, cfg=copy.deepcopy(cfg))

        # bind_viewer must be on Qt thread
        if self._viewer is not None:
            box = self._invoker.call(lambda: dm.bind_viewer(self._viewer))
            if not box.get("ok", False):
                raise RuntimeError(f"bind_viewer failed:\n{box.get('exc')}")

        dm.set_order_manager(om)
        dm.set_store_manager(sm)
        dm.set_bus_manager(bus_manager)
        dm.set_ue(ue)
        dm.bind_simworld()
        dm.register_to_comms()

        dm.run_dir = run_dir
        dm.map_exportor = map_exportor

        dm.manual_step = True
        dm._step_cv = self._cv

        # VLM setup
        agent_cfg = _get_agent_model_config("1", models_config)
        provider = (agent_cfg.get("provider") or "openai").lower()

        openai_key = os.getenv("OPENAI_KEY") or os.getenv("OPENAI_API_KEY") or ""
        openrouter_key = os.getenv("OPENROUTER_KEY") or os.getenv("OPENROUTER_API_KEY") or ""
        api_key = openai_key if provider == "openai" else openrouter_key
        if not api_key:
            raise RuntimeError(f"Missing API key for provider={provider}. Set OPENAI_API_KEY or OPENROUTER_API_KEY.")

        llm = BaseModel(url=agent_cfg.get("url"), api_key=api_key, model=agent_cfg.get("model"))
        dm.set_vlm_client(llm)
        dm.set_vlm_executor(self._executor)

        self._wait_until_ready(ue, ["1"], timeout_s=self.ready_timeout_s)

        # save handles
        self.map, self.nodes = m, nodes
        self.clock, self.comms = clock, comms
        self.om, self.sm, self.bus_manager = om, sm, bus_manager
        self.ue = ue
        self.dms = [dm]
        self.map_exportor = map_exportor

        # timers must be created/started on Qt thread
        box = self._invoker.call(self._start_sim_timer)
        if not box.get("ok", False):
            raise RuntimeError(f"start_sim_timer failed:\n{box.get('exc')}")
        box = self._invoker.call(self._start_vlm_pump_timer)
        if not box.get("ok", False):
            raise RuntimeError(f"start_vlm_pump_timer failed:\n{box.get('exc')}")

        obs = self._build_obs()
        info = {
            "sim_time": self._get_sim_time(),
            "seed": seed,
            "options": options or {},
            "sim_tick_ms": self.sim_tick_ms,
            "vlm_pump_ms": self.vlm_pump_ms,
            "run_dir": self._run_dir,
        }
        return obs, info

    def step(self, action: Any):
        """
        Route A:
        - action is not None: execute exactly ONE DMAction (manual_step boundary)
        - action is None: dispatch ONE VLM request on Qt thread -> wait until exactly ONE action finishes

        Robustness:
        - Use dm._manual_step_done_seq as the step boundary signal (never misses very fast actions).
        """
        if not self.dms:
            raise RuntimeError("Env not reset() yet")
        if self._invoker is None:
            raise RuntimeError("Env not bootstrapped. Call bootstrap_qt() first.")

        self.elapsed_steps += 1
        dm = self.dms[0]

        def _done_seq() -> int:
            try:
                return int(getattr(dm, "_manual_step_done_seq", 0) or 0)
            except Exception:
                return 0

        # wait idle
        with self._cv:
            ok_idle = self._cv.wait_for(
                lambda: (getattr(dm, "_current", None) is None and (not getattr(dm, "_queue", None))),
                timeout=self.idle_timeout_s,
            )
            if not ok_idle:
                return self._build_obs(), 0.0, False, True, {"error": "timeout_waiting_idle_before_step"}

        done0 = _done_seq()

        # direct action
        if action is not None:
            box = self._invoker.call(lambda: dm.enqueue_action(action))
            if not box.get("ok", False):
                return self._build_obs(), 0.0, False, True, {"error": "enqueue_failed", "enqueue_exc": box.get("exc")}

            with self._cv:
                ok_done = self._cv.wait_for(lambda: _done_seq() > done0, timeout=self.action_timeout_s)
                if not ok_done:
                    return self._build_obs(), 0.0, False, True, {
                        "error": "timeout_waiting_action_done",
                        "done0": done0,
                        "done_now": _done_seq(),
                        "current": repr(getattr(dm, "_current", None)),
                        "queue_len": len(getattr(dm, "_queue", []) or []),
                    }

            return self._finalize_step({"mode": "direct_action", "done0": done0, "done1": _done_seq()})

        # VLM dispatch
        t0 = time.time()
        dispatch_box = self._invoker.call(lambda: self._dispatch_vlm_once(dm))
        print("invoker elapsed:", time.time() - t0, "ok=", dispatch_box.get("ok"), "exc=", dispatch_box.get("exc"))

        with self._cv:
            self._cv.notify_all()

        if not dispatch_box.get("ok", False):
            return self._build_obs(), 0.0, False, True, {"error": "dispatch_failed_on_qt_thread", "dispatch_exc": dispatch_box.get("exc")}

        # best-effort wait VLM in-flight marked
        with self._cv:
            ok_sent = self._cv.wait_for(
                lambda: bool(getattr(dm, "_waiting_vlm", False)) and bool(getattr(dm, "_vlm_future", None)),
                timeout=3.0,
            )
            if not ok_sent:
                return self._build_obs(), 0.0, False, True, {
                    "error": "vlm_not_dispatched",
                    "waiting_vlm": getattr(dm, "_waiting_vlm", None),
                    "has_future": bool(getattr(dm, "_vlm_future", None)),
                    "done0": done0,
                    "done_now": _done_seq(),
                }

        # wait one action finishes
        with self._cv:
            ok_done = self._cv.wait_for(lambda: _done_seq() > done0, timeout=self.action_timeout_s)
            if not ok_done:
                return self._build_obs(), 0.0, False, True, {
                    "error": "timeout_waiting_action_done_after_vlm",
                    "done0": done0,
                    "done_now": _done_seq(),
                    "waiting_vlm": getattr(dm, "_waiting_vlm", None),
                    "has_future": bool(getattr(dm, "_vlm_future", None)),
                    "current": repr(getattr(dm, "_current", None)),
                    "queue_len": len(getattr(dm, "_queue", []) or []),
                }

        return self._finalize_step({"mode": "vlm_decision", "done0": done0, "done1": _done_seq()})

    # -------------------------------------------------------------------------
    # Jupyter-safe event pumping (NO app.exec_)
    # -------------------------------------------------------------------------
    def run_qt_loop_jupyter(self, poll_ms: int = 10):
        """
        For Jupyter ONLY (if you do NOT use %gui qt):
        - Run this in the MAIN thread cell.
        - Start RL in a background thread.
        """
        if threading.current_thread() is not threading.main_thread():
            raise RuntimeError("run_qt_loop_jupyter() must run on Python MAIN thread.")
        if self._app is None:
            self._ensure_qt_app()

        self._jupyter_loop_running = True
        while self._jupyter_loop_running:
            try:
                self._app.processEvents()
            except Exception:
                pass
            time.sleep(max(poll_ms, 1) / 1000.0)

    def stop_qt_loop_jupyter(self):
        self._jupyter_loop_running = False

    def close(self):
        try:
            if self._invoker is not None:
                self._invoker.call(self._stop_sim_timer)
                self._invoker.call(self._stop_vlm_timer)
            else:
                self._stop_sim_timer()
                self._stop_vlm_timer()
        except Exception:
            pass

        self.dms = []
        self.ue = None

    def run_qt_loop(self):
        """For normal script execution (NOT Jupyter)."""
        if threading.current_thread() is not threading.main_thread():
            raise RuntimeError("run_qt_loop() must be called on Python MAIN thread.")
        if self._app is None:
            self._ensure_qt_app()
        self._app.exec_()

    # -------------------------------------------------------------------------
    # VLM dispatch (Qt thread)
    # -------------------------------------------------------------------------
    def _dispatch_vlm_once(self, dm) -> bool:
        if not hasattr(dm, "_waiting_vlm"):
            dm._waiting_vlm = False

        if getattr(dm, "_waiting_vlm", False) or getattr(dm, "_vlm_future", None) is not None:
            return True

        try:
            dm._viewer_agent_id = int(getattr(dm, "agent_id", 1))
        except Exception:
            dm._viewer_agent_id = 1

        if hasattr(dm, "build_vlm_input") and callable(getattr(dm, "build_vlm_input")):
            prompt = dm.build_vlm_input()
        elif hasattr(dm, "build_user_prompt") and callable(getattr(dm, "build_user_prompt")):
            prompt = dm.build_user_prompt()
        else:
            prompt = "Decide the next action. Output one valid action."

        from ..utils.vlm_runtime import vlm_request_async
        ok = vlm_request_async(dm, prompt)

        with self._cv:
            self._cv.notify_all()

        return bool(ok)

    # -------------------------------------------------------------------------
    # Qt timers (Qt thread)
    # -------------------------------------------------------------------------
    def _ensure_qt_app(self):
        if self._app is not None:
            return
        if threading.current_thread() is not threading.main_thread():
            raise RuntimeError("QApplication must be created on Python MAIN thread.")
        from PyQt5.QtWidgets import QApplication
        inst = QApplication.instance()
        self._app = inst if inst is not None else QApplication([])

    def _start_sim_timer(self):
        if self._sim_timer is not None:
            return
        from PyQt5.QtCore import QTimer
        parent = self._viewer if self._viewer is not None else None
        self._sim_timer = QTimer(parent)
        self._sim_timer.setInterval(self.sim_tick_ms)

        def tick_sim():
            for dm in self.dms:
                dm.poll_time_events()
            with self._cv:
                self._cv.notify_all()

        self._sim_timer.timeout.connect(tick_sim)
        self._sim_timer.start()

    def _stop_sim_timer(self):
        try:
            if self._sim_timer is not None:
                self._sim_timer.stop()
        except Exception:
            pass
        self._sim_timer = None

    def _start_vlm_pump_timer(self):
        if self._vlm_timer is not None:
            return
        from PyQt5.QtCore import QTimer
        parent = self._viewer if self._viewer is not None else None
        self._vlm_timer = QTimer(parent)
        self._vlm_timer.setInterval(self.vlm_pump_ms)

        def tick_vlm():
            for dm in self.dms:
                try:
                    dm.pump_vlm_results()
                except Exception as e:
                    try:
                        dm._log(f"[GymEnv] pump_vlm_results exception: {e}")
                    except Exception:
                        pass
            with self._cv:
                self._cv.notify_all()

        self._vlm_timer.timeout.connect(tick_vlm)
        self._vlm_timer.start()

    def _stop_vlm_timer(self):
        try:
            if self._vlm_timer is not None:
                self._vlm_timer.stop()
        except Exception:
            pass
        self._vlm_timer = None

    # -------------------------------------------------------------------------
    # Helpers
    # -------------------------------------------------------------------------
    def _finalize_step(self, info_extra: Optional[dict] = None):
        obs = self._build_obs()
        reward, reward_info = self._compute_reward()
        terminated, term_info = self._is_terminated()
        truncated = self.elapsed_steps >= self.max_steps

        info = {
            "sim_time": self._get_sim_time(),
            "elapsed_steps": self.elapsed_steps,
            "reward_info": reward_info,
            "termination_info": term_info,
        }
        if info_extra:
            info.update(info_extra)
        return obs, float(reward), bool(terminated), bool(truncated), info

    def _make_paths(self, base_dir: Path, map_name: str) -> EnvPaths:
        simworld_dir = base_dir / "SimWorld"
        vlm_delivery_dir = base_dir / "vlm_delivery"
        return EnvPaths(
            base_dir=base_dir,
            simworld_dir=simworld_dir,
            vlm_delivery_dir=vlm_delivery_dir,
            roads_json=base_dir / "maps" / map_name / "roads.json",
            world_json=base_dir / "maps" / map_name / "progen_world_enriched.json",
            store_items_json=vlm_delivery_dir / "input" / "store_items.json",
            food_json=vlm_delivery_dir / "input" / "food.json",
            experiment_config_json=vlm_delivery_dir / "input" / "experiment_config.json",
            game_mechanics_config_json=vlm_delivery_dir / "input" / "game_mechanics_config.json",
            special_notes_json=vlm_delivery_dir / "input" / "special_notes.json",
            models_json=vlm_delivery_dir / "input" / "models.json",
        )

    def _lazy_imports(self):
        from ..map.map import Map
        from ..entities.order import OrderManager
        from ..entities.delivery_man import DeliveryMan, TransportMode
        from ..entities.store import StoreManager
        from ..entities.bus_manager import BusManager
        from ..base.timer import VirtualClock
        from ..gameplay.comms import init_comms, reset_comms
        from ..communicator.communicator import Communicator
        from ..map.map_observer import MapObserver
        from ..map.map_exportor import MapExportor
        from ..utils.trajectory_recorder import make_run_folder
        from ..vlm.base_model import BaseModel
        return (
            Map, OrderManager, DeliveryMan, TransportMode, StoreManager, BusManager,
            VirtualClock, init_comms, reset_comms, Communicator, MapObserver, MapExportor, make_run_folder, BaseModel
        )

    def _wait_until_ready(self, ue, agent_ids, timeout_s: float):
        t0 = time.time()
        remaining = set(agent_ids)
        while remaining:
            for aid in list(remaining):
                rec = ue.get_position_and_direction(str(aid))
                tup = rec.get(str(aid)) if rec else None
                if tup:
                    remaining.remove(aid)
            if not remaining:
                break
            if time.time() - t0 > timeout_s:
                raise RuntimeError(f"UE actor(s) not ready within timeout: {sorted(list(remaining))}")
            time.sleep(0.1)

    def _get_sim_time(self):
        if self.clock is None:
            return None
        for attr in ["t", "time", "now", "ts", "ts_sim"]:
            if hasattr(self.clock, attr):
                return getattr(self.clock, attr)
        return None

    def _build_obs(self) -> Dict[str, np.ndarray]:
        dm = self.dms[0]
        x = float(getattr(dm, "x", 0.0))
        y = float(getattr(dm, "y", 0.0))
        money = float(getattr(dm, "money", getattr(dm, "cash", 0.0)))
        battery = float(getattr(dm, "battery_pct", getattr(dm, "battery", 0.0)))

        pending = 0.0
        if self.om is not None:
            if hasattr(self.om, "active_orders"):
                pending = float(_safe_len(getattr(self.om, "active_orders")))
            elif hasattr(self.om, "orders"):
                pending = float(_safe_len(getattr(self.om, "orders")))
        state = np.array([x, y, money, battery, pending], dtype=np.float32)
        return {"state": state}

    def _compute_reward(self):
        return 0.0, {"base": 0.0}

    def _is_terminated(self):
        return False, {}

    def _ensure_viewer(self, MapObserver, clock, m, om, comms, bus_manager):
        if self._viewer is not None:
            return
        v = MapObserver(title="GymEnvQt RouteA Viewer", clock=clock)
        v.draw_map(
            m,
            str(self.paths.world_json),
            show_bus=True,
            show_docks=False,
            show_building_links=True,
            show_road_names=True,
            plain_mode="pudo",
        )
        v.resize(1200, 900)
        v.show()
        v.attach_order_manager(om)
        v.attach_comms(comms)
        v.attach_bus_manager(bus_manager)
        self._viewer = v