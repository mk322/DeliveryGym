"""
Text-only (no UE) gym-like environment for DeliveryBench.

Design:
- All actions complete immediately (no background simulation loop).
- Simulation time still advances deterministically via VirtualClock.advance().
- Movement uses routed distance (Map.route_xy_to_xy_mode) as ground truth.
- Map images (global/local) can be produced either:
  - via a lightweight Pillow renderer, or
  - via the original Qt-based MapExportor (for pixel-identical outputs to the UE version).
"""

from __future__ import annotations

import copy
import json
import os
import random
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

import numpy as np


def _load_json(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f) or {}


def _deep_merge_dicts(base: dict, override: dict) -> dict:
    merged = dict(base or {})
    for k, v in (override or {}).items():
        if isinstance(v, dict) and isinstance(merged.get(k), dict):
            merged[k] = _deep_merge_dicts(merged[k], v)
        else:
            merged[k] = v
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
    vlm_delivery_dir: Path
    roads_json: Path
    world_json: Path
    store_items_json: Path
    food_json: Path
    experiment_config_json: Path
    game_mechanics_config_json: Path
    special_notes_json: Path
    models_json: Path


class DeliveryBenchGymEnvText:
    """
    Gym-like API (reset/step/close) without UE/Qt.

    - reset() initializes map, order pool, agent(s), VLM client, and exporter.
    - step(None) runs one VLM decision + executes exactly one parsed action.
    - step(action) executes exactly one provided action (string or DMAction).
    """

    def __init__(
        self,
        *,
        base_dir: str,
        map_name: str = "medium-city-22",
        time_scale: float = 1.0,
        max_steps: int = 2000,
        enable_map_images: bool = True,
        map_renderer: str = "qt",  # "qt" (original look) or "pil" (lightweight)
        enable_vlm: bool = True,
    ):
        self.paths = self._make_paths(Path(base_dir), map_name=map_name)
        self.map_name = str(map_name)
        self.time_scale = float(time_scale)
        self.max_steps = int(max_steps)
        self.enable_map_images = bool(enable_map_images)
        self.map_renderer = str(map_renderer).strip().lower() or "qt"
        # If False, the env will NOT initialize any VLM client during reset().
        # This is useful when an external agent (e.g., AgentGym) drives actions
        # via env.step(action_str) and we never call env.step(None).
        self.enable_vlm = bool(enable_vlm)

        # Runtime handles
        self.cfg: Optional[dict] = None
        self.map = None
        self.nodes = None
        self.clock = None
        self.comms = None
        self.om = None
        self.sm = None
        self.bus_manager = None
        self.dms = []

        self.elapsed_steps = 0
        self._run_dir: Optional[str] = None
        self._qt_app = None

    # ------------------------------------------------------------------
    # Paths / bootstrap
    # ------------------------------------------------------------------
    def _make_paths(self, base_dir: Path, map_name: str) -> EnvPaths:
        vlm_delivery_dir = base_dir / "vlm_delivery"
        return EnvPaths(
            base_dir=base_dir,
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

    # ------------------------------------------------------------------
    # Gym-like API
    # ------------------------------------------------------------------
    def reset(self, seed: Optional[int] = None, options: Optional[dict] = None):
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

        os.environ["DELIVERYBENCH_MULTI_AGENT"] = "1" if bool(cfg.get("multi_agent", False)) else "0"

        from ..map.map import Map
        from ..entities.order import OrderManager
        from ..entities.delivery_man import DeliveryMan, TransportMode
        from ..entities.store import StoreManager
        from ..entities.bus_manager import BusManager
        from ..base.timer import VirtualClock
        from ..gameplay.comms import init_comms, reset_comms
        from ..utils.trajectory_recorder import make_run_folder
        # Map exporter backend is selected later (qt vs pil).

        # Tear down stale comms so scooter/help state doesn't leak across episodes.
        reset_comms()

        # Map/world
        m = Map(cfg.get("map", {}))
        m.import_roads(str(self.paths.roads_json))
        m.import_pois(str(self.paths.world_json))
        nodes = _load_world_nodes(self.paths.world_json)

        # Menu + notes
        food_data = _load_json(self.paths.food_json)
        menu_items = food_data.get("items", [])
        special_notes_data = _load_json(self.paths.special_notes_json)

        # Clock + comms
        clock = VirtualClock(time_scale=self.time_scale)
        comms = init_comms(
            clock=clock,
            ambient_temp_c=cfg.get("ambient_temp_c", 22.0),
            k_food_per_s=cfg.get("k_food_per_s", 1.0 / 1200.0),
        )

        # Orders
        om = OrderManager(
            capacity=cfg.get("order_pool_capacity", 10),
            menu=menu_items,
            clock=clock,
            special_notes_map=special_notes_data,
            note_prob=cfg.get("special_note_prob", 0.4),
        )
        om.fill_pool(m, nodes, _ue=None)

        # Store
        sm = StoreManager()
        sm.load_items(str(self.paths.store_items_json))

        # Bus
        bus_cfg = cfg.get("bus", {}) or {}
        world_data = _load_json(self.paths.world_json)
        bus_manager = BusManager(
            clock=clock,
            waiting_time_s=bus_cfg.get("waiting_time_s", 180.0),
            speed_cm_s=bus_cfg.get("speed_cm_s", 1200.0),
        )
        bus_manager.init_bus_system(world_data)

        # Run dir
        root = (cfg.get("trajectory_output_dir") or "outputs/trajectories") or "outputs/trajectories"
        run_name = datetime.now().strftime("run_%Y%m%d_%H%M%S")
        run_dir = make_run_folder(root, run_name)
        self._run_dir = str(run_dir)

        # Spawn agent at a random road node (fallback to origin)
        initial_x, initial_y = 0.0, 0.0
        try:
            road_nodes = [
                n for n in (getattr(m, "nodes", []) or [])
                if getattr(n, "type", "") in ("normal", "intersection")
            ]
            if road_nodes:
                spawn = random.choice(road_nodes)
                initial_x = float(spawn.position.x)
                initial_y = float(spawn.position.y)
        except Exception:
            pass

        dm = DeliveryMan(
            "1",
            m,
            nodes,
            initial_x,
            initial_y,
            mode=TransportMode.SCOOTER,
            clock=clock,
            cfg=copy.deepcopy(cfg),
        )
        dm.set_order_manager(om)
        dm.set_store_manager(sm)
        dm.set_bus_manager(bus_manager)
        dm.set_ue(None)  # no UE in text-only mode
        dm.register_to_comms()

        dm.run_dir = str(run_dir)

        # Scenario directory (holds roads.json / progen_world_enriched.json).
        # Exposed so query-only runtime actions (e.g. VISUAL_NAVIGATE_WALK) can
        # build map overlays without re-deriving paths. Set unconditionally.
        dm.scenario_dir = str(self.paths.world_json.parent)

        # Map exporter: prefer original Qt renderer for pixel-identical images.
        if self.enable_map_images:
            if self.map_renderer == "qt":
                # Headless Qt is sufficient; no UE required.
                os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
                from PyQt5.QtWidgets import QApplication
                from ..map.map_exportor import MapExportor

                if QApplication.instance() is None:
                    _app = QApplication([])
                    # Keep a ref so it isn't GC'ed.
                    self._qt_app = _app

                ex = MapExportor(
                    map_obj=m,
                    world_json_path=str(self.paths.world_json),
                    show_road_names=True,
                )
                ex.prepare_base()
                dm.map_exportor = ex
            else:
                from ..map.map_exportor_pil import MapExportorPil

                dm.map_exportor = MapExportorPil(
                    map_obj=m,
                    world_json_path=str(self.paths.world_json),
                    show_road_names=True,
                )

        # Manual-step mode: env.step() boundaries are one finished action.
        dm.manual_step = True

        # VLM setup (optional).
        # If you run this env in "external-agent" mode (AgentGym, RL trainers, etc.),
        # you should disable VLM and always call step(action_str).
        if self.enable_vlm:
            from ..vlm.base_model import BaseModel

            agent_cfg = _get_agent_model_config("1", models_config)
            provider = (agent_cfg.get("provider") or "openai").lower()
            api_key = (
                (os.getenv("OPENAI_API_KEY") or "")
                if provider == "openai"
                else (os.getenv("OPENROUTER_API_KEY") or "")
            )
            if not api_key:
                raise RuntimeError(
                    f"Missing API key for provider={provider}. "
                    "Set OPENAI_API_KEY or OPENROUTER_API_KEY."
                )
            llm = BaseModel(
                url=agent_cfg.get("url"),
                api_key=api_key,
                model=agent_cfg.get("model"),
            )
            dm.set_vlm_client(llm)

        # Save handles
        self.map, self.nodes = m, nodes
        self.clock, self.comms = clock, comms
        self.om, self.sm, self.bus_manager = om, sm, bus_manager
        self.dms = [dm]

        obs = self._build_obs()
        info = {
            "sim_time": self.clock.now_sim() if self.clock else None,
            "seed": seed,
            "options": options or {},
            "run_dir": self._run_dir,
        }
        return obs, info

    def step(self, action: Any):
        if not self.dms:
            raise RuntimeError("Env not reset() yet")

        self.elapsed_steps += 1
        dm = self.dms[0]

        # Execute exactly one action
        info_extra: Dict[str, Any] = {}
        try:
            if action is None:
                if dm._vlm_client is None:
                    raise RuntimeError("VLM client not set.")
                prompt = dm.build_vlm_input()
                step_idx = int(getattr(dm, "current_step", 0) or 0)
                images = None
                if self.enable_map_images:
                    try:
                        from ..utils.vlm_runtime import (
                            vlm_collect_images,
                            export_vlm_images_debug_once,
                        )
                        images = vlm_collect_images(dm)
                        # Save global/local images + prompt for inspection (trajectory artifacts).
                        export_vlm_images_debug_once(dm)
                    except Exception:
                        images = None
                raw = dm._vlm_client.generate(user_prompt=prompt, images=images)

                # Save raw model output per step (to match *_prompt.txt and images).
                try:
                    from ..utils.util import sanitize_filename
                    from ..utils.trajectory_recorder import save_text

                    safe_model = sanitize_filename(getattr(dm._vlm_client, "model", "unknown_model"))
                    save_text(dm.run_dir, f"{safe_model}_{step_idx}_output.txt", str(raw), encoding="utf-8")
                except Exception:
                    pass

                act, _plan = self._parse_action(raw, dm)
            else:
                act, _plan = self._coerce_action(action, dm)

            # Run the action synchronously.
            dm._start_action(act)

            # Apply any time-driven updates at the new sim time.
            dm.poll_time_events()

        except Exception as e:
            info_extra["error"] = str(e)

        obs = self._build_obs()
        reward, reward_info = self._compute_reward()
        terminated, term_info = self._is_terminated()
        truncated = self.elapsed_steps >= self.max_steps
        info = {
            "sim_time": self.clock.now_sim() if self.clock else None,
            "elapsed_steps": self.elapsed_steps,
            "reward_info": reward_info,
            "termination_info": term_info,
            "mode": "vlm_decision" if action is None else "manual_action",
        }
        info.update(info_extra)
        return obs, float(reward), bool(terminated), bool(truncated), info

    def close(self):
        self.dms = []

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------
    def _coerce_action(self, action: Any, dm: Any):
        from ..base.defs import DMAction
        from ..gameplay.action_space import parse_action
        if isinstance(action, DMAction):
            return action, None
        if isinstance(action, str):
            return parse_action(action, dm)
        raise TypeError(f"Unsupported action type: {type(action)}")

    def _parse_action(self, model_text: str, dm: Any):
        from ..gameplay.action_space import parse_action
        return parse_action(model_text, dm)

    def _build_obs(self) -> Dict[str, np.ndarray]:
        dm = self.dms[0]
        x = float(getattr(dm, "x", 0.0))
        y = float(getattr(dm, "y", 0.0))
        money = float(getattr(dm, "earnings_total", 0.0))
        batt = 0.0
        try:
            if getattr(dm, "e_scooter", None) is not None:
                batt = float(dm.e_scooter.battery_pct)
        except Exception:
            batt = 0.0

        pending = float(_safe_len(getattr(dm, "active_orders", []) or []))
        state = np.array([x, y, money, batt, pending], dtype=np.float32)
        return {"state": state}

    def _compute_reward(self):
        return 0.0, {"base": 0.0}

    def _is_terminated(self):
        return False, {}

