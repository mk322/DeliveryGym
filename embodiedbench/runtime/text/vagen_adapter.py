"""Adapt the vendored DeliveryBench text environment to the runtime contract.

the design plan M1: "Wrap the existing VAGEN text env without changing its behavior",
and accept only when "a fixed existing DeliveryBench action trace runs through
the adapter three times and matches the pinned pre-adapter state/event hash".

So this is a translation layer and nothing more. It does not re-implement a
transition, re-order an event, or reshape a reward. The one thing it adds is
structure: typed actions in, a typed ``StepResult`` out, with terminated and
truncated separated — the vendored env returns a single ``done`` flag, and
design plan §7 requires the distinction. That separation is derived from the
environment's own step budget and success check, not invented.

Determinism patches (ADR-0003) are applied centrally here, so no consumer of
the runtime can forget them and silently get address-dependent routes.
"""

from __future__ import annotations

import asyncio
import dataclasses
import json
from typing import Any

from embodiedbench.artifacts.state_digest import digest_of, extract_state, state_digest
from embodiedbench.baseline.compat import apply_map_compatibility_patches
from embodiedbench.baseline.determinism import apply_deterministic_patches
from embodiedbench.baseline.replay import STATE_POLICY, load_vendor_env_module
from embodiedbench.runtime.core import EpisodeNotStarted, RuntimeError_, StepIndexMismatch
from embodiedbench.schemas.environment import NavigationMode
from embodiedbench.schemas.episode import EpisodeSpec
from embodiedbench.schemas.geometry import FrameName, Pose, Vec3
from embodiedbench.schemas.runtime import (
    ActionEnvelope,
    ActionResult,
    ActionStatus,
    Event,
    MarkCandidate,
    Observation,
    ResetInfo,
    RuntimeCapabilities,
    RuntimeMode,
    StepResult,
    TerminationReason,
)

# The vendored engine's action grammar mixes positional and keyword forms:
# ACCEPT_ORDER(0) is positional, PICKUP(orders=[0]) and MOVE(direction="left")
# are keyword, VIEW_ORDERS() takes nothing. POSITIONAL_KEY carries the
# positional list so a TaskAction can express either without a second field.
POSITIONAL_KEY = "_args"
_QUOTED_ARGS = {"direction", "target", "mode", "address"}


def _render_value(value: Any) -> str:
    if isinstance(value, bool):
        return str(value).lower()
    if isinstance(value, (list, tuple)):
        return f"[{', '.join(_render_value(v) for v in value)}]"
    return str(value)


def render_vendor_action(name: str, arguments: dict[str, Any]) -> str:
    """Render a task action into the vendored engine's action string.

    Keyword arguments are emitted in sorted order so the same ``ActionEnvelope``
    always produces byte-identical vendored input, which is what makes the
    adapter's replay determinism testable.
    """
    parts: list[str] = []
    for value in arguments.get(POSITIONAL_KEY, []) or []:
        parts.append(_render_value(value))
    for key in sorted(k for k in arguments if k != POSITIONAL_KEY):
        value = arguments[key]
        if isinstance(value, str) and key in _QUOTED_ARGS:
            parts.append(f'{key}="{value}"')
        else:
            parts.append(f"{key}={_render_value(value)}")
    return f"{name}({', '.join(parts)})"


class VagenTextRuntime:
    """The text runtime (design plan §7.1) backed by the vendored engine."""

    def __init__(
        self,
        *,
        map_name: str,
        preset: str = "nav",
        max_steps: int = 400,
        config_overrides: dict[str, Any] | None = None,
        repair_graph: bool = True,
    ):
        apply_deterministic_patches()
        # Without this the runtime cannot load Paris at all: the vendored bus
        # manager indexes route[0] unconditionally and Paris ships an empty
        # bus_routes list (PARIS-F1). Every compiler entry point already applied
        # it, so the gap only showed up in the runtime -- the one component that
        # actually has to run for a policy to be trained.
        apply_map_compatibility_patches()
        self._module = load_vendor_env_module()
        self._map_name = map_name
        self._preset = preset
        self._max_steps = max_steps
        self._config_overrides = dict(config_overrides or {})
        self._repair_graph = repair_graph
        self.graph_repair_report: dict[str, Any] | None = None
        self._env: Any = None
        self._episode_id: str | None = None
        self._step_index = 0
        self._event_seq = 0
        self._sim_time_s = 0.0
        self._closed = False

        self.capabilities = RuntimeCapabilities(
            mode=RuntimeMode.TEXT,
            observation_channels=["text"],
            navigation_modes=[NavigationMode.NAV_WAYPOINT],
            # the design plan M6 requires text and cached runtimes to advertise
            # snapshot support; the text engine is pure Python, so a deep state
            # copy is a faithful snapshot.
            supports_snapshot=True,
        )

    # ── lifecycle ────────────────────────────────────────────────────────────

    def _build_env(self) -> Any:
        config = dataclasses.asdict(self._module.PRESETS[self._preset])
        config.update(
            map_name=self._map_name, render_mode="text", max_steps=self._max_steps
        )
        config.update(self._config_overrides)
        return self._module.DeliveryBench(config)

    def reset(self, instance: EpisodeSpec) -> tuple[Observation, ResetInfo]:
        if self._closed:
            raise RuntimeError_("runtime is closed")
        if self._env is not None:
            asyncio.run(self._env.close())
        self._env = self._build_env()
        self._episode_id = instance.instance_id
        self._step_index = 0
        self._event_seq = 0
        self._sim_time_s = 0.0

        raw_obs, _raw_info = asyncio.run(self._env.reset(seed=instance.seed))
        if self._repair_graph:
            # Node the graph before the first observation, so the candidate
            # waypoints the agent is offered match the graph it will traverse.
            from embodiedbench.compiler.graph_repair import repair_graph as _node_graph

            agent = self._dm()
            if agent is not None:
                self.graph_repair_report = _node_graph(agent.city_map).to_dict()
        observation = self._observation(raw_obs, instance, action_result=None)
        info = ResetInfo(
            episode_id=instance.instance_id,
            seed=instance.seed,
            environment_id=instance.environment_id,
            environment_version=instance.environment_version,
            runtime_mode=RuntimeMode.TEXT,
            spawn_pose=self._agent_pose(),
            privileged_state_ref=f"private://{instance.instance_id}/reset",
        )
        return observation, info

    def close(self) -> None:
        if self._env is not None:
            asyncio.run(self._env.close())
            self._env = None
        self._closed = True

    # ── stepping ─────────────────────────────────────────────────────────────

    def step(self, action: ActionEnvelope) -> StepResult:
        if self._env is None:
            raise EpisodeNotStarted("call reset() before step()")
        if action.step_index != self._step_index:
            raise StepIndexMismatch(
                f"expected step {self._step_index}, got {action.step_index}"
            )
        self.capabilities.require_action(action)

        vendor_action = self._to_vendor_action(action)
        raw_obs, reward, done, info = asyncio.run(
            self._env.step(json.dumps({"action": vendor_action}))
        )
        info = info or {}

        error = self._action_error(info)
        if error:
            action_result = ActionResult(
                status=ActionStatus.REJECTED,
                error_code="vendor_action_error",
                message=str(error)[:2000],
            )
        else:
            action_result = ActionResult(
                status=ActionStatus.ACCEPTED,
                resolved_target_node=self._resolved_target(action),
            )

        events = self._events_for_step(info, error)
        self._sim_time_s = self._current_sim_time()

        terminated, truncated, reason = self._termination(done, info)
        instance_stub = None
        observation = self._observation(raw_obs, instance_stub, action_result=action_result)

        result = StepResult(
            observation=observation,
            reward=float(reward),
            # The vendored engine reports one scalar. Claiming a decomposition
            # it did not produce would be a fabricated reward_components block,
            # so the single component is named for what it is.
            reward_components={"vendor_total": float(reward)},
            terminated=terminated,
            truncated=truncated,
            termination_reason=reason,
            events=events,
            action_result=action_result,
            metrics_delta={"sim_time_s": self._sim_time_s},
            privileged_state_ref=f"private://{self._episode_id}/step_{self._step_index}",
        )
        self._step_index += 1
        return result

    # ── translation ──────────────────────────────────────────────────────────

    def _to_vendor_action(self, envelope: ActionEnvelope) -> str:
        action = envelope.action
        if action.type == "task":
            return render_vendor_action(action.name, action.arguments)
        if action.type == "nav_waypoint":
            node = action.target_node
            if node is None:
                node = self._node_for_mark(action.target_mark)
            if node is None:
                raise RuntimeError_(f"could not resolve mark {action.target_mark} to a node")
            return render_vendor_action("MOVE_TO", {"target": node})
        # Unreachable in practice: capabilities.require_action screens point
        # modes out first, because this runtime advertises nav_waypoint only.
        # The reason is worth stating where someone hits it rather than leaving
        # a bare type name -- the vendored engine's action space is
        # node-to-node, so there is no primitive that can end at the arbitrary
        # lattice pose a point action resolves to (design plan §9.5).
        raise RuntimeError_(
            f"the text runtime cannot execute {action.type!r}: its action space is "
            "graph-neighbour moves, which cannot terminate at an off-node pose. "
            "Point modes need the live UE runtime or a pose-lattice album."
        )

    def _resolved_target(self, envelope: ActionEnvelope) -> str | None:
        action = envelope.action
        if action.type == "nav_waypoint":
            return action.target_node or self._node_for_mark(action.target_mark)
        return None

    def _node_for_mark(self, mark: int | None) -> str | None:
        if mark is None:
            return None
        candidates = self._mark_candidates()
        for candidate in candidates:
            if candidate.mark == mark:
                return candidate.node_id
        return None

    def _mark_candidates(self) -> list[MarkCandidate]:
        """Reachable neighbours as numbered Set-of-Marks candidates (design plan §9.1)."""
        dm = self._dm()
        if dm is None:
            return []
        try:
            from vagen.envs.deliverybench.vlm_delivery.actions.move import available_moves

            moves = available_moves(dm)
        except Exception:  # noqa: BLE001 - candidate listing must never break a step
            return []
        out: list[MarkCandidate] = []
        for index, direction in enumerate(sorted(moves)):
            candidate = moves.get(direction)
            if not candidate:
                continue
            node = candidate.get("node")
            node_id = getattr(node, "waypoint_id", None)
            if node_id is None:
                continue
            position = getattr(node, "position", None)
            distance_m = 0.0
            if position is not None:
                distance_m = (
                    ((float(position.x) - float(dm.x)) ** 2 + (float(position.y) - float(dm.y)) ** 2)
                    ** 0.5
                ) / 100.0
            out.append(
                MarkCandidate(
                    mark=len(out),
                    node_id=str(node_id),
                    bearing_deg=float(getattr(dm, "facing_deg", 0.0) or 0.0),
                    distance_m=distance_m,
                )
            )
        return out

    # ── state access ─────────────────────────────────────────────────────────

    def _dm(self) -> Any:
        if self._env is None or self._env._env is None:
            return None
        dms = self._env._env.dms
        return dms[0] if dms else None

    def _agent_pose(self) -> Pose | None:
        dm = self._dm()
        if dm is None:
            return None
        # The engine's facing_deg is a *compass* bearing: map.py's _bearing_deg
        # is atan2(dx, dy), so 0 means north (+Y) and it increases clockwise.
        # Pose.yaw_deg is the mathematical convention used by the camera
        # intrinsics, the point-navigation chain and the UE renderer alike --
        # atan2(dy, dx), 0 means +X, increasing anticlockwise. Copying one into
        # the other without converting rotates every pose by 90 degrees and then
        # mirrors it, which is why an observation could say "you are facing
        # South" while the same axis was described as "east".
        compass = float(getattr(dm, "facing_deg", 0.0) or 0.0)
        return Pose(
            frame=FrameName.BUNDLE_WORLD,
            position=Vec3(x_cm=float(dm.x), y_cm=float(dm.y)),
            yaw_deg=(90.0 - compass) % 360.0,
        )

    def _current_sim_time(self) -> float:
        if self._env is None or not hasattr(self._env, "_get_sim_hours"):
            return 0.0
        return float(self._env._get_sim_hours()) * 3600.0

    def _observation(
        self, raw_obs: dict[str, Any], instance: EpisodeSpec | None, *, action_result: ActionResult | None
    ) -> Observation:
        text = str((raw_obs or {}).get("obs_str", ""))
        budgets: dict[str, float] = {}
        if instance is not None:
            budgets = {"steps": float(instance.budgets.steps - self._step_index)}
        elif self._env is not None and self._env._env is not None:
            remaining = float(getattr(self._env._env, "max_steps", 0)) - self._step_index
            budgets = {"steps": max(0.0, remaining)}
        return Observation(
            episode_id=self._episode_id or "unknown",
            step_index=self._step_index,
            text=text,
            marks=self._mark_candidates(),
            agent_pose=self._agent_pose(),
            available_actions=["task", "nav_waypoint"],
            budgets_remaining=budgets,
            last_action_result=action_result,
        )

    def _events_for_step(self, info: dict[str, Any], error: str | None) -> list[Event]:
        """One structured event per step, plus an error event when one occurred.

        The vendored engine does not emit an event stream, so this synthesizes a
        minimal one from what it does report. Event ids are derived from the
        episode and a monotonic counter, never from wall-clock or object
        identity, so design plan §12.6's "retrying a step cannot duplicate events"
        holds and replays produce identical ids.
        """
        events: list[Event] = []
        self._event_seq += 1
        events.append(
            Event(
                event_id=f"{self._episode_id}_ev_{self._event_seq:06d}",
                episode_id=self._episode_id or "unknown",
                step_index=self._step_index,
                kind="step_completed",
                sim_time_s=max(0.0, self._current_sim_time()),
                payload={"is_tool": bool(info.get("is_tool", False))},
            )
        )
        if error:
            self._event_seq += 1
            events.append(
                Event(
                    event_id=f"{self._episode_id}_ev_{self._event_seq:06d}",
                    episode_id=self._episode_id or "unknown",
                    step_index=self._step_index,
                    kind="action_error",
                    sim_time_s=max(0.0, self._current_sim_time()),
                    payload={"message": str(error)[:2000]},
                )
            )
        return events

    @staticmethod
    def _action_error(info: dict[str, Any]) -> str | None:
        """Extract an action failure from the vendored info dict.

        The vendored env surfaces handler-level failures as ``action_error``
        (deliverybench_env.py:1001) and keeps the underlying message in
        ``raw_info["error"]``. Reading only ``error`` at the top level, as an
        earlier version of this adapter did, silently reported every failed
        PICKUP and DROP_OFF as accepted.
        """
        error = info.get("action_error")
        if error:
            return str(error)
        raw = info.get("raw_info") or {}
        error = raw.get("error") if isinstance(raw, dict) else None
        return str(error) if error else None

    def effective_step_budget(self) -> int | None:
        """The step budget the environment is actually enforcing.

        The ``nav`` preset sets ``dynamic_max_steps_mult=2.5``, so the engine
        recomputes ``max_steps`` per seed from an oracle route estimate and the
        value passed to the constructor is only a floor. Reporting the requested
        budget would misstate when truncation is due.
        """
        inner = self._env._env if self._env is not None else None
        budget = getattr(inner, "max_steps", None)
        return int(budget) if budget else None

    def _termination(
        self, done: bool, info: dict[str, Any]
    ) -> tuple[bool, bool, TerminationReason | None]:
        """Split the vendored single ``done`` into terminated vs truncated.

        design plan §7 requires the distinction and the design plan M1 requires separately
        tested causes. The vendored env can end an episode four ways, and each
        maps to exactly one side of the split:

        - simulated shift over  -> truncation, sim-time budget
        - step budget reached   -> truncation, step budget
        - same action failed repeatedly (the engine's anti-stuck guard)
                                -> termination, unrecoverable state
        - otherwise             -> termination, success or failure
        """
        if not done:
            return False, False, None

        metrics = (info.get("metrics") or {}).get("traj_metrics") or {}
        if metrics.get("time_limit_reached") or (
            hasattr(self._env, "_sim_time_exceeded") and self._env._sim_time_exceeded()
        ):
            return False, True, TerminationReason.SIM_TIME_BUDGET_EXHAUSTED

        budget = self.effective_step_budget()
        if budget and (self._step_index + 1) >= budget:
            return False, True, TerminationReason.STEP_BUDGET_EXHAUSTED

        error = self._action_error(info) or ""
        if error.startswith("Terminated:"):
            # The anti-stuck guard fired: the agent repeated a failing action.
            # That is an unrecoverable task state, not a budget running out.
            return True, False, TerminationReason.UNRECOVERABLE_STATE

        success = bool(metrics.get("success")) if "success" in metrics else (
            bool(self._env._check_success()) if hasattr(self._env, "_check_success") else False
        )
        return True, False, (
            TerminationReason.TASK_SUCCESS if success else TerminationReason.TASK_FAILURE
        )

    # ── conformance ──────────────────────────────────────────────────────────

    def authoritative_state(self) -> dict[str, Any]:
        """The state roots design plan §7.2 compares across runtimes."""
        if self._env is None:
            return {}
        inner = self._env._env
        dm = inner.dms[0] if inner.dms else None
        return {
            "dm": dm,
            "order_manager": inner.om,
            "store_manager": getattr(inner, "sm", None),
            "step_count": getattr(inner, "step_count", getattr(inner, "steps", None)),
            "max_steps": getattr(inner, "max_steps", None),
        }

    def authoritative_state_digest(self) -> str:
        return state_digest(self.authoritative_state(), policy=STATE_POLICY)

    def state_tree(self) -> Any:
        return extract_state(self.authoritative_state(), policy=STATE_POLICY)

    @staticmethod
    def event_digest(events: list[Event]) -> str:
        return digest_of([e.to_dict() for e in events])
