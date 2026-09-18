# utils/action_runtime.py
# -*- coding: utf-8 -*-

"""
Action runtime helpers for DeliveryMan agents.

"""

import threading
from typing import Any, Optional, Callable, List

from ..base.defs import *
from ..gameplay.action_space import action_to_text


# ---------------------------------------------------------------------------
# Core decision logic
# ---------------------------------------------------------------------------

def dm_default_decider(agent: Any) -> Optional[DMAction]:
    """
    Original: DeliveryMan._default_decider(self)

    Decide the next high-level action by:
    - Skipping decision if the lifecycle is done, the agent is rescued,
      currently in hospital context, or has no energy.
    - Skipping if we are already waiting for a VLM result.
    - Building a VLM prompt and dispatching an async VLM request.
    The function always returns immediately to avoid blocking the UI.
    """
    # States in which no decision should be made: immediately return.
    if getattr(agent, "_lifecycle_done", False):
        return None
    if agent.is_rescued or getattr(agent, "_hospital_ctx", None) is not None or agent.energy_pct <= 0.0:
        return None

    # If a VLM request is already in flight, do not send another one.
    if getattr(agent, "_waiting_vlm", False):
        return None

    # Build the prompt for the VLM.
    prompt = agent.build_vlm_input()
    agent.logger.debug(f"[VLM] User Prompt:\n{prompt}")

    # Dispatch the VLM request asynchronously (non-blocking).
    try:
        agent.request_vlm_async(prompt)
    except Exception as e:
        agent.vlm_add_error(f"VLM dispatch error: {e}")

    # Return immediately; do not block the UI.
    return None


def dm_kickstart(agent: Any) -> None:
    """
    Original: DeliveryMan.kickstart(self)

    Kick off the decision loop when the agent is idle and has no queued actions.
    """
    if agent._current is None and not agent._queue:
        agent.timers_pause()
        act = dm_default_decider(agent)
        agent.timers_resume()
        if act is not None:
            agent.enqueue_action(act)


# ---------------------------------------------------------------------------
# Queue management
# ---------------------------------------------------------------------------

def dm_enqueue_action(agent: Any, action: DMAction, *, allow_interrupt: bool = False) -> None:
    """
    Original: DeliveryMan.enqueue_action(self, action, allow_interrupt=False)

    Enqueue a new action for the agent. Optionally allows interrupting
    the current action and clearing the remaining queue.
    """
    if not isinstance(action, DMAction):
        agent._log(f"ignore invalid action enqueued: {type(action)}")
        return
    if agent.is_rescued or getattr(agent, "_hospital_ctx", None) is not None:
        return

    if allow_interrupt and agent._current is not None:
        agent._queue.clear()
        agent._current = None
        dm_start_action(agent, action, allow_interrupt=True)
        return

    agent._queue.append(action)
    dm_start_next_if_idle(agent)


def dm_clear_queue(agent: Any) -> None:
    """
    Original: DeliveryMan.clear_queue(self)

    Remove all pending actions from the queue.
    """
    agent._queue.clear()


def dm_start_next_if_idle(agent: Any) -> None:
    """
    Original: DeliveryMan._start_next_if_idle(self)

    If there is no current action but the queue is non-empty,
    start the next action in FIFO order.
    """
    if agent._current is None and agent._queue:
        act = agent._queue.pop(0)
        dm_start_action(agent, act)


# ---------------------------------------------------------------------------
# Action lifecycle
# ---------------------------------------------------------------------------

def dm_start_action(agent: Any, act: DMAction, allow_interrupt: bool = True) -> None:
    """
    Original: DeliveryMan._start_action(self, act, allow_interrupt=True)

    Start executing a specific action by:
    - Setting it as the current action.
    - Looking up and invoking the registered handler.
    - Recording an attempt in the optional recorder.
    """
    agent._current = act
    agent.last_action_kind = act.kind  # read by deliverybench_env.step() for info["is_tool"]
    handler = agent._action_handlers.get(act.kind)
    if handler is None:
        dm_finish_action(agent, success=False)
        return

    if getattr(agent, "_recorder", None):
        agent._recorder.inc_nested(f"action_attempts.{act.kind.value}")

    # Handler signature: handler(act, allow_interrupt)
    handler(act, allow_interrupt)


def dm_finish_action(agent: Any, *, success: bool) -> None:
    """
    Complete the current action, trigger callbacks, update stats, and
    dispatch the next decision if the lifecycle is not finished.

    In manual_step (gym-like) mode:
    - Do NOT auto-trigger dm_default_decider()
    - Instead notify the env that one action finished (a step boundary)

    Extra (for robust gym-like stepping):
    - Maintain agent._manual_step_done_seq: increments once per finished action,
      so env.step() can wait on a monotonic edge (never misses "fast" actions).
    """
    # 1) callbacks / stats
    if agent._current and callable(agent._current.on_done):
        agent._current.on_done(agent)

    if success and agent._current:
        agent._register_success(action_to_text(agent._current))

    if getattr(agent, "_recorder", None) and agent._current and success:
        action_name = agent._current.kind.value
        agent._recorder.inc_nested(f"action_successes.{action_name}")

    # 2) clear current action
    agent._current = None

    # If the lifecycle is marked as done, do not schedule further actions.
    if getattr(agent, "_lifecycle_done", False):
        agent._current = None
        return

    # ---------------------------
    # gym-like manual step mode
    # ---------------------------
    if getattr(agent, "manual_step", False):
        # ✅ Robust "edge" signal: increment done sequence (never misses fast actions)
        try:
            if getattr(agent, "_manual_step_done_seq", None) is None:
                agent._manual_step_done_seq = 0
            agent._manual_step_done_seq += 1
        except Exception:
            # don't let instrumentation break the runtime
            pass

        # Notify env condition variable (idle boundary)
        cv = getattr(agent, "_step_cv", None)
        if cv is not None:
            try:
                # cv is expected to be threading.Condition
                with cv:
                    cv.notify_all()
            except Exception:
                pass
        return

    # ---------------------------
    # Original behavior: auto-trigger next VLM decision
    # ---------------------------
    agent.timers_pause()
    next_act = dm_default_decider(agent)
    if next_act is not None:
        delay = int(agent.cfg.get("vlm", {}).get("next_action_delay_ms", 300))
        threading.Timer(delay / 1000.0, lambda: agent.enqueue_action(next_act)).start()
    agent.timers_resume()


# ---------------------------------------------------------------------------
# Registration helpers
# ---------------------------------------------------------------------------

def dm_register_action(
    agent: Any,
    kind: DMActionKind,
    handler: Callable[[Any, DMAction, bool], None],
) -> None:
    """
    Original: DeliveryMan.register_action(self, kind, handler)

    Register an action handler for a given DMActionKind.
    """
    agent._action_handlers[kind] = handler