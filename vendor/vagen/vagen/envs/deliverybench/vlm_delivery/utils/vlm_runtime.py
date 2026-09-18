# utils/vlm_runtime.py
# -*- coding: utf-8 -*-

"""
Runtime utilities for VLM interaction.

This module contains helper functions to:
- Collect visual observations (global map, local map, first-person view).
- Save debug images and prompts for inspection.
- Dispatch asynchronous VLM calls and track in-flight requests.
- Parse VLM responses into actions, handle retries, and apply results
  back to the DeliveryMan state machine.
"""

import os
from typing import Any, Optional, List
from concurrent.futures import Future

from .util import (
    sanitize_filename,
    ensure_png_bytes,
)
from .trajectory_recorder import save_text, save_png_bytes
from ..gameplay.action_space import parse_action as parse_vlm_action


def vlm_collect_images(agent: Any) -> List[bytes]:
    """
    Collect visual observations for the VLM input (text-only compatible):

        0: Global map view (PNG bytes)
        1: Local map view (PNG bytes)

    All images are converted to PNG bytes so that they can be passed
    directly to the VLM client.
    """
    imgs: List[Optional[bytes]] = [None, None]

    # 0/1: two map views (already bytes or ndarray) → normalize to PNG bytes
    exp = getattr(agent, "map_exportor", None)
    if exp is not None:
        orders = list(agent.active_orders) if getattr(agent, "active_orders", None) else []
        g, l = exp.export(
            agent_xy=(float(agent.x), float(agent.y)),
            orders=orders,
        )
        imgs[0] = ensure_png_bytes(g) if g is not None else None
        imgs[1] = ensure_png_bytes(l) if l is not None else None

    return [b for b in imgs if b is not None]


def export_vlm_images_debug_once(agent: Any) -> List[str]:
    """
    Save the current VLM visual inputs and prompt to disk for debugging.

    For each step, this function:
        - Collects global/local map and first-person images.
        - Saves each image under the agent's run directory.
        - Saves the compiled VLM prompt text for inspection.
    """
    imgs = vlm_collect_images(agent)
    save_dir = getattr(agent, "run_dir", "debug_snaps")

    names = ["global", "local"]
    saved_paths: List[str] = []

    safe_model = sanitize_filename(getattr(agent._vlm_client, "model", "unknown_model"))
    step = getattr(agent, "current_step", 0)

    # print(f"Exporting VLM debug images for model: {safe_model}")

    for i, img in enumerate((imgs or [])[:2]):
        if img is None:
            continue
        fname = f"{safe_model}_{step}_{names[i]}.png"
        path = save_png_bytes(save_dir, fname, img)
        saved_paths.append(path)

    # Save the current VLM prompt
    prompt = agent.build_vlm_input()
    prompt_fname = f"{safe_model}_{step}_prompt.txt"
    save_text(save_dir, prompt_fname, prompt)

    agent.current_step = step + 1
    return saved_paths


def vlm_request_async(agent: Any, prompt: str) -> bool:
    """
    Submit an asynchronous VLM request for the given prompt.

    This function:
        - Validates that executor and client are available.
        - Ensures only one in-flight request at a time.
        - Captures images on the main thread.
        - Exports debug snapshots (images + prompt).
        - Submits a background network call to the VLM client.
        - Queues the result (or error) into `agent._vlm_results_q`.

    Returns:
        True if a new request was successfully dispatched, else False.
    """
    if agent._vlm_executor is None or agent._vlm_client is None:
        agent._log("[VLM] executor/client not set")
        return False

    # If a request is still in flight, do not start a new one
    if agent._vlm_future and not agent._vlm_future.done():
        return False

    # Pause internal timers while waiting for the VLM decision
    agent.timers_pause()

    # Capture images on the main thread
    images = vlm_collect_images(agent)

    # Export debug images and prompt for inspection
    export_vlm_images_debug_once(agent)

    # Assign a token so late responses can be discarded if outdated
    agent._vlm_token_ctr += 1
    token = agent._vlm_token_ctr
    agent._vlm_inflight_token = token
    agent._waiting_vlm = True

    # Background task: perform the network call only
    def _call():
        resp = agent._vlm_client.generate(user_prompt=prompt, images=images)
        return {"token": token, "resp": resp}

    fut = agent._vlm_executor.submit(_call)
    agent._vlm_future = fut

    # Callback: only enqueue results; do not touch UI or main state machine
    def _done(f: Future):
        try:
            res = f.result()
            agent._vlm_results_q.append(res)
        except Exception as e:
            agent._vlm_results_q.append({"token": token, "error": str(e)})

    fut.add_done_callback(_done)
    return True


def vlm_pump_results(agent: Any) -> bool:
    """
    Apply any completed VLM results to the agent's state machine.

    This should be called periodically on the main thread. It:
        - Processes pending items in `agent._vlm_results_q`.
        - Discards stale results whose tokens do not match the current token.
        - Clears in-flight flags when a matching result is handled.
        - Routes success to `_vlm_handle_response` and failures to `_vlm_on_failed`.
        - Resumes timers if they were paused for VLM processing.

    Returns:
        True if at least one result was processed, otherwise False.
    """
    processed = False

    while agent._vlm_results_q:
        rec = agent._vlm_results_q.popleft()
        if rec.get("token") != agent._vlm_inflight_token:
            # Stale result; ignore it
            continue

        # Clear in-flight state
        agent._waiting_vlm = False
        agent._vlm_inflight_token = None
        agent._vlm_future = None

        if "error" in rec:
            _vlm_on_failed(agent, rec["error"])
        else:
            _vlm_handle_response(agent, rec["resp"])
        processed = True

    # If we applied a result and are no longer waiting, resume timers if paused
    if processed and not getattr(agent, "_waiting_vlm", False) and agent._timers_paused:
        agent.timers_resume()

    return processed


def _vlm_handle_response(agent: Any, resp: Any) -> None:
    """
    Handle a successful VLM response.

    Responsibilities:
        - Log the raw response.
        - Increment call statistics.
        - Parse the response into a DMAction and optional language plan.
        - Save the raw output for debugging.
        - Enqueue the parsed action for execution.

    On parse failure, it delegates to `_vlm_retry`.
    """
    raw = str(resp)

    # Record VLM call statistics
    if agent._recorder:
        agent._recorder.inc("vlm_calls")

    try:
        agent._log(f"[VLM] raw output: {raw}")

        # Parse VLM output into a DMAction
        act, language_plan = parse_vlm_action(raw, agent)
        agent._previous_language_plan = language_plan

        # Local import to avoid circular dependencies
        from ..entities.delivery_man import DMAction  # type: ignore

        if not isinstance(act, DMAction):
            raise ValueError(f"bad return type: {type(act)}")
        else:
            # Save raw output for debugging
            safe_model = sanitize_filename(getattr(agent._vlm_client, "model", "unknown_model"))
            step = getattr(agent, "current_step", 0)
            path = os.path.join(agent.run_dir, f"{safe_model}_{step}_output.txt")
            save_text(agent.run_dir, path, raw, encoding="utf-8")

    except Exception as e:
        # Parsing failed → record and trigger a retry
        if agent._recorder:
            agent._recorder.inc("vlm_parse_failures")
        _vlm_retry(agent, str(e), sample=raw)
        return

    # Successful parse: reset retry counters and hints
    agent._vlm_retry_count = 0
    agent.vlm_ephemeral.pop("format_hint", None)
    if agent._recorder:
        agent._recorder.inc("vlm_successes")

    agent.logger.info(f"[VLM] parsed action: {act.kind} {act.data if act.data else ''}")
    agent.enqueue_action(act)


def _vlm_retry(agent: Any, reason: str, sample: Optional[str] = None) -> None:
    """
    Retry logic when a VLM call fails or returns invalid output.

    This function:
        - Increments retry counters and statistics.
        - Stores a truncated snapshot of the last invalid output.
        - Adds a format hint into `vlm_ephemeral` to guide the next response.
        - Records a recent error string.
        - If under the retry limit, issues a new VLM request with the
          latest prompt; otherwise, enqueues a safe fallback action.
    """
    agent._vlm_retry_count += 1
    if agent._recorder:
        agent._recorder.inc("vlm_retries")

    agent._vlm_last_bad_output = str(sample)[:160] if sample is not None else None

    # This hint is injected into the ephemeral_context block of the next prompt
    agent.vlm_ephemeral["format_hint"] = (
        "Your previous output was invalid. Reply with exactly ONE action call from the Action API. "
        "No explanations or apologies."
    )

    # recent_error is also surfaced in the next prompt
    agent.vlm_add_error(
        f"VLM invalid output (attempt {agent._vlm_retry_count}/{agent._vlm_retry_max}): {reason}"
    )

    if agent._vlm_retry_count <= agent._vlm_retry_max:
        # Reissue a request with an updated prompt that includes format_hint and recent_error
        vlm_request_async(agent, agent.build_vlm_input())
    else:
        # Exceeded retry limit: reset state, remove hint, and enqueue a gentle fallback action
        agent._vlm_retry_count = 0
        agent.vlm_ephemeral.pop("format_hint", None)
        try:
            from ..entities.delivery_man import DMAction, DMActionKind  # type: ignore
            agent.enqueue_action(DMAction(DMActionKind.VIEW_ORDERS, data={}))
        except Exception:
            # If fallback enqueue fails, avoid blocking the agent
            pass


def _vlm_on_failed(agent: Any, msg: str) -> None:
    """
    Handle a transport-level or runtime failure of a VLM call.

    Logs the error and forwards it to the retry logic.
    """
    agent._log(f"[VLM] error: {msg}")
    _vlm_retry(agent, msg)