"""
rollout_qwen.py
───────────────
Run N rollouts of DeliveryBench with a Qwen model on OpenRouter, a local
vLLM endpoint, or a local sglang endpoint, and report the mean / std of
hourly profit across rollouts.

All settings come from a single YAML config with two sections:
    rollout:  how to run (backend / model / sampling / concurrency / logging)
    env:      DeliveryBenchEnvConfig fields (the task definition)
See configs/rollout_config.yaml for the full, explicit set of fields.

Usage:
    python -m vagen.envs.deliverybench.rollout_qwen [CONFIG_YAML]

    # default config (configs/rollout_config.yaml):
    python -m vagen.envs.deliverybench.rollout_qwen
    # explicit config:
    python -m vagen.envs.deliverybench.rollout_qwen configs/my_run.yaml

The only secret read from the environment is the OpenRouter API key
(OPENROUTER_API_KEY or OPENAI_API_KEY), used when rollout.backend == "openrouter".

Outputs (under outputs/rollout_<timestamp>/):
    log.jsonl       one record per step / rollout / summary event
    images/         per-step images sent to the model
    trajectory.html auto-rendered agent<->env trajectory viewer
"""

import argparse
import asyncio
import json
import logging
import os
import re
import statistics
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from openai import AsyncOpenAI

from .deliverybench_env import DeliveryBench, DeliveryBenchEnvConfig

# ──────────────────────────────────────────────────────────────────────────────
# Config
# ──────────────────────────────────────────────────────────────────────────────


import dataclasses
import yaml


@dataclasses.dataclass
class RolloutConfig:
    """Rollout settings, loaded from the `rollout:` section of the config YAML.

    Every field is required (no defaults) — the YAML lists them all explicitly.
    """
    backend: str                      # "vllm" | "sglang" | "openrouter"
    vllm_base_url: str
    vllm_model_name: str
    vllm_api_key: str
    sglang_base_url: str
    sglang_model_name: str
    sglang_api_key: str
    sglang_grammar: bool
    openrouter_base_url: str
    openrouter_model_id: str
    num_rollouts: int
    seed_list: List[int]
    max_steps: int
    concurrency: int
    temperature: float
    top_p: float
    max_tokens: int
    request_timeout: float
    max_consecutive_api_failures: int
    image_turns_kept: int
    context_token_threshold: int
    chars_per_token: int
    tokens_per_image: int
    metrics: List[str]                # trajectory metrics to report (see benchmark.py)

    @property
    def model_id(self) -> str:
        return {
            "openrouter": self.openrouter_model_id,
            "sglang": self.sglang_model_name,
        }.get(self.backend, self.vllm_model_name)

    @property
    def seeds(self) -> List[int]:
        """Seed for each rollout to run.

        Every seed in `seed_list` is repeated `num_rollouts` times, so the
        total number of rollouts is len(seed_list) * num_rollouts.
        """
        return [s for s in self.seed_list for _ in range(self.num_rollouts)]


def _load_config(path: Path) -> Tuple[RolloutConfig, DeliveryBenchEnvConfig]:
    """Load the `rollout:` and `env:` sections of the config YAML into a
    RolloutConfig and a DeliveryBenchEnvConfig. Unknown keys are rejected."""
    if not path.exists():
        raise FileNotFoundError(f"rollout config YAML not found: {path}")
    with path.open("r", encoding="utf-8") as f:
        raw = yaml.safe_load(f) or {}
    rollout = dict(raw.get("rollout", {}) or {})
    env = dict(raw.get("env", {}) or {})

    def _reject_unknown(d: dict, cls, label: str) -> None:
        valid = {fld.name for fld in dataclasses.fields(cls)}
        unknown = set(d) - valid
        if unknown:
            raise ValueError(f"{path.name} `{label}` has unknown fields: {sorted(unknown)}")

    _reject_unknown(rollout, RolloutConfig, "rollout")
    preset = env.pop("preset", None)
    _reject_unknown(env, DeliveryBenchEnvConfig, "env")
    if preset is not None:
        from .deliverybench_env import preset_config
        return RolloutConfig(**rollout), preset_config(str(preset), **env)
    return RolloutConfig(**rollout), DeliveryBenchEnvConfig(**env)


# Config path is taken from the command line:
#   python -m vagen.envs.deliverybench.rollout_qwen <config.yaml>
# Falls back to configs/rollout_config.yaml when omitted.
_parser = argparse.ArgumentParser(description="DeliveryBench rollout")
_parser.add_argument(
    "config",
    nargs="?",
    default=str(Path(__file__).parent / "configs" / "rollout_config.yaml"),
    help="path to the rollout config YAML (with `rollout:` and `env:` sections)",
)
_CONFIG_PATH = Path(_parser.parse_known_args()[0].config)
cfg, ENV_CONFIG = _load_config(_CONFIG_PATH)

# ──────────────────────────────────────────────────────────────────────────────
# Prompt policy — FROZEN operational prompt for the curriculum rollout scan
# (B0/B1/B2/B3/...).
#
#   Scaffolded Task-Navigation Prompt v2.1
#
# This is the FIXED operational prompt used for the difficulty scan. It restores
# the scaffolded suffix that achieved the first end-to-end B0 delivery
# (run B0_optA_20260615_145107: 1 delivery, DROP_OFF at step 36, +$7.97), under
# FixA/FixB action normalization + the Option-A 10 m door tolerance. It states the
# objective, the response format, and light task-navigation scaffolding: the
# per-order workflow (view→accept→pickup→dropoff), how STEP_TO adjacency works,
# how to use NAVIGATE for routing, the visual observation format, and to read
# error messages instead of repeating a failed action.
#
# Ablation history (do NOT restore these — both FAILED B0):
#   - "Minimal Interface Prompt v1" (interface-only): too sparse — the agent
#     picked up but never attempted drop-off. 0 deliveries.
#   - "Minimal Task-Interface Prompt v1.1" / "Protocol-Aware Visual Prompt v1":
#     added delivery semantics + env protocol facts but removed the workflow
#     scaffold; the agent pursued the customer yet circled / died to the
#     repeat-guard without delivering. 0 deliveries.
# The scaffolded prompt is the one that actually completes B0, so it is the fixed
# baseline for the scan; the minimal/protocol-aware prompts are recorded as
# ablations.
#
# DO NOT change this prompt while changing stage/difficulty parameters — difficulty
# comparisons are only valid under a fixed prompt. To compare prompt variants, bump
# the version and treat it as its own axis.
# ──────────────────────────────────────────────────────────────────────────────
PROMPT_VERSION = "Scaffolded Task-Navigation Prompt v2.1"

_ACTION_SCHEMA: Dict[str, Any] = {
    "type": "object",
    "properties": {
        "reasoning_and_reflection": {"type": "string"},
        "action": {"type": "string"},
        "future_plan": {"type": "string"},
    },
    "required": ["reasoning_and_reflection", "action", "future_plan"],
    "additionalProperties": False,
}

# ──────────────────────────────────────────────────────────────────────────────
# Logging
# ──────────────────────────────────────────────────────────────────────────────

# Output root and run label are overridable so sweep runs land under
# experiments/<...>/ with a descriptive band name (e.g. RUN_LABEL="B0_sanity").
_TS = datetime.now().strftime("%Y%m%d_%H%M%S")
_RUN_LABEL = os.environ.get("RUN_LABEL", "rollout")
_OUT_ROOT = os.environ.get("ROLLOUT_OUT_DIR")
_RUN_DIR = (
    Path(_OUT_ROOT) / f"{_RUN_LABEL}_{_TS}" if _OUT_ROOT
    else Path(__file__).parent / "outputs" / f"{_RUN_LABEL}_{_TS}"
).resolve()
_IMG_DIR = _RUN_DIR / "images"
_RUN_DIR.mkdir(parents=True, exist_ok=True)
_IMG_DIR.mkdir(parents=True, exist_ok=True)

logging.basicConfig(level=logging.INFO, format="[%(levelname)s] %(message)s",
                    handlers=[logging.StreamHandler(sys.stdout)])
log = logging.getLogger("rollout_qwen")

_log_fh = (_RUN_DIR / "log.jsonl").open("w", encoding="utf-8")
_log_lock = asyncio.Lock()


def _emit(record: Dict[str, Any]) -> None:
    _log_fh.write(json.dumps(record, ensure_ascii=False) + "\n")
    _log_fh.flush()


def _save_step_images(obs: Dict[str, Any], rollout_id: int, step: int) -> List[str]:
    mmi = obs.get("multi_modal_input")
    if not mmi:
        return []
    out = _IMG_DIR / f"rollout_{rollout_id}"
    out.mkdir(parents=True, exist_ok=True)
    paths = []
    imgs = [im for v in mmi.values() for im in (v if isinstance(v, list) else [v])]
    for i, img in enumerate(imgs):
        p = out / f"step_{step:03d}_img{i}.png"
        img.save(p)
        paths.append(str(p.relative_to(_RUN_DIR)))
    return paths


# ──────────────────────────────────────────────────────────────────────────────
# Model client
# ──────────────────────────────────────────────────────────────────────────────

def _make_client() -> AsyncOpenAI:
    if cfg.backend == "vllm":
        log.info(f"Backend: vLLM  url={cfg.vllm_base_url}  model={cfg.vllm_model_name}")
        return AsyncOpenAI(base_url=cfg.vllm_base_url, api_key=cfg.vllm_api_key, timeout=cfg.request_timeout)
    if cfg.backend == "sglang":
        log.info(f"Backend: sglang  url={cfg.sglang_base_url}  model={cfg.sglang_model_name}")
        return AsyncOpenAI(base_url=cfg.sglang_base_url, api_key=cfg.sglang_api_key, timeout=cfg.request_timeout)
    api_key = os.environ.get("OPENROUTER_API_KEY") or os.environ.get("OPENAI_API_KEY")
    if not api_key:
        raise RuntimeError("Set OPENROUTER_API_KEY (or OPENAI_API_KEY) before running.")
    log.info(f"Backend: OpenRouter  model={cfg.openrouter_model_id}")
    return AsyncOpenAI(base_url=cfg.openrouter_base_url, api_key=api_key, timeout=cfg.request_timeout)


def _make_system_msg(content: str) -> Dict[str, Any]:
    if cfg.backend == "openrouter":
        # cache_control caches the large static system prompt on supported providers
        return {"role": "system",
                "content": [{"type": "text", "text": content,
                             "cache_control": {"type": "ephemeral"}}]}
    return {"role": "system", "content": content}


async def _call_model(client: AsyncOpenAI, messages: List[Dict[str, Any]], *,
                      structured: bool = True, session_id: Optional[str] = None) -> str:
    kwargs: Dict[str, Any] = {"model": cfg.model_id, "messages": messages,
                              "temperature": cfg.temperature, "top_p": cfg.top_p,
                              "max_tokens": cfg.max_tokens}
    if cfg.backend == "openrouter":
        if structured:
            kwargs["response_format"] = {
                "type": "json_schema",
                "json_schema": {"name": "delivery_action", "strict": True,
                                "schema": _ACTION_SCHEMA},
            }
        extra: Dict[str, Any] = {
            "reasoning": {"effort": "none"},          # no CoT tokens
            "plugins": [{"id": "response-healing"}],  # auto-repair malformed JSON
        }
        if session_id:
            extra["session_id"] = session_id          # sticky routing → warm cache
        kwargs["extra_body"] = extra
    elif cfg.backend == "sglang":
        # sglang enforces JSON via response_format json_schema (xgrammar backend).
        # NOTE: grammar-constrained sampling triggers a TP all-reduce
        # (sampler._sync_token_ids_across_tp) that, on nodes whose NVIDIA driver is
        # older than the env's CUDA runtime (e.g. driver 550/CUDA 12.4 with a cu128
        # build), crashes the server with an NCCL "insufficient driver" error. The
        # prompt already mandates pure JSON and DeliveryBench parses it from raw text,
        # so we leave grammar OFF by default. Set SGLANG_GRAMMAR=1 to re-enable once
        # the driver/runtime mismatch is resolved.
        if structured and cfg.sglang_grammar:
            kwargs["response_format"] = {
                "type": "json_schema",
                "json_schema": {"name": "delivery_action", "schema": _ACTION_SCHEMA},
            }
    elif structured:  # vllm
        kwargs["extra_body"] = {"guided_json": _ACTION_SCHEMA}

    resp = await client.chat.completions.create(**kwargs)
    return resp.choices[0].message.content or ""


_THINK_RE = re.compile(r"^.*?</think>\s*", re.DOTALL)


def _sanitize_response(text: str) -> str:
    """
    Reduce a raw model response to the JSON action object.

    Handles the failure modes seen with Qwen: a <think>...</think> preamble
    before the JSON, markdown code fences, and stray prose around the object.
    Returning clean JSON lets the env json.loads it (so escaped quotes inside
    the action string are correctly unescaped) instead of falling back to a
    regex scrape of the raw text.
    """
    t = (text or "").strip()
    if "</think>" in t:                       # drop the leading think block
        t = _THINK_RE.sub("", t, count=1).strip()
    if t.startswith("```"):                   # strip markdown fences
        t = re.sub(r"^```[a-zA-Z0-9]*\s*", "", t)
        t = re.sub(r"\s*```$", "", t).strip()
    if not t.startswith("{"):                 # extract the first {...} object
        m = re.search(r"\{.*\}", t, re.DOTALL)
        if m:
            t = m.group(0)
    return t


# ──────────────────────────────────────────────────────────────────────────────
# History management
# ──────────────────────────────────────────────────────────────────────────────

import re as _re
# Fix A — tolerant action normalizer. Qwen3-VL frequently emits bareword waypoint
# args, e.g. STEP_TO(int_10), which the env's AST parser rejects
# ("Unsupported expression: Name"). This quotes bare identifier args for the
# locomotion/handling actions so a correct intent is not lost to a quoting slip.
# It does NOT invent or change which waypoint the model chose — only adds quotes.
# Every rewrite is counted and logged so the format-tax remains measurable.
_BAREWORD_CALL = _re.compile(r'\b(STEP_TO|MOVE|PICKUP|DROP_OFF)\(\s*([A-Za-z][\w]*)\s*\)')
# JS-style boolean kwargs, e.g. NAVIGATE(target="x", text=true). The AST parser
# rejects bare `true`/`false` (Name nodes), which silently breaks the entire
# NAVIGATE(text=True) routing procedure → the agent never gets the waypoint chain.
_JS_BOOL = _re.compile(r'(\b\w+\s*=\s*)(true|false)\b')


def _normalize_action_text(resp: str) -> tuple:
    """Return (normalized_response, n_rewrites). Tolerant fixes that preserve the
    model's intent: quote bare waypoint ids, and Python-case kwarg booleans.
    Never changes which waypoint/target/flag value the model chose."""
    n = 0

    def _quote(m):
        nonlocal n
        n += 1
        return f"{m.group(1)}('{m.group(2)}')"

    def _bool(m):
        nonlocal n
        n += 1
        return m.group(1) + ("True" if m.group(2) == "true" else "False")

    out = _BAREWORD_CALL.sub(_quote, resp)
    out = _JS_BOOL.sub(_bool, out)
    return out, n


def _extract_action_field(text: str) -> str:
    """Best-effort pull of the raw "action" string the model emitted, for logging
    the pre-normalization action alongside the normalized one. Returns "" on miss."""
    try:
        obj = json.loads(text)
        if isinstance(obj, dict) and "action" in obj:
            return str(obj["action"])
    except Exception:
        pass
    m = _re.search(r'"action"\s*:\s*"((?:[^"\\]|\\.)*)"', text)
    return m.group(1) if m else ""


def _pil_to_data_url(img) -> str:
    import base64
    from io import BytesIO
    buf = BytesIO()
    img.save(buf, format="PNG")
    return "data:image/png;base64," + base64.b64encode(buf.getvalue()).decode()


def _obs_to_user_message(obs: Dict[str, Any], step: int, action_error: Optional[str]) -> Dict[str, Any]:
    """Build the user message for one step: images + obs text (+ error feedback)."""
    text = obs.get("obs_str", "").replace("<image>", "").strip()
    if action_error:
        text = f"⚠ Your previous action FAILED: {action_error}\n\n{text}"
    text = f"[Step {step}]\n{text}"

    mmi = obs.get("multi_modal_input")
    if not mmi:
        return {"role": "user", "content": text}
    imgs = [im for v in mmi.values() for im in (v if isinstance(v, list) else [v])]
    content = [{"type": "image_url", "image_url": {"url": _pil_to_data_url(im)}} for im in imgs]
    content.append({"type": "text", "text": text})
    return {"role": "user", "content": content}


def _strip_old_images(history: List[Dict[str, Any]], keep_last: Optional[int] = None) -> None:
    """Drop image parts from all but the last `keep_last` user turns.

    Old map/FPV snapshots are stale (the text state is refreshed every turn)
    and re-sending them every step multiplies cost and latency.
    """
    if keep_last is None:
        keep_last = cfg.image_turns_kept
    user_idxs = [i for i, m in enumerate(history)
                 if m["role"] == "user" and isinstance(m.get("content"), list)]
    for i in user_idxs[:-keep_last] if keep_last else user_idxs:
        parts = history[i]["content"]
        n_imgs = sum(1 for p in parts if p.get("type") == "image_url")
        if not n_imgs:
            continue
        text = "\n".join(p.get("text", "") for p in parts if p.get("type") == "text")
        history[i]["content"] = f"[{n_imgs} image(s) omitted]\n{text}"


def _estimate_tokens(messages: List[Dict[str, Any]]) -> int:
    total = 0
    for msg in messages:
        content = msg.get("content", "")
        if isinstance(content, str):
            total += len(content) // cfg.chars_per_token
        else:
            for part in content:
                if part.get("type") == "text":
                    total += len(part.get("text", "")) // cfg.chars_per_token
                elif part.get("type") == "image_url":
                    total += cfg.tokens_per_image
    return total


async def _compress_history(client: AsyncOpenAI, system_msg: Dict[str, Any],
                            history: List[Dict[str, Any]],
                            session_id: Optional[str]) -> List[Dict[str, Any]]:
    """Replace the history with a short model-written summary of past actions."""
    prompt = ("The conversation history is getting long. Summarize ONLY your own past "
              "actions and their outcomes (what you did, whether it succeeded). "
              "Do NOT include state the environment refreshes every turn. Under 300 words.")
    summary = await _call_model(client, [system_msg] + history +
                                [{"role": "user", "content": prompt}],
                                structured=False, session_id=session_id)
    return [
        {"role": "user", "content": f"[History summary — your actions so far]\n{summary}"},
    ]


def _delivery_boundary_history(prev_deliveries: int, deliveries: int, action: str) -> List[Dict[str, Any]]:
    """Reset stale per-order history after a successful delivery."""
    return [
        {
            "role": "user",
            "content": (
                "[Delivery boundary]\n"
                f"A DROP_OFF action succeeded and completed delivery count increased "
                f"from {prev_deliveries} to {deliveries}.\n"
                f"Completed action: {action}\n"
                "Previous pickup/dropoff navigation and failed route attempts are no "
                "longer relevant. Use the next current observation for any remaining "
                "active orders, carried orders, or new order selection."
            ),
        }
    ]


# ──────────────────────────────────────────────────────────────────────────────
# Single rollout
# ──────────────────────────────────────────────────────────────────────────────

async def run_one_rollout(rollout_id: int, seed: int, client: AsyncOpenAI) -> Dict[str, Any]:
    tag = f"[r{rollout_id}]"
    log.info(f"{tag} start  seed={seed}")
    async with _log_lock:
        _emit({"event": "rollout_start", "rollout_id": rollout_id, "seed": seed})

    env = DeliveryBench(ENV_CONFIG)
    session_id = f"delivery-r{rollout_id}-s{seed}"

    sys_obs = await env.system_prompt()
    system_text = sys_obs["obs_str"]
    system_msg = _make_system_msg(system_text)
    async with _log_lock:
        _emit({"event": "system", "rollout_id": rollout_id, "system_text": system_text})

    obs, _ = await env.reset(seed=seed)
    _save_step_images(obs, rollout_id, 0)

    history: List[Dict[str, Any]] = []
    action_error: Optional[str] = None
    step = 0
    earnings, sim_hours, deliveries = 100.0, 0.0, 0
    success = False
    api_failures = 0
    aborted: Optional[str] = None

    while step < cfg.max_steps:
        step += 1

        if _estimate_tokens([system_msg] + history) > cfg.context_token_threshold and len(history) >= 4:
            log.info(f"{tag} compressing history at step {step}")
            try:
                history = await _compress_history(client, system_msg, history, session_id)
            except Exception as exc:
                # Compression is best-effort; on failure just drop the oldest
                # half of the turns rather than killing the rollout.
                log.warning(f"{tag} history compression failed ({exc}); truncating instead")
                history = history[len(history) // 2:]

        # Faithful copy of the text the agent receives this turn (mirrors
        # _obs_to_user_message), captured for the trajectory viewer.
        prev_error = action_error
        obs_in_text = obs.get("obs_str", "").replace("<image>", "").strip()
        if prev_error:
            obs_in_text = f"⚠ Your previous action FAILED: {prev_error}\n\n{obs_in_text}"

        history.append(_obs_to_user_message(obs, step, action_error))
        _strip_old_images(history)

        t0 = time.time()
        try:
            response = await _call_model(client, [system_msg] + history, session_id=session_id)
            api_failures = 0
        except Exception as exc:
            api_failures += 1
            log.warning(f"{tag} model error ({api_failures}/{cfg.max_consecutive_api_failures}): {exc}")
            if api_failures >= cfg.max_consecutive_api_failures:
                aborted = f"{api_failures} consecutive API failures: {exc}"
                log.error(f"{tag} ABORTED — {aborted}")
                break
            history.pop()   # retry the same observation next iteration
            step -= 1
            await asyncio.sleep(5 * api_failures)
            continue
        latency = round(time.time() - t0, 2)
        action_raw = _extract_action_field(response)          # pre-normalization
        response, n_norm = _normalize_action_text(response)   # transparent parser tolerance
        history.append({"role": "assistant", "content": response})

        prev_deliveries = deliveries
        obs, reward, done, info = await env.step(response)
        obs_out_text = obs.get("obs_str", "").replace("<image>", "").strip()

        parsed = info.get("parsed", {})
        action = parsed.get("action") or "?"
        action_error = info.get("action_error")
        traj = info["metrics"]["traj_metrics"]
        turn = info["metrics"]["turn_metrics"]
        dm0 = env._env.dms[0] if env._env and env._env.dms else None
        traffic_light_check = getattr(dm0, "last_traffic_light_check", None) if dm0 is not None else None
        earnings = float(getattr(dm0, "earnings_total", earnings))
        sim_hours = traj["sim_hours"]
        deliveries = traj["deliveries_completed"]
        success = traj["success"]
        img_paths = _save_step_images(obs, rollout_id, step)

        err_tag = f"  ✗ {action_error[:70]}" if action_error else ""
        log.info(f"{tag} step={step:03d} sim={sim_hours:.2f}h earn=${earnings:.2f} "
                 f"deliv={deliveries} lat={latency}s {action[:50]!r}{err_tag}")

        async with _log_lock:
            _emit({
                "event": "step", "rollout_id": rollout_id, "step": step,
                "sim_hours": round(sim_hours, 4), "earnings": round(earnings, 4),
                "reward": round(reward, 4), "deliveries": deliveries,
                "action_raw": action_raw,             # exactly what the model emitted
                "action": action,                     # normalized + env-parsed action
                "action_valid": turn["action_is_valid"],
                "action_normalized": n_norm,           # # of tolerant rewrites applied
                "action_error": action_error,
                "traffic_light_check": traffic_light_check,
                "traffic_violations": traj.get("traffic_violations", 0),
                "pedestrian_traffic_light_checks": traj.get("pedestrian_traffic_light_checks", 0),
                "pedestrian_traffic_light_violations": traj.get("pedestrian_traffic_light_violations", 0),
                "oracle_next_move": info.get("oracle_next_move"),
                "oracle_next_action": info.get("oracle_next_action"),
                "oracle_next_move_before_action": info.get("oracle_next_move_before_action"),
                "oracle_next_action_before_action": info.get("oracle_next_action_before_action"),
                "route_arrived": info.get("route_arrived"),
                "reasoning": (parsed.get("reasoning") or "")[:500],
                # Full interaction record for the trajectory viewer:
                "obs_in_text": obs_in_text,        # observation shown to the model this turn
                "model_response": response,        # raw model output (verbatim)
                "obs_out_text": obs_out_text,      # observation after the action executed
                "images": img_paths, "latency_s": latency,
            })

        delivered_now = (
            action_error is None
            and str(action).lstrip().upper().startswith("DROP_OFF")
            and int(deliveries) > int(prev_deliveries)
        )
        if delivered_now and not done:
            log.info(f"{tag} resetting history after successful delivery boundary")
            history = _delivery_boundary_history(prev_deliveries, deliveries, action)

        if done:
            break

    net_profit = earnings - 100.0
    hourly = net_profit / sim_hours if sim_hours > 0 else 0.0
    result = {
        "rollout_id": rollout_id, "seed": seed, "steps": step,
        "sim_hours": round(sim_hours, 3), "deliveries": deliveries,
        "net_profit": round(net_profit, 2), "hourly_profit": round(hourly, 2),
        "success": success, "aborted": aborted,
    }
    log.info(f"{tag} done  steps={step} sim={sim_hours:.2f}h deliveries={deliveries} "
             f"net=${net_profit:.2f} (${hourly:.2f}/h)")
    async with _log_lock:
        _emit({"event": "rollout_end", **result})
    await env.close()
    return result


# ──────────────────────────────────────────────────────────────────────────────
# Main
# ──────────────────────────────────────────────────────────────────────────────

async def main() -> None:
    client = _make_client()
    seeds = cfg.seeds

    # Self-describing run header: backend, model, frozen model settings, full env
    # config and seed list — so each sweep run's log.jsonl records its own conditions.
    run_meta = {
        "event": "run_meta", "backend": cfg.backend, "model": cfg.model_id,
        "prompt_version": PROMPT_VERSION,
        "run_label": _RUN_LABEL, "run_dir": str(_RUN_DIR),
        "model_settings": {"temperature": cfg.temperature, "top_p": cfg.top_p,
                           "max_tokens": cfg.max_tokens, "request_timeout": cfg.request_timeout,
                           "concurrency": cfg.concurrency, "max_steps": cfg.max_steps},
        "seeds": seeds, "num_rollouts": cfg.num_rollouts,
        "env_config": dataclasses.asdict(ENV_CONFIG),
    }
    _emit(run_meta)
    log.info(f"run_label={_RUN_LABEL}  backend={cfg.backend}  model={cfg.model_id}  "
             f"seeds={seeds}  concurrency={cfg.concurrency}")

    _sem = asyncio.Semaphore(cfg.concurrency)

    async def _bounded(i: int, s: int) -> Dict[str, Any]:
        async with _sem:
            return await run_one_rollout(i, s, client)

    raw = await asyncio.gather(
        *(_bounded(i, seeds[i]) for i in range(len(seeds))),
        return_exceptions=True,
    )
    results = [r for r in raw if isinstance(r, dict)]
    for r in raw:
        if not isinstance(r, dict):
            log.error(f"rollout crashed: {r!r}")

    clean = [r for r in results if not r.get("aborted")]
    profits = [r["hourly_profit"] for r in clean]
    mean = statistics.mean(profits) if profits else 0.0
    std = statistics.pstdev(profits) if len(profits) > 1 else 0.0

    log.info("═" * 60)
    log.info(f"{'rollout':<8}{'seed':<6}{'steps':<7}{'sim_h':<7}{'deliv':<7}{'$/h':<9}{'note'}")
    for r in results:
        note = "ABORTED" if r.get("aborted") else ""
        log.info(f"{r['rollout_id']:<8}{r['seed']:<6}{r['steps']:<7}"
                 f"{r['sim_hours']:<7}{r['deliveries']:<7}{r['hourly_profit']:<9}{note}")
    log.info(f"hourly_profit ({len(clean)} clean rollouts): mean=${mean:.2f}/h  std=${std:.2f}  "
             f"deliveries: {[r['deliveries'] for r in clean]}")
    log.info(f"log dir: {_RUN_DIR}")

    _emit({"event": "final_summary", "results": results,
           "mean_hourly_profit": round(mean, 2), "std_hourly_profit": round(std, 2)})

    # Trajectory metrics (config-selected) over all rollouts. Computed from the
    # flushed log records; emitted as a `metrics` event and logged.
    try:
        from .benchmark import compute_metrics, format_report
        _metrics = compute_metrics(_RUN_DIR, cfg.metrics)
        log.info("─" * 60)
        log.info("trajectory metrics:\n" + format_report(_metrics))
        _emit({"event": "metrics",
               "metrics": _metrics["metrics"],
               "counts": _metrics["counts"],
               "pct": _metrics["pct"],
               "n_trajectories": _metrics["n_trajectories"]})
    except Exception as exc:
        log.warning(f"metrics computation failed: {exc}")

    _log_fh.close()

    # Auto-render the interaction trajectory to HTML (best-effort; never let a
    # viewer error mask the rollout result).
    try:
        from .tools.render_trajectory_html import render as _render_html
        html_path = _render_html(run_dir=str(_RUN_DIR))
        log.info(f"trajectory html: {html_path}")
    except Exception as exc:
        log.warning(f"trajectory html generation failed: {exc}")


if __name__ == "__main__":
    asyncio.run(main())
