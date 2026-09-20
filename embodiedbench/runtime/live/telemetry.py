"""One durable line per finished episode, so the run can be audited later.

The env already gathers everything worth knowing -- ``embodied_log`` holds a
dict per ``/walk``, ``summary()`` aggregates it -- and then throws it away when
the episode object is dropped. That is not a small gap. Measured on the development workstation on
2026-08-11: a live GRPO run had been walking Paris for hours, and a grep across
every Ray worker log for the episode id, for "walk", for "observe", returned
nothing at all. The rollouts were real and the evidence was not written down.

Nothing here fights Python's logging config, and that is deliberate. These
episodes run inside Ray actor processes whose handlers the trainer owns and
reconfigures; a ``logger.info`` from an env is at the mercy of that. A file
descriptor is not. One O_APPEND write of one line under PIPE_BUF is atomic on
Linux, so every worker process appends to its own file and no reader ever sees
a torn record.

Where it lands, in order of precedence: ``EB_LIVE_TELEMETRY_DIR``, else a
``_telemetry`` sibling of the per-instance cache dir. A sibling, never a child:
the cache dir is a TemporaryDirectory in the default configuration and takes
its contents down with it at episode end, which is precisely when this is
written.

Failure to record is never allowed to fail a rollout. An unwritable directory
costs the audit, not the run -- but it says so, once, at WARNING.
"""

from __future__ import annotations

import json
import logging
import os
import time
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)

#: Overrides the default location.
TELEMETRY_DIR_ENV = "EB_LIVE_TELEMETRY_DIR"
SCHEMA = "nav-live-telemetry/v1"

#: Warn once per process, not once per episode: a broken telemetry dir would
#: otherwise print a line per episode for the length of the run.
_warned = False


def telemetry_dir(cache_root: Path | str | None) -> Path | None:
    """Resolve the directory, or ``None`` when there is nowhere to write."""
    explicit = os.environ.get(TELEMETRY_DIR_ENV)
    if explicit:
        return Path(explicit)
    if cache_root is None:
        return None
    return Path(cache_root).parent / "_telemetry"


#: Model replies are capped rather than dropped. Whole replies would dominate
#: the file (the prompt alone carries an observation), and dropping them costs
#: the only record of what the policy actually SAID -- which is the question
#: "is the agent even running" reduces to, and it took a manual dig through a
#: frame cache to answer it once.
REPLY_CHARS = 1500


def record_episode(
    env: Any,
    *,
    cache_root: Path | str | None,
    config: dict[str, Any] | None = None,
    opened_at: float | None = None,
    turns: Any = None,
) -> Path | None:
    """Append one episode's record. Returns the file written, or ``None``.

    ``env`` is duck-typed on purpose: Track A's env has no ``embodied_log`` and
    Track B's has no ``live_album`` in every configuration, and an audit that
    only works when every optional attribute is present is an audit that stops
    working the first time the env is subclassed.
    """
    global _warned
    directory = telemetry_dir(cache_root)
    if directory is None:
        return None
    try:
        directory.mkdir(parents=True, exist_ok=True)
        record = _build(env, config=config, opened_at=opened_at,
                        turns=turns)
        path = directory / f"episodes-{os.getpid()}.jsonl"
        line = json.dumps(record, default=str, ensure_ascii=False) + "\n"
        # O_APPEND: concurrent writers in one process cannot interleave a
        # single write, and separate processes have separate files anyway.
        fd = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_APPEND, 0o644)
        try:
            os.write(fd, line.encode("utf-8"))
        finally:
            os.close(fd)
        return path
    except Exception as error:  # noqa: BLE001 — an audit never fails a rollout
        if not _warned:
            _warned = True
            logger.warning(
                "episode telemetry could not be written to %s (%s: %s); the "
                "run continues unrecorded", directory, type(error).__name__,
                error)
        return None


def _turn_rows(turns: Any) -> list[dict[str, Any]]:
    """What the policy said, and what became of it.

    The prompt is summarised rather than stored: it carries a whole
    observation and would swamp the record. The reply is kept, capped,
    because it is the evidence layer nothing else has -- the hops say where
    the courier went, and only this says whether a model chose it.
    """
    rows: list[dict[str, Any]] = []
    for turn in list(turns or []):
        reply = str(getattr(turn, "reply", "") or "")
        rows.append({
            "step": getattr(turn, "step", None),
            "status": getattr(turn, "status", None),
            "error": getattr(turn, "error", None),
            "prompt_chars": len(str(getattr(turn, "prompt", "") or "")),
            "images": len(getattr(turn, "image_paths", []) or []),
            "reply_chars": len(reply),
            "reply": reply[:REPLY_CHARS],
            "reply_truncated_in_log": len(reply) > REPLY_CHARS,
        })
    return rows


def _build(
    env: Any,
    *,
    config: dict[str, Any] | None,
    opened_at: float | None,
    turns: Any = None,
) -> dict[str, Any]:
    now = time.time()
    hops = list(getattr(env, "embodied_log", []) or [])
    walks = [h for h in hops if "ticks" in h]
    outcomes: dict[str, int] = {}
    for hop in walks:
        key = str(hop.get("outcome", "unknown"))
        outcomes[key] = outcomes.get(key, 0) + 1

    try:
        summary = env.summary()
    except Exception as error:  # noqa: BLE001 — a partial record beats none
        summary = {"summary_failed": f"{type(error).__name__}: {error}"}

    return {
        "schema": SCHEMA,
        "episode_id": getattr(env, "episode_id", None),
        "pid": os.getpid(),
        "closed_at": now,
        "wall_seconds": round(now - opened_at, 3) if opened_at else None,
        "config": config or {},
        "fixed_dt": getattr(env, "fixed_dt", None),
        # The counters an auditor reads first: every one of them is a way the
        # SIMULATOR, not the policy, can have shaped this episode.
        "counters": {
            "hops": len(walks),
            "outcomes": outcomes,
            "recoveries": sum(1 for h in hops if "recovery" in h),
            "degraded": bool(getattr(env, "live_degraded", False)),
            "busy_waits": int(getattr(env, "busy_waits", 0)),
            "busy_wait_seconds": round(
                float(getattr(env, "busy_wait_seconds", 0.0)), 3),
        },
        "summary": summary,
        "turns": _turn_rows(turns),
        "hops": hops,
    }
