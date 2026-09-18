"""One event per CALL, with the picture the courier was looking at.

``telemetry.py`` writes one line per episode and is the right grain for
"how did the fleet behave over 248 episodes". It is the wrong grain for the
question this exists to answer, which is "what did the courier see, what did
it decide, and what did the world do about it" -- and that question is asked
one call at a time.

The difference matters most for a chunked turn. One reply names up to K
waypoints; the second and third are named relative to positions the courier
has not walked to yet, and the world's answer to each is joined into a single
feedback paragraph before the turn log sees it. Read at turn grain, a chunk is
one action with one outcome. Read here, it is three, each with the position it
was judged from.

What a row carries:

* the turn's observation -- the text and the frames, copied into the trace so
  they outlive the episode's temporary cache directory (which is a
  ``TemporaryDirectory`` in the default configuration and takes the pictures
  down with it at episode end, which is exactly when this is written);
* the model's reply, whole;
* every call the reply named, in order, with the position it ran from, what
  the world said back, and whether it ran at all;
* the walk, where there was one: ticks, engine seconds, metres, the pose it
  ended at, and how far the graph node being described sits from that pose.

Off unless ``EB_LIVE_TRACE_DIR`` is set. It writes a frame per look and a row
per call, which is the right cost for a diagnostic run and the wrong one for
a training job.

Failure to record never fails a rollout: an unwritable directory costs the
trace, not the run, and says so once at WARNING.
"""

from __future__ import annotations

import json
import logging
import os
import shutil
import time
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)

#: Where to write. Unset, nothing is recorded.
TRACE_DIR_ENV = "EB_LIVE_TRACE_DIR"
SCHEMA = "nav-live-trace/v1"

_warned = False


def trace_dir() -> Path | None:
    raw = os.environ.get(TRACE_DIR_ENV)
    return Path(raw) if raw else None


def _rows_for(turn: Any) -> list[dict[str, Any]]:
    """The calls a turn made, chunked or not, in one shape.

    A stock single-call turn has no ``chunk``; it is a chunk of one, and
    synthesising the row here is what lets a reader of the trace never ask
    which session produced it.
    """
    rows = list(getattr(turn, "chunk", None) or [])
    if rows:
        return rows
    action = getattr(turn, "action", "") or ""
    if not action:
        return []
    return [{
        "action": action,
        "status": getattr(turn, "status", None),
        "code": getattr(turn, "error", "") or "",
        "sim_seconds": getattr(turn, "sim_seconds", 0.0),
        "reward": getattr(turn, "reward", 0.0),
        "from_xy": getattr(turn, "from_xy", None),
        "message": getattr(turn, "feedback", "") or "",
    }]


class EpisodeTrace:
    """Accumulates one episode's events and writes them at the end.

    Held in memory rather than appended live for one reason: the frames have
    to be copied while the episode's cache still exists, and the walk detail
    for a turn is only complete once the turn has finished. Episodes are tens
    of turns, so the memory is not the constraint.
    """

    def __init__(self, root: Path, episode_id: str, config: dict[str, Any]):
        self.root = Path(root) / episode_id
        self.frames = self.root / "frames"
        self.episode_id = episode_id
        self.config = dict(config or {})
        self.events: list[dict[str, Any]] = []
        self.opened_at = time.time()
        self._hops_seen = 0
        self._frames_seen: dict[str, str] = {}

    # ── one turn ─────────────────────────────────────────────────────────────

    def record(self, env: Any, turn: Any, observation: Any = None) -> None:
        """One turn: what it was shown, what it said, what each call did."""
        try:
            self._record(env, turn, observation)
        except Exception as error:  # noqa: BLE001 — a trace never fails a turn
            _warn_once("could not record a turn", error)

    def _record(self, env: Any, turn: Any, observation: Any) -> None:
        # The walks this turn produced: everything the env logged since the
        # last time we looked. Joined to calls by order, which is the order
        # they happened in.
        log = list(getattr(env, "embodied_log", []) or [])
        fresh = log[self._hops_seen:]
        self._hops_seen = len(log)
        walks = [h for h in fresh if "ticks" in h or "kind" in h]

        rows = _rows_for(turn)
        wi = 0
        calls: list[dict[str, Any]] = []
        for index, row in enumerate(rows):
            call = {
                "index": index,
                "action": row.get("action"),
                "status": row.get("status"),
                "code": row.get("code") or "",
                "from_xy_m": _metres(row.get("from_xy")),
                "sim_seconds": round(float(row.get("sim_seconds") or 0.0), 3),
                "reward": row.get("reward"),
                "feedback": row.get("message") or "",
                "walk": None,
            }
            # A dropped call never reached the world, so it consumed no walk.
            if row.get("status") != "dropped" and wi < len(walks):
                call["walk"] = _walk(walks[wi])
                wi += 1
            calls.append(call)

        # WHERE THE PICTURES WERE TAKEN, which is not where the turn ended.
        #
        # `session.step` observes first and acts second, so a turn's frames are
        # rendered at the pose BEFORE its calls run, while this method is
        # called after them. Recorded as one "pose" they read as the same
        # place, and a playback showing the two together has the camera
        # trailing the position by one step -- which looks, watching it, like
        # the courier walking backwards.
        #
        # The first call's own `from_xy` is that pose exactly. A turn that made
        # no call (a format error) moved nothing, so the previous event's
        # after-pose still stands.
        after = _metres(_pose(env))
        before = calls[0]["from_xy_m"] if calls and calls[0]["from_xy_m"] else None
        if before is None:
            before = self.events[-1]["pose_after_m"] if self.events else after
        self.events.append({
            "turn": getattr(turn, "step", None),
            "at": round(time.time() - self.opened_at, 3),
            "sim_seconds": round(float(getattr(env, "sim_seconds", 0.0) or 0.0), 2),
            "node": getattr(env, "node_id", None),
            # The frames belong to `pose_before_m`; everything the calls did
            # belongs between the two.
            "pose_before_m": before,
            "pose_after_m": after,
            # Three orientations, and they are three different facts:
            # `facing` is the way the courier travelled (what the candidate
            # list's left/right come from), `yaw_after` is where the body
            # points now, and each frame below carries the yaw it was shot at.
            "facing_deg": _facing(env),
            "yaw_after_deg": _yaw(env),
            "observation": getattr(turn, "prompt", "") or "",
            "frames": self._keep(getattr(turn, "image_paths", None) or []),
            # Which way each photograph looks, in the same order as `frames`.
            # Carried on the turn rather than looked up here: the harness
            # renames frames on the way out, so the album path the yaw was
            # recorded against is not the path this sees.
            "frame_yaws_deg": list(getattr(turn, "frame_yaws", None) or []),
            "reply": getattr(turn, "reply", "") or "",
            "status": getattr(turn, "status", None),
            "error": getattr(turn, "error", None),
            "calls": calls,
        })

    def _keep(self, paths: list[str]) -> list[str]:
        """Copy the frames out of the episode cache, which is about to vanish.

        Deduplicated by source path: a junction photographed once and shown on
        three turns is one file and three references to it.
        """
        kept = []
        for raw in paths:
            if not raw:
                continue
            if raw in self._frames_seen:
                kept.append(self._frames_seen[raw])
                continue
            source = Path(raw)
            if not source.exists():
                continue
            self.frames.mkdir(parents=True, exist_ok=True)
            name = f"{len(self._frames_seen):04d}_{source.name}"
            try:
                shutil.copyfile(source, self.frames / name)
            except OSError as error:
                _warn_once("could not copy a frame", error)
                continue
            self._frames_seen[raw] = f"frames/{name}"
            kept.append(f"frames/{name}")
        return kept

    # ── the end ──────────────────────────────────────────────────────────────

    def close(self, env: Any) -> Path | None:
        """Write the episode. Returns the file, or ``None`` if it could not."""
        try:
            self.root.mkdir(parents=True, exist_ok=True)
            summary: Any
            try:
                summary = env.summary()
            except Exception as error:  # noqa: BLE001 — a partial trace beats none
                summary = {"summary_failed": f"{type(error).__name__}: {error}"}
            path = self.root / "trace.json"
            path.write_text(json.dumps({
                "schema": SCHEMA,
                "episode_id": self.episode_id,
                "config": self.config,
                "opened_at": self.opened_at,
                "wall_seconds": round(time.time() - self.opened_at, 3),
                "summary": summary,
                "events": self.events,
            }, default=str, ensure_ascii=False, indent=1), encoding="utf-8")
            return path
        except Exception as error:  # noqa: BLE001
            _warn_once("could not write the trace", error)
            return None


def _warn_once(what: str, error: Exception) -> None:
    global _warned
    if not _warned:
        _warned = True
        logger.warning("trace: %s (%s: %s); the run continues untraced",
                       what, type(error).__name__, error)


def _pose(env: Any) -> tuple[float, float] | None:
    try:
        return tuple(env.position())
    except Exception:  # noqa: BLE001
        return None


def _yaw(env: Any) -> float | None:
    """The pawn's OWN yaw, straight off the last pose the engine returned.

    Not the same thing as ``facing`` and recorded beside it on purpose.
    ``facing`` is the direction the courier travelled, which is what every
    "on your left" in the observation is derived from. This is where the body
    is actually pointing -- and because ``/observe`` aims the camera by
    turning the pawn, after a look it is the bearing of the last photograph
    taken rather than the way the courier was walking. Keeping both is what
    lets a reader tell those two apart instead of assuming they agree.
    """
    pose = getattr(env, "ue_pose", None)
    if pose is None:
        return None
    try:
        return round(float(pose.yaw_deg) % 360.0, 1)
    except Exception:  # noqa: BLE001
        return None


def _facing(env: Any) -> float | None:
    try:
        value = env.facing()
    except Exception:  # noqa: BLE001
        return None
    return round(float(value), 1) if value is not None else None


def _metres(xy: Any) -> list[float] | None:
    """Centimetres to metres, at the precision the courier is told them in."""
    if not xy:
        return None
    return [round(float(v) / 100.0, 2) for v in xy]


def _walk(hop: dict[str, Any]) -> dict[str, Any]:
    """One walk, or one request that never became a walk, in metres."""
    if "ticks" not in hop:
        return {"kind": hop.get("kind"), "code": hop.get("code"),
                "asked_m": _metres(hop.get("asked_xy")),
                "from_m": _metres(hop.get("from_xy")),
                "gap_m": hop.get("gap_m")}
    return {
        "kind": hop.get("kind", "hop"),
        "outcome": hop.get("outcome"),
        "asked_m": _metres(hop.get("asked_xy")),
        "target_m": _metres(hop.get("target_xy")),
        "end_m": _metres((hop["end_pose"]["x_cm"], hop["end_pose"]["y_cm"]))
                 if hop.get("end_pose") else None,
        # Where the body pointed when the walk stopped: a navmesh route round
        # a corner leaves it facing along the last leg, not at what it named.
        "end_yaw_deg": (round(float(hop["end_pose"]["yaw_deg"]) % 360.0, 1)
                        if hop.get("end_pose") else None),
        "walked_m": round(float(hop.get("walked_cm") or 0.0) / 100.0, 2),
        "ticks": hop.get("ticks"),
        "sim_seconds": hop.get("sim_seconds"),
        "snap_m": round(float(hop["snap_cm"]) / 100.0, 2) if "snap_cm" in hop else None,
        "landed_node": hop.get("landed_node") or hop.get("target_node"),
    }
