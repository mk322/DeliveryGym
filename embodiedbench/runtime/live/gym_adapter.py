"""The courier benchmark behind VAGEN's interface, observed through live UE.

A subclass of ``CourierGymEnv`` that replaces exactly one thing: where the
frames come from. The prompt, the observation text, the image budget, the
lamp-pairing rules, the reward bases and the shaping all stay the inherited
code, because the entire point of the live backend is that a training run
cannot tell it from an album run except by the frames being fresh.

``reset`` is overridden whole rather than hooked, and the reason is worth
recording: the stock ``reset`` builds ``CourierEnv`` inline, entangled with
the album-path defaulting for the baked albums, and offers no
construction seam. Copying its bookkeeping (session, counters, first
observation) is eight lines; threading a factory hook through the stock class
would be an edit to a file this branch is trying not to touch.

``reset``/``step``/``close`` run their (synchronous) bodies on a worker
thread via ``asyncio.to_thread``: verl shares one event loop across every
sample in an AgentLoopWorker, and a blocking HTTP render inside an async
method would freeze all of them for up to the render timeout. See "the async
boundary" below.

Config keys, on top of the inherited ones (``difficulty``, ``stride``,
``embodiment``, ``hazards``, ``max_images``, ``max_turns``, ``city``,
``reward_basis``, ``progress_weight``, ``image_max_side``, ``map_dir``):

=========================  =================================================
``backend``                must be ``"live"``; the key exists so a config
                           that reaches the wrong class fails loudly
``ue_endpoints``           path to endpoints.json (default: $EB_UE_ENDPOINTS)
``live_cache_root``        parent under which this instance creates its own
                           private cache directory; unset, a temporary
                           directory that lives as long as this env object
``obstacle_sidecar_root``  the baked obstacle album directory whose
                           obstacle_visibility.json is copied into each
                           episode's album -- a valid transfer, because
                           obstacle frames use the same street camera pose
                           as the bake; unset, obstacles are silently off,
                           exactly as for a bare album
``signal_sidecar_root``    EXPLICIT OPT-IN, invalid for scored runs: the
                           signal bake certified lens-aimed close-ups the
                           v0 renderer cannot reproduce, so copying its
                           claims can charge red crossings on frames that
                           do not show the lamp. Logs a WARNING when set.
=========================  =================================================

``album_root`` is refused: the live backend renders its own frames, and a
config carrying both would be two sources of truth about one directory.

Launch line (the same CLI registration the stock adapter uses, so nothing in
the gitignored vendor/ checkout changes):

    +env_registry.Courier=embodiedbench.runtime.live.gym_adapter.LiveCourierGymEnv
"""

from __future__ import annotations

import asyncio
import hashlib
import json
import os
import tempfile
import time
import uuid
from pathlib import Path
from typing import Any, Coroutine

from embodiedbench.training.vagen_courier_env import CourierGymEnv

from .embodied_env import (
    ACTION_SPACE_STREET,
    CAMERA_VIEW_STREETS,
    DEFAULT_MAX_STEP_M,
)
from .pool import RenderPool, shared_pool
from .protocol import (
    DEFAULT_ARRIVE_CM,
    DEFAULT_MAX_WALK_SIM_SECONDS,
    DEFAULT_TICK_CHUNK,
)


class LiveCourierGymEnv(CourierGymEnv):
    """One live-rendered courier episode, driven through the training harness."""

    def __init__(self, env_config: dict[str, Any] | None = None):
        config = dict(env_config or {})
        backend = config.get("backend", "live")
        if backend != "live":
            raise ValueError(
                f"LiveCourierGymEnv got backend={backend!r}; a config meant for "
                "the album adapter should name CourierGymEnv, not this class.")
        if config.get("album_root"):
            raise ValueError(
                "album_root has no meaning on the live backend -- frames are "
                "rendered, not read. To reuse a baked album's visibility "
                "claims, pass obstacle_sidecar_root (and, opt-in, "
                "signal_sidecar_root).")
        if config.get("sidecar_source_root"):
            raise ValueError(
                "sidecar_source_root no longer exists: sidecar transfer is "
                "split by validity. Pass obstacle_sidecar_root for the "
                "obstacle claims (they transfer); signal_sidecar_root is a "
                "separate, warned opt-in whose claims do NOT transfer in v0.")
        super().__init__(config)

        self.ue_endpoints = config.get("ue_endpoints")  # None -> $EB_UE_ENDPOINTS
        # Which facts the WORDS may state (none/route/all). Read here and
        # passed to the env, because CourierGymEnv does not thread it: it
        # reached CourierEnv only from the evaluation tooling, so a training
        # config naming a setting would have been silently ignored and the
        # run would have been narration="none" wearing another name.
        self.narration = config.get("narration", "none")
        # Render at the size the policy is actually given, instead of
        # rendering 640x480 and throwing three quarters of it away.
        #
        # Off by default, because it is a real if small change to what the
        # model sees: STREET_CAMERA's 640x480 is the carriageway bake's
        # camera, and a frame rendered at 256x192 is not bit-identical to one
        # rendered at 640x480 and downscaled -- a downscale averages, a direct
        # render aliases. Measured on the development workstation: mean absolute difference 2.13,
        # 2.03% of pixels differing by more than 8. The geometry (4:3, 90 deg
        # FOV) is untouched, so framing is identical; only sampling changes.
        #
        # What it buys, measured on the same instance: 240.8 ms per capture at
        # 640x480 against 161.2 ms at 256x192, and 486 KB of PNG against 75 KB.
        self.capture_at_served_size = bool(config.get("capture_at_served_size", False))
        raw_cache = config.get("live_cache_root")
        self._cache_scratch: tempfile.TemporaryDirectory | None = None
        if raw_cache:
            self.live_cache_root = Path(raw_cache)
        else:
            self._cache_scratch = tempfile.TemporaryDirectory(prefix="courier-live-")
            self.live_cache_root = Path(self._cache_scratch.name)
        # The cache directory is PRIVATE to this adapter instance: a
        # per-process, per-instance unique subdirectory of the configured
        # root. Same-seed resets of this instance reuse it -- that keeps
        # idempotency, and a GRPO group multiplied within one worker reuses
        # its frames -- but nothing else shares it. Cross-worker sharing was
        # considered and rejected: it invites write races and config
        # cross-contamination for zero training benefit (frames are simply
        # re-rendered per worker, and renders are cheap next to rollouts).
        self.live_instance_dir = (
            self.live_cache_root / f"{os.getpid()}-{uuid.uuid4().hex[:8]}")
        self.live_instance_dir.mkdir(parents=True, exist_ok=True)
        def _sidecar_root(key: str) -> Path | None:
            raw = config.get(key)
            root = Path(raw) if raw else None
            if root is not None and not root.exists():
                # The same refusal the stock adapter makes for a missing
                # album: a path that silently degrades to "mechanics off" is
                # how training and evaluation drift apart without anyone
                # deciding they should.
                raise FileNotFoundError(f"{key} does not exist: {root}")
            return root

        self.obstacle_sidecar_root = _sidecar_root("obstacle_sidecar_root")
        self.signal_sidecar_root = _sidecar_root("signal_sidecar_root")
        # Every config axis that changes pixels or keys, folded into a short
        # digest the episode id carries. Seed alone was not an identity:
        # embodiment moves the camera (kerb offset), difficulty/stride change
        # which frames a turn asks for, hazards and the sidecar roots decide
        # which claims exist -- two runs differing on any of these must never
        # share an album directory.
        axes = {
            "difficulty": self.difficulty,
            "stride": self.stride,
            "embodiment": self.embodiment,
            "hazards": self.hazards,
            "obstacle_sidecar_root": (str(self.obstacle_sidecar_root)
                                      if self.obstacle_sidecar_root else None),
            "signal_sidecar_root": (str(self.signal_sidecar_root)
                                    if self.signal_sidecar_root else None),
        }
        self._cfg8 = hashlib.blake2b(
            json.dumps(axes, sort_keys=True).encode("utf-8"),
            digest_size=4).hexdigest()
        # The inherited ``_load_images`` sends the phone map only when
        # ``album_root`` is truthy -- its gate for "this is a sighted
        # condition". Live is sighted, so the gate opens; evaluation sends the
        # map and training must see the same world.
        self.album_root = self.live_instance_dir
        self._render_pool: RenderPool | None = None


    def _street_camera(self) -> Any:
        """The capture spec: the bake's camera, or the size actually served.

        Only the sampling density changes -- 4:3 and 90 degrees are kept, so
        the framing is identical and a frame remains comparable to the bake in
        everything except how finely it was sampled.
        """
        from .env import STREET_CAMERA

        if not self.capture_at_served_size or not self.image_max_side:
            return STREET_CAMERA
        long_edge = int(self.image_max_side)
        short_edge = max(1, round(long_edge * STREET_CAMERA.height
                                  / STREET_CAMERA.width))
        return type(STREET_CAMERA)(width=long_edge, height=short_edge,
                                   fov_deg=STREET_CAMERA.fov_deg)

    def _pool(self) -> RenderPool:
        """Built on first use, not in the constructor: verl instantiates env
        objects before rollout starts, and the endpoints file is written by
        the fleet, which may still be launching at that moment."""
        if self._render_pool is None:
            # Shared per endpoints file: exclusive Track B leases only
            # serialize if every env in this process consults one pool.
            self._render_pool = shared_pool(self.ue_endpoints)
        return self._render_pool

    # ── the async boundary ───────────────────────────────────────────────────
    #
    # verl runs one asyncio loop per AgentLoopWorker with one task per batch
    # sample; every await in every sample shares that loop. The stock adapter
    # bodies are synchronous code in async clothing -- they never actually
    # await -- which was tolerable while the blocking work was a local PIL
    # open (~ms) and becomes a worker-wide freeze when it is an HTTP render
    # with a 120 s timeout. So the sync bodies are driven to completion on a
    # worker thread: the loop stays free to run every other sample's env
    # interaction and generation awaits while this env waits on the wire.
    #
    # Thread safety is not at issue: verl drives one env instance strictly
    # sequentially (reset, then step, then step...), so the env object is
    # only ever touched by one thread at a time.

    @staticmethod
    def _drive(coro: Coroutine[Any, Any, Any]) -> Any:
        """Run a sync-bodied coroutine to completion (stock adapter methods
        never actually await). Loud failure if that assumption ever breaks.

        ``coro.close()`` in the ``finally`` is a no-op after StopIteration
        (the coroutine already finished) and a proper cleanup when the body
        yielded; the placement matters -- a ``finally`` must not swallow the
        returned value, and this one does not.
        """
        try:
            coro.send(None)
        except StopIteration as stop:
            return stop.value
        finally:
            coro.close()
        raise RuntimeError(
            "stock adapter awaited mid-body; thread offload assumption broken")

    async def reset(self, seed: int) -> tuple[dict[str, Any], dict[str, Any]]:
        return await asyncio.to_thread(self._drive, self._reset_impl(seed))

    async def step(self, action_str: str) -> tuple[dict[str, Any], float, bool, dict[str, Any]]:
        return await asyncio.to_thread(self._drive, super().step(action_str))

    async def close(self) -> None:
        # Off-thread too: cleanup deletes the instance's rendered frames,
        # which is real I/O on a big cache.
        return await asyncio.to_thread(self._drive, self._close_impl())

    # ── VAGEN interface ──────────────────────────────────────────────────────

    async def _reset_impl(self, seed: int) -> tuple[dict[str, Any], dict[str, Any]]:
        from embodiedbench.agent.courier.session import CourierSession
        from embodiedbench.compiler.road_network import build_road_network

        from .env import LiveCourierEnv

        if self._network is None:
            self._network = build_road_network(self.map_dir,
                                               map_name=self.map_dir.name)

        kwargs: dict[str, Any] = {
            "seed": int(seed),
            "difficulty": self.difficulty,
            "stride": self.stride,
            "embodiment": self.embodiment,
            "narration": self.narration,
        }
        if self.image_max_side:
            # Same reason as the stock adapter: charge for a red light only
            # where the lamp survives the downscale the policy actually gets.
            kwargs["served_long_edge"] = float(self.image_max_side)

        # The episode id names the album directory inside this instance's
        # private cache dir. The seed makes same-seed resets of this instance
        # one album (idempotency across resets, not merely within one); the
        # cfg8 digest makes runs that differ on any pixel- or key-changing
        # axis different albums, so a rider's centreline frames can never be
        # served to a walker that happens to share the seed.
        episode_id = f"courier-{self.map_dir.name}-s{int(seed)}-{self._cfg8}"
        self._env = LiveCourierEnv(
            self._network,
            self._pool(),
            self._street_camera(),
            episode_id=episode_id,
            cache_root=self.live_instance_dir,
            # hazards=false drops the visibility claims instead of the album
            # kwargs, which is the same lever the stock adapter pulls: no
            # claim, no charge, and the frames stay clean street views. (A
            # reused episode dir cannot smuggle stale claims back in either:
            # LiveAlbum re-syncs the sidecars to these params on open.)
            obstacle_sidecar_root=(self.obstacle_sidecar_root
                                   if self.hazards else None),
            signal_sidecar_root=(self.signal_sidecar_root
                                 if self.hazards else None),
            **kwargs,
        )
        self._env.reset()
        self._episode_opened_at = time.time()
        self._last_seed = int(seed)
        self._session = CourierSession(self._env, city=self.city)
        self._turns = 0
        self._progress_cm = 0.0
        self._last_earnings = 0.0

        obs, dropped = self._observation()
        return obs, {"seed": int(seed), "images_dropped": dropped,
                     "difficulty": self.difficulty, "stride": self.stride,
                     "embodiment": self.embodiment,
                     "backend": "live", "episode_id": episode_id}

    async def _close_impl(self) -> None:
        # Awaiting the parent's sync-bodied coroutine does not suspend, so
        # this whole body still drives to completion in one send.
        await super().close()
        if self._cache_scratch is not None:
            self._cache_scratch.cleanup()
            self._cache_scratch = None


class EmbodiedCourierGymEnv(CourierGymEnv):
    """The Track B backend behind the same training interface.

    Identical adapter shape to ``LiveCourierGymEnv`` -- the same private
    cache-dir discipline, the same config-digested episode id, the same
    worker-thread offload for the async methods -- with two differences that
    are the whole point:

    * the env underneath is ``EmbodiedCourierEnv``, so UE owns locomotion and
      the clock's movement seconds, and the pool lease it takes is exclusive
      (spec 3b: one embodied episode per instance);
    * ``hazards`` must be false. v1 embodied has no obstacle dressing and no
      signal charging -- locomotion realism is the thing under test -- so a
      config asking for hazards is asking for a mode that does not exist yet
      and is refused rather than silently stripped. Unset, it defaults to
      false here (the stock default is true, which would make every embodied
      config carry boilerplate for the only value that works).

    Extra config keys beyond the stock set: ``ue_endpoints``,
    ``live_cache_root`` (both exactly as on the live adapter),
    ``spawn_z_cm`` (where the pawn spawns on the z axis, default 100) and
    ``action_chunk`` (how many calls one turn may carry, default 1).

    **On ``action_chunk``.** This is the backend chunking is *for*: a turn
    here costs 25-105 s of engine time for the walk and a few seconds of
    generation, so the round trip -- and above all the ~5k-token multimodal
    prefill in front of it -- is what the schedule is made of, not the
    inference. Above 1, the episode runs on ``ChunkedCourierSession``: one
    prompt buys up to K actions, they execute in order, and the turn stops at
    the first refusal. The obs/info contract is unchanged; ``chunk_len`` and
    ``chunk_executed`` are added to ``info`` at every K, including 1, so a
    log parser does not have to know which session ran.

    Launch line:

        +env_registry.Courier=embodiedbench.runtime.live.gym_adapter.EmbodiedCourierGymEnv
    """

    def __init__(self, env_config: dict[str, Any] | None = None):
        config = dict(env_config or {})
        backend = config.get("backend", "embodied")
        if backend != "embodied":
            raise ValueError(
                f"EmbodiedCourierGymEnv got backend={backend!r}; a config "
                "meant for the album or live adapter should name its own "
                "class, not this one.")
        if config.get("album_root"):
            raise ValueError(
                "album_root has no meaning on the embodied backend -- frames "
                "come from the pawn's own camera.")
        for key in ("obstacle_sidecar_root", "signal_sidecar_root",
                    "sidecar_source_root"):
            if config.get(key):
                raise ValueError(
                    f"{key} has no meaning on the embodied backend: v1 runs "
                    "hazards off (spec 3b), so there are no visibility "
                    "claims to transfer.")
        # False unless the config says otherwise; a config that says
        # otherwise is refused below, loudly.
        config.setdefault("hazards", False)
        super().__init__(config)
        if self.hazards:
            raise ValueError(
                "hazards=true is not available on the embodied backend: v1 "
                "runs hazards OFF (spec 3b -- obstacle dressing and signal "
                "charging need stateful scene dressing that is reserved, not "
                "built). Drop the key or set it false.")

        self.ue_endpoints = config.get("ue_endpoints")  # None -> $EB_UE_ENDPOINTS
        # Which facts the WORDS may state (none/route/all). Read here and
        # passed to the env, because CourierGymEnv does not thread it: it
        # reached CourierEnv only from the evaluation tooling, so a training
        # config naming a setting would have been silently ignored and the
        # run would have been narration="none" wearing another name.
        self.narration = config.get("narration", "none")
        # Render at the size the policy is actually given, instead of
        # rendering 640x480 and throwing three quarters of it away.
        #
        # Off by default, because it is a real if small change to what the
        # model sees: STREET_CAMERA's 640x480 is the carriageway bake's
        # camera, and a frame rendered at 256x192 is not bit-identical to one
        # rendered at 640x480 and downscaled -- a downscale averages, a direct
        # render aliases. Measured on the development workstation: mean absolute difference 2.13,
        # 2.03% of pixels differing by more than 8. The geometry (4:3, 90 deg
        # FOV) is untouched, so framing is identical; only sampling changes.
        #
        # What it buys, measured on the same instance: 240.8 ms per capture at
        # 640x480 against 161.2 ms at 256x192, and 486 KB of PNG against 75 KB.
        self.capture_at_served_size = bool(config.get("capture_at_served_size", False))
        self.spawn_z_cm = float(config.get("spawn_z_cm", 100.0))
        # Which action space this run measures. "street" is what every
        # embodied run before this one used, so an unset config is the old
        # behaviour exactly; "coordinate" is the harder one under test. The
        # env refuses anything else, and both the telemetry record and the
        # reset info carry the answer -- a pair of runs whose action spaces
        # have to be recalled from a launch command is not a comparison.
        self.action_space = str(config.get("action_space", ACTION_SPACE_STREET))
        self.max_step_m = float(config.get("max_step_m", DEFAULT_MAX_STEP_M))
        # What the turn's photographs are OF. "streets" is the default and
        # the comparable one; "forward" shows what is in front of the courier
        # instead of one frame per street, and is a second axis rather than a
        # free improvement -- it changes what an arm SEES, so a run using it
        # is not comparable to one that does not.
        self.camera_view = str(config.get("camera_view", CAMERA_VIEW_STREETS))
        # Whether the observation states the courier's own coordinates.
        # Defaulted by the env to on under the coordinate space (where the
        # task is unanswerable without it) and off under the street one. Set
        # it explicitly for the controlled comparison: street space with the
        # pose shown isolates the action space from the extra fact.
        self.show_pose = config.get("show_pose")
        # Whether a dead renderer may finish an episode on cached frames.
        # The env has had this knob since the cross-machine work and the
        # quickstart's own settings table says an experiment must turn it
        # off -- but no config key reached it, so the only value any run
        # could have was the training-friendly default. Measured the hard
        # way on 2026-08-13: one of three instances died mid-validation with
        # UE's malloc crash, and the episodes it was serving carried on
        # against an album with every walking metric still reading green.
        self.allow_album_fallback = bool(
            config.get("allow_album_fallback", True))
        self.action_chunk = int(config.get("action_chunk", 1))
        if self.action_chunk < 1:
            raise ValueError(
                "action_chunk is how many calls one turn may carry, so it is "
                f"at least 1; got {config.get('action_chunk')!r}")
        #: Increments per reset so no two live episodes share an id.
        self._episode_seq = 0
        # Stamped at reset, read by the telemetry record on the way out.
        self._episode_opened_at: float | None = None
        self._last_seed: int | None = None
        # Walk geometry. The arrival radius must not be finer than the
        # distance the pawn covers between two arrival checks: at
        # fixed_dt 0.2 and 140 cm/s a tick is 28 cm, so a chunk of 10 moves
        # 280 cm between looks and a 50 cm radius is invisible -- the courier
        # sails past its node and the walk reports stuck. Callers that raise
        # the engine's dt must raise these together, so they are config.
        self.tick_chunk = int(config.get("tick_chunk", DEFAULT_TICK_CHUNK))
        self.arrive_cm = float(config.get("arrive_cm", DEFAULT_ARRIVE_CM))
        self.max_walk_seconds = float(
            config.get("max_walk_seconds", DEFAULT_MAX_WALK_SIM_SECONDS))
        raw_cache = config.get("live_cache_root")
        self._cache_scratch: tempfile.TemporaryDirectory | None = None
        if raw_cache:
            self.live_cache_root = Path(raw_cache)
        else:
            self._cache_scratch = tempfile.TemporaryDirectory(
                prefix="courier-embodied-")
            self.live_cache_root = Path(self._cache_scratch.name)
        # Private per-(pid, instance) dir, for the same reasons as the live
        # adapter: idempotent same-seed resets, zero cross-worker sharing.
        self.live_instance_dir = (
            self.live_cache_root / f"{os.getpid()}-{uuid.uuid4().hex[:8]}")
        self.live_instance_dir.mkdir(parents=True, exist_ok=True)
        # Every axis that changes pixels or keys -- plus the backend itself,
        # so an embodied album can never collide with a live one that
        # happens to share every other axis.
        axes = {
            "backend": "embodied",
            "difficulty": self.difficulty,
            "stride": self.stride,
            "embodiment": self.embodiment,
            "spawn_z_cm": self.spawn_z_cm,
            # The action space changes both the frame keys and where the pawn
            # stands when a frame is taken, so two runs that differ only in it
            # must not share a cache directory.
            "action_space": self.action_space,
            "max_step_m": self.max_step_m,
            "camera_view": self.camera_view,
            "show_pose": self.show_pose,
            "allow_album_fallback": self.allow_album_fallback,
        }
        self._cfg8 = hashlib.blake2b(
            json.dumps(axes, sort_keys=True).encode("utf-8"),
            digest_size=4).hexdigest()
        # Embodied is sighted: open the stock adapter's phone-map gate.
        self.album_root = self.live_instance_dir
        self._render_pool: RenderPool | None = None

    _drive = staticmethod(LiveCourierGymEnv._drive)


    def _street_camera(self) -> Any:
        """The capture spec: the bake's camera, or the size actually served.

        Only the sampling density changes -- 4:3 and 90 degrees are kept, so
        the framing is identical and a frame remains comparable to the bake in
        everything except how finely it was sampled.
        """
        from .env import STREET_CAMERA

        if not self.capture_at_served_size or not self.image_max_side:
            return STREET_CAMERA
        long_edge = int(self.image_max_side)
        short_edge = max(1, round(long_edge * STREET_CAMERA.height
                                  / STREET_CAMERA.width))
        return type(STREET_CAMERA)(width=long_edge, height=short_edge,
                                   fov_deg=STREET_CAMERA.fov_deg)

    def _pool(self) -> RenderPool:
        """Lazy for the same reason as the live adapter: the endpoints file
        is written by the fleet, which may still be launching."""
        if self._render_pool is None:
            # Shared per endpoints file: exclusive Track B leases only
            # serialize if every env in this process consults one pool.
            self._render_pool = shared_pool(self.ue_endpoints)
        return self._render_pool

    # ── the async boundary (same offload as LiveCourierGymEnv) ───────────────

    async def reset(self, seed: int) -> tuple[dict[str, Any], dict[str, Any]]:
        return await asyncio.to_thread(self._drive, self._reset_impl(seed))

    async def step(self, action_str: str) -> tuple[dict[str, Any], float, bool, dict[str, Any]]:
        return await asyncio.to_thread(self._drive, self._step_impl(action_str))

    async def close(self) -> None:
        return await asyncio.to_thread(self._drive, self._close_impl())

    # ── VAGEN interface ──────────────────────────────────────────────────────

    async def _step_impl(self, action_str: str) -> tuple[dict[str, Any], float, bool, dict[str, Any]]:
        """The stock turn, plus what the chunk did with it.

        Two keys, added and never substituted: the contract a trainer reads is
        the inherited one, and a run that cannot tell how many of its calls
        actually happened cannot tell a policy that chains well from one that
        chains hopefully and loses the tail of every turn.
        """
        from embodiedbench.agent.courier.chunk import chunk_info

        obs, reward, done, info = await super().step(action_str)
        turns = self._session.run.turns if self._session is not None else []
        last = turns[-1] if turns else None
        info.update(chunk_info(last))
        info.update(self._live_health())
        if getattr(self, "_trace", None) is not None and last is not None:
            self._trace.record(self._env, last)
        return obs, reward, done, info

    def _live_health(self) -> dict[str, Any]:
        """What the SIMULATOR did to this turn, in the trainer's own dict.

        Both of these already existed inside the env and neither reached a
        consumer. ``degraded`` means the episode stopped receiving live
        frames and fell back to whatever the album held -- a change of
        observation distribution mid-rollout that ``images_dropped`` does not
        report, because from its point of view the world simply offered fewer
        photographs. ``sim_failures`` counts hops that ended in stuck or
        timeout: at block stride those are absorbed into the macro's prose
        and the turn arrives as accepted with error None, so this is the only
        channel that distinguishes "the policy chose badly" from "the navmesh
        could not walk there". A run that cannot separate those two is
        training on the map's defects.
        """
        env = self._env
        if env is None:
            return {}
        hops = [h for h in getattr(env, "embodied_log", []) if "ticks" in h]
        return {
            "live_degraded": bool(getattr(env, "live_degraded", False)),
            "sim_failures": sum(
                1 for h in hops if h.get("outcome") in ("stuck", "timeout")),
            "sim_hops": len(hops),
        }

    async def _reset_impl(self, seed: int) -> tuple[dict[str, Any], dict[str, Any]]:
        from embodiedbench.agent.courier.chunk import ChunkedCourierSession
        from embodiedbench.agent.courier.session import CourierSession
        from embodiedbench.compiler.road_network import build_road_network

        from .embodied_env import EmbodiedCourierEnv

        if self._network is None:
            self._network = build_road_network(self.map_dir,
                                               map_name=self.map_dir.name)
        # A new reset means a new env object, and the old one holds an
        # exclusive lease: give it back first or a reset storm starves the
        # fleet one instance per reset. Record it on the way out -- this and
        # _close_impl are the only two places an episode ends, and an episode
        # that ends unrecorded is one nobody can audit afterwards.
        if self._env is not None:
            self._flush_telemetry()
            self._env.close()

        kwargs: dict[str, Any] = {
            "seed": int(seed),
            "difficulty": self.difficulty,
            "stride": self.stride,
            "embodiment": self.embodiment,
            "narration": self.narration,
        }
        if self.image_max_side:
            kwargs["served_long_edge"] = float(self.image_max_side)
        # Unset means "let the action space decide", which is the env's own
        # default. Passing None through would override that decision with a
        # falsy value and leave a coordinate courier unable to answer.
        if self.show_pose is not None:
            kwargs["show_pose"] = bool(self.show_pose)

        # Unique per EPISODE, not per (seed, config). GRPO's group runs the
        # same seed n times concurrently -- that is where its advantage comes
        # from -- so a seed-derived id makes two live episodes collide on one
        # instance, and the service's idempotent same-id re-spawn then has
        # them tearing down each other's pawn. Measured on the development workstation: four
        # spawns at one position inside 1.1 s, and the wall clock going to
        # re-spawns instead of walking.
        self._episode_seq += 1
        episode_id = (f"courier-{self.map_dir.name}-s{int(seed)}-{self._cfg8}"
                      f"-{self.live_instance_dir.name}-{self._episode_seq}")
        self._env = EmbodiedCourierEnv(
            self._network,
            self._pool(),
            self._street_camera(),
            episode_id=episode_id,
            cache_root=self.live_instance_dir,
            spawn_z_cm=self.spawn_z_cm,
            arrive_cm=self.arrive_cm,
            tick_chunk=self.tick_chunk,
            max_walk_seconds=self.max_walk_seconds,
            action_space=self.action_space,
            max_step_m=self.max_step_m,
            camera_view=self.camera_view,
            allow_album_fallback=self.allow_album_fallback,
            **kwargs,
        )
        self._env.reset()
        self._episode_opened_at = time.time()
        self._last_seed = int(seed)
        self._trace = self._open_trace(episode_id)
        # The chunked session only where chunking was asked for: at K=1 it is
        # the stock session by delegation anyway, and building the stock one
        # keeps "chunking off" and "chunking never installed" the same run.
        self._session = (
            ChunkedCourierSession(self._env, city=self.city,
                                  action_chunk=self.action_chunk)
            if self.action_chunk > 1 else
            CourierSession(self._env, city=self.city))
        self._turns = 0
        self._progress_cm = 0.0
        self._last_earnings = 0.0

        obs, dropped = self._observation()
        return obs, {"seed": int(seed), "images_dropped": dropped,
                     "difficulty": self.difficulty, "stride": self.stride,
                     "embodiment": self.embodiment,
                     "action_chunk": self.action_chunk,
                     "action_space": self.action_space,
                     "backend": "embodied", "episode_id": episode_id}

    def _open_trace(self, episode_id: str) -> Any:
        """A per-call trace, when one was asked for. See ``trace.py``."""
        from .trace import EpisodeTrace, trace_dir

        root = trace_dir()
        if root is None:
            return None
        return EpisodeTrace(root, episode_id, {
            "seed": self._last_seed, "map": self.map_dir.name,
            "action_space": self.action_space, "max_step_m": self.max_step_m,
            "camera_view": self.camera_view,
            "arrive_cm": self.arrive_cm, "tick_chunk": self.tick_chunk,
            "action_chunk": self.action_chunk, "narration": self.narration,
            "difficulty": self.difficulty, "stride": self.stride,
        })

    def _flush_telemetry(self) -> None:
        """Write the finished episode's record. Never raises."""
        from .telemetry import record_episode

        if getattr(self, "_trace", None) is not None and self._env is not None:
            self._trace.close(self._env)
            self._trace = None

        record_episode(
            self._env,
            cache_root=self.live_instance_dir,
            opened_at=self._episode_opened_at,
            turns=(self._session.run.turns
                   if self._session is not None else None),
            config={
                "seed": self._last_seed,
                "map": self.map_dir.name,
                "difficulty": self.difficulty,
                "stride": self.stride,
                "embodiment": self.embodiment,
                "action_chunk": self.action_chunk,
                "arrive_cm": self.arrive_cm,
                "tick_chunk": self.tick_chunk,
                "max_walk_seconds": self.max_walk_seconds,
                "image_max_side": self.image_max_side,
                "backend": "embodied",
                # Which action space, and the cap that bounds one call of it.
                # In the record because the comparison this run exists for is
                # between two files of these, read weeks apart.
                "action_space": self.action_space,
                "max_step_m": self.max_step_m,
                "camera_view": self.camera_view,
                "allow_album_fallback": self.allow_album_fallback,
                "show_pose": self.show_pose,
                "narration": self.narration,
            },
        )

    async def _close_impl(self) -> None:
        # End the embodied episode -- despawn, lease back -- before the stock
        # cleanup nulls the reference to it.
        if self._env is not None:
            self._flush_telemetry()
            self._env.close()
        await super().close()
        if self._cache_scratch is not None:
            self._cache_scratch.cleanup()
            self._cache_scratch = None
