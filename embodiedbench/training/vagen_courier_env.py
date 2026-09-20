"""The courier benchmark, behind VAGEN's ``GymImageEnv`` interface.

Why this file exists at all: the RL machinery worth training with -- PPO, GRPO,
a real rollout engine, ray -- already exists in VAGEN (`ymzhang0303/VAGEN`,
vendored at ``vendor/vagen``). What it could not see was *this* benchmark.
VAGEN ships its own ``DeliveryBench``, but that is the older environment with a
different action space, different observations and none of the fixes in this
repo. So rather than reimplement PPO, this adapts the environment.

The contract is four async methods and one observation shape:

    obs = {"obs_str": "... <image> ...",
           "multi_modal_input": {"<image>": [PIL.Image, ...]}}

with one ``<image>`` placeholder per image, in the order the images appear.

Three things this deliberately does *not* do:

* **It does not re-render the prompt.** ``CourierSession`` builds the system
  prompt and the per-turn observation, and this returns them unchanged. A
  training-only prompt would optimise a policy for text it is never scored on,
  which is the single easiest way to produce a number that does not transfer.
* **It does not shape the reward by default.** ``step`` returns the turn reward
  the harness charged, and ``info["env_return"]`` is always the unshaped
  episode return the benchmark scores. ``progress_weight`` can add a dense
  term, but it is off unless asked for and it never touches ``env_return``.
* **It does not silently drop images.** A turn can offer nine pictures. The cap
  is a config value and the number dropped is reported in ``info``, because a
  quietly truncated observation looks identical to a policy that ignored what
  it was shown.

The class is importable without ray, verl, or a GPU, so the whole environment
side can be tested on its own -- which is what ``tests/test_vagen_courier.py``
does.
"""

from __future__ import annotations

import os
import re
import tempfile
from pathlib import Path
from typing import Any

from embodiedbench.training.image_groups import (
    required_image_groups,
    validate_atomic_image_capacity,
)

REPO = Path(__file__).resolve().parents[2]
DEFAULT_MAP = REPO / "vendor/vagen/vagen/envs/deliverybench/maps/citycore-paris"


def _albums_base(hint: Path | None = None) -> Path:
    """The directory the photograph albums sit in on *this* machine.

    The albums are the one asset whose location genuinely differs between
    machines. Writing either
    into a config makes that config wrong on the other machine, and it fails
    a long way from the config -- inside reset(), on the first episode, after
    the model has loaded and the engine has warmed.

    Resolved in order of how much the source knows:

        ALBUMS_DIR          set deliberately, so it wins
        <album_root>/../..  a sibling of whatever album the config named,
                            which is right by construction wherever they were
                            unpacked together

    Nothing else: a machine that has neither must say where the albums are,
    and the reset that follows names the path it could not find.
    """
    env = os.environ.get("ALBUMS_DIR")
    if env:
        return Path(env)
    if hint is not None and hint.parent.parent.is_dir():
        return hint.parent.parent
    return Path(hint.parent.parent) if hint is not None else Path("albums")


def _resolve_album(path: Path | None) -> Path | None:
    """A configured album path, re-rooted if it is not on this machine.

    The last two components -- <album name>/<map name> -- are what identifies
    an album; everything before them is where that machine keeps them. So a
    config saying /data/albums/paris_streets_v2/citycore-paris still means
    the right thing on a node that unpacked the same album somewhere else.
    """
    if path is None or path.exists():
        return path
    if len(path.parts) < 2:
        return path
    relocated = _albums_base() / path.parent.name / path.name
    return relocated if relocated.exists() else path
IMAGE_PLACEHOLDER = "<image>"
# One "unit" of shaped progress. A block on this map is 60-110 m, so 100 m
# makes a good block worth about a tenth of a delivery.
PROGRESS_SCALE_CM = 10_000.0


def _replace_photo_caption(text: str, labels: list[str]) -> str:
    """Write the photographs block from the frames that were actually sent.

    Built from the frames' own labels rather than by editing the harness's
    sentence, because that sentence changes shape: three street views at a
    plain junction, street views plus "[light k]" lines at a signalised one,
    plus the phone map once a route has been asked for. A pattern that handles
    the first quietly destroys the others.
    """
    match = re.search(r"(### photographs\n)(.*?)(\n###|\Z)", text, re.S)
    if match is None:
        return text
    body = ("  (no photographs this turn)" if not labels
            else "\n".join(f"  {label}" for label in labels))
    return text[:match.start(2)] + body + text[match.end(2):]


def _base_class() -> type:
    """VAGEN's base class if it is importable, else a stand-in.

    The adapter is useful without VAGEN on the path -- the tests run it that
    way -- but when VAGEN *is* present it must be a real subclass, because the
    agent loop checks the interface it got.
    """
    try:
        from vagen.envs.gym_image_env import GymImageEnv

        return GymImageEnv
    except Exception:  # noqa: BLE001 - VAGEN is optional at import time

        class _Standalone:
            def __init__(self, env_config: dict[str, Any]):
                self.config = env_config

        return _Standalone


class CourierGymEnv(_base_class()):  # type: ignore[misc]
    """One courier episode, driven through the same harness evaluation uses.

    Config keys, all optional:

    ==================  =======================================================
    ``map_dir``         road network to load (default: the vendored Paris map)
    ``album_root``      street photographs; without it the episode is text-only
    ``difficulty``      ``solo``/``pair``/``triple``/``shift``/``endless``
    ``stride``          ``block`` or ``waypoint``
    ``embodiment``      ``human_on_foot``/``human_on_scooter``/``human_in_car``
    ``hazards``         lights and obstacles on (default true)
    ``max_images``      per-turn image cap (default 5)
    ``max_turns``       episode length cap; 0 means the harness's own budget
    ``city``            name used in the prompt (default ``Paris``)
    ``reward_basis``    ``earnings_per_hour`` (the job's own measure),
                        ``earnings`` or ``env_return``
    ``progress_weight`` dense reward for closing on the target (default 0.0)
    ``image_max_side``  downscale frames to this long side (0 = leave alone)
    ==================  =======================================================

    Optional constraints, each a boolean the yaml may set; an absent key keeps
    the environment's default, so an old config builds the old environment:

    ==============================  ==========================================
    ``enable_earning_jitter``       fees swing ±20% per order, own RNG stream
                                    (default false)
    ``enable_food_temperature``     food cools 8 min after collect; a cold
                                    drop pays 70% (default false)
    ``enable_special_notes``        half of slips carry a note that changes
                                    what the door costs (default false)
    ``enable_walking_energy``       the stamina drain and REST (default TRUE:
                                    this has always been on)
    ``enable_phone_battery``        routes cost 4%, the lit map 0.5%/block;
                                    a dead phone loses the map for the rest
                                    of the shift (default false)
    ==============================  ==========================================

    **On ``progress_weight``.** GRPO normalises the advantage within a group of
    ``rollout.n`` samples of the same prompt. A policy that never delivers earns
    exactly 0.0 on every sample, so the group has no variance, every advantage
    is zero, and the first training step reported ``reward_variance: 0.0``,
    ``pg_loss: 0.0`` and ``grad_norm: 0.0`` -- a step that cost two minutes and
    changed nothing. Distance closed on the active order's target is dense
    enough to have variance and still points at the job. It is potential-based
    (Ng, Harada & Russell 1999), so it cannot change which policy is optimal.
    """

    def __init__(self, env_config: dict[str, Any] | None = None):
        config = dict(env_config or {})
        super().__init__(config)
        self.config = config

        _md = Path(config.get("map_dir") or DEFAULT_MAP)
        # Relative paths anchor to the repository, not to whatever the
        # process's cwd happens to be on a node.
        self.map_dir = _md if _md.is_absolute() else (REPO / _md)
        # RQ3 (map scaling): a list of map names; each episode's city is
        # picked by seed % len(maps), so the mixture is deterministic,
        # reproducible, and balanced without a scheduler. Album roots follow
        # the naming convention every city's bake used; networks are cached
        # per map because building one costs seconds. Empty list = the
        # single-map behaviour, byte for byte.
        self.map_rotation = list(config.get("map_rotation") or [])
        self._networks: dict[str, Any] = {}
        self._album_cache: dict[str, Any] = {}
        album = config.get("album_root")
        self.album_root = _resolve_album(Path(album)) if album else None

        # Where this machine keeps its albums, learned from the one the config
        # named. On a node that unpacked them all together this is exact; the
        # environment variable and the fallbacks cover the rest.
        base = _albums_base(self.album_root)

        def _album(key: str, name: str):
            """An album path: whatever the config says, or its sibling here."""
            raw = config.get(key, str(base / name) if self.album_root else None)
            return _resolve_album(Path(raw)) if raw else None

        name = self.album_root.name if self.album_root else "citycore-paris"
        self.pavement_album_root = _album(
            "pavement_album_root", f"paris_streets_pavement/{name}")
        self.signal_album_root = _album(
            "signal_album_root", f"paris_lamps_real/{name}")
        self.obstacle_album_root = _album(
            "obstacle_album_root", f"paris_obstacles/{name}")
        self.pavement_obstacle_album_root = _album(
            "pavement_obstacle_album_root", f"paris_obstacles_pavement/{name}")
        self.difficulty = config.get("difficulty", "solo")
        # Present so training can run ``difficulty: endless`` one order at a
        # time: the tier's own depth is 3 concurrent jobs, which is a
        # scheduling problem on top of a navigation one. An explicit depth
        # wins over the tier's (the env says so), so endless + depth 1 is a
        # fixed one-hour shift whose dispatcher refills after every delivery
        # -- under which TOTAL earnings vary within a GRPO group (two
        # deliveries against one against none), where the one-order fee, paid
        # identically for any successful route, does not.
        self.queue_depth = config.get("queue_depth", None)
        self.stride = config.get("stride", "block")
        self.embodiment = config.get("embodiment", "human_on_foot")
        self.hazards = bool(config.get("hazards", True))
        self.max_images = config.get("max_images", 5)
        self.max_turns = int(config.get("max_turns", 0))
        # Derived from the map unless the config insists. "Paris" was the
        # hardcoded default, which under map rotation told the courier it was
        # in Paris while it walked small-city-13.
        self.city = config.get("city") or self._city_name(self.map_dir.name)
        self.progress_weight = float(config.get("progress_weight", 0.0))
        # What the policy is ultimately being paid to maximise.
        #
        # ``env_return`` is +1.0 a delivery, +/-0.5 for punctuality, +0.1 a
        # collection and -1.0 for a red light: four constants chosen in this
        # repository, none of them derived from the task. ``earnings`` is the
        # fee the job actually pays -- 3.00 plus a cent a metre, in full when
        # on time and in part when late -- so punctuality and job size are
        # inside the number rather than bolted onto it, and a policy that
        # maximises it is a courier that earns.
        #
        # Progress shaping sits on top of whichever basis is chosen. It is
        # potential-based, so it cannot change which policy is optimal: the
        # optimum under ``earnings`` remains the best-earning courier.
        # RQ1's scalpel: additive per-event charges layered on any basis, so
        # an arm can pay pure money and still be fined for running a red or
        # walking into a barrier. env_return bakes its four constants in;
        # these compose instead. Units are the charge per event, positive
        # numbers (the sign is applied here).
        self.red_penalty = float(config.get("red_penalty", 0.0))
        self.block_penalty = float(config.get("block_penalty", 0.0))
        self._last_red = 0
        self._blocked_attempts = 0
        # RQ2's leash: with adaptive on, only this fraction of episodes draw
        # from the biased distribution; the rest stay on the base one. The
        # E4 collapse taught that an unbounded distribution shift plus a
        # starving reward signal is how a policy walks off a cliff.
        self.adaptive_mix = float(config.get("adaptive_mix", 1.0))
        self.reward_basis = str(config.get("reward_basis", "env_return"))
        if self.reward_basis not in ("earnings_per_hour", "earnings",
                                     "env_return", "deliveries"):
            raise ValueError(f"unknown reward_basis {self.reward_basis!r}")
        self._last_earnings = 0.0
        self._last_delivered = 0
        self._last_red = 0
        self._blocked_attempts = 0
        self._earnings_at = {}
        # Optional-constraint flags, forwarded to CourierEnv only when the yaml
        # actually writes them: an absent key leaves the env's own default in
        # charge, so this file never needs to know what those defaults are and
        # a config from before the flags existed builds the same environment.
        self.constraint_flags = {
            key: bool(config[key]) for key in (
                "enable_earning_jitter", "enable_food_temperature",
                "enable_special_notes", "enable_walking_energy",
                "enable_phone_battery", "enable_food_categories",
                "enable_phone_recharge",
            ) if key in config
        }
        # Numeric difficulty knobs, forwarded only when the yaml writes them.
        # deadline_slack scales every per-job window; food_warm_seconds sets
        # the hot-meal spoil fuse (the 480 s default never bound -- c2 held a
        # warm rate of 1.0 throughout -- so the arms that study spoilage
        # tighten it here).
        self.env_knobs = {}
        if "deadline_slack" in config:
            self.env_knobs["deadline_slack"] = float(config["deadline_slack"])
        if "food_warm_seconds" in config:
            self.env_knobs["food_warm_seconds"] = float(config["food_warm_seconds"])

        # ── adaptive curriculum (opt-in; off is the benchmark, exactly) ─────
        #
        # `adaptive: true` makes reset() read a weights profile -- written by
        # tools/adaptive_profile.py from evaluation failures -- and hand it to
        # the dispatcher as order_bias, which reweights (never widens) the
        # order distribution toward the classes the policy currently fails.
        # The profile is re-read when its mtime changes, so a run picks up
        # each validation's diagnosis without restarting.
        #
        # With `adaptive` false or the file absent, order_bias is None and the
        # dispatcher takes its pre-existing branch, RNG stream untouched:
        # baseline shifts are byte-identical to a tree without this feature.
        # From the config only. An ADAPTIVE=1 fallback read from the process
        # environment was tried and switched the curriculum on inside every
        # VALIDATION env as well (no val block carries the key), which biased
        # the held-out seeds the diagnosis is drawn from. Arms that want the
        # curriculum say so in their training yaml (`train_*_cur.yaml`); the
        # sidecar's ADAPTIVE_MODE only chooses how the profile is built.
        self.adaptive = bool(config.get("adaptive", False))
        self.adaptive_profile = (config.get("adaptive_profile")
                                 or os.environ.get("ADAPTIVE_PROFILE") or "")
        self._profile_cache: tuple[float, dict] | None = None
        # Where episodes report their failure taxonomy for the profile builder.
        # Written whenever the path is set, adaptive or not -- validation must
        # write it (it is where the diagnosis comes from) and validation does
        # not run with adaptive on.
        self.episode_log = (config.get("episode_log")
                            or os.environ.get("ADAPTIVE_EPISODE_LOG") or "")
        # Turns are what this task needs and images are what crowds them out.
        # A 640x480 frame costs ~380 tokens after the vision merge, so at four
        # turns -- which is what the token budget forced -- the window covers
        # two of the ten successful collections measured on this model, which
        # happened on turns 2, 3, 5, 5, 8, 14, 14, 17, 26 and 36. Halving the
        # long side quarters the pixels and roughly quarters the token cost,
        # and buys back the turns that the signal actually lives in.
        self.image_max_side = int(config.get("image_max_side", 0))

        # The road network is the expensive part of a reset -- parsing it per
        # episode would dominate rollout time in a trainer that resets
        # thousands of times -- so it is built once and shared.
        self._network: Any = None
        self._env: Any = None
        self._session: Any = None
        self._turns = 0
        self._scratch: Any = None
        self._progress_cm = 0.0

    # ── adaptive curriculum plumbing ─────────────────────────────────────────

    def _load_bias(self) -> dict[str, float] | None:
        """The current weights profile, or None -- which means baseline.

        mtime-cached: thousands of resets must not each stat-and-parse a file
        that changes once per validation. A malformed or missing profile is
        None, never an exception: the curriculum degrades to the benchmark,
        it does not take training down.
        """
        if not self.adaptive or not self.adaptive_profile:
            return None
        try:
            import json
            path = Path(self.adaptive_profile)
            mtime = path.stat().st_mtime
            if self._profile_cache and self._profile_cache[0] == mtime:
                return self._profile_cache[1] or None
            data = json.loads(path.read_text())
            bias = {str(k): float(v) for k, v in (data.get("bias") or {}).items()}
            cities = {str(k): float(v)
                      for k, v in (data.get("city_weights") or {}).items()}
            self._profile_cache = (mtime, bias, cities)
            return bias or None
        except (OSError, ValueError, TypeError):
            return None

    def _load_city_weights(self) -> dict[str, float] | None:
        """City sampling weights from the same profile, same cache, same
        degradation rule: anything wrong means uniform rotation, never a
        crash. Only cities actually in this run's rotation count."""
        self._load_bias()   # refresh the cache
        if not self._profile_cache or len(self._profile_cache) < 3:
            return None
        cities = {k: v for k, v in (self._profile_cache[2] or {}).items()
                  if k in self.map_rotation and v > 0}
        return cities or None

    def _log_episode(self, summary: dict[str, Any], seed: int) -> None:
        """One JSON line of failure taxonomy per finished episode.

        This is the measurement half of the adaptive loop: the profile builder
        reads these lines (validation seeds only) and turns failure rates into
        sampling weights. Append with a single os.write so concurrent env
        workers interleave lines, not bytes.
        """
        if not self.episode_log:
            return
        try:
            import json
            walked = float(summary.get("walked_m") or 0.0)
            optimal = summary.get("optimal_walk_m")
            row = {
                "seed": int(seed),
                "delivered": int(summary.get("delivered") or 0),
                "late": int(summary.get("late") or 0),
                "expired": int(summary.get("expired") or 0),
                "orders_issued": int(summary.get("orders_issued") or 0),
                "red_crossings": int(summary.get("red_crossings") or 0),
                "waits_at_red": int(summary.get("waits_at_red") or 0),
                "earnings": float(summary.get("earnings") or 0.0),
                "walked_m": walked,
                "detour": (round(walked / float(optimal), 3)
                           if optimal else None),
                "turns": self._turns,
                # Which curriculum classes this shift's issued orders fell in,
                # so the frontier profile builder can credit a seed's group
                # variance to the classes it actually exercised.
                "order_classes": (self._env.order_class_counts()
                                  if self._env is not None else {}),
                "city": self.map_dir.name,
                "phone_alive": (None if summary.get("phone_battery_left") is None
                                else summary.get("phone_died_at_s") is None),
                "recharges": int(summary.get("phone_recharges") or 0),
            }
            line = (json.dumps(row) + "\n").encode()
            fd = os.open(self.episode_log,
                         os.O_WRONLY | os.O_CREAT | os.O_APPEND, 0o644)
            try:
                os.write(fd, line)
            finally:
                os.close(fd)
        except OSError:
            pass  # telemetry must never take a rollout down

    # ── VAGEN interface ──────────────────────────────────────────────────────

    async def system_prompt(self) -> dict[str, Any]:
        if self._session is None:
            raise RuntimeError("reset() before system_prompt()")
        return {"obs_str": self._session.system_prompt()}

    @property
    def session(self):
        """The harness driving the current episode (None before reset())."""
        return self._session

    @property
    def world(self):
        """The CourierEnv underneath the current episode (None before reset())."""
        return self._env

    @staticmethod
    def _city_name(map_name: str) -> str:
        """citycore-paris -> Paris; small-city-13 -> Small City 13."""
        if "paris" in map_name.lower():
            return "Paris"
        return map_name.replace("-", " ").title()

    def _select_map(self, name: str) -> None:
        """Point map_dir, network and every album root at one city."""
        from embodiedbench.compiler.road_network import build_road_network
        maps_dir = DEFAULT_MAP.parent
        self.map_dir = maps_dir / name
        if name not in self._networks:
            self._networks[name] = build_road_network(self.map_dir,
                                                      map_name=name)
        self._network = self._networks[name]
        if name not in self._album_cache:
            base = _albums_base(self.album_root)
            street = (base / "paris_streets_v2" / name
                      if name == "citycore-paris"
                      else base / "city_streets_v1" / name)
            self._album_cache[name] = {
                "album_root": _resolve_album(street),
                "pavement_album_root": _resolve_album(
                    base / "paris_streets_pavement" / name),
                "signal_album_root": _resolve_album(
                    base / "paris_lamps_real" / name),
                "obstacle_album_root": _resolve_album(
                    base / "paris_obstacles" / name),
                "pavement_obstacle_album_root": _resolve_album(
                    base / "paris_obstacles_pavement" / name),
            }
        for key, value in self._album_cache[name].items():
            setattr(self, key, value)
        if not self.config.get("city"):
            self.city = self._city_name(name)

    def _check_album_city(self) -> None:
        """The album must photograph the city being walked. Node ids repeat
        across cities (every map numbers its streets s000_n000 up), so an
        album root pointing at the wrong city serves the wrong city's
        photographs SILENTLY -- discovered when a duplicate YAML key left a
        medium-city-18 validation block reading Paris pavement frames. The
        manifest names its map; a mismatch is an error, not a fallback."""
        import json as _json
        for root in (self.album_root, self.pavement_album_root):
            if root is None:
                continue
            manifest = Path(root) / "manifest.jsonl"
            if not manifest.exists():
                continue
            try:
                with manifest.open() as fh:
                    row = _json.loads(fh.readline())
            except (OSError, ValueError):
                continue
            album_map = str(row.get("map_name") or "")
            if album_map and album_map != self.map_dir.name:
                raise RuntimeError(
                    f"album at {root} photographs {album_map!r} but the map "
                    f"is {self.map_dir.name!r} -- with node ids shared "
                    f"across cities this would serve the wrong city's "
                    f"pictures without a trace")

    async def reset(self, seed: int) -> tuple[dict[str, Any], dict[str, Any]]:
        from embodiedbench.agent.courier.session import CourierSession
        from embodiedbench.compiler.road_network import build_road_network
        from embodiedbench.runtime.city.courier_env import CourierEnv

        if self.map_rotation:
            name = self.map_rotation[int(seed) % len(self.map_rotation)]
            # City-level curriculum (the dispatcher chooses WHERE, not just
            # what): with a profile carrying city_weights and the episode's
            # mixing coin landing on the biased side, the city is drawn from
            # those weights -- deterministically in the seed, so a shift
            # replays identically. Every other path is the old uniform
            # rotation, byte for byte.
            weights = self._load_city_weights() if self.adaptive else None
            if weights:
                import random as _random
                if (self.adaptive_mix >= 1.0
                        or _random.Random(f"adaptive-{seed}").random() < self.adaptive_mix):
                    names = sorted(weights)
                    total = sum(weights[c] for c in names)
                    draw = _random.Random(f"city-{seed}").random() * total
                    for c in names:
                        draw -= weights[c]
                        if draw <= 0:
                            name = c
                            break
            self._select_map(name)
        if self._network is None:
            self._network = build_road_network(self.map_dir,
                                               map_name=self.map_dir.name)
            self._networks[self.map_dir.name] = self._network
            self._check_album_city()

        kwargs: dict[str, Any] = {
            "seed": int(seed),
            "difficulty": self.difficulty,
            "stride": self.stride,
            "embodiment": self.embodiment,
            **self.constraint_flags,
            **self.env_knobs,
        }
        # RQ-V: how much is vision worth? "route" narrates the next street in
        # text (lights and barriers stay pictures-only); "all" narrates
        # everything -- the text-solvable floor. Absent means "none", the
        # benchmark as designed.
        if self.config.get("narration"):
            kwargs["narration"] = str(self.config["narration"])
        bias = self._load_bias()
        if bias and self.adaptive_mix < 1.0:
            # The leash (RQ2): a per-episode coin, seeded by the episode seed
            # so the choice reproduces, keeps most shifts on the base
            # distribution. The bias may steer learning; it may not become
            # the curriculum.
            import random as _random
            if _random.Random(f"adaptive-{seed}").random() >= self.adaptive_mix:
                bias = None
        if bias:
            kwargs["order_bias"] = bias
        if self.queue_depth is not None:
            kwargs["queue_depth"] = int(self.queue_depth)
        # Named, not guessed.
        #
        # These used to be derived as album_root.parent/"signals"/name,
        # which resolves to a directory that does not exist, and the
        # `if path.exists()` guard then skipped it in silence. Training
        # therefore ran without pedestrian lamps, without barrier frames,
        # and -- worse -- without the pavement views, so an on-foot courier
        # was shown the carriageway. Evaluation had all three. The two were
        # not the same environment, and nothing said so. An album the config
        # names has to exist, whether or not a street album is configured:
        # on a machine without albums the check used to be skipped with the
        # rest of this block, and a wrong path went unreported.
        named_albums = (("pavement_album_root", self.pavement_album_root),
                        ("signal_album_root", self.signal_album_root),
                        ("obstacle_album_root", self.obstacle_album_root),
                        ("pavement_obstacle_album_root",
                         self.pavement_obstacle_album_root))
        for key, value in named_albums:
            if value is not None and not value.exists():
                raise FileNotFoundError(
                    f"{key} does not exist: {value}. A missing album used to "
                    "be skipped quietly, which is how training and evaluation "
                    "drifted apart.\n"
                    f"    albums base resolved to: {_albums_base(self.album_root)}\n"
                    f"    ALBUMS_DIR={os.environ.get('ALBUMS_DIR', '<unset>')} "
                    "    If this is a node that unpacked the bundle, set "
                    "ALBUMS_DIR to the directory holding the album folders.")
        if self.album_root is not None:
            kwargs["album_root"] = self.album_root
            for key, value in named_albums:
                if value is None:
                    continue
                if key.startswith(("signal", "obstacle", "pavement_obstacle")) and not self.hazards:
                    continue
                kwargs[key] = value

        # Tell the environment how small the frames arrive, so it charges for
        # a red light only where the lamp survives the downscale. The album
        # certifies legibility at 768 px on the long edge; at 320 that keeps 96
        # of 130 approaches, and the other 34 would be penalties on a lamp too
        # few pixels wide to read.
        if self.image_max_side:
            kwargs["served_long_edge"] = float(self.image_max_side)
        self._env = CourierEnv(self._network, **kwargs)
        self._env.reset()
        self._session = CourierSession(self._env, city=self.city)
        self._turns = 0
        self._progress_cm = 0.0
        self._last_earnings = 0.0
        self._last_delivered = 0
        self._last_red = 0
        self._blocked_attempts = 0
        self._earnings_at = {}
        self._episode_seed = int(seed)

        obs, dropped = self._observation()
        return obs, {"seed": int(seed), "images_dropped": dropped,
                     "difficulty": self.difficulty, "stride": self.stride,
                     "embodiment": self.embodiment}

    async def step(self, action_str: str) -> tuple[dict[str, Any], float, bool, dict[str, Any]]:
        if self._session is None:
            raise RuntimeError("reset() before step()")

        before_cm, before_target = self._remaining_cm()
        log = self._session.step(action_str)
        self._turns += 1

        # The turn's pay, as a delta, so the episode's rewards sum to what the
        # shift earned. The harness's own step reward is the alternative basis.
        # Hard key: a renamed key must raise, not silently zero the reward.
        earned_now = float(self._env.summary()["earnings"])
        earned_this_turn = earned_now - self._last_earnings
        self._last_earnings = earned_now
        # ``deliveries``: +1 the turn an order arrives, so the episode's sum is
        # the count. The experiment it exists for: the fee varies with order
        # length (3.00 + 0.01/m), so under ``earnings`` two samples that both
        # deliver still differ by which orders they happened to draw -- fee
        # magnitude is variance GRPO spends without it meaning skill. A pure
        # count asks whether removing that noise buys convergence, at the cost
        # of no longer preferring the better-paying route.
        delivered_now = int(self._env.summary().get("delivered") or 0)
        delivered_this_turn = delivered_now - self._last_delivered
        self._last_delivered = delivered_now
        reward = (earned_this_turn if self.reward_basis == "earnings"
                  else 0.0 if self.reward_basis == "earnings_per_hour"
                  else float(delivered_this_turn) if self.reward_basis == "deliveries"
                  else float(log.reward or 0.0))

        # Additive charges (RQ1). Applied as deltas so an episode's fines sum
        # to charge x events, on whatever basis is paying.
        red_now = int(self._env.summary().get("red_crossings") or 0)
        red_this_turn = red_now - self._last_red
        self._last_red = red_now
        if log.status == "rejected" and (log.error or "") == "way_blocked":
            self._blocked_attempts += 1
            if self.block_penalty:
                reward -= self.block_penalty
        if self.red_penalty and red_this_turn > 0:
            reward -= self.red_penalty * red_this_turn

        # Potential-based shaping, per turn rather than end-to-end. Collecting a
        # parcel switches the target from the pickup to the dropoff and the two
        # are streets apart, so an end-to-end difference would book that switch
        # as a reward the policy never walked for. Turns where the target
        # changed are skipped; collection is already worth +0.1 unshaped.
        after_cm, after_target = self._remaining_cm()
        step_progress = 0.0
        if (self.progress_weight and before_cm is not None and after_cm is not None
                and before_target == after_target):
            step_progress = (before_cm - after_cm) / PROGRESS_SCALE_CM
            self._progress_cm += before_cm - after_cm
            reward += self.progress_weight * step_progress

        done = bool(self._session.finished)
        if self.max_turns and self._turns >= self.max_turns:
            done = True

        # The judging metric's raw material: cumulative money photographed at
        # fixed turn numbers, all out of the single long validation episode.
        # A shift that ends early keeps earning nothing, so later checkpoints
        # inherit the final figure at done.
        for horizon in (20, 40, 60, 80, 100):
            if self._turns == horizon:
                self._earnings_at[horizon] = earned_now
        if done:
            for horizon in (20, 40, 60, 80, 100):
                self._earnings_at.setdefault(horizon, earned_now)

        obs, dropped = ({"obs_str": ""}, 0) if done else self._observation()
        summary = self._env.summary()

        # Paid at the end, because a rate is not a sum of per-turn deltas.
        #
        # Under ``earnings`` a group of four samples of the same order scored
        # identically whenever they all delivered -- the fee is 3.00 plus a
        # cent a metre of the ORDER, so it pays the same for a ten-turn run and
        # a twenty-one-turn one -- and identically at 0.0 whenever none did.
        # GRPO normalises within the group, so every advantage was 0.0 and the
        # first real training step logged pg_loss 0.0 and grad_norm 0.0 against
        # scores that ranged 0.0 to 5.43 across the batch. The variance was all
        # between orders, where GRPO cannot see it.
        #
        # Per hour, the same delivery pays differently for taking longer, and
        # walking time is what a route choice actually costs: sim_seconds
        # accrues per street walked, per wait at a red, and per red crossed.
        # It is also the courier's own measure of a shift rather than a
        # constant chosen here, and it stays 0.0 for a shift that delivered
        # nothing, so it does not pay a policy to give up early -- which an
        # additive charge for time would.
        if done and self.reward_basis == "earnings_per_hour":
            reward += float(summary.get("earnings_per_hour") or 0.0)
        if done:
            self._log_episode(summary, getattr(self, "_episode_seed", -1))
        info = {
            "status": log.status,          # accepted / rejected / format_error
            "action": log.action,
            "error": log.error,
            "sim_seconds": log.sim_seconds,
            "turns": self._turns,
            "images_dropped": dropped,
            "delivered": summary.get("delivered"),
            "orders_issued": summary.get("orders_issued"),
            # VAGEN reads trajectory success out of info["success"], and this
            # dict did not have the key, so extract_success returned False on
            # every turn of every run. traj_success was reported as 0.0
            # throughout and read here as "the courier never delivered" -- a
            # claim about the policy that was really a claim about a missing
            # dictionary entry. It also gates early termination in the agent
            # loop, so a finished delivery could not end its episode.
            #
            # The definition is mode-aware, because "delivered everything you
            # were issued" cannot be satisfied under ENDLESS: the dispatcher
            # refills after every delivery, so there is an open order at every
            # instant of the shift and delivered >= orders_issued is false by
            # construction. A 110-step H100 run reported traj_success 0.0
            # throughout while held-out earnings rose 20% -- the metric was
            # measuring the dispatcher, not the courier. An ENDLESS shift
            # succeeds when it ends having delivered at least once; and it is
            # judged only at done, because success=True is also the agent
            # loop's early-termination gate, and ending an ENDLESS shift at
            # the first delivery would destroy the total-earnings objective
            # it exists to train.
            "success": bool(
                done and (summary.get("delivered") or 0) >= 1
                if self.difficulty == "endless"
                else (summary.get("orders_issued")
                      and (summary.get("delivered") or 0)
                      >= summary.get("orders_issued"))),
            "on_time": summary.get("on_time"),
            # The episode return the benchmark scores. Carried so a trainer can
            # log the real number next to whatever objective it optimises.
            # What the job actually paid. This is the courier's own measure of
            # a shift and it is defined by the task rather than by constants
            # chosen here: the fee is 3.00 plus 0.01 a metre, full on time and
            # a fraction late, so punctuality and distance are already inside
            # it. env_return, by contrast, is +1.0 a delivery, +/-0.5 for
            # punctuality, +0.1 a collection and -1.0 for a red light -- four
            # numbers with no external justification.
            #
            # It is reported and never optimised. Earnings are zero unless a
            # delivery completes, and traj_success has been zero on every RL
            # measurement so far, so training against money directly would hand
            # the optimiser a constant and no gradient at all. That is the same
            # sparsity that progress shaping exists to bridge.
            "reward_basis": self.reward_basis,
            "earnings": float(summary.get("earnings") or 0.0),
            "red_crossings": int(summary.get("red_crossings") or 0),
            "blocked_attempts": int(self._blocked_attempts),
            "earnings_at": dict(self._earnings_at),
            "earnings_per_hour": float(summary.get("earnings_per_hour") or 0.0),
            "env_return": float(self._session.run.total_reward),
            "progress_score": round(self._progress_cm / PROGRESS_SCALE_CM, 4),
            "step_progress": round(step_progress, 4),
            "progress_weight": self.progress_weight,
            "termination": self._session.run.termination_reason,
        }
        # The compliance ledger for the optional constraints: every flag that
        # is ON reports its execution rate (a project rule). Red lights get a
        # wandb curve, so the rules these arms exist to study must get theirs
        # -- an arm whose compliance is invisible can only be judged on money,
        # which conflates "learned the rule" with "got faster".
        #
        # Emitted UNCONDITIONALLY, with well-defined inert values where the
        # mechanic is off. The first version emitted a key only when its flag
        # was on, which was tidier and wrong one layer down: the constraint
        # val yaml mixes flags-on and flags-off blocks in one validation pass,
        # Ray splits the episodes across workers, and DataProto.concat
        # (vendor verl/protocol.py list_of_dict_to_dict_of_list) ASSERTS that
        # every worker's dict carries exactly the first worker's keys --
        # courier-c2-battery died at step-0 validation on precisely that.
        # The tolerant key-union lives in the two reward aggregators; the
        # concat layer is strict by design (it also carries tensors), so the
        # uniformity has to come from here: same nine keys, every episode,
        # every block, whatever the flags.
        delivered = int(summary.get("delivered") or 0)
        cold = int(summary.get("cold_deliveries") or 0)
        melted = int(summary.get("melted_deliveries") or 0)
        noted = int(summary.get("notes_followed") or 0)
        battery = summary.get("phone_battery_left")
        info["cold_deliveries"] = cold
        info["melted_deliveries"] = melted
        # Every parcel that arrived inside its spoil window, whatever was in
        # the bag. The categories arm's headline compliance number.
        info["intact_delivery_rate"] = (
            (delivered - cold - melted) / delivered if delivered else 1.0)
        info["phone_recharges"] = int(summary.get("phone_recharges") or 0)
        # Share of deliveries handed over warm. Vacuously 1.0 when nothing
        # was delivered (an episode with no deliveries broke no warm-food
        # rule) and identically 1.0 when the mechanic is off (cold stays 0).
        info["warm_delivery_rate"] = (
            (delivered - cold) / delivered if delivered else 1.0)
        # The note is executed by the world at the door, so there is no
        # disobey channel; the rate that exists is exposure -- what share of
        # completed deliveries carried a note. The *learning* signal for
        # notes is time: door seconds saved or spent, visible in
        # earnings_per_hour against base2. 0.0 wherever the mechanic is off.
        info["notes_followed"] = noted
        info["noted_delivery_rate"] = noted / delivered if delivered else 0.0
        # 1.0 = finished the shift with a live phone; the fleet mean is the
        # share of episodes that kept their map. A shift with no battery
        # mechanic reports a full, alive phone -- the inert constant.
        info["phone_battery_left"] = 100.0 if battery is None else float(battery)
        info["phone_alive_rate"] = float(summary.get("phone_died_at_s") is None)
        # Jitter has no behaviour to comply with -- it moves the fees, not
        # the courier. What is reportable is whether the policy *reads* the
        # moved fee: mean realised fee per delivery, against base2's fixed
        # formula on the same seeds. Meaningful on every arm.
        info["mean_fee_paid"] = (
            float(summary.get("earnings") or 0.0) / delivered if delivered else 0.0)
        info["rests"] = int(summary.get("rests") or 0)
        info["stamina_left"] = float(summary.get("stamina_left") or 0.0)
        return obs, reward, done, info

    async def close(self) -> None:
        if self._scratch is not None:
            self._scratch.cleanup()
            self._scratch = None
        self._session = self._env = None

    def _remaining_cm(self) -> tuple[float | None, str | None]:
        """How far the courier still has to walk, and what it is walking to.

        ``route_length_cm`` is the environment's own bookkeeping and never
        appears in an observation. Using it for a *training* signal is
        legitimate for the same reason a simulator may compute a reward it does
        not show: the policy is scored on ``env_return``, which this never
        touches.
        """
        if not self.progress_weight:
            return None, None
        order = self._env.active_order()
        if order is None:
            return None, None
        target = order.target.kerb_node
        return self._env.route_length_cm(self._env.node_id, target), target

    # ── observations ─────────────────────────────────────────────────────────

    def _observation(self) -> tuple[dict[str, Any], int]:
        """The harness's own text, with a caption naming exactly what was sent.

        The photographs block is rebuilt from the labels of the frames that
        went out. The harness writes a caption for every frame the turn offers,
        and this cap sends a subset, so leaving its text alone tells the policy
        it can see streets and lamps whose pictures never arrived.
        """
        observation = self._session.observe()
        images, dropped, labels = self._load_images(observation)
        # Tell the environment which lamps survived the image budget. It gates
        # the red-light charge on what the album can show; this narrows that to
        # what was actually sent, because at max_images the pair for a dropped
        # street goes with it and the courier would otherwise be penalised for
        # a light that never reached it.
        legs = getattr(self._session, "lamp_legs", {}) or {}
        sent = {legs[key] for label in labels
                for key in [self._lamp_key(label)]
                if key is not None and key in legs}
        self._env.show_only_these_signals(sent if legs else None)
        text = observation.text
        # Always, not only when something was dropped. This adapter sends the
        # frames interleaved -- street, its lamp, next street, its lamp -- while
        # the harness's own caption lists every street view first and then every
        # lamp. When nothing is dropped the harness text survived and disagreed
        # with the order of the images beside it: at max_images=5 the second
        # image is street 1's lamp and the caption calls it the view down
        # street 2. Latent at max_images=1 and live at this adapter's own
        # default, which is exactly where raising the image budget would land.
        text = _replace_photo_caption(text, labels)
        if images:
            text = f"{text}\n\n{' '.join([IMAGE_PLACEHOLDER] * len(images))}"
        obs: dict[str, Any] = {"obs_str": text}
        if images:
            obs["multi_modal_input"] = {IMAGE_PLACEHOLDER: images}
        return obs, dropped

    @staticmethod
    def _lamp_key(label: str) -> str | None:
        """The ``street, bearing`` a lamp caption names, or None if not a lamp."""
        match = re.match(r"\[light:\s*([^\]]+)\]", label)
        return match.group(1).strip() if match else None

    def _load_images(self, observation: Any) -> tuple[list[Any], int, list[str]]:
        """The pictures for one turn, and the captions that describe exactly them.

        Two rules beyond "take the first n".

        A street view and its pedestrian lamp travel together. The benchmark
        charges for crossing on red, and its own rule is that a mechanic is only
        charged when the album can show it -- so sending the street without the
        lamp reintroduces the very defect the visibility gate exists to prevent,
        and adds a coin-flip penalty the policy cannot avoid. ``max_images``
        therefore counts street views; a lamp rides along with its street.

        The captions are rebuilt from the labels of the frames that actually
        went out. Rewriting the harness's text with a pattern was fine while a
        turn was three street views and became wrong the moment a signalised
        junction added "[light k]" lines to the same block.
        """
        from PIL import Image

        validate_atomic_image_capacity(observation, self.max_images)
        groups = required_image_groups(observation)
        required_ids = {id(frame) for frames in groups.values() for frame in frames}
        required = [frame for frame in observation.frames if id(frame) in required_ids]
        streets, lamps, drawings = [], {}, []
        for frame in observation.frames:
            if frame.kind == "map" and frame.svg:
                drawings.append(frame)
            elif id(frame) in required_ids:
                continue
            elif frame.kind == "photograph" and frame.path:
                # The label names the street and its bearing -- "[Rue de la
                # Paix, west]" for a view, "[light: Rue de la Paix, west]" for
                # its lamp -- so the two are paired on that text. They used to
                # be paired on an index, which went away with the numbers.
                match = re.match(r"\[light:\s*([^\]]+)\]", frame.label)
                if match:
                    lamps[match.group(1).strip()] = frame
                else:
                    streets.append(frame)

        chosen: list[Any] = []
        labels: list[str] = []
        dropped = 0

        def load(frame, *, is_required: bool = False) -> bool:
            try:
                chosen.append(self._fit(Image.open(frame.path).convert("RGB")))
                labels.append(frame.label)
                return True
            except Exception as exc:  # noqa: BLE001 - required frames fail the turn
                if is_required:
                    raise RuntimeError(
                        f"required image {frame.view_id} could not be loaded"
                    ) from exc
                return False

        for frame in required:
            load(frame, is_required=True)

        sent_streets = 0
        for frame in streets:
            named = re.match(r"\[([^\]]+)\]", frame.label)
            key = named.group(1).strip() if named else None
            optional_count = 1 + (1 if key in lamps else 0)
            if ((groups and len(chosen) + optional_count > self.max_images)
                    or (not groups and sent_streets >= self.max_images)):
                dropped += 1 + (1 if key in lamps else 0)
                continue
            if not load(frame):
                dropped += 1
                continue
            sent_streets += 1
            if key in lamps and not load(lamps[key]):
                dropped += 1

        # The blind control has no album, and the map is a picture: sending it
        # would give the text-only condition the one visual the sighted
        # condition navigates by, which is the comparison the control exists to
        # make.
        for frame in (drawings if self.album_root else []):
            if groups and len(chosen) >= self.max_images:
                dropped += 1
                continue
            raster = self._rasterise(frame.svg, len(chosen))
            if raster is None:
                dropped += 1
            else:
                chosen.append(raster)
                labels.append(frame.label)
        return chosen, dropped, labels

    def _fit(self, image: Any) -> Any:
        """Downscale to the configured long side, preserving aspect ratio."""
        if not self.image_max_side:
            return image
        longest = max(image.size)
        if longest <= self.image_max_side:
            return image
        scale = self.image_max_side / longest
        size = (max(1, round(image.width * scale)), max(1, round(image.height * scale)))
        return image.resize(size)

    def _rasterise(self, svg: str, index: int) -> Any:
        """The phone map as pixels, or nothing.

        Nothing rather than a text description: a drawing described in words is
        a different observation from a drawing, and swapping one for the other
        silently would change what the ``visual`` condition is measuring.
        """
        try:
            import cairosvg
            from PIL import Image
        except ImportError as exc:
            # A missing library is a machine configuration error, not a bad
            # frame's luck; swallowed per frame it removes every map silently.
            raise RuntimeError(
                "the phone map cannot render: cairosvg/PIL is not installed "
                "in this environment. Install cairosvg or run a condition "
                "without the phone."
            ) from exc
        try:
            if self._scratch is None:
                self._scratch = tempfile.TemporaryDirectory(prefix="courier-map-")
            out = Path(self._scratch.name) / f"map_{index}.png"
            cairosvg.svg2png(bytestring=svg.encode(), write_to=str(out),
                             output_width=720, output_height=540)
            # Through the same downscale as a photograph. The map used to go
            # out at 720x540 -- roughly 400 tokens against a photograph's 80 --
            # because it was rendered rather than loaded and never met _fit. It
            # survives the resize: at 320 px the route line, the position dot,
            # the destination and the compass are all legible, and only the
            # street labels are lost, which is deliberate.
            return self._fit(Image.open(out).convert("RGB"))
        except Exception:  # noqa: BLE001
            # A genuinely bad SVG stays a per-frame miss, counted in dropped.
            return None
