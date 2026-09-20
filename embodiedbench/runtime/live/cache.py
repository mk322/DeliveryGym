"""The per-episode album a live episode fills in as it looks around.

The contract (spec section 4) is that nothing downstream can tell this
directory from a baked album, because it *is* an album -- the same
``images/<node>/toward_<neighbour>[suffix].png`` shape ``CourierEnv`` already
reads, the same two visibility sidecars at its root -- that happens to start
empty and gain frames as the episode renders them. CourierSession,
FrameAliases and the training adapter's ``PIL.Image.open`` all keep working
because none of them is told anything changed.

Two properties are load-bearing:

* **Idempotency per (episode_id, key).** ``observation_media_hash`` and
  FrameAliases both assume the same picture has the same bytes forever, and a
  GPU renderer only guarantees that within one service process. So the first
  render of a key is the only render of it: ``has`` before every request,
  atomic write after, and a replay of the episode reads the file instead of
  asking the GPU to agree with itself.

* **The sidecars come from a bake, never from the renderer -- and only where
  the bake's certification actually transfers.** Visibility is a property of
  scene + camera geometry, so a measured claim carries over exactly when the
  live camera stands where the bake's did. Street and obstacle frames do:
  the live env renders them from the same street camera pose the obstacle
  bake photographed, so ``obstacle_visibility.json`` transfers from
  ``obstacle_sidecar_root``. Lamp close-ups do NOT: the bake certified
  lens-aimed shots (camera between lamp and junction, pitched at the head)
  and the v0 renderer stands at the node with pitch 0, so
  ``signal_visibility.json`` is copied only on the explicit -- and loudly
  warned -- ``signal_sidecar_root`` opt-in. Absent a source, the album stays
  silent and the mechanic stays off, the same "silence is not consent"
  default as a bare album; the live env's summary() reports the backend
  state so the silence stays visible.

* **The sidecars match the constructor, not history.** A reused episode
  directory has its sidecars rewritten from the current sources and stripped
  of any file the current config did not ask for, so a hazards=False run
  cannot inherit a hazards=True run's claims.

The frame filenames keep the album's leaky names (``_road_block``, ``_red``)
*inside the cache* on purpose: the existing ``FrameAliases`` layer already
launders them at the harness boundary, and re-inventing that here would be a
second implementation of a solved problem.
"""

from __future__ import annotations

import base64
import logging
import os
import shutil
import tempfile
import threading
from pathlib import Path

from .protocol import RenderResult

logger = logging.getLogger(__name__)


class LiveAlbum:
    """A lazily-materialised album directory for one episode.

    ``obstacle_sidecar_root`` names the directory holding the baked
    ``obstacle_visibility.json`` (the stock obstacle album); its claims
    transfer because obstacle frames are rendered from the same street camera
    pose the bake photographed.

    ``signal_sidecar_root`` is an EXPLICIT OPT-IN, invalid for scored runs:
    the bake certified lens-aimed close-ups (camera between lamp and
    junction, pitched at the head), while the v0 live renderer stands at the
    node with pitch 0, so the copied certification does not transfer and
    red-light charges can attach to frames that do not show the lamp. Do not
    use it for scored runs until a lamp_pose export lands and ``_lamp_item``
    sends the aimed pose with ``pitch_deg``. Passing it logs a WARNING.

    The two roots are separate because they are separate in the stock layout
    too: signal_visibility.json lives in the real-lamp album and
    obstacle_visibility.json in the (viewpoint-matched) obstacle album -- two
    different trees no single source directory could express.
    """

    def __init__(
        self,
        cache_root: str | Path,
        episode_id: str,
        *,
        obstacle_sidecar_root: str | Path | None = None,
        signal_sidecar_root: str | Path | None = None,
    ):
        if not episode_id or "/" in episode_id or episode_id in (".", ".."):
            # The episode id becomes a directory name; a slash in it would
            # silently nest albums and break the one-directory-per-episode
            # contract the cache root is organised by.
            raise ValueError(f"episode_id is not a directory name: {episode_id!r}")
        self.episode_id = episode_id
        self.root = Path(cache_root) / episode_id
        self.images = self.root / "images"
        self.images.mkdir(parents=True, exist_ok=True)
        self.obstacle_sidecar_root = (
            Path(obstacle_sidecar_root) if obstacle_sidecar_root else None)
        self.signal_sidecar_root = (
            Path(signal_sidecar_root) if signal_sidecar_root else None)
        self._sync_sidecars()
        # One writer at a time *within this instance*. The courier env is
        # single-threaded, but the training adapter runs under an async loop
        # and the cheap lock removes in-process double-writes. Across
        # instances the lock is no protection at all -- there, safety comes
        # from ``store``'s unique temp names and idempotent atomic publish.
        self._lock = threading.Lock()

    def _sync_sidecars(self) -> None:
        """Make the album's sidecars match this constructor's params, exactly.

        Copied per source, split by validity (spec section 4): obstacle claims
        from ``obstacle_sidecar_root``, signal claims -- opt-in only -- from
        ``signal_sidecar_root``. A sidecar the current config did not ask for
        is *removed*: the episode directory is deliberately reusable across
        same-seed resets, and a stale file left by a differently-configured
        run would silently switch on a mechanic -- hazards=False inheriting a
        hazards=True run's claims was exactly that bug.
        """
        if self.signal_sidecar_root is not None:
            logger.warning(
                "signal_sidecar_root is set for episode %s: the signal bake "
                "certified lens-aimed close-ups (camera between lamp and "
                "junction, pitched at the head), but the v0 live renderer "
                "stands at the node with pitch 0, so the copied certification "
                "does not transfer and red-light charges can attach to frames "
                "that do not show the lamp. Do not use for scored runs until "
                "a lamp_pose export lands and _lamp_item sends the aimed pose "
                "with pitch_deg.", self.episode_id)
        for name, source_root in (
                ("obstacle_visibility.json", self.obstacle_sidecar_root),
                ("signal_visibility.json", self.signal_sidecar_root)):
            target = self.root / name
            source = None if source_root is None else source_root / name
            if source is not None and source.exists():
                shutil.copyfile(source, target)
            else:
                target.unlink(missing_ok=True)

    # ── keys and paths ───────────────────────────────────────────────────────

    def path_for(self, key: str) -> Path:
        """Where a render key's frame lives: ``images/<node>/<basename>.png``.

        The key is the spec's ``<node>/toward_<neighbour>[suffix]`` -- exactly
        the album path relative to ``images/``, minus the extension. Keeping
        key == relative path is what makes the cache auditable with ``ls``.
        """
        node, _, basename = key.partition("/")
        if not node or not basename or "/" in basename:
            raise ValueError(f"render key is not '<node>/<frame>': {key!r}")
        return self.images / node / f"{basename}.png"

    def has(self, key: str) -> bool:
        return self.path_for(key).exists()

    # ── materialisation ──────────────────────────────────────────────────────

    def store(self, key: str, result: RenderResult) -> Path | None:
        """Put one render result's PNG at its album path. Idempotent.

        A key that already has a frame keeps it -- first render wins, which is
        the within-process determinism rule. Returns the path, or ``None`` for
        a failed result, which the caller treats exactly like an album that
        never had the frame.
        """
        if not result.ok:
            return None
        target = self.path_for(key)
        with self._lock:
            if target.exists():
                return target
            target.parent.mkdir(parents=True, exist_ok=True)
            # A *unique* temp name per writer, in the target's own directory
            # so os.replace stays a same-filesystem rename. The old fixed
            # "<name>.part" was a shared name: two albums over one directory
            # could truncate each other's half-written bytes and the loser's
            # os.replace raised FileNotFoundError. Unique names make the race
            # harmless -- each writer publishes a complete file, last replace
            # wins, and both were renders of the same key.
            handle = tempfile.NamedTemporaryFile(
                dir=target.parent, prefix=".part-", delete=False)
            temporary = Path(handle.name)
            try:
                with handle:
                    if result.png_base64 is not None:
                        handle.write(base64.b64decode(result.png_base64))
                    elif result.path:
                        # return_mode=path: the service is co-located and
                        # wrote the frame to its own cache; copy rather than
                        # link so the album survives the service recycling
                        # its scratch space.
                        with open(result.path, "rb") as source:
                            shutil.copyfileobj(source, handle)
                    else:
                        return None
                # Atomic publish: a concurrent reader sees no file or the
                # whole file, never a partial PNG that PIL half-decodes. No
                # fsync, deliberately: a torn file cannot survive os.replace,
                # and crash-consistency is not a requirement for a cache -- a
                # frame lost to a power cut is re-rendered on the next miss.
                os.replace(temporary, target)
            except OSError:
                # Losing any race whose winner published the frame is a
                # success: the file this call exists to produce exists.
                if target.exists():
                    return target
                raise
            finally:
                temporary.unlink(missing_ok=True)
        return target
