"""``LiveCourierEnv``: the courier world with a renderer instead of a bake.

The subclass overrides exactly the three frame lookups (plus the coverage
report) and nothing else, because those three methods are the only places
``CourierEnv`` constructs an album path. Everything that decides *what* to
show -- the visibility gates, the signal phase, which obstacle is in effect,
which viewpoint this body is entitled to -- stays the stock code running
unchanged over a directory that happens to fill itself in. Transitions never
touch the renderer at all: a dead UE fleet degrades this env to album mode
(it serves whatever frames the cache already has, which may be none) and the
walk, the clock and the charges continue exactly as they would over a bare
album. That is design plan §7.1's "never required for every RL worker", honoured
at the level where it is true.

The render request for any frame is fully determined by state the env already
owns:

* street view -- camera at the node (kerb-offset when the served viewpoint is
  the pavement, by the pavement bake's own rule: half the street's width plus
  60 cm, perpendicular-right of travel), eye 160 cm, yaw along the directed
  edge, 640x480 at 90 degrees -- the carriageway bake's camera exactly;
* lamp -- the close-up camera the real-lamp bake used (1280 px long edge,
  FOV 40, eye 165 cm), with the phase computed from ``sim_seconds`` by the
  same ``signal_state`` the charge uses, so picture and penalty cannot
  disagree;
* obstacle -- the street camera plus the obstacle's kind, edge endpoints and
  street width, because the *caller* owns which edges have obstacles
  (deterministic in (map, seed)) and the service owns only prop geometry.
"""

from __future__ import annotations

import logging
import math
from pathlib import Path
from typing import Any

from embodiedbench.compiler.road_network import RoadNetwork, bearing_deg
from embodiedbench.runtime.city.courier_env import CourierEnv, signal_state
from embodiedbench.runtime.city.embodiment import Viewpoint

from .cache import LiveAlbum
from .client import RenderServiceError, ServiceBusy
from .protocol import (
    RENDER_KIND_LAMP,
    RENDER_KIND_OBSTACLE,
    RENDER_KIND_STREET,
    RETURN_MODE_PATH,
    CameraSpec,
    ObstacleSpec,
    ProtocolViolation,
    RenderBatch,
    RenderItem,
    SignalSpec,
)

logger = logging.getLogger(__name__)

# The carriageway bake's camera, exactly (tools/ue/bake_obstacles.py and the
# street bake pin the same numbers; embodiment/oracle.ALBUM_INTRINSICS is the
# certified statement of them). Changing these makes live frames differ from
# the baked albums in ways that are not the scene.
STREET_CAMERA = CameraSpec(width=640, height=480, fov_deg=90.0)
STREET_EYE_CM = 160.0
# The real-lamp bake's close-up camera (compiler/bake_real_lamps.py: EYE_CM,
# FOV_DEG, RENDER_LONG_EDGE) -- the sizes signal_visibility.json's lamp_px
# measurements are expressed in, which is why they must match.
LAMP_CAMERA = CameraSpec(width=1280, height=960, fov_deg=40.0)
LAMP_EYE_CM = 165.0
# How far past the kerbstone the pavement camera stands
# (tools/ue/bake_pavement_views.py KERB_MARGIN_CM).
KERB_MARGIN_CM = 60.0
# Per-item failures are the caller's information -- until every item of this
# many consecutive batches has failed, at which point "the engine's capture
# path is broken while HTTP stays healthy" is the only reading left, and the
# episode is degraded so album_coverage stops reporting renderable-on-demand
# health it can no longer deliver.
ALL_FAILED_DEGRADE_BATCHES = 3

# The album roots the live env owns. A caller that passes its own is asking
# for two sources of truth about one directory.
_OWNED_ROOTS = ("album_root", "signal_album_root", "obstacle_album_root",
                "pavement_album_root", "pavement_obstacle_album_root")


class LiveCourierEnv(CourierEnv):
    """CourierEnv whose album is rendered on demand by a UE fleet.

    ``renderer`` is anything with ``render(batch) -> results`` -- normally a
    ``RenderPool``, but a bare ``UERenderClient`` works for a single-instance
    setup and tests, which is why the parameter is duck-typed rather than
    typed to the pool.
    """

    def __init__(
        self,
        network: RoadNetwork,
        renderer: Any,
        street_camera: Any = None,
        *,
        episode_id: str,
        cache_root: str | Path,
        obstacle_sidecar_root: str | Path | None = None,
        signal_sidecar_root: str | Path | None = None,
        return_mode: str = RETURN_MODE_PATH,
        **courier_kwargs: Any,
    ):
        # Sidecar transfer is split by validity (spec section 4):
        # obstacle_sidecar_root's claims transfer because obstacle frames use
        # the same street camera pose as the bake; signal_sidecar_root is an
        # explicit opt-in that does NOT transfer in v0 (the bake aimed at the
        # lens with pitch, this env stands at the node with pitch 0) and
        # LiveAlbum warns loudly when it is set. Default: obstacles
        # chargeable when a root is given, signals off.
        clash = sorted(set(_OWNED_ROOTS) & set(courier_kwargs))
        if clash:
            raise ValueError(
                f"LiveCourierEnv owns the album roots; got {clash}. To reuse a "
                "baked album's visibility claims, pass obstacle_sidecar_root "
                "(and, opt-in, signal_sidecar_root).")
        self.live_album = LiveAlbum(cache_root, episode_id,
                                    obstacle_sidecar_root=obstacle_sidecar_root,
                                    signal_sidecar_root=signal_sidecar_root)
        self.renderer = renderer
        self.return_mode = return_mode
        # Sticky for the episode once the backend dies, so a dead fleet costs
        # one warning and zero per-frame timeouts. reset() re-arms it -- an
        # episode boundary is the natural moment to ask the fleet again.
        self.live_degraded = False
        # The bake's camera unless the caller renders at serving size.
        self.street_camera = street_camera or STREET_CAMERA
        self.live_render_failures = 0
        self.live_rendered = 0
        self.live_busy_skips = 0
        self._all_failed_batches = 0
        # Every root is the one cache directory. The suffix convention keeps
        # the frame kinds apart within it, and pointing the pavement roots at
        # the same place means the stock viewpoint-selection logic runs
        # unchanged -- an on-foot body is served the pavement viewpoint, which
        # this env honours with the camera offset rather than a second bake.
        root = self.live_album.root
        super().__init__(
            network,
            album_root=root,
            signal_album_root=root,
            obstacle_album_root=root,
            pavement_album_root=root,
            pavement_obstacle_album_root=root,
            **courier_kwargs,
        )

    @property
    def episode_id(self) -> str:
        return self.live_album.episode_id

    # ── lifecycle ────────────────────────────────────────────────────────────

    def reset(self) -> None:
        self.live_degraded = False
        self._all_failed_batches = 0
        super().reset()

    # ── the render plumbing ──────────────────────────────────────────────────

    def _render(self, items: list[RenderItem]) -> None:
        """Render whichever of these frames the album does not have yet.

        Failure never escapes: a backend error marks the episode degraded and
        the lookup falls through to the album, where a missing frame means
        exactly what it means for a baked album with a hole in it. Transitions
        cannot be touched from here by construction -- this method only ever
        adds files to a directory.

        The failure taxonomy, spelled out:

        * ``ServiceBusy`` -- transient load, NOT degradation: skip the batch
          at info level, leave the miss, retry on the next lookup;
        * ``RenderServiceError`` / ``ProtocolViolation`` / ``OSError`` --
          dead fleet, version skew, unreadable transport: all deterministic
          for the rest of the episode, so degrade once and stop asking.
          Arbitrary ``Exception`` is deliberately NOT caught -- a genuine bug
          in this code should crash the test that finds it, not hide as one
          more degraded episode;
        * per-item ``failed`` results -- the caller's information, counted in
          ``live_render_failures``, until ``ALL_FAILED_DEGRADE_BATCHES``
          consecutive batches fail every item, which is an engine whose
          capture path is broken behind a healthy HTTP front, and degrades.
        """
        missing = [item for item in items if not self.live_album.has(item.key)]
        if not missing or self.live_degraded:
            return
        batch = RenderBatch(
            episode_id=self.live_album.episode_id,
            return_mode=self.return_mode,
            camera=self.street_camera,
            requests=tuple(missing),
        )
        try:
            results = self.renderer.render(batch)
        except ServiceBusy as error:
            # Busy is transient by spec: the fleet is loaded, not dead, so the
            # episode is NOT degraded. The batch is skipped, the cache miss
            # remains, and the next lookup at these keys simply asks again.
            self.live_busy_skips += 1
            self.live_render_failures += 1
            logger.info(
                "render backend busy (%s); skipped a batch of %d for episode "
                "%s -- the next lookup retries", error, len(missing),
                self.live_album.episode_id)
            return
        except (RenderServiceError, ProtocolViolation, OSError) as error:
            # ProtocolViolation is version skew: deterministic for the whole
            # episode, so degrading -- one warning, album mode -- is correct,
            # where letting it escape killed the turn (and, under gather, the
            # rollout chunk) for a service that answered 200 in a shape this
            # client refuses.
            self.live_degraded = True
            logger.warning(
                "live render backend failed (%s: %s); episode %s continues in "
                "album mode on %d cached frame(s)",
                type(error).__name__, error, self.live_album.episode_id,
                sum(1 for _ in self.live_album.images.rglob("*.png")))
            return
        by_key = {result.key: result for result in results}
        every_item_failed = True
        for item in missing:
            result = by_key.get(item.key)
            if result is None or not result.ok:
                self.live_render_failures += 1
                continue
            every_item_failed = False
            try:
                if self.live_album.store(item.key, result) is not None:
                    self.live_rendered += 1
            except OSError as error:
                # A cache directory that cannot be written is as dead as the
                # fleet, and deterministically so: degrade, do not crash.
                self.live_degraded = True
                logger.warning(
                    "live cache write failed (%s: %s); episode %s continues "
                    "in album mode", type(error).__name__, error,
                    self.live_album.episode_id)
                return
        if every_item_failed:
            self._all_failed_batches += 1
            if self._all_failed_batches >= ALL_FAILED_DEGRADE_BATCHES:
                self.live_degraded = True
                logger.warning(
                    "every item failed in %d consecutive render batches; the "
                    "engine's capture path is broken behind a healthy HTTP "
                    "front. Episode %s degrades to album mode.",
                    self._all_failed_batches, self.live_album.episode_id)
        else:
            self._all_failed_batches = 0

    def _camera_xy(self, node_id: str, yaw_deg: float) -> tuple[float, float]:
        """Where the street camera stands, honouring the served viewpoint.

        The pavement rule is the pavement bake's, verbatim
        (tools/ue/bake_pavement_views.plan_jobs): offset perpendicular to the
        direction of travel, toward the right-hand kerb, by half the street's
        own width plus 60 cm. UE yaw is degrees clockwise from +X, so the
        right-hand normal is yaw + 90.
        """
        node = self.network.nodes[node_id]
        if self.viewpoint_served != Viewpoint.PAVEMENT:
            return node.x_cm, node.y_cm
        street = self.streets[node.street_index]
        offset = street.width_cm / 2.0 + KERB_MARGIN_CM
        right = math.radians(yaw_deg + 90.0)
        return (node.x_cm + offset * math.cos(right),
                node.y_cm + offset * math.sin(right))

    def _edge_width_cm(self, a: str, b: str) -> float:
        """The width of the street an edge runs along -- ``edge_street``'s own
        rule, answered in centimetres instead of a name."""
        street_a = self.network.nodes[a].street_index
        street_b = self.network.nodes[b].street_index
        return self.streets[street_a if street_a == street_b else street_b].width_cm

    def _street_item(self, key: str, node_id: str, toward: str) -> RenderItem:
        yaw = bearing_deg(self.position(node_id), self.position(toward))
        x_cm, y_cm = self._camera_xy(node_id, yaw)
        return RenderItem(key=key, x_cm=x_cm, y_cm=y_cm, z_cm=STREET_EYE_CM,
                          yaw_deg=yaw, render_kind=RENDER_KIND_STREET)

    def _obstacle_item(self, key: str, start: str, step: str, kind: str) -> RenderItem:
        yaw = bearing_deg(self.position(start), self.position(step))
        x_cm, y_cm = self._camera_xy(start, yaw)
        a = self.position(start)
        b = self.position(step)
        return RenderItem(
            key=key, x_cm=x_cm, y_cm=y_cm, z_cm=STREET_EYE_CM, yaw_deg=yaw,
            render_kind=RENDER_KIND_OBSTACLE,
            obstacle=ObstacleSpec(
                kind=kind, a_cm=a, b_cm=b,
                street_width_cm=self._edge_width_cm(start, step),
                viewpoint=self.viewpoint_served,
            ),
        )

    def _lamp_item(self, key: str, node_id: str, toward: str,
                   state: str, bearing: float) -> RenderItem:
        # v0 aiming simplification, stated plainly: the real-lamp bake
        # (compiler/bake_real_lamps.poses) stands the camera between the lamp
        # and the junction and aims at the *lens*, using the lamp_pose sidecar
        # its analysis produced. The live env has no lamp_pose export, so v0
        # stands at the node, eye 165 cm, yawed along the crossing -- same
        # close-up camera, aimed down the approach rather than at the lens.
        # This is why the bake's signal_visibility.json does NOT certify these
        # frames and signal_sidecar_root is a warned opt-in. When a lamp_pose
        # export lands, this method is the one place to refine: stand on the
        # lamp-junction line and send the computed pitch through the
        # protocol's pitch_deg, which exists for exactly that day.
        node = self.network.nodes[node_id]
        return RenderItem(
            key=key, x_cm=node.x_cm, y_cm=node.y_cm, z_cm=LAMP_EYE_CM,
            yaw_deg=bearing, render_kind=RENDER_KIND_LAMP,
            signal=SignalSpec(approach=f"{node_id}|{toward}", state=state),
            camera=LAMP_CAMERA,
        )

    # ── the three lookups, rendered on miss then answered by the stock code ──

    def _plain_frame(self, node_id: str, toward: str) -> str | None:
        key = f"{node_id}/toward_{toward}"
        if not self.live_album.has(key):
            self._render([self._street_item(key, node_id, toward)])
        return super()._plain_frame(node_id, toward)

    def signal_frame_for(self, node_id: str, toward: str) -> str | None:
        # The stock gates, reproduced so a lamp is only rendered where the
        # stock path would serve it: no sidecar claim, no frame, no charge --
        # rendering an unclaimed lamp would waste a GPU on a picture the env
        # is forbidden to show.
        if (self.signal_album_root is not None and node_id in self.signalised
                and self.signal_is_visible(node_id, toward)):
            rows = {row["node"]: row for row in self._raw_candidates()}
            row = rows.get(toward)
            if row is not None:
                state = signal_state(node_id, row["bearing"], self.sim_seconds)
                key = f"{node_id}/toward_{toward}_{state}"
                if not self.live_album.has(key):
                    self._render([self._lamp_item(key, node_id, toward,
                                                  state, row["bearing"])])
        return super().signal_frame_for(node_id, toward)

    def obstacle_frame_for(self, node_id: str, toward: str) -> str | None:
        if self.obstacle_album_root is not None:
            # One batch for the whole block: at block stride several hops can
            # each hold something, and the stock scan will ask about all of
            # them anyway.
            items: list[RenderItem] = []
            for start, step in self.block_chain(node_id, toward):
                kind = self.obstacles.in_effect(start, step)
                if kind is None:
                    continue
                key = f"{start}/toward_{step}_{kind}"
                if not self.live_album.has(key):
                    items.append(self._obstacle_item(key, start, step, kind))
            if items:
                self._render(items)
        return super().obstacle_frame_for(node_id, toward)

    # ── reporting ────────────────────────────────────────────────────────────

    def summary(self) -> dict[str, Any]:
        """The stock summary plus a ``live`` block.

        Without it, an episode whose every render failed was indistinguishable
        from a healthy run in every report a trainer reads -- the charges gate
        on sidecars, not on frames, so the numbers kept moving while the
        courier ran blind. The block is additive: nothing stock is renamed or
        removed, so every existing consumer keeps working.
        """
        out = super().summary()
        out["live"] = {
            "degraded": self.live_degraded,
            "render_failures": self.live_render_failures,
            "rendered": self.live_rendered,
            "busy_skips": self.live_busy_skips,
        }
        return out

    # ── coverage ─────────────────────────────────────────────────────────────

    def album_coverage(self) -> dict[str, Any]:
        """Coverage of the live backend, in the stock report's shape.

        Healthy, every directed edge has a frame -- renderable on demand is
        covered, which is the property the number exists to assert. Degraded,
        the honest answer is what is on disk, because that is all album mode
        can serve. ``rendered`` carries the on-disk count either way, so the
        two claims stay distinguishable. Computed directly rather than via the
        stock method, which would call the overridden ``_plain_frame`` and
        render the entire city to answer a bookkeeping question.
        """
        total = signalled_total = rendered = rendered_signal = 0
        for node_id, node in self.network.nodes.items():
            for neighbour in node.neighbours:
                total += 1
                if self.live_album.has(f"{node_id}/toward_{neighbour}"):
                    rendered += 1
                if node_id in self.signalised:
                    signalled_total += 1
                    if self.live_album.has(f"{node_id}/toward_{neighbour}_red"):
                        rendered_signal += 1
        covered = rendered if self.live_degraded else total
        signalled = rendered_signal if self.live_degraded else signalled_total
        return {
            "directed_edges": total, "with_frame": covered,
            "fraction": round(covered / total, 4) if total else 0.0,
            "signalised_with_frame": signalled,
            "album_root": str(self.album_root) if self.album_root else None,
            "backend": "live",
            "rendered": rendered,
            "degraded": self.live_degraded,
            # Carried so "renderable on demand" can be weighed against how
            # often rendering has actually been failing.
            "render_failures": self.live_render_failures,
        }
