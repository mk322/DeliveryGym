"""Does the obstacle frame actually show the obstacle?

The runtime will only block a street, or charge for squeezing past one, where
the picture it hands the agent shows what is there. That is not a claim the
renderer gets to make about itself: a prop can be spawned behind a wall, over a
brow, or beyond the point where a 640 x 480 frame resolves it, and every one of
those produces a file on disk that looks like a successful render.

So the album is measured the same way the signal album is, and for the same
reason. An obstructed frame and the clear frame of the *same* approach are shot
from the identical camera, so everything that is not the obstacle is identical
between them and the pixels that differ are the obstacle and nothing else. No
threshold has to find it.

What is then asked of that region is only that it be big enough to see at the
resolution the frame is seen at -- ``MIN_MODEL_PIXELS`` of the model's own
input, the same rule and the same numbers the signal measurement uses -- and
that it be one connected thing rather than a scatter. Unlike the signal there is
no colour question: a barrier is not defined by its hue, so the sign tests that
separate a lamp from foliage are not needed here and are not used.

An approach that fails is not put in the sidecar, and an obstacle whose two
approaches are not both in the sidecar is inert at runtime: not drawn, not
blocking, not charged. This module is what makes that gate mean something.

    python -m embodiedbench.compiler.obstacle_visibility ALBUM \\
        --clear $ALBUMS_DIR/paris_streets_v2/citycore-paris --write-sidecar
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterable, Iterator

import numpy as np
from PIL import Image

from embodiedbench.compiler.signal_legibility import (
    CHANGE_THRESHOLD,
    MIN_MODEL_PIXELS,
    MODEL_LONG_EDGE_PX,
    _largest_component,
    _load_pair,
    _median,
)
from embodiedbench.runtime.city.obstacles import OBSTACLE_TYPES, OBSTACLE_VISIBILITY_FILE


@dataclass
class ObstacleVisibility:
    """One (approach, kind): how much of the frame the obstacle occupies."""

    key: str
    kind: str
    changed_px: int = 0
    blob_px: int = 0
    width: int = 0
    height: int = 0
    error: str = ""

    def model_pixels(self, long_edge: float = MODEL_LONG_EDGE_PX) -> float:
        if not self.width or not self.height:
            return 0.0
        scale = min(1.0, long_edge / max(self.width, self.height))
        return self.blob_px * scale * scale

    def visible(self, min_model_px: float = MIN_MODEL_PIXELS) -> bool:
        return self.model_pixels() >= min_model_px

    def to_dict(self) -> dict[str, Any]:
        return {
            "key": self.key, "kind": self.kind, "changed_px": self.changed_px,
            "blob_px": self.blob_px, "model_px": round(self.model_pixels(), 2),
            "visible": self.visible(), "error": self.error,
        }


def measure_frame(key: str, kind: str, obstructed: Path, clear: Path) -> ObstacleVisibility:
    out = ObstacleVisibility(key=key, kind=kind)
    if not clear.exists():
        out.error = "no clear frame to compare against"
        return out
    try:
        blocked, plain = _load_pair(obstructed, clear)
    except (OSError, ValueError) as error:
        out.error = f"{type(error).__name__}: {error}"
        return out
    if blocked.shape != plain.shape:
        out.error = "frames differ in size"
        return out
    out.height, out.width = blocked.shape[0], blocked.shape[1]
    changed = np.abs(blocked - plain).max(axis=2) >= CHANGE_THRESHOLD
    out.changed_px = int(changed.sum())
    if out.changed_px == 0:
        return out
    out.blob_px = int(_largest_component(changed).sum())
    return out


def walk_album(album: Path) -> Iterator[tuple[str, str, Path]]:
    """Every ``(approach key, kind, path)`` in an obstacle album."""
    images = Path(album) / "images"
    for node_dir in sorted(p for p in images.iterdir() if p.is_dir()):
        for kind in OBSTACLE_TYPES:
            for path in sorted(node_dir.glob(f"toward_*_{kind}.png")):
                toward = path.name[len("toward_"): -len(f"_{kind}.png")]
                yield f"{node_dir.name}|{toward}", kind, path


@dataclass
class AlbumVisibility:
    album: str
    clear: str
    rows: list[ObstacleVisibility] = field(default_factory=list)

    def approaches(self) -> set[str]:
        return {r.key for r in self.rows}

    def visible_keys(self, min_model_px: float = MIN_MODEL_PIXELS) -> list[str]:
        """Approaches whose obstacle shows *whichever* kind is standing there.

        Both kinds have to be legible on an approach, because which one is live
        is chosen per episode and the gate is consulted before that choice is
        known. An approach where the barrier reads but the column does not would
        otherwise be charged for a frame that does not show anything.
        """
        by_key: dict[str, dict[str, bool]] = {}
        for row in self.rows:
            by_key.setdefault(row.key, {})[row.kind] = row.visible(min_model_px)
        return sorted(
            key for key, kinds in by_key.items()
            if len(kinds) == len(OBSTACLE_TYPES) and all(kinds.values())
        )

    def curve(self, floors: Iterable[float] = (1.0, 2.0, 4.0, 8.0, 16.0, 64.0, 256.0)) -> list[dict]:
        return [
            {"min_model_px": floor, "approaches": len(self.visible_keys(floor))}
            for floor in floors
        ]

    def summary(self, min_model_px: float = MIN_MODEL_PIXELS) -> dict[str, Any]:
        keys = self.visible_keys(min_model_px)
        by_kind = {}
        for kind in OBSTACLE_TYPES:
            rows = [r for r in self.rows if r.kind == kind]
            by_kind[kind] = {
                "frames": len(rows),
                "visible": sum(1 for r in rows if r.visible(min_model_px)),
                "median_model_px": round(_median([r.model_pixels() for r in rows]), 1),
            }
        return {
            "album": self.album, "clear_album": self.clear,
            "frames": len(self.rows),
            "approaches": len(self.approaches()),
            "visible_approaches": len(keys),
            "visible_fraction": round(len(keys) / max(len(self.approaches()), 1), 4),
            "by_kind": by_kind,
            "curve": self.curve(),
            "errors": sum(1 for r in self.rows if r.error),
        }

    def sidecar(self, map_name: str, min_model_px: float = MIN_MODEL_PIXELS,
                measured_at: str = "") -> dict[str, Any]:
        keys = self.visible_keys(min_model_px)
        return {
            "map": map_name,
            "measured_at": measured_at,
            "method": (
                "obstructed frame minus clear frame, same camera. The pixels "
                "that differ are the obstacle; an approach is listed when the "
                "largest connected difference, for *both* obstacle kinds, "
                f"survives a vision model's resize to {MODEL_LONG_EDGE_PX:.0f} "
                f"px on the long edge at {min_model_px:.0f} px of area."
            ),
            "approaches": len(self.approaches()),
            "visible_count": len(keys),
            "visible": keys,
        }


def measure_album(album: Path, clear: Path) -> AlbumVisibility:
    out = AlbumVisibility(album=str(album), clear=str(clear))
    for key, kind, path in walk_album(Path(album)):
        node, toward = key.split("|")
        plain = Path(clear) / "images" / node / f"toward_{toward}.png"
        out.rows.append(measure_frame(key, kind, path, plain))
    return out


def main(argv: list[str] | None = None) -> int:
    import argparse
    import datetime

    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("album", type=Path)
    parser.add_argument("--clear", type=Path, required=True,
                        help="the street album the obstructed frames are compared with")
    parser.add_argument("--map-name", default="")
    parser.add_argument("--min-model-px", type=float, default=MIN_MODEL_PIXELS)
    parser.add_argument("--write-sidecar", action="store_true")
    parser.add_argument("--rows", type=Path, default=None)
    args = parser.parse_args(argv)

    measured = measure_album(args.album, args.clear)
    print(json.dumps(measured.summary(args.min_model_px), indent=1))
    if args.rows:
        with Path(args.rows).open("w", encoding="utf-8") as handle:
            for row in measured.rows:
                handle.write(json.dumps(row.to_dict()) + "\n")
    if args.write_sidecar:
        path = Path(args.album) / OBSTACLE_VISIBILITY_FILE
        path.write_text(json.dumps(measured.sidecar(
            args.map_name or Path(args.album).name, args.min_model_px,
            datetime.date.today().isoformat()), indent=1))
        print(f"wrote {path}")
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
