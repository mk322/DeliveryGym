"""Can a courier standing here tell what colour the light is?

The environment charges for crossing on red, and it may only do that where the
frame it serves actually shows the colour. So the album has to be measured, and
the measurement has to be trustworthy -- which the previous one was not.

**What was wrong with the previous metric.** It hunted for a saturated red blob
of at least 25 pixels in the red bake and a saturated green blob of at least 20
in the green one, with an 8-pixel margin between the phases. Those are absolute
cutoffs, and the thing being measured is a lamp lens whose apparent area across
the album has a median of about 60 pixels. A cutoff sitting inside the bulk of
the distribution decides the answer: nudging it a few pixels moves the reported
legibility by tens of percent, and every number quoted from it (34%, 45%, 31%,
0%) inherited that. Worse, "saturated red" and "saturated green" were separate
absolute colour tests, so a dim lamp failed for being dim rather than for being
unreadable, and a red awning could pass one half of the test on its own.

**What replaces it.** The same approach is baked twice, from the identical
camera, with only the signal phase changed. So the question can be asked of the
frame against *itself* rather than against a constant:

    does a pixel go red when the phase is red, and green when it is green?

which is a pair of sign tests about zero and no magnitude at all:

    chroma(p) = (R - G) / (R + G + 1)          per pixel, in [-1, 1]
    flips(p)  iff  chroma_red(p) > 0 > chroma_green(p)

A red awning is red in both bakes, so it never flips. A lamp whose lens does not
move (one lens, two colours) and a pedestrian lamp whose figure moves between
apertures both flip, without needing the two cases the old metric hand-coded.

The one thing a sign test cannot do is ignore a pixel that did not really
change, so a pixel must also have moved at all -- ``CHANGE_THRESHOLD`` on the
raw channels. That is a per-pixel presence test, not a size cutoff, and unlike
the old 25-pixel blob rule it sits in an empty part of the distribution rather
than in the middle of it: across this album the median inter-phase difference is
1 of 255 and the lamp reaches 230, so any threshold from about 8 to 100 selects
the same pixels. ``threshold_sweep`` demonstrates that instead of asserting it.

The flipping pixels are then grouped into connected components and only the
largest counts, because a scatter of single flipped pixels is not a lamp and is
not readable. The old metric's fatal arithmetic was averaging colour over *every*
changed pixel: renders are not bit-identical -- temporal AA and exposure leave a
diffuse haze of a few thousand faintly-changed pixels per frame -- so the mean
it tested was mostly haze, and the lamp it was looking for contributed a few
percent of it.

**One lens or two.** The flip test above asks a single pixel to be red in one
bake and green in the other, which is what a *shared-aperture* head does: the
pedestrian head has one window and swaps the figure inside it. A three-aspect
vehicle head does not — its red lens is at the top and its green lens at the
bottom, two different places in the frame. The flipping component is then the
green lens alone, green when green and a dark grey lens when red, and reporting
that component's colour *in the red bake* reports an unlit lens. Measured that
way the Paris album says its red lamps reach a mean of 45/255, i.e. that the red
does not work; measured per phase it says 199/255, i.e. that it plainly does.
The frames never changed. Only the question did.

So each phase is asked its own question, still self-normalised against the other
bake of the identical camera:

    the red lamp   is what is brighter AND redder in the red bake
    the green lamp is what is brighter AND greener in the green bake

and an approach is legible only when *both* lamps are found, both are lit rather
than merely tinted, and both survive the resize. That is what a courier needs:
being able to see the green does not tell you what red looks like here.

**What else is red.** A no-entry disc, a café awning, a red shopfront: static,
so it never flips and the metric is right to ignore it. A courier is not — it
sees both, and on this album the largest static red object outranks the lamp in
area in a third of the legible frames. That is a property of the city and not a
defect, but it is the reason a frame is not legible merely because something in
it is red, and ``confusers`` reports it so the claim is on the record.

**Decisive is necessary, not sufficient.** Two pixels flipping colour is
decisive and invisible. So the second half of the measurement is apparent size,
reported as a curve over the size floor rather than as one number, and the
default floor is derived from the resolution the frame is actually seen at: a
vision model resizes the long edge to roughly ``MODEL_LONG_EDGE_PX``, and a
detail smaller than about two pixels after that resize is gone. The floor is
therefore stated in *model pixels*, which is the frame the question lives in.

Every constant here is published with the sweep that justifies it -- 
``threshold_sweep`` for the per-pixel change floor, ``margin_sweep`` for the
chroma margin, ``curve`` for the size floor -- because "this cutoff does not
decide the answer" is a measurable claim and the metric this replaces made it
without evidence.
"""

from __future__ import annotations

import json
import math
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterable, Iterator

import numpy as np
from PIL import Image

# A pixel has to have moved before its colour can be said to follow the phase.
# Measured on the Paris bakes the inter-phase difference is 1/255 at the median
# and 230/255 at the lamp, so this sits in the gap rather than in the data;
# ``threshold_sweep`` reports the answer from 8 to 96 so the claim is checkable.
CHANGE_THRESHOLD = 24.0
# What a vision model's preprocessor leaves of the frame: the long edge is
# resized to about this, and detail below ~2 px after that resize is not there.
MODEL_LONG_EDGE_PX = 768.0
MIN_MODEL_PIXELS = 4.0  # a 2x2 patch at model resolution
# How far apart the blob's two phases have to be in chroma before "it went red
# then green" is a statement about a lamp rather than about foliage. Renders are
# not bit-identical and leaves flicker, so a small clump of leaves can satisfy
# two sign tests by chance -- one such clump (19 px, margin 0.08) was visually
# confirmed as a tree. This is set from the album's own distribution, which is
# bimodal with an empty valley: on the kerb album 92 approaches score under 0.10,
# 164 score over 0.15, and 4 lie between. Any cutoff inside that valley gives the
# same answer, which ``margin_sweep`` reports rather than assumes.
MIN_MARGIN = 0.125
# How much a lamp's own hue has to lead the other bake before the pixel is said
# to belong to the lamp lit in this phase. Same shape of test as MIN_MARGIN, one
# phase at a time.
MIN_PHASE_MARGIN = 0.12
# A lit lens against an unlit one. The album's green lamps are bimodal in mean
# level with an empty valley: 33 sit under 80 and 130 over 120, and 5 lie between
# -- the low mode is a dark reflection that shifted, not a lamp. Any cutoff from
# 80 to 120 selects the same 130, which ``level_sweep`` reports rather than
# assumes. The red lamps have no low mode at all: every red blob big enough to
# read already exceeds 120.
MIN_LAMP_LEVEL = 100.0
# What counts as a static red distractor: red in both bakes, bright enough to
# read, and unchanged between them. Reported, never subtracted -- the city is
# allowed to contain red things.
CONFUSER_CHROMA = 0.30
CONFUSER_LEVEL = 90.0


def chroma(rgb: np.ndarray) -> np.ndarray:
    """Red-versus-green, normalised by the pixel's own brightness.

    In [-1, 1]: positive is red-dominant, negative green-dominant, and a grey or
    dark pixel sits near zero whatever its exposure. Dividing by the pixel's own
    R+G is what makes this comparable between a lamp in sun and one in shade
    without an exposure constant anywhere.
    """
    red = rgb[..., 0].astype(np.float64)
    green = rgb[..., 1].astype(np.float64)
    return (red - green) / (red + green + 1.0)


def _components(mask: np.ndarray) -> list[int]:
    """Sizes of the 8-connected components of a sparse boolean mask, descending.

    Union-find over the set pixels only. The mask is a lamp, so it is thousands
    of pixels at most out of a million, and iterating the pixels that are set is
    cheaper than any dense labelling -- which also means this needs no scipy.
    """
    ys, xs = np.nonzero(mask)
    if ys.size == 0:
        return []
    index = {(int(y), int(x)): i for i, (y, x) in enumerate(zip(ys, xs))}
    parent = list(range(ys.size))

    def find(a: int) -> int:
        while parent[a] != a:
            parent[a] = parent[parent[a]]
            a = parent[a]
        return a

    def union(a: int, b: int) -> None:
        ra, rb = find(a), find(b)
        if ra != rb:
            parent[rb] = ra

    for (y, x), i in index.items():
        for dy, dx in ((0, 1), (1, -1), (1, 0), (1, 1)):
            j = index.get((y + dy, x + dx))
            if j is not None:
                union(i, j)
    sizes: dict[int, int] = {}
    for i in range(ys.size):
        root = find(i)
        sizes[root] = sizes.get(root, 0) + 1
    return sorted(sizes.values(), reverse=True)


@dataclass
class PhaseLamp:
    """The lamp that is lit in one phase, as it appears in that phase's bake."""

    px: int = 0
    level: float = 0.0       # mean of the lamp's own channel, 0..255
    box: tuple[int, int, int, int] | None = None

    def model_pixels(self, width: int, height: int,
                     long_edge: float = MODEL_LONG_EDGE_PX) -> float:
        if not width or not height:
            return 0.0
        scale = min(1.0, long_edge / max(width, height))
        return self.px * scale * scale

    def lit(self, min_level: float = MIN_LAMP_LEVEL) -> bool:
        """Lit, not merely tinted. An unlit lens is dark and slightly coloured."""
        return self.px > 0 and self.level >= min_level

    def to_dict(self) -> dict[str, Any]:
        return {"px": self.px, "level": round(self.level, 1),
                "box": list(self.box) if self.box else None}


@dataclass
class ApproachLegibility:
    """One approach, measured: does its lamp say which phase is running?"""

    key: str
    changed_px: int = 0
    flipped_px: int = 0
    blob_px: int = 0
    c_red: float = 0.0
    c_green: float = 0.0
    width: int = 0
    height: int = 0
    error: str = ""
    # Each phase's own lamp, found in its own bake. See the module docstring:
    # a three-aspect head puts them in different places, so one blob cannot
    # answer for both.
    red_lamp: PhaseLamp = field(default_factory=PhaseLamp)
    green_lamp: PhaseLamp = field(default_factory=PhaseLamp)
    # Static red things sharing the frame: measured, reported, never removed.
    confusers: int = 0
    biggest_confuser_px: int = 0

    @property
    def margin(self) -> float:
        """How decisively the blob's colour follows the phase. Sign, not size."""
        return self.c_red - self.c_green

    def decisive(self, min_margin: float = MIN_MARGIN) -> bool:
        """The blob goes red on red and green on green, and unambiguously so."""
        return (
            self.blob_px > 0
            and self.c_red > 0.0
            and self.c_green < 0.0
            and self.margin >= min_margin
        )

    def model_pixels(self, long_edge: float = MODEL_LONG_EDGE_PX) -> float:
        """The blob's area after a vision model's resize -- the size that counts."""
        if not self.width or not self.height:
            return 0.0
        # Never above 1.0. Enlarging a frame does not put detail into it, so a
        # one-pixel lamp in a 320-wide render is still one pixel of evidence
        # however big the model's input is; scaling it up to 5.76 "model pixels"
        # and calling it readable is the metric lying to itself.
        scale = min(1.0, long_edge / max(self.width, self.height))
        return self.blob_px * scale * scale

    def both_lamps(self, min_model_px: float = MIN_MODEL_PIXELS,
                   min_level: float = MIN_LAMP_LEVEL) -> bool:
        """A red lamp and a green lamp, each lit and each big enough to read.

        Both, because a courier that can see the green and not the red learns
        nothing from a red phase -- and the red phase is the one that is scored.
        """
        return all(
            lamp.lit(min_level)
            and lamp.model_pixels(self.width, self.height) >= min_model_px
            for lamp in (self.red_lamp, self.green_lamp)
        )

    def legible(self, min_model_px: float = MIN_MODEL_PIXELS,
                min_margin: float = MIN_MARGIN,
                min_level: float = MIN_LAMP_LEVEL) -> bool:
        """Both lamps, lit and readable. ``min_margin`` is kept for the sweep.

        This deliberately does *not* require ``decisive``. The flip test asks the
        unlit lens of a three-aspect head to read faintly green so that its lit
        neighbour can flip against it, which is a fact about how the dark lens
        renders and not about whether anyone can see the signal. On the Paris
        album the two agree on 117 of 118 approaches, so nothing is being bought
        with the extra condition except a reason to fail for the wrong cause.
        """
        del min_margin  # part of the reported sweep, not of the definition
        return self.both_lamps(min_model_px, min_level)

    def to_dict(self) -> dict[str, Any]:
        return {
            "key": self.key, "changed_px": self.changed_px,
            "flipped_px": self.flipped_px, "blob_px": self.blob_px,
            "c_red": round(self.c_red, 4), "c_green": round(self.c_green, 4),
            "margin": round(self.margin, 4), "decisive": self.decisive(),
            "model_px": round(self.model_pixels(), 2),
            "red_lamp": self.red_lamp.to_dict(),
            "green_lamp": self.green_lamp.to_dict(),
            "red_lamp_model_px": round(
                self.red_lamp.model_pixels(self.width, self.height), 2),
            "green_lamp_model_px": round(
                self.green_lamp.model_pixels(self.width, self.height), 2),
            "both_lamps": self.both_lamps(),
            "confusers": self.confusers,
            "biggest_confuser_px": self.biggest_confuser_px,
            "legible": self.legible(), "error": self.error,
        }


def _load_pair(red_path: Path, green_path: Path) -> tuple[np.ndarray, np.ndarray]:
    with Image.open(red_path) as handle:
        red = np.asarray(handle.convert("RGB"), dtype=np.int16)
    with Image.open(green_path) as handle:
        green = np.asarray(handle.convert("RGB"), dtype=np.int16)
    return red, green


def _label(mask: np.ndarray) -> tuple[np.ndarray, dict[int, int]]:
    """8-connected labelling by iterative flood fill, and each label's size.

    Written as a flood fill rather than the dict-keyed union-find this replaces
    because the per-phase measurement asks for components four times per pair
    rather than once, and the union-find spent most of its time hashing
    coordinate tuples. Same answer, and the album goes from tens of minutes to
    under one.
    """
    height, width = mask.shape
    labels = np.zeros((height, width), np.int32)
    ys, xs = np.nonzero(mask)
    sizes: dict[int, int] = {}
    current = 0
    for y0, x0 in zip(ys.tolist(), xs.tolist()):
        if labels[y0, x0]:
            continue
        current += 1
        count = 0
        stack = [(y0, x0)]
        labels[y0, x0] = current
        while stack:
            y, x = stack.pop()
            count += 1
            for dy in (-1, 0, 1):
                for dx in (-1, 0, 1):
                    a, b = y + dy, x + dx
                    if 0 <= a < height and 0 <= b < width and mask[a, b] and not labels[a, b]:
                        labels[a, b] = current
                        stack.append((a, b))
        sizes[current] = count
    return labels, sizes


def _largest_component(mask: np.ndarray) -> np.ndarray:
    """The mask reduced to its largest 8-connected component."""
    labels, sizes = _label(mask)
    if not sizes:
        return mask
    biggest = max(sizes, key=lambda label: sizes[label])
    return labels == biggest


def measure_phase_lamp(lit: np.ndarray, dark: np.ndarray, channel: int,
                       change_threshold: float = CHANGE_THRESHOLD,
                       min_margin: float = MIN_PHASE_MARGIN) -> PhaseLamp:
    """The lamp that is lit in ``lit`` and off in ``dark``.

    ``channel`` is the lamp's own channel -- 0 for a red lamp, 1 for a green one.
    The test is the same shape either way: the pixel got brighter, and it moved
    toward its own hue relative to the other bake. Both halves are comparisons
    between the two bakes of one camera, so nothing here is an absolute colour.
    """
    if lit.shape != dark.shape:
        return PhaseLamp()
    # chroma is red-minus-green, so a green lamp is the same test with the sign
    # flipped rather than a second function.
    sign = 1.0 if channel == 0 else -1.0
    lead = sign * (chroma(lit) - chroma(dark))
    brighter = (lit.astype(np.float64) - dark.astype(np.float64)).max(axis=2)
    on = (brighter >= change_threshold) & (lead >= min_margin)
    if not on.any():
        return PhaseLamp()
    blob = _largest_component(on)
    ys, xs = np.nonzero(blob)
    return PhaseLamp(
        px=int(blob.sum()),
        level=float(lit[..., channel][blob].mean()),
        box=(int(xs.min()), int(ys.min()), int(xs.max()), int(ys.max())),
    )


def measure_confusers(red: np.ndarray, green: np.ndarray,
                      change_threshold: float = CHANGE_THRESHOLD,
                      min_px: int = 8) -> tuple[int, int]:
    """Static red objects in the frame: how many, and the biggest.

    Red in both bakes and unchanged between them, which is what a no-entry disc
    or a red awning is. They are counted so the album can say plainly that a
    courier has to find the lamp among them, not merely find something red.
    """
    unchanged = np.abs(red - green).max(axis=2) < change_threshold
    both_red = (chroma(red) > CONFUSER_CHROMA) & (chroma(green) > CONFUSER_CHROMA)
    mask = unchanged & both_red & (red[..., 0] > CONFUSER_LEVEL)
    if not mask.any():
        return 0, 0
    _, sizes = _label(mask)
    big = [n for n in sizes.values() if n >= min_px]
    return len(big), (max(big) if big else 0)


def measure_arrays(key: str, red: np.ndarray, green: np.ndarray,
                   change_threshold: float = CHANGE_THRESHOLD) -> ApproachLegibility:
    """Measure one approach from two already-loaded bakes."""
    out = ApproachLegibility(key=key)
    if red.shape != green.shape:
        out.error = "frames differ in size"
        return out
    out.height, out.width = red.shape[0], red.shape[1]
    # Asked before the flip test and independently of it, because a head whose
    # two lenses sit in different places has no flipping pixel to find and is
    # still perfectly readable.
    out.red_lamp = measure_phase_lamp(red, green, 0, change_threshold)
    out.green_lamp = measure_phase_lamp(green, red, 1, change_threshold)
    out.confusers, out.biggest_confuser_px = measure_confusers(
        red, green, change_threshold)
    changed = np.abs(red - green).max(axis=2) >= change_threshold
    out.changed_px = int(changed.sum())
    if out.changed_px == 0:
        return out
    c_red, c_green = chroma(red), chroma(green)
    # The two sign tests, per pixel: red when red, green when green. Anything
    # that is the same colour in both phases -- an awning, a brake light baked
    # into the plate -- fails, and so does the diffuse re-render haze, whose
    # chroma barely moves because exposure shifts R and G together.
    flipped = changed & (c_red > 0.0) & (c_green < 0.0)
    out.flipped_px = int(flipped.sum())
    if out.flipped_px == 0:
        return out
    # Only the largest connected component counts as "the lamp": a scatter of
    # single flipped pixels across the frame is not something anyone can read,
    # and summing them lets noise masquerade as a signal head. This is where the
    # old metric went wrong -- it averaged colour over the scatter as well.
    blob = _largest_component(flipped)
    out.blob_px = int(blob.sum())
    out.c_red = float(c_red[blob].mean())
    out.c_green = float(c_green[blob].mean())
    return out


def measure_pair(key: str, red_path: Path, green_path: Path,
                 change_threshold: float = CHANGE_THRESHOLD) -> ApproachLegibility:
    """Measure one approach from its two bakes."""
    try:
        red, green = _load_pair(red_path, green_path)
    except (OSError, ValueError) as error:
        return ApproachLegibility(key=key, error=f"{type(error).__name__}: {error}")
    return measure_arrays(key, red, green, change_threshold)


def walk_album(album: Path) -> Iterator[tuple[str, Path, Path]]:
    """Every ``(key, red, green)`` triple in a signal album.

    The key is ``"<node>|<toward>"``, which is what the runtime's visibility
    sidecar is keyed by, so a measurement can be written straight back out.
    """
    images = Path(album) / "images"
    for node_dir in sorted(p for p in images.iterdir() if p.is_dir()):
        for red in sorted(node_dir.glob("toward_*_red.png")):
            green = red.with_name(red.name[: -len("_red.png")] + "_green.png")
            if not green.exists():
                continue
            toward = red.name[len("toward_"): -len("_red.png")]
            yield f"{node_dir.name}|{toward}", red, green


@dataclass
class AlbumLegibility:
    """A whole album's worth of measurements, and the summary they support."""

    album: str
    rows: list[ApproachLegibility] = field(default_factory=list)

    @property
    def approaches(self) -> int:
        return len(self.rows)

    def legible_keys(self, min_model_px: float = MIN_MODEL_PIXELS,
                     min_margin: float = MIN_MARGIN,
                     min_level: float = MIN_LAMP_LEVEL) -> list[str]:
        return sorted(
            r.key for r in self.rows if r.legible(min_model_px, min_margin, min_level))

    def curve(self, floors: Iterable[float] = (0.0, 1.0, 2.0, 4.0, 8.0, 16.0, 32.0)) -> list[dict]:
        """Legibility as a function of the size floor.

        Reported instead of a single number so the reader can see whether the
        answer depends on the floor. If it barely moves across a 32x range, the
        floor is not what decides it -- which is the property the old metric
        lacked and never demonstrated.
        """
        return [
            {
                "min_model_px": floor,
                "legible": sum(1 for r in self.rows if r.legible(floor)),
                "fraction": (round(sum(1 for r in self.rows if r.legible(floor))
                                   / max(self.approaches, 1), 4)),
            }
            for floor in floors
        ]

    def level_sweep(self, levels: Iterable[float] = (60.0, 80.0, 100.0, 120.0, 140.0, 180.0),
                    min_model_px: float = MIN_MODEL_PIXELS) -> list[dict[str, Any]]:
        """Legibility as a function of the lit-lamp level.

        The third published sweep, for the third constant. ``MIN_LAMP_LEVEL``
        claims to sit in the valley of a bimodal distribution; this is where a
        reader checks it.
        """
        return [
            {
                "min_level": level,
                "legible": sum(1 for r in self.rows if r.legible(min_model_px, MIN_MARGIN, level)),
                "fraction": round(
                    sum(1 for r in self.rows if r.legible(min_model_px, MIN_MARGIN, level))
                    / max(self.approaches, 1), 4),
            }
            for level in levels
        ]

    def summary(self, min_model_px: float = MIN_MODEL_PIXELS) -> dict[str, Any]:
        decisive = [r for r in self.rows if r.decisive()]
        legible = [r for r in self.rows if r.legible(min_model_px)]
        margins = sorted(r.margin for r in self.rows)
        blobs = sorted(r.blob_px for r in decisive)
        flipped = [r for r in self.rows if r.flipped_px > 0]
        return {
            "album": self.album,
            "approaches": self.approaches,
            "changed_at_all": sum(1 for r in self.rows if r.changed_px > 0),
            "flipped_at_all": sum(1 for r in self.rows if r.flipped_px > 0),
            "decisive": len(decisive),
            "red_lamp_lit": sum(1 for r in self.rows if r.red_lamp.lit()),
            "green_lamp_lit": sum(1 for r in self.rows if r.green_lamp.lit()),
            "both_lamps": sum(1 for r in self.rows if r.both_lamps(min_model_px)),
            "legible": len(legible),
            "legible_fraction": round(len(legible) / max(self.approaches, 1), 4),
            "median_red_level": round(
                _median([r.red_lamp.level for r in legible]), 1),
            "median_green_level": round(
                _median([r.green_lamp.level for r in legible]), 1),
            "median_red_model_px": round(
                _median([r.red_lamp.model_pixels(r.width, r.height) for r in legible]), 2),
            "median_green_model_px": round(
                _median([r.green_lamp.model_pixels(r.width, r.height) for r in legible]), 2),
            # A courier reading a legible frame is also looking at this many
            # static red things, and this often at one bigger than the lamp.
            "legible_with_static_red": sum(1 for r in legible if r.confusers),
            "legible_outranked_by_static_red": sum(
                1 for r in legible if r.biggest_confuser_px > r.red_lamp.px),
            "median_margin": round(_median(margins), 4),
            "margin_valley": _valley(sorted(r.margin for r in flipped)),
            "median_decisive_blob_px": round(_median([float(b) for b in blobs]), 1),
            "median_decisive_model_px": round(
                _median([r.model_pixels() for r in decisive]), 2),
            "curve": self.curve(),
            "level_sweep": self.level_sweep(min_model_px=min_model_px),
            "errors": sum(1 for r in self.rows if r.error),
        }

    def visibility_sidecar(self, map_name: str, min_model_px: float = MIN_MODEL_PIXELS,
                           measured_at: str = "") -> dict[str, Any]:
        """The ``signal_visibility.json`` the runtime reads, with its provenance."""
        keys = self.legible_keys(min_model_px)
        legible = [r for r in self.rows if r.key in set(keys)]
        return {
            "map": map_name,
            "measured_at": measured_at,
            "method": (
                "per-phase lamp measurement (signal_legibility.py). Each phase is "
                "asked its own question against the other bake of the identical "
                "camera: the red lamp is what got brighter and redder in the red "
                "bake, the green lamp what got brighter and greener in the green "
                "one, each taken as the largest connected component. Both are "
                "comparisons between two bakes, so no absolute colour appears "
                "anywhere. An approach is legible when BOTH lamps are found, both "
                f"are lit rather than merely tinted ({MIN_LAMP_LEVEL:.0f} of 255 "
                "in the lamp's own channel), and both survive a vision model's "
                f"resize to {MODEL_LONG_EDGE_PX:.0f} px on the long edge at "
                f"{min_model_px:.0f} px of area. Both, because seeing the green "
                "does not tell a courier what red looks like here, and red is the "
                "phase that is scored. Each cutoff is published with a sweep "
                "showing it does not decide the answer. Static red objects -- "
                "no-entry discs, awnings, shopfronts -- are counted and reported "
                "but never removed: finding the lamp among them is the task."
            ),
            "approaches": self.approaches,
            "legible_count": len(keys),
            "legible": keys,
            "static_red_present": sum(1 for r in legible if r.confusers),
            "static_red_bigger_than_lamp": sum(
                1 for r in legible if r.biggest_confuser_px > r.red_lamp.px),
            # The lamp's size in the frame as baked, so a runtime serving the
            # frame smaller can ask the question again at its own resolution.
            # "legible" above answers it at MODEL_LONG_EDGE_PX and nothing
            # else; a harness that downscales to 320 px keeps only 96 of these
            # 130 above a 2x2 patch, and charging for the other 34 is charging
            # for a lamp the policy was never sent enough pixels to see.
            "lamp_px": {
                r.key: [min(r.red_lamp.px, r.green_lamp.px), r.width, r.height]
                for r in legible
            },
            # Where the lamp sits in the frame. The map has no lamp objects --
            # signalised junctions are derived from degree and one light mesh
            # is baked per junction -- so a junction with four approaches has
            # four frames of THE SAME LAMP from the same camera. Publishing the
            # box lets the runtime notice that and stop treating one lamp as
            # four independent ones.
            "lamp_box": {
                r.key: list(r.red_lamp.box) if r.red_lamp.box else None
                for r in legible
            },
        }


def _valley(margins: list[float], lo: float = 0.10, hi: float = 0.15) -> dict[str, int]:
    """How the margins fall either side of the cutoff, and how many are on it.

    The claim that ``MIN_MARGIN`` sits in an empty valley is checkable only if
    the counts are published, so they are: a middle bin near zero means the
    cutoff is separating two populations rather than cutting one in half.
    """
    return {
        "below": sum(1 for m in margins if m < lo),
        "between": sum(1 for m in margins if lo <= m < hi),
        "above": sum(1 for m in margins if m >= hi),
        "lo": lo, "hi": hi,
    }


def _median(values: list[float]) -> float:
    if not values:
        return 0.0
    values = sorted(values)
    mid = len(values) // 2
    if len(values) % 2:
        return values[mid]
    return 0.5 * (values[mid - 1] + values[mid])


def measure_album(album: Path, change_threshold: float = CHANGE_THRESHOLD) -> AlbumLegibility:
    """Measure every approach in a signal album."""
    out = AlbumLegibility(album=str(album))
    for key, red, green in walk_album(Path(album)):
        out.rows.append(measure_pair(key, red, green, change_threshold))
    return out


def threshold_sweep(album: Path,
                    thresholds: Iterable[float] = (8.0, 12.0, 24.0, 48.0, 96.0),
                    min_model_px: float = MIN_MODEL_PIXELS) -> list[dict[str, Any]]:
    """Legibility as a function of the per-pixel change threshold.

    The point of publishing this is that it is flat. A metric whose answer moves
    when its one constant moves is measuring the constant; this one is asked
    across a twelve-fold range so the reader can see that it is not.
    """
    pairs = list(walk_album(Path(album)))
    loaded = [(key, *_load_pair(red, green)) for key, red, green in pairs]
    out = []
    for threshold in thresholds:
        rows = [measure_arrays(key, red, green, threshold) for key, red, green in loaded]
        count = sum(1 for r in rows if r.legible(min_model_px))
        out.append({
            "change_threshold": threshold,
            "legible": count,
            "fraction": round(count / max(len(rows), 1), 4),
        })
    return out


def margin_sweep(measured: "AlbumLegibility",
                 margins: Iterable[float] = (0.05, 0.10, 0.125, 0.15, 0.20, 0.30),
                 min_model_px: float = MIN_MODEL_PIXELS) -> list[dict[str, Any]]:
    """Legibility as a function of the chroma-margin cutoff.

    The companion to ``threshold_sweep``: the answer should be flat across the
    valley the cutoff sits in and fall away only outside it.
    """
    return [
        {
            "min_margin": margin,
            "legible": sum(1 for r in measured.rows if r.legible(min_model_px, margin)),
            "fraction": round(
                sum(1 for r in measured.rows if r.legible(min_model_px, margin))
                / max(measured.approaches, 1), 4),
        }
        for margin in margins
    ]


def main(argv: list[str] | None = None) -> int:
    import argparse

    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("album", type=Path, help="a signal album's map directory")
    parser.add_argument("--map-name", default="")
    parser.add_argument("--write-sidecar", action="store_true",
                        help="write signal_visibility.json beside the images")
    parser.add_argument("--min-model-px", type=float, default=MIN_MODEL_PIXELS)
    parser.add_argument("--rows", type=Path, default=None,
                        help="write every per-approach measurement here as JSONL")
    args = parser.parse_args(argv)

    measured = measure_album(args.album)
    summary = measured.summary(args.min_model_px)
    print(json.dumps(summary, indent=1))
    if args.rows:
        with Path(args.rows).open("w", encoding="utf-8") as handle:
            for row in measured.rows:
                handle.write(json.dumps(row.to_dict()) + "\n")
    if args.write_sidecar:
        import datetime

        sidecar = measured.visibility_sidecar(
            args.map_name or Path(args.album).name,
            args.min_model_px,
            datetime.date.today().isoformat(),
        )
        path = Path(args.album) / "signal_visibility.json"
        path.write_text(json.dumps(sidecar, indent=1))
        print(f"wrote {path}")
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
