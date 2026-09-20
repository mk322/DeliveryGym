"""Content-addressed store for observation images.

design plan §10.4 requires trajectories to carry "hashes linking large images/video/
snapshots in artifact storage" rather than inline pixels, and design plan §5.1
requires artifact paths to be relative and portable.

Content addressing matters more here than it looks. A cached RGB-D rollout
revisits the same nodes constantly, so the same frame recurs many times per
episode and across episodes in a batch. Keying by sha256 means each distinct
frame is written once no matter how often it is observed, which is the
difference between a bounded media directory and one that grows with step count.
"""

from __future__ import annotations

import io
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from embodiedbench.artifacts.hashing import sha256_bytes


@dataclass
class MediaStore:
    """Writes images under ``root`` keyed by content hash."""

    root: Path
    image_format: str = "PNG"
    _written: set[str] = field(default_factory=set)
    writes: int = 0
    deduplicated: int = 0

    def __post_init__(self) -> None:
        self.root = Path(self.root)
        self.root.mkdir(parents=True, exist_ok=True)

    @property
    def extension(self) -> str:
        return "png" if self.image_format.upper() == "PNG" else "jpg"

    def _encode(self, image: Any) -> bytes:
        buffer = io.BytesIO()
        # PNG so a stored frame is byte-identical to what the policy saw; a lossy
        # re-encode would make the trajectory's image hash describe a different
        # image than the one the model was shown.
        image.save(buffer, format=self.image_format)
        return buffer.getvalue()

    def put(self, image: Any) -> tuple[str, str]:
        """Store an image, returning ``(relative_path, sha256)``."""
        payload = self._encode(image)
        digest = sha256_bytes(payload)
        # Shard by the first two hex characters so one directory does not end up
        # with hundreds of thousands of entries.
        relpath = f"{digest[:2]}/{digest}.{self.extension}"
        if digest in self._written:
            self.deduplicated += 1
            return relpath, digest
        target = self.root / relpath
        if target.exists():
            self._written.add(digest)
            self.deduplicated += 1
            return relpath, digest
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(payload)
        self._written.add(digest)
        self.writes += 1
        return relpath, digest

    def path_for(self, relpath: str) -> Path:
        return self.root / relpath

    def stats(self) -> dict[str, int]:
        return {
            "distinct_images": len(self._written),
            "writes": self.writes,
            "deduplicated": self.deduplicated,
        }
