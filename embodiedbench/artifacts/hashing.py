"""Content hashing for reproducible artifact identity.

Two digest levels exist because the source content packs are large (CityCore
Paris alone is 7.3 GB over ~3k files) and milestone gates must stay runnable as
a single non-interactive command:

``structure`` digest
    sha256 over sorted ``relpath\\0size`` records. Cheap (a stat walk), catches
    added/removed/renamed/resized files. Does *not* catch in-place edits that
    preserve size.

``content`` digest
    sha256 over sorted ``relpath\\0sha256(bytes)`` records. Reads every byte.
    Authoritative, and what a release pins.

Both are stable under directory relocation, which design plan §5.1 requires ("no
absolute machine-specific paths in portable artifacts"). Symlinks are recorded
by their target string rather than followed, so a re-pointed symlink is a
detectable change instead of a silent one.
"""

from __future__ import annotations

import hashlib
import json
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Iterator

CHUNK_BYTES = 1 << 20
EMPTY_TREE_DIGEST = hashlib.sha256(b"").hexdigest()


def sha256_file(path: Path) -> str:
    """sha256 of a single file's bytes."""
    h = hashlib.sha256()
    with path.open("rb") as fh:
        while chunk := fh.read(CHUNK_BYTES):
            h.update(chunk)
    return h.hexdigest()


def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def canonical_json(obj: Any) -> bytes:
    """Deterministic JSON encoding used wherever a dict must be hashed.

    Sorted keys, no insignificant whitespace, UTF-8, no NaN/Infinity (which are
    not valid JSON and would not round-trip through other consumers).
    """
    return json.dumps(
        obj, sort_keys=True, separators=(",", ":"), ensure_ascii=False, allow_nan=False
    ).encode("utf-8")


def sha256_json(obj: Any) -> str:
    return hashlib.sha256(canonical_json(obj)).hexdigest()


@dataclass(frozen=True)
class TreeEntry:
    """One filesystem entry in a walked tree."""

    relpath: str
    size: int
    kind: str  # "file" | "symlink"
    target: str | None = None  # symlink target, verbatim and unresolved


@dataclass(frozen=True)
class TreeDigest:
    """Digest of a directory tree at one of the two levels."""

    level: str  # "structure" | "content"
    digest: str
    file_count: int
    total_bytes: int

    def to_dict(self) -> dict[str, Any]:
        return {
            "level": self.level,
            "digest": self.digest,
            "file_count": self.file_count,
            "total_bytes": self.total_bytes,
        }


def walk_tree(
    root: Path, *, exclude_dirs: Iterable[str] = (), follow_symlinks: bool = False
) -> Iterator[TreeEntry]:
    """Yield tree entries under ``root`` in unspecified order, relative to root.

    ``exclude_dirs`` matches directory *basenames* at any depth (e.g. ``.git``,
    ``__pycache__``).
    """
    excluded = set(exclude_dirs)
    root = root.resolve() if follow_symlinks else Path(os.path.abspath(root))

    for dirpath, dirnames, filenames in os.walk(root, followlinks=follow_symlinks):
        dirnames[:] = sorted(d for d in dirnames if d not in excluded)
        for name in sorted(filenames):
            full = Path(dirpath) / name
            rel = os.path.relpath(full, root)
            if full.is_symlink() and not follow_symlinks:
                yield TreeEntry(rel, 0, "symlink", os.readlink(full))
            else:
                try:
                    size = full.stat().st_size
                except OSError:
                    # Broken symlink or a file that vanished mid-walk. Record it
                    # as a zero-length entry rather than aborting the digest, so
                    # the difference still shows up as a digest change.
                    yield TreeEntry(rel, 0, "file")
                    continue
                yield TreeEntry(rel, size, "file")


def _record(entry: TreeEntry, per_file_digest: str | None) -> bytes:
    if entry.kind == "symlink":
        return f"{entry.relpath}\0symlink\0{entry.target}\n".encode("utf-8")
    if per_file_digest is None:
        return f"{entry.relpath}\0{entry.size}\n".encode("utf-8")
    return f"{entry.relpath}\0{per_file_digest}\n".encode("utf-8")


def digest_tree(
    root: Path,
    *,
    level: str = "structure",
    exclude_dirs: Iterable[str] = (),
    follow_symlinks: bool = False,
) -> TreeDigest:
    """Digest a directory tree at ``structure`` or ``content`` level.

    Raises ``FileNotFoundError`` if ``root`` does not exist, so a missing
    baseline is a hard failure rather than an empty-tree digest that silently
    matches another empty tree.
    """
    if level not in ("structure", "content"):
        raise ValueError(f"unknown digest level: {level!r}")
    root = Path(root)
    if not root.exists():
        raise FileNotFoundError(root)
    if root.is_file():
        size = root.stat().st_size
        digest = sha256_file(root) if level == "content" else sha256_bytes(str(size).encode())
        return TreeDigest(level, digest, 1, size)

    entries = sorted(
        walk_tree(root, exclude_dirs=exclude_dirs, follow_symlinks=follow_symlinks),
        key=lambda e: e.relpath,
    )
    h = hashlib.sha256()
    total = 0
    for entry in entries:
        per_file = None
        if level == "content" and entry.kind == "file":
            try:
                per_file = sha256_file(root / entry.relpath)
            except OSError:
                per_file = EMPTY_TREE_DIGEST
        h.update(_record(entry, per_file))
        total += entry.size
    return TreeDigest(level, h.hexdigest(), len(entries), total)
