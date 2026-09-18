"""Policy-visible names for album frames, carrying no semantics.

The albums name their files after what is in them -- ``toward_s004_n008_road_block.png``,
``toward_s005_n011_red.png`` -- which is exactly right for a human checking a bake
and exactly wrong for the thing the courier is handed. Measured on the shipped
Paris albums, **39.9% of the frames served to a policy had the answer in the
path**, and ``docs/RUNNING.md`` tells integrators to pass those paths straight to
the model:

    images=[f.path for f in observation.frames if f.kind == "photograph"]

So a text-only policy that never decodes a pixel could read ``road_block`` off
the filename and score as though it had perfect sight -- which is the
``sighted=True`` reference arm, the very thing a vision policy is supposed to be
measured against. Two independent reviewers found this.

The fix is a rename at the harness boundary, not in the albums: the album keeps
its readable names, ``CourierEnv.candidates()`` keeps returning them (the
privileged reference policies are *documented* to read the frame's name as a
stand-in for perfect recognition, and they must go on being able to), and only
the frames handed to a policy are renamed.

Names are content-addressed, so the same picture is always the same name --
episodes replay, trajectories stay comparable, and a frame served twice is one
file. What the name no longer does is say what the picture contains.
"""

from __future__ import annotations

import getpass
import os
import tempfile
from pathlib import Path

from embodiedbench.artifacts.hashing import sha256_file


def default_root() -> Path:
    """Where aliases live. Overridable so a harness can put them on fast disk.

    Per user, because /tmp is not. The first person to run this on a shared
    machine created /tmp/embodiedbench_frames owned by themselves, and the
    second got

        PermissionError: [Errno 13] Permission denied:
        '/tmp/embodiedbench_frames/56bacfe52e5dd539.png'

    on the first frame of the first episode -- a run that dies at startup for
    a reason that has nothing to do with the run.
    """
    override = os.environ.get("EMBODIEDBENCH_FRAME_CACHE")
    if override:
        return Path(override)
    try:
        who = getpass.getuser()
    except Exception:  # noqa: BLE001 - no passwd entry in some containers
        who = str(os.getuid())
    return Path(tempfile.gettempdir()) / f"embodiedbench_frames_{who}"


class FrameAliases:
    """Maps an album path to an opaque, stable, policy-safe path.

    Linked rather than copied: the albums are ~600 MB and a copy per run is a
    cost with no benefit. A hard link is preferred because it leaves no trace of
    the original name; a symlink is the fallback when the cache and the album sit
    on different filesystems, which is the usual case.
    """

    def __init__(self, root: Path | None = None):
        self.root = Path(root) if root is not None else default_root()
        self._by_path: dict[str, str] = {}
        self._ready = False

    def _ensure_root(self) -> None:
        if not self._ready:
            self.root.mkdir(parents=True, exist_ok=True)
            self._ready = True

    def alias(self, path: str | None) -> str:
        """The name a policy may see for ``path``. Empty in, empty out."""
        if not path:
            return ""
        cached = self._by_path.get(path)
        if cached is not None:
            return cached

        source = Path(path)
        if not source.exists():
            # A frame the album promised and did not deliver is the album's
            # problem, not this one's. Passing the original through would leak;
            # dropping it silently would hide a bad bake. Neither: keep it
            # nameless and let the caller's own coverage check fail loudly.
            self._by_path[path] = ""
            return ""

        self._ensure_root()
        digest = sha256_file(source)[:16]
        alias = self.root / f"{digest}{source.suffix.lower() or '.png'}"
        if alias.is_symlink() and not alias.exists():
            # A link left by an earlier run whose album has since moved: it
            # answers exists() with False and would have been written through.
            alias.unlink()
        if not alias.exists():
            try:
                os.link(source, alias)
            except FileExistsError:
                pass  # a concurrent worker aliased the same frame first
            except OSError:
                try:
                    alias.symlink_to(source.resolve())
                except FileExistsError:
                    pass
                except OSError:
                    # Last resort: copy -- into a private temp name and then an
                    # atomic rename, so no write ever goes through a path that
                    # may already be a link to the album's own file.
                    tmp = alias.with_name(f"{alias.name}.{os.getpid()}.tmp")
                    tmp.write_bytes(source.read_bytes())
                    os.replace(tmp, alias)

        out = str(alias)
        self._by_path[path] = out
        return out
