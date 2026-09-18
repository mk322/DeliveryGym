"""Map compiler: inspect -> extract -> normalize -> annotate -> validate.

design plan §6 makes the existing exporter the first structure-aware producer. At M1
only the procgen producer exists, and it exists so the walking skeleton runs
against a real ``WorldBundle`` rather than a hand-written stub that could agree
with nothing.
"""

from embodiedbench.compiler.procgen import compile_procgen_world

__all__ = ["compile_procgen_world"]
