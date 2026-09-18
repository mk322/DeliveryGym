#!/usr/bin/env python3
"""Fail-closed identity checks for the front/rear probe's UE process.

The launcher always supplies the project target as ``--expected-editor``.
Tests may point the helper at an isolated copied build bundle, but this module
does not select or launch an executable itself.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import re
import shutil
import stat
import subprocess
import sys
import time
from collections.abc import Mapping
from pathlib import Path
from typing import Any


MODULE_NAME = "libSimWorldEditor-SimWorld.so"
BUILD_ID = re.compile(
    r"^[0-9a-fA-F]{8}-[0-9a-fA-F]{4}-[0-9a-fA-F]{4}-"
    r"[0-9a-fA-F]{4}-[0-9a-fA-F]{12}$"
)
REQUIRED_EDITOR_LIBRARIES = {
    "libSimWorldEditor-Core.so",
    "libSimWorldEditor-CoreUObject.so",
    "libSimWorldEditor-Engine.so",
}
REQUIRED_MODULE_LIBRARIES = {
    "libSimWorldEditor-Core.so",
    "libSimWorldEditor-CoreUObject.so",
    "libSimWorldEditor-SpCore.so",
}
APPROVED_BUNDLE_SHA256 = {
    "SimWorldEditor":
        "f87e4acfeb9e8d3ed235deb0171b7877480591694d2364fe340ebb05fc30f71a",
    "SimWorldEditor.debug":
        "99ef9097d230631ff527f3ab6645490208039b96c7cb219e239e5a5ad416de7c",
    "SimWorldEditor.modules":
        "cbf7a632f96bb51c8c9b4bb5a98006268e737bf5d2c17e6b05a1a0a93c9b23b9",
    "SimWorldEditor.target":
        "fe0b0c677b836a15448c701b8f62d89fbc6efa236c2f6eb4af952f079355d89c",
    "SimWorldEditor.version":
        "f7bf20b2fbc91c352bcce49503c210cabdea4ab1bbeb85cf43c4026790af29f2",
    "libSimWorldEditor-SimWorld.so":
        "ec6b65ef90b3f5e1bc12d471848d9901d5ccacb17c5041fad706620ed6465f60",
    "libSimWorldEditor-SimWorld.debug":
        "cb93b2d2c3c0b9560f14586e68c92bcbbf2c31f8cd8098e71f7158fbfe681bf5",
}


class IdentityError(RuntimeError):
    """The requested launcher identity could not be authenticated."""


def _read_json(path: Path, label: str) -> Mapping[str, Any]:
    if path.is_symlink() or not path.is_file():
        raise IdentityError(f"{label} is missing or is not a regular file: {path}")
    try:
        parsed = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, ValueError) as error:
        raise IdentityError(f"{label} is unreadable: {error}") from error
    if not isinstance(parsed, Mapping):
        raise IdentityError(f"{label} is not a JSON object")
    return parsed


def _tool(name: str) -> str:
    path = shutil.which(name)
    if path is None:
        raise IdentityError(f"required identity tool is unavailable: {name}")
    return path


def _run_tool(name: str, *arguments: Path | str) -> str:
    try:
        completed = subprocess.run(
            [_tool(name), *(str(argument) for argument in arguments)],
            check=False,
            text=True,
            capture_output=True,
            timeout=15.0,
        )
    except (OSError, subprocess.SubprocessError) as error:
        raise IdentityError(f"{name} could not inspect the build: {error}") from error
    if completed.returncode != 0:
        diagnostic = completed.stderr.strip() or completed.stdout.strip()
        raise IdentityError(f"{name} rejected the build artifact: {diagnostic}")
    return completed.stdout


def _elf_header(path: Path, *, expected_type: str, label: str) -> str:
    header = _run_tool("readelf", "-W", "-h", path)
    if "Class:                             ELF64" not in header \
            or "Data:                              2's complement, little endian" \
            not in header \
            or f"Type:                              {expected_type}" not in header \
            or "Machine:                           Advanced Micro Devices X86-64" \
            not in header:
        raise IdentityError(
            f"{label} is not the required Linux x86-64 ELF {expected_type} artifact"
        )
    notes = _run_tool("readelf", "-W", "-n", path)
    build_id = re.search(r"Build ID: ([0-9a-fA-F]{16,})", notes)
    if build_id is None:
        raise IdentityError(f"{label} has no ELF Build ID")
    return build_id.group(1).lower()


def _dynamic_entries(path: Path) -> tuple[set[str], str | None]:
    dynamic = _run_tool("readelf", "-W", "-d", path)
    needed = set(re.findall(r"\(NEEDED\).*?\[([^]]+)\]", dynamic))
    match = re.search(r"\(SONAME\).*?\[([^]]+)\]", dynamic)
    return needed, match.group(1) if match else None


def _build_id(value: Any, label: str) -> str:
    if not isinstance(value, str) or not BUILD_ID.fullmatch(value):
        raise IdentityError(f"{label} BuildId is missing or malformed")
    return value


def _require_approved_digest(path: Path) -> None:
    expected = APPROVED_BUNDLE_SHA256[path.name]
    digest = hashlib.sha256()
    try:
        with path.open("rb") as stream:
            for chunk in iter(lambda: stream.read(1024 * 1024), b""):
                digest.update(chunk)
    except OSError as error:
        raise IdentityError(f"approved bundle artifact is unreadable: {path}") \
            from error
    if digest.hexdigest() != expected:
        raise IdentityError(
            f"{path.name} is not from the approved project build")


def validate_project_editor_bundle(
    expected_editor: Path, selected_editor: Path,
) -> Path:
    """Authenticate one selected path against the exact project build bundle."""

    if expected_editor.name != "SimWorldEditor":
        raise IdentityError("expected project target must be named SimWorldEditor")
    try:
        expected = expected_editor.resolve(strict=True)
        selected = selected_editor.resolve(strict=True)
    except OSError as error:
        raise IdentityError(f"project editor path is unavailable: {error}") from error
    if selected != expected or not os.path.samefile(selected, expected):
        raise IdentityError(
            "selected editor is not the exact resolved project SimWorldEditor"
        )
    mode = expected.stat().st_mode
    if not stat.S_ISREG(mode) or not os.access(expected, os.X_OK):
        raise IdentityError("project SimWorldEditor is not an executable regular file")

    binary_dir = expected.parent
    manifest_path = binary_dir / "SimWorldEditor.modules"
    target_path = binary_dir / "SimWorldEditor.target"
    version_path = binary_dir / "SimWorldEditor.version"
    manifest = _read_json(manifest_path, "SimWorldEditor.modules")
    target = _read_json(target_path, "SimWorldEditor.target")
    version = _read_json(version_path, "SimWorldEditor.version")

    target_version = target.get("Version")
    if not isinstance(target_version, Mapping):
        raise IdentityError("SimWorldEditor.target has no Version object")
    manifest_build_id = _build_id(manifest.get("BuildId"), "manifest")
    target_build_id = _build_id(target_version.get("BuildId"), "target")
    version_build_id = _build_id(version.get("BuildId"), "version")
    if len({manifest_build_id, target_build_id, version_build_id}) != 1:
        raise IdentityError(
            "SimWorldEditor manifest, target, and version BuildId values do not match"
        )

    exact_target_fields = {
        "TargetName": "SimWorldEditor",
        "Platform": "Linux",
        "TargetType": "Editor",
        "Project": "../../SimWorld.uproject",
        "Launch": "$(ProjectDir)/Binaries/Linux/SimWorldEditor",
    }
    for key, expected_value in exact_target_fields.items():
        if target.get(key) != expected_value:
            raise IdentityError(
                f"SimWorldEditor.target has incompatible {key}: {target.get(key)!r}"
            )
    if target.get("Architecture") not in ("x64", "x86_64"):
        raise IdentityError("SimWorldEditor.target has an incompatible Architecture")

    products = target.get("BuildProducts")
    if not isinstance(products, list):
        raise IdentityError("SimWorldEditor.target has no BuildProducts")
    product_paths = {
        item.get("Path") for item in products if isinstance(item, Mapping)
    }
    required_products = {
        "$(ProjectDir)/Binaries/Linux/SimWorldEditor",
        "$(ProjectDir)/Binaries/Linux/SimWorldEditor.debug",
        "$(ProjectDir)/Binaries/Linux/SimWorldEditor.modules",
        "$(ProjectDir)/Binaries/Linux/libSimWorldEditor-SimWorld.debug",
        "$(ProjectDir)/Binaries/Linux/libSimWorldEditor-SimWorld.so",
    }
    if not required_products.issubset(product_paths):
        raise IdentityError(
            "SimWorldEditor.target does not bind the editor, manifest, and SimWorld module"
        )
    modules = manifest.get("Modules")
    if not isinstance(modules, Mapping) or modules.get("SimWorld") != MODULE_NAME:
        raise IdentityError(
            "SimWorldEditor.modules must map SimWorld to the matching "
            "SimWorldEditor module"
        )
    module_path = binary_dir / MODULE_NAME
    if module_path.is_symlink() or not module_path.is_file():
        raise IdentityError(f"matching SimWorldEditor module is missing: {module_path}")
    for metadata_path in (manifest_path, target_path, version_path):
        _require_approved_digest(metadata_path)

    editor_build_id = _elf_header(
        expected, expected_type="EXEC", label="SimWorldEditor")
    editor_needed, _editor_soname = _dynamic_entries(expected)
    if not REQUIRED_EDITOR_LIBRARIES.issubset(editor_needed):
        raise IdentityError(
            "SimWorldEditor does not link the required SimWorldEditor target libraries"
        )
    editor_debug = binary_dir / "SimWorldEditor.debug"
    if editor_debug.is_symlink() or not editor_debug.is_file():
        raise IdentityError("SimWorldEditor debug identity companion is missing")
    editor_debug_build_id = _elf_header(
        editor_debug, expected_type="EXEC", label="SimWorldEditor.debug")
    if editor_debug_build_id != editor_build_id:
        raise IdentityError(
            "SimWorldEditor and SimWorldEditor.debug ELF Build ID values do not match"
        )
    _require_approved_digest(expected)
    _require_approved_digest(editor_debug)

    module_build_id = _elf_header(
        module_path, expected_type="DYN", label=MODULE_NAME)
    symbols = _run_tool(
        "nm", "-D", "--defined-only", "--format=posix", module_path)
    exported_symbols = {
        line.split(maxsplit=1)[0]
        for line in symbols.splitlines()
        if line.strip()
    }
    required_symbols = {
        "_ZN21USpPixelGoalSubsystem29PixelGoal_CaptureViewPairJsonERK7FString",
        "_ZN21USpPixelGoalSubsystem33execPixelGoal_CaptureViewPairJsonEP7UObjectR6FFramePv",
    }
    if not required_symbols.issubset(exported_symbols):
        raise IdentityError(
            "matching SimWorldEditor module does not export both the native and "
            "reflected PixelGoal_CaptureViewPairJson wrappers"
        )
    module_needed, module_soname = _dynamic_entries(module_path)
    if module_soname != MODULE_NAME:
        raise IdentityError(
            f"matching SimWorldEditor module has incompatible SONAME {module_soname!r}"
        )
    if not REQUIRED_MODULE_LIBRARIES.issubset(module_needed):
        raise IdentityError(
            "matching SimWorldEditor module does not link the required project libraries"
        )
    module_debug = binary_dir / "libSimWorldEditor-SimWorld.debug"
    if module_debug.is_symlink() or not module_debug.is_file():
        raise IdentityError("SimWorldEditor module debug identity companion is missing")
    module_debug_build_id = _elf_header(
        module_debug, expected_type="DYN", label=module_debug.name)
    if module_debug_build_id != module_build_id:
        raise IdentityError(
            "SimWorldEditor module and debug companion ELF Build ID values do not match"
        )
    _require_approved_digest(module_path)
    _require_approved_digest(module_debug)
    return expected


def verify_process_executable_identity(
    pid: int,
    expected_editor: Path,
    *,
    timeout_s: float = 1.0,
    stable_s: float = 0.05,
) -> Path:
    """Require one exact PID to settle on the expected executable identity."""

    if isinstance(pid, bool) or not isinstance(pid, int) or pid <= 0:
        raise IdentityError("started editor PID is invalid")
    if not math.isfinite(timeout_s) or timeout_s < 0.0:
        raise IdentityError("process identity timeout must be finite and non-negative")
    try:
        expected = expected_editor.resolve(strict=True)
    except OSError as error:
        raise IdentityError(f"expected editor is unavailable: {error}") from error
    proc_exe = Path("/proc") / str(pid) / "exe"
    deadline = time.monotonic() + timeout_s
    matched_at: float | None = None
    last_actual = "unavailable"
    while True:
        now = time.monotonic()
        try:
            actual = proc_exe.resolve(strict=True)
            last_actual = str(actual)
        except OSError:
            actual = None
        if actual == expected:
            if matched_at is None:
                matched_at = now
            if now - matched_at >= stable_s:
                return expected
        elif matched_at is not None:
            raise IdentityError(
                "process executable identity mismatch after initially matching: "
                f"expected {expected}, got {last_actual}"
            )
        if now >= deadline:
            raise IdentityError(
                "process executable identity mismatch: "
                f"expected {expected}, got {last_actual}"
            )
        time.sleep(0.01)


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    bundle = commands.add_parser("validate-bundle")
    bundle.add_argument("--expected-editor", required=True, type=Path)
    bundle.add_argument("--selected-editor", required=True, type=Path)
    process = commands.add_parser("verify-process")
    process.add_argument("--pid", required=True, type=int)
    process.add_argument("--expected-editor", required=True, type=Path)
    process.add_argument("--timeout-s", default=1.0, type=float)
    return parser


def main() -> int:
    arguments = _build_parser().parse_args()
    try:
        if arguments.command == "validate-bundle":
            result = validate_project_editor_bundle(
                arguments.expected_editor, arguments.selected_editor)
        else:
            result = verify_process_executable_identity(
                arguments.pid,
                arguments.expected_editor,
                timeout_s=arguments.timeout_s,
            )
    except (IdentityError, OSError) as error:
        print(f"Launcher identity check failed: {error}", file=sys.stderr)
        return 2
    print(result)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
