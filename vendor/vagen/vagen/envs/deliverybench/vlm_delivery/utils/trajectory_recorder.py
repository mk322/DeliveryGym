# utils/trajectory_recorder.py
# -*- coding: utf-8 -*-

"""
Trajectory / episode data recorder.

This module provides lightweight utilities for storing all artifacts
generated during a trajectory run, including:

- Text logs (e.g., prompts, reasoning traces, debug messages)
- Image files (maps, observations, rendered frames)
- Raw PNG byte streams (useful for model- or API-generated images)

All saving utilities ensure directory creation and return absolute paths.
"""

import os
from datetime import datetime
from typing import Optional

from PIL import Image  # Used for saving images; can be replaced if needed.


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _ensure_dir(path: str) -> None:
    """Ensure the target directory exists."""
    os.makedirs(path, exist_ok=True)


# ---------------------------------------------------------------------------
# Text saving
# ---------------------------------------------------------------------------

def save_text(
    folder: str,
    filename: str,
    text: str,
    encoding: str = "utf-8",
    overwrite: bool = True,
) -> str:
    """
    Save a block of text into a .txt file.

    Args:
        folder: Directory where the file will be stored.
        filename: File name such as "step_0001_log.txt".
        text: Text content to write.
        encoding: File encoding.
        overwrite: If False, append a timestamp to avoid overwriting.

    Returns:
        Absolute path to the written file.
    """
    _ensure_dir(folder)

    if not filename.endswith(".txt"):
        filename = filename + ".txt"

    full_path = os.path.join(folder, filename)

    # Add timestamp if overwriting is disabled and the file exists.
    if (not overwrite) and os.path.exists(full_path):
        base, ext = os.path.splitext(filename)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%%S")
        filename = f"{base}_{timestamp}{ext}"
        full_path = os.path.join(folder, filename)

    with open(full_path, "w", encoding=encoding, newline="\n") as f:
        f.write(text)

    return os.path.abspath(full_path)


# ---------------------------------------------------------------------------
# Image saving
# ---------------------------------------------------------------------------

def save_image(
    folder: str,
    filename: str,
    image,
    format: Optional[str] = None,
    overwrite: bool = True,
) -> str:
    """
    Save an image to disk.

    The input can be a PIL.Image.Image instance or a numpy array (H, W, C).
    The resulting image format is inferred from the filename unless
    explicitly specified through `format`.

    Args:
        folder: Target directory.
        filename: File name such as "step_0001_obs.png".
        image: PIL image or numpy array.
        format: Optional output format (e.g., "PNG").
        overwrite: If False, append timestamp to avoid overwriting.

    Returns:
        Absolute path to the saved image.
    """
    _ensure_dir(folder)

    if "." not in filename:
        filename = filename + ".png"

    full_path = os.path.join(folder, filename)

    # Avoid overwriting if explicitly disabled.
    if (not overwrite) and os.path.exists(full_path):
        base, ext = os.path.splitext(filename)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"{base}_{timestamp}{ext}"
        full_path = os.path.join(folder, filename)

    # Convert to PIL Image if needed.
    if isinstance(image, Image.Image):
        img = image
    else:
        import numpy as np
        img = Image.fromarray(image.astype(np.uint8))

    img.save(full_path, format=format)
    return os.path.abspath(full_path)


# ---------------------------------------------------------------------------
# PNG bytes saving
# ---------------------------------------------------------------------------

def save_png_bytes(
    folder: str,
    filename: str,
    data: bytes,
    overwrite: bool = True,
) -> str:
    """
    Save raw PNG byte content as a file.

    Useful when image bytes come directly from an API response or
    another imaging backend.

    Args:
        folder: Target directory.
        filename: File name such as "step_0001_obs.png".
        data: Raw PNG byte sequence.
        overwrite: If False, append timestamp when file exists.

    Returns:
        Absolute path to the saved file.
    """
    _ensure_dir(folder)

    if "." not in filename:
        filename = filename + ".png"

    full_path = os.path.join(folder, filename)

    # Prevent overwriting if necessary.
    if (not overwrite) and os.path.exists(full_path):
        base, ext = os.path.splitext(filename)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"{base}_{timestamp}{ext}"
        full_path = os.path.join(folder, filename)

    with open(full_path, "wb") as f:
        f.write(data)

    return os.path.abspath(full_path)


# ---------------------------------------------------------------------------
# Run folder creation
# ---------------------------------------------------------------------------

def make_run_folder(root: str, run_name: Optional[str] = None) -> str:
    """
    Create and return a dedicated directory for a full episode run.

    Args:
        root: Base directory, e.g. "outputs/trajectories".
        run_name: Optional explicit subdirectory name. If omitted,
                  a timestamp-based name is generated.

    Returns:
        Absolute path to the created run directory.
    """
    if run_name is None:
        run_name = datetime.now().strftime("run_%Y%m%d_%H%M%S")

    folder = os.path.join(root, run_name)
    _ensure_dir(folder)
    return os.path.abspath(folder)