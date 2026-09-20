"""Validation shared by model-ingress adapters for required image groups."""

from __future__ import annotations

from typing import Any


def required_image_groups(observation: Any) -> dict[str, list[Any]]:
    """Return required frames grouped in observation order, rejecting bad pairs."""
    groups: dict[str, list[Any]] = {}
    for frame in observation.frames:
        if not getattr(frame, "required_group", False):
            continue
        group_id = getattr(frame, "capture_group_id", None)
        if not isinstance(group_id, str) or not group_id.strip():
            raise ValueError("a required image frame is missing capture_group_id")
        groups.setdefault(group_id, []).append(frame)

    for group_id, frames in groups.items():
        if [getattr(frame, "view_id", None) for frame in frames] != ["front", "rear"]:
            raise ValueError(f"required image group {group_id} is not front, rear")
    return groups


def validate_atomic_image_capacity(observation: Any, max_images: int) -> None:
    """Fail before I/O when an image cap cannot hold a complete required group."""
    if type(max_images) is not int or max_images < 0:
        raise ValueError(
            f"max_images must be a non-negative int; got {max_images!r}"
        )

    groups = required_image_groups(observation)
    required_count = sum(len(frames) for frames in groups.values())
    if max_images >= required_count:
        return
    if len(groups) == 1:
        group_id, frames = next(iter(groups.items()))
        raise ValueError(
            f"{group_id} requires {len(frames)} photographs; "
            f"max_images={max_images}"
        )
    raise ValueError(
        f"required image groups require {required_count} photographs; "
        f"max_images={max_images}"
    )
