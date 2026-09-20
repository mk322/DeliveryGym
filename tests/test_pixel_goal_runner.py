"""Unit tests for the live calibration runner's SPEAR wire boundary."""

from __future__ import annotations

import pytest

from tools.run_pixel_goal_m1a import (
    _build_argument_parser,
    _cleanup_session,
    _decode_object_response,
)


class _RecordingSession:
    def __init__(self):
        self.calls = []

    def end_play(self):
        self.calls.append("end_play")

    def shutdown(self):
        self.calls.append("shutdown")


@pytest.mark.parametrize(
    ("wire_value", "expected"),
    [
        ('{"success":true,"snapshot":"frame-1"}', {"success": True, "snapshot": "frame-1"}),
        ({"success": True, "snapshot": "frame-1"}, {"success": True, "snapshot": "frame-1"}),
    ],
)
def test_decode_object_response_accepts_json_text_or_reflected_mapping(
    wire_value, expected
):
    assert _decode_object_response("PixelGoal_Test", wire_value) == expected


def test_decode_object_response_rejects_non_object_values():
    with pytest.raises(RuntimeError, match="non-object"):
        _decode_object_response("PixelGoal_Test", "[]")


@pytest.mark.parametrize(
    ("launch_mode", "shutdown_attached_editor", "play_started", "expected"),
    [
        ("attach", False, True, ["end_play"]),
        ("attach", False, False, []),
        ("attach", True, True, ["shutdown"]),
        ("launch", False, True, ["shutdown"]),
    ],
)
def test_cleanup_session_respects_editor_ownership(
    launch_mode, shutdown_attached_editor, play_started, expected
):
    session = _RecordingSession()

    _cleanup_session(
        session,
        launch_mode=launch_mode,
        shutdown_attached_editor=shutdown_attached_editor,
        play_started=play_started,
    )

    assert session.calls == expected


def test_attach_shutdown_flag_is_opt_in():
    parser = _build_argument_parser()

    assert (
        parser.parse_args(
            ["--launch-mode", "attach"]
        ).shutdown_attached_editor
        is False
    )
    assert (
        parser.parse_args(
            ["--launch-mode", "attach", "--shutdown-attached-editor"]
        ).shutdown_attached_editor
        is True
    )
