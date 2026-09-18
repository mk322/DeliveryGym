"""Smoke tests for visual route-following SFT export helpers.

Run:
    PYTHONPATH=. python -m vagen.envs.deliverybench.test_visual_sft_export
"""

from __future__ import annotations

import json
import shutil
from pathlib import Path

import pandas as pd
from PIL import Image


def _as_list(value):
    if isinstance(value, list):
        return value
    if hasattr(value, "tolist"):
        return value.tolist()
    return list(value)


def _assert_no_policy_leak(messages) -> None:
    leaked = ("next_move", "oracle_next_move", "oracle_next_action")
    for msg in _as_list(messages):
        if msg.get("role") == "assistant":
            continue
        text = str(msg.get("content", "")).lower()
        for token in leaked:
            assert token not in text, f"policy message leaked {token}: {text[:500]}"


def _assert_images(messages, images) -> None:
    messages = _as_list(messages)
    images = _as_list(images)
    user_msgs = [m for m in messages if m.get("role") == "user"]
    assert len(user_msgs) == 1
    user_text = str(user_msgs[0].get("content", ""))
    assert user_text.count("<image>") == len(images)
    for image in images:
        assert Path(image["image"]).exists(), image


def test_visual_sft_export_smoke() -> None:
    from vagen.envs.deliverybench.tools.generate_visual_sft_data import (
        _json_response,
        _save_pre_action_images,
        _validate_prompt_images,
        parse_seed_list,
    )

    out_dir = Path("/tmp/deliverybench_visual_sft_export_test")
    if out_dir.exists():
        shutil.rmtree(out_dir)
    image_dir = out_dir / "images"
    out_dir.mkdir(parents=True, exist_ok=True)

    assert parse_seed_list("100,102-104") == [100, 102, 103, 104]

    obs = {
        "obs_str": "<image><image>\n\n### agent_state\nYou are at 100 Test Ave.\n### ephemeral_context\n[navigation]\nto: 200 Test Ave",
        "multi_modal_input": {
            "<image>": [
                Image.new("RGB", (16, 16), (255, 255, 255)),
                Image.new("RGB", (16, 16), (0, 0, 255)),
            ]
        },
    }
    images = _save_pre_action_images(obs, image_dir=image_dir, seed=100, turn_index=1)
    _validate_prompt_images(obs["obs_str"], images)

    rows = [
        {
            "messages": [
                {"role": "system", "content": "You are a delivery agent."},
                {"role": "user", "content": obs["obs_str"]},
                {"role": "assistant", "content": _json_response('NAVIGATE(target="200 Test Ave", mode="walk")', stage="navigate_pickup")},
            ],
            "images": images,
            "seed": 100,
            "turn_index": 1,
            "stage": "navigate_pickup",
            "action": 'NAVIGATE(target="200 Test Ave", mode="walk")',
        },
        {
            "messages": [
                {"role": "system", "content": "You are a delivery agent."},
                {"role": "user", "content": obs["obs_str"]},
                {"role": "assistant", "content": _json_response('MOVE(direction="forward")', stage="move")},
            ],
            "images": images,
            "seed": 100,
            "turn_index": 2,
            "stage": "move",
            "action": 'MOVE(direction="forward")',
        },
        {
            "messages": [
                {"role": "system", "content": "You are a delivery agent."},
                {"role": "user", "content": obs["obs_str"]},
                {"role": "assistant", "content": _json_response("PICKUP(orders=[0])", stage="pickup")},
            ],
            "images": images,
            "seed": 100,
            "turn_index": 3,
            "stage": "pickup",
            "action": "PICKUP(orders=[0])",
        },
        {
            "messages": [
                {"role": "system", "content": "You are a delivery agent."},
                {"role": "user", "content": obs["obs_str"]},
                {"role": "assistant", "content": _json_response("DROP_OFF(oid=0)", stage="dropoff")},
            ],
            "images": images,
            "seed": 100,
            "turn_index": 4,
            "stage": "dropoff",
            "action": "DROP_OFF(oid=0)",
        },
    ]
    parquet = out_dir / "tiny.parquet"
    pd.DataFrame(rows).to_parquet(parquet, index=False)
    assert parquet.exists()

    df = pd.read_parquet(parquet)
    assert len(df) > 0
    for col in ("messages", "images", "seed", "turn_index", "stage", "action"):
        assert col in df.columns

    stages = set(df["stage"].tolist())
    assert "move" in stages
    assert "navigate_pickup" in stages
    assert "pickup" in stages

    actions = df["action"].tolist()
    assert any(str(a).startswith("MOVE(") for a in actions)
    assert any(str(a).startswith("NAVIGATE(") for a in actions)
    assert any(str(a).startswith("DROP_OFF(") for a in actions)

    for row in df.to_dict("records"):
        messages = _as_list(row["messages"])
        images = _as_list(row["images"])
        assert isinstance(messages, list)
        assert [m["role"] for m in messages] == ["system", "user", "assistant"]
        _assert_no_policy_leak(messages)
        _assert_images(messages, images)

        assistant = messages[-1]["content"]
        payload = json.loads(assistant)
        assert payload["action"] == row["action"]
        assert isinstance(payload.get("reasoning_and_reflection"), str)
        assert isinstance(payload.get("future_plan"), str)

    print(f"PASS test_visual_sft_export_smoke rows={len(df)} parquet={parquet}")


if __name__ == "__main__":
    test_visual_sft_export_smoke()
