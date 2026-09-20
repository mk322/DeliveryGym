"""Lightweight tests for balanced visual SFT sampling.

Run:
    PYTHONPATH=. python -m vagen.envs.deliverybench.test_balanced_visual_sft_data
"""

from __future__ import annotations

import json


def _row(stage: str, action: str, idx: int) -> dict:
    return {
        "messages": [
            {"role": "system", "content": "system"},
            {"role": "user", "content": "<image>\n\nobs"},
            {
                "role": "assistant",
                "content": json.dumps(
                    {
                        "reasoning_and_reflection": "keep full rollout-style target",
                        "action": action,
                        "future_plan": "continue",
                    }
                ),
            },
        ],
        "images": [{"image": f"/abs/fake_{idx}.png"}],
        "seed": idx,
        "turn_index": idx,
        "stage": stage,
        "action": action,
        "repeat_index": 0,
        "map_name": "small-city-11",
    }


def test_balanced_sampler_keeps_workflow_and_full_json() -> None:
    from vagen.envs.deliverybench.tools.build_balanced_visual_sft_data import (
        MOVE_DIRECTIONS,
        direction_counts,
        sample_balanced_rows,
        stage_counts,
    )

    rows = []
    idx = 0
    for direction in MOVE_DIRECTIONS:
        for _ in range(3):
            rows.append(_row("move", f'MOVE(direction="{direction}")', idx))
            idx += 1
    for stage, action in [
        ("view_orders", "VIEW_ORDERS()"),
        ("accept", "ACCEPT_ORDER(0)"),
        ("navigate_pickup", 'NAVIGATE(target="100 Main St", mode="walk")'),
        ("pickup", "PICKUP(orders=[0])"),
        ("navigate_dropoff", 'NAVIGATE(target="200 Main St", mode="walk")'),
        ("dropoff", "DROP_OFF(oid=0)"),
    ]:
        rows.append(_row(stage, action, idx))
        idx += 1

    selected, stats = sample_balanced_rows(
        rows,
        target_total=10,
        workflow_fraction=0.2,
        include_workflow=True,
        allow_repeat=False,
        seed=7,
    )

    assert len(selected) == 10
    assert direction_counts(selected) == {direction: 2 for direction in MOVE_DIRECTIONS}
    assert sum(stage_counts(selected).get(stage, 0) for stage in ("view_orders", "accept", "navigate_pickup", "pickup", "navigate_dropoff", "dropoff")) == 2
    assert stats["selected_total"] == 10
    for row in selected:
        content = row["messages"][-1]["content"]
        parsed = json.loads(content)
        assert set(parsed) == {"reasoning_and_reflection", "action", "future_plan"}


def test_multicity_config_defaults_to_map_only_images() -> None:
    from vagen.envs.deliverybench.tools.build_balanced_visual_sft_data import make_env_config

    cfg = make_env_config(
        map_name="medium-city-18",
        max_steps=25,
        feasible_order_step_budget=20,
        enable_fpv=False,
    )
    assert cfg.map_name == "medium-city-18"
    assert cfg.task_mode == "visual_route_following"
    assert cfg.render_mode == "vision"
    assert cfg.enable_map_images is True
    assert cfg.enable_fpv is False
    assert cfg.fixed_spawn_position is None


if __name__ == "__main__":
    test_balanced_sampler_keeps_workflow_and_full_json()
    test_multicity_config_defaults_to_map_only_images()
    print("PASS test_balanced_visual_sft_data")
