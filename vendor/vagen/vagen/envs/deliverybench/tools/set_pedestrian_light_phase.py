"""Switch generated pedestrian-light states in a DeliveryBench world JSON."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from vagen.envs.deliverybench.utils.pedestrian_lights import PHASE_A, PHASE_B, apply_pedestrian_light_phase


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("world_json", type=Path)
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument(
        "--phase",
        choices=(PHASE_A, PHASE_B, "left_green_right_red", "left_red_right_green"),
        help="Explicit pedestrian-light phase. Old left/right aliases are accepted.",
    )
    group.add_argument(
        "--seconds",
        type=float,
        help="Episode time in seconds; phases alternate over --period-s.",
    )
    parser.add_argument("--period-s", type=float, default=120.0)
    args = parser.parse_args()

    with args.world_json.open("r", encoding="utf-8") as f:
        world = json.load(f)

    phase = apply_pedestrian_light_phase(
        world,
        phase=args.phase,
        seconds=args.seconds,
        period_s=args.period_s,
    )

    with args.world_json.open("w", encoding="utf-8") as f:
        json.dump(world, f, indent=2)
        f.write("\n")

    print(f"applied pedestrian-light phase: {phase}")


if __name__ == "__main__":
    main()
