"""
Replay a saved trajectory JSON against the DeliveryBench environment.

Reads the actions from a trajectory file, feeds them to a fresh env instance
with the same seed, and prints a side-by-side comparison of the original vs
replayed observations, rewards, and done flags.

Usage:
    python scripts/replay_trajectory.py \
        --trajectory exps/run_.../trajectories/ep0021_seed10126.json \
        --val-yaml scripts/train/earning_reward/val_deliverybench.yaml
"""

import argparse
import asyncio
import json
import sys
from pathlib import Path
from typing import Any, Dict

import yaml


def load_trajectory(path: str) -> Dict[str, Any]:
    with open(path) as f:
        return json.load(f)


def load_env_config(val_yaml: str) -> Dict[str, Any]:
    with open(val_yaml) as f:
        data = yaml.safe_load(f)
    return dict(data["envs"][0].get("config", {}))


async def replay(traj_path: str, val_yaml: str):
    traj = load_trajectory(traj_path)
    env_config = load_env_config(val_yaml)
    seed = traj["seed"]
    turns = traj["turns"]

    from vagen.envs.deliverybench.deliverybench_env import DeliveryBench

    env = DeliveryBench(env_config)
    sys_obs = await env.system_prompt()
    obs, info = await env.reset(seed=seed)

    print(f"=== Replaying {traj_path} | seed={seed} | {len(turns)} turns ===\n")
    print(f"[System Prompt] (first 200 chars):\n{sys_obs['obs_str'][:200]}...\n")
    print(f"[Initial Obs] (first 300 chars):\n{obs['obs_str'][:300]}...\n")
    print("=" * 80)

    cumulative_reward = 0.0
    for turn_data in turns:
        turn_num = turn_data["turn"]
        action = turn_data["action"]
        orig_reward = turn_data["reward"]
        orig_done = turn_data["done"]
        orig_sim_time = turn_data.get("sim_time", "?")

        obs, reward, done, step_info = await env.step(action)
        cumulative_reward += reward

        traj_metrics = (step_info.get("metrics") or {}).get("traj_metrics", {})
        sim_hours = traj_metrics.get("sim_hours", 0.0)
        total_min = round(sim_hours * 60)
        sim_time_str = f"{total_min // 60}h {total_min % 60:02d}m"

        match_reward = "OK" if abs(reward - orig_reward) < 1e-4 else "MISMATCH"
        match_done = "OK" if done == orig_done else "MISMATCH"

        print(f"\n--- Turn {turn_num} ---")
        print(f"  Action     : {action[:120]}...")
        print(f"  Reward     : {reward:+.4f}  (orig: {orig_reward:+.4f})  [{match_reward}]")
        print(f"  Cumulative : {cumulative_reward:+.4f}")
        print(f"  Done       : {done}  (orig: {orig_done})  [{match_done}]")
        print(f"  Sim Time   : {sim_time_str}  (orig: {orig_sim_time})")

        action_error = step_info.get("action_error")
        if action_error:
            print(f"  Error      : {action_error}")

        dm_errors = None
        if env._env and env._env.dms:
            dm_errors = getattr(env._env.dms[0], "vlm_errors", None)
        if dm_errors:
            print(f"  VLM Error  : {dm_errors}")

        print(f"  Obs (first 200 chars): {obs['obs_str'][:200]}...")

        if done:
            print(f"\n  Episode ended at turn {turn_num}.")
            break

    print("\n" + "=" * 80)
    print(f"Final reward: {cumulative_reward:+.4f}  (orig: {traj['reward']:+.4f})")
    print(f"Deliveries : {env.deliveries_completed}")
    await env.close()


def main():
    parser = argparse.ArgumentParser(description="Replay a trajectory against DeliveryBench")
    parser.add_argument("--trajectory", required=True, help="Path to trajectory JSON")
    parser.add_argument(
        "--val-yaml",
        default="scripts/train/earning_reward/val_deliverybench.yaml",
        help="Path to val YAML for env config",
    )
    args = parser.parse_args()
    asyncio.run(replay(args.trajectory, args.val_yaml))


if __name__ == "__main__":
    main()
