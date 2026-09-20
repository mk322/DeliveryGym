# run_deliverybench.py (text-only)
import json
import os
import sys
import traceback
from pathlib import Path

import json
import os
import sys
import traceback


from vagen.envs.deliverybench.vlm_delivery.gym_like_interface import DeliveryBenchGymEnvText


base_dir = "/root/VAGEN2/vagen/envs/deliverybench"

from vlm_delivery.gym_like_interface import DeliveryBenchGymEnvText


def main():
    exp_cfg_path = os.path.join(base_dir, "vlm_delivery", "input", "experiment_config.json")
    with open(exp_cfg_path, "r", encoding="utf-8") as f:
        exp_cfg = json.load(f) or {}
    gym_env_cfg = exp_cfg.get("gym_env", {}) or {}

    env = DeliveryBenchGymEnvText(
        base_dir=base_dir,
        map_name=gym_env_cfg.get("map_name", "medium-city-22"),
        max_steps=20,
    )

    try:
        obs, info = env.reset(seed=0)
        print("reset info:", info)
        print("obs:", obs)

        for step_i in range(1, 999999):
            obs, r, term, trunc, info2 = env.step(None)
            print(f"[RL] step={step_i} info:", info2)

            if info2.get("error"):
                print("STEP ERROR:", info2["error"])
                break

            if term or trunc:
                break

    except Exception as e:
        print("[RL] Exception:", e)
        traceback.print_exc()

    finally:
        try:
            env.close()
        except Exception:
            pass


if __name__ == "__main__":
    main()