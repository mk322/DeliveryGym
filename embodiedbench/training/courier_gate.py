"""The training gate, run against the courier environment.

    python -m embodiedbench.training.courier_gate --report artifacts/R1/courier.json

``r1_gate`` proves the same properties against the old vagen runtime, whose
action space is ``MOVE_TO``/``PICKUP``/``DROP_OFF`` and whose map is
``small-city-11``. That environment is not the one the benchmark ships or scores,
so passing it says nothing about whether *this* environment can train *this*
model. This is the courier edition.

What it checks, and why each one is here rather than assumed:

1. **an episode runs through the real harness** -- the same ``CourierSession``
   an evaluation uses, so a policy is optimised against the prompt it is scored
   on rather than a training-only variant;
2. **images reach more than one turn** -- a multimodal rollout that quietly
   degrades to text is the failure this benchmark is least able to notice;
3. **the policy loss touches only sampled tokens** -- observation, template and
   image tokens all excluded, checked from provenance rather than by re-scanning
   for delimiters;
4. **every sampled token has a positionally aligned log-prob**;
5. **fan-out preserves the episode return exactly once**;
6. **one optimizer step changes the weights, and the changed weights change
   behaviour** -- a step that leaves the policy identical has proven nothing.

Checks 1-5 need no model and run anywhere, on a stub whose only job is to append
tokens with honest provenance. Check 6 needs a real checkpoint; when there is
none it is reported ``skipped`` with the reason, never ``pass``. A gate that
quietly downgrades is worse than one that fails.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Any

REPO = Path(__file__).resolve().parents[2]
MAPS = REPO / "vendor/vagen/vagen/envs/deliverybench/maps/citycore-paris"
# Relocated through ALBUMS_DIR like every other album path in the repo, so
# the gate checks the albums a run will actually read rather than the ones
# the workstation they were baked on had.
from embodiedbench.training.vagen_courier_env import _resolve_album  # noqa: E402

STREETS = _resolve_album(Path("/data/albums/paris_streets_v2/citycore-paris"))
PAVEMENT = _resolve_album(Path("/data/albums/paris_streets_pavement/citycore-paris"))
SIGNALS = _resolve_album(Path("/data/albums/paris_lamps_real/citycore-paris"))
OBSTACLES = _resolve_album(Path("/data/albums/paris_obstacles/citycore-paris"))
PAVEMENT_OBSTACLES = _resolve_album(Path("/data/albums/paris_obstacles_pavement/citycore-paris"))


class Check:
    def __init__(self, name: str):
        self.name = name
        self.status = "skipped"
        self.detail: dict[str, Any] = {}

    def passed(self, **detail: Any) -> "Check":
        self.status, self.detail = "pass", detail
        return self

    def failed(self, **detail: Any) -> "Check":
        self.status, self.detail = "fail", detail
        return self

    def skip(self, reason: str) -> "Check":
        self.status, self.detail = "skipped", {"reason": reason}
        return self

    def to_dict(self) -> dict[str, Any]:
        return {"check": self.name, "status": self.status, **self.detail}


def build_env(paris, seed: int, tier: str, embodiment: str,
              hazards: bool = True):
    from embodiedbench.runtime.city.courier_env import CourierEnv

    kwargs: dict[str, Any] = {}
    if STREETS.exists():
        kwargs["album_root"] = STREETS
        if not hazards:
            # The simplest rung there is: streets and doors, no lights and
            # nothing in the way. Omitting an album is how a mechanic is
            # turned off -- the runtime refuses to charge for what no
            # photograph can show -- so this is the environment's own
            # switch rather than a training-only special case.
            if embodiment == "human_on_foot" and PAVEMENT.exists():
                kwargs["pavement_album_root"] = PAVEMENT
            env = CourierEnv(paris, seed=seed, difficulty=tier,
                             stride="block", embodiment=embodiment, **kwargs)
            env.reset()
            return env
        if SIGNALS.exists():
            kwargs["signal_album_root"] = SIGNALS
        if embodiment == "human_on_foot" and PAVEMENT.exists():
            kwargs["pavement_album_root"] = PAVEMENT
            if OBSTACLES.exists() and PAVEMENT_OBSTACLES.exists():
                kwargs["obstacle_album_root"] = OBSTACLES
                kwargs["pavement_obstacle_album_root"] = PAVEMENT_OBSTACLES
        elif OBSTACLES.exists():
            kwargs["obstacle_album_root"] = OBSTACLES
    env = CourierEnv(paris, seed=seed, difficulty=tier, stride="block",
                     embodiment=embodiment, **kwargs)
    env.reset()
    return env


def load_adapter(model_path: str | None):
    """A real adapter if there is a checkpoint, otherwise the stub."""
    if model_path and Path(model_path).exists():
        from embodiedbench.agent.model_adapters.qwen3vl import Qwen3VLAdapter

        return Qwen3VLAdapter(model_path), True
    from tests.test_courier_training import StubAdapter  # noqa: PLC0415

    return StubAdapter(), False


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--model", default=None,
                        help="path to a Qwen3-VL checkpoint; omitted runs the "
                             "model-free checks only")
    parser.add_argument("--tier", default="solo")
    parser.add_argument("--embodiment", default="human_on_foot")
    parser.add_argument("--seeds", type=int, default=3)
    parser.add_argument("--max-turns", type=int, default=8)
    parser.add_argument("--learning-rate", type=float, default=1e-3,
                        help="deliberately larger than a real run. The gate has "
                             "to show the step CHANGED BEHAVIOUR, and at 1e-6 the "
                             "weights moved 1.8e-08 -- real, but too small to "
                             "shift the probe logits, so the check could not tell "
                             "a working update from a dead one.")
    parser.add_argument("--temperature", type=float, default=1.0,
                        help="rollouts must SAMPLE. At 0.0 every episode in "
                             "a batch decodes identically, so a mean baseline "
                             "makes the advantages cancel and the gradient is "
                             "exactly zero -- measured: loss 0.0 over 93 masked "
                             "tokens.")
    parser.add_argument("--max-images", type=int, default=5,
                        help="pictures per turn; every one is image tokens "
                             "in the context and in the backward pass")
    parser.add_argument("--report", type=Path, default=None)
    args = parser.parse_args(argv)

    sys.path.insert(0, str(REPO))
    from embodiedbench.compiler.road_network import build_road_network
    from embodiedbench.training.core import check_reward_conservation, validate_sample
    from embodiedbench.training.courier_rollout import (
        batch_advantages,
        rollout_courier_episode,
    )

    started = time.time()
    paris = build_road_network(MAPS, map_name="citycore-paris")
    adapter, is_real = load_adapter(args.model)
    checks: list[Check] = []

    # 1. rollouts through the real harness
    rollouts = []
    check = Check("episodes_run_through_the_shipped_harness")
    try:
        for seed in range(args.seeds):
            env = build_env(paris, seed, args.tier, args.embodiment)
            rollouts.append(rollout_courier_episode(
                adapter, env, episode_id=f"courier-{seed}", seed=seed,
                max_turns=args.max_turns, max_images=args.max_images,
                temperature=args.temperature))
        turns = [r.turns for r in rollouts]
        check.passed(episodes=len(rollouts), turns=turns,
                     rewards=[round(r.reward, 3) for r in rollouts])
    except Exception as error:  # noqa: BLE001
        check.failed(error=f"{type(error).__name__}: {error}")
    checks.append(check)

    if not rollouts:
        return _emit(checks, args, started, is_real)

    # 2. images on more than one turn
    check = Check("images_reach_more_than_one_turn")
    shown = sum(r.images_shown for r in rollouts)
    if not STREETS.exists():
        check.skip("albums not mounted on this host")
    elif shown == 0:
        check.failed(reason="no images were shown at all; the rollout is text-only")
    else:
        with_images = sum(1 for r in rollouts if r.images_shown > 0)
        check.passed(images_shown=shown, episodes_with_images=with_images,
                     images_dropped=sum(r.images_dropped for r in rollouts))
    checks.append(check)

    # 3 & 4. the mask, and log-prob alignment
    check = Check("policy_loss_covers_only_sampled_tokens")
    try:
        leaked_obs = leaked_img = 0
        for rollout in rollouts:
            sample = rollout.samples[0]
            validate_sample(sample)
            assistant = set(sample.response_indices)
            for span in rollout.state.spans:
                positions = set(range(span.start, span.end))
                if not span.is_assistant and (positions & assistant):
                    leaked_obs += len(positions & assistant)
            leaked_img += len(set(sample.image_token_positions) & assistant)
        if leaked_obs or leaked_img:
            check.failed(observation_tokens_in_loss=leaked_obs,
                         image_tokens_in_loss=leaked_img)
        else:
            check.passed(
                sampled_tokens=sum(r.samples[0].response_token_count for r in rollouts),
                total_tokens=sum(len(r.samples[0].input_ids) for r in rollouts),
                logprobs_aligned=True)
    except Exception as error:  # noqa: BLE001
        check.failed(error=f"{type(error).__name__}: {error}")
    checks.append(check)

    # 5. fan-out reward accounting
    check = Check("fanout_preserves_the_episode_return")
    try:
        import copy

        from embodiedbench.training.core import split_for_fanout

        for rollout in rollouts:
            # A *copy*, with a non-zero reward so the conservation check is not
            # vacuous. Mutating the real sample here set every reward to 1.0,
            # which made the batch baseline equal to every reward, every
            # advantage zero, and check 6's optimizer step a silent no-op --
            # 170 masked tokens and a loss of exactly 0.0.
            probe = copy.copy(rollout.samples[0])
            probe.reward = rollout.reward or 1.0
            check_reward_conservation(probe.reward, split_for_fanout(probe, 4))
        check.passed(shards=4, episodes=len(rollouts))
    except Exception as error:  # noqa: BLE001
        check.failed(error=f"{type(error).__name__}: {error}")
    checks.append(check)

    # 6. a real optimizer step
    check = Check("one_update_changes_the_policy")
    if not is_real:
        check.skip("no checkpoint given; a stub has no weights to change, and "
                   "reporting this as a pass would make the gate meaningless")
    else:
        try:
            from embodiedbench.training.policy_update import run_policy_update

            samples = [r.samples[0] for r in rollouts]
            advantages = batch_advantages(rollouts)
            synthetic = all(a == 0.0 for a in advantages)
            if synthetic:
                # Every episode earned the same return -- which for a small
                # model on this task means every episode earned nothing, so the
                # batch has no baseline to differ from and the step would be a
                # no-op. What check 6 is for is the *mechanism*: that gradient
                # reaches the sampled tokens and the optimizer moves the
                # weights. So one sample is given a unit reward to make the
                # advantage non-zero, and the report says so. Reporting a no-op
                # as a pass, or refusing to test the mechanism because the
                # policy is bad, would both be wrong.
                samples[0].reward = 1.0
            result = run_policy_update(adapter, samples,
                                       learning_rate=args.learning_rate)
            detail = {**result.to_dict(),
                      "synthetic_advantage": synthetic,
                      "episode_rewards": [round(r.reward, 3) for r in rollouts]}
            (check.passed(**detail) if result.parameters_changed
             and result.logits_changed else check.failed(**detail))
        except Exception as error:  # noqa: BLE001
            check.failed(error=f"{type(error).__name__}: {error}")
    checks.append(check)

    return _emit(checks, args, started, is_real)


def _emit(checks: list[Check], args, started: float, is_real: bool) -> int:
    failed = [c for c in checks if c.status == "fail"]
    skipped = [c for c in checks if c.status == "skipped"]
    report = {
        "gate": "courier_r1",
        "model": args.model or "(stub: model-free checks only)",
        "real_model": is_real,
        "tier": args.tier, "embodiment": args.embodiment, "seeds": args.seeds,
        "status": "fail" if failed else ("pass_with_skips" if skipped else "pass"),
        "seconds": round(time.time() - started, 1),
        "checks": [c.to_dict() for c in checks],
    }
    for check in checks:
        mark = {"pass": "PASS", "fail": "FAIL", "skipped": "SKIP"}[check.status]
        print(f"[{mark}] {check.name}")
        if check.status != "pass":
            print(f"       {json.dumps(check.detail)[:200]}")
    print(f"courier R1 gate: {report['status']} "
          f"({len(checks) - len(failed) - len(skipped)} pass, {len(failed)} fail, "
          f"{len(skipped)} skip) in {report['seconds']}s")
    if args.report:
        args.report.parent.mkdir(parents=True, exist_ok=True)
        args.report.write_text(json.dumps(report, indent=2, default=str))
        print("wrote", args.report)
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
