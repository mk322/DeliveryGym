"""The episode loop and trajectory writer (design plan §10).

the design plan M6 will require this harness to run text and cached backends "without
backend-specific branches in the episode loop". That property is established
here at M1, while there is only one backend, because it is far easier to keep
than to retrofit: the loop below touches only the ``EmbodiedRuntime`` protocol.

Budget accounting is the harness's job, not the environment's. When a budget is
exhausted the harness truncates with the matching ``TerminationReason``, which
is how design plan §7's termination/truncation split stays honest even for budgets
the environment knows nothing about (tool calls, output tokens).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from embodiedbench.artifacts.hashing import sha256_bytes
from embodiedbench.artifacts.state_digest import digest_of
from embodiedbench.schemas.episode import EpisodeSpec
from embodiedbench.schemas.runtime import ActionResult, ActionStatus, TerminationReason
from embodiedbench.schemas.trajectory import (
    CostAccounting,
    TokenAccounting,
    Trajectory,
    TrajectoryTurn,
)

HARNESS_VERSION = "0.1.0"


@dataclass
class EpisodeOutcome:
    """A finished episode and the hashes that identify it."""

    trajectory: Trajectory
    privileged: dict[str, Any] = field(default_factory=dict)
    state_digests: list[str] = field(default_factory=list)

    @property
    def trajectory_hash(self) -> str:
        return self.trajectory.content_hash()

    @property
    def transition_hash(self) -> str:
        """Fold over per-step state digests: any divergence changes it."""
        acc = ""
        for digest in self.state_digests:
            acc = sha256_bytes((acc + "|" + digest).encode())
        return acc

    @property
    def terminal_state_hash(self) -> str:
        return self.state_digests[-1] if self.state_digests else ""


class AgentHarness:
    """Runs one episode of any policy against any runtime."""

    def __init__(self, *, model_id: str = "scripted", harness_version: str = HARNESS_VERSION):
        self.model_id = model_id
        self.harness_version = harness_version

    def run_episode(
        self,
        *,
        runtime: Any,
        policy: Any,
        instance: EpisodeSpec,
        task_plugin: str = "delivery",
        task_plugin_version: str = "0.1.0",
    ) -> EpisodeOutcome:
        if hasattr(policy, "reset"):
            policy.reset()

        observation, _reset_info = runtime.reset(instance)
        state_digests = [runtime.authoritative_state_digest()]
        turns: list[TrajectoryTurn] = []
        total_reward = 0.0
        terminated = truncated = False
        reason: TerminationReason | None = None
        tool_calls_used = 0

        for step_index in range(instance.budgets.steps):
            envelope = policy.act(observation, runtime, step_index)
            if envelope is None:
                break

            # Budget checks happen before the step, so an exhausted budget never
            # produces a transition the trajectory then has to explain.
            if instance.budgets.tool_calls is not None and tool_calls_used >= instance.budgets.tool_calls:
                truncated, reason = True, TerminationReason.TOOL_CALL_BUDGET_EXHAUSTED
                break

            result = runtime.step(envelope)
            tool_calls_used += 1
            total_reward += result.reward
            state_digests.append(runtime.authoritative_state_digest())

            raw_output = ""
            outputs = getattr(policy, "raw_outputs", None)
            if outputs and len(outputs) > step_index:
                raw_output = outputs[step_index]

            turns.append(
                TrajectoryTurn(
                    step_index=step_index,
                    observation=observation,
                    observation_text=observation.text,
                    raw_model_output=raw_output,
                    action=envelope,
                    action_result=result.action_result,
                    events=result.events,
                    reward=result.reward,
                    reward_components=result.reward_components,
                    terminated=result.terminated,
                    truncated=result.truncated,
                    termination_reason=result.termination_reason,
                    # A scripted policy samples no tokens, so the mask is empty
                    # rather than fabricated. the design plan M6's mask assertions run
                    # against a model trajectory, not this one.
                    tokens=TokenAccounting(prompt_tokens=0, response_tokens=0),
                    cost=CostAccounting(),
                )
            )

            observation = result.observation
            if result.terminated or result.truncated:
                terminated, truncated, reason = result.terminated, result.truncated, result.termination_reason
                break
        else:
            truncated, reason = True, TerminationReason.STEP_BUDGET_EXHAUSTED

        trajectory = Trajectory(
            episode_id=instance.instance_id,
            instance_id=instance.instance_id,
            environment_id=instance.environment_id,
            environment_version=instance.environment_version,
            task_plugin=task_plugin,
            task_plugin_version=task_plugin_version,
            model_id=self.model_id,
            harness_version=self.harness_version,
            seed=instance.seed,
            turns=turns,
            total_reward=total_reward,
            terminated=terminated,
            truncated=truncated,
            termination_reason=reason,
        )
        return EpisodeOutcome(
            trajectory=trajectory,
            privileged=self._privileged_summary(runtime, total_reward, len(turns)),
            state_digests=state_digests,
        )

    @staticmethod
    def _privileged_summary(runtime: Any, total_reward: float, steps: int) -> dict[str, Any]:
        """Evaluator-only outcome facts. Never placed in an Observation."""
        dm = runtime._dm() if hasattr(runtime, "_dm") else None
        env = getattr(runtime, "_env", None)
        completed = getattr(dm, "completed_orders", None) if dm is not None else None
        return {
            "delivered_count": len(completed) if completed is not None else 0,
            "earnings_total": float(getattr(dm, "earnings_total", 0.0)) if dm is not None else 0.0,
            "energy_pct": float(getattr(dm, "energy_pct", 0.0)) if dm is not None else 0.0,
            "success": bool(env._check_success()) if env is not None and hasattr(env, "_check_success") else False,
            "total_reward": total_reward,
            "steps": steps,
        }


def trajectory_digest(trajectory: Trajectory) -> str:
    return digest_of(trajectory.to_dict())
