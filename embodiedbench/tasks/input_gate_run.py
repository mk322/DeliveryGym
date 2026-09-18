"""Drive the input-quality gate against a live runtime.

Separated from ``input_gate`` so the checks themselves stay testable without the
vendored engine: the rules are properties of a string and an EnvSpec, and only
this module needs an environment to be running.
"""

from __future__ import annotations

import re

from typing import Any

from embodiedbench.schemas.env_spec import EnvSpec
from embodiedbench.schemas.runtime import ActionEnvelope, ActionStatus, TaskAction
from embodiedbench.tasks.observation_rewrite import compute_guidance, rewrite_observation
from embodiedbench.tasks.input_gate import (
    GateReport,
    ObservationOnlyOracle,
    OracleStep,
    check_observation,
    routing_hint,
)


def run_gate(
    *,
    map_name: str,
    env_spec: EnvSpec,
    max_steps: int = 40,
    seed: int = 0,
    preset: str = "nav",
    config_overrides: dict[str, Any] | None = None,
    rewrite: bool = True,
) -> GateReport:
    """Run the observation-only oracle and report what the input supports."""
    from embodiedbench.runtime.text.vagen_adapter import VagenTextRuntime
    from embodiedbench.schemas.fixtures import _episode_spec

    report = GateReport(map_name=map_name)
    overrides = {"enable_waypoint_marks": True}
    overrides.update(config_overrides or {})

    runtime = VagenTextRuntime(
        map_name=map_name, preset=preset, max_steps=max_steps,
        config_overrides=overrides,
    )
    try:
        spec = _episode_spec()
        observation, _info = runtime.reset(spec)

        def present(raw: str) -> str:
            """What the policy actually sees."""
            if not rewrite:
                return raw
            guidance = compute_guidance(runtime, raw)
            return rewrite_observation(
                raw, guidance, enabled_actions=list(env_spec.enabled_actions)
            )

        text = present(observation.text or "")
        report.static_findings = check_observation(text, env_spec)

        oracle = ObservationOnlyOracle(list(env_spec.enabled_actions))
        for index in range(max_steps):
            name, arguments, reason = oracle.act(text)
            if name not in env_spec.enabled_actions:
                report.notes.append(
                    f"oracle chose {name}, which this environment does not enable"
                )
                break
            envelope = ActionEnvelope(
                episode_id=spec.instance_id, step_index=index,
                action=TaskAction(name=name, arguments=arguments),
            )
            result = runtime.step(envelope)
            text = present(result.observation.text or "")
            distance = _objective_distance(text)
            if distance is not None:
                if report.distance_start_m is None:
                    report.distance_start_m = distance
                report.distance_end_m = distance
            if result.action_result.status is not ActionStatus.ACCEPTED:
                report.rejected_actions += 1
            if "matches no candidate bearing" in reason:
                report.hint_match_failures += 1
            report.reward_total += float(result.reward)
            report.steps.append(OracleStep(
                step=index, action=f"{name}{arguments or ''}", reason=reason,
                distance_m=distance, choices=text.count("MOVE_TO("),
            ))
            if result.terminated or result.truncated:
                report.notes.append(
                    f"episode ended at step {index}: "
                    f"{result.termination_reason.value if result.termination_reason else 'unknown'}"
                )
                break
    finally:
        runtime.close()

    if report.distance_start_m is None:
        report.notes.append(
            "no routing distance ever appeared, so progress could not be measured "
            "from the observation alone -- which is itself an input defect"
        )
    return report


_OBJECTIVE_DISTANCE = re.compile(r"Your destination is\s+([\d.]+)\s*m away")


def _objective_distance(text: str) -> float | None:
    """Distance to the goal as the rewritten observation states it.

    Read from the text, not from the runtime, because the whole question is
    whether a policy restricted to the observation can tell it is making
    progress. Taking it from privileged state would answer a different question.
    """
    match = _OBJECTIVE_DISTANCE.search(text or "")
    return float(match.group(1)) if match else None
