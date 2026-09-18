"""A zero-shot VLM policy: the model half of the harness/model split.

design plan §10 separates model access from agent behaviour. This class is the model
half and nothing else -- it turns an observation into an action envelope. The
harness owns the loop, budgets, memory, and trajectory writing.

The design follows a coding agent's shape rather than a bespoke embodied one:
a fixed system prompt describing the tools, one observation per turn, one action
per turn, and typed feedback when an action fails. The action space is read from
the environment's ``EnvSpec`` instead of hardcoded, so on a non-cardinal map the
model is never offered directional MOVE.

Parse failures are ordinary events, not errors. design plan §10.2 requires the raw
output to be preserved and execution to depend only on the validated action, so
an unparseable reply is recorded verbatim, counted, and turned into a safe
no-op instead of aborting the episode.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Any

from embodiedbench.runtime.text.vagen_adapter import POSITIONAL_KEY
from embodiedbench.schemas.env_spec import EnvSpec
from embodiedbench.schemas.runtime import ActionEnvelope, Observation, TaskAction

ACTION_PATTERN = re.compile(
    r"\b(VIEW_ORDERS|ACCEPT_ORDER|MOVE_TO|MOVE|PICKUP|DROP_OFF|WAIT|NAVIGATE)\b\s*\(([^)]*)\)",
    re.IGNORECASE,
)


def build_system_prompt(env: EnvSpec, instruction: str) -> str:
    """A tool-style system prompt built from the environment's real action space."""
    lines = [
        instruction,
        "",
        "Each turn you receive the current state and, when available, a first-person "
        "view and a map. Reply with EXACTLY ONE action and nothing else.",
        "",
        "Available actions:",
    ]
    descriptions = {
        "VIEW_ORDERS": "VIEW_ORDERS()  - list orders you could accept",
        "ACCEPT_ORDER": "ACCEPT_ORDER(k)  - accept order number k",
        "MOVE_TO": 'MOVE_TO(k)  - step to numbered neighbouring waypoint k',
        "MOVE": 'MOVE(direction="forward"|"left"|"right"|"backward")  - step in a direction',
        "PICKUP": "PICKUP(orders=[k])  - collect order k at a restaurant",
        "DROP_OFF": "DROP_OFF(oid=k)  - deliver order k at its destination",
        "WAIT": "WAIT()  - do nothing this turn",
        "NAVIGATE": 'NAVIGATE(target="restaurant 1")  - ask for a route (does not move you)',
    }
    for action in env.enabled_actions:
        if action in descriptions:
            lines.append(f"  {descriptions[action]}")
    lines += [
        "",
        "Reply with one action only, for example: MOVE_TO(1)",
    ]
    return "\n".join(lines)


def parse_action(text: str, allowed: list[str]) -> tuple[str, dict[str, Any], bool]:
    """Parse a reply into ``(name, arguments, parsed_ok)``.

    Falls back to WAIT so a malformed reply costs a turn instead of ending the
    episode, and reports that it fell back so the harness can count it.
    """
    for match in ACTION_PATTERN.finditer(text or ""):
        name = match.group(1).upper()
        if name not in allowed:
            continue
        raw = match.group(2).strip()
        if name == "MOVE":
            direction = re.search(r'"?\b(forward|left|right|backward)\b"?', raw, re.IGNORECASE)
            if direction:
                return name, {"direction": direction.group(1).lower()}, True
            continue
        if name == "MOVE_TO":
            digits = re.search(r"-?\d+", raw)
            if digits:
                return name, {POSITIONAL_KEY: [int(digits.group())]}, True
            quoted = re.search(r'"([^"]+)"', raw)
            if quoted:
                return name, {"target": quoted.group(1)}, True
            continue
        if name == "ACCEPT_ORDER":
            digits = re.search(r"\d+", raw)
            return name, {POSITIONAL_KEY: [int(digits.group())]} if digits else {}, True
        if name == "PICKUP":
            digits = re.search(r"\d+", raw)
            return name, {"orders": [int(digits.group()) if digits else 0]}, True
        if name == "DROP_OFF":
            digits = re.search(r"\d+", raw)
            return name, {"oid": int(digits.group()) if digits else 0}, True
        if name == "NAVIGATE":
            quoted = re.search(r'"([^"]+)"', raw)
            return name, {"target": quoted.group(1)} if quoted else {}, True
        return name, {}, True
    return "WAIT", {}, False


@dataclass
class TurnRecord:
    """Exactly what the model saw and said on one turn."""

    step: int
    prompt_text: str
    image_count: int
    image_shapes: list[tuple[int, int]]
    raw_output: str
    parsed_action: str
    parsed_ok: bool
    prompt_tokens: int = 0
    response_tokens: int = 0
    latency_s: float = 0.0

    def to_dict(self) -> dict[str, Any]:
        return {
            "step": self.step,
            "prompt_text": self.prompt_text,
            "image_count": self.image_count,
            "image_shapes": [list(s) for s in self.image_shapes],
            "raw_output": self.raw_output,
            "parsed_action": self.parsed_action,
            "parsed_ok": self.parsed_ok,
            "prompt_tokens": self.prompt_tokens,
            "response_tokens": self.response_tokens,
            "latency_s": round(self.latency_s, 3),
        }


class Qwen3VLPolicy:
    """Zero-shot policy driving Qwen3-VL through the rollout adapter."""

    name = "qwen3vl_zero_shot"
    uses_privileged_state = False

    def __init__(
        self,
        adapter: Any,
        env: EnvSpec,
        instruction: str,
        *,
        max_new_tokens: int = 32,
        max_observation_chars: int = 1800,
        max_images_per_turn: int = 2,
    ):
        self.adapter = adapter
        self.env = env
        self.system_prompt = build_system_prompt(env, instruction)
        self.allowed = [a.upper() for a in env.enabled_actions]
        self.max_new_tokens = max_new_tokens
        self.max_observation_chars = max_observation_chars
        self.max_images_per_turn = max_images_per_turn
        self.state: Any = None
        self.turns: list[TurnRecord] = []
        self.raw_outputs: list[str] = []
        self.parse_failures = 0

    def reset(self) -> None:
        self.state = self.adapter.start_episode(self.system_prompt)
        self.turns = []
        self.raw_outputs = []
        self.parse_failures = 0

    def act(self, observation: Observation, runtime: Any, step_index: int) -> ActionEnvelope | None:
        import time

        if self.state is None:
            self.reset()

        images = list(getattr(runtime, "last_images", []) or [])[: self.max_images_per_turn]
        text = observation.text[: self.max_observation_chars]
        if observation.last_action_result is not None and observation.last_action_result.message:
            # the design plan M6: an invalid action gets typed feedback on the next turn.
            text += f"\n[previous action failed: {observation.last_action_result.message[:300]}]"

        started = time.time()
        self.adapter.observe(self.state, text, images)
        reply, span = self.adapter.generate(self.state, max_new_tokens=self.max_new_tokens)
        self.adapter.close_assistant_turn(self.state)
        latency = time.time() - started

        name, arguments, ok = parse_action(reply, self.allowed)
        if not ok:
            self.parse_failures += 1

        self.raw_outputs.append(reply)
        self.turns.append(
            TurnRecord(
                step=step_index,
                prompt_text=text,
                image_count=len(images),
                image_shapes=[(im.width, im.height) for im in images],
                raw_output=reply,
                parsed_action=f"{name}({arguments})",
                parsed_ok=ok,
                prompt_tokens=span.start,
                response_tokens=span.length,
                latency_s=latency,
            )
        )
        return ActionEnvelope(
            episode_id=observation.episode_id,
            step_index=step_index,
            action=TaskAction(name=name, arguments=arguments),
        )
