"""Tests for the model half of the harness/model split.

The parser is where a zero-shot policy silently degrades: a reply the parser
misreads becomes a wrong action nobody notices, and a reply it rejects when it
should not becomes a wasted turn. Both are tested here, along with the property
that matters most on Paris -- an action the environment does not offer must
never be executed just because the model wrote it.
"""

from __future__ import annotations

import pytest

from embodiedbench.agent.vlm_policy import build_system_prompt, parse_action
from embodiedbench.schemas.env_spec import (
    AffordanceInventory,
    EnvSpec,
    GraphSummary,
    NavigationStyle,
    SolvabilityEvidence,
)
from embodiedbench.schemas.environment import CertificationGrade, NavigationMode

GRAPH_ACTIONS = ["VIEW_ORDERS", "ACCEPT_ORDER", "PICKUP", "DROP_OFF", "WAIT", "MOVE_TO", "NAVIGATE"]
CARDINAL_ACTIONS = GRAPH_ACTIONS + ["MOVE"]


def _env(style: NavigationStyle, actions: list[str]) -> EnvSpec:
    return EnvSpec(
        env_id="e", map_name="m",
        navigation_style=style,
        navigation_modes=[NavigationMode.NAV_WAYPOINT],
        enabled_actions=actions,
        navigation_rationale="test",
        graph=GraphSummary(
            node_count=10, edge_count=10, mean_degree=2.0, max_degree=4,
            cardinal_fraction=1.0 if style is NavigationStyle.CARDINAL_AND_GRAPH else 0.2,
            largest_component_fraction=1.0,
        ),
        affordances=AffordanceInventory(counts={"restaurant": 2, "building": 5}),
        grade=CertificationGrade.B,
        quality_flags=[],
        solvability=SolvabilityEvidence(
            episodes=1, delivered_episodes=1, solvability_rate=1.0
        ),
    )


@pytest.mark.parametrize(
    "text,expected",
    [
        ("MOVE_TO(3)", ("MOVE_TO", {"_args": [3]})),
        ("MOVE_TO(0)", ("MOVE_TO", {"_args": [0]})),
        ("VIEW_ORDERS()", ("VIEW_ORDERS", {})),
        ("ACCEPT_ORDER(2)", ("ACCEPT_ORDER", {"_args": [2]})),
        ("PICKUP(orders=[0])", ("PICKUP", {"orders": [0]})),
        ("DROP_OFF(oid=1)", ("DROP_OFF", {"oid": 1})),
        ("WAIT()", ("WAIT", {})),
        ('NAVIGATE(target="restaurant 1")', ("NAVIGATE", {"target": "restaurant 1"})),
    ],
)
def test_well_formed_replies_parse(text, expected):
    name, arguments, ok = parse_action(text, GRAPH_ACTIONS)
    assert ok
    assert (name, arguments) == expected


@pytest.mark.parametrize(
    "text",
    [
        "I will now MOVE_TO(3) because it is closest.",
        "Action: MOVE_TO(3)",
        "```\nMOVE_TO(3)\n```",
        "move_to(3)",
    ],
)
def test_actions_are_found_inside_prose(text):
    name, arguments, ok = parse_action(text, GRAPH_ACTIONS)
    assert ok and name == "MOVE_TO" and arguments == {"_args": [3]}


@pytest.mark.parametrize("text", ["", "I am not sure what to do.", "GO NORTH", "MOVE_TO()"])
def test_unparseable_replies_fall_back_to_wait_and_say_so(text):
    """A malformed reply costs a turn; it must not end the episode."""
    name, _arguments, ok = parse_action(text, GRAPH_ACTIONS)
    assert name == "WAIT"
    assert ok is False


def test_an_action_the_environment_does_not_offer_is_refused():
    """The Paris invariant, at the policy boundary."""
    name, _arguments, ok = parse_action('MOVE(direction="forward")', GRAPH_ACTIONS)
    assert name == "WAIT" and ok is False
    # The same reply on a cardinal map is executed.
    name, arguments, ok = parse_action('MOVE(direction="forward")', CARDINAL_ACTIONS)
    assert ok and name == "MOVE" and arguments == {"direction": "forward"}


def test_first_offered_action_wins_when_several_are_mentioned():
    name, _arguments, ok = parse_action(
        "I could WAIT() but instead MOVE_TO(2)", GRAPH_ACTIONS
    )
    assert ok and name == "WAIT"


def test_system_prompt_lists_only_available_actions():
    graph_prompt = build_system_prompt(
        _env(NavigationStyle.GRAPH, GRAPH_ACTIONS), "deliver things"
    )
    assert "MOVE_TO(k)" in graph_prompt
    assert "MOVE(direction=" not in graph_prompt

    cardinal_prompt = build_system_prompt(
        _env(NavigationStyle.CARDINAL_AND_GRAPH, CARDINAL_ACTIONS), "deliver things"
    )
    assert "MOVE(direction=" in cardinal_prompt


def test_system_prompt_carries_the_instruction():
    prompt = build_system_prompt(_env(NavigationStyle.GRAPH, GRAPH_ACTIONS), "deliver in Paris")
    assert "deliver in Paris" in prompt
    assert "EXACTLY ONE action" in prompt
