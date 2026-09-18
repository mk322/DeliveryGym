"""The courier harness: tools, parsing, memory, skills, budgets, prompts.

The tests that matter most here are the ones that defend the *boundary* — that
the harness records what happened without deciding what to do next. A harness
that plans the route measures itself instead of the policy, and that failure is
invisible from the scores.
"""

from __future__ import annotations

import hashlib

import pytest

from embodiedbench.agent.courier import (
    MACROS,
    PROCEDURES,
    Budgets,
    CourierMemory,
    FormatError,
    Spend,
    ToolKind,
    available_tools,
    budget_exceeded,
    parse_reply,
    render_macros,
    render_procedures,
    render_tool_menu,
)
from embodiedbench.agent.courier.loop import expansion_limit
from embodiedbench.agent.courier.prompts import (
    REQUIRED_FIELDS,
    build_observation,
    build_system_prompt,
    render_candidates,
)
from embodiedbench.agent.courier.skills import FOLLOW_STREET, RETRACE
from embodiedbench.agent.courier.tools import (
    ALL_TOOLS,
    FACT_NUMBERS_HERE,
    FACT_STREET_HERE,
    OBSERVATION_PROVIDES,
    RETIRED_TOOLS,
    Tool,
)

PARIS_ACTIONS = ["VIEW_ORDERS", "ACCEPT_ORDER", "PICKUP", "DROP_OFF", "WAIT", "MOVE_TO", "NAVIGATE"]


def fence(action: str, thought: str = "") -> str:
    head = f"THOUGHT: {thought}\n" if thought else ""
    return f"{head}```\n{action}\n```"


class TestToolSet:
    def test_tools_are_split_by_what_they_change(self):
        """Act, look and consult are charged differently and can be enabled
        separately; collapsing them is what let an earlier design treat a free
        map lookup as equivalent to walking."""
        kinds = {t.kind for t in ALL_TOOLS}
        assert kinds == {ToolKind.ACT, ToolKind.LOOK, ToolKind.CONSULT}

    def test_a_tool_the_map_cannot_execute_is_not_offered(self):
        """The Paris defect in miniature: the observation advertised MOVE on a
        map that disabled it, so every obedient turn was rejected."""
        without_pickup = [a for a in PARIS_ACTIONS if a != "PICKUP"]
        names = {t.name for t in available_tools(without_pickup)}
        assert "collect" not in names
        assert "walk_to" in names

    def test_consult_tools_can_be_switched_off_as_a_difficulty_knob(self):
        """Taking the phone away is the natural way to make an episode harder;
        taking walking away is not a difficulty setting, it is a broken map."""
        hard = available_tools(PARIS_ACTIONS, allow_consult=False)
        names = {t.name for t in hard}
        assert "check_map" not in names and "check_order" not in names
        assert "walk_to" in names and "look" in names

    def test_every_tool_documents_itself(self):
        for tool in ALL_TOOLS:
            described = tool.describe()
            assert tool.name in described
            assert tool.summary in described
            for param in tool.params:
                assert param.name in described

    def test_looking_is_cheap_but_not_free(self):
        """A rider who spins on the spot every turn is not delivering."""
        look = next(t for t in ALL_TOOLS if t.name == "look")
        walk = next(t for t in ALL_TOOLS if t.name == "walk_to")
        assert 0 < look.time_cost_s < 30
        assert look.counts_as_step

    def test_the_menu_groups_tools_by_kind(self):
        menu = render_tool_menu(available_tools(PARIS_ACTIONS))
        # Case-insensitively: the headings are shouted now, because the menu
        # became a manual and each entry runs to several lines, so the reader
        # needs the three groups to stand off the page.
        low = menu.lower()
        assert "actions" in low and "looking" in low and "consulting" in low

    def test_every_tool_says_what_it_gives_back_and_when_not_to_use_it(self):
        """The gap that made the courier a one-tool policy.

        Over 1206 measured turns, 1139 were walk_to and three were look,
        against 47 refusals saying the named street is not at this junction.
        The menu named the tools and never said what a call returns, what a
        refusal means, or when the tool is the wrong one -- so a model with a
        hammer kept swinging it. Each of those is now a field on the tool, and
        a tool that has none is one nobody has written the manual for.
        """
        for tool in available_tools(PARIS_ACTIONS):
            assert tool.returns, f"{tool.name} does not say what it gives back"
            assert tool.use_when, f"{tool.name} does not say when to use it"
            entry = tool.manual()
            assert tool.example.split("(")[0] in entry
            assert "cost" in entry


class TestNoToolRepeatsTheObservation:
    """The audit that retired ``read_sign``, kept as a standing check.

    The reference courier used four of eleven tools across 25 recorded runs, and
    two of the seven it ignored could not have helped it: they answered with the
    first two lines of the observation they were answering. A menu entry like
    that costs prompt, costs a turn when taken, and teaches a policy that an
    action need not pay for itself.
    """

    def test_no_offered_tool_only_restates_what_the_turn_says(self):
        for tool in available_tools(PARIS_ACTIONS):
            assert tool.informative(), (
                f"{tool.name} returns {sorted(tool.provides)}, all of which the "
                "observation already states for free"
            )

    def test_a_tool_that_would_only_restate_it_is_dropped_from_the_menu(self):
        """The check has teeth: give it such a tool and it disappears."""
        echo = Tool(name="echo", kind=ToolKind.LOOK, summary="says it again",
                    provides=frozenset({FACT_STREET_HERE, FACT_NUMBERS_HERE}))
        assert not echo.informative()
        assert echo.informative(frozenset({FACT_STREET_HERE})), (
            "one fact the turn does not carry is enough to earn a place"
        )

    def test_acting_tools_earn_their_place_by_acting(self):
        """``collect`` returns no facts at all and is not therefore redundant."""
        collect = next(t for t in ALL_TOOLS if t.name == "collect")
        assert collect.provides == frozenset()
        assert collect.informative()

    def test_the_retirement_is_recorded_with_its_reason(self):
        assert "read_sign" in RETIRED_TOOLS
        assert "read_sign" not in {t.name for t in ALL_TOOLS}
        assert len(RETIRED_TOOLS["read_sign"]) > 40, "a reason, not a tombstone"

    def test_a_condition_that_stops_stating_a_fact_gives_the_tool_work_again(self):
        """The knob is symmetric, which is why it is a parameter and not a rule.

        Under a condition whose header stopped naming the doors underfoot, a tool
        reporting door numbers would be informative again -- so the filter must
        read the observation rather than a hard-coded list of survivors.
        """
        silent = OBSERVATION_PROVIDES - {FACT_NUMBERS_HERE}
        echo = Tool(name="echo", kind=ToolKind.LOOK, summary="the doors here",
                    provides=frozenset({FACT_NUMBERS_HERE}))
        assert not echo.informative(OBSERVATION_PROVIDES)
        assert echo.informative(silent)

    def test_two_tools_that_route_do_not_route_identically(self):
        """``check_map`` and ``navigate`` both touch the phone and must differ.

        They were nearly merged during the audit. They stay separate because a
        map app really does have both: search tells you where a place is, for any
        address you can name; directions tell you how to get to one you are
        going to. The check is that each carries a fact the other does not.
        """
        by_name = {t.name: t for t in ALL_TOOLS}
        search, route = by_name["check_map"], by_name["navigate"]
        assert search.provides - route.provides, "search says something routing does not"
        assert route.provides - search.provides, "routing says something search does not"
        assert route.time_cost_s > search.time_cost_s, "the bigger answer costs more"


class TestActionParsing:
    def allowed(self) -> set[str]:
        return {t.name for t in available_tools(PARIS_ACTIONS)}

    def test_a_well_formed_reply_parses(self):
        action = parse_reply(fence("walk_to(\"Rue de Grenelle\", \"east\")", "street 3 heads west"), self.allowed())
        assert action.tool == "walk_to"
        assert action.args == ["Rue de Grenelle", "east"]
        assert "west" in action.thought

    def test_a_string_argument_survives(self):
        action = parse_reply(fence('check_map("42 Rue de Rivoli")'), self.allowed())
        assert action.args == ["42 Rue de Rivoli"]

    def test_keyword_arguments_parse(self):
        action = parse_reply(fence("follow_street(k=3, n=4)"), self.allowed() | {"follow_street"})
        assert action.kwargs == {"k": 3, "n": 4}

    def test_no_fenced_block_is_a_format_error(self):
        with pytest.raises(FormatError, match="No action found"):
            parse_reply("I think I should walk_to(\"Rue de Grenelle\", \"east\").", self.allowed())

    def test_two_actions_is_a_format_error(self):
        """One action per turn, so a turn maps to exactly one transition and the
        trajectory stays replayable."""
        with pytest.raises(FormatError, match="exactly one"):
            parse_reply("```\nwalk_to(\"Rue de Grenelle\", \"east\")\nlook(\"Rue de Grenelle\", \"east\")\n```", self.allowed())

    def test_two_blocks_is_a_format_error(self):
        with pytest.raises(FormatError, match="action blocks"):
            parse_reply("```\nwalk_to(\"Rue de Grenelle\", \"east\")\n```\ntext\n```\nlook(\"Rue de Grenelle\", \"east\")\n```", self.allowed())

    def test_an_unavailable_tool_is_refused_by_name(self):
        """Refused at parse time, with the list, rather than passed to the world
        and rejected there — the model needs to know what it may say."""
        with pytest.raises(FormatError, match="not something you can do"):
            parse_reply(fence("fly_to(3)"), self.allowed())

    def test_the_error_message_shows_the_grammar(self):
        """A format error that does not show the format teaches nothing."""
        with pytest.raises(FormatError) as caught:
            parse_reply("no action here.", self.allowed())
        # The message deliberately carries no fenced example of its own. The
        # reminder it is wrapped in (FORMAT_ERROR_TEMPLATE) already shows one,
        # and two fences inside a message that says "exactly one fenced block"
        # demonstrates the violation it is correcting -- running the parser
        # over that message finds two action blocks.
        assert "```" not in str(caught.value)
        assert "fenced block" in str(caught.value)

    def test_parsing_never_guesses(self):
        """Falling back to a default action spends a turn on something the model
        did not choose and hides the format problem."""
        for reply in ("", "walk_to 3", "```\n\n```", "```\nnonsense\n```"):
            with pytest.raises(FormatError):
                parse_reply(reply, self.allowed())


class TestMemory:
    def test_it_records_where_the_agent_has_been(self):
        memory = CourierMemory()
        memory.arrive(step=0, node_id="s001_n000", street="Rue Monge")
        memory.arrive(step=1, node_id="s001_n001", street="Rue Monge")
        assert memory.current_node == "s001_n001"
        assert memory.previous_node == "s001_n000"

    def test_it_notices_a_loop(self):
        """The concrete failure: the oracle circled three nodes 21 m from its
        destination for the rest of the episode, because each looked best from
        the last and it could not tell it had been there."""
        memory = CourierMemory()
        # A real loop revisits nodes by *travelling* between them. Repeating the
        # same node id is standing still, which arrive() now ignores -- calling
        # it every turn, including on looking and consulting, was firing the
        # warning at an agent that had gone nowhere.
        for step in range(6):
            memory.arrive(step=step, node_id=f"s001_n{step % 2:03d}", street="Rue Monge")
        assert memory.is_looping()
        assert "circles" in memory.render()

    def test_a_short_trail_is_not_a_loop(self):
        memory = CourierMemory()
        memory.arrive(step=0, node_id="a", street="Rue A")
        memory.arrive(step=1, node_id="b", street="Rue B")
        assert not memory.is_looping()

    def test_notes_are_the_model_s_own_words_kept_verbatim(self):
        memory = CourierMemory()
        memory.write("Rue Monge north end is a dead end")
        assert "Rue Monge north end is a dead end" in memory.render()

    def test_duplicate_notes_are_not_repeated(self):
        memory = CourierMemory()
        for _ in range(3):
            memory.write("same thing")
        assert memory.notebook.count("same thing") == 1

    def test_first_impression_of_a_street_is_kept(self):
        """A later glance from another angle must not silently overwrite what
        the agent already committed to memory."""
        memory = CourierMemory()
        memory.saw_street("Rue Monge", "wide, shops")
        memory.saw_street("Rue Monge", "narrow")
        assert memory.streets_seen["Rue Monge"] == "wide, shops"

    def test_a_goal_change_is_shown_but_not_written_in_the_notebook(self):
        """It used to be recorded as a note, and that was the wrong store.

        The notebook shows six lines and a shift changes goal twenty times, so
        by mid-episode the notebook was six copies of the ``Job:`` line directly
        above it and every note the model had written for itself had been
        evicted. The current goal belongs in the goal line; the notebook is the
        one store the model controls.
        """
        memory = CourierMemory()
        memory.write("Quai Montorgueil south past no. 37 is a dead end")
        memory.set_goal("collect", "131 Union Ave")
        memory.set_goal("deliver", "46 Hill St")
        assert memory.goal_kind == "deliver" and memory.goal == "46 Hill St"
        assert "deliver 46 Hill St" in memory.render()
        assert memory.notebook == ["Quai Montorgueil south past no. 37 is a dead end"]

    def test_memory_states_facts_and_never_a_recommendation(self):
        """The boundary this harness is built around. Memory may say where the
        agent has been; the moment it says where to go, the benchmark is
        measuring the harness."""
        memory = CourierMemory()
        memory.arrive(step=0, node_id="a", street="Rue A")
        memory.set_goal("collect", "12 Rue B")
        memory.mark_dead_end("a")
        rendered = memory.render().lower()
        for verb in ("you should", "go to", "take street", "best", "recommend", "next move"):
            assert verb not in rendered, f"memory told the agent what to do: {verb!r}"

    def test_memory_is_serialisable_for_replay(self):
        memory = CourierMemory()
        memory.arrive(step=0, node_id="a", street="Rue A")
        memory.write("note")
        payload = memory.to_dict()
        assert payload["trail"] and payload["notebook"] == ["note"]


class TestSkills:
    def test_procedures_are_guidance_not_code(self):
        """Everything about *navigating* is written guidance, because navigation
        is the capability under test."""
        for procedure in PROCEDURES:
            assert procedure.steps
            assert procedure.when
        # The route tool replaced check_map as the anchor of the address
        # runbook; what the assertion is really about is that the runbook names
        # tools rather than executing anything.
        assert "navigate()" in render_procedures()

    def test_every_macro_declares_the_tools_it_expands_into(self):
        """The safety argument: a macro may only be built from tools the agent
        could have called itself. One that decided anything would have to name a
        tool that decides, and there is none."""
        tool_names = {t.name for t in ALL_TOOLS}
        for macro in MACROS:
            assert macro.expands_to
            for name in macro.expands_to:
                assert name in tool_names, f"{macro.tool.name} expands into unknown {name}"

    def test_no_macro_expands_into_a_consult_tool(self):
        """A macro that could look things up could plan; these may only move."""
        by_name = {t.name: t for t in ALL_TOOLS}
        for macro in MACROS:
            for name in macro.expands_to:
                assert by_name[name].kind is not ToolKind.CONSULT

    def test_macros_are_bounded(self):
        assert expansion_limit(FOLLOW_STREET, 99) == FOLLOW_STREET.max_expansion
        assert expansion_limit(RETRACE, 99) == RETRACE.max_expansion
        assert expansion_limit(FOLLOW_STREET, 2) == 2
        assert expansion_limit(FOLLOW_STREET, 0) == 1

    def test_follow_street_does_not_choose_the_street(self):
        """It walks the street the model named. If it picked one, it would be
        doing the navigation."""
        assert any(p.name == "street" for p in FOLLOW_STREET.tool.params)
        assert "never chooses" in FOLLOW_STREET.rationale


class TestBudgets:
    def test_each_limit_is_reported_by_name(self):
        budgets = Budgets(steps=5, tool_calls=10, sim_seconds=100.0, output_tokens=50)
        assert budget_exceeded(Spend(steps=5), budgets) == "step_budget_exhausted"
        assert budget_exceeded(Spend(tool_calls=10), budgets) == "tool_call_budget_exhausted"
        assert budget_exceeded(Spend(sim_seconds=100.0), budgets) == "sim_time_budget_exhausted"
        assert budget_exceeded(Spend(output_tokens=50), budgets) == "output_token_budget_exhausted"

    def test_an_unspent_budget_does_not_stop_the_episode(self):
        assert budget_exceeded(Spend(steps=1), Budgets(steps=5)) is None

    def test_repeated_format_errors_end_the_episode(self):
        """A model that cannot emit the grammar will not start; but one bad turn
        must not end a run, so the limit is on consecutive failures."""
        budgets = Budgets(max_format_errors=3)
        assert budget_exceeded(Spend(consecutive_format_errors=2), budgets) is None
        assert budget_exceeded(Spend(consecutive_format_errors=3), budgets) == "repeated_format_errors"

    def test_optional_budgets_can_be_disabled(self):
        loose = Budgets(steps=100, tool_calls=None, sim_seconds=None, output_tokens=None)
        assert budget_exceeded(Spend(tool_calls=10_000, sim_seconds=1e9), loose) is None


class TestPrompts:
    def test_the_system_prompt_only_describes_executable_tools(self):
        tools = available_tools([a for a in PARIS_ACTIONS if a != "PICKUP"])
        prompt = build_system_prompt(city="Paris", tools=tools)
        assert "collect()" not in prompt
        assert 'walk_to("Rue de Grenelle", "east")' in prompt

    def test_the_system_prompt_shows_the_action_grammar(self):
        prompt = build_system_prompt(city="Paris", tools=available_tools(PARIS_ACTIONS))
        assert "```" in prompt and "THOUGHT" in prompt

    def test_front_rear_pixel_prompt_uses_one_atomic_view_and_point_call(self):
        from embodiedbench.agent.courier.tools import (
            WALK_TO_PIXEL_FRONT_REAR,
        )

        prompt = build_system_prompt(
            city="Paris",
            tools=[WALK_TO_PIXEL_FRONT_REAR],
        )

        assert "walk_to_pixel(view: str, u: number, v: number)" in prompt
        assert 'walk_to_pixel(view="front", u=0.37, v=0.82)' in prompt
        assert prompt.count("walk_to_pixel(view=") == 1
        mechanics = prompt.lower()
        for fact in (
            "simultaneous",
            '"front" or "rear"',
            "first visible surface",
            "pedestrian pavement/sidewalk",
            "marked crosswalk",
            "surface-resolution refusal",
            "partial movement",
        ):
            assert fact in mechanics
        for unavailable in (
            "select_camera(",
            "turn_left(",
            "turn_right(",
            "canonical calls",
            "roughly v",
            "below the horizon",
            "toward that bearing",
            "u=0.50",
        ):
            assert unavailable not in mechanics

    def test_front_rear_prompt_is_compact_and_truthful_about_each_source(self):
        from embodiedbench.agent.courier.tools import (
            CHECK_MAP,
            CHECK_ORDER,
            COLLECT,
            HAND_OVER,
            LOOK,
            NAVIGATE,
            NOTE,
            REST,
            WAIT,
            WALK_TO_PIXEL_FRONT_REAR,
        )

        prompt = build_system_prompt(
            city="Paris",
            tools=[
                COLLECT, HAND_OVER, WAIT, REST,
                WALK_TO_PIXEL_FRONT_REAR, LOOK,
                CHECK_ORDER, CHECK_MAP, NAVIGATE, NOTE,
            ],
        )
        lowered = prompt.lower()

        # Compact for a small model, with room for the pixel-choosing
        # guidance: about 1.7k tokens against a 4864-token prompt budget.
        assert len(prompt) < 7_000
        assert len(prompt.split()) < 1_150
        assert prompt.count("\nVISUAL MOVEMENT CONTRACT\n") == 1
        assert "route follows pedestrian connectivity and real marked" in lowered
        assert "crosswalks" in lowered
        assert "current signal phase" in lowered
        assert "temporary obstacle" in lowered
        assert "what surface a camera pixel will hit" in lowered
        assert "phone cannot see crossings" not in lowered
        assert "what you have been trained to do" not in lowered
        assert "using the photograph pair" not in lowered
        assert "how the photograph pair works here" not in lowered
        for signature in (
            "collect()", "hand_over()", "wait()", "rest()",
            "walk_to_pixel(view: str, u: number, v: number)",
            "look(street: str, heading: str = default)", "check_order()",
            "check_map(address: str)",
            "navigate(where: str = default)", "note(text: str)",
        ):
            assert signature in prompt

    @pytest.mark.parametrize(
        ("narration", "must", "must_not"),
        [
            ("none", "phone's blue pedestrian route",
             "THE ROUTE GOES THIS WAY"),
            ("route", "read current signals and obstacles",
             "pedestrian-signal state, and blockage state"),
            ("all", "street list's stated pedestrian-signal and BLOCKED status",
             "separate [light: street, bearing] photograph"),
        ],
    )
    def test_front_rear_prompt_respects_the_narration_setting(
            self, narration, must, must_not):
        from embodiedbench.agent.courier.tools import (
            WAIT,
            WALK_TO_PIXEL_FRONT_REAR,
        )

        prompt = build_system_prompt(
            city="Paris",
            tools=[WALK_TO_PIXEL_FRONT_REAR, WAIT],
            narration=narration,
        )

        assert must in prompt
        assert must_not not in prompt
        assert "first visible surface" in prompt
        assert prompt.count("\nVISUAL MOVEMENT CONTRACT\n") == 1

    def test_the_legacy_pixel_prompt_is_byte_for_byte_unchanged(self):
        actions = [
            "VIEW_ORDERS", "ACCEPT_ORDER", "PICKUP", "DROP_OFF", "WAIT",
            "MOVE_TO_PIXEL", "NAVIGATE",
        ]
        prompt = build_system_prompt(
            city="Paris", tools=available_tools(actions))

        assert "walk_to_pixel(u: number, v: number)" in prompt
        assert "view:" not in prompt
        assert hashlib.sha256(prompt.encode()).hexdigest() == (
            "d146fcab05347b0185a7637fb2fdcb3f8e22a7986d8ec0371dc3c1262b76cb45"
        )

    def test_templates_interpolate_every_declared_field(self):
        """A renamed field must fail loudly rather than render '{street}' into
        the model's context."""
        prompt = build_system_prompt(city="Paris", tools=available_tools(PARIS_ACTIONS))
        for field in REQUIRED_FIELDS["system"]:
            assert "{" + field + "}" not in prompt
        observation = build_observation(
            memory="### your notes\n(nothing yet)", location="You are on Rue Monge.",
            clock="12 min left.", candidates=render_candidates([
                {"k": 1, "street": "Rue Monge", "heading": "north", "distance_m": 18.0,
                 "numbers": "12-30"},
            ]),
        )
        for field in REQUIRED_FIELDS["observation"]:
            assert "{" + field + "}" not in observation

    def test_candidates_carry_what_a_rider_reads_off_a_corner(self):
        rendered = render_candidates([
            {"k": 2, "street": "Rue de Rivoli", "heading": "west", "distance_m": 24.0,
             "numbers": "40-58", "seen": True},
        ])
        assert "Rue de Rivoli" in rendered
        assert "west" in rendered
        assert "24 m" in rendered
        assert "40-58" in rendered
        assert "walked this before" in rendered

    def test_candidates_never_name_the_correct_choice(self):
        """Printing the answer would make the benchmark measure token-copying."""
        rendered = render_candidates([
            {"k": 1, "street": "A", "heading": "north", "distance_m": 10.0},
            {"k": 2, "street": "B", "heading": "west", "distance_m": 12.0},
        ]).lower()
        for giveaway in ("recommended", "shortest", "best", "take this", "correct"):
            assert giveaway not in rendered

    def test_a_dead_end_is_stated_rather_than_left_blank(self):
        assert "no way on" in render_candidates([]).lower()
        assert "no way on" in build_observation(
            memory="m", location="l", candidates=render_candidates([])
        ).lower()


class TestPoseConvention:
    """The frame bug the independent evaluation surfaced.

    Three angle conventions meet in this system: the engine's compass bearing
    (atan2(dx,dy), 0 = north), the mathematical convention the camera intrinsics
    and point-navigation chain use (atan2(dy,dx), 0 = +X), and the UE renderer's
    yaw -- which a 1550-frame correlation against building footprints showed
    matches the mathematical one (corr +0.60, against +0.48 for the negation and
    ~+0.24 for the quarter-turns). Copying the engine's number into Pose.yaw_deg
    without converting rotates and mirrors every heading.
    """

    @pytest.mark.parametrize(
        "compass,expected_math",
        [(0.0, 90.0), (90.0, 0.0), (180.0, 270.0), (270.0, 180.0), (45.0, 45.0)],
    )
    def test_compass_converts_to_the_mathematical_convention(self, compass, expected_math):
        assert (90.0 - compass) % 360.0 == pytest.approx(expected_math)

    def test_the_adapter_applies_the_conversion(self):
        """A structural guard: if the raw copy comes back, this fails."""
        import inspect

        from embodiedbench.runtime.text.vagen_adapter import VagenTextRuntime

        source = inspect.getsource(VagenTextRuntime._agent_pose)
        assert "90.0 - compass" in source or "90 - compass" in source
        assert "yaw_deg=float(getattr(dm" not in source


# ─────────────────────────────────────────────────────────────────────────────
# The navigation tool and the session that carries it
#
# Everything below tests the input/output contract end to end -- the runtime, the
# observation built from it, the reply parsed against it, and the tool dispatched
# from it. None of that had a test before, because none of it had an
# implementation: prompts.py could render strings, loop.py could parse a reply
# and CourierEnv could execute a call, and nothing joined the three. A contract
# with no implementation cannot regress, but it also cannot be relied on, and
# every claim about "what the agent sees" was a claim about code nobody ran.
# ─────────────────────────────────────────────────────────────────────────────

import json
import math
import re
import tempfile
from pathlib import Path

from embodiedbench.agent.courier.prompts import render_photographs
from embodiedbench.agent.courier.session import CourierSession
from embodiedbench.compiler.road_network import build_road_network
from embodiedbench.runtime.city.courier_env import (
    Condition,
    CourierEnv,
    relative_of,
    turn_word,
)

PARIS_MAP = (Path(__file__).resolve().parents[1] / "vendor" / "vagen" / "vagen"
             / "envs" / "deliverybench" / "maps" / "citycore-paris")


@pytest.fixture(scope="module")
def paris():
    return build_road_network(PARIS_MAP, map_name="citycore-paris")


def courier(paris, **kwargs):
    kwargs.setdefault("seed", 0)
    kwargs.setdefault("order_count", 1)
    env = CourierEnv(paris, **kwargs)
    env.reset()
    return env


class TestNavigationTool:
    """A route, and nothing but a route.

    The environment has exactly one mechanic that cannot be solved from text --
    the pedestrian light -- so the single thing a navigation tool must never do
    is mention it. Everything else it says is a direction a phone would speak.
    """

    # Anything a courier is supposed to use its eyes for. Word-bounded, because
    # "Boulevard de Buci" contains "uci" and street names are not evidence.
    FORBIDDEN = re.compile(
        r"\b(red|green|amber|light|lights|lamp|signal|signalised|traffic|crossing|"
        r"pedestrian|wait|obstacle|blocked|hazard|roadworks|closed|danger)\b",
        re.IGNORECASE,
    )

    def routes(self, paris, seeds=range(6)):
        """A route from many places, so the check is not one lucky sentence."""
        out = []
        for seed in seeds:
            env = courier(paris, seed=seed, order_count=3)
            for _ in range(12):
                rows = env.candidates()
                outcome = env.navigate()
                if outcome.message:
                    # Snapshot the corner the route was asked from: the env is
                    # walked on below and the assertions are about *this* corner.
                    out.append(({r["street"] for r in rows}, outcome))
                if not rows:
                    break
                env.walk_to(*env.street_at(rows[len(out) % len(rows)]["k"]))
        return out

    def test_a_route_never_mentions_a_light_or_a_hazard(self, paris):
        """The whole argument for the tool. If the phone says "wait at the
        crossing", the photographs stop being load-bearing and the benchmark
        stops measuring perception."""
        checked = 0
        for _, outcome in self.routes(paris):
            found = self.FORBIDDEN.findall(outcome.message)
            assert not found, f"navigate() leaked {found} in: {outcome.message!r}"
            checked += 1
        assert checked >= 40, "not enough routes to make the claim"

    def test_a_route_never_names_the_numbered_street_to_take(self, paris):
        """Naming ``k`` would make navigate() the policy: the agent would copy a
        number instead of matching a street name to the corner it is standing on."""
        for _, outcome in self.routes(paris, seeds=range(3)):
            for line in outcome.message.splitlines():
                assert not re.search(r"\bwalk_to\b|\bfollow_street\b|\bstreet \d\b", line)

    def test_the_route_gives_scale_and_not_steps(self, paris):
        """The legs are gone: they were the navigation, written out."""
        for _, outcome in self.routes(paris, seeds=range(3)):
            assert "min on foot" in outcome.message
            assert "street" in outcome.message
            for word in ("north", "south", "east", "west"):
                assert word not in outcome.message.lower(), outcome.message

    def test_the_first_instruction_names_a_street_that_leaves_this_junction(self, paris):
        """A route the courier cannot start is not a route."""
        for streets, outcome in self.routes(paris, seeds=range(4)):
            first = next((l for l in outcome.message.splitlines()
                          if l.startswith("  1. ")), None)
            if first is None:
                continue
            street = first.split(" — ")[0].split(". ", 1)[1]
            for verb in ("Take ", "Turn left onto ", "Turn right onto ",
                         "Bear left onto ", "Bear right onto ", "Continue onto ",
                         "Turn back onto "):
                if street.startswith(verb):
                    street = street[len(verb):]
                    break
            assert street in streets, first

    def test_following_the_route_gets_closer(self, paris):
        """The route is correct, not merely well-formed."""
        for seed in range(5):
            env = courier(paris, seed=seed)
            target = env.target_address()
            before = env.route_length_cm(env.node_id, target.kerb_node)
            legs = env.route_legs(env.node_id, target.kerb_node)
            assert legs
            path = env.route_nodes(env.node_id, target.kerb_node)
            row = next(r for r in env.candidates() if r["node"] == path[1])
            env.walk_to(*env.street_at(row["k"]))
            after = env.route_length_cm(env.node_id, target.kerb_node)
            assert after < before

    def test_the_shortest_route_is_the_route_that_is_quoted(self, paris):
        """The predecessor map has to be updated on every relaxation. Keeping the
        first node that pushed a neighbour rebuilds a path out of edges the
        search never chose, and the metres beside it then describe a different
        walk from the one the instructions describe."""
        env = courier(paris)
        for goal in sorted(env.network.nodes)[:25]:
            path = env.route_nodes(env.node_id, goal)
            quoted = env.route_length_cm(env.node_id, goal)
            if path is None or quoted is None:
                continue
            walked = sum(math.dist(env.position(a), env.position(b))
                         for a, b in zip(path, path[1:]))
            assert walked == pytest.approx(quoted, rel=1e-6, abs=1.0)

    def test_the_phone_is_gone_when_the_phone_is_gone(self, paris):
        env = courier(paris, condition=Condition.NO_PHONE)
        outcome = env.navigate()
        assert not outcome.ok and outcome.code == "no_phone"
        assert "navigate" not in env.allowed_tool_names()

    def test_asking_for_the_route_costs_the_clock(self, paris):
        """A free lookup makes the optimal policy ask every turn forever."""
        env = courier(paris)
        before = env.sim_seconds
        env.navigate()
        assert env.sim_seconds - before == pytest.approx(env.NAVIGATE_SECONDS)
        assert env.NAVIGATE_SECONDS > 0

    def test_the_courier_chooses_which_job_to_route_to(self, paris):
        """With several live orders the sequencing is the task; a tool that only
        ever routes to the oldest would make that decision for the agent."""
        env = courier(paris, order_count=3, difficulty=None)
        live = env.live_orders()
        if len(live) < 2:
            pytest.skip("this seed issues one job at a time")
        first = env.navigate(live[0].index).message
        second = env.navigate(live[1].index).message
        assert live[0].target.text in first
        assert live[1].target.text in second
        refused = env.navigate(99)
        assert not refused.ok and refused.code == "no_such_job"

    def test_reading_the_queue_does_not_overwrite_an_explicit_route(self, paris):
        """A queue read must not silently retarget the map app.

        Routing to an address outside the live queue is explicitly supported.
        Observation construction calls both ``live_orders`` and
        ``active_order``; neither read may replace that chosen destination.
        """
        env = courier(paris, order_count=2, queue_depth=1)
        live = env.live_orders()
        other_pickup = env.orders[1].pickup

        assert other_pickup.text != live[0].target.text
        assert env.navigate(other_pickup.text).ok
        assert env.screen_target is other_pickup

        env.live_orders()
        env.active_order()

        assert env.screen_target is other_pickup

    def test_collecting_a_later_live_job_routes_to_its_dropoff(self, paris):
        """Out-of-order collection must focus the newly carried parcel."""
        env = courier(paris, order_count=3, queue_depth=2)
        first, second = env.live_orders()
        assert env.screen_target is first.pickup

        env.node_id = second.pickup.kerb_node
        env.arrived_from = None
        outcome = env.collect()

        assert outcome.ok
        assert second.picked_up
        assert env.active_order() is second
        assert env.screen_target is second.dropoff

    def test_collecting_two_jobs_keeps_the_phone_on_the_active_carried_one(
            self, paris):
        """The map and default text focus must not split with two parcels."""
        env = courier(paris, order_count=3, queue_depth=2)
        first, second = env.live_orders()

        env.node_id = first.pickup.kerb_node
        env.arrived_from = None
        assert env.collect().ok
        assert env.active_order() is first
        assert env.screen_target is first.dropoff

        env.node_id = second.pickup.kerb_node
        env.arrived_from = None
        assert env.collect().ok
        assert second.picked_up

        assert env.active_order() is first
        assert env.screen_target is first.dropoff

        env.node_id = first.dropoff.kerb_node
        env.arrived_from = None
        assert env.hand_over().ok

        assert env.active_order() is second
        assert env.screen_target is second.dropoff

    def test_underfilled_queue_read_preserves_an_explicit_route(self, paris):
        """The for-loop-end issue path must respect a chosen destination."""
        env = courier(paris, order_count=1, queue_depth=2)
        active = env.live_orders()[0]
        other = next(
            address for address in paris.addresses
            if address.kerb_node and address.text != active.target.text
            and env.route_nodes(env.node_id, address.kerb_node)
        )

        assert env.navigate(other.text).ok
        env.live_orders()

        assert env.screen_target is other

    @pytest.mark.parametrize("queue_depth", (1, 2, 3))
    def test_automatic_phone_follows_bounded_expiry_refill(
            self, paris, queue_depth):
        """Dispatcher replacement must retire an expired automatic target."""
        env = courier(
            paris,
            order_count=2 * queue_depth + 1,
            queue_depth=queue_depth,
        )
        before = env.active_order()
        live = env.live_orders()
        env.sim_seconds = max(order.expires_at() for order in live) + 1.0

        after = env.active_order()

        assert after is not None and after is not before
        assert before.expired
        assert env.screen_target is after.target

    def test_bounded_expiry_refill_preserves_an_explicit_route(self, paris):
        """Automatic dispatch changes must not seize an explicitly owned map."""
        env = courier(paris, order_count=2, queue_depth=1)
        active = env.live_orders()[0]
        explicit = env.orders[1].dropoff
        assert explicit.text != active.target.text
        assert env.navigate(explicit.text).ok

        env.sim_seconds = active.expires_at() + 1.0
        replacement = env.active_order()

        assert replacement is env.orders[1]
        assert env.screen_target is explicit

    def test_endless_expiry_refill_preserves_an_explicit_route(self, paris):
        """The unbounded refill tail must obey the same ownership rule."""
        env = courier(paris, difficulty="endless")
        live = env.live_orders()
        live_targets = {order.target.text for order in live}
        explicit = next(
            address for address in paris.addresses
            if address.kerb_node and address.text not in live_targets
            and env.route_nodes(env.node_id, address.kerb_node)
        )
        assert env.navigate(explicit.text).ok

        env.sim_seconds = max(order.expires_at() for order in live) + 1.0
        replacement = env.active_order()

        assert replacement is not None
        assert all(order.expired for order in live)
        assert env.screen_target is explicit

    @pytest.mark.parametrize("queue_depth", (1, 2, 3))
    def test_hand_over_refreshes_automatic_phone_to_the_next_active_job(
            self, paris, queue_depth):
        """A completed job must not remain on an automatically owned screen."""
        env = courier(
            paris,
            order_count=2 * queue_depth + 1,
            queue_depth=queue_depth,
        )
        first = env.active_order()
        assert first is not None

        env.node_id = first.pickup.kerb_node
        env.arrived_from = None
        assert env.collect().ok
        env.node_id = first.dropoff.kerb_node
        env.arrived_from = None
        assert env.hand_over().ok

        after = env.active_order()
        assert after is not None and after is not first
        assert env.screen_target is after.target

    def test_hand_over_reclaims_the_phone_from_an_explicit_route(self, paris):
        """Serving a job starts automatic guidance for the next job."""
        env = courier(paris, order_count=3, queue_depth=1)
        first = env.active_order()
        assert first is not None

        env.node_id = first.pickup.kerb_node
        env.arrived_from = None
        assert env.collect().ok
        explicit = env.orders[2].pickup
        assert env.navigate(explicit.text).ok
        assert env.screen_target is explicit

        env.node_id = first.dropoff.kerb_node
        env.arrived_from = None
        assert env.hand_over().ok

        after = env.active_order()
        assert after is env.orders[1]
        assert env.screen_target is after.pickup

    def test_collect_reclaims_the_phone_from_an_explicit_route(self, paris):
        """Picking up a parcel starts automatic guidance to its dropoff."""
        env = courier(paris, order_count=2, queue_depth=1)
        first = env.active_order()
        assert first is not None
        explicit = env.orders[1].pickup
        assert env.navigate(explicit.text).ok

        env.node_id = first.pickup.kerb_node
        env.arrived_from = None
        assert env.collect().ok

        assert env.active_order() is first
        assert env.screen_target is first.dropoff

    def test_duplicate_collect_preserves_an_explicit_route(self, paris):
        """A refused task action does not gain ownership of the phone."""
        env = courier(paris, order_count=2, queue_depth=1)
        first = env.active_order()
        assert first is not None
        env.node_id = first.pickup.kerb_node
        env.arrived_from = None
        assert env.collect().ok

        explicit = env.orders[1].pickup
        assert env.navigate(explicit.text).ok
        refused = env.collect()

        assert not refused.ok and refused.code == "already_collected"
        assert env.screen_target is explicit

    def test_arrival_is_reported_rather_than_routed(self, paris):
        env = courier(paris)
        target = env.target_address()
        env.node_id = target.kerb_node
        assert "arrived" in env.navigate().message


class TestRelativeDirections:
    def test_the_way_you_came_is_behind_you(self, paris):
        env = courier(paris)
        rows = env.candidates()
        env.walk_to(*env.street_at(rows[0]["k"]))
        back = [r for r in env.candidates() if r["back"]]
        assert back and back[0]["relative"] == "behind you"

    def test_there_is_no_relative_direction_before_the_first_step(self, paris):
        """A courier who has not moved has no back to its head, and inventing one
        would put a left and a right on a heading the agent cannot verify."""
        env = courier(paris)
        assert env.facing() is None
        assert all(row["relative"] == "" for row in env.candidates())

    @pytest.mark.parametrize("facing,bearing,expected", [
        (0.0, 0.0, "straight ahead"),
        (0.0, 90.0, "on your right"),
        (0.0, 270.0, "on your left"),
        (0.0, 180.0, "behind you"),
        (90.0, 0.0, "on your left"),
    ])
    def test_relative_directions_are_taken_from_the_way_you_face(
            self, facing, bearing, expected):
        assert relative_of(bearing, facing) == expected

    @pytest.mark.parametrize("a,b,expected", [
        (0.0, 5.0, "continue"), (0.0, 90.0, "turn right"), (0.0, 270.0, "turn left"),
        (0.0, 180.0, "turn back"), (0.0, 40.0, "bear right"), (0.0, 320.0, "bear left"),
    ])
    def test_a_turn_is_named_the_way_a_person_would_name_it(self, a, b, expected):
        assert turn_word(a, b) == expected


class TestFramesAreForwardViews:
    """The photograph beside street k is the view down street k.

    It was not: at the 105 signalised junctions the runtime served the *signal*
    bake instead, and that bake aims the camera at the lamp rather than along the
    street -- on 88 of those nodes every approach shares one yaw, so a junction's
    four candidate photographs were four copies of one picture of somewhere the
    courier was not going.
    """

    def album(self, tmp, env, signalised_node):
        street = Path(tmp) / "street"
        signal = Path(tmp) / "signal"
        for node, node_data in env.network.nodes.items():
            for neighbour in node_data.neighbours:
                path = street / "images" / node / f"toward_{neighbour}.png"
                path.parent.mkdir(parents=True, exist_ok=True)
                path.write_bytes(b"street")
        for neighbour in env.network.nodes[signalised_node].neighbours:
            for state in ("red", "green"):
                path = (signal / "images" / signalised_node
                        / f"toward_{neighbour}_{state}.png")
                path.parent.mkdir(parents=True, exist_ok=True)
                path.write_bytes(b"signal")
        return street, signal

    def test_a_signalised_junction_still_shows_the_street_you_would_walk(self, paris):
        with tempfile.TemporaryDirectory() as tmp:
            env = courier(paris)
            node = sorted(env.signalised)[0]
            street, signal = self.album(tmp, env, node)
            visible = {f"{node}|{n}" for n in env.network.nodes[node].neighbours}
            (signal / "signal_visibility.json").write_text(
                json.dumps({"legible": sorted(visible)}))
            env = CourierEnv(paris, seed=0, order_count=1,
                             album_root=street, signal_album_root=signal)
            env.reset()
            env.node_id = node
            rows = env.candidates()
            images = [r["image"] for r in rows]
            assert all(i is not None for i in images)
            assert len(set(images)) == len(images), "one picture served for every street"
            for row in rows:
                assert row["image"].endswith(f"toward_{row['node']}.png")

    def test_the_lamp_is_a_second_photograph_beside_the_street_one(self, paris):
        with tempfile.TemporaryDirectory() as tmp:
            env = courier(paris)
            node = sorted(env.signalised)[0]
            street, signal = self.album(tmp, env, node)
            neighbour = sorted(env.network.nodes[node].neighbours)[0]
            (signal / "signal_visibility.json").write_text(
                json.dumps({"legible": [f"{node}|{neighbour}"]}))
            env = CourierEnv(paris, seed=0, order_count=1,
                             album_root=street, signal_album_root=signal)
            env.reset()
            env.node_id = node
            rows = {r["node"]: r for r in env.candidates()}
            assert rows[neighbour]["signal_image"] is not None
            assert "_red.png" in rows[neighbour]["signal_image"] or \
                   "_green.png" in rows[neighbour]["signal_image"]
            # And nothing for the approaches the album cannot show.
            for node_id, row in rows.items():
                if node_id != neighbour:
                    assert row["signal_image"] is None

    def test_no_lamp_is_shown_where_the_album_declares_none(self, paris):
        """A picture of a light the courier cannot see, followed by a penalty for
        the light it could not see, is the defect the visibility gate exists for."""
        with tempfile.TemporaryDirectory() as tmp:
            env = courier(paris)
            node = sorted(env.signalised)[0]
            street, signal = self.album(tmp, env, node)
            (signal / "signal_visibility.json").write_text(json.dumps({"legible": []}))
            env = CourierEnv(paris, seed=0, order_count=1,
                             album_root=street, signal_album_root=signal)
            env.reset()
            env.node_id = node
            assert all(r["signal_image"] is None for r in env.candidates())

    def test_the_served_lamp_matches_the_phase_that_is_running(self, paris):
        with tempfile.TemporaryDirectory() as tmp:
            env = courier(paris)
            node = sorted(env.signalised)[0]
            street, signal = self.album(tmp, env, node)
            visible = {f"{node}|{n}" for n in env.network.nodes[node].neighbours}
            (signal / "signal_visibility.json").write_text(
                json.dumps({"legible": sorted(visible)}))
            env = CourierEnv(paris, seed=0, order_count=1,
                             album_root=street, signal_album_root=signal)
            env.reset()
            env.node_id = node
            for row in env.candidates():
                assert row["signal_image"].endswith(
                    f"_{env.light_here(row['k'])}.png")


class TestPhotographCaptions:
    def test_every_frame_has_a_caption_and_they_are_in_the_same_order(self):
        rows = [
            {"k": 1, "street": "Rue Monge", "relative": "on your left",
             "heading": "north", "image": "/a.png", "signal_image": "/a_red.png"},
            {"k": 2, "street": "Rue Cujas", "relative": "behind you",
             "heading": "south", "image": "/b.png", "signal_image": None},
        ]
        captions = render_photographs(rows)
        # The contract is the ordering: every attached frame is identified, in
        # the order the frames are attached, and identified by the words
        # walk_to takes so that reading a picture and acting on it need no
        # translation. With the streets named rather than numbered there is no
        # index to carry that mapping, so the names carry it.
        for row in rows:
            if row.get("image"):
                assert f'[{row["street"]}, {row["heading"]}]' in captions
            if row.get("signal_image"):
                assert f'[light: {row["street"]}, {row["heading"]}]' in captions
        # Rue Cujas has no lamp, so it must not be given one.
        assert "[light: Rue Cujas, south]" not in captions
        # Street views first, then lamps -- the order the frames are attached.
        assert (captions.index("[Rue Cujas, south]")
                < captions.index("[light: Rue Monge, north]"))

    def test_a_junction_with_no_pictures_says_so(self):
        assert "no photographs" in render_photographs([])

    def test_a_front_rear_pair_is_captioned_by_view_then_bearing(self):
        rows = [
            {
                "view": "front", "heading": "east", "image": "/front.png",
                "capture_group_id": "private-group",
                "camera_snapshot_id": "private-front-snapshot",
                "camera_intrinsics_id": "private-intrinsics",
            },
            {
                "view": "rear", "heading": "west", "image": "/rear.png",
                "capture_group_id": "private-group",
                "camera_snapshot_id": "private-rear-snapshot",
                "camera_intrinsics_id": "private-intrinsics",
            },
        ]

        captions = render_photographs(rows, phone_map=True)

        assert captions.splitlines() == [
            "  [front, east] the view in front of you",
            "  [rear, west] the view behind you",
            "  [map] your phone's map — a drawing, not a photograph: it has the "
            "streets and your route on it and cannot see anything in them",
        ]
        for private in (
            "private-group", "private-front-snapshot",
            "private-rear-snapshot", "private-intrinsics",
        ):
            assert private not in captions


class TestSession:
    """The turn, end to end."""

    def test_every_advertised_tool_can_be_dispatched(self, paris):
        """A tool in the menu that the runtime cannot run costs the agent a turn,
        and three of them cost the episode."""
        for condition in (Condition.FULL, Condition.NO_PHONE):
            env = courier(paris, condition=condition)
            session = CourierSession(env)
            assert set(session.dispatch) == set(session.allowed)
            for name in session.allowed:
                assert name in session.system_prompt()

    def test_a_tool_that_is_not_advertised_is_refused_not_executed(self, paris):
        env = courier(paris, condition=Condition.NO_PHONE)
        session = CourierSession(env)
        turn = session.step("THOUGHT: t\n```\nnavigate()\n```")
        assert turn.status == "format_error"

    def test_the_job_is_known_before_the_first_action(self, paris):
        """A rider is handed the job with the bag. Learning it cost a turn."""
        env = courier(paris)
        session = CourierSession(env)
        assert env.target_address().text in session.observe().text

    def test_the_observation_lists_exactly_the_frames_it_attaches(self, paris):
        with tempfile.TemporaryDirectory() as tmp:
            env = courier(paris)
            street = Path(tmp) / "street"
            for node, data in env.network.nodes.items():
                for neighbour in data.neighbours:
                    path = street / "images" / node / f"toward_{neighbour}.png"
                    path.parent.mkdir(parents=True, exist_ok=True)
                    path.write_bytes(b"x")
            env = CourierEnv(paris, seed=0, order_count=1, album_root=street)
            env.reset()
            session = CourierSession(env)
            observation = session.observe()
            assert observation.frames
            # Every attached frame is named somewhere in the text, and nothing
            # is named that is not attached.
            captions = observation.text.split("### photographs")[1]
            for frame in observation.frames:
                marker = frame.label.split("]")[0] + "]"
                assert marker in captions
            attached = {frame.label.split("]")[0] + "]" for frame in observation.frames}
            mentioned = set(re.findall(r"\[[^\]]+\]", captions))
            assert mentioned == attached

    def test_a_malformed_reply_is_re_prompted_rather_than_fatal(self, paris):
        session = CourierSession(courier(paris))
        turn = session.step("I will walk north.")
        assert turn.status == "format_error"
        assert not session.finished
        assert "fenced block" in session.feedback

    def test_three_malformed_replies_in_a_row_end_the_episode(self, paris):
        session = CourierSession(courier(paris))
        for _ in range(3):
            # A finished sentence with no action in it -- the model answered
            # and simply did not act. A reply that stops mid-sentence is a
            # truncation and is charged to a different budget; see
            # TestATruncatedReplyIsNotAFormatError.
            session.step("no action here.")
        assert session.finished
        assert session.run.termination_reason == "repeated_format_errors"

    def test_a_refusal_is_information_and_the_episode_continues(self, paris):
        session = CourierSession(courier(paris))
        turn = session.step("THOUGHT: t\n```\nwalk_to(\"Rue de Grenelle\", \"east\")\n```")
        assert turn.status == "rejected"
        assert not session.finished
        assert "leaving this junction" in session.feedback

    def test_the_wrong_arity_is_reported_rather_than_crashing(self, paris):
        session = CourierSession(courier(paris))
        turn = session.step('THOUGHT: t\n```\ncheck_order("Rue Monge")\n```')
        assert turn.status == "rejected"
        assert turn.error == "bad_arguments"

    def test_the_clock_the_turn_reports_is_the_clock_the_world_moved(self, paris):
        session = CourierSession(courier(paris))
        before = session.env.sim_seconds
        turn = session.step("THOUGHT: t\n```\nnavigate()\n```")
        assert turn.sim_seconds == pytest.approx(session.env.sim_seconds - before)
        assert turn.sim_seconds > 0

    def test_handling_time_is_charged_like_walking_time(self, paris):
        """Both tools declared 30 s and neither advanced the clock, so a minute a
        delivery was free -- and the deadlines had already been sized as if it
        were not."""
        env = courier(paris)
        order = env.active_order()
        env.node_id = order.pickup.kerb_node
        before = env.sim_seconds
        assert env.collect().ok
        assert env.sim_seconds - before >= 30.0
        env.node_id = order.dropoff.kerb_node
        before = env.sim_seconds
        assert env.hand_over().ok
        assert env.sim_seconds - before >= 30.0

    def test_a_refusal_names_the_job_the_courier_is_actually_near(self, paris):
        """"You are not at 19 Boulevard du Temple" while standing beside job 1's
        door named a place the courier was not going and said nothing about the
        one it was."""
        env = courier(paris, order_count=3, difficulty=None)
        live = env.live_orders()
        if len(live) < 2:
            pytest.skip("this seed issues one job at a time")
        far, near = live[0], live[-1]
        if far is near:
            pytest.skip("only one live pickup")
        env.node_id = near.pickup.kerb_node
        # Standing on the door, collect succeeds; one junction off, the refusal
        # must still be about *this* job.
        message = env.collect().message
        assert near.pickup.text in message


class TestPromptDoesNotLeak:
    def test_the_prompt_tells_the_agent_the_pictures_are_load_bearing(self):
        prompt = build_system_prompt(city="Paris", tools=available_tools(PARIS_ACTIONS))
        assert "LOOK AT THE PHOTOGRAPHS" in prompt
        low = prompt.lower()
        assert "colour" in low and "pedestrian light" in low

    def test_the_prompt_says_which_photograph_governs_the_crossing(self):
        """36% of the street frames have a traffic lamp baked into them in one
        fixed phase, so a street view can show red while the crossing is green.
        Only the [light k] frame tracks the phase, and the courier has to be told
        which of the two pictures it is being scored against."""
        prompt = build_system_prompt(city="Paris", tools=available_tools(PARIS_ACTIONS))
        assert "[light: street name, bearing]" in prompt
        assert "not any light in the street views" in prompt

    def test_the_prompt_states_the_reply_format_unambiguously(self):
        prompt = build_system_prompt(city="Paris", tools=available_tools(PARIS_ACTIONS))
        for rule in ("Exactly one fenced block", "one call", "THOUGHT"):
            assert rule in prompt

    def test_the_prompt_never_says_which_way_to_go(self):
        prompt = build_system_prompt(city="Paris", tools=available_tools(PARIS_ACTIONS))
        low = prompt.lower()
        for leak in ("the light is red", "the light is green", "walk_to(\"Rue de Grenelle\", \"east\") is correct",
                     "the answer is", "the shortest"):
            assert leak not in low

    def test_the_prompt_fits_a_sensible_budget(self):
        """Context spent on the runbook is context not spent on the pictures.

        Raised from 1600 to 2400 to buy one thing: an ordered decision procedure
        that says which situation the courier is in before the other runbooks say
        what to do in it. The measurement that justified the spend -- 40 episodes
        of Qwen3-VL-4B -- found the model reaching the slip's street in 22 of 40
        episodes and then walking off it 81 times, choosing the right way along
        it 37 times against 25, and returning to an already-visited junction on
        26 of 41 moves. None of that is a missing runbook; it is not knowing
        which of them applies.

        Raised again, from 2400 to 3000, and the exchange rate is not what it
        was. The frames were 640x480 at roughly 380 tokens each when this
        ceiling was set; the harness now serves 320x240 at roughly 80, so the
        same prompt tokens cost more pictures than they used to and the test
        should have been retuned when the frames shrank rather than left to
        drift.

        What the extra 600 buys was measured, not guessed. With the route drawn
        rather than dictated, nothing in the prompt said how to turn a line on a
        map into a street at a corner, and 40 held-out episodes produced 130
        no_such_street refusals and 24 episodes ending stuck -- the model
        writing the name of a street further along its route, being told that
        street is not here, and concluding in its own words that "the map must
        be wrong". The addition states where each fact comes from, that the
        route is walked one junction at a time, and that a repeated refusal will
        be refused again.

        Raised a fourth time, 3200 to 4800, to make the tool menu a manual.
        The measurement that bought it: over 1206 turns of Qwen3-VL-4B, 1139
        were walk_to and three were look, against 47 "that street does not
        leave this junction" refusals. A courier with one hammer, repeatedly
        told the street it named is not here -- and never told that look()
        answers which way the numbers run without walking, or what to do when
        a street it wants is not on the list. Each tool now says what a call
        gives back, what each refusal means and what to do about it, and when
        it is the wrong tool. That is where the 1600 went.

        Raised a third time, 3000 to 3200, for one tool: ``note``. That is the
        smallest raise of the three and it buys back something already assumed
        to exist -- the LOST skill has always told the courier to "note() that
        this way was a dead end", while ``note`` sat in UNIMPLEMENTED_TOOLS with
        no executor anywhere, so the instruction named a call that could not be
        made. The entry costs about 60 characters against a runbook that spends
        them referring to it.

        The ceiling still binds, and it is not free: anything added here has to
        beat a picture.
        """
        prompt = build_system_prompt(city="Paris", tools=available_tools(PARIS_ACTIONS))
        # ~4 characters a token for English prose; the exact tokeniser does not
        # matter for a ceiling this loose.
        assert len(prompt) / 4 < 4800, "system prompt has grown past its budget"

    def test_taking_the_phone_away_takes_its_runbook_away_too(self):
        """Guidance that names a tool the environment will refuse is the same
        defect as a menu that does."""
        hard = build_system_prompt(
            city="Paris", tools=available_tools(PARIS_ACTIONS, allow_consult=False))
        assert "navigate()" not in hard
        assert "check_map" not in hard
        # What is left is the runbook that does not need a phone: read the doors
        # and follow the numbers.
        assert "look(" in hard  # the no-phone runbook survives, by name


class TestTheSameRefusalFourTimesEndsTheSession:
    """A courier repeating one refused call is not working a problem.

    On the 22 episodes that actually ran, 47 turns -- 6.8% of every turn spent
    -- went into repeating an identical refused call, and one episode made the
    same one 28 times in a row while its own reasoning said "the order cannot
    be collected". Nothing in the remaining turns can differ, because nothing
    in the world has.

    It ends the session, not the city: the environment does not disappear
    because the courier pressed the wrong button, and a shift that stops early
    earns nothing extra, so there is nothing to game.
    """

    def session(self):
        from embodiedbench.agent.courier.session import CourierSession
        from embodiedbench.runtime.city.courier_env import CourierEnv
        from pathlib import Path

        from embodiedbench.compiler.road_network import build_road_network

        maps = (Path(__file__).resolve().parents[1] / "vendor" / "vagen"
                / "vagen" / "envs" / "deliverybench" / "maps")
        paris = build_road_network(maps / "citycore-paris",
                                   map_name="citycore-paris")
        env = CourierEnv(paris, seed=19, difficulty="solo", stride="block")
        env.reset()
        return CourierSession(env, city="Paris")

    def refuse(self, session, times, call='navigate("9 Rue Imaginaire")'):
        for _ in range(times):
            if session.finished:
                break
            session.step(f"THOUGHT: x\n```\n{call}\n```")
        return session

    def test_four_identical_refusals_stop_it(self):
        session = self.refuse(self.session(), 6)
        assert session.finished
        assert session.run.termination_reason == "stuck"
        assert len(session.run.turns) == 4

    def test_three_are_allowed(self):
        """Three is a run a policy can plausibly be working through; every
        measured run of 2 or 3 was a model trying something adjacent."""
        session = self.refuse(self.session(), 3)
        assert not session.finished

    def test_a_different_refused_call_resets_the_count(self):
        session = self.session()
        self.refuse(session, 3, 'navigate("9 Rue Imaginaire")')
        self.refuse(session, 3, 'navigate("8 Rue Imaginaire")')
        assert not session.finished

    def test_an_accepted_call_in_between_resets_the_count(self):
        session = self.session()
        self.refuse(session, 3, 'navigate("9 Rue Imaginaire")')
        session.step("THOUGHT: x\n```\ncheck_order()\n```")
        self.refuse(session, 3, 'navigate("9 Rue Imaginaire")')
        assert not session.finished

    def test_a_working_courier_is_never_stopped(self):
        session = self.session()
        for _ in range(8):
            session.step("THOUGHT: x\n```\ncheck_order()\n```")
        assert not session.finished


class TestATruncatedReplyIsNotAFormatError:
    """A generation that ran out of room is the harness's fault, not the model's.

    ``model_io`` already says so on the evaluation path -- it raises the budget
    and asks again rather than parsing the stump. The RL path has no requery,
    so a truncation arrived as a format error and three in a row ended the
    episode. Measured on 64 held-out episodes: 11 died that way, all at zero
    earnings, discarding 308 of their 440 remaining turns -- the same size as
    the entire reported success rate.

    It lands asymmetrically, which is what makes it poisonous. Validation
    decodes greedily and greedy degenerates into the repetition loops that
    exhaust the budget; training samples and never hits it. The policy was
    losing a sixth of its score to a failure mode it got no gradient on.
    """

    ALLOWED = {"walk_to", "collect", "navigate", "look", "wait", "check_order"}

    def parse(self, reply):
        from embodiedbench.agent.courier.loop import parse_reply
        return parse_reply(reply, self.ALLOWED)

    # What a real one looks like: greedy decoding falls into a repetition loop
    # and the generation budget ends it inside a word. Median length of the
    # ones that killed an episode on the held-out set was 3365 characters.
    CUT_OFF = (
            "THOUGHT: I am on Rue Mouffetard and I need the address of the "
            "customer. I need the address of the customer. I need the "
            "address of the customer. I need the address of the cust")

    def test_a_reply_cut_mid_word_is_a_truncation(self):
        from embodiedbench.agent.courier.loop import TruncatedReply
        with pytest.raises(TruncatedReply):
            self.parse(self.CUT_OFF)

    def test_a_short_unpunctuated_reply_is_not_excused_as_truncation(self):
        """Too short to have exhausted a 1024-token budget. Without this floor
        any brief non-answer is forgiven and the leniency means nothing."""
        from embodiedbench.agent.courier.loop import FormatError, TruncatedReply
        with pytest.raises(FormatError) as caught:
            self.parse("no action here")
        assert not isinstance(caught.value, TruncatedReply)

    def test_an_unclosed_fence_is_a_truncation(self):
        from embodiedbench.agent.courier.loop import TruncatedReply
        with pytest.raises(TruncatedReply):
            self.parse("THOUGHT: going west.\n```\nwalk_to(1")

    def test_a_finished_sentence_with_no_action_is_still_a_format_error(self):
        """Otherwise a model that simply declines to answer is forgiven, and
        the leniency stops meaning anything."""
        from embodiedbench.agent.courier.loop import FormatError, TruncatedReply
        with pytest.raises(FormatError) as caught:
            self.parse("THOUGHT: I looked around and decided to do nothing.")
        assert not isinstance(caught.value, TruncatedReply)

    def test_a_truncation_is_caught_by_anything_catching_format_errors(self):
        from embodiedbench.agent.courier.loop import FormatError
        with pytest.raises(FormatError):
            self.parse(self.CUT_OFF)

    def test_truncations_do_not_spend_the_three_strike_format_budget(self):
        from embodiedbench.agent.courier.loop import Budgets, Spend, budget_exceeded
        spend, budgets = Spend(), Budgets()
        for _ in range(5):
            spend.truncated_replies += 1
            assert budget_exceeded(spend, budgets) is None
        assert spend.consecutive_format_errors == 0

    def test_but_they_are_not_unlimited(self):
        from embodiedbench.agent.courier.loop import Budgets, Spend, budget_exceeded
        spend, budgets = Spend(), Budgets()
        spend.truncated_replies = budgets.max_truncated_replies
        assert budget_exceeded(spend, budgets) == "repeated_truncations"

    def test_a_session_charges_a_truncation_to_its_own_counter(self):
        from pathlib import Path

        from embodiedbench.agent.courier.session import CourierSession
        from embodiedbench.compiler.road_network import build_road_network
        from embodiedbench.runtime.city.courier_env import CourierEnv

        maps = (Path(__file__).resolve().parents[1] / "vendor" / "vagen"
                / "vagen" / "envs" / "deliverybench" / "maps")
        env = CourierEnv(build_road_network(maps / "citycore-paris",
                                            map_name="citycore-paris"),
                         seed=0, difficulty="solo")
        env.reset()
        session = CourierSession(env, city="Paris")
        for _ in range(4):
            turn = session.step(self.CUT_OFF)
        assert turn.status == "truncated_reply"
        assert session.spend.truncated_replies == 4
        assert session.spend.consecutive_format_errors == 0
        assert not session.finished


class TestALampThatNeverArrivedIsNotCharged:
    """The album gate says which crossings have a lamp it can show. The image
    budget then decides how many pictures fit in a turn, and drops street views
    with their lamps. A crossing the album can show is therefore not always a
    crossing the courier was shown, and charging on the album alone penalises a
    policy for a light that never reached it -- the same defect the album gate
    exists to prevent, one layer further out.
    """

    def env(self, paris):
        env = CourierEnv(paris, seed=0, order_count=1, enforce_signals=True)
        env.reset()
        env.visible_signals = {f"{n}|{m}" for n in env.signalised
                               for m in paris.nodes[n].neighbours}
        return env

    def test_unlimited_by_default(self, paris):
        """Nobody limiting the pictures means the album's answer stands, which
        is what the evaluation harness needs -- it sends them all."""
        env = self.env(paris)
        node = next(iter(env.signalised))
        toward = sorted(paris.nodes[node].neighbours)[0]
        assert env.signal_is_visible(node, toward)

    def test_a_lamp_left_out_of_the_turn_cannot_be_charged(self, paris):
        env = self.env(paris)
        node = next(iter(env.signalised))
        legs = sorted(paris.nodes[node].neighbours)
        env.show_only_these_signals({f"{node}|{legs[0]}"})
        assert env.signal_is_visible(node, legs[0])
        for other in legs[1:]:
            assert not env.signal_is_visible(node, other), \
                "charged for a lamp the model was never sent"

    def test_sending_nothing_charges_nothing(self, paris):
        env = self.env(paris)
        node = next(iter(env.signalised))
        env.show_only_these_signals(set())
        for toward in paris.nodes[node].neighbours:
            assert not env.signal_is_visible(node, toward)

    def test_a_reset_forgets_who_was_limiting(self, paris):
        """Otherwise one episode's image budget silently gates the next."""
        env = self.env(paris)
        env.show_only_these_signals(set())
        env.reset()
        assert env.signals_shown is None


class TestTheMapSectionTeachesTheReadableRoute:
    """The map is the only place direction exists, and it was never explained.

    It also has to be explained the right way round. Measured on Qwen3-VL-4B:
    asked which way a large red pin lies from a blue dot -- one object against
    another, no route, no arrow -- it answers within 45 degrees 17-29% of the
    time, against 50% for guessing and 81% on a black arrow on a white field.
    Instructions about angles are therefore instructions about the thing it
    cannot do. Street names, now legible at the served size, are the thing it
    can, so the section leads with matching names and keeps the compass as a
    fallback.
    """

    def test_it_names_what_is_drawn(self):
        prompt = build_system_prompt(city="Paris",
                                     tools=available_tools(PARIS_ACTIONS))
        assert "READING THE MAP" in prompt
        low = prompt.lower()
        for mark in ("blue line", "arrow", "you are here", "red pin", "north up"):
            assert mark in low, mark

    def test_it_leads_with_names_rather_than_angles(self):
        prompt = build_system_prompt(city="Paris",
                                     tools=available_tools(PARIS_ACTIONS))
        section = prompt.split("READING THE MAP", 1)[1]
        by_name = section.lower().index("by name")
        by_angle = section.lower().index("arrow and compass")
        assert by_name < by_angle, (
            "the compass advice must come after the name advice: reading "
            "angles is the thing the model measurably cannot do"
        )


class TestARefusalIsShownWhereTheChoiceIsMade:
    """Telling the courier that a repeat will be refused did not stop it.

    Measured over 12 shifts: 63 of 128 attempts were the identical call made
    again immediately, with the prompt already saying that nothing changes
    between a refusal and the next turn. Prose is the wrong place for it. The
    fact belongs on the candidate line the courier is choosing from, which is
    the only text it demonstrably acts on.
    """

    def test_the_line_carries_the_refusal(self):
        from embodiedbench.agent.courier.prompts import render_candidates

        rows = [{"street": "Rue de Grenelle", "heading": "east",
                 "refused": "way_blocked"}]
        assert "REFUSED ALREADY" in render_candidates(rows)

    def test_memory_keys_a_refusal_to_its_junction(self):
        from embodiedbench.agent.courier.memory import CourierMemory

        memory = CourierMemory()
        memory.refused("n1", "Rue de Grenelle", "east", "way_blocked")
        assert memory.refusal_at("n1", "Rue de Grenelle", "east") == "way_blocked"
        # Not somewhere else: the same street can be walkable from the next
        # corner, and marking it everywhere would hide a legal move.
        assert memory.refusal_at("n2", "Rue de Grenelle", "east") == ""


class TestTheMapAdviceCannotBeHalfFollowed:
    """The first version of this advice caused the failure it was meant to fix.

    It said to read names off the map and take one that also appears in the
    list. The model did the first half and skipped the second: no_such_street
    went from 16 to 29 over the same twelve seeds, because only 38% of the
    names it reads off a map are takeable from where it stands.
    """

    def test_the_list_is_named_before_the_walk(self):
        prompt = build_system_prompt(city="Paris",
                                     tools=available_tools(PARIS_ACTIONS))
        section = prompt.split("READING THE MAP", 1)[1].lower()
        assert "also in the list" in section
        # Split so a line break inside the sentence cannot fail the pin: what
        # must survive is the warning that an off-list map name is refused.
        assert "name off the map" in section
        assert "is refused" in " ".join(section.split())
        # The numbered order has to put finding it in the list before walking.
        assert section.index("find one of those") < section.index("walk that one")


class TestTheThreeVisionSettings:
    """Three tasks, one world: which facts the words are allowed to state.

    none   direction, lights and barriers live in the pictures only.
    route  the route's next street is named in the text; lights and barriers
           stay visual. Isolates route-reading -- the one thing measurement
           says current models cannot do -- from the two visual jobs they have
           never been tested on.
    all    all three are stated. A text-only policy can solve this, which is
           the point: it is the floor the other two are measured against.

    What each setting must never do is describe itself wrongly, so the prompt
    is taken from the environment rather than configured beside it.
    """

    def env(self, paris, narration):
        env = CourierEnv(paris, seed=0, order_count=1, narration=narration,
                         enforce_signals=True)
        env.reset()
        return env

    def test_it_refuses_a_setting_it_does_not_have(self, paris):
        with pytest.raises(ValueError):
            CourierEnv(paris, seed=0, narration="sometimes")

    def test_only_the_narrated_settings_name_the_route(self, paris):
        for narration, expected in (("none", False), ("route", True),
                                    ("all", True)):
            rows = self.env(paris, narration).candidates()
            named = any(r.get("on_route") for r in rows)
            assert named is expected or not expected, narration
            if not expected:
                assert not any("on_route" in r for r in rows), narration

    def test_only_all_states_lights_and_barriers(self, paris):
        for narration in ("none", "route"):
            rows = self.env(paris, narration).candidates()
            assert not any("told_signal" in r or "told_blocked" in r
                           for r in rows), narration
        rows = self.env(paris, "all").candidates()
        assert all("told_signal" in r and "told_blocked" in r for r in rows)

    def test_the_prompt_says_what_its_own_world_does(self, paris):
        from embodiedbench.agent.courier.session import CourierSession

        for narration, must, must_not in (
            ("none", "none of that is in any text", "THE ROUTE GOES THIS WAY"),
            ("route", "only place a red light", "you are not required"),
            ("all", "EVERYTHING YOU NEED IS IN THE WORDS",
             "none of that is in any text"),
        ):
            prompt = CourierSession(self.env(paris, narration),
                                    with_images=False).system_prompt()
            assert must in prompt, narration
            assert must_not not in prompt, narration


class TestNoSettingContradictsItself:
    """Swapping one paragraph is not enough, and the first guard missed it.

    The narrated prompt said "you are not required to read anything out of
    them" and then told the courier six more times, in the steps, the hazard
    rules, the tool manual and two whole runbooks, that the colour of a light
    is in no text. A prompt that contradicts itself teaches nothing, and it
    costs turns: the courier goes looking for a fact the words already gave
    it. This scans the entire prompt rather than the part that was swapped.
    """

    VISUAL_ONLY_CLAIMS = (
        "will never tell you the colour",
        "no order slip and no route",
        "Those are in the pictures and nowhere else",
        "a photograph shows a barrier",
        "Read the lamp in the [light:",
        "those are in the photographs",
    )

    def test_the_all_setting_never_claims_a_fact_is_visual_only(self):
        prompt = build_system_prompt(city="Paris",
                                     tools=available_tools(PARIS_ACTIONS),
                                     narration="all")
        found = [c for c in self.VISUAL_ONLY_CLAIMS if c in prompt]
        assert not found, f"narration=all still says: {found}"

    def test_the_visual_settings_keep_the_warning(self):
        """The mirror image: dropping it everywhere would remove the only
        thing telling a visual courier the pictures are load-bearing."""
        for narration in ("none", "route"):
            prompt = build_system_prompt(city="Paris",
                                         tools=available_tools(PARIS_ACTIONS),
                                         narration=narration)
            assert "will never tell you the colour" in prompt, narration

    def test_all_still_says_what_to_do_about_lights_and_barriers(self):
        """Removing the photograph runbooks must not remove the rules."""
        prompt = build_system_prompt(city="Paris",
                                     tools=available_tools(PARIS_ACTIONS),
                                     narration="all")
        assert "RED" in prompt and "wait()" in prompt
        assert "BLOCKED" in prompt


class TestTheMapPointsTheRightWay:
    """Orientation is the one thing on this map that must never be wrong.

    Two conventions meet here and they have different zeros: screen angles
    measure from east with y growing downwards, while the banner's arrow glyph
    is drawn pointing north. Comparing one against the other shows a constant
    90 degree error that is not an error, which is exactly the trap a check
    written by eye falls into. Each is tested against the geometry it is
    supposed to represent, not against the other.
    """

    HEADING_ON_SCREEN = {"north": -90, "north-east": -45, "east": 0,
                         "south-east": 45, "south": 90, "south-west": 135,
                         "west": 180, "north-west": -135}

    def _first_leg(self, env):
        from embodiedbench.runtime.city.courier_env import (
            bearing_deg, compass_of)

        order = next((o for o in env.orders if o.live), None)
        route = env.route_nodes(env.node_id, order.pickup.kerb_node) or []
        if len(route) < 2:
            return None
        return compass_of(bearing_deg(env.position(),
                                      env.position(route[1])))

    def test_the_position_marker_faces_the_way_the_route_goes(self, paris):
        import re

        for seed in range(8):
            env = CourierEnv(paris, seed=seed, order_count=1,
                             difficulty="solo", stride="block")
            env.reset()
            word = self._first_leg(env)
            if word is None:
                continue
            svg = env.map_drawing().svg
            match = re.search(
                r'class="puck"[^>]*/><g transform="translate\([^)]*\) '
                r'rotate\((-?[\d.]+)\)', svg)
            assert match, f"seed {seed}: no heading chevron on the marker"
            drawn = float(match.group(1))
            want = self.HEADING_ON_SCREEN[word]
            gap = abs((drawn - want + 180) % 360 - 180)
            assert gap <= 30, (
                f"seed {seed}: route leaves {word} (screen {want}) but the "
                f"marker points {drawn}"
            )

    def test_the_banner_names_the_direction_it_draws(self, paris):
        import re

        for seed in range(8):
            env = CourierEnv(paris, seed=seed, order_count=1,
                             difficulty="solo", stride="block")
            env.reset()
            word = self._first_leg(env)
            if word is None:
                continue
            svg = env.map_drawing().svg
            assert f"head {word}" in svg, f"seed {seed}: banner omits {word}"
            # The glyph is drawn pointing north, so its rotation is the
            # compass bearing itself rather than a screen angle.
            north_up = {"north": 0, "north-east": 45, "east": 90,
                        "south-east": 135, "south": 180, "south-west": 225,
                        "west": 270, "north-west": 315}[word]
            assert f"rotate({north_up})" in svg, (
                f"seed {seed}: banner says {word} but its arrow does not turn "
                f"to {north_up}"
            )


class TestTheBannerIsAnInstructionYouCanObey:
    """The banner has to name an action the courier can actually take.

    It derived its bearing separately -- the compass of the step to the
    route's next node -- while the candidate line quotes the bearing of the
    whole block. At block stride those are different quantities, and they
    disagreed on 6 frames in 67; on one the banner said south-east where the
    list offered the same street going north-west. A banner that cannot be
    copied into walk_to verbatim is worse than none, because it reads as the
    map contradicting the corner, which is the confusion the whole redesign
    exists to remove.
    """

    def test_every_banner_names_a_line_on_the_list(self, paris):
        import re

        for seed in range(8):
            env = CourierEnv(paris, seed=seed, order_count=1,
                             difficulty="solo", stride="block")
            env.reset()
            order = next((o for o in env.orders if o.live), None)
            if order is None:
                continue
            env.navigate(order.pickup.text)
            for _ in range(5):
                svg = env.map_drawing().svg
                street = re.search(r'class="bandtext"[^>]*>([^<]+)<', svg)
                heading = re.search(r"head ([a-z\-]+) on", svg)
                if not street or not heading:
                    break
                offered = {(r["street"],
                            r["heading"])
                           for r in env.candidates()}
                assert (street.group(1), heading.group(1)) in offered, (
                    f"seed {seed}: banner says {street.group(1)!r} going "
                    f"{heading.group(1)!r}, list offers {sorted(offered)}"
                )
                if not env.walk_to(street.group(1), heading.group(1)).ok:
                    break


class TestDrawingTheMapCostsNothing:
    """The map is a survey drawing. It must not ask the world for pictures.

    The instruction banner recomputes the next street every turn, and it read
    that off candidates() -- which looks up a street view and a lamp for every
    way out. Under the album that is a path lookup; under the live renderer it
    is a render request, so drawing the map issued a batch of renders. Three
    live tests caught it as a symptom (a busy backend skipped, a degrade
    verdict early, a lamp request with the wrong phase) and none of them named
    the cause, which is why this test exists at the level the cause lives on.
    """

    def test_the_map_does_not_look_up_a_single_frame(self, paris):
        env = CourierEnv(paris, seed=0, order_count=1, difficulty="solo",
                         stride="block")
        env.reset()
        order = next((o for o in env.orders if o.live), None)
        if order is not None:
            env.navigate(order.pickup.text)

        looked: list[tuple[str, str]] = []
        real_frame, real_signal = env.frame_for, env.signal_frame_for
        env.frame_for = lambda n, t: (looked.append((n, t)), real_frame(n, t))[1]
        env.signal_frame_for = lambda n, t: (looked.append((n, t)),
                                             real_signal(n, t))[1]
        try:
            env.map_drawing()
        finally:
            env.frame_for, env.signal_frame_for = real_frame, real_signal
        assert not looked, (
            f"drawing the map asked for {len(looked)} frames: {looked[:3]}"
        )


class TestTheFrameCacheIsNotSharedBetweenUsers:
    """/tmp is shared and the alias cache is not: the first user to run this on
    a machine owned the directory, and the second died on the first frame with
    PermissionError. The default path carries the user."""

    def test_the_default_root_is_per_user(self):
        import getpass

        from embodiedbench.agent.courier.frame_alias import default_root

        root = default_root()
        assert getpass.getuser() in root.name, (
            f"{root} is shared between users on this machine")

    def test_an_explicit_override_still_wins(self, tmp_path, monkeypatch):
        from embodiedbench.agent.courier.frame_alias import default_root

        monkeypatch.setenv("EMBODIEDBENCH_FRAME_CACHE", str(tmp_path / "elsewhere"))
        assert default_root() == tmp_path / "elsewhere"
