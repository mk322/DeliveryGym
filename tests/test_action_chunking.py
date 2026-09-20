"""Action chunking: up to K calls in one turn, and what that must not change.

Chunking exists for one measured reason. In the embodied condition UE owns
locomotion, so a turn is 25-105 s of engine time for the walk against a few
seconds of generation -- the round trip and its ~5k-token multimodal prefill
are the schedule, not the inference. A chunk pays for one prompt and buys K
actions, which divides both by K.

Everything worth defending here is a way that saving could quietly become a
different benchmark instead:

* **K=1 must be the thing it always was.** Same prompt bytes, same TurnLog
  shape, same dispatch. A chunking switch that changes the unchunked run makes
  every number measured before it incomparable with every number after it.
* **A chunk saves turns and nothing else.** Three hops in one reply cost three
  steps, three tool calls and the sum of their seconds; only the *turn* count
  is one. Otherwise a policy discovers that the cheapest route through the city
  is a long reply.
* **Execution stops at the first refusal**, and the calls after it are dropped
  and reported rather than run. The courier plans blind past its own mistake --
  that is the risk the prompt tells it about, and it has to be real.
* **Over the limit is refused, not trimmed.** Truncating a five-call plan to
  three runs a prefix the model never chose, which is the exact "execute
  something it did not decide" failure the parser exists to prevent.
"""

from __future__ import annotations

import asyncio
from pathlib import Path

import pytest

from embodiedbench.agent.courier.chunk import (
    ChunkedCourierSession,
    ChunkTooLong,
    ChunkTurnLog,
    chunk_info,
    parse_chunk,
)
from embodiedbench.agent.courier.loop import Budgets, FormatError, TurnLog
from embodiedbench.agent.courier.session import CourierSession
from embodiedbench.compiler.road_network import build_road_network
from embodiedbench.runtime.city.courier_env import (
    REJECTED_ACTION_SECONDS,
    CourierEnv,
)
from embodiedbench.runtime.live.client import UERenderClient  # noqa: F401
from embodiedbench.runtime.live.gym_adapter import EmbodiedCourierGymEnv

from live_stub import FakeTrackBService, write_endpoints

MAPS = (Path(__file__).resolve().parents[1] / "vendor" / "vagen" / "vagen"
        / "envs" / "deliverybench" / "maps")
PARIS = MAPS / "citycore-paris"
needs_maps = pytest.mark.skipif(not PARIS.exists(),
                                reason="vendored maps not present")


@pytest.fixture(scope="module")
def paris():
    return build_road_network(PARIS, map_name="citycore-paris")


def courier(paris, **kwargs) -> CourierEnv:
    kwargs.setdefault("seed", 0)
    kwargs.setdefault("order_count", 1)
    env = CourierEnv(paris, **kwargs)
    env.reset()
    return env


def fence(*calls: str, thought: str = "t") -> str:
    body = "\n".join(calls)
    return f"THOUGHT: {thought}\n```\n{body}\n```"


def walk_chain(paris, count: int, **kwargs):
    """``count`` walk calls that really chain, and the nodes they land on.

    Found by walking a twin of the environment under test rather than by
    reading the map: two ``CourierEnv(seed=0)`` are the same city down to the
    hazards, so a call the twin accepts is a call the env under test accepts,
    and the twin's clock is the clock the chunk should end up with. Each hop
    is chosen to a junction the chain has not been to, so "three hops" and
    "three different junctions" are the same claim.

    Returns the calls, the node trail (starting node first) and the twin,
    whose ``sim_seconds`` is the reference the chunk's summed clock is
    compared against.
    """
    probe = courier(paris, **kwargs)
    calls, nodes = [], [probe.node_id]
    for _ in range(count):
        rows = probe.candidates()
        row = next((r for r in rows if r["node"] not in nodes), rows[0])
        calls.append(f'walk_to("{row["street"]}", "{row["heading"]}")')
        outcome = probe.walk_to(row["street"], row["heading"])
        assert outcome.ok, outcome.message
        nodes.append(probe.node_id)
    return calls, nodes, probe


# ─────────────────────────────────────────────────────────────────────────────


@needs_maps
class TestKOfOneIsTheStockTurn:
    """The switch has an off position, and off means untouched."""

    def test_the_prompt_at_k_of_one_is_the_stock_prompt_byte_for_byte(self, paris):
        """Every courier number ever measured was measured against these
        bytes. A chunking parameter that reworded the prompt at K=1 would make
        the comparison it exists to enable impossible."""
        stock = CourierSession(courier(paris)).system_prompt()
        chunked = ChunkedCourierSession(courier(paris),
                                        action_chunk=1).system_prompt()
        assert chunked == stock

    def test_the_same_tools_are_dispatched(self, paris):
        stock = CourierSession(courier(paris))
        chunked = ChunkedCourierSession(courier(paris), action_chunk=1)
        assert set(chunked.dispatch) == set(stock.dispatch)
        assert chunked.allowed == stock.allowed

    def test_the_turn_log_is_the_stock_record_field_for_field(self, paris):
        """Not merely 'compatible': the same class, the same keys and the same
        values, because at K=1 the chunked session runs the stock step rather
        than a re-implementation of it."""
        calls, _, _ = walk_chain(paris, 1)
        stock = CourierSession(courier(paris)).step(fence(*calls))
        chunked = ChunkedCourierSession(courier(paris),
                                        action_chunk=1).step(fence(*calls))
        assert type(chunked) is TurnLog
        assert chunked.to_dict() == stock.to_dict()

    def test_the_parser_at_one_call_is_the_stock_parser(self, paris):
        """Including its refusals: two calls in a block is the stock format
        error, not a chunk of two silently permitted."""
        allowed = {"walk_to", "look"}
        one = parse_chunk(fence('walk_to("Rue Monge", "north")'), allowed, 1)
        assert len(one) == 1 and one[0].tool == "walk_to"
        with pytest.raises(FormatError, match="Give exactly one"):
            parse_chunk(fence('walk_to("Rue Monge", "north")', "look()"),
                        allowed, 1)

    def test_a_stock_turn_reports_as_a_chunk_of_one(self, paris):
        """``chunk_info`` is what the adapter puts in ``info``, and it has to
        answer for an unchunked log too -- otherwise a log parser has to know
        which session ran before it can read the run."""
        turn = CourierSession(courier(paris)).step(fence("check_order()"))
        assert chunk_info(turn) == {"chunk_len": 1, "chunk_executed": 1}
        missed = CourierSession(courier(paris)).step("no action here.")
        assert missed.status == "format_error"
        assert chunk_info(missed) == {"chunk_len": 1, "chunk_executed": 0}
        assert chunk_info(None) == {"chunk_len": 0, "chunk_executed": 0}


@needs_maps
class TestAChunkIsAPlanAndOneStep:
    """A chunk is K waypoints estimated and ONE walked.

    It was built as "K actions for one prompt", which is what a throughput
    argument wants and the wrong contract for a plan: waypoints two and three
    are named relative to positions the courier has not reached, so running
    them commits to a forecast the first walk has already had a chance to
    falsify -- it stops a metre short, or against a wall, and the rest was
    computed from somewhere the courier is not.

    So the later calls are a plan. Naming three makes the model look a few
    steps ahead and puts that thinking in the trajectory; only the first
    happens, and the next turn re-plans from where the walk actually ended."""

    def test_three_named_one_walked(self, paris):
        calls, nodes, probe = walk_chain(paris, 3)
        env = courier(paris)
        session = ChunkedCourierSession(env, action_chunk=3)
        turn = session.step(fence(*calls))

        assert turn.status == "accepted"
        assert turn.chunk_len == 3, "three were named"
        assert turn.chunk_executed == 1, "one was walked"
        assert env.node_id == nodes[1], "it is one hop along, not three"

    def test_the_rest_of_the_plan_is_recorded_not_run(self, paris):
        calls, _, _ = walk_chain(paris, 3)
        session = ChunkedCourierSession(courier(paris), action_chunk=3)
        turn = session.step(fence(*calls))
        status = [row["status"] for row in turn.chunk]
        assert status == ["accepted", "planned", "planned"]
        # A plan never made and a plan made and superseded must not look the
        # same in the log -- the difference is the reason for asking for three.
        assert all(row["action"] for row in turn.chunk)

    def test_a_turn_costs_one_step_and_one_call(self, paris):
        calls, _, _ = walk_chain(paris, 3)
        session = ChunkedCourierSession(courier(paris), action_chunk=3)
        session.step(fence(*calls))
        assert len(session.run.turns) == 1
        assert session.spend.steps == 1
        assert session.spend.tool_calls == 1

    def test_the_clock_is_the_one_call_that_happened(self, paris):
        calls, _, _ = walk_chain(paris, 3)
        env = courier(paris)
        session = ChunkedCourierSession(env, action_chunk=3)
        turn = session.step(fence(*calls))
        walked = [r for r in turn.chunk if r["status"] == "accepted"]
        assert turn.sim_seconds == pytest.approx(
            sum(row["sim_seconds"] for row in walked))
        assert env.sim_seconds == pytest.approx(turn.sim_seconds)
        assert session.spend.sim_seconds == pytest.approx(turn.sim_seconds)

    def test_the_chunk_detail_rides_beside_the_stock_fields(self, paris):
        """New keys, not changed ones: everything a consumer read off a stock
        TurnLog is still there with its stock type."""
        calls, _, _ = walk_chain(paris, 2)
        session = ChunkedCourierSession(courier(paris), action_chunk=3)
        turn = session.step(fence(*calls))
        assert isinstance(turn, ChunkTurnLog) and isinstance(turn, TurnLog)
        record = turn.to_dict()
        stock = CourierSession(courier(paris)).step(fence(calls[0])).to_dict()
        assert set(stock) <= set(record)
        for key, value in stock.items():
            assert isinstance(record[key], type(value)), key
        assert set(record) - set(stock) == {
            "chunk", "chunk_len", "chunk_executed", "chunk_aborted_at"}
        # `from_xy` and `message` joined them so a trace can say what each
        # call was judged from and what the world said back to it -- a chunk's
        # later calls are named relative to positions the courier has not
        # reached yet, and the turn-level feedback is the calls' messages
        # joined, so neither is recoverable from the turn alone.
        assert [set(row) for row in turn.chunk] == [
            {"action", "status", "code", "sim_seconds", "reward",
             "from_xy", "message"}] * 2

    def test_a_chunk_of_one_call_is_still_allowed(self, paris):
        """The permission is 'up to K'. A courier that is unsure must be able
        to spend a turn on one call without breaking the grammar."""
        calls, nodes, _ = walk_chain(paris, 1)
        env = courier(paris)
        turn = ChunkedCourierSession(env, action_chunk=3).step(fence(*calls))
        assert turn.status == "accepted"
        assert turn.chunk_len == turn.chunk_executed == 1
        assert env.node_id == nodes[-1]


@needs_maps
class TestOnlyTheFirstCallReachesTheWorld:
    """Under a plan, "the turn stops at the first refusal" is not a rule so
    much as an arithmetic consequence: there is only ever one call to stop
    at. What has to hold is that the rest never runs, however tempting the
    plan looked."""

    def test_a_refused_first_call_leaves_the_plan_unrun(self, paris):
        calls, nodes, _ = walk_chain(paris, 3)
        env = courier(paris)
        session = ChunkedCourierSession(env, action_chunk=3)
        before = env.node_id
        turn = session.step(
            fence('walk_to("Rue Nowhere", "north")', calls[0], calls[1]))

        assert [row["status"] for row in turn.chunk] == [
            "rejected", "planned", "planned"]
        assert turn.chunk[0]["code"] == "no_such_street"
        assert turn.status == "rejected" and turn.error == "no_such_street"
        assert env.node_id == before, "the plan behind it did not run"
        assert not session.finished

    def test_a_good_first_call_still_does_not_run_the_rest(self, paris):
        """The one that matters. The second waypoint was named relative to a
        position the first walk was predicted to reach, and predictions are
        exactly what a re-plan exists to correct."""
        calls, nodes, _ = walk_chain(paris, 3)
        env = courier(paris)
        session = ChunkedCourierSession(env, action_chunk=3)
        turn = session.step(fence(*calls))
        assert env.node_id == nodes[1], "one hop"
        assert env.node_id != nodes[-1], "not three"
        assert turn.chunk_executed == 1

    def test_the_refusal_is_charged_exactly_once(self, paris):
        """Not zero times (a free refusal is a rangefinder) and not twice (the
        env charges per call; a session that also charged its own floor would
        double-bill)."""
        env = courier(paris)
        session = ChunkedCourierSession(env, action_chunk=3)
        turn = session.step(
            fence('walk_to("Rue Nowhere", "north")', "look()", "look()"))
        assert env.rejected_actions == 1
        assert turn.chunk[0]["sim_seconds"] == pytest.approx(
            REJECTED_ACTION_SECONDS)
        assert env.sim_seconds == pytest.approx(REJECTED_ACTION_SECONDS)
        assert turn.sim_seconds == pytest.approx(env.sim_seconds)

    def test_the_plan_is_named_rather_than_missing(self, paris):
        """A plan never made and a plan made and superseded must not read the
        same, and only one of them is worth showing a trainer."""
        calls, _, _ = walk_chain(paris, 3)
        session = ChunkedCourierSession(courier(paris), action_chunk=3)
        turn = session.step(fence(*calls))
        assert [row["action"] for row in turn.chunk[1:]] == [
            calls[1], calls[2]]
        assert all(row["status"] == "planned" for row in turn.chunk[1:])

    def test_a_terminating_first_call_ends_the_episode_cleanly(self, paris):
        env = courier(paris)
        order = env.active_order()
        env.node_id = order.pickup.kerb_node
        assert env.collect().ok
        env.node_id = order.dropoff.kerb_node
        session = ChunkedCourierSession(env, action_chunk=3)

        turn = session.step(fence("hand_over()", "look()", "look()"))
        assert turn.chunk[0]["status"] == "accepted"
        assert [row["status"] for row in turn.chunk[1:]] == ["planned", "planned"]
        assert session.finished
        assert session.run.termination_reason == "delivered"
        assert session.spend.tool_calls == 1


@needs_maps
class TestOverTheLimitIsRefusedNotTrimmed:
    def test_more_calls_than_the_turn_allows_is_refused_whole(self, paris):
        """Trimming to K would run a prefix of a plan the model wrote as a
        whole -- executing something it did not choose, which is the failure
        the parser exists to refuse."""
        calls, _, _ = walk_chain(paris, 4)
        env = courier(paris)
        start = env.node_id
        session = ChunkedCourierSession(env, action_chunk=3)
        turn = session.step(fence(*calls))

        assert turn.status == "rejected" and turn.error == "too_many_calls"
        assert turn.chunk_executed == 0 and turn.chunk == []
        assert turn.chunk_len == 4, "the report says how many were written"
        assert env.node_id == start, "nothing in the reply was carried out"

    def test_the_refusal_names_the_limit(self, paris):
        """A limit the courier is refused against and never told is a rule it
        can only find by losing turns to it."""
        calls, _, _ = walk_chain(paris, 4)
        session = ChunkedCourierSession(courier(paris), action_chunk=3)
        session.step(fence(*calls))
        assert "at most 3" in session.feedback
        assert "4 calls" in session.feedback

    def test_the_parser_raises_the_refusal_rather_than_a_format_error(self):
        """The reply was well formed; the courier simply asked for more than
        the turn allows. The two get different words back, so they must be
        different exceptions."""
        with pytest.raises(ChunkTooLong) as raised:
            parse_chunk(fence("look()", "look()", "look()"), {"look"}, 2)
        assert raised.value.found == 3 and raised.value.limit == 2
        assert raised.value.code == "too_many_calls"

    def test_an_over_long_chunk_costs_what_a_refusal_costs(self, paris):
        """A free retry would make overflowing the cheapest way to buy a turn's
        thinking time, and neither currency this benchmark reports would show
        it happening."""
        calls, _, _ = walk_chain(paris, 4)
        env = courier(paris)
        session = ChunkedCourierSession(env, action_chunk=3)
        turn = session.step(fence(*calls))
        assert env.sim_seconds == pytest.approx(REJECTED_ACTION_SECONDS)
        assert turn.sim_seconds == pytest.approx(REJECTED_ACTION_SECONDS)
        assert env.rejected_actions == 1
        assert session.spend.steps == 1 and session.spend.tool_calls == 1
        assert session.spend.format_errors == 0, (
            "it is a refusal, so it must not spend the three-strike budget")

    def test_repeating_an_over_long_chunk_ends_the_session_stuck(self, paris):
        """The same rule the stock session applies to a repeated refused call:
        nothing about the world has changed, so nothing about the reply can."""
        calls, _, _ = walk_chain(paris, 4)
        session = ChunkedCourierSession(courier(paris), action_chunk=3)
        for _ in range(4):
            session.step(fence(*calls))
        assert session.finished
        assert session.run.termination_reason == "stuck"


@needs_maps
class TestTheBudgetStopsAChunkAtTheBoundary:
    def test_a_turn_spends_one_call_against_the_budget(self, paris):
        """A chunk cannot straddle a budget boundary any more: it spends one
        call, so it either fits or the turn was already over."""
        env = courier(paris)
        session = ChunkedCourierSession(
            env, action_chunk=3, budgets=Budgets(tool_calls=1))
        calls, _, _ = walk_chain(paris, 3)
        session.step(fence(*calls))
        assert session.spend.tool_calls == 1
        # The next turn is the one the budget stops.
        turn = session.step(fence(*calls))
        assert turn.status == "truncated"

    def test_a_budget_already_spent_ends_the_turn_as_it_always_did(self, paris):
        """Chunking does not add a way past a limit that was already reached."""
        calls, _, _ = walk_chain(paris, 1)
        session = ChunkedCourierSession(courier(paris), action_chunk=3,
                                        budgets=Budgets(steps=0))
        turn = session.step(fence(*calls))
        assert turn.status == "truncated"
        assert turn.error == "step_budget_exhausted"
        assert session.finished


@needs_maps
class TestThePromptStatesTheRule:
    def test_above_one_the_prompt_teaches_the_plan_and_which_call_happens(
            self, paris):
        """A model told it may write three calls and not told that only the
        first happens will write three it means, and plan its next turn from
        where the third was supposed to land."""
        # Whitespace-normalised: the rule is wrapped for the prompt, so a
        # phrase that matters can straddle a line break.
        prompt = " ".join(ChunkedCourierSession(
            courier(paris), action_chunk=3).system_prompt().lower().split())
        assert "up to 3 calls" in prompt
        assert "only the first one happens" in prompt
        assert "plan" in prompt
        # ...and it says why the plan will be wrong, which is the reason to
        # re-write it rather than trust it.
        assert "not land exactly where you predicted" in prompt

    def test_at_one_the_prompt_says_exactly_one_call_and_nothing_about_chunks(
            self, paris):
        stock = CourierSession(courier(paris)).system_prompt()
        assert "containing exactly one call" in stock
        assert "up to" not in stock.lower().split("how to reply")[1]

    def test_the_two_rules_are_never_in_the_prompt_together(self, paris):
        """One replaces the other. Both at once is a contradiction the policy
        has to guess its way out of."""
        chunked = ChunkedCourierSession(courier(paris),
                                        action_chunk=4).system_prompt()
        assert "containing exactly one call" not in chunked
        assert "UP TO 4 calls" in chunked

    def test_the_worked_example_carries_as_many_calls_as_the_rule_allows(
            self, paris):
        """The example is the rule in miniature; a chunked prompt whose only
        example is a single call teaches the shape it is moving away from."""
        chunked = ChunkedCourierSession(courier(paris),
                                        action_chunk=2).system_prompt()
        block = chunked.split("HOW TO REPLY")[1].split("```")[1]
        assert block.strip().count("walk_to(") == 2

    def test_a_chunk_of_less_than_one_call_is_refused_at_construction(self, paris):
        with pytest.raises(ValueError, match="action_chunk"):
            ChunkedCourierSession(courier(paris), action_chunk=0)


# ── the embodied path ────────────────────────────────────────────────────────


class _AdapterUnderTest(EmbodiedCourierGymEnv):
    """The embodied adapter with the phone map rasterised by PIL instead of
    cairosvg, exactly as the embodied runtime's own tests do; the rasteriser
    is inherited stock code and not what these tests defend."""

    def _rasterise(self, svg, index):
        from PIL import Image

        return self._fit(Image.new("RGB", (720, 540), (240, 240, 240)))


def run(coro):
    return asyncio.run(coro)


@pytest.fixture()
def service(tmp_path):
    stub = FakeTrackBService(tmp_path / "svc").start()
    yield stub
    stub.stop()


@needs_maps
class TestTheEmbodiedAdapterDrivesAChunk:
    """Track B is the backend chunking is for: the walk is engine time and the
    prefill is the thing being amortised."""

    @pytest.fixture()
    def config(self, service, tmp_path):
        endpoints = write_endpoints(tmp_path / "endpoints.json", [service])
        return {"backend": "embodied",
                "ue_endpoints": str(endpoints),
                "live_cache_root": str(tmp_path / "cache"),
                "difficulty": "solo", "stride": "waypoint",
                "max_turns": 3, "max_images": 1}

    def test_a_chunk_of_three_walks_moves_the_pawn_once(
            self, paris, service, tmp_path):
        """End to end over the embodied backend: three waypoints reach the
        env, one reaches the engine."""
        endpoints = write_endpoints(tmp_path / "endpoints.json", [service])
        adapter = _AdapterUnderTest({
            "backend": "embodied", "ue_endpoints": str(endpoints),
            "live_cache_root": str(tmp_path / "cache"),
            "difficulty": "solo", "stride": "waypoint",
            "max_turns": 3, "max_images": 1, "action_chunk": 3})
        run(adapter.reset(0))
        env = adapter._env
        rows = env.candidates()
        calls = [f'walk_to("{r["street"]}", "{r["heading"]}")'
                 for r in rows[:1] * 3]
        walks_before = len(service.walks)
        _, _, _, info = run(adapter.step(fence(*calls)))

        assert info["chunk_len"] == 3
        assert info["chunk_executed"] == 1
        assert len(service.walks) - walks_before >= 1
        run(adapter.close())

    def test_the_info_contract_is_unchanged_and_two_keys_wider(
            self, paris, config, service):
        """The trainer reads the inherited keys; the two chunk keys are added
        beside them, never instead of them."""
        env = _AdapterUnderTest({**config, "action_chunk": 3})
        run(env.reset(0))
        calls, _, _ = walk_chain(paris, 2, difficulty="solo", stride="waypoint")
        _, reward, _, info = run(env.step(fence(*calls)))
        assert isinstance(reward, float)
        for key in ("status", "action", "error", "sim_seconds", "turns",
                    "images_dropped", "success", "env_return", "earnings",
                    "termination", "chunk_len", "chunk_executed"):
            assert key in info
        assert info["chunk_len"] == 2
        run(env.close())

    def test_without_the_key_the_backend_runs_the_stock_session(
            self, paris, config, service):
        """Default off, and off is the session the embodied measurements were
        taken with -- plus the two keys, which answer for it as a chunk of
        one."""
        from embodiedbench.agent.courier.session import CourierSession

        env = _AdapterUnderTest(config)
        run(env.reset(0))
        assert env.action_chunk == 1
        assert type(env._session) is CourierSession
        calls, _, _ = walk_chain(paris, 1, difficulty="solo", stride="waypoint")
        _, _, _, info = run(env.step(fence(*calls)))
        assert info["chunk_len"] == 1 and info["chunk_executed"] == 1
        run(env.close())

    def test_a_chunk_smaller_than_one_is_refused_at_construction(self):
        with pytest.raises(ValueError, match="action_chunk"):
            EmbodiedCourierGymEnv({"backend": "embodied", "action_chunk": 0})


def _call_args(call: str) -> tuple[str, str]:
    """The two quoted arguments of a rendered ``walk_to`` call."""
    import re

    return tuple(re.findall(r'"([^"]*)"', call))  # type: ignore[return-value]


def test_a_chunked_session_still_obeys_the_environment_narration():
    """Chunking and narration are independent axes and must stay that way.

    ChunkedCourierSession overrides system_prompt, and the first version of
    that override dropped the narration argument -- so every chunked run was
    pinned to narration="none" however the environment was configured. Two
    settings would then have differed by whether chunking happened to be on,
    which is not a difference anybody chose.
    """
    from embodiedbench.agent.courier.chunk import ChunkedCourierSession
    from embodiedbench.agent.courier.tools import available_tools

    actions = ["VIEW_ORDERS", "ACCEPT_ORDER", "PICKUP", "DROP_OFF",
               "WAIT", "MOVE_TO", "NAVIGATE"]

    class _Env:
        narration = "all"
        allowed_actions = ("walk_to", "wait")

        def candidates(self):
            return []

        def summary(self):
            return {}

    for chunk in (1, 3):
        session = ChunkedCourierSession.__new__(ChunkedCourierSession)
        session.env = _Env()
        session.city = "Paris"
        session.action_chunk = chunk
        session.tools = available_tools(actions)
        prompt = session.system_prompt()
        assert "EVERYTHING YOU NEED IS IN THE WORDS" in prompt, (
            f"narration=all was dropped at action_chunk={chunk}")
        multi = "UP TO 3 calls" in prompt
        assert multi is (chunk == 3), "the two axes are leaking into each other"
