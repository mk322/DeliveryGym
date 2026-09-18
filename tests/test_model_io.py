"""The model-facing side of the harness, checked without a model.

Everything pinned here is a mistake this benchmark actually made and then
believed: a truncated reply scored as a format failure, a transport error fed
to the parser as though the model had said it, a reasoning model refused for
thinking out loud, and a format error charged as a world step. They are cheap
to test and expensive to rediscover.
"""

from __future__ import annotations

import pytest

from embodiedbench.eval import stats

from embodiedbench.agent.courier.model_io import (
    ModelClient,
    looks_truncated,
    split_reasoning,
)


class Boom(Exception):
    pass


def parse_walk(reply: str):
    """Stand-in for the reply contract: accepts exactly one fenced call."""
    if reply.count("```") != 2 or "walk_to" not in reply:
        raise Boom("no single fenced call")
    return reply.split("```")[1].strip()


def client(replies, **kwargs):
    """A ModelClient whose transport returns canned chat-completion bodies."""
    c = ModelClient("http://unused", "stub", **kwargs)
    calls = []

    def fake_post(payload):
        calls.append(payload)
        return replies[min(len(calls) - 1, len(replies) - 1)]

    c._post_with_retries = fake_post  # type: ignore[assignment]
    c.sent = calls  # type: ignore[attr-defined]
    return c


def body(content, finish="stop", **message):
    return {"choices": [{"finish_reason": finish,
                         "message": {"content": content, **message}}]}


class TestReasoningIsNotTheAnswer:
    def test_inline_think_block_is_cut_off(self):
        answer, reasoning = split_reasoning(
            {"content": "<think>\nmaybe ```walk_to(\"Rue Cujas\", \"north\")```\n</think>\n"
                        "THOUGHT: go\n```\nwalk_to(\"Rue de Grenelle\", \"east\")\n```"})
        assert "Rue Cujas" not in answer
        assert "walk_to(\"Rue de Grenelle\", \"east\")" in answer
        assert "maybe" in reasoning

    def test_a_separate_reasoning_field_is_respected(self):
        """vLLM with a reasoning parser puts the thought in its own field and
        leaves content clean. The same model does the opposite without one, so
        neither shape can be assumed."""
        answer, reasoning = split_reasoning(
            {"content": "```\nwalk_to(\"Rue de Grenelle\", \"east\")\n```", "reasoning_content": "thinking"})
        assert answer == "```\nwalk_to(\"Rue de Grenelle\", \"east\")\n```"
        assert reasoning == "thinking"

    def test_an_unclosed_thought_yields_no_answer(self):
        """Cut off mid-thought. The calls inside are the ones it was arguing
        itself out of, and running one is a guess."""
        answer, _ = split_reasoning({"content": "<think>\nmaybe ```walk_to(\"Rue de Grenelle\", \"east\")```"})
        assert answer == ""

    @pytest.mark.parametrize("open_tag,close_tag", [
        ("<think>", "</think>"), ("<thinking>", "</thinking>"),
        ("<reasoning>", "</reasoning>"),
    ])
    def test_the_common_conventions_are_all_handled(self, open_tag, close_tag):
        answer, _ = split_reasoning(
            {"content": f"{open_tag}x{close_tag}\n```\nwalk_to(\"Rue de Grenelle\", \"east\")\n```"})
        assert answer == "```\nwalk_to(\"Rue de Grenelle\", \"east\")\n```"


class TestATruncatedReplyIsNotAModelFailure:
    def test_length_finish_raises_the_budget_and_retries(self):
        c = client([body("THOUGHT: I am thinking about", finish="length"),
                    body("THOUGHT: go\n```\nwalk_to(\"Rue de Grenelle\", \"east\")\n```")],
                   max_tokens=100)
        reply, parsed, rejected = c.act([{"role": "user", "content": "x"}], parse_walk)
        assert parsed == "walk_to(\"Rue de Grenelle\", \"east\")"
        assert c.stats.truncations == 1
        assert rejected == [], "a truncated reply must not count as a bad reply"
        assert c.sent[1]["max_tokens"] > c.sent[0]["max_tokens"]

    def test_looks_truncated_reads_finish_reason(self):
        assert looks_truncated({"finish_reason": "length"})
        assert not looks_truncated({"finish_reason": "stop"})


class TestAFormatErrorIsRequeriedNotCharged:
    """mini-swe-agent re-prompts on an unparseable reply and never lets the
    world see it. Charging it as a turn measures the model's luck at
    formatting rather than its competence at the task."""

    def test_a_bad_reply_is_retried_with_the_error_fed_back(self):
        c = client([body("I will walk north."),
                    body("THOUGHT: ok\n```\nwalk_to(\"Rue de Grenelle\", \"east\")\n```")])
        reply, parsed, rejected = c.act([{"role": "user", "content": "x"}], parse_walk)
        assert parsed == "walk_to(\"Rue de Grenelle\", \"east\")"
        assert c.stats.requeries == 1
        assert rejected == ["I will walk north."]
        # the failed attempt and the complaint are both in the retry
        assert c.sent[1]["messages"][-2]["role"] == "assistant"
        assert "could not be read" in c.sent[1]["messages"][-1]["content"]

    def test_giving_up_is_reported_rather_than_guessed(self):
        c = client([body("nope")], max_requeries=2)
        reply, parsed, rejected = c.act([{"role": "user", "content": "x"}], parse_walk)
        assert parsed is None, "the harness must not invent an action"
        assert len(rejected) == 3
        assert c.stats.unparseable == 1

    def test_requeries_are_counted_not_hidden(self):
        """A model needing three attempts a turn should still look worse than
        one needing none, so the count is part of the result."""
        c = client([body("bad"), body("bad"), body("THOUGHT: x\n```\nwalk_to(\"Rue de Grenelle\", \"east\")\n```")])
        c.act([{"role": "user", "content": "x"}], parse_walk)
        assert c.stats.as_dict()["requeries"] == 2
        assert c.stats.as_dict()["model_calls"] == 3


class TestTransportFailuresAreNeverReplies:
    def test_an_http_error_carries_the_servers_own_message(self):
        c = ModelClient("http://unused", "stub")

        def explode(payload):
            raise RuntimeError('HTTP 400: {"message":"At most 6 image(s)"}')

        c._post_with_retries = explode  # type: ignore[assignment]
        with pytest.raises(RuntimeError, match="At most 6 image"):
            c.act([{"role": "user", "content": "x"}], parse_walk)

    def test_a_4xx_is_not_retried(self):
        """It is a request this harness built wrongly; sending it again
        unchanged only spends the clock."""
        c = ModelClient("http://unused", "stub", max_transport_retries=3)
        tries = []

        def explode(payload):
            tries.append(1)
            raise RuntimeError("HTTP 400: bad")

        c._post = explode  # type: ignore[assignment]
        with pytest.raises(RuntimeError):
            c._post_with_retries({})
        assert len(tries) == 1


class TestTheBudgetNeverWalksIntoTheContextLimit:
    """Raising max_tokens after a truncation is right until it is not.

    Escalating to 10000 output tokens against a 10240-token model leaves 240
    for the prompt, and every turn 400s. Qwen3.5-9B episodes ended after a mean
    of 2.6 turns that way, with zero format errors -- the harness had broken its
    own requests while adapting. The server states both numbers in the refusal,
    so the fix is to read them rather than carry a per-model ceiling.
    """

    def _clamping_client(self):
        c = ModelClient("http://unused", "stub", max_tokens=8000,
                        max_tokens_ceiling=10000)
        seen = []

        def fake(payload):
            # A realistic server: it refuses exactly when the request cannot
            # fit, and says so with both numbers.
            seen.append(payload["max_tokens"])
            if payload["max_tokens"] + 8000 > 10240:
                raise RuntimeError(
                    'HTTP 400: {"message":"This model\'s maximum context length '
                    'is 10240 tokens. However, you requested '
                    f'{payload["max_tokens"]} output tokens and your prompt '
                    'contains 8000 tokens"}')
            return body("THOUGHT: ok\n```\nwalk_to(\"Rue de Grenelle\", \"east\")\n```")

        c._post_with_retries = fake  # type: ignore[assignment]
        c.seen = seen  # type: ignore[attr-defined]
        return c

    def test_it_backs_off_to_the_room_the_server_reports(self):
        c = self._clamping_client()
        _, parsed, _ = c.act([{"role": "user", "content": "x"}], parse_walk)
        assert parsed == "walk_to(\"Rue de Grenelle\", \"east\")"
        assert c.seen[1] == 10240 - 8000 - 64
        assert c.stats.budget_clamps == 1

    def test_the_clamp_sticks_for_later_turns(self):
        c = self._clamping_client()
        c.act([{"role": "user", "content": "x"}], parse_walk)
        assert c.max_tokens_ceiling <= 10240 - 8000 - 64

    def test_an_unrelated_400_still_raises(self):
        """Only a context-limit refusal carries a usable number. Swallowing
        every 400 would hide the six-image limit that started all this."""
        c = ModelClient("http://unused", "stub")

        def fake(payload):
            raise RuntimeError('HTTP 400: {"message":"At most 6 image(s)"}')

        c._post_with_retries = fake  # type: ignore[assignment]
        with pytest.raises(RuntimeError, match="At most 6 image"):
            c.act([{"role": "user", "content": "x"}], parse_walk)


class TestAnOverlongPromptShedsHistoryRatherThanDying:
    """When the budget cannot shrink any further, the prompt is what is too big.

    A reasoning model behind a 10k context fills it after a handful of turns of
    history, and every turn then 400s. The alternative to shedding history here
    is a per-model history setting, which is one more number to guess wrong for
    each new model.
    """

    def test_the_oldest_exchange_is_dropped_and_the_call_retried(self):
        c = ModelClient("http://unused", "stub", max_tokens=512)
        sizes = []

        def fake(payload):
            n = len(payload["messages"])
            sizes.append(n)
            if n > 3:
                raise RuntimeError(
                    'HTTP 400: {"message":"This model\'s maximum context length '
                    'is 10240 tokens. However, you requested 512 output tokens '
                    'and your prompt contains 10200 tokens"}')
            return body("THOUGHT: ok\n```\nwalk_to(\"Rue de Grenelle\", \"east\")\n```")

        c._post_with_retries = fake  # type: ignore[assignment]
        convo = [{"role": "system", "content": "s"}]
        convo += [{"role": "user", "content": "u"}, {"role": "assistant", "content": "a"}] * 3
        _, parsed, _ = c.act(convo, parse_walk)
        assert parsed == "walk_to(\"Rue de Grenelle\", \"east\")"
        assert c.stats.history_drops >= 1
        assert sizes[-1] < sizes[0]

    def test_it_gives_up_rather_than_looping_when_nothing_is_left_to_drop(self):
        c = ModelClient("http://unused", "stub")

        def fake(payload):
            raise RuntimeError(
                'HTTP 400: {"message":"This model\'s maximum context length is '
                '10240 tokens. However, you requested 100 output tokens and '
                'your prompt contains 10200 tokens"}')

        c._post_with_retries = fake  # type: ignore[assignment]
        with pytest.raises(RuntimeError):
            c.act([{"role": "system", "content": "s"},
                   {"role": "user", "content": "u"}], parse_walk)


class TestBothWordingsOfTheContextRefusalAreUnderstood:
    """vLLM phrases the same refusal two ways depending on where it fires.

    Handling only the first shape let the second through as fatal and ended
    Qwen3.5-9B episodes after a mean of 6.6 turns -- the harness reading the
    server's limits, but only when the server used the words it expected.
    """

    @pytest.mark.parametrize("message", [
        "This model's maximum context length is 10240 tokens. However, you "
        "requested 4000 output tokens and your prompt contains 10200 tokens",
        "Input length (10384) exceeds model's maximum context length (10240).",
    ])
    def test_an_overlong_prompt_sheds_history_either_way(self, message):
        c = ModelClient("http://unused", "stub", max_tokens=512)
        seen = []

        def fake(payload):
            seen.append(len(payload["messages"]))
            if len(payload["messages"]) > 3:
                raise RuntimeError('HTTP 400: {"message":"%s"}' % message)
            return body("THOUGHT: ok\n```\nwalk_to(\"Rue de Grenelle\", \"east\")\n```")

        c._post_with_retries = fake  # type: ignore[assignment]
        convo = [{"role": "system", "content": "s"}]
        convo += [{"role": "user", "content": "u"}, {"role": "assistant", "content": "a"}] * 3
        _, parsed, _ = c.act(convo, parse_walk)
        assert parsed == "walk_to(\"Rue de Grenelle\", \"east\")", f"not handled: {message[:40]}"
        assert c.stats.history_drops >= 1


class TestATimeoutIsQueueingNotAnEndedEpisode:
    """A busy server is not a result.

    Running two evaluations against one server doubled latency past the
    deadline; 16 of 40 episodes ended on a timeout and were recorded as having
    finished, which pulled the mean episode length from 35 turns to 18 and made
    the delivery rate an underestimate of unknown size. Same class of mistake
    as parsing an HTTP error as the model's reply.
    """

    def test_a_timeout_is_retried_more_patiently_than_other_faults(self):
        c = ModelClient("http://unused", "stub", max_transport_retries=1,
                        max_timeout_retries=4)
        tries = []

        def flaky(payload):
            tries.append(1)
            if len(tries) < 4:
                raise TimeoutError("timed out")
            return body("THOUGHT: ok\n```\nwalk_to(\"Rue de Grenelle\", \"east\")\n```")

        c._post = flaky  # type: ignore[assignment]
        c._sleep_patch = True
        import embodiedbench.agent.courier.model_io as mio
        real_sleep, mio.time.sleep = mio.time.sleep, lambda _: None
        try:
            out = c._post_with_retries({})
        finally:
            mio.time.sleep = real_sleep
        assert out["choices"][0]["message"]["content"].startswith("THOUGHT")
        assert c.stats.timeouts == 3

    def test_timeouts_are_counted_in_the_summary(self):
        c = ModelClient("http://unused", "stub")
        c.stats.timeouts = 2
        assert c.stats.as_dict()["timeouts"] == 2


class TestDowntimeIsNotFailure:
    """An episode the server cut short is not an episode the model failed.

    ``run_vlm.py`` already refused to hand a failed request to the parser --
    that fix came after 26 of 40 episodes were killed by HTTP 400s reported as
    format errors. It then counted those same episodes in the denominator
    anyway. One run lost seeds 24-39 to a server that stopped answering:
    sixteen episodes that ended on turn 1, reported as sixteen non-deliveries,
    putting the delivery rate at 8/40 = 20% when the model had been asked 24
    times and delivered 8, which is 33%.
    """

    def summarise(self, runs):
        """The arithmetic run_vlm.py does, isolated so it can be pinned."""
        aborted = [r for r in runs
                   if any(t["status"] == "infra_error" for t in r["transcript"])]
        scored = [r for r in runs if r not in aborted]
        return {
            "delivered": sum(r["summary"]["delivered"] for r in scored),
            "issued": sum(r["summary"]["orders_issued"] for r in scored),
            "scored": len(scored),
            "aborted": len(aborted),
        }

    def episode(self, seed, delivered, statuses):
        return {"seed": seed,
                "summary": {"delivered": delivered, "orders_issued": 1},
                "transcript": [{"status": s} for s in statuses]}

    def test_an_aborted_episode_leaves_the_denominator(self):
        runs = ([self.episode(i, 1, ["accepted"] * 3) for i in range(8)]
                + [self.episode(i, 0, ["accepted"] * 3) for i in range(8, 24)]
                + [self.episode(i, 0, ["infra_error"]) for i in range(24, 40)])
        out = self.summarise(runs)
        assert out["aborted"] == 16
        assert (out["delivered"], out["issued"]) == (8, 24)
        assert out["delivered"] / out["issued"] == pytest.approx(1 / 3)

    def test_a_run_with_no_downtime_is_unchanged(self):
        runs = [self.episode(i, i % 2, ["accepted"]) for i in range(10)]
        out = self.summarise(runs)
        assert out["aborted"] == 0
        assert out["issued"] == 10

    def test_downtime_late_in_an_episode_still_excludes_it(self):
        """The model never got to finish, so its zero says nothing."""
        runs = [self.episode(0, 0, ["accepted"] * 19 + ["infra_error"])]
        assert self.summarise(runs)["scored"] == 0


class TestThePairedComparison:
    """The arithmetic that decides whether a training run worked.

    It has to be right, because the whole point of it is that the unpaired
    comparison it replaces was reading noise as signal.
    """

    def scores(self, values, start=0):
        return {start + i: float(v) for i, v in enumerate(values)}

    def test_the_unchanged_seeds_do_not_enter_the_test(self):
        """Sixty seeds that failed both times say nothing about a change; a
        test that counts them would call any run insignificant."""
        before = self.scores([0.0] * 65)
        after = self.scores([5.0] * 5 + [0.0] * 60)
        out = stats.paired_compare(before, after)
        assert (out["wins_b"], out["losses_b"], out["ties"]) == (5, 0, 60)
        assert out["p_sign_test"] == pytest.approx(2 / 2 ** 5)
        assert out["p_sign_test"] < 0.07

    def test_a_symmetric_change_is_not_a_change(self):
        before = self.scores([0.0] * 6 + [5.0] * 6)
        after = self.scores([5.0] * 6 + [0.0] * 6)
        assert stats.paired_compare(before, after)["p_sign_test"] == pytest.approx(1.0)

    def test_nothing_moved_is_not_significant(self):
        same = self.scores([1.0] * 8)
        assert stats.paired_compare(same, dict(same))["p_sign_test"] == 1.0

    def test_a_run_that_only_improved_is_reported_as_improvement(self):
        before = self.scores([0.0] * 10)
        after = self.scores([5.0] * 6 + [0.0] * 4)
        out = stats.paired_compare(before, after)
        assert (out["wins_b"], out["losses_b"]) == (6, 0)
        assert out["p_sign_test"] < 0.05
        assert out["mean_diff_b_minus_a"] == pytest.approx(3.0)

    def test_mismatched_seed_sets_are_reported_not_truncated(self):
        """Silently zipping to the shorter one would compare seed i with seed
        j and report it as paired. Only the shared seeds are compared, and the
        ones left out are counted where a reader will see them."""
        out = stats.paired_compare(self.scores([0.0] * 8), self.scores([0.0] * 6))
        assert out["n"] == 6
        assert (out["dropped_a_only"], out["dropped_b_only"]) == (2, 0)

    def test_earnings_that_move_without_crossing_zero_still_count(self):
        """A seed delivering late for half the fee and then on time for all of
        it has improved, and a delivered/not-delivered test cannot see it --
        the paired difference can."""
        out = stats.paired_compare(self.scores([2.5] * 8), self.scores([5.0] * 8))
        assert (out["wins_b"], out["losses_b"]) == (8, 0)
        assert out["mean_diff_b_minus_a"] == pytest.approx(2.5)
        assert out["diff_ci"][0] > 0
