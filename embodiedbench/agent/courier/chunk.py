"""Several calls in one turn: the chunked courier turn.

WHY. In the embodied condition UE owns locomotion, and one turn costs 25-105 s
of engine time for the walk against a few seconds of generation -- the GPU is
idle for most of the episode. Chunking does not fix that by overlapping
inference with execution. It cuts the number of round trips, and with them the
number of ~5k-token multimodal prefills, by a factor of K: the expensive part
of a turn is the prompt, and a chunk pays for one prompt and buys K actions.

The environment already believed this in one direction. ``follow_street(street,
n)`` walks up to n waypoints for one call, and the block stride collapses a
whole block into one decision -- both are chunks of a fixed, single-tool shape.
This generalises that to a chunk the *model* composes: up to K calls of any
tools it has, in the order it wants them.

WHAT A CHUNK COSTS, which is the part that must not be quietly generous:

  * one TURN -- one prompt, one generation, one ``TurnLog``. That is the whole
    saving and it is the only thing chunking is allowed to save;
  * K STEPS and K TOOL CALLS against ``Budgets``, one per executed call,
    because each of them is an action in the world and the world does not care
    how they were transmitted. A chunk that charged one step for three hops
    would be a policy discovering that the cheapest route is a long chunk;
  * every second the world charges for each of them, summed.

WHAT A CHUNK RISKS, which is what makes it a decision rather than a free lunch:
execution stops at the FIRST refusal, termination, or budget boundary, and the
calls after it are thrown away unexecuted. The courier plans blind past its
first mistake -- a chunk is a bet that the world will still look the way the
model predicted after each call, and the prompt says so.

The grammar, stated exactly:

  * one fenced block, as before. Two blocks is still a format error -- the
    thought-then-answer split ``strip_reasoning`` handles is not a chunk;
  * inside it, 1..K calls, found by the same ``split_calls`` scanner, each
    checked by the same ``build_call`` (tool set, then argument types), so a
    chunk is held to exactly the rules one call is held to;
  * MORE than K calls is refused, never trimmed. Truncating to K would run a
    prefix of a plan the model wrote as a whole, which is the "execute
    something it did not choose" failure that ``parse_reply`` exists to
    refuse. It is refusal-shaped rather than format-shaped (``RejectedAction``,
    code ``too_many_calls``) because the reply is well-formed and the courier
    simply asked for more than the turn allows -- and it costs what a refusal
    costs, so overflowing is not a free retry;
  * K=1 is the stock parser and the stock turn, by delegation rather than by
    resemblance: ``parse_chunk`` calls ``parse_reply`` and ``step`` calls
    ``CourierSession.step``. The chunked path cannot regress the unchunked one
    because at K=1 it is not running.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from embodiedbench.agent.courier.loop import (
    ACTION_BLOCK,
    FormatError,
    ParsedAction,
    RejectedAction,
    TruncatedReply,
    TurnLog,
    _bare_call,
    budget_exceeded,
    build_call,
    looks_cut_off,
    parse_reply,
    split_calls,
    strip_reasoning,
)
from embodiedbench.agent.courier.prompts import (
    CHUNK_FEEDBACK_TEMPLATE,
    FORMAT_ERROR_TEMPLATE,
    REJECTED_TEMPLATE,
    REJECTED_TEMPLATE_ACTION,
    REJECTED_TEMPLATE_PIXEL,
    TRUNCATED_TEMPLATE,
    build_system_prompt,
)
from embodiedbench.agent.courier.session import CourierSession
from embodiedbench.agent.courier.tools import Tool

#: One parsed call. The same object a single-call turn produces -- a chunk is
#: a list of ordinary decisions, not a new kind of decision.
Call = ParsedAction


class ChunkTooLong(RejectedAction):
    """More calls than the turn allows: refused whole, never trimmed to fit.

    A ``RejectedAction`` rather than a ``FormatError`` because nothing about
    the reply was malformed -- the grammar was right and the courier asked for
    more than it is allowed. That distinction is the one the feedback rests on:
    a format error asks for the shape again, and this asks for a shorter plan.
    """

    def __init__(self, found: int, limit: int):
        self.found = found
        self.limit = limit
        super().__init__(
            f"You wrote {found} calls in one block, and a turn takes at most "
            f"{limit}. Nothing in that reply was carried out. Write at most "
            f"{limit} calls, in the order you want them done.",
            code="too_many_calls",
        )


@dataclass
class ChunkTurnLog(TurnLog):
    """A ``TurnLog`` with the per-call detail beside it.

    A subclass rather than a parallel record, so every existing consumer --
    the training adapter's ``info``, the trajectory writer, the report --
    keeps reading the fields it already read and gets the same types for them.
    ``status``, ``action``, ``sim_seconds`` and ``reward`` describe the turn as
    a whole (the status of the call that ended it, every call the reply named,
    the summed clock, the summed reward); the chunk fields describe its parts.

    ``chunk`` carries one row per call that was BUILT, which for an over-long
    chunk is none of them -- the limit is checked before the calls are typed,
    because type-checking calls that will not run is work done to produce a
    number nobody may act on. ``chunk_len`` is what the reply contained, so
    those two disagree in exactly that one case, and it is the case where the
    disagreement is the report.

    ``chunk_aborted_at`` is the index the turn stopped at, and the row at that
    index says whether it ran: ``rejected`` for a refusal (it ran and was
    refused), ``dropped`` for a budget boundary (it never ran).
    """

    chunk: list[dict[str, Any]] = field(default_factory=list)
    chunk_len: int = 0
    chunk_executed: int = 0
    chunk_aborted_at: int | None = None


def parse_chunk(
    reply: str,
    allowed: set[str],
    max_calls: int = 1,
    *,
    tools_by_name: dict[str, Tool] | None = None,
) -> list[Call]:
    """Up to ``max_calls`` calls out of one model reply, in the order written.

    At ``max_calls == 1`` this *is* ``parse_reply``: same errors, same words,
    same leniency about the fence. Above 1 the only rule that changes is how
    many calls one block may hold; everything else -- the reasoning strip, the
    one-block rule, the bare-call fallback, the truncation test, the tool-set
    check and the argument types -- is the same code doing the same thing.
    """
    if max_calls <= 1:
        return [parse_reply(
            reply, allowed, tools_by_name=tools_by_name)]

    text = strip_reasoning(reply or "")
    blocks = ACTION_BLOCK.findall(text)
    unfenced = False
    if not blocks:
        # The same kindness the single-call parser extends, and for the same
        # measured reason: a correct call alone on the last line is not
        # ambiguous, and rejecting it measures markdown rather than
        # navigation. A bare line carries one call by construction, so an
        # unfenced reply is a chunk of one.
        bare = _bare_call(text, allowed)
        if bare is None:
            if looks_cut_off(text):
                raise TruncatedReply(
                    "Your reply stopped before it named an action -- it ran out "
                    "of room. Put the action first next time, and keep the "
                    "reasoning to one line."
                )
            raise FormatError(
                "No action found. End your reply with a fenced block "
                f"containing up to {max_calls} calls, one per line, and "
                "nothing else."
            )
        blocks, unfenced = [bare], True
    if len(blocks) > 1:
        # Still one block. Chunking says a block may hold several calls; it
        # does not say a reply may hold several blocks, and the second block
        # is overwhelmingly a rehearsal rather than a plan.
        raise FormatError(
            f"Found {len(blocks)} action blocks. Put every call in ONE block, "
            f"up to {max_calls} of them, one per line."
        )
    calls = split_calls(blocks[0].strip())
    if not calls:
        raise FormatError(
            f"Could not read an action from {blocks[0].strip()[:80]!r}. "
            "Use the form tool_name(arguments)."
        )
    if len(calls) > max_calls:
        raise ChunkTooLong(len(calls), max_calls)

    thought = ""
    head = text.split("```")[0].strip()
    if head:
        thought = head[-600:]
    raw = blocks[0].strip()
    parsed: list[Call] = []
    for name, raw_args in calls:
        tool, args, kwargs = build_call(
            name, raw_args, allowed, tools_by_name=tools_by_name)
        parsed.append(ParsedAction(tool=tool, args=args, kwargs=kwargs,
                                   thought=thought, raw=raw, unfenced=unfenced))
    return parsed


def chunk_info(log: TurnLog | None) -> dict[str, int]:
    """``chunk_len`` and ``chunk_executed`` for any turn, chunked or not.

    A stock single-call turn is a chunk of one which executed if it reached
    the world at all, so a trainer reading these two keys never has to know
    which session produced the log -- and a log parser written against a
    chunked run keeps working against an unchunked one.
    """
    if log is None:
        return {"chunk_len": 0, "chunk_executed": 0}
    reached_the_world = 1 if log.status in ("accepted", "rejected") else 0
    return {
        "chunk_len": int(getattr(log, "chunk_len", 1)),
        "chunk_executed": int(getattr(log, "chunk_executed", reached_the_world)),
    }


class ChunkedCourierSession(CourierSession):
    """A ``CourierSession`` whose turn may carry up to ``action_chunk`` calls.

    At ``action_chunk == 1`` this is the stock session: same prompt bytes,
    same ``TurnLog`` class, same dispatch, by delegating rather than by
    reimplementing.
    """

    def __init__(self, env: Any, *, action_chunk: int = 1, **kwargs: Any):
        chunk = int(action_chunk)
        if chunk < 1:
            raise ValueError(
                f"action_chunk is how many calls a turn may carry, so it is at "
                f"least 1; got {action_chunk!r}")
        # Before ``super().__init__``: the base constructor renders the system
        # prompt (to check that everything it advertises is callable), and the
        # prompt it renders depends on this.
        self.action_chunk = chunk
        super().__init__(env, **kwargs)

    # ── what the courier is shown ────────────────────────────────────────────

    def system_prompt(self) -> str:
        from embodiedbench.agent.courier.prompts import render_special_rules
        # narration is read off the environment, exactly as the stock session
        # reads it, and for the reason its comment gives: a prompt that
        # promises narrated lights to a courier whose lights are only in the
        # pictures teaches a rule that does not hold. Overriding this method
        # without carrying that argument silently pinned every chunked run to
        # narration="none" -- the two settings would then differ by whether
        # chunking happened to be on, which is not a difference anybody chose.
        # The special-rules block rides along for the same reason: a chunked
        # shift under a constraint flag must be taught the same rule the
        # stock session teaches.
        return build_system_prompt(
            city=self.city, tools=self.tools,
            narration=getattr(self.env, "narration", "none"),
            action_chunk=self.action_chunk,
            special_rules=render_special_rules(self._active_constraints()))

    # ── one turn ─────────────────────────────────────────────────────────────

    def step(self, reply: str) -> TurnLog:
        """Parse one reply into up to K calls, run them in order, charge them.

        At K=1 the stock turn runs, untouched: chunking that is switched off
        must not be a different code path with the same intentions.
        """
        if self.action_chunk == 1:
            return super().step(reply)
        return self._chunked_step(reply)

    def _chunked_step(self, reply: str) -> ChunkTurnLog:
        observation = self._consume_observation()
        turn = ChunkTurnLog(
            step=len(self.run.turns) + 1,
            prompt=observation.text,
            image_paths=observation.image_paths,
            reply=reply,
        )
        turn.frame_metadata = [
            frame.metadata_dict() for frame in observation.frames]
        turn.frame_yaws = list(self._frame_yaws)
        # The pose the turn STARTED at, which is where its photographs were
        # taken. Set on both session paths or the two produce different record
        # shapes -- and a reader of one would find a field the other lacks.
        try:
            turn.from_xy = [round(v, 1) for v in self.env.position()]
        except Exception:  # noqa: BLE001 — a recording never fails a turn
            turn.from_xy = None
        stopped = budget_exceeded(self.spend, self.budgets)
        if stopped:
            turn.status, turn.error = "truncated", stopped
            self._finish(stopped)
            self.run.turns.append(turn)
            return turn

        try:
            calls = parse_chunk(
                reply, set(self.allowed), self.action_chunk,
                tools_by_name=self._tools_by_name)
        except TruncatedReply as error:
            self.spend.truncated_replies += 1
            turn.status, turn.error = "truncated_reply", str(error)
            self.feedback = TRUNCATED_TEMPLATE.format(error=error)
            self.run.turns.append(turn)
            if budget_exceeded(self.spend, self.budgets) == "repeated_truncations":
                self._finish("repeated_truncations")
            return turn
        except FormatError as error:
            self.spend.format_errors += 1
            self.spend.consecutive_format_errors += 1
            turn.status, turn.error = "format_error", str(error)
            self.feedback = FORMAT_ERROR_TEMPLATE.format(
                error=error, example=self._reply_example())
            self.run.turns.append(turn)
            if budget_exceeded(self.spend, self.budgets) == "repeated_format_errors":
                self._finish("repeated_format_errors")
            return turn
        except ChunkTooLong as error:
            return self._refuse_over_long(turn, error)

        self.spend.consecutive_format_errors = 0
        if calls[0].unfenced:
            self.spend.unfenced_actions += 1
        turn.thought = calls[0].thought
        # Every call the reply named, in the grammar the prompt teaches. The
        # chunk rows say which of them actually happened.
        turn.action = "; ".join(call.render() for call in calls)
        # The first call's kind, because the first call is the one a chunk
        # always makes: whatever else a chunk turns out to be, it is at least
        # that. The per-call detail carries the rest.
        turn.tool_kind = self._tools_by_name[calls[0].tool].kind.value
        turn.chunk_len = len(calls)

        rows: list[dict[str, Any]] = []
        messages: list[str] = []
        executed = 0
        aborted_at: int | None = None
        stop_code = ""
        finish_reason = ""
        refused: Any = None
        refused_tool = ""
        # RECEDING HORIZON: the reply names K waypoints, and exactly ONE of
        # them happens.
        #
        # The earlier reading was that a chunk buys K actions for one prompt,
        # which is what the throughput argument for chunking wanted. It is the
        # wrong contract for a plan. The later waypoints are named relative to
        # positions the courier has not reached yet, so executing them commits
        # to a forecast the world has already had a chance to falsify: the
        # first walk stops a metre short, or ends against a wall, and the
        # second and third were computed from somewhere the courier is not.
        #
        # So they are a PLAN, not a queue. Naming three makes the model think a
        # few steps ahead and shows that thinking in the trajectory; only the
        # first is carried out, and the next turn re-plans from where the walk
        # actually left it. The turn budget is sized for one action a turn
        # because that is what a turn now buys.
        planned = calls[1:]
        for index, action in enumerate(calls[:1]):
            # Before the call, never after: an exhausted budget must not leave
            # a half-applied transition for the trajectory to explain. This is
            # the same rule ``budget_exceeded`` is documented with, applied
            # per call because a call is what a budget counts.
            stopped = budget_exceeded(self.spend, self.budgets)
            if stopped:
                aborted_at, stop_code = index, stopped
                break
            before = self.env.sim_seconds
            self._leaving = self.env.node_id
            # Where this call is being judged FROM. The second and third
            # waypoints of a chunk are named relative to positions the courier
            # has not reached yet, so a trace that records only the reply
            # cannot say what each call was actually asking for.
            from_xy = None
            if hasattr(self.env, "position"):
                try:
                    from_xy = [round(v, 1) for v in self.env.position()]
                except Exception:  # noqa: BLE001 — a trace never fails a turn
                    from_xy = None
            outcome = self._execute(action)
            seconds = self.env.sim_seconds - before
            executed += 1
            self.spend.steps += 1
            self.spend.tool_calls += 1
            self.spend.sim_seconds += seconds
            self.run.total_reward += outcome.reward
            rows.append({
                "action": action.render(),
                "status": "accepted" if outcome.ok else "rejected",
                "code": "" if outcome.ok else (outcome.code or "refused"),
                "sim_seconds": seconds,
                "reward": outcome.reward,
                "from_xy": from_xy,
                # What the world said back to THIS call. The turn-level
                # feedback is the calls' messages joined, so by the time a
                # reader sees it there is no telling which call said what.
                "message": outcome.message,
            })
            messages.append(outcome.message)

            if outcome.moved:
                self._record_arrival()
            self._remember(action, outcome)
            if not outcome.ok and self._leaving is not None and action.args:
                self.memory.refused(
                    self._leaving, str(action.args[0]),
                    str(action.args[1]) if len(action.args) > 1 else "",
                    outcome.code or "refused")
            self._track_repeat(action.render(), outcome.ok)

            if not outcome.ok:
                aborted_at, stop_code, refused = index, rows[-1]["code"], outcome
                refused_tool = action.tool
                break
            if outcome.finished or self.env.shift_over:
                aborted_at = index
                finish_reason = "delivered" if outcome.finished else "shift_over"
                stop_code = finish_reason
                break

        # The rest of the plan, recorded and not run. Named rather than
        # silently missing: a plan that was never made and a plan that was
        # made and superseded look identical in a log that keeps only what
        # happened, and the difference is the whole reason for asking for
        # three.
        for action in planned:
            rows.append({"action": action.render(), "status": "planned",
                         "code": "", "sim_seconds": 0.0, "reward": 0.0,
                         "from_xy": None, "message": ""})

        turn.chunk = rows
        turn.chunk_executed = executed
        turn.chunk_aborted_at = aborted_at
        turn.sim_seconds = sum(row["sim_seconds"] for row in rows)
        turn.reward = sum(row["reward"] for row in rows)
        turn.status = "rejected" if refused is not None else "accepted"
        turn.error = stop_code if (refused is not None or not finish_reason) else ""
        turn.memory = self.memory.to_dict()
        self.run.turns.append(turn)
        self.feedback = self._chunk_feedback(
            rows, messages, refused, refused_tool)

        if finish_reason:
            self._finish(finish_reason)
        elif self._repeats >= self.STUCK_REPEATS:
            self._finish("stuck")
        else:
            stopped = budget_exceeded(self.spend, self.budgets)
            if stopped:
                self._finish(stopped)
        return turn

    # ── the refusals a chunk adds ────────────────────────────────────────────

    def _refuse_over_long(self, turn: ChunkTurnLog,
                          error: ChunkTooLong) -> ChunkTurnLog:
        """Charge an over-long chunk the way the world charges any refusal.

        Through ``CourierEnv._refuse`` -- the same turn, the same rejected
        action, the same five-second floor -- rather than as a free re-prompt.
        A refusal that costs nothing is a rangefinder: the courier would learn
        that the cheapest way to find out what the limit is, or to buy thinking
        time, is to overflow it, and neither currency this benchmark reports
        would show it happening.
        """
        from embodiedbench.runtime.city.courier_env import StepOutcome

        before = self.env.sim_seconds
        self.env._refuse(StepOutcome(ok=False, code=error.code,
                                     message=str(error)))
        seconds = self.env.sim_seconds - before
        self.spend.consecutive_format_errors = 0
        self.spend.steps += 1
        self.spend.tool_calls += 1
        self.spend.sim_seconds += seconds
        turn.status, turn.error = "rejected", error.code
        turn.sim_seconds = seconds
        # What the reply asked for, against a ``chunk`` that is empty because
        # nothing was built: see ``ChunkTurnLog``.
        turn.chunk_len = error.found
        turn.chunk_aborted_at = 0
        turn.memory = self.memory.to_dict()
        self.feedback = str(error)
        self.run.turns.append(turn)
        # Named by the fault rather than by a call, because there is no call:
        # a courier that answers the limit by re-sending the same over-long
        # chunk is as stuck as one repeating a refused walk, and the same rule
        # should end it.
        self._track_repeat(f"<{error.code}:{error.found}>", ok=False)
        if self._repeats >= self.STUCK_REPEATS:
            self._finish("stuck")
        else:
            stopped = budget_exceeded(self.spend, self.budgets)
            if stopped:
                self._finish(stopped)
        return turn

    # ── bookkeeping ──────────────────────────────────────────────────────────

    def _track_repeat(self, action: str, ok: bool) -> None:
        """The stock stuck detector, fed one call at a time.

        Kept here rather than reached for in the base class because the base
        class runs it once a turn over one call, and a chunk has to run it once
        a call over several -- the same rule, applied at the resolution
        chunking made the actions arrive in.
        """
        if not ok and action == self._last_refused:
            self._repeats += 1
        else:
            self._repeats = 0 if ok else 1
        self._last_refused = None if ok else action

    def _chunk_feedback(self, rows: list[dict[str, Any]], messages: list[str],
                        refused: Any, refused_tool: str = "") -> str:
        """What the next prompt says about the step that was taken.

        Says which one happened and which were the plan, because the courier
        wrote them as one reply and only one of them is a fact. A courier told
        "you did all three" plans its next turn from a position it is not at.
        """
        lines, done = [], 0
        for index, row in enumerate(rows):
            if row["status"] in ("planned", "dropped"):
                lines.append(f"  {index + 1}. {row['action']} — NOT CARRIED OUT")
            else:
                said = messages[done] if done < len(messages) else ""
                done += 1
                lines.append(f"  {index + 1}. {row['action']} — {said}")
        rest = sum(1 for row in rows if row["status"] in ("planned", "dropped"))
        many = "the call" if rest == 1 else f"the {rest} calls"
        tail = ""
        if rest:
            tail = (
                f"\nOnly the first call is ever carried out, so {many} after "
                "it did not happen. They were your plan, and the step you just "
                "took is the only thing that has changed the world -- work the "
                "next one out again from where you are now, which is what the "
                "turn above tells you."
            )
        if refused is not None:
            if refused_tool == "walk_to_pixel":
                rejected_template = REJECTED_TEMPLATE_PIXEL
            elif refused_tool in {"walk_to", "follow_street"}:
                rejected_template = REJECTED_TEMPLATE
            else:
                rejected_template = REJECTED_TEMPLATE_ACTION
            tail += "\n\n" + rejected_template.format(reason=refused.message)
        return CHUNK_FEEDBACK_TEMPLATE.format(
            count=len(rows), lines="\n".join(lines), tail=tail).rstrip()
