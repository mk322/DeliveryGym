"""The courier loop: how a turn is assembled, parsed, executed and charged.

Shape borrowed from mini-SWE-agent, because the shape is the good part: prompts
are data rather than code, the model emits exactly one action per turn inside a
fenced block, a malformed reply is re-prompted rather than fatal, and every limit
is explicit and checked in one place. What changes is everything embodied --
a courier has a tool set rather than a shell, carries memory between turns, and
acts in a world where a wrong move costs simulated time it cannot get back.

One turn, in order:

    1. sense       the runtime's state becomes an observation: where am I, what
                   streets leave this junction, what does the job slip say
    2. remember    memory is updated with the arrival, then rendered into the
                   prompt. Recording happens before rendering so the current
                   junction is already in the trail the model reads.
    3. compose     system prompt (once) + memory + observation + images +
                   any feedback from the previous turn's failure
    4. decide      the model returns THOUGHT then exactly one fenced action
    5. parse       into (tool, arguments), against the tool set this environment
                   actually enables. A parse failure is a FormatError.
    6. execute     dispatch to the runtime; a macro expands to several tool calls
                   and stops early if the situation changes
    7. charge      steps, simulated seconds, tool calls, output tokens
    8. record      the turn, its images, the raw reply, the typed result

Two failure modes are deliberately non-terminating, because ending an episode on
them would score the prompt rather than the policy:

FormatError        the reply had no action, or a malformed one. Re-prompt with
                   the format reminder. After ``max_format_errors`` in a row the
                   episode truncates -- a model that cannot emit the grammar is
                   not going to start.
RejectedAction     the action parsed but the world refused it (walked into a
                   wall, collected at the wrong address). The typed reason goes
                   into the next prompt. This is information, not an error, and
                   a courier gets it constantly.
"""

from __future__ import annotations

import json

import re
from dataclasses import dataclass, field
from typing import Any

from embodiedbench.agent.courier.memory import CourierMemory
from embodiedbench.agent.courier.skills import MACROS_BY_NAME, Macro
from embodiedbench.agent.courier.tools import Tool, ToolKind

# The action grammar. One fenced block, one call. Fenced rather than bare so a
# model that narrates around its action still parses, which mini-SWE-agent found
# necessary and which held here too.
ACTION_BLOCK = re.compile(r"```(?:action)?\s*\n?(.+?)\n?```", re.DOTALL)
# Where a call starts. Finding its *end* needs a scan, not a pattern -- see
# split_calls.
CALL_START = re.compile(r"([a-z_][a-z_0-9]*)\s*\(", re.IGNORECASE)


class FormatError(Exception):
    """The reply did not contain exactly one well-formed action."""


class TruncatedReply(FormatError):
    """The reply stops mid-sentence: the generation budget ran out.

    This is a configuration fault, not a model output. ``model_io`` already
    says so for the evaluation path -- it raises the budget and asks again
    rather than parsing the stump -- but the RL path has no requery, so a
    truncation arrived here as a format error and three in a row ended the
    episode. Measured on 64 held-out episodes: 11 died that way, every one at
    zero earnings, discarding 308 of their 440 remaining turns. That is the
    same size as the whole reported success rate.

    Worse, it lands asymmetrically. Validation decodes greedily and greedy is
    what degenerates into the repetition loops that exhaust the budget;
    training samples and never hits it (0 truncations in 1531 training turns).
    So the policy was losing a sixth of its score to a failure mode it
    received no gradient on.

    It is a FormatError subclass so that anything catching FormatError still
    catches this; what changes is the counter it charges and the words it gets
    back.
    """


class RejectedAction(Exception):
    """The action was understood but the world refused it."""

    def __init__(self, message: str, code: str = "rejected"):
        self.code = code
        super().__init__(message)


@dataclass
class Budgets:
    """Every limit in one place, so none is enforced twice or not at all."""

    steps: int = 120
    tool_calls: int | None = 160
    sim_seconds: float | None = 3600.0
    output_tokens: int | None = 60000
    max_format_errors: int = 3
    # Truncations get their own, looser budget. They are the harness's fault
    # and the reply the model was trying to give is unknown, so charging them
    # against the same three-strike count meant a generation cap could end a
    # shift that was going fine.
    max_truncated_replies: int = 8
    wall_seconds: float | None = 1800.0

    def to_dict(self) -> dict[str, Any]:
        return dict(self.__dict__)


@dataclass
class Spend:
    steps: int = 0
    tool_calls: int = 0
    sim_seconds: float = 0.0
    output_tokens: int = 0
    format_errors: int = 0
    consecutive_format_errors: int = 0
    # Replies that carried a correct call with no code fence. Accepted,
    # and counted, so format-following stays a reportable number rather
    # than a silent leniency.
    unfenced_actions: int = 0
    # Replies that stopped mid-sentence because the generation budget ran out.
    # Charged separately from format errors: see ``TruncatedReply``.
    truncated_replies: int = 0

    def to_dict(self) -> dict[str, Any]:
        return dict(self.__dict__)


@dataclass
class ParsedAction:
    """One decision, after parsing and before execution."""

    tool: str
    args: list[Any]
    kwargs: dict[str, Any]
    thought: str = ""
    raw: str = ""
    unfenced: bool = False

    def render(self) -> str:
        """The call as the prompt asks for it to be written.

        Double quotes, not Python's ``repr``. The prompt teaches
        walk_to("Rue de Grenelle", "east") and the transcript is read beside
        it -- a log that renders the same call with single quotes reports an
        action in a grammar the model was told not to use.
        """
        def one(value: Any) -> str:
            return json.dumps(value) if isinstance(value, str) else str(value)

        inner = ", ".join(
            [one(a) for a in self.args]
            + [f"{k}={one(v)}" for k, v in self.kwargs.items()]
        )
        return f"{self.tool}({inner})"


def _bare_call(text: str, allowed: set[str]) -> str | None:
    """The last line that is nothing but one call to a tool this env allows.

    Deliberately strict about *where*: the call has to be the whole line. A
    tool name mentioned inside a sentence -- "I will walk_to(2) once I am
    past the barrier" -- is the model narrating, not acting, and executing its
    narration is the guessing this parser exists to avoid.
    """
    for line in reversed([ln.strip() for ln in text.splitlines() if ln.strip()]):
        line = line.rstrip(".")
        calls = split_calls(line)
        if len(calls) != 1:
            continue
        name, _ = calls[0]
        if name.lower() not in allowed:
            continue
        # the call must be the entire line, not a fragment of prose
        if line.endswith(")") and line.lower().startswith(name.lower()):
            return line
    return None


#: A number argument that is an unevaluated sum: two numbers with an operator
#: between them. Deliberately narrow -- it only has to recognise the shape the
#: refusal below names, and anything it does not match falls through to the
#: general "that is not a number" message.
_ARITHMETIC = re.compile(r"^\s*[-+]?[\d.]+\s*[-+*/]\s*[-+]?[\d.]+")

REASONING_START = "<think>"
REASONING_END = "</think>"


# A sentence the model finished ends in one of these. A generation that ran
# out of budget almost never does -- it stops inside a word or a clause.
_ENDINGS = (".", "!", "?", "`", ")", '"', "'", ":", "\u3002", "\uff1f", "\uff01")
_SHORTEST_PLAUSIBLE_TRUNCATION = 120


def looks_cut_off(reply: str) -> bool:
    """Did the generation stop because it ran out of room?

    Two signals, both from the text alone, because the harness that parses a
    reply does not always know the budget that produced it. An odd number of
    fences means one was opened and never closed. Otherwise, a reply that ends
    without any terminal punctuation stopped mid-sentence: on the held-out set
    this separates the 26 replies killed by the generation cap from the 7 that
    genuinely said nothing actionable.

    Deliberately conservative in the other direction: a reply that ends cleanly
    but names no action is a format error, not a truncation, and calling it a
    truncation would forgive a model that simply did not answer.
    """
    text = reply.strip()
    if not text:
        return False
    if text.count("```") % 2 == 1:
        return True
    # Too short to have run out of a 1024-token budget. On the held-out set the
    # replies killed by the cap have a median length of 3365 characters, while
    # the ones that simply forgot the action are 60 to 100 -- a finished
    # sentence like "I am at the pickup address. I should collect the order."
    # Without a floor, any short unpunctuated reply is excused as a truncation,
    # and the leniency stops distinguishing anything.
    if len(text) < _SHORTEST_PLAUSIBLE_TRUNCATION:
        return False
    return not text.endswith(_ENDINGS)


def strip_reasoning(reply: str) -> str:
    """Everything after the model stops thinking.

    A reasoning model emits its chain of thought first, and inside that thought
    it writes candidate calls in fenced blocks -- exactly the shape this parser
    looks for. Qwen3.5-9B produced two fenced blocks on every single turn, one
    rehearsed inside <think> and one real answer after it, and the parser
    rejected all of them as "Found 2 action blocks": 66 format errors in 80
    turns, 82.5%, on a model whose answers were in fact perfectly formed.

    The thought is not the action. Only what follows the closing tag is the
    reply, which is also how the serving stack and the chat template treat it.
    A reply with no reasoning section is returned unchanged, so this costs
    nothing for models that do not think out loud.
    """
    end = reply.rfind(REASONING_END)
    if end != -1:
        return reply[end + len(REASONING_END):]
    if REASONING_START in reply:
        # Opened a thought and never closed it: the model was cut off before it
        # answered. The calls inside are rehearsal, and running one is exactly
        # the guess this parser exists to refuse -- it would execute a move the
        # model was in the middle of arguing itself out of.
        return ""
    return reply


def parse_reply(
    reply: str,
    allowed: set[str],
    *,
    tools_by_name: dict[str, Tool] | None = None,
) -> ParsedAction:
    """Pull exactly one tool call out of a model reply.

    Raises ``FormatError`` rather than guessing. Guessing is how an agent ends up
    executing something it did not choose: an earlier text policy fell back to
    WAIT on any unparseable reply, which spent a turn and told the model nothing
    about what went wrong.
    """
    text = strip_reasoning(reply or "")
    blocks = ACTION_BLOCK.findall(text)
    unfenced = False
    if not blocks:
        # A bare call on its own line counts. The fence exists to make the
        # action unambiguous, and `walk_to(2)` alone on the last line is not
        # ambiguous -- so rejecting it measures markdown, not navigation.
        #
        # This is not a hypothetical kindness. Qwen2-VL-2B ends every reply with
        # exactly the right call and no fence, so all three of its opening turns
        # were format errors and the three-strikes rule ended the episode before
        # it had acted once. Every seed scored 0.0 for a reason that had nothing
        # to do with the city. A benchmark that discards a correct action
        # conflates instruction-following with the thing it means to measure.
        #
        # It is counted, not forgiven: ``Spend.unfenced_actions`` tracks it, so
        # "how well does this model follow the reply format" stays answerable.
        bare = _bare_call(text, allowed)
        if bare is None:
            if looks_cut_off(text):
                raise TruncatedReply(
                    "Your reply stopped before it named an action -- it ran out "
                    "of room. Put the action first next time, and keep the "
                    "reasoning to one line."
                )
            raise FormatError(
                "No action found. End your reply with a fenced block containing "
                "exactly one call. The block holds the call and nothing else."
            )
        blocks, unfenced = [bare], True
    if len(blocks) > 1:
        raise FormatError(
            f"Found {len(blocks)} action blocks. Give exactly one action per turn."
        )
    calls = split_calls(blocks[0].strip())
    if not calls:
        raise FormatError(
            f"Could not read an action from {blocks[0].strip()[:80]!r}. "
            "Use the form tool_name(arguments)."
        )
    if len(calls) > 1:
        raise FormatError(
            f"Found {len(calls)} calls in one block: "
            f"{', '.join(c[0] for c in calls)}. Give exactly one."
        )
    called, raw_args = calls[0]
    name, args, kwargs = build_call(
        called, raw_args, allowed, tools_by_name=tools_by_name)

    thought = ""
    head = text.split("```")[0].strip()
    if head:
        thought = head[-600:]
    return ParsedAction(tool=name, args=args, kwargs=kwargs, thought=thought,
                        raw=blocks[0].strip(), unfenced=unfenced)


def build_call(
    name: str,
    raw_args: str,
    allowed: set[str],
    *,
    tools_by_name: dict[str, Tool] | None = None,
) -> tuple[str, list[Any], dict[str, Any]]:
    """One scanned ``name(raw_args)``, held to the tool set and to the types.

    Lifted out of ``parse_reply`` unchanged -- same wording, same order of
    checks -- so that the chunked parser in ``chunk.py`` can hold each call of
    a multi-call block to exactly the rules a single-call block is held to.
    Written twice it would be two grammars, and the difference between them
    would be found by a policy rather than by a test.
    """
    name = name.lower()
    if name not in allowed:
        raise FormatError(
            f"{name!r} is not something you can do here. Available: "
            f"{', '.join(sorted(allowed))}."
        )

    args: list[Any] = []
    kwargs: dict[str, Any] = {}
    for piece in _split_args(raw_args):
        if not piece:
            continue
        if "=" in piece and not piece.lstrip().startswith(('"', "'")):
            key, _, value = piece.partition("=")
            kwargs[key.strip()] = _coerce(value.strip())
        else:
            args.append(_coerce(piece.strip()))

    _check_argument_types(
        name, args, kwargs,
        tool=(tools_by_name or {}).get(name) if tools_by_name is not None else None,
        use_global=tools_by_name is None,
    )
    return name, args, kwargs


def _check_argument_types(
    name: str,
    args: list[Any],
    kwargs: dict[str, Any],
    *,
    tool: Tool | None = None,
    use_global: bool = True,
) -> None:
    """Hold the caller to the types the prompt states.

    The prompt says "any text argument goes in double quotes", and
    ``check_map(42)`` was dispatched anyway: the address lookup then searched for
    the *integer* 42, failed, and charged 5 s for a refusal caused by a rule the
    runtime had declined to enforce. A rule stated to the agent and not enforced
    is worse than no rule -- it teaches that the prompt is approximate.
    """
    if use_global:
        from embodiedbench.agent.courier.tools import TOOLS_BY_NAME

        tool = TOOLS_BY_NAME.get(name)
    if tool is None:
        return
    supplied = list(zip(tool.params, args)) + [
        (param, kwargs[param.name]) for param in tool.params if param.name in kwargs
    ]
    for param, value in supplied:
        if param.type == "str" and not isinstance(value, str):
            raise FormatError(
                f"{name}({param.name}=…) takes text, so it goes in double quotes: "
                f'{name}("{value}") rather than {name}({value}).'
            )
        if param.type == "int" and not isinstance(value, int):
            raise FormatError(
                f"{name}({param.name}=…) takes a whole number without quotes, "
                f"like {tool.example or name + '(2)'}."
            )
        # A coordinate arrives as something other than a number often enough
        # to be worth its own messages, and the two ways it happens want
        # different answers.
        #
        # Measured on Qwen3-VL-4B over 205 coordinate turns: 62 of 81 format
        # errors were an unevaluated SUM -- ``walk_to_xy(-53.2, 297.7 - 18)``,
        # the model saying "eighteen metres west of where I am" in the most
        # direct way it knows. Answering that with a sentence about quotes is
        # advice it cannot act on, and it repeated the same reply until the
        # three-strike rule ended the episode. The sum is still refused --
        # working the position out is the task, and a harness that does the
        # subtraction is measuring something easier than it claims -- but the
        # refusal has to say which thing went wrong.
        if param.type == "number" and not isinstance(value, (int, float)):
            if isinstance(value, str) and _ARITHMETIC.search(value):
                raise FormatError(
                    f"{name}({param.name}=…) got the sum {value.strip()!r} "
                    "rather than a number. Work it out and type the answer: "
                    f"{tool.example or name + '(1.0, 2.0)'}."
                )
            raise FormatError(
                f"{name}({param.name}=…) takes a number, with no quotes around "
                f"it: {tool.example or name + '(1.0, 2.0)'}."
            )


def split_calls(text: str) -> list[tuple[str, str]]:
    """Every ``name(args)`` in a block, as ``(name, raw_args)``.

    A regex cannot do this correctly. ``[^)]*`` stops at the first bracket and
    silently truncated ``check_map("12 Avenue de Rivoli (2)")`` into a corrupt
    address -- 27 of 86 street names carry a "(2)" suffix, so that was routine,
    not an edge case. Making it greedy to the last bracket fixed the quoting and
    broke the opposite check: ``walk_to(3)`` followed by ``look(2)`` then matched
    as a single call, so a reply containing two actions was accepted as one.

    Scanning is the only thing that satisfies both. Track quote state and bracket
    depth, and a close bracket only ends the call when it is unquoted and at
    depth zero.
    """
    calls: list[tuple[str, str]] = []
    position = 0
    while True:
        match = CALL_START.search(text, position)
        if match is None:
            return calls
        depth, quote, index = 1, "", match.end()
        while index < len(text) and depth:
            char = text[index]
            if quote:
                if char == quote:
                    quote = ""
            elif char in "\"'":
                quote = char
            elif char in "([{":
                depth += 1
            elif char in ")]}":
                depth -= 1
            index += 1
        if depth:
            # Unbalanced: report the fragment so the error names what was seen.
            calls.append((match.group(1), text[match.end():]))
            return calls
        calls.append((match.group(1), text[match.end():index - 1]))
        position = index


def _split_args(raw: str) -> list[str]:
    out, depth, current, quote = [], 0, [], ""
    for ch in raw:
        if quote:
            current.append(ch)
            if ch == quote:
                quote = ""
            continue
        if ch in "\"'":
            quote = ch
            current.append(ch)
        elif ch in "[({":
            depth += 1
            current.append(ch)
        elif ch in "])}":
            depth -= 1
            current.append(ch)
        elif ch == "," and depth == 0:
            out.append("".join(current))
            current = []
        else:
            current.append(ch)
    out.append("".join(current))
    return [piece.strip() for piece in out]


def _coerce(value: str) -> Any:
    text = value.strip()
    if len(text) >= 2 and text[0] == text[-1] and text[0] in "\"'":
        return text[1:-1]
    try:
        return int(text)
    except ValueError:
        pass
    try:
        return float(text)
    except ValueError:
        pass
    return text


@dataclass
class TurnLog:
    """Everything about one turn, for replay and for the trajectory."""

    step: int
    prompt: str
    image_paths: list[str] = field(default_factory=list)
    frame_metadata: list[dict[str, Any]] = field(default_factory=list)
    reply: str = ""
    thought: str = ""
    action: str = ""
    tool_kind: str = ""
    expanded: list[str] = field(default_factory=list)
    status: str = "accepted"
    error: str = ""
    sim_seconds: float = 0.0
    reward: float = 0.0
    memory: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return dict(self.__dict__)


@dataclass
class CourierRun:
    """The result of one shift."""

    turns: list[TurnLog] = field(default_factory=list)
    spend: Spend = field(default_factory=Spend)
    budgets: Budgets = field(default_factory=Budgets)
    memory: CourierMemory = field(default_factory=CourierMemory)
    finished: bool = False
    termination_reason: str = ""
    total_reward: float = 0.0

    def to_dict(self) -> dict[str, Any]:
        return {
            "turns": [t.to_dict() for t in self.turns],
            "spend": self.spend.to_dict(),
            "budgets": self.budgets.to_dict(),
            "memory": self.memory.to_dict(),
            "finished": self.finished,
            "termination_reason": self.termination_reason,
            "total_reward": round(self.total_reward, 4),
            "turn_count": len(self.turns),
        }


def budget_exceeded(spend: Spend, budgets: Budgets) -> str | None:
    """Which limit stopped the episode, checked in one place.

    Checked *before* an action executes, so an exhausted budget never leaves a
    half-applied transition the trajectory then has to explain.
    """
    if spend.steps >= budgets.steps:
        return "step_budget_exhausted"
    if budgets.tool_calls is not None and spend.tool_calls >= budgets.tool_calls:
        return "tool_call_budget_exhausted"
    if budgets.sim_seconds is not None and spend.sim_seconds >= budgets.sim_seconds:
        return "sim_time_budget_exhausted"
    if budgets.output_tokens is not None and spend.output_tokens >= budgets.output_tokens:
        return "output_token_budget_exhausted"
    if spend.consecutive_format_errors >= budgets.max_format_errors:
        return "repeated_format_errors"
    if spend.truncated_replies >= budgets.max_truncated_replies:
        return "repeated_truncations"
    return None


def expansion_limit(macro: Macro, requested: int) -> int:
    """Clamp a macro to its declared maximum, reporting rather than truncating silently."""
    return max(1, min(int(requested), macro.max_expansion))
