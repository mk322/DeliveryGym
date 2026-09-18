"""Talking to a model, without the benchmark learning that model's habits.

Three times in one evaluation session a model was recorded as unable to follow
the reply contract when the truth was a setting on this side of the wire:

* ``max_new_tokens=48`` cut Qwen3-VL-4B's reasoning off before its fenced call
  and reported a held-out format score of 0.125 for a model that scores 1.0;
* ``--limit-mm-per-prompt 6`` against a turn that offers nine pictures returned
  HTTP 400, whose text was fed to the parser as though the model had said it,
  and killed 26 of 40 episodes as "format errors";
* Qwen3.5-9B rehearses candidate calls inside ``<think>`` in fenced blocks, so
  the parser found two action blocks and refused 66 of 80 turns from a model
  whose answers were all well formed.

Each was found by hand, after the number had already been believed once. The
pattern is the same every time: **a harness that only works for the model it was
debugged against measures its own configuration.** So the adaptation lives here,
once, rather than in the reader's head.

What this does, following mini-swe-agent's separation of a *model call* from an
*environment step*:

* **Unparseable output is requeried, not charged.** mini-swe-agent re-prompts on
  a format error and keeps the bad output in the conversation; the environment
  never sees it. Here likewise: the turn is retried with the error fed back, up
  to ``max_requeries``, and only a turn that never parses reaches the world. The
  count is reported rather than hidden, so a model that needs three attempts
  every turn still looks worse than one that needs none.
* **Reasoning is separated from the answer**, whichever way the model marks it:
  a ``reasoning_content`` field, a ``reasoning`` field, or inline ``<think>``.
* **Truncation is a configuration fault, not a model output.** ``finish_reason
  == "length"`` means the budget was too small; the reply is not parsed, the
  budget is raised, and the call is retried.
* **Transport failures are never mistaken for replies.** They raise, carrying
  the server's own message.
"""

from __future__ import annotations

import json
import re
import time
import urllib.error
import urllib.request
from dataclasses import dataclass, field
from typing import Any, Callable

# Every convention seen in the wild for "this part was me thinking". Matched
# non-greedily and only when closed: an unterminated block means the model was
# cut off mid-thought, and the calls inside it are the ones it was arguing
# itself out of.
THINK_BLOCKS = (
    ("<think>", "</think>"),
    ("<thinking>", "</thinking>"),
    ("<|thinking|>", "<|/thinking|>"),
    ("<reasoning>", "</reasoning>"),
)


# The server knows its own limits and says so in plain text when they are
# broken. Reading that back is what keeps this layer model-agnostic: nothing
# here has to be told a context size per model.
# vLLM alone phrases this two ways depending on where the check fires:
#   "This model's maximum context length is 10240 tokens. However, you
#    requested 10000 output tokens and your prompt contains 8000 tokens"
#   "Input length (10384) exceeds model's maximum context length (10240)."
# Matching only the first let the second through as a fatal error and ended
# Qwen3.5-9B episodes after a mean of 6.6 turns. Reading the server's limits
# only works if it covers what the server actually says.
CONTEXT_LIMIT = re.compile(
    r"maximum context length is (\d+) tokens|maximum context length \((\d+)\)", re.I)
PROMPT_TOKENS = re.compile(
    r"prompt contains (\d+) tokens|Input length \((\d+)\)", re.I)


def _first_group(match: re.Match[str] | None) -> int | None:
    """The one group that matched, whichever alternative it came from."""
    if match is None:
        return None
    for value in match.groups():
        if value is not None:
            return int(value)
    return None


@dataclass
class CallStats:
    """What it cost to get an action out of the model, in the model's own terms."""

    calls: int = 0
    requeries: int = 0
    truncations: int = 0
    budget_clamps: int = 0
    history_drops: int = 0
    transport_retries: int = 0
    timeouts: int = 0
    unparseable: int = 0

    def as_dict(self) -> dict[str, int]:
        return {
            "model_calls": self.calls,
            "requeries": self.requeries,
            "truncations": self.truncations,
            "budget_clamps": self.budget_clamps,
            "history_drops": self.history_drops,
            "transport_retries": self.transport_retries,
            "timeouts": self.timeouts,
            "unparseable_turns": self.unparseable,
        }


def split_reasoning(message: dict[str, Any]) -> tuple[str, str]:
    """``(answer, reasoning)`` from one chat-completion message.

    The serving stack may hand the thought back in its own field, in which case
    ``content`` is already only the answer. Otherwise the thought is inline and
    has to be cut off the front. Both happen, on the same model, depending on
    whether a reasoning parser is configured -- so neither can be assumed.
    """
    content = message.get("content") or ""
    field_reasoning = message.get("reasoning_content") or message.get("reasoning") or ""

    for open_tag, close_tag in THINK_BLOCKS:
        end = content.rfind(close_tag)
        if end != -1:
            return content[end + len(close_tag):].strip(), content[:end]
        if open_tag in content:
            # Opened and never closed: there is no answer in here at all.
            return "", content
    return content.strip(), field_reasoning


def looks_truncated(choice: dict[str, Any]) -> bool:
    return choice.get("finish_reason") == "length"


class ModelClient:
    """An OpenAI-compatible endpoint, made to behave the same for every model.

    ``parse`` is the benchmark's own reply parser. It is passed in rather than
    imported so this module stays a transport concern and the reply contract
    stays where it is defined.
    """

    def __init__(
        self,
        endpoint: str,
        model: str,
        *,
        max_tokens: int = 2048,
        max_requeries: int = 3,
        max_transport_retries: int = 2,
        max_timeout_retries: int = 5,
        truncation_growth: float = 2.0,
        max_tokens_ceiling: int = 8192,
        timeout: float = 600.0,
        api_key: str | None = None,
    ) -> None:
        self.endpoint = endpoint
        self.model = model
        # For hosted OpenAI-compatible endpoints. Local servers ignore it, so
        # None costs nothing; without it every hosted model in a baseline
        # table would need its own transport fork.
        self.api_key = api_key
        self.max_tokens = max_tokens
        self.max_requeries = max_requeries
        self.max_transport_retries = max_transport_retries
        self.max_timeout_retries = max_timeout_retries
        self.truncation_growth = truncation_growth
        self.max_tokens_ceiling = max_tokens_ceiling
        self.timeout = timeout
        self.stats = CallStats()
        self.last_reasoning = ""

    # ── transport ────────────────────────────────────────────────────────────

    def _post(self, payload: dict[str, Any]) -> dict[str, Any]:
        headers = {"Content-Type": "application/json"}
        if self.api_key:
            headers["Authorization"] = f"Bearer {self.api_key}"
        request = urllib.request.Request(
            self.endpoint, data=json.dumps(payload).encode(),
            headers=headers, method="POST")
        try:
            with urllib.request.urlopen(request, timeout=self.timeout) as response:
                return json.loads(response.read().decode())
        except urllib.error.HTTPError as error:
            # The server's own words. "HTTP Error 400: Bad Request" on its own
            # sent a previous investigation looking at the model for a day.
            detail = error.read().decode(errors="replace")[:500]
            raise RuntimeError(f"HTTP {error.code}: {detail}") from None

    def _post_with_retries(self, payload: dict[str, Any]) -> dict[str, Any]:
        last: Exception | None = None
        attempt = 0
        while True:
            try:
                return self._post(payload)
            except Exception as error:  # noqa: BLE001
                last = error
                # A 4xx is a request this harness built wrongly; retrying it
                # unchanged just spends the clock. Only transient faults are
                # worth another go.
                # A 408 or 429 is the server asking for patience, not a
                # malformed request; a hosted API's first rate-limit used to
                # end the episode as an infrastructure error.
                if (isinstance(error, RuntimeError) and str(error).startswith("HTTP 4")
                        and not str(error).startswith(("HTTP 408", "HTTP 429"))):
                    raise
                # A timeout is a busy server, not a broken request, and it gets
                # its own larger allowance. Two evaluations sharing one server
                # doubled latency past the deadline and killed 16 of 40
                # episodes outright -- a queueing artefact recorded as the
                # episode ending, which is the same class of mistake as an
                # HTTP error recorded as the model's reply.
                timed_out = isinstance(error, TimeoutError) or "timed out" in str(error)
                allowance = self.max_timeout_retries if timed_out else self.max_transport_retries
                if attempt >= allowance:
                    break
                if timed_out:
                    self.stats.timeouts += 1
                else:
                    self.stats.transport_retries += 1
                time.sleep(min(2 ** attempt, 30))
                attempt += 1
        raise last  # type: ignore[misc]

    # ── one turn ─────────────────────────────────────────────────────────────

    def act(
        self,
        messages: list[dict[str, Any]],
        parse: Callable[[str], Any],
        *,
        on_requery: Callable[[str, str], None] | None = None,
        temperature: float = 0.0,
    ) -> tuple[str, Any | None, list[str]]:
        """Get one *parseable* action, or report that none was forthcoming.

        Returns ``(reply, parsed, rejected)``: the reply that finally parsed,
        what it parsed to, and every reply that did not. ``parsed`` is None when
        every attempt failed, and the caller decides what a turn like that costs
        -- this layer does not step the world.
        """
        budget = self.max_tokens
        conversation = list(messages)
        rejected: list[str] = []

        for attempt in range(self.max_requeries + 1):
            try:
                body = self._post_with_retries({
                    "model": self.model, "messages": conversation,
                    "max_tokens": budget, "temperature": temperature,
                })
            except RuntimeError as error:
                # Raising the budget after a truncation can walk it straight
                # into the context limit: escalating to 10000 output tokens on
                # a 10240-token model leaves 240 for the prompt, and every turn
                # 400s. The server names both numbers in the refusal, so use
                # them rather than guessing a ceiling per model.
                limit = _first_group(CONTEXT_LIMIT.search(str(error)))
                used = _first_group(PROMPT_TOKENS.search(str(error)))
                room = None
                if limit is not None and used is not None:
                    room = limit - used - 64
                elif limit is not None:
                    room = limit // 4
                if room is not None and 128 <= room < budget:
                    self.stats.budget_clamps += 1
                    self.max_tokens_ceiling = min(self.max_tokens_ceiling, room)
                    budget = room
                    continue
                # No room left to give back: the prompt itself is too long for
                # this model. Shed the oldest exchange and try again. A
                # reasoning model behind a 10k context runs out this way after
                # a handful of turns, and the alternative is a per-model
                # history setting -- another number to guess wrong.
                if limit is not None and len(conversation) > 2:
                    # Shed the oldest user/assistant exchange, never the
                    # system message in front of it: dropping conversation[0]
                    # took the tool manual with it, and every later turn of
                    # that episode ran without a rulebook.
                    has_system = conversation[0].get("role") == "system"
                    if has_system and len(conversation) <= 3:
                        raise
                    self.stats.history_drops += 1
                    conversation = (conversation[:1] + conversation[3:]
                                    if has_system else conversation[2:])
                    budget = min(budget, self.max_tokens)
                    continue
                raise
            self.stats.calls += 1
            choice = body["choices"][0]
            reply, reasoning = split_reasoning(choice.get("message") or {})
            self.last_reasoning = reasoning

            if looks_truncated(choice) and budget < self.max_tokens_ceiling:
                # Not a malformed answer -- an unfinished one. Parsing it would
                # record a configuration fault as a model failure.
                self.stats.truncations += 1
                budget = min(int(budget * self.truncation_growth), self.max_tokens_ceiling)
                continue

            try:
                return reply, parse(reply), rejected
            except Exception as error:  # noqa: BLE001 - parser raises its own type
                rejected.append(reply)
                if attempt >= self.max_requeries:
                    break
                self.stats.requeries += 1
                if on_requery is not None:
                    on_requery(reply, str(error))
                conversation = conversation + [
                    {"role": "assistant", "content": reply},
                    {"role": "user", "content":
                        f"That reply could not be read as an action: {error}\n"
                        "Reply again, in the required shape, with exactly one call."},
                ]

        self.stats.unparseable += 1
        return (rejected[-1] if rejected else ""), None, rejected


def probe(endpoint: str, model: str, *, timeout: float = 300.0) -> dict[str, Any]:
    """Ask the model one trivial question and note how it answers.

    Cheap, and it turns three separate hand-debugged surprises into a line of
    setup: whether it thinks out loud, and how many tokens a bare answer costs
    it -- which is what the generation budget has to clear.
    """
    client = ModelClient(endpoint, model, max_tokens=2048, max_requeries=0,
                         timeout=timeout)
    body = client._post_with_retries({
        "model": model, "max_tokens": 2048, "temperature": 0.0,
        "messages": [{"role": "user", "content":
                      "Reply with exactly this and nothing else:\n```\nok()\n```"}],
    })
    choice = body["choices"][0]
    answer, reasoning = split_reasoning(choice.get("message") or {})
    usage = body.get("usage") or {}
    return {
        "reasons_out_loud": bool(reasoning),
        "reasoning_chars": len(reasoning),
        "completion_tokens": usage.get("completion_tokens"),
        "finish_reason": choice.get("finish_reason"),
        "answer": answer[:120],
    }
