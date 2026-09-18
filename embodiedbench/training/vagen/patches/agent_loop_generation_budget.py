"""End the episode when the context is spent, instead of asking for -62 tokens.

``vagen/agent_loop/gym_agent_loop.py`` gates a rollout on the response
budget alone: it terminates when ``len(response_mask) >= response_length``.
The engine's ceiling is different -- ``vllm_async_server.py`` serves at most
``prompt_length + response_length`` tokens in total and computes each
request's allowance as ``max_model_len - len(prompt_ids)``. The two disagree
whenever the initial prompt runs past ``prompt_length``, which the loop
treats as a warning rather than a cut. The sequence then reaches the engine's
ceiling while the loop still believes there is response budget left, the next
generate request asks for a negative number of tokens, and the whole training
job dies mid-rollout on

    ValueError: max_tokens must be at least 1, got -62.

So before each generation, cap the request by what the engine will actually
serve, and terminate the episode -- the way the response-budget rule already
does -- when nothing meaningful is left. The floor is 16 tokens rather than 1
because a turn that can only fit a fragment of a call produces a format error,
not an action.

This patches the vendored checkout in place. It lives here rather than as an
edit to ``vendor/`` because ``vendor/`` is gitignored, so an edit made there
is invisible to git and lost on the next clone.

Idempotent; run it after ``git submodule update``:

    python -m embodiedbench.training.vagen.patches.agent_loop_generation_budget
"""

from __future__ import annotations

from pathlib import Path

REPO = Path(__file__).resolve().parents[4]
TARGET = REPO / "vendor/vagen/vagen/agent_loop/gym_agent_loop.py"

ANCHOR = """        sampling_params_for_turn = sampling_params.copy()
        max_new_tokens=sampling_params_for_turn.get("max_new_tokens", None) or agent_data.response_limit
        max_new_tokens = min(max_new_tokens, agent_data.response_limit)
        sampling_params_for_turn["max_new_tokens"] = max_new_tokens
"""

REPLACEMENT = """        sampling_params_for_turn = sampling_params.copy()
        max_new_tokens=sampling_params_for_turn.get("max_new_tokens", None) or agent_data.response_limit
        max_new_tokens = min(max_new_tokens, agent_data.response_limit)
        # Patched by embodiedbench/training/vagen/patches/: the engine serves
        # at most prompt_length + response_length tokens in total and sizes
        # each request as that minus len(prompt_ids). The response-budget rule
        # alone does not imply that bound -- an initial prompt past
        # prompt_length is only warned about -- and an unclamped request at
        # the ceiling asks vLLM for a negative allowance and kills the run.
        engine_remaining = (self.prompt_length + self.response_length
                            - len(agent_data.prompt_ids))
        if engine_remaining < 16:
            return AgentState.TERMINATED
        max_new_tokens = min(max_new_tokens, engine_remaining)
        sampling_params_for_turn["max_new_tokens"] = max_new_tokens
"""

MARKER = "engine_remaining"


def apply(target: Path = TARGET) -> str:
    if not target.exists():
        raise FileNotFoundError(f"vendored VAGEN not found at {target}")
    source = target.read_text()
    if MARKER in source:
        return "already patched"
    if ANCHOR not in source:
        raise RuntimeError(
            f"{target} does not contain the expected sampling-params block; "
            "VAGEN has moved and this patch needs rewriting")
    target.write_text(source.replace(ANCHOR, REPLACEMENT, 1))
    return "patched"


if __name__ == "__main__":
    print(f"{apply()}: {TARGET}")
