"""Only the most recent frames survive into the next prompt.

The vendored agent loop accumulates them for the whole episode and never
drops any, so a forty-turn episode carries forty turns of pictures into every
later prompt. The sequence overruns, and the truncation that catches it drops
the OLDEST TEXT -- the memory the courier needs -- to keep frames it can no
longer act on.
"""

from __future__ import annotations

import types

import pytest

from embodiedbench.training.vagen.patches import recent_images_only as patch


class _Agent:
    def __init__(self, images, messages):
        self.image_data = list(images)
        self.messages = messages


def test_the_window_keeps_the_last_frames_and_their_placeholders():
    agent = _Agent(
        list(range(6)),
        [{"content": "turn 1 <image> <image>"},
         {"content": "turn 2 <image> <image>"},
         {"content": "turn 3 <image> <image>"}],
    )
    patch._keep_recent_images(agent, keep=3)

    assert agent.image_data == [3, 4, 5], "the newest survive"
    # Placeholders and images are one contract: the processor pairs them by
    # position, so a marker left behind shifts every later pairing by one.
    left = sum(m["content"].count("<image>") for m in agent.messages)
    assert left == len(agent.image_data)
    # ...and the oldest turns still say a photograph was taken.
    assert "(a photograph)" in agent.messages[0]["content"]
    assert "<image>" not in agent.messages[0]["content"]


def test_a_short_episode_is_untouched():
    agent = _Agent([1, 2], [{"content": "turn 1 <image> <image>"}])
    patch._keep_recent_images(agent, keep=3)
    assert agent.image_data == [1, 2]
    assert agent.messages[0]["content"].count("<image>") == 2


def test_the_window_is_configurable_from_the_environment(monkeypatch):
    monkeypatch.setenv("VAGEN_RECENT_IMAGES", "1")
    agent = _Agent([1, 2, 3], [{"content": "<image> <image> <image>"}])
    patch._keep_recent_images(agent)
    assert agent.image_data == [3]


def test_the_patch_is_idempotent_and_refuses_a_moved_anchor():
    source = "        agent_data.image_data.extend(new_images)\n"
    once = patch.patch(source)
    assert patch.MARKER in once
    assert patch.patch(once) == once, "running it twice changes nothing"
    with pytest.raises(SystemExit):
        patch.patch("nothing that looks like the anchor")
