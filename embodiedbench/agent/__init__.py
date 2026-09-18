"""Agent harness (design plan §10, Contract E).

Model access and agent behavior are separate: a ``BaseModelAdapter`` only turns
messages into a turn, while the ``AgentHarness`` owns context policy, action
parsing, budgets, memory, the episode loop, and trajectory emission.
"""

from embodiedbench.agent.harness import AgentHarness, EpisodeOutcome
from embodiedbench.agent.policies import ScriptedCourierPolicy

__all__ = ["AgentHarness", "EpisodeOutcome", "ScriptedCourierPolicy"]
