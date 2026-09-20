"""Task plugins (design plan §8, Contract C)."""

from embodiedbench.tasks.core import TaskPlugin, TaskRequirements
from embodiedbench.tasks.delivery import DeliveryTask

__all__ = ["DeliveryTask", "TaskPlugin", "TaskRequirements"]
