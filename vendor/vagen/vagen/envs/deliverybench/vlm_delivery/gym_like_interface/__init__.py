"""
Gym-like interfaces for DeliveryBench.

This package provides:
- `DeliveryBenchGymEnvText`: pure-Python text-only environment (no UE/Qt).
- `DeliveryBenchGymEnvQtRouteA`: optional Qt/UE-backed environment (requires PyQt5 + UE).
"""

from .text_env import DeliveryBenchGymEnvText

try:
    from .gym_like_interface import DeliveryBenchGymEnvQtRouteA  # optional
except Exception:  # pragma: no cover
    DeliveryBenchGymEnvQtRouteA = None

__all__ = [
    "DeliveryBenchGymEnvText",
    "DeliveryBenchGymEnvQtRouteA",
]
