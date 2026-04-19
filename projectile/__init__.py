"""Symulacja rzutu ukośnego z oporem liniowym F_d = -k v."""

from .model import ProjectileParams, state_derivative
from .analytical import analytical_state
from .simulation import Method, integrate_until_ground

__all__ = [
    "ProjectileParams",
    "state_derivative",
    "analytical_state",
    "Method",
    "integrate_until_ground",
]
