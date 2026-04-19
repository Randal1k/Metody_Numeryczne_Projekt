"""
Model rzutu z oporem powietrza proporcjonalnym do prędkości.

Stan: y = [x, y, v_x, v_y]^T

    dx/dt = v_x
    dy/dt = v_y
    dv_x/dt = -(k/m) v_x
    dv_y/dt = -g - (k/m) v_y
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class ProjectileParams:
    """Parametry fizyczne i początkowe."""

    x0: float = 0.0
    # Domyślnie nad ziemią — unika degeneracji y=0 przy starcie z płaszczyzny z=0.
    y0: float = 1.0
    vx0: float = 20.0
    vy0: float = 25.0
    m: float = 1.0
    k: float = 0.1
    g: float = 9.81

    @property
    def drag_coeff(self) -> float:
        """Współczynnik k/m występujący w równaniach."""
        return self.k / self.m

    @staticmethod
    def from_speed_angle(
        v0: float,
        angle_deg: float,
        *,
        x0: float = 0.0,
        y0: float = 1.0,
        m: float = 1.0,
        k_drag: float = 0.1,
        g: float = 9.81,
    ) -> ProjectileParams:
        """
        Konstruktor z prędkością początkową i kątem rzutu (w stopniach, od osi x).

        v0 [m/s], kąt w stopniach; składowe: vx0 = v0 cos(α), vy0 = v0 sin(α).
        """
        rad = np.deg2rad(angle_deg)
        return ProjectileParams(
            x0=x0,
            y0=y0,
            vx0=float(v0 * np.cos(rad)),
            vy0=float(v0 * np.sin(rad)),
            m=m,
            k=k_drag,
            g=g,
        )


def state_derivative(_t: float, state: np.ndarray, p: ProjectileParams) -> np.ndarray:
    """
    Prawa strona układu ODE: dy/dt = f(t, y).

    state: wektor (x, y, vx, vy).
    """
    x, y, vx, vy = state
    alpha = p.drag_coeff
    return np.array(
        [vx, vy, -alpha * vx, -p.g - alpha * vy],
        dtype=float,
    )
