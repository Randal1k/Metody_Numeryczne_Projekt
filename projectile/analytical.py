"""
Rozwiązanie analityczne rzutu ukośnego z oporem liniowym F_d = -k v.

Wzory (v0, kąt θ od osi poziomej; vx0 = v0 cos θ, vy0 = v0 sin θ):

    x(t) = x0 + (m / k) v0 cos(θ) (1 - e^{-k t / m})
         = x0 + (m / k) vx0 (1 - e^{-k t / m})

    y(t) = y0 + (m / k) (v0 sin(θ) + m g / k) (1 - e^{-k t / m}) - (m g / k) t
         = y0 + (m / k) (vy0 + m g / k) (1 - e^{-k t / m}) - (m g / k) t

Prędkości (pochodne po czasie):

    v_x(t) = v0 cos(θ) e^{-k t / m}
    v_y(t) = (v0 sin(θ) + m g / k) e^{-k t / m} - m g / k
"""

from __future__ import annotations

import numpy as np

from .model import ProjectileParams


def analytical_state(t: float | np.ndarray, p: ProjectileParams) -> np.ndarray:
    """
    Zwraca wektor [x, y, v_x, v_y] dla skalarnego t lub tablicy czasów t.

    Dla t jako tablicy kształt wyniku to (len(t), 4).
    """
    t_arr = np.atleast_1d(np.asarray(t, dtype=float))
    m, k, g = p.m, p.k, p.g
    vx0, vy0 = p.vx0, p.vy0

    if k == 0.0:
        vx = np.full_like(t_arr, vx0, dtype=float)
        vy = vy0 - g * t_arr
        x = p.x0 + vx0 * t_arr
        y = p.y0 + vy0 * t_arr - 0.5 * g * t_arr**2
        out = np.stack([x, y, vx, vy], axis=-1)
    else:
        decay = np.exp(-k * t_arr / m)
        one_minus_decay = 1.0 - decay

        # Składowa pozioma:
        x = p.x0 + (m / k) * vx0 * one_minus_decay

        # Składowa pionowa:
        y = (
            p.y0
            + (m / k) * (vy0 + m * g / k) * one_minus_decay
            - (m * g / k) * t_arr
        )
        vx = vx0 * decay
        vy = (vy0 + m * g / k) * decay - m * g / k

        out = np.stack([x, y, vx, vy], axis=-1)

    if np.isscalar(t):
        return out[0]
    return out
