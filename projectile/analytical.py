"""
Rozwiązanie analityczne dla oporu liniowego (składowe rozkłają się).

Oznaczenia: α = k/m.

    v_x(t) = v_x0 * exp(-α t)
    v_y(t) = (v_y0 + g/α) * exp(-α t) - g/α

    x(t) = x0 + (v_x0/α) * (1 - exp(-α t))
    y(t) = y0 + ((v_y0 + g/α)/α) * (1 - exp(-α t)) - (g/α) * t
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
    alpha = p.drag_coeff
    # Unikamy dzielenia przez zero przy braku oporu
    if alpha == 0.0:
        vx = np.full_like(t_arr, p.vx0, dtype=float)
        vy = p.vy0 - p.g * t_arr
        x = p.x0 + p.vx0 * t_arr
        y = p.y0 + p.vy0 * t_arr - 0.5 * p.g * t_arr**2
        out = np.stack([x, y, vx, vy], axis=-1)
    else:
        ea = np.exp(-alpha * t_arr)
        g_over_a = p.g / alpha
        vx = p.vx0 * ea
        vy = (p.vy0 + g_over_a) * ea - g_over_a
        x = p.x0 + (p.vx0 / alpha) * (1.0 - ea)
        y = (
            p.y0
            + ((p.vy0 + g_over_a) / alpha) * (1.0 - ea)
            - g_over_a * t_arr
        )
        out = np.stack([x, y, vx, vy], axis=-1)

    if np.isscalar(t):
        return out[0]
    return out
