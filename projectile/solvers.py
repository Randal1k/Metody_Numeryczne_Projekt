"""
Solvery ODE dla układu rzutu z oporem liniowym.

Zawiera: Eulera, RK4 oraz adaptacyjną parę Rungego-Kutty-Fehlberga 4(5) (RKF45).
"""

from __future__ import annotations

from typing import Callable

import numpy as np

from .model import ProjectileParams, state_derivative

StateFn = Callable[[float, np.ndarray], np.ndarray]


def _make_rhs(p: ProjectileParams) -> StateFn:
    def rhs(t: float, y: np.ndarray) -> np.ndarray:
        return state_derivative(t, y, p)

    return rhs


def euler_step(rhs: StateFn, t: float, y: np.ndarray, h: float) -> np.ndarray:
    """Jeden krok metody Eulera w przód."""
    return y + h * rhs(t, y)


def rk4_step(rhs: StateFn, t: float, y: np.ndarray, h: float) -> np.ndarray:
    """Jeden krok klasycznego RK4."""
    k1 = rhs(t, y)
    k2 = rhs(t + 0.5 * h, y + 0.5 * h * k1)
    k3 = rhs(t + 0.5 * h, y + 0.5 * h * k2)
    k4 = rhs(t + h, y + h * k3)
    return y + (h / 6.0) * (k1 + 2.0 * k2 + 2.0 * k3 + k4)


def rkf45_step(
    rhs: StateFn,
    t: float,
    y: np.ndarray,
    h: float,
    *,
    atol: float = 1e-9,
    rtol: float = 1e-6,
) -> tuple[np.ndarray, np.ndarray, float, bool]:
    """
    Jeden krok metody Fehlberga RKF45 (para rzędów 4 i 5, 6 ewaluacji f).

    Zwraca:
        y5 — propozycja stanu rzędu 5 (używana jako akceptowany krok),
        err_est — estymator błędu (norma różnicy y5 - y4),
        h_new — proponowany następny krok czasu,
        accept — czy krok zaakceptowany (err_est w tolerancji).
    """
    # Współczynniki Fehlberga (6 etapów)
    a21 = 1.0 / 4.0
    a31 = 3.0 / 32.0
    a32 = 9.0 / 32.0
    a41 = 1932.0 / 2197.0
    a42 = -7200.0 / 2197.0
    a43 = 7296.0 / 2197.0
    a51 = 439.0 / 216.0
    a52 = -8.0
    a53 = 3680.0 / 513.0
    a54 = -845.0 / 4104.0
    a61 = -8.0 / 27.0
    a62 = 2.0
    a63 = -3544.0 / 2565.0
    a64 = 1859.0 / 4104.0
    a65 = -11.0 / 40.0

    b4 = np.array(
        [25.0 / 216.0, 0.0, 1408.0 / 2565.0, 2197.0 / 4104.0, -1.0 / 5.0, 0.0],
        dtype=float,
    )
    b5 = np.array(
        [
            16.0 / 135.0,
            0.0,
            6656.0 / 12825.0,
            28561.0 / 56430.0,
            -9.0 / 50.0,
            2.0 / 55.0,
        ],
        dtype=float,
    )

    k1 = rhs(t, y)
    k2 = rhs(t + 0.25 * h, y + h * a21 * k1)
    k3 = rhs(t + (3.0 / 8.0) * h, y + h * (a31 * k1 + a32 * k2))
    k4 = rhs(t + (12.0 / 13.0) * h, y + h * (a41 * k1 + a42 * k2 + a43 * k3))
    k5 = rhs(t + h, y + h * (a51 * k1 + a52 * k2 + a53 * k3 + a54 * k4))
    k6 = rhs(
        t + 0.5 * h,
        y + h * (a61 * k1 + a62 * k2 + a63 * k3 + a64 * k4 + a65 * k5),
    )

    ks = np.stack([k1, k2, k3, k4, k5, k6], axis=0)
    y4 = y + h * np.dot(b4, ks)
    y5 = y + h * np.dot(b5, ks)

    scale = atol + rtol * np.maximum(np.abs(y), np.abs(y5))
    err_vec = (y5 - y4) / np.maximum(scale, 1e-15)
    err_est = float(np.linalg.norm(err_vec))

    if err_est == 0.0:
        factor = 2.0
    else:
        factor = 0.9 * err_est ** (-0.2)

    factor = float(np.clip(factor, 0.2, 10.0))
    h_new = h * factor
    accept = err_est <= 1.0
    return y5, np.asarray(err_est), h_new, accept
