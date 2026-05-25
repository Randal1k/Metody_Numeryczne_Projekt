"""
Integracja trajektorii do momentu uderzenia w podłoże (y <= 0).

Dla metod stałokrokowych używany jest stały krok h; dla RKF45 adaptacyjny krok
z kontrolą tolerancji i ograniczeniami h_min / h_max.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum, auto

import numpy as np

from .analytical import analytical_state
from .model import ProjectileParams
from .solvers import _make_rhs, euler_step, rk4_step, rkf45_step


class Method(Enum):
    EULER = auto()
    RK4 = auto()
    RKF45 = auto()
    ANALYTICAL = auto()


@dataclass
class TrajectoryResult:
    """Wynik symulacji: czasy, stany oraz ewentualnie estymatory błędu kroku."""

    times: np.ndarray
    states: np.ndarray
    # Długość local_error_estimates może być o 1 krótsza niż times (tylko RKF45)
    local_error_estimates: np.ndarray | None = None


def _initial_state(p: ProjectileParams) -> np.ndarray:
    return np.array([p.x0, p.y0, p.vx0, p.vy0], dtype=float)


def _intersect_ground_linear(
    t0: float, y0: np.ndarray, t1: float, y1: np.ndarray
) -> tuple[float, np.ndarray]:
    """Liniowa interpolacja przecięcia y=0 między (t0,y0) a (t1,y1)."""
    y0p, y1p = float(y0[1]), float(y1[1])
    if y1p == y0p:
        theta = 0.5
    else:
        theta = -y0p / (y1p - y0p)
    theta = float(np.clip(theta, 0.0, 1.0))
    t_hit = t0 + theta * (t1 - t0)
    state_hit = (1.0 - theta) * y0 + theta * y1
    state_hit[1] = 0.0
    return t_hit, state_hit


def integrate_until_ground(
    p: ProjectileParams,
    method: Method,
    *,
    dt: float = 0.01,
    t_max: float = 200.0,
    rkf_atol: float = 1e-9,
    rkf_rtol: float = 1e-6,
    h0_rkf: float | None = None,
) -> TrajectoryResult:
    """
    Całkuje od t=0 do pierwszego przejścia y przez zero (z góry na dół).

    Dla ANALYTICAL generowana jest siatka czasu co dt do t_hit z rozwiązania
    analitycznego (wyszukiwanie t_hit bisection po y(t)).
    """
    if method is Method.ANALYTICAL:
        return _analytical_trajectory_on_grid(p, dt=dt, t_max=t_max)

    y = _initial_state(p)
    rhs = _make_rhs(p)
    times: list[float] = [0.0]
    states: list[np.ndarray] = [y.copy()]
    local_errs: list[float] = []

    if method in (Method.EULER, Method.RK4):
        t = 0.0
        while t < t_max:
            if method is Method.EULER:
                y_next = euler_step(rhs, t, y, dt)
            else:
                y_next = rk4_step(rhs, t, y, dt)
            t_next = t + dt
            if y[1] >= 0.0 >= y_next[1] or y_next[1] <= 0.0 <= y[1]:
                t_hit, y_hit = _intersect_ground_linear(t, y, t_next, y_next)
                times.append(t_hit)
                states.append(y_hit)
                break
            t, y = t_next, y_next
            times.append(t)
            states.append(y.copy())
    else:
        # RKF45
        t = 0.0
        h = h0_rkf if h0_rkf is not None else min(0.05, dt * 5)
        h_min = 1e-8
        h_max = min(0.5, t_max / 10.0)
        step_count = 0
        max_steps = 2_000_000

        while t < t_max and step_count < max_steps:
            step_count += 1
            h = float(np.clip(h, h_min, min(h_max, t_max - t)))
            if h <= h_min * 1.01 and t > 0:
                break

            y_try, err_est, h_sug, accept = rkf45_step(
                rhs, t, y, h, atol=rkf_atol, rtol=rkf_rtol
            )
            if not accept:
                h = min(h_sug, 0.5 * h)
                continue

            t_next = t + h
            if y[1] >= 0.0 >= y_try[1] or y_try[1] <= 0.0 <= y[1]:
                t_hit, y_hit = _intersect_ground_linear(t, y, t_next, y_try)
                times.append(t_hit)
                states.append(y_hit)
                local_errs.append(float(err_est))
                break

            t, y = t_next, y_try
            times.append(t)
            states.append(y.copy())
            local_errs.append(float(err_est))
            h = min(h_sug, h_max)

    return TrajectoryResult(
        times=np.asarray(times, dtype=float),
        states=np.vstack(states),
        local_error_estimates=np.asarray(local_errs, dtype=float)
        if local_errs
        else None,
    )


def _analytical_trajectory_on_grid(
    p: ProjectileParams, dt: float, t_max: float
) -> TrajectoryResult:
    """Analityczne y(t), x(t) na siatce do czasu uderzenia w ziemię."""
    t_hit = _find_ground_time_analytical(p, t_max=t_max)
    if t_hit is None:
        t_grid = np.arange(0.0, t_max + dt, dt)
    else:
        t_grid = np.arange(0.0, min(t_hit, t_max) + 1e-12, dt)
        if t_grid[-1] < t_hit:
            t_grid = np.append(t_grid, t_hit)

    states = analytical_state(t_grid, p)
    return TrajectoryResult(times=t_grid, states=states, local_error_estimates=None)


def reference_analytical_trajectory(
    p: ProjectileParams,
    *,
    n_points: int = 1000,
    t_max: float = 200.0,
) -> TrajectoryResult:
    """
    Dokładna trajektoria analityczna na gęstej siatce czasu (niezależnej od dt symulacji).

    Używana jako tło odniesienia na wykresach y(x).
    """
    t_hit = _find_ground_time_analytical(p, t_max=t_max)
    if t_hit is None:
        t_hit = t_max
    t_grid = np.linspace(0.0, t_hit, max(2, n_points))
    states = analytical_state(t_grid, p)
    return TrajectoryResult(times=t_grid, states=states, local_error_estimates=None)


def _find_ground_time_analytical(p: ProjectileParams, t_max: float) -> float | None:
    """Pierwsze t >= 0, że y(t) <= 0 po locie (dla startu na y=0 z v_y>0 pomijamy t=0)."""

    def y_at(t: float) -> float:
        return float(analytical_state(t, p)[1])

    y0 = y_at(0.0)
    if y0 < 0:
        return 0.0
    if y0 == 0.0 and p.vy0 <= 0:
        return 0.0

    t_lo = 0.0
    if y0 == 0.0 and p.vy0 > 0:
        t_lo = 1e-12
    y_lo = y_at(t_lo)
    if y_lo <= 0:
        return 0.0

    t_hi = 0.05
    while y_at(t_hi) > 0 and t_hi < t_max:
        t_hi = min(t_hi + 0.05, t_max)
    if y_at(t_hi) > 0:
        return None

    while t_hi - t_lo > 1e-10:
        tm = 0.5 * (t_lo + t_hi)
        if y_at(tm) > 0:
            t_lo = tm
        else:
            t_hi = tm
    return t_hi


def pointwise_error_vs_analytical(
    times: np.ndarray, states: np.ndarray, p: ProjectileParams
) -> np.ndarray:
    """
    Norma euklidesowa ||y_num(t) - y_an(t)|| w każdym zapisanym punkcie czasu.

    Traktujemy to jako wykres „błędu w punktach siatki” względem rozwiązania
    analitycznego (dla trajektorii zbieżnego porównania z dokładnym rozwiązaniem).
    """
    exact = analytical_state(times, p)
    diff = states - exact
    return np.linalg.norm(diff, axis=1)
