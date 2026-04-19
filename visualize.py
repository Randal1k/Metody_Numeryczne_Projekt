"""
Wykresy: trajektorie y(x) oraz błąd punktowy względem rozwiązania analitycznego.
"""

from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np

from projectile.model import ProjectileParams
from projectile.simulation import (
    Method,
    integrate_until_ground,
    pointwise_error_vs_analytical,
)


def run_plots(
    p: ProjectileParams | None = None,
    *,
    dt: float = 0.01,
    outfile_trajectory: str | None = None,
    outfile_error: str | None = None,
    show: bool = True,
) -> None:
    p = p or ProjectileParams()

    res_euler = integrate_until_ground(p, Method.EULER, dt=dt)
    res_rk4 = integrate_until_ground(p, Method.RK4, dt=dt)
    res_rkf = integrate_until_ground(p, Method.RKF45, dt=dt)
    res_ana = integrate_until_ground(p, Method.ANALYTICAL, dt=dt)

    fig1, ax1 = plt.subplots(figsize=(8, 5))
    for res, label, ls in (
        (res_euler, "Euler", "-"),
        (res_rk4, "RK4", "--"),
        (res_rkf, "RKF45", "-."),
        (res_ana, "Analityczne", ":"),
    ):
        xs = res.states[:, 0]
        ys = res.states[:, 1]
        ax1.plot(xs, ys, ls, lw=1.8, label=label)
    ax1.set_xlabel("x [m]")
    ax1.set_ylabel("y [m]")
    ax1.set_title("Trajektorie rzutu ukośnego z oporem liniowym $F_d=-kv$")
    ax1.grid(True, alpha=0.35)
    ax1.legend()
    ax1.set_aspect("equal", adjustable="box")
    fig1.tight_layout()
    if outfile_trajectory:
        fig1.savefig(outfile_trajectory, dpi=150)

    fig2, ax2 = plt.subplots(figsize=(8, 5))
    for res, label, ls in (
        (res_euler, "Euler", "-"),
        (res_rk4, "RK4", "--"),
        (res_rkf, "RKF45", "-."),
    ):
        err = pointwise_error_vs_analytical(res.times, res.states, p)
        # Ostatni punkt to interpolacja y=0; analityczne w tym samym t ma y>0 — pomijamy.
        if len(err) > 1:
            t_plot, e_plot = res.times[:-1], err[:-1]
        else:
            t_plot, e_plot = res.times, err
        ax2.semilogy(t_plot, e_plot + 1e-20, ls, lw=1.5, label=label)
    ax2.set_xlabel("t [s]")
    ax2.set_ylabel(r"$\|y_{\mathrm{num}}(t) - y_{\mathrm{an}}(t)\|_2$")
    ax2.set_title(
        "Błąd stanu w punktach siatki względem rozwiązania analitycznego "
        "(norma euklidesowa wektora stanu)"
    )
    ax2.grid(True, which="both", alpha=0.35)
    ax2.legend()
    fig2.tight_layout()
    if outfile_error:
        fig2.savefig(outfile_error, dpi=150)
    if show:
        plt.show()


if __name__ == "__main__":
    run_plots()
