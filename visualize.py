"""
Wykresy: trajektorie y(x) oraz błąd punktowy względem rozwiązania analitycznego.
"""

from __future__ import annotations

import matplotlib.pyplot as plt

from projectile.model import ProjectileParams
from projectile.simulation import (
    Method,
    integrate_until_ground,
    reference_analytical_trajectory,
)

METHOD_COLORS: dict[str, str] = {
    "Euler": "tab:blue",
    "RK4": "tab:orange",
    "RKF45": "tab:green",
}


def run_plots(
    p: ProjectileParams | None = None,
    *,
    dt: float = 0.01,
    rkf_atol: float = 1e-9,
    rkf_rtol: float = 1e-6,
    show_analytical_background: bool = True,
    outfile_trajectory: str | None = None,
    outfile_error: str | None = None,
    show: bool = True,
) -> None:
    p = p or ProjectileParams()

    res_euler = integrate_until_ground(p, Method.EULER, dt=dt)
    res_rk4 = integrate_until_ground(p, Method.RK4, dt=dt)
    res_rkf = integrate_until_ground(
        p, Method.RKF45, dt=dt, rkf_atol=rkf_atol, rkf_rtol=rkf_rtol
    )

    fig1, ax1 = plt.subplots(figsize=(8, 5))
    if show_analytical_background:
        res_ana = reference_analytical_trajectory(p)
        ax1.plot(
            res_ana.states[:, 0],
            res_ana.states[:, 1],
            "-",
            color="0.55",
            lw=2.2,
            alpha=0.45,
            zorder=0,
            label="Analityczne",
        )
    for res, label in (
        (res_euler, "Euler"),
        (res_rk4, "RK4"),
        (res_rkf, "RKF45"),
    ):
        color = METHOD_COLORS[label]
        ax1.plot(
            res.states[:, 0],
            res.states[:, 1],
            "-",
            lw=1.8,
            color=color,
            zorder=1,
            label=label,
        )
    ax1.set_xlabel("x [m]")
    ax1.set_ylabel("y [m]")
    ax1.set_title("Trajektorie rzutu ukośnego z oporem liniowym $F_d=-kv$")
    ax1.grid(True, alpha=0.35)
    ax1.legend(loc="best")
    ax1.set_xlim(left=0.0)
    ax1.set_ylim(bottom=0.0)
    ax1.set_aspect("equal", adjustable="box")
    fig1.tight_layout()
    if outfile_trajectory:
        fig1.savefig(outfile_trajectory, dpi=150)

    # --- wykres błędu stanu (tymczasowo wyłączony) ---
    # from projectile.simulation import pointwise_error_vs_analytical
    #
    # fig2, ax2 = plt.subplots(figsize=(8, 5))
    # for res, label in (
    #     (res_euler, "Euler"),
    #     (res_rk4, "RK4"),
    #     (res_rkf, "RKF45"),
    # ):
    #     err = pointwise_error_vs_analytical(res.times, res.states, p)
    #     if len(err) > 1:
    #         t_plot, e_plot = res.times[:-1], err[:-1]
    #     else:
    #         t_plot, e_plot = res.times, err
    #     ax2.semilogy(t_plot, e_plot + 1e-20, "-", lw=1.5, color=METHOD_COLORS[label], label=label)
    # ax2.set_xlabel("t [s]")
    # ax2.set_ylabel(r"$\|y_{\mathrm{num}}(t) - y_{\mathrm{an}}(t)\|_2$")
    # ax2.set_title(
    #     "Błąd stanu w punktach siatki względem rozwiązania analitycznego "
    #     "(norma euklidesowa wektora stanu)"
    # )
    # ax2.grid(True, which="both", alpha=0.35)
    # fig2.tight_layout()
    # if outfile_error:
    #     fig2.savefig(outfile_error, dpi=150)
    _ = outfile_error

    if show:
        plt.show()


if __name__ == "__main__":
    run_plots()
