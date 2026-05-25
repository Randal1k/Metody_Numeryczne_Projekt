"""
Aplikacja okienkowa: parametry v0, kąt α, m, k, g — wybór metod na wykresach,
zakładka benchmarku (średni / pełny czas, min, max, odchylenie).

Uruchomienie: python gui_app.py
"""

from __future__ import annotations

import threading
import tkinter as tk
from tkinter import messagebox, scrolledtext, ttk

from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg, NavigationToolbar2Tk
from matplotlib.figure import Figure

from benchmark import MethodBenchStats, format_benchmark_report, run_benchmark_detailed
from projectile.model import ProjectileParams
from projectile.tolerances import format_tolerance, parse_positive_float
from projectile.simulation import (
    Method,
    TrajectoryResult,
    integrate_until_ground,
    reference_analytical_trajectory,
)

NUM_METHOD_ORDER: list[tuple[Method, str]] = [
    (Method.EULER, "Euler"),
    (Method.RK4, "RK4"),
    (Method.RKF45, "RKF45"),
]

METHOD_COLORS: dict[str, str] = {
    "Euler": "tab:blue",
    "RK4": "tab:orange",
    "RKF45": "tab:green",
}

METHOD_LINEWIDTH: dict[str, float] = {
    "Euler": 1.8,
    "RK4": 1.8,
    "RKF45": 1.8,
}


def _read_params(entries: dict[str, tk.Entry]) -> ProjectileParams:
    try:
        v0 = float(entries["v0"].get())
        angle = float(entries["angle"].get())
        m = float(entries["m"].get())
        k = float(entries["k"].get())
        g = float(entries["g"].get())
        x0 = float(entries["x0"].get())
        y0 = float(entries["y0"].get())
    except ValueError as e:
        raise ValueError("Niepoprawna liczba w polach parametrów.") from e
    if m <= 0 or k < 0 or g < 0 or v0 < 0:
        raise ValueError("Wymagane: m > 0, k ≥ 0, g ≥ 0, v0 ≥ 0.")
    return ProjectileParams.from_speed_angle(
        v0,
        angle,
        x0=x0,
        y0=y0,
        m=m,
        k_drag=k,
        g=g,
    )


class ProjectileGUI(tk.Tk):
    def __init__(self) -> None:
        super().__init__()
        self.title("Rzut ukośny z oporem liniowym — symulacja i benchmark")
        self.geometry("1100x820")
        self.minsize(900, 700)

        self._param_entries: dict[str, tk.Entry] = {}
        self._traj_vars: dict[str, tk.BooleanVar] = {}
        self._show_ana_bg = tk.BooleanVar(value=True)
        # self._err_vars: dict[str, tk.BooleanVar] = {}

        self._build_param_frame()
        self._build_notebook()

    def _build_param_frame(self) -> None:
        outer = ttk.LabelFrame(self, text="Parametry fizyczne i początkowe", padding=8)
        outer.pack(fill=tk.X, padx=8, pady=6)

        row1_fields = [
            ("v0", "v₀ [m/s]", "32.0"),
            ("angle", "Kąt α [°]", "51.0"),
            ("m", "m [kg]", "1.0"),
            ("k", "k [N·s/m]", "0.1"),
            ("g", "g [m/s²]", "9.81"),
            ("x0", "x₀ [m]", "0.0"),
            ("y0", "y₀ [m]", "1.0"),
            ("dt", "Krok dt [s] (Euler/RK4)", "0.01"),
            ("atol", "RKF45 atol", "1e-9"),
            ("rtol", "RKF45 rtol", "1e-6"),
        ]
        row1 = ttk.Frame(outer)
        row1.pack(fill=tk.X)
        for key, label, default in row1_fields:
            f = ttk.Frame(row1)
            f.pack(side=tk.LEFT, padx=4, pady=2)
            ttk.Label(f, text=label).pack(anchor=tk.W)
            e = ttk.Entry(f, width=12)
            e.insert(0, default)
            e.pack()
            self._param_entries[key] = e

    def _build_notebook(self) -> None:
        nb = ttk.Notebook(self)
        nb.pack(fill=tk.BOTH, expand=True, padx=8, pady=4)

        tab_sim = ttk.Frame(nb, padding=6)
        tab_bench = ttk.Frame(nb, padding=6)
        nb.add(tab_sim, text="Symulacja i porównanie")
        nb.add(tab_bench, text="Benchmark")

        self._build_tab_simulation(tab_sim)
        self._build_tab_benchmark(tab_bench)

    def _build_tab_simulation(self, parent: ttk.Frame) -> None:
        ctrl = ttk.Frame(parent)
        ctrl.pack(side=tk.LEFT, fill=tk.Y, padx=(0, 8))

        ttk.Label(ctrl, text="Krzywe na wykresie trajektorii y(x):").pack(anchor=tk.W)
        for _, name in NUM_METHOD_ORDER:
            var = tk.BooleanVar(value=True)
            self._traj_vars[name] = var
            ttk.Checkbutton(ctrl, text=name, variable=var).pack(anchor=tk.W)

        ttk.Separator(ctrl, orient=tk.HORIZONTAL).pack(fill=tk.X, pady=8)
        ttk.Checkbutton(
            ctrl,
            text="Tło analityczne (szare)",
            variable=self._show_ana_bg,
        ).pack(anchor=tk.W)

        ttk.Label(
            ctrl,
            text=(
                "RKF45: zielona linia łączy wyłącznie punkty "
                "zaakceptowanych kroków przy podanych atol/rtol "
                "(łamana, nie interpolacja wygładzająca)."
            ),
            wraplength=220,
            justify=tk.LEFT,
            font=("", 8),
        ).pack(anchor=tk.W, pady=(8, 0))

        # --- błąd vs analityczne (tymczasowo wyłączony) ---
        # ttk.Separator(ctrl, orient=tk.HORIZONTAL).pack(fill=tk.X, pady=10)
        # ttk.Label(ctrl, text="Błąd vs analityczne (tylko metody numeryczne):").pack(anchor=tk.W)
        # for _, name in NUM_METHOD_ORDER:
        #     var = tk.BooleanVar(value=True)
        #     self._err_vars[name] = var
        #     ttk.Checkbutton(ctrl, text=name, variable=var).pack(anchor=tk.W)

        ttk.Button(ctrl, text="Symuluj / odśwież wykresy", command=self._run_simulation).pack(
            pady=(16, 4), fill=tk.X
        )

        plot_frame = ttk.Frame(parent)
        plot_frame.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)

        self.fig = Figure(figsize=(7.5, 5.5), dpi=100, layout="tight")
        self.ax_traj = self.fig.add_subplot(1, 1, 1)
        # self.ax_err = self.fig.add_subplot(2, 1, 2)

        self.canvas = FigureCanvasTkAgg(self.fig, master=plot_frame)
        self.canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)
        NavigationToolbar2Tk(self.canvas, plot_frame).pack(fill=tk.X)

        self._run_simulation()

    def _build_tab_benchmark(self, parent: ttk.Frame) -> None:
        top = ttk.Frame(parent)
        top.pack(fill=tk.X)

        ttk.Label(top, text="Liczba symulacji na metodę:").pack(side=tk.LEFT, padx=(0, 6))
        self._bench_n = tk.StringVar(value="1000")
        ttk.Entry(top, textvariable=self._bench_n, width=10).pack(side=tk.LEFT, padx=4)

        self._bench_btn = ttk.Button(top, text="Uruchom benchmark", command=self._start_benchmark)
        self._bench_btn.pack(side=tk.LEFT, padx=16)

        self._bench_status = ttk.Label(top, text="")
        self._bench_status.pack(side=tk.LEFT, padx=8)

        self._bench_text = scrolledtext.ScrolledText(parent, height=28, width=90, font=("Consolas", 10))
        self._bench_text.pack(fill=tk.BOTH, expand=True, pady=8)
        self._bench_text.insert(
            tk.END,
            "Parametry fizyczne i dt, atol, rtol — z górnego panelu (jak w symulacji).\n",
        )
        self._bench_text.insert(
            tk.END,
            "Po uruchomieniu pojawi się raport z rozróżnieniem: czas **jednej** pełnej "
            "symulacji (ms i s) vs **suma N** takich symulacji (sekundy).\n\n",
        )

    def _run_simulation(self) -> None:
        try:
            p = _read_params(self._param_entries)
            dt = parse_positive_float(self._param_entries["dt"].get(), name="dt")
            atol = parse_positive_float(self._param_entries["atol"].get(), name="atol")
            rtol = parse_positive_float(self._param_entries["rtol"].get(), name="rtol")
        except ValueError as e:
            messagebox.showerror("Błąd parametrów", str(e))
            return

        self.ax_traj.clear()

        if self._show_ana_bg.get():
            res_ana = reference_analytical_trajectory(p)
            self.ax_traj.plot(
                res_ana.states[:, 0],
                res_ana.states[:, 1],
                "-",
                color="0.55",
                lw=2.2,
                alpha=0.45,
                zorder=0,
                label="Analityczne",
            )

        cache: dict[Method, TrajectoryResult] = {}

        def get_traj(method: Method) -> TrajectoryResult:
            if method not in cache:
                if method is Method.RKF45:
                    cache[method] = integrate_until_ground(
                        p, method, dt=dt, rkf_atol=atol, rkf_rtol=rtol
                    )
                else:
                    cache[method] = integrate_until_ground(p, method, dt=dt)
            return cache[method]

        for method, name in NUM_METHOD_ORDER:
            if not self._traj_vars[name].get():
                continue
            res = get_traj(method)
            self.ax_traj.plot(
                res.states[:, 0],
                res.states[:, 1],
                "-",
                color=METHOD_COLORS[name],
                lw=METHOD_LINEWIDTH[name],
                zorder=1,
                label=name,
            )
            if name == "RKF45":
                self.ax_traj.plot(
                    res.states[:, 0],
                    res.states[:, 1],
                    "o",
                    color=METHOD_COLORS[name],
                    ms=4,
                    zorder=2,
                    label="_nolegend_",
                )

        self.ax_traj.set_xlabel("x [m]")
        self.ax_traj.set_ylabel("y [m]")
        title = "Trajektorie"
        self.ax_traj.set_title(title)
        self.ax_traj.grid(True, alpha=0.35)
        self.ax_traj.legend(loc="best", fontsize=8)
        self.ax_traj.set_xlim(left=0.0)
        self.ax_traj.set_ylim(bottom=0.0)
        self.ax_traj.set_aspect("equal", adjustable="box")

        # --- wykres błędu stanu (tymczasowo wyłączony) ---
        # from projectile.simulation import pointwise_error_vs_analytical
        #
        # self.ax_err.clear()
        # for method, name in NUM_METHOD_ORDER:
        #     if not self._err_vars[name].get():
        #         continue
        #     res = get_traj(method)
        #     err = pointwise_error_vs_analytical(res.times, res.states, p)
        #     if len(err) > 1:
        #         t_plot, e_plot = res.times[:-1], err[:-1]
        #     else:
        #         t_plot, e_plot = res.times, err
        #     self.ax_err.semilogy(
        #         t_plot,
        #         e_plot + 1e-20,
        #         "-",
        #         color=METHOD_COLORS[name],
        #         lw=METHOD_LINEWIDTH[name] * 0.85,
        #         label=name,
        #     )
        # self.ax_err.set_xlabel("t [s]")
        # self.ax_err.set_ylabel("‖y_num − y_an‖₂")
        # self.ax_err.set_title("Błąd stanu w punktach siatki (ostatni punkt pominięty)")
        # self.ax_err.grid(True, which="both", alpha=0.35)

        self.fig.tight_layout()
        self.canvas.draw()

    def _start_benchmark(self) -> None:
        try:
            p = _read_params(self._param_entries)
            n = int(self._bench_n.get())
            dt = parse_positive_float(self._param_entries["dt"].get(), name="dt")
            atol = parse_positive_float(self._param_entries["atol"].get(), name="atol")
            rtol = parse_positive_float(self._param_entries["rtol"].get(), name="rtol")
            if n < 1:
                raise ValueError("N ≥ 1")
        except ValueError as e:
            messagebox.showerror("Błąd", str(e))
            return

        self._bench_btn.configure(state=tk.DISABLED)
        self._bench_status.configure(text="Trwa benchmark…")

        def worker() -> None:
            stats = run_benchmark_detailed(
                n_runs=n, dt=dt, p=p, rkf_atol=atol, rkf_rtol=rtol
            )
            self.after(0, lambda: self._benchmark_done(stats, dt, atol, rtol))

        threading.Thread(target=worker, daemon=True).start()

    def _benchmark_done(
        self,
        stats: dict[str, MethodBenchStats],
        dt: float,
        atol: float,
        rtol: float,
    ) -> None:
        self._bench_btn.configure(state=tk.NORMAL)
        self._bench_status.configure(text="Gotowe.")

        block = format_benchmark_report(
            stats, dt=dt, rkf_atol=atol, rkf_rtol=rtol
        )
        self._bench_text.insert(tk.END, block + "\n\n")
        self._bench_text.see(tk.END)


def main() -> None:
    app = ProjectileGUI()
    app.mainloop()


if __name__ == "__main__":
    main()
