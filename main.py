"""
Punkt wejścia: wizualizacja trajektorii i błędów oraz opcjonalnie benchmark.

Uruchomienie:
    python main.py              # wykresy + podsumowanie benchmarku
    python main.py --plots-only
    python main.py --bench-only
    python main.py --gui        # aplikacja okienkowa (tkinter)
"""

from __future__ import annotations

import argparse

from benchmark import format_benchmark_report, run_benchmark_detailed
from projectile.model import ProjectileParams
from visualize import run_plots


def main() -> None:
    parser = argparse.ArgumentParser(description="Rzut ukośny z oporem liniowym.")
    parser.add_argument("--plots-only", action="store_true")
    parser.add_argument("--bench-only", action="store_true")
    parser.add_argument("--dt", type=float, default=0.01, help="Krok dla Eulera/RK4")
    parser.add_argument("--gui", action="store_true", help="Uruchom aplikację okienkową")
    args = parser.parse_args()

    if args.gui:
        from gui_app import main as gui_main

        gui_main()
        return

    p = ProjectileParams()

    if not args.bench_only:
        run_plots(p, dt=args.dt, show=True)

    if not args.plots_only:
        print()
        detailed = run_benchmark_detailed(p=p, dt=args.dt)
        print(format_benchmark_report(detailed))


if __name__ == "__main__":
    main()
