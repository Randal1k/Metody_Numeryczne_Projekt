"""
Benchmark: wielokrotne powtórzenia integracji dla każdej metody (Euler, RK4, RKF45).

Czas mierzony przez time.perf_counter().
"""

from __future__ import annotations

import time
from dataclasses import dataclass
from statistics import mean, pstdev

from projectile.model import ProjectileParams
from projectile.simulation import Method, integrate_until_ground

N_RUNS = 1000


def _bench_one(
    method: Method,
    p: ProjectileParams,
    dt: float,
    *,
    rkf_atol: float = 1e-9,
    rkf_rtol: float = 1e-6,
) -> float:
    t0 = time.perf_counter()
    if method is Method.RKF45:
        integrate_until_ground(
            p, method, dt=dt, rkf_atol=rkf_atol, rkf_rtol=rkf_rtol
        )
    else:
        integrate_until_ground(p, method, dt=dt)
    return time.perf_counter() - t0


@dataclass(frozen=True)
class MethodBenchStats:
    """Statystyki czasu pojedynczej symulacji dla jednej metody (w sekundach)."""

    n_runs: int
    mean_s: float
    total_s: float
    min_s: float
    max_s: float
    stdev_s: float


def run_benchmark(
    n_runs: int = N_RUNS,
    *,
    dt: float = 0.01,
    p: ProjectileParams | None = None,
) -> dict[str, float]:
    """Średni czas [s] na jedną symulację — interfejs uproszczony (kompatybilność)."""
    detailed = run_benchmark_detailed(n_runs=n_runs, dt=dt, p=p)
    return {name: s.mean_s for name, s in detailed.items()}


def run_benchmark_detailed(
    n_runs: int = N_RUNS,
    *,
    dt: float = 0.01,
    rkf_atol: float = 1e-9,
    rkf_rtol: float = 1e-6,
    p: ProjectileParams | None = None,
) -> dict[str, MethodBenchStats]:
    """
    Dla każdej metody numerycznej: średnia, suma, min, max, odchylenie standardowe
    (dla n_runs pomiarów czasu jednej pełnej symulacji do uderzenia w ziemię).
    """
    p = p or ProjectileParams()
    out: dict[str, MethodBenchStats] = {}
    for method, name in (
        (Method.EULER, "Euler"),
        (Method.RK4, "RK4"),
        (Method.RKF45, "RKF45"),
    ):
        times: list[float] = []
        for _ in range(n_runs):
            times.append(
                _bench_one(method, p, dt, rkf_atol=rkf_atol, rkf_rtol=rkf_rtol)
            )
        total = sum(times)
        out[name] = MethodBenchStats(
            n_runs=n_runs,
            mean_s=mean(times),
            total_s=total,
            min_s=min(times),
            max_s=max(times),
            stdev_s=pstdev(times) if len(times) > 1 else 0.0,
        )
    return out


def format_benchmark_report(
    stats: dict[str, MethodBenchStats],
    *,
    dt: float | None = None,
    rkf_atol: float | None = None,
    rkf_rtol: float | None = None,
) -> str:
    """
    Czytelny opis wyników: rozróżnienie czasu jednej pełnej symulacji vs sumy N uruchomień,
    jednostki ms vs s, rozrzut pomiarów (OS, GC itd.).
    """
    lines: list[str] = []
    if dt is not None:
        lines.append(f"dt (Euler/RK4) = {dt:g} s")
    if rkf_atol is not None and rkf_rtol is not None:
        from projectile.tolerances import format_tolerance

        lines.append(
            f"RKF45: atol = {format_tolerance(rkf_atol)}, "
            f"rtol = {format_tolerance(rkf_rtol)}"
        )
    if lines:
        lines.append("")

    grand_total = 0.0
    for name in ("Euler", "RK4", "RKF45"):
        s = stats[name]
        grand_total += s.total_s
        n = s.n_runs
        n_times_mean = n * s.mean_s
        lines.append("")
        lines.append(f"=== {name} ===")
        lines.append(
            f"  Dla N = {n}:\n"
            f"    Średni czas jednej pełnej symulacji:     {s.mean_s * 1000:.4f} ms   \n"
            f"    Suma czasu N symulacji : {s.total_s:.4f} s   \n"
            f"    Min / max czasu pojedynczej symulacji: {s.min_s * 1000:.4f} ms / {s.max_s * 1000:.4f} ms\n"
            f"    Odchylenie std między {n} pomiarami : {s.stdev_s * 1000:.4f} ms \n"
        )

    lines.append("")
    lines.append(
        f"--- Suma łącznych czasów (Euler + RK4 + RKF45, każda po N symulacji): "
        f"{grand_total:.4f} s ---"
    )
    return "\n".join(lines)


def main() -> None:
    print(f"Benchmark: {N_RUNS} symulacji na metodę, dt (Euler/RK4) = 0.01 s\n")
    detailed = run_benchmark_detailed(N_RUNS)
    print(format_benchmark_report(detailed))
    avgs = {name: s.mean_s for name, s in detailed.items()}
    fastest = min(avgs, key=avgs.get)  # type: ignore[arg-type]
    print(f"\nNajszybsza średnio (jedna symulacja): {fastest}")


if __name__ == "__main__":
    main()
