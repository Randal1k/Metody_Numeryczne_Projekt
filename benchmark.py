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


def _bench_one(method: Method, p: ProjectileParams, dt: float) -> float:
    t0 = time.perf_counter()
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
            times.append(_bench_one(method, p, dt))
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


def format_benchmark_report(stats: dict[str, MethodBenchStats]) -> str:
    """
    Czytelny opis wyników: rozróżnienie czasu jednej pełnej symulacji vs sumy N uruchomień,
    jednostki ms vs s, rozrzut pomiarów (OS, GC itd.).
    """
    lines: list[str] = []
    lines.append(
        "Znaczenie pomiaru:\n"
        "  - „Symulacja” to jedno pełne całkowanie trajektorii (wiele kroków numerycznych "
        "w środku) aż do uderzenia w ziemię.\n"
        "  - Benchmark: N kolejnych takich symulacji; mierzony jest czas każdej z osobna.\n"
        "  - Średni czas = średnia z N pomiarów czasu jednej pełnej symulacji (poniżej: ms i s).\n"
        "  - Łączny czas N symulacji = suma tych N pomiarów (w sekundach), czyli ok. "
        "N × (średnia w sekundach), a nie „N × liczba w ms” bez przeliczenia jednostki.\n"
        "    Przykład: średnia 65 ms = 0,065 s na jedną symulację; N = 1000 daje sumę ok. 65 s, nie 65 ms.\n"
        "  - Odchylenie std: rozrzut między N czasami całej symulacji (kolejne uruchomienia), "
        "nie między krokami całkowania ODE.\n"
        "    Źródła rozrzutu: planista OS, cache, GC Pythona, inne procesy (nie błąd metody).\n"
    )

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
            f"    Średni czas jednej pełnej symulacji:     {s.mean_s * 1000:.4f} ms   "
            f"({s.mean_s:.6f} s)\n"
            f"    Suma czasu N symulacji (N kolejnych uruchomień): {s.total_s:.4f} s   "
            f"(suma {n} pomiarów; kontrola: N × średnia = {n_times_mean:.4f} s)\n"
            f"    Min / max czasu pojedynczej symulacji: {s.min_s * 1000:.4f} ms / {s.max_s * 1000:.4f} ms\n"
            f"    Odchylenie std między {n} pomiarami (całe symulacje): {s.stdev_s * 1000:.4f} ms "
            f"({s.stdev_s:.6f} s)"
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
