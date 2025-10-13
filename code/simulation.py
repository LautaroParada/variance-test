"""High-level driver to simulate price paths and evaluate the VRT statistic."""

from __future__ import annotations

import argparse
import time
from dataclasses import dataclass
from typing import Dict, Iterable, Optional, Tuple

import numpy as np

from price_paths import PricePaths
from variance_test import EMH
from visuals import VRTVisuals


@dataclass
class SimulationConfig:
    """Configuration bundle for a variance ratio test experiment."""

    num_series: int = 10
    horizon: int = 100
    initial_price: float = 100.0
    mu: float = 0.09
    sigma: float = 0.1
    jump_intensity: float = 50.0
    kappa: float = 0.1
    theta: float = 0.06
    risk_free_rate: float = 0.02
    aggregation_horizon: int = 5
    heteroskedastic: bool = True
    seed: Optional[int] = None

    def __post_init__(self) -> None:
        if self.num_series < 1:
            raise ValueError("num_series must be a positive integer.")
        if self.horizon < 2:
            raise ValueError("horizon must be at least 2 observations.")
        if self.initial_price <= 0:
            raise ValueError("initial_price must be strictly positive.")
        if self.sigma <= 0:
            raise ValueError("sigma must be strictly positive.")
        if self.jump_intensity <= 0:
            raise ValueError("jump_intensity must be strictly positive.")
        if self.kappa <= 0:
            raise ValueError("kappa must be strictly positive.")
        if self.theta <= 0:
            raise ValueError("theta must be strictly positive.")
        if self.aggregation_horizon <= 0:
            raise ValueError("aggregation_horizon must be a positive integer.")
        if self.aggregation_horizon >= self.horizon:
            raise ValueError(
                "aggregation_horizon must be smaller than the simulation horizon."
            )


@dataclass
class SimulationResults:
    """Outputs of a complete simulation run."""

    processes: Dict[str, np.ndarray]
    z_scores: np.ndarray
    p_values: np.ndarray
    reference_distribution: np.ndarray
    simulation_time: float
    test_time: float

    @property
    def combined_paths(self) -> np.ndarray:
        """Convenience accessor for the stacked price processes."""

        if not self.processes:
            return np.empty((0, 0))

        return np.column_stack(tuple(self.processes.values()))


def _set_random_seed(seed: Optional[int]) -> Optional[Tuple]:
    """Store the current RNG state and seed the generators if required."""

    if seed is None:
        return None

    state = np.random.get_state()
    np.random.seed(seed)
    return state


def _restore_random_seed(state: Optional[Tuple]) -> None:
    """Restore the RNG state captured by :func:`_set_random_seed`."""

    if state is not None:
        np.random.set_state(state)


def simulate_price_processes(config: SimulationConfig) -> Tuple[Dict[str, np.ndarray], float]:
    """Generate price trajectories for the supported stochastic models."""

    rng_state = _set_random_seed(config.seed)
    start = time.perf_counter()

    try:
        simulator = PricePaths(config.num_series, config.horizon, config.initial_price)
        gbm = np.asarray(simulator.gbm_prices(mu=config.mu, sigma=config.sigma), dtype=float)
        merton = np.asarray(
            simulator.merton_prices(mu=config.mu, sigma=config.sigma, lambda_=config.jump_intensity),
            dtype=float,
        )
        heston = np.asarray(
            simulator.heston_prices(
                rf=config.risk_free_rate,
                k=config.kappa,
                theta=config.theta,
                sigma=config.sigma,
            ),
            dtype=float,
        )

        processes = {
            "gbm": gbm,
            "merton": merton,
            "heston": heston,
        }
    finally:
        _restore_random_seed(rng_state)

    elapsed = time.perf_counter() - start
    return processes, elapsed


def _iter_series(matrix: np.ndarray) -> Iterable[np.ndarray]:
    """Yield 1-D price series from an array of stacked paths."""

    data = np.asarray(matrix, dtype=float)
    if data.ndim == 1:
        yield data
        return

    for idx in range(data.shape[1]):
        yield data[:, idx]


def compute_vrt_statistics(
    paths: np.ndarray,
    q: int,
    *,
    heteroskedastic: bool,
    emh: Optional[EMH] = None,
) -> Tuple[np.ndarray, np.ndarray, float]:
    """Evaluate the VRT statistic for each provided price trajectory."""

    if q <= 0:
        raise ValueError("q must be a positive integer.")

    model = emh or EMH()
    start = time.perf_counter()
    z_values = []
    p_values = []

    for series in _iter_series(paths):
        z_score, p_value = model.vrt(
            X=series,
            q=q,
            heteroskedastic=heteroskedastic,
            centered=True,
            unbiased=True,
            annualize=True,
        )
        z_values.append(z_score)
        p_values.append(p_value)

    elapsed = time.perf_counter() - start
    return np.asarray(z_values, dtype=float), np.asarray(p_values, dtype=float), elapsed


def run_simulation(config: SimulationConfig) -> SimulationResults:
    """Execute the full pipeline: simulate paths, compute VRT, and collect outputs."""

    processes, simulation_time = simulate_price_processes(config)
    combined = np.column_stack(tuple(processes.values()))
    z_scores, p_values, test_time = compute_vrt_statistics(
        combined,
        config.aggregation_horizon,
        heteroskedastic=config.heteroskedastic,
    )

    rng = np.random.default_rng(config.seed)
    reference = rng.standard_normal(size=z_scores.size)

    return SimulationResults(
        processes=processes,
        z_scores=z_scores,
        p_values=p_values,
        reference_distribution=reference,
        simulation_time=simulation_time,
        test_time=test_time,
    )


def print_summary(results: SimulationResults) -> None:
    """Display a textual summary of the simulation outputs."""

    total_paths = sum(series.shape[1] if series.ndim > 1 else 1 for series in results.processes.values())
    print(
        "It took {:.3f} seconds to simulate {} trajectories across {} models.".format(
            results.simulation_time,
            total_paths,
            len(results.processes),
        )
    )
    print(
        "It took {:.3f} seconds to evaluate the VRT statistic for {} trajectories.".format(
            results.test_time,
            results.z_scores.size,
        )
    )
    print("Mean p-value: {:.4f}".format(float(np.mean(results.p_values))))


def plot_results(results: SimulationResults, *, mostrar: bool = True) -> None:
    """Render the density comparison using the visual helper class."""

    if results.z_scores.size == 0:
        print("No statistics available to plot.")
        return

    VRTVisuals().graficar_densidades(
        results.reference_distribution,
        results.z_scores,
        mostrar=mostrar,
    )


def build_parser() -> argparse.ArgumentParser:
    """Create the command line parser used by :func:`main`."""

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--series", type=int, default=10, help="Número de trayectorias a simular por modelo.")
    parser.add_argument("--horizon", type=int, default=100, help="Número de observaciones por trayectoria.")
    parser.add_argument("--initial-price", type=float, default=100.0, help="Precio inicial de cada trayectoria.")
    parser.add_argument("--mu", type=float, default=0.09, help="Media de las trayectorias de retorno.")
    parser.add_argument("--sigma", type=float, default=0.1, help="Volatilidad base de las trayectorias.")
    parser.add_argument(
        "--jump-intensity",
        type=float,
        default=50.0,
        help="Intensidad de saltos para el proceso de Merton.",
    )
    parser.add_argument("--kappa", type=float, default=0.1, help="Velocidad de reversión del modelo de Heston.")
    parser.add_argument("--theta", type=float, default=0.06, help="Nivel objetivo de varianza en Heston.")
    parser.add_argument("--rf", type=float, default=0.02, help="Tasa libre de riesgo para Heston.")
    parser.add_argument(
        "--aggregation",
        type=int,
        default=5,
        help="Horizonte de agregación para la estadística del VRT.",
    )
    parser.add_argument(
        "--homoskedastic",
        action="store_true",
        help="Evalúa la hipótesis nula homocedástica en lugar de la heterocedástica.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=None,
        help="Semilla para reproducibilidad.",
    )
    parser.add_argument(
        "--no-plot",
        action="store_true",
        help="Evita mostrar la comparación de densidades.",
    )
    return parser


def main() -> None:
    """Entry point for CLI execution."""

    parser = build_parser()
    args = parser.parse_args()

    config = SimulationConfig(
        num_series=args.series,
        horizon=args.horizon,
        initial_price=args.initial_price,
        mu=args.mu,
        sigma=args.sigma,
        jump_intensity=args.jump_intensity,
        kappa=args.kappa,
        theta=args.theta,
        risk_free_rate=args.rf,
        aggregation_horizon=args.aggregation,
        heteroskedastic=not args.homoskedastic,
        seed=args.seed,
    )

    results = run_simulation(config)
    print_summary(results)

    if not args.no_plot:
        plot_results(results)


if __name__ == "__main__":
    main()

