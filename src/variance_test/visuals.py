"""Visual helpers for the variance ratio test simulations."""

from typing import Iterable, Optional, Sequence, Tuple

import numpy as np
import matplotlib.pyplot as plt
from .core import EMH
from .price_paths import PricePaths

__all__ = ["VRTVisuals"]


class VRTVisuals:
    """Colección de utilidades para visualizar simulaciones del VRT."""

    def graficar_densidades(
        self,
        ref_stats: Sequence[float],
        z_stats: Sequence[float],
        *,
        ax: Optional[plt.Axes] = None,
        mostrar: bool = True,
        etiquetas: Tuple[str, str] = (
            "Distribución de referencia",
            "Puntajes Z*",
        ),
    ) -> plt.Axes:
        """Comparar densidades de referencia y estadísticas simuladas."""

        ref_values = np.asarray(ref_stats, dtype=float)
        z_values = np.asarray(z_stats, dtype=float)

        if ref_values.ndim != 1 or ref_values.size < 2:
            raise ValueError("ref_stats debe contener al menos dos valores unidimensionales.")
        if z_values.ndim != 1 or z_values.size < 2:
            raise ValueError("z_stats debe contener al menos dos valores unidimensionales.")

        axis = ax or plt.gca()
        axis.hist(
            ref_values,
            bins="auto",
            density=True,
            color="r",
            alpha=0.4,
            label=etiquetas[0],
        )
        axis.hist(
            z_values,
            bins="auto",
            density=True,
            color="g",
            alpha=0.4,
            label=etiquetas[1],
        )
        axis.set_title(
            "Comparación de densidades entre una referencia normal (rojo)\n"
            "y los puntajes Z* obtenidos de las trayectorias analizadas"
        )
        axis.legend()

        if mostrar:
            plt.show()

        return axis

    def graficar_estadisticas(
        self,
        proceso: str = "brownian",
        rango_q: Sequence[int] = (5, 10),
        *,
        total_muestras: int = 500,
        precio_inicial: float = 1.0,
        tipo_estadistica: str = "mr",
        numero_caminos: int = 250,
        mostrar: bool = True,
        **kwargs,
    ) -> Tuple[plt.Figure, np.ndarray]:
        """Generar diagramas de dispersión para diferentes horizontes de agregación."""

        if numero_caminos < 1:
            raise ValueError("Se requiere al menos una trayectoria para generar estadísticas.")

        if len(rango_q) != 2:
            raise ValueError("rango_q debe contener exactamente dos horizontes a comparar.")

        q1, q2 = (int(q) for q in rango_q)
        if q1 <= 0 or q2 <= 0:
            raise ValueError("Los horizontes de agregación deben ser enteros positivos.")
        if total_muestras <= max(q1, q2):
            raise ValueError("total_muestras debe exceder el mayor horizonte de agregación.")

        sims = PricePaths(n=numero_caminos, T=total_muestras, s0=precio_inicial)

        simuladores = {
            "brownian": sims.brownian_prices,
            "gbm": sims.gbm_prices,
            "gmb": sims.gbm_prices,
            "merton": sims.merton_prices,
        }

        simulador = simuladores.get(proceso.lower())
        if simulador is None:
            raise ValueError(
                "Proceso desconocido. Selecciona uno de: brownian, gbm o merton."
            )

        emh = EMH()
        estadisticas = {
            "md": emh._EMH__md,
            "mr": emh._EMH__mr,
        }

        estadistica = estadisticas.get(tipo_estadistica.lower())
        if estadistica is None:
            raise ValueError("Estadística no válida; utiliza 'md' o 'mr'.")

        def generar_estadisticas(q: int, unbiased: bool) -> Iterable[float]:
            trayectorias = np.asarray(simulador(**kwargs), dtype=float)
            if trayectorias.ndim == 1:
                trayectorias = trayectorias[np.newaxis, :]
            elif trayectorias.shape[0] == total_muestras:
                trayectorias = trayectorias.T
            elif trayectorias.shape[1] != total_muestras:
                raise ValueError("Las trayectorias simuladas no tienen la longitud esperada.")

            return [
                estadistica(X=serie, q=q, unbiased=unbiased)
                for serie in trayectorias
            ]

        stats_q1 = generar_estadisticas(q1, True)
        stats_q2 = generar_estadisticas(q2, True)
        stats_vol_q1 = generar_estadisticas(q1, False)
        stats_vol_q2 = generar_estadisticas(q2, False)

        fig, axes = plt.subplots(2, 2, figsize=(14, 12))
        fig.suptitle(
            f"Valores de {tipo_estadistica.upper()} para trayectorias {proceso.capitalize()}",
            fontsize=16,
        )

        axes[0, 0].plot(stats_q1, marker="o", linestyle="None", alpha=0.7)
        axes[0, 0].set_title("Sesgo corregido")
        axes[0, 0].set_ylabel(f"{tipo_estadistica.upper()} (q={q1})")

        axes[0, 1].plot(stats_q2, marker="o", linestyle="None", alpha=0.7)
        axes[0, 1].set_title("Sesgo corregido")
        axes[0, 1].set_ylabel(f"{tipo_estadistica.upper()} (q={q2})")

        axes[1, 0].plot(stats_vol_q1, marker="o", linestyle="None", alpha=0.7)
        axes[1, 0].set_title("Sin corrección")
        axes[1, 0].set_ylabel(f"{tipo_estadistica.upper()} (q={q1})")

        axes[1, 1].plot(stats_vol_q2, marker="o", linestyle="None", alpha=0.7)
        axes[1, 1].set_title("Sin corrección")
        axes[1, 1].set_ylabel(f"{tipo_estadistica.upper()} (q={q2})")

        for axis in axes.flat:
            axis.set_xlabel("Trayectoria")

        fig.tight_layout(rect=(0, 0, 1, 0.97))

        if mostrar:
            plt.show()

        return fig, axes
