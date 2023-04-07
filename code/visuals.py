import matplotlib.pyplot as plt
import seaborn as sns
sns.set_style('whitegrid')

from variance_test import EMH
from price_paths import PricePaths


class VRTVisuals:

    def graficar_densidades(self, ref_stats, z_stats):
        """
        Grafica las densidades para una serie de datos

        Los valores ref_stats (distribución normal) se muestran en rojo, ya que representan la luz que 'detiene'
        el rechazo de la hipótesis nula. Se hace una representación análoga para los puntajes z del test.
        """
        # Graficar los valores de la distribución normal
        sns.kdeplot(ref_stats, shade=True, color="r")
        # Graficar los valores de la estadística
        sns.kdeplot(z_stats, shade=True, color="g")
        titulo = "Comparación de densidades de una variable con distribución normal (rojo)\ncontra los puntajes Z* calculados para la serie de precios logarítmicos analizada"
        plt.title(titulo)
        plt.show()

    def graficar_estadisticas(self, proceso: str = 'brownian', rango_q: list = [5, 10],
                               total_muestras: int = 500, precio_inicial: float = 1.0,
                               tipo_estadistica: str = 'mr', **kwargs):
        """
        Genera las gráficas para las DIFERENCIAS UTILIZANDO EL ESTIMADOR DE MUESTRAS SUPERPUESTAS

        """
        # Crear el objeto para las trayectorias de precios
        sims = PricePaths(n=1, T=total_muestras * 2, s0=precio_inicial)

        # Diccionario para asignar el simulador
        simuladores = {
            'brownian': sims.brownian_prices,
            'gmb': sims.gbm_prices,
            'merton': sims.merton_prices
        }

        # Obtener el simulador correspondiente al proceso
        simulador = simuladores.get(proceso)
        if not simulador:
            print(f'Tu opción es {proceso}. Por favor, selecciona una de estas: brownian, gbm, merton')
            return None

        # Manejo de errores
        if len(rango_q) != 2:
            print('Por favor, selecciona al menos 2 rangos para analizar, por ejemplo, [3,6]')
            return None

        # Estableciendo la estadística deseada
        estadisticas = {
            'md': EMH()._EMH__md,
            'mr': EMH()._EMH__mr
        }

        estadistica = estadisticas.get(tipo_estadistica)
        if not estadistica:
            print('Estadística no válida, por favor, intenta con md o mr')
            return None

        nombre_estadistica = tipo_estadistica.capitalize()

        # Función interna para generar estadísticas
        def generar_estadisticas(q, unbiased):
            return [estadistica(X=simulador(**kwargs), q=q, unbiased=unbiased) for muestra in range(q, total_muestras)]

        # Generando las estadísticas
        statsQ1 = generar_estadisticas(rango_q[0], True)
        statsQ2 = generar_estadisticas(rango_q[1], True)
        statsVolQ1 = generar_estadisticas(rango_q[0], False)
        statsVolQ2 = generar_estadisticas(rango_q[1], False)

        # Creando las gráficas
        fig, axes = plt.subplots(2, 2, figsize=(15, 15))
        fig.suptitle(f'Valores de {nombre_estadistica} para las trayectorias de precios {proceso.capitalize()}', fontsize=16)

        # Valores de MD sin Volatilidad Estocástica
        axes[0, 0].plot(statsQ1, marker='o', linestyle='None')
        axes[0, 0].set_title(f'Valores de {nombre_estadistica} sin Volatilidad Estocástica')
        axes[0, 0].set_ylabel(f'Valores de {nombre_estadistica}(q={rango_q[0]})')

        axes[0, 1].plot(statsQ2, marker='o', linestyle='None')
        axes[0, 1].set_title(f'Valores de {nombre_estadistica} sin Volatilidad Estocástica')
        axes[0, 1].set_ylabel(f'Valores de {nombre_estadistica}(q={rango_q[1]})')

        # Valores de MD con Volatilidad Estocástica
        axes[1, 0].plot(statsVolQ1, marker='o', linestyle='None')
        axes[1, 0].set_title(f'Valores de {nombre_estadistica} con Volatilidad Estocástica')
        axes[1, 0].set_ylabel(f'Valores de {nombre_estadistica}(q={rango_q[0]})')

        axes[1, 1].plot(statsVolQ2, marker='o', linestyle='None')
        axes[1, 1].set_title(f'Valores de {nombre_estadistica} con Volatilidad Estocástica')
        axes[1, 1].set_ylabel(f'Valores de {nombre_estadistica}(q={rango_q[1]})')

        plt.show()

        return