import numpy as np
from scipy.stats import halfnorm

class PricePaths(object):
    
    def __init__(self, n:int, T:int, s0:float):

        if n <= 0 or T <= 0 or s0 <= 0:
            raise ValueError("Los parámetros n, T y s0 deben ser números positivos")

        self.n = n                      # number of paths to generate
        self.T = T                      # number of observations to generate
        self.s0 = s0                    # initial price
        
        self.h = self.n / self.T        # step to move in each step
        self.r0 = self.s0 / 100         # Initial rate, based on the price
        

    # -------------------------------------------
    # Brownian motion
    # -------------------------------------------

    def brownian_prices(self, mu, sigma):
        """
        Genera una simulación de precios siguiendo un movimiento browniano geométrico.

        Parámetros:
        mu (float): Tasa de rendimiento esperado de la acción.
        sigma (float): Volatilidad de la acción.

        Retorna:
        numpy.ndarray: Array 1D de precios simulados.
        """

        # Validación de parámetros
        if not (isinstance(mu, (int, float)) and isinstance(sigma, (int, float))):
            raise ValueError("Los parámetros mu y sigma deben ser números")

        # Inicialización de los precios con el precio inicial
        prices = np.zeros(self.n + 1)
        prices[0] = self.s0

        # Generación de incrementos brownianos
        brownian_increments = np.random.normal(
            loc=(mu - 0.5 * sigma**2) * self.dt,
            scale=np.sqrt(self.dt) * sigma,
            size=self.n
        )

        # Cálculo de precios
        for t in range(1, self.n + 1):
            prices[t] = prices[t - 1] * np.exp(brownian_increments[t - 1])

        return prices

	# -------------------------------------------
	# Geometric Brownian motion
	# -------------------------------------------
    
    def gbm_prices(self, mu, sigma):
        """
        Genera una simulación de precios siguiendo un proceso de movimiento browniano geométrico (GBM).

        Parámetros:
        mu (float): Tasa de rendimiento esperado de la acción.
        sigma (float): Volatilidad de la acción.

        Retorna:
        numpy.ndarray: Array 1D de precios simulados.
        """

        # Validación de parámetros
        if not (isinstance(mu, (int, float)) and isinstance(sigma, (int, float))):
            raise ValueError("Los parámetros mu y sigma deben ser números")

        # Inicialización de los precios con el precio inicial
        prices = np.zeros(self.n + 1)
        prices[0] = self.s0

        # Generación de incrementos brownianos
        brownian_increments = np.random.normal(
            loc=0,
            scale=np.sqrt(self.dt),
            size=self.n
        )

        # Cálculo de precios utilizando GBM
        for t in range(1, self.n + 1):
            prices[t] = prices[t - 1] * np.exp((mu - 0.5 * sigma**2) * self.dt + sigma * brownian_increments[t - 1])

        return prices  
    
	# -------------------------------------------
	# Merton Jump Diffusion Stochastic Process
	# -------------------------------------------
    

    
	# -------------------------------------------
	# Vasicek Interest Rate Model
	# -------------------------------------------
    

    
	# -------------------------------------------
	# Cox Ingersoll Ross (CIR) stochastic proces - RATES
	# -------------------------------------------
    

    # -------------------------------------------
	# Heston Stochastic Volatility Process
	# -------------------------------------------
    

    
    # -------------------------------------------
	# Ornstein–Uhlenbeck Process (Mean reverting)
	# -------------------------------------------
    
