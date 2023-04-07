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
    
    def merton_prices(self, mu, sigma, lambda_jump, mu_jump, sigma_jump):
        """
        Genera una simulación de precios siguiendo un proceso de salto-difusión de Merton.

        Parámetros:
        mu (float): Tasa de rendimiento esperado de la acción.
        sigma (float): Volatilidad de la acción.
        lambda_jump (float): Intensidad de saltos.
        mu_jump (float): Media de la distribución log-normal de saltos.
        sigma_jump (float): Desviación estándar de la distribución log-normal de saltos.

        Retorna:
        numpy.ndarray: Array 1D de precios simulados.
        """

        # Validación de parámetros
        if not all(isinstance(param, (int, float)) for param in (mu, sigma, lambda_jump, mu_jump, sigma_jump)):
            raise ValueError("Los parámetros mu, sigma, lambda_jump, mu_jump y sigma_jump deben ser números")

        # Inicialización de los precios con el precio inicial
        prices = np.zeros(self.n + 1)
        prices[0] = self.s0

        # Generación de incrementos brownianos
        brownian_increments = np.random.normal(
            loc=0,
            scale=np.sqrt(self.dt),
            size=self.n
        )

        # Generación de saltos
        num_jumps = np.random.poisson(
            lam=lambda_jump * self.dt,
            size=self.n
        )

        jump_sizes = np.random.normal(
            loc=mu_jump,
            scale=sigma_jump,
            size=self.n
        )

        # Cálculo de precios utilizando el proceso de salto-difusión de Merton
        for t in range(1, self.n + 1):
            prices[t] = prices[t - 1] * np.exp((mu - 0.5 * sigma**2) * self.dt + sigma * brownian_increments[t - 1]) * np.exp(num_jumps[t - 1] * (np.exp(jump_sizes[t - 1]) - 1))

        return prices
    
	# -------------------------------------------
	# Vasicek Interest Rate Model
	# -------------------------------------------
    
    def vas_rates(self, kappa, theta, sigma):
        """
        Genera una simulación de tasas de interés siguiendo un proceso de Vasicek.

        Parámetros:
        kappa (float): Velocidad de reversión a la media.
        theta (float): Nivel de reversión a la media.
        sigma (float): Volatilidad de la tasa de interés.

        Retorna:
        numpy.ndarray: Array 1D de tasas de interés simuladas.
        """

        # Validación de parámetros
        if not all(isinstance(param, (int, float)) for param in (kappa, theta, sigma)):
            raise ValueError("Los parámetros kappa, theta y sigma deben ser números")

        # Inicialización de las tasas de interés con la tasa inicial
        rates = np.zeros(self.n + 1)
        rates[0] = self.s0

        # Generación de incrementos brownianos
        brownian_increments = np.random.normal(
            loc=0,
            scale=np.sqrt(self.dt),
            size=self.n
        )

        # Cálculo de tasas de interés utilizando el proceso de Vasicek
        for t in range(1, self.n + 1):
            rates[t] = rates[t - 1] + kappa * (theta - rates[t - 1]) * self.dt + sigma * brownian_increments[t - 1]

        return rates
    
	# -------------------------------------------
	# Cox Ingersoll Ross (CIR) stochastic proces - RATES
	# -------------------------------------------

    def cir_rates(self, kappa, theta, sigma):
        """
        Genera una simulación de tasas de interés siguiendo un proceso de Cox-Ingersoll-Ross (CIR).

        Parámetros:
        kappa (float): Velocidad de reversión a la media.
        theta (float): Nivel de reversión a la media.
        sigma (float): Volatilidad de la tasa de interés.

        Retorna:
        numpy.ndarray: Array 1D de tasas de interés simuladas.
        """

        # Validación de parámetros
        if not all(isinstance(param, (int, float)) for param in (kappa, theta, sigma)):
            raise ValueError("Los parámetros kappa, theta y sigma deben ser números")

        # Inicialización de las tasas de interés con la tasa inicial
        rates = np.zeros(self.n + 1)
        rates[0] = self.s0

        # Generación de incrementos brownianos
        brownian_increments = np.random.normal(
            loc=0,
            scale=np.sqrt(self.dt),
            size=self.n
        )

        # Cálculo de tasas de interés utilizando el proceso de Cox-Ingersoll-Ross
        for t in range(1, self.n + 1):
            rates[t] = rates[t - 1] + kappa * (theta - rates[t - 1]) * self.dt + sigma * np.sqrt(rates[t - 1]) * brownian_increments[t - 1]

            # Si la tasa de interés es negativa, se establece en cero
            rates[t] = max(rates[t], 0)

        return rates

    # -------------------------------------------
	# Heston Stochastic Volatility Process
	# -------------------------------------------
    
    def heston_prices(self, kappa, theta, sigma, rho, v0):
        """
        Genera una simulación de precios de activos siguiendo el modelo de Heston.

        Parámetros:
        kappa (float): Velocidad de reversión a la media de la varianza.
        theta (float): Nivel de reversión a la media de la varianza.
        sigma (float): Volatilidad de la varianza.
        rho (float): Correlación entre el movimiento del activo y su varianza.
        v0 (float): Valor inicial de la varianza.

        Retorna:
        numpy.ndarray: Array 1D de precios de activos simulados.
        """

        # Validación de parámetros
        if not all(isinstance(param, (int, float)) for param in (kappa, theta, sigma, rho, v0)):
            raise ValueError("Los parámetros kappa, theta, sigma, rho y v0 deben ser números")

        # Inicialización de los precios de activos y varianzas
        prices = np.zeros(self.n + 1)
        prices[0] = self.s0
        variances = np.zeros(self.n + 1)
        variances[0] = v0

        # Generación de incrementos brownianos correlacionados
        normal_sample = np.random.multivariate_normal(
            mean=[0, 0],
            cov=[[self.dt, rho * self.dt], [rho * self.dt, self.dt]],
            size=self.n
        )
        brownian_increments_s = normal_sample[:, 0]
        brownian_increments_v = normal_sample[:, 1]

        # Cálculo de precios y varianzas utilizando el modelo de Heston
        for t in range(1, self.n + 1):
            variances[t] = variances[t - 1] + kappa * (theta - variances[t - 1]) * self.dt + sigma * np.sqrt(variances[t - 1]) * brownian_increments_v[t - 1]
            variances[t] = max(variances[t], 0)  # Garantizar que la varianza no sea negativa

            prices[t] = prices[t - 1] * np.exp((self.r - 0.5 * variances[t - 1]) * self.dt + np.sqrt(variances[t - 1]) * brownian_increments_s[t - 1])

        return prices
    
    # -------------------------------------------
	# Ornstein–Uhlenbeck Process (Mean reverting)
	# -------------------------------------------
    
    def ou_prices(self, alpha, mu):
        """
        Genera una simulación de precios de activos siguiendo el modelo de Ornstein-Uhlenbeck.

        Parámetros:
        alpha (float): Velocidad de reversión a la media.
        mu (float): Nivel de reversión a la media.

        Retorna:
        numpy.ndarray: Array 1D de precios de activos simulados.
        """

        # Validación de parámetros
        if not all(isinstance(param, (int, float)) for param in (alpha, mu)):
            raise ValueError("Los parámetros alpha y mu deben ser números")

        # Inicialización de los precios de activos
        prices = np.zeros(self.n + 1)
        prices[0] = self.s0

        # Generación de incrementos brownianos
        brownian_increments = np.random.normal(loc=0, scale=np.sqrt(self.dt), size=self.n)

        # Cálculo de precios utilizando el modelo de Ornstein-Uhlenbeck
        for t in range(1, self.n + 1):
            prices[t] = prices[t - 1] + alpha * (mu - prices[t - 1]) * self.dt + brownian_increments[t - 1]

        return prices