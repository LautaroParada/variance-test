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
        self.dt = self.h
        self.r0 = self.s0 / 100         # Initial rate, based on the price
        
    # -------------------------------------------
    # Geometric Brownian motion
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
    
    def merton_prices(self, mu:float, sigma:float, lambda_:int):
        """
        This model superimposes a jump component on a diffusion component.
        The diffusion component is the familiar geometric Brownian motion.
        The jump component is composed of lognormal jumps driven by a Poisson
        process.
            - It models the sudden changes in the stock price because of the 
            arrival of important new information.
            - The lognormal jumps, and the Poisson process are assumed to
            be independent.
        
        The jump event is governed by a compound Poisson process qt with 
        intensity λ, where k denotes the magnitude of the random jump.

        Parameters
        ----------
        mu : float
            mean returns for the instrument (first moment).
        sigma : float
            volatility of the instrument (second moment).
        lambda_ : int
            intensity/frequency of the jump.

        Returns
        -------
        mert_prices : numy array
            prices with at least a jump.

        """
        # preallocate the data
        mert_prices = self.__zeros()
        
        # check the size of the output matrix
        if self.n > 1:
            for i in range(self.n):
                # simulate n price paths
                mert_prices[:, i] = self.__merton_returns(mu, sigma, lambda_, sto_vol=False)
        else:
            # case for only 1 simulation
            mert_prices = self.__merton_returns(mu, sigma, lambda_, sto_vol=False)
        
        return mert_prices
    
    # -------------------------------------------
    # Vasicek Interest Rate Model
    # -------------------------------------------
    
    def vas_rates(self, mu:float, sigma:float, lambda_:float, sto_vol:bool=False):
        
        ou_rates = self.__zeros()
        if self.n > 1:
            for i in range(self.n):
                ou_rates[:, i] = self.__vas_returns(mu, sigma, lambda_, sto_vol)
        
        else:
            ou_rates = self.__vas_returns(mu, sigma, lambda_, sto_vol)
        
        return ou_rates
    
    # -------------------------------------------
    # Cox Ingersoll Ross (CIR) stochastic proces - RATES
    # -------------------------------------------
    
    def cir_rates(self, mu:float, sigma:float, lambda_:float, sto_vol:bool=False):
        cir_rates_ = self.__zeros()
        
        if self.n > 1:
            for i in range(self.n):
                cir_rates_[:, i] = self.__cir_return(mu, sigma, lambda_, sto_vol)
        else:
            cir_rates_ = self.__cir_return(mu, sigma, lambda_, sto_vol)
        
        return cir_rates_
    
   # -------------------------------------------
    # Heston Stochastic Volatility Process
    # -------------------------------------------
    
    def heston_prices(self, mu:float, k:float, theta:float, sigma:float):
        hes_prices = self.__zeros()
        
        if self.n > 1:
            for i in range(self.n):
                hes_prices[:, i] = self.__heston_returns(mu, k, theta, sigma)
                
        else:
            hes_prices = self.__heston_returns(mu, k, theta, sigma)
            
        return hes_prices
    
   # -------------------------------------------
    # Ornstein–Uhlenbeck Process (Mean reverting)
    # -------------------------------------------
    
    def ou_prices(self, mu:float, sigma:float, lambda_:float, sto_vol:bool=False):
        ou_prices = self.__zeros()
        if self.n > 1:
            for i in range(self.n):
                ou_prices[:, i] = self.__ou_returns(mu, sigma, lambda_, sto_vol)
        
        else:
            ou_prices = self.__ou_returns(mu, sigma, lambda_, sto_vol)
        
        return ou_prices

    # -------------------------------------------
    # Helper methods
    # -------------------------------------------
    
    # Ornstein–Uhlenbeck Process (Mean reverting)
    
    def __ou_discrete(self, mu:float, sigma:float, lambda_:float, st:float, vol:float):
        return np.exp(-lambda_*self.h)*st + (1-np.exp(-lambda_*self.h))*mu + sigma*( (1-np.exp(-2*lambda_*self.h)) / (2*lambda_)) * vol
    
    def __ou_returns(self, mu:float, sigma:float, lambda_:float, sto_vol:float):
        volatility = self.__random_disturbance(sto_vol, rd_mu=mu, rd_sigma=sigma)
        ou_rets = np.zeros(self.T)
        ou_rets[0] = self.s0
        
        for t in range(1, self.T):
            ou_rets[t] = ou_rets[t] + self.__ou_discrete(mu=mu*100, sigma=sigma*100, lambda_=lambda_, st=ou_rets[t-1], vol=volatility[t])
        
        return ou_rets      
    
    # Heston Stochastic Volatility Process
    
    def __heston_dis_vol(self, k:float, theta:float, vt:float, sigma:float, w2:float):
        # heston mean reverting volatility recurrence
        return k * (sigma - np.abs(vt)) * self.h + theta * np.sqrt(np.abs(vt) * self.h) * w2
    
    def __heston_discrete(self, mu:float, st:float, V, w1):
        # Discrete form of the Heston model
        return mu * st * self.h + np.sqrt(np.abs(V)) * st * np.sqrt(self.h) * w1
    
    def __heston_returns(self, mu:float, k:float, theta:float, sigma:float):
        
        # integrate a random correlation level
        corr_wn1, corr_wn2 = self.__corr_noise()
        
        # integrating the mean reverting volatility of two Wiener processes
        dw2 = np.zeros(self.T)
        dw2[0] = corr_wn2[0]
        
        for t in range(1, self.T):
            dw2[t] = corr_wn2[t-1] + self.__heston_dis_vol(k=k, 
                                                       theta=theta, 
                                                       vt=dw2[t-1], 
                                                       sigma=sigma, 
                                                       w2=corr_wn2[t])
            
        # creating the actual data for the process
        heston_ret = np.zeros(self.T)
        heston_ret[0] = self.s0
        
        for t in range(1, self.T):
            heston_ret[t] = heston_ret[t-1] + self.__heston_discrete(mu=mu,
                                                                     st=heston_ret[t-1], 
                                                                     V=dw2[t-1], 
                                                                     w1=corr_wn1[t])
            
        return heston_ret.ravel()
    
    # Cox Ingersoll Ross
    
    def __cir_discrete(self, mu:float, sigma:float, lambda_:float, xt:float, vol:float):
        return lambda_ * (mu-xt) * self.h + sigma * np.sqrt(xt*self.h) * vol
    
    def __cir_return(self, mu:float, sigma:float, lambda_:float, sto_vol:bool):
        
        volatility = self.__random_disturbance(sto_vol, mu, sigma)
        cir_ret = np.zeros(self.T)
        
        cir_ret[0] = self.r0
            
        for t in range(1, self.T):
            cir_ret[t] = cir_ret[t-1] + self.__cir_discrete(mu, sigma, lambda_, cir_ret[t-1], volatility[t])
            
        return cir_ret
    
    # Vasicek Interest Rate Model and Ornstein–Uhlenbeck Process - Mean reverting
    
    def __vas_discrete(self, mu:float, sigma:float, lambda_:float, rt:float, vol:float):
        
        return lambda_ * (mu-rt) * self.h + sigma * np.sqrt(self.h) * vol
    
    def __vas_returns(self, mu:float, sigma:float, lambda_:float, sto_vol:float):
        volatility = self.__random_disturbance(sto_vol, rd_mu=mu, rd_sigma=sigma)
        vas_rets = np.zeros((self.T))
        vas_rets[0] = self.r0
        
        for t in range(1, self.T):
            vas_rets[t] = vas_rets[t-1] + self.__vas_discrete(mu=mu, sigma=sigma, lambda_=lambda_, rt=vas_rets[t-1], vol=volatility[t])
            
        return vas_rets
        
    # Merton Jump Diffusion Stochastic Process
    
    def __jumps_diffusion(self, lambda_:int):
        t = 0
        jumps = np.zeros((self.T, 1))
        lambda__ = lambda_ / self.T
        small_lambda = -(1.0/lambda__) # k or magnitude of the jump
        pd_ = np.random.poisson(lam=lambda__, size=(self.T))
        
        # applying the psudo-code of the algorithm
        for i in range(self.T):
            t += small_lambda * np.log(np.random.uniform())
            if t >= self.T:
                jumps[i:] = ( (np.mean(pd_) + np.std(pd_)) * np.random.uniform() ) * np.random.choice([-1, 1])
                # the t parameter is restituted to the original value
                # for several jumps in the future
                t = 0
                
        return jumps.reshape(-1, 1)
                
    def __merton_returns(self, mu:float, sigma:float, lambda_:int, sto_vol:bool):
        geometric_brownian_motion = self.__brownian_returns(mu, sigma, sto_vol).reshape(-1, 1)
        jump_diffusion = self.__jumps_diffusion(lambda_)
        return (geometric_brownian_motion + jump_diffusion).ravel()
    
    # Brownian Motion Stochastic Process (Wiener Process)
    
    def __brownian_discrete(self, mu:float, sigma:float, st:float, vol:float):
        return ( mu * st * self.h ) + ( sigma * st * np.sqrt(self.h) * vol )
    
    def __brownian_returns(self, mu:float, sigma:float, sto_vol:bool):
        # preallocate the volatility
        volatility = self.__random_disturbance(sto_vol, rd_mu=mu, rd_sigma=sigma)
        bro_returns = np.zeros((self.T))
        # initial condition X(0) = X_0
        bro_returns[0] = self.s0
        for t in range(1, self.T):
            bro_returns[t] = bro_returns[t-1] + \
                self.__brownian_discrete(mu, sigma, st=bro_returns[t-1], vol=volatility[t])
                
        return bro_returns
    
    # -------------------------------------------
    # General utilities
    # -------------------------------------------
    
    def __random_disturbance(self, sto_vol:bool, rd_mu:float, rd_sigma:float):
        
        # negative volatility doesnt make sense
        if not sto_vol:
            return np.random.normal(size=(self.T, 1))
        else:
            return np.random.normal(loc=rd_mu * self.h,
                                    scale=rd_sigma * halfnorm.rvs(1) * np.sqrt(self.h),
                                    size=(self.T, 1)) * 100
        return
        
    def __zeros(self):
        return np.zeros((self.T, self.n))
    
    def __corr_noise(self):
        #fuente https://lup.lub.lu.se/luur/download?func=downloadFile&recordOId=1436827&fileOId=1646914
        
        # generate two uncorrelated Brownian processes
        z1 = self.__random_disturbance(sto_vol=False, rd_mu=0, rd_sigma=1)
        z2 = self.__random_disturbance(sto_vol=False, rd_mu=0, rd_sigma=1)
        
        # randomly create a correlation coeficient
        rho = np.random.uniform(low=0.5, high=0.7) * np.random.choice([-1, 1])
        
        z_tV = ( rho * z1 ) + ( np.sqrt( 1 - rho**2 ) * z2 )
        
        return z1, z_tV