import numpy as np

class PricePaths(object):
    
    def __init__(self, n:int, T:int, s0:float):
        
        self.n = n      # number of paths to generate
        self.T = T      # number of observations to generate
        self.s0 = s0    # initial price
        
        self.h = self.n / self.T

	# -------------------------------------------
	# The Brownian Motion Stochastic Process (Wiener Process)
	# -------------------------------------------
    
    def brownian_prices(self, mu:float, sigma:float, sto_vol:bool=False):
        
        # preallocate the data
        bro_prices = np.zeros((self.T, self.n))
        
        # check the size of the output matrix
        if self.n > 1:
            for i in range(self.n):
                # simulate n price paths
                bro_prices[:, i] = self.__brownian_returns(mu, sigma, sto_vol)
        else:
            # case for only 1 simulation
            bro_prices = self.__brownian_returns(mu, sigma, sto_vol).reshape(-1, 1)
            
        return bro_prices
    
    def gbm_prices(self, mu:float, sigma:float, sto_vol:bool=True):
        # preallocate the data
        gbm_prices = np.zeros((self.T, self.n))
        
        # check the size of the output matrix
        if self.n > 1:
            for i in range(self.n):
                # simulate n price paths
                gbm_prices[:, i] = self.__brownian_returns(mu, sigma, sto_vol)
        else:
            # case for only 1 simulation
            gbm_prices = self.__brownian_returns(mu, sigma, sto_vol).reshape(-1, 1)
            
        return gbm_prices

	# -------------------------------------------
	# Helper methods
	# -------------------------------------------

	# Brownian Motion Stochastic Process (Wiener Process)
    
    def __brownian_discrete(self, mu:float, sigma:float, st:float, vol:float):
        return ( mu * st * self.h ) + ( sigma * st * np.sqrt(self.h) * vol )
    
    def __brownian_returns(self, mu:float, sigma:float, sto_vol:bool):
        # preallocate the volatility
        volatility = self.__random_disturbance(sto_vol, rd_mu=mu, rd_sigma=sigma)
        bro_returns = np.zeros((self.T))
        bro_returns[0] = self.s0
        for t in range(1, self.T):
            bro_returns[t] = bro_returns[t-1] + \
                self.__brownian_discrete(mu, sigma, st=bro_returns[t-1], vol=volatility[t])
                
        return bro_returns
    
	# -------------------------------------------
	# General utilities
	# -------------------------------------------
    
    def __random_disturbance(self, sto_vol:bool, rd_mu:float=0.0, rd_sigma:float=1.0):
        
        if not sto_vol:
            return np.random.normal(size=(self.T, 1))
        else:
            # error handling for scale < 0, because negative volatilities 
            # doesn makes sense.
            return np.random.normal(loc=rd_mu * self.h,
                                    scale=np.abs(rd_sigma * np.random.normal() * np.sqrt(self.h)),
                                    size=(self.T, 1))
