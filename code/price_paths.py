import numpy as np
from scipy.stats import halfnorm

class PricePaths(object):
    
    def __init__(self, n:int, T:int, s0:float):
        
        self.n = n      # number of paths to generate
        self.T = T      # number of observations to generate
        self.s0 = s0    # initial price
        
        self.h = self.n / self.T
        
        # DEBBUING SPACE
        self.t = None
        self.jumps = None
        self.lambda__ = None
        self.small_lambda = None
        self.pd = None
        
        self.geometric_brownian_motion = None
        self.jumps_diffusion = None
        self.mert_prices_ = None

	# -------------------------------------------
	# The Brownian Motion Stochastic Process (Wiener Process)
	# -------------------------------------------
    
    def brownian_prices(self, mu:float, sigma:float, sto_vol:bool=False):
        
        # preallocate the data
        bro_prices = self.__zeros()
        
        # check the size of the output matrix
        if self.n > 1:
            for i in range(self.n):
                # simulate n price paths
                bro_prices[:, i] = self.__brownian_returns(mu, sigma, sto_vol)
        else:
            # case for only 1 simulation
            bro_prices = self.__brownian_returns(mu, sigma, sto_vol).reshape(-1, 1)
            
        return bro_prices
    
	# -------------------------------------------
	# Geometric Brownian motion
	# -------------------------------------------
    
    def gbm_prices(self, mu:float, sigma:float, sto_vol:bool=True):
        # preallocate the data
        gbm_prices = self.__zeros()

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
	# Merton Jump Diffusion Stochastic Process
	# -------------------------------------------
    
    def merton_prices(self, mu:float, sigma:float, lambda_:int, sto_vol:bool=False):
        # preallocate the data
        mert_prices = self.__zeros()
        
        # check the size of the output matrix
        if self.n > 1:
            for i in range(self.n):
                # simulate n price paths
                mert_prices[:, i] = self.__merton_returns(mu, sigma, lambda_, sto_vol)
        else:
            # case for only 1 simulation
            self.mert_prices_ = self.__merton_returns(mu, sigma, lambda_, sto_vol)
            return self.mert_prices_

	# -------------------------------------------
	# Helper methods
	# -------------------------------------------
    
    # Merton Jump Diffusion Stochastic Process
    
    def __jumps_diffusion(self, lambda_:int):
        self.t = 0
        self.jumps = np.zeros((self.T, 1))
        self.lambda__ = lambda_ / self.T
        self.small_lambda = -(1.0/self.lambda__)
        self.pd = np.random.poisson(lam=self.lambda__, size=(self.T))
        
        # applying the psudo-code of the algorithm
        for i in range(self.T):
            self.t += self.small_lambda * np.log(np.random.uniform())
            if self.t > self.T:
                self.jumps[i:] = (np.mean(self.pd)*np.random.uniform()+np.std(self.pd)) * np.random.choice([-1, 1])
                # the t parameter is restituted to the original value
                # for several jumps in the future
                self.t = self.small_lambda
                
        return self.jumps.reshape(-1, 1)
                
    def __merton_returns(self, mu:float, sigma:float, lambda_:int, sto_vol:bool):
        self.geometric_brownian_motion = self.__brownian_returns(mu, sigma, sto_vol).reshape(-1, 1)
        self.jump_diffusion = self.__jumps_diffusion(lambda_)
        return self.geometric_brownian_motion + self.jump_diffusion
    
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
    
    def __random_disturbance(self, sto_vol:bool, rd_mu:float, rd_sigma:float):
        
        if not sto_vol:
            return np.random.normal(size=(self.T, 1))
        else:
            # error handling for scale < 0, because negative volatilities 
            # doesnt makes sense.
            return np.random.normal(loc=rd_mu * self.h,
                                    scale=rd_sigma * halfnorm.rvs(1) * np.sqrt(self.h),
                                    size=(self.T, 1)) * 100
        return
        
    def __zeros(self):
        return np.zeros((self.T, self.n))


if __name__=='__main__':
    
    import matplotlib.pyplot as plt
    
    n = 1
    T = 1000
    s0 = 1
    
    sim = PricePaths(n, T, s0)
    
    mu = 0.05
    sigma = 0.1
    lam = T/20
    
    bro = sim.brownian_prices(mu, sigma)
    gbm = sim.gbm_prices(mu, sigma)
    merton = sim.merton_prices(mu, sigma, lambda_=lam)
    
    plt.plot(bro, label='Brownian')
    plt.plot(gbm, label='GBM')
    plt.plot(merton, label='Merton')
    plt.title('Simulated price paths')
    plt.ylabel('price')
    plt.xlabel('step')
    plt.legend()