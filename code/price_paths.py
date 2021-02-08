import numpy as np
from scipy.stats import halfnorm

class PricePaths(object):
    
    def __init__(self, n:int, T:int, s0:float):
        
        self.n = n                      # number of paths to generate
        self.T = T                      # number of observations to generate
        self.s0 = s0                    # initial price
        
        self.h = self.n / self.T        # step to move in each step
        self.r0 = self.s0 / 100         # Initial rate, based on the price
        

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
            bro_prices = self.__brownian_returns(mu, sigma, sto_vol)
            
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
            gbm_prices = self.__brownian_returns(mu, sigma, sto_vol)

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
            mert_prices = self.__merton_returns(mu, sigma, lambda_, sto_vol)
        
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
	# Helper methods
	# -------------------------------------------
    
    # Vasicek Interest Rate Model
    
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
        small_lambda = -(1.0/lambda__)
        pd = np.random.poisson(lam=lambda__, size=(self.T))
        
        # applying the psudo-code of the algorithm
        for i in range(self.T):
            t += small_lambda * np.log(np.random.uniform())
            if t > self.T:
                jumps[i:] = ( np.mean(pd)*np.random.uniform() ) * np.random.choice([-1, 1])
                # the t parameter is restituted to the original value
                # for several jumps in the future
                t = small_lambda
                break
                
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

#%% Run the simulation

if __name__=='__main__':
    
    import matplotlib.pyplot as plt
    import seaborn as sns
    
    n = 1                   # number of time series to simulate
    T = 1000                # number of steps
    s0 = 1                  # Initial price or initial short term rate
    
    sim = PricePaths(n, T, s0)
    
    mu = 0.05               # Long term mean return
    sigma = 0.05            # Volatility
    lam = 100               # Intensity of the Jump (Merton process)
    
    bro = sim.brownian_prices(mu, sigma)
    gbm = sim.gbm_prices(mu, sigma)
    merton = sim.merton_prices(mu, sigma, lam)
    
    all_proc = np.vstack((bro, gbm, merton))
    sns.heatmap(np.corrcoef(all_proc), xticklabels=False, yticklabels=False)
    plt.title('Correlation across the simulated instruments')
    plt.show()
    
    # Price plots
    plt.plot(bro, label='Brownian')
    plt.plot(gbm, label='GBM')
    plt.plot(merton, label='Merton')
    plt.title('Simulated price paths')
    plt.ylabel('price')
    plt.xlabel('step')
    plt.legend()
    plt.show()
    
    # Rate plots
    
    n = 1                   # number of time series to simulate
    T = 1000                # number of steps
    r0 = 3.0                # Initial short term rate
    
    sim = PricePaths(n, T, r0)
    
    mu = r0/100             # Long term rate %
    sigma = 0.01            # Volatility %
    lam = 0.7               # Reversion speed -> 0 <= lambda <= 1
    
    vas = sim.vas_rates(mu, sigma, lambda_=lam)
    
    plt.plot(vas, label='Vasicek')
    plt.plot(np.ones(sim.T) * mu, color='red')
    plt.title('Simulated rate paths')
    plt.ylabel('rate')
    plt.xlabel('step')
    plt.legend()
    plt.show()