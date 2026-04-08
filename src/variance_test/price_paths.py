import numpy as np
from scipy.stats import halfnorm


class PricePaths(object):

    def __init__(self, n: int, T: int, s0: float):
        if n < 1:
            raise ValueError("Number of paths 'n' must be positive")
        if T < 1:
            raise ValueError("Number of observations 'T' must be positive")

        self.n = n                      # number of paths to generate
        self.T = T                      # number of observations to generate
        self.s0 = s0                    # initial price

        # Use a time step that depends on the observation horizon instead of
        # the number of simulated paths. When only one observation is
        # requested the step defaults to unity.
        self.h = 1.0 / max(self.T - 1, 1)
        self.r0 = self.s0 / 100         # Initial rate, based on the price

    # -------------------------------------------
    # The Brownian Motion Stochastic Process (Wiener Process)
    # -------------------------------------------

    def brownian_prices(self, mu: float, sigma: float):
        return self.__generate_paths(lambda: self.__brownian_returns(mu, sigma, sto_vol=True))

    # -------------------------------------------
    # Geometric Brownian motion
    # -------------------------------------------

    def gbm_prices(self, mu: float, sigma: float):
        return self.__generate_paths(lambda: self.__brownian_returns(mu, sigma, sto_vol=False))

    # -------------------------------------------
    # Merton Jump Diffusion Stochastic Process
    # -------------------------------------------

    def merton_prices(self, mu: float, sigma: float, lambda_: int, sto_vol: bool = False):
        return self.__generate_paths(lambda: self.__merton_returns(mu, sigma, lambda_, sto_vol))

    # -------------------------------------------
    # Vasicek Interest Rate Model
    # -------------------------------------------

    def vas_rates(self, mu: float, sigma: float, lambda_: float, sto_vol: bool = False):
        return self.__generate_paths(lambda: self.__vas_returns(mu, sigma, lambda_, sto_vol))

    # -------------------------------------------
    # Cox Ingersoll Ross (CIR) stochastic proces - RATES
    # -------------------------------------------

    def cir_rates(self, mu: float, sigma: float, lambda_: float, sto_vol: bool = False):
        return self.__generate_paths(lambda: self.__cir_return(mu, sigma, lambda_, sto_vol))

    # -------------------------------------------
    # Heston Stochastic Volatility Process
    # -------------------------------------------

    def heston_prices(self, rf: float, k: float, theta: float, sigma: float, sto_vol: bool = False):
        return self.__generate_paths(lambda: self.__heston_returns(rf, k, theta, sigma, sto_vol))

    # -------------------------------------------
    # Ornstein–Uhlenbeck Process (Mean reverting)
    # -------------------------------------------

    def ou_prices(self, mu: float, sigma: float, lambda_: float, sto_vol: bool = False):
        return self.__generate_paths(lambda: self.__ou_returns(mu, sigma, lambda_, sto_vol))

    # -------------------------------------------
    # Helper methods
    # -------------------------------------------

    # Ornstein–Uhlenbeck Process (Mean reverting)

    def __ou_discrete(self, mu: float, sigma: float, lambda_: float, st: float, vol: float):
        return (
            np.exp(-lambda_ * self.h) * st
            + (1 - np.exp(-lambda_ * self.h)) * mu
            + sigma * ((1 - np.exp(-2 * lambda_ * self.h)) / (2 * lambda_)) * vol
        )

    def __ou_returns(self, mu: float, sigma: float, lambda_: float, sto_vol: float):
        volatility = self.__random_disturbance(sto_vol, rd_mu=mu, rd_sigma=sigma)
        ou_rets = np.zeros(self.T)
        ou_rets[0] = self.s0

        for t in range(1, self.T):
            ou_rets[t] = ou_rets[t] + self.__ou_discrete(
                mu=mu * 100,
                sigma=sigma * 100,
                lambda_=lambda_,
                st=ou_rets[t - 1],
                vol=volatility[t],
            )

        return ou_rets

    # Heston Stochastic Volatility Process

    def __heston_dis_vol(self, k: float, theta: float, vt: float, sigma: float, w2: float):
        # heston mean reverting volatility recurrence
        return k * (theta - vt) * self.h + sigma * np.sqrt(np.abs(vt) * self.h) * w2

    def __heston_discrete(self, rf: float, st: float, V, w1):
        # Discrete form of the Heston model
        return rf * st * self.h + np.sqrt(np.abs(V) * self.h) * st * w1

    def __heston_returns(self, rf: float, k: float, theta: float, sigma: float, sto_vol: bool):

        # integrate a random correlation level
        corr_wn1, corr_wn2 = self.__corr_noise()

        # integrating the mean reverting volatility
        wn2 = self.__random_disturbance(sto_vol=sto_vol, rd_mu=0, rd_sigma=0)
        dw2 = np.zeros(self.T)
        dw2[0] = corr_wn2[0]

        for t in range(1, self.T):
            dw2[t] = self.__heston_dis_vol(
                k=k,
                theta=theta,
                vt=corr_wn2[t],
                sigma=sigma,
                w2=wn2[t],
            )

        # creating the actual data for the process
        heston_ret = np.zeros(self.T)
        heston_ret[0] = self.s0

        for t in range(1, self.T):
            heston_ret[t] = heston_ret[t - 1] + self.__heston_discrete(
                rf=rf,
                st=heston_ret[t - 1],
                V=dw2[t],
                w1=corr_wn1[t],
            )

        return heston_ret.ravel()

    # Cox Ingersoll Ross

    def __cir_discrete(self, mu: float, sigma: float, lambda_: float, xt: float, vol: float):
        return lambda_ * (mu - xt) * self.h + sigma * np.sqrt(np.maximum(xt, 0) * self.h) * vol

    def __cir_return(self, mu: float, sigma: float, lambda_: float, sto_vol: bool):

        volatility = self.__random_disturbance(sto_vol, mu, sigma)
        cir_ret = np.zeros(self.T)

        cir_ret[0] = self.r0

        for t in range(1, self.T):
            cir_ret[t] = cir_ret[t - 1] + self.__cir_discrete(mu, sigma, lambda_, cir_ret[t - 1], volatility[t])

        return cir_ret

    # Vasicek Interest Rate Model and Ornstein–Uhlenbeck Process - Mean reverting

    def __vas_discrete(self, mu: float, sigma: float, lambda_: float, rt: float, vol: float):

        return lambda_ * (mu - rt) * self.h + sigma * np.sqrt(self.h) * vol

    def __vas_returns(self, mu: float, sigma: float, lambda_: float, sto_vol: float):
        volatility = self.__random_disturbance(sto_vol, rd_mu=mu, rd_sigma=sigma)
        vas_rets = np.zeros((self.T))
        vas_rets[0] = self.r0

        for t in range(1, self.T):
            vas_rets[t] = vas_rets[t - 1] + self.__vas_discrete(
                mu=mu,
                sigma=sigma,
                lambda_=lambda_,
                rt=vas_rets[t - 1],
                vol=volatility[t],
            )

        return vas_rets

    # Merton Jump Diffusion Stochastic Process

    def __jumps_diffusion(self, lambda_: int):
        t = 0
        jumps = np.zeros((self.T, 1))
        lambda__ = lambda_ / self.T
        small_lambda = -(1.0 / lambda__)
        pd = np.random.poisson(lam=lambda__, size=(self.T))

        # applying the psudo-code of the algorithm
        for i in range(self.T):
            t += small_lambda * np.log(np.random.uniform())
            if t > self.T:
                jumps[i:] = ((np.mean(pd) + np.std(pd)) * np.random.uniform()) * np.random.choice([-1, 1])
                # the t parameter is restituted to the original value
                # for several jumps in the future
                t = small_lambda
                break

        return jumps.reshape(-1, 1)

    def __merton_returns(self, mu: float, sigma: float, lambda_: int, sto_vol: bool):
        geometric_brownian_motion = self.__brownian_returns(mu, sigma, sto_vol).reshape(-1, 1)
        jump_diffusion = self.__jumps_diffusion(lambda_)
        return (geometric_brownian_motion + jump_diffusion).ravel()

    # Brownian Motion Stochastic Process (Wiener Process)

    def __brownian_discrete(self, mu: float, sigma: float, st: float, vol: float):
        return (mu * st * self.h) + (sigma * st * np.sqrt(self.h) * vol)

    def __brownian_returns(self, mu: float, sigma: float, sto_vol: bool):
        # preallocate the volatility
        volatility = self.__random_disturbance(sto_vol, rd_mu=mu, rd_sigma=sigma)
        bro_returns = np.zeros((self.T))
        # initial condition X(0) = X_0
        bro_returns[0] = self.s0
        for t in range(1, self.T):
            bro_returns[t] = bro_returns[t - 1] + self.__brownian_discrete(
                mu, sigma, st=bro_returns[t - 1], vol=volatility[t]
            )

        return bro_returns

    # -------------------------------------------
    # General utilities
    # -------------------------------------------

    def __random_disturbance(self, sto_vol: bool, rd_mu: float, rd_sigma: float):

        if not sto_vol:
            return np.random.normal(size=self.T)

        scale = abs(rd_sigma) * halfnorm.rvs(loc=0.0, scale=1.0)
        if scale == 0.0:
            return np.full(self.T, rd_mu * self.h)

        return np.random.normal(
            loc=rd_mu * self.h,
            scale=scale * np.sqrt(self.h),
            size=self.T,
        ) * 100.0

    def __corr_noise(self):

        # generate two uncorrelated Brownian processes
        z1 = self.__random_disturbance(sto_vol=False, rd_mu=0, rd_sigma=1)
        z2 = self.__random_disturbance(sto_vol=False, rd_mu=0, rd_sigma=1)

        # randomly create an absolute correlation
        rho = np.random.uniform(low=0.5, high=1)

        corr1 = np.sqrt((1 + rho) / 2)
        corr2 = np.sqrt((1 - rho) / 2)

        # correlating the brownian processes
        dw1 = corr1 * z1 + corr2 * z2
        dw2 = corr1 * z1 - corr2 * z2

        return dw1, dw2

    def __generate_paths(self, generator):
        if self.n == 1:
            return generator()

        return np.column_stack([generator() for _ in range(self.n)])
