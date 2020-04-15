import numpy as np

class SimPaths(object):

	def __init__(self, n:int, T:int, s0:float=1, rd_mu=0, rd_sigma=1):

		self.n = n
		self.T = T
		self.s0 = s0
		self.r0 = self.s0 / 100 # initial interest rate for cir
		self.dt = 1 / self.T
		# inputs for the random disturbance for each time step of the processes
		self.rd_mu = rd_mu
		self.rd_sigma = rd_sigma

	# -------------------------------
	# Helper methods
	# -------------------------------

	def _random_disturbance(self, stochastic_volatility:bool):
		"""
		Sample a random disturbance from either a normal distribution with a 
	    constant standard deviation (Geometric Brownian Motion model) or from a 
	    distribution with a stochastic standard deviation (Stochastic Volatility GBM)
	    
	    Given a long run random disturbance, sample a random disturbance from either 
	    a standard normal distribution with mean mu and standard deviation sigma, or 
	    sample a random disturbance from a normal distribution with mean mu and a 
	    stochastic standard deviation sampled from a normal distribution centered around 
	    sigma with a standard deviation equal to half of sigma. 
	    This is maxed out at 0.01 because a negative standard deviation is incoherent. 
	    
	    When the stochastic_volatility parameter is enabled, the resulting 
	    distribution of means for the random disturbance has much fatter tails. 
		
		Args:
			mu(float): mean of a normal distribuition
			sigma(float): standard deviation of a normal distribuition
			stochastic_volatility(boolean): choose if the disturbance is 
				Homoskedastic or Heteroskedastic variance. 
		Returns
			a float number, that represents the disturbance in a series of prices
		"""
		if not stochastic_volatility:
			return np.random.normal(self.rd_mu, self.rd_sigma, 1)

		sigma_stochastic = max(np.random.normal(self.rd_sigma, self.rd_sigma/2, 1), 0.01)

		return np.random.normal(self.rd_mu, sigma_stochastic, 1)

	def ohlc_candles(self, prices:np.array):
		"""
		Generate OHLC candles from a series of simulated prices
		source https://towardsdatascience.com/advanced-candlesticks-for-machine-learning-ii-volume-and-dollar-bars-6cda27e3201d
		"""
		ROWS = len(prices)
		ans = np.zeros(shape=(ROWS, 4)) # 4 => OHLC
		candle_counter = 0
		lasti = 0
		freq = 0

		for i in range(ROWS):
			freq+=1
			if freq == 4:
				ans[candle_counter][0] = prices[lasti] # open
				ans[candle_counter][1] = np.max(prices[lasti:i+1]) # high
				ans[candle_counter][2] = np.min(prices[lasti:i+1]) # low
				ans[candle_counter][3] = prices[i] # close
				candle_counter+=1
				lasti+=1
				freq = 0

		return ans[:candle_counter]

	def _corr_noise(self, T:int, rho:float):
		"""
		A straightforward way to create noise processes with a specified correlation 
		is through the singular value decomposition (SVD) 
		see Sauer (2006) for a description).
		"""
		# Two correlated brownian motions
		# generate two uncorrelated Brownian processes
		z1 = np.array( [ self._random_disturbance(stochastic_volatility=False) for _ in range(T) ] )
		z2 = np.array( [ self._random_disturbance(stochastic_volatility=False) for _ in range(T) ] )

		_corr1 = np.power( ( 1 + rho ) / 2, 0.5)
		_corr2 = np.power( ( 1 - rho ) / 2, 0.5)

		# correlating the brownian processes
		dw1 = (_corr1 * z1 + _corr2 * z2).reshape(-1, 1)
		dw2 = (_corr1 * z1 - _corr2 * z2).reshape(-1, 1)

		return dw1, dw1

	# -------------------------------
	# The Brownian Motion Stochastic Process (Wiener Process)
	# -------------------------------

	def _brownian_continuous(self, mu:float, xt:float, rd:float):
		"""
		Brownian motion X with the drift parameter µ, and the variance
		parameter σ^2, and the initial time t = 0 satisfies the following 
		stochastic differential equation, where W is a standard 
		Brownian motion
		dX(t) = µdt + σdW(t) , X(0) = 0
		where µ=0, σ=1 and dW(t)∼N(0, dt)
		"""
		return mu * self.dt + xt + rd * np.sqrt(self.dt)

	def _brownian_returns(self, x0:float, T:float, mu:float, stochastic_volatility:bool):
		"""
		"""
		# preallocate the data
		X = np.zeros(T)
		X[0] = x0

		for t in range(1, T):
			_rd = self._random_disturbance(stochastic_volatility=stochastic_volatility)
			X[t] = self._brownian_continuous(mu=mu, xt=X[t-1], rd=_rd)

		return X

	def brownian_prices(self, mu:float=0, stochastic_volatility:bool=False):
		"""
		"""
		_data = np.empty((self.T, self.n))

		if self.n > 1:
			for i in range(self.n):
				_data[:, i] = np.exp(self._brownian_returns(x0=np.log(self.s0), T=self.T, mu=mu, stochastic_volatility=stochastic_volatility))
		else:
			_data = np.exp(self._brownian_returns(x0=np.log(self.s0), T=self.T, mu=mu, stochastic_volatility=stochastic_volatility))

		return _data

	# -------------------------------
	# The Geometric Brownian Motion Stochastic Process
	# -------------------------------

	def gbm_prices(self, mu:float=1, stochastic_volatility:bool=True):
		"""
		GBM says the change in stock price is the stock 
		price "S" multiplied by the two terms found inside
		the parenthesis below
		ΔS = S × (μΔt + σϵΔt)
		
		The first term is a "drift" and the second term 
		is a "shock." For each time period, our model 
		assumes the price will "drift" up by the expected 
		return. But the drift will be shocked (added or 
		subtracted) by a random shock. The random shock 
		will be the standard deviation "s" multiplied by
		a random number "e." This is simply a way of 
		scaling the standard deviation.
		more details at 
		https://en.wikipedia.org/wiki/Geometric_Brownian_motion
		"""
		_data = np.empty((self.T, self.n))

		if self.n > 1:
			for i in range(self.n):
				_data[:, i] = np.exp(self._brownian_returns(x0=np.log(self.s0), T=self.T, mu=mu, stochastic_volatility=stochastic_volatility))
		else:
			_data = np.exp(self._brownian_returns(x0=np.log(self.s0), T=self.T, mu=mu, stochastic_volatility=stochastic_volatility))

		return _data

	# -------------------------------
	# The Merton Jump Diffusion Stochastic Process
	# -------------------------------

	def _jumps_diffusion(self, lam:int):
		"""
		Suppose that we wish to simulate a stationary 
		compound Poisson process at rate λ with iid 
		Bi distributed as (say) G (could be continuous 
		or discrete). Suppose that we already have an 
		algorithm for generating from G
		Algorithm for generating a compound Poisson process 
		up to a desired time T to get X(T)
		1. t = 0, N = 0, X = 0.
		2. Generate a U
		3. t = t + [−(1/λ) ln (U)]. If t > T, then stop.
		4. Generate B distributed as G.
		5. Set N = N + 1 and set X = X + B
		6. Go back to 2.
		source http://www.columbia.edu/~ks20/4703-Sigman/4703-07-Notes-PP-NSPP.pdf
		page 8
		https://www.statstodo.com/Poisson_Exp.php
		https://www.lpsm.paris/pageperso/tankov/tankov_voltchkova.pdf
		pag 5
		"""
		t = 0
		x_jumps = np.zeros(self.T)
		_lambda = lam / self.T
		small_lambda = -(1.0/_lambda)

		for n in range(self.T):
			t += small_lambda * np.log(np.random.uniform(0,1))
			if t > self.T:
				x_jumps[n:] = ( np.random.poisson(_lambda, 1) + np.random.uniform(0, 1) ) * np.random.normal(0, 0.5)
				# this is a test for several jumps
				# the t parameter is restituted to the original value
				t = small_lambda

		return x_jumps

	def _merton_returns(self, x0:float, T:float, jumps:int, mu:float, stochastic_volatility:bool):
		"""
		"""
		geometric_brownian_motion = self._brownian_returns(x0=x0, T=T, mu=mu, stochastic_volatility=stochastic_volatility)

		jump_diffusion = self._jumps_diffusion(lam=jumps)

		return np.add(geometric_brownian_motion, jump_diffusion)

	def merton_prices(self, jumps:int=5, mu:float=1, stochastic_volatility:bool=False):
		"""
		"""
		_data = np.empty((self.T, self.n))

		if self.n > 1:
			for i in range(self.n):
				_data[:, i] = np.exp(self._merton_returns(x0=np.log(self.s0), T=self.T, jumps=jumps, mu=mu, stochastic_volatility=stochastic_volatility))
		else:
			_data = np.exp(self._merton_returns(x0=np.log(self.s0), T=self.T, jumps=jumps, mu=mu, stochastic_volatility=stochastic_volatility))

		return _data

	# -------------------------------
	# Heston Stochastic Volatility Process (Heston Model)
	# -------------------------------

	################## special volatility for Heston model ##################

	def _heston_vol_discrete(self, kappa:float, lamba:float, rt:float, rd:float, t:int):
		# consider the process Yt = √Xt
		dt = t / self.T
		sigma = self.rd_sigma / 100

		return kappa * ( lamba - rt ) * dt + sigma * np.power(rt*dt, 0.5) * rd

	def _heston_vol_returns(self, x0:float, T:int, kappa:float, lamba:float, rd:float, t:int):
		"""
		"""
		X = np.zeros(T)
		X[0] = x0

		for t in range(1, T):
			X[t] = X[t-1] + self._heston_vol_discrete(kappa=kappa, lamba=lamba, rt=X[t-1], rd=rd[t-1], t=t)

		return X

	def _heston_vol_rates(self, kappa:float, lamba:float, rd:np.array, t:int=1):
		"""
		"""
		_data = np.zeros((self.T, self.n))

		if self.n > 1:
			for i in range(self.n):
				_data[:, i] = self._heston_vol_returns(x0=self.r0, T=self.T, kappa=kappa, lamba=lamba, rd=rd, t=t)

		else:
			_data = self._heston_vol_returns(x0=self.r0, T=self.T, kappa=kappa, lamba=lamba, rd=rd, t=t)

		return _data

	################## Actual Stochastic Differential model ##################

	def _heston_discrete(self, mu:float, rd:float, st:float, wt:float):
		"""
		"""
		# print(f'mu={mu} | rd={rd} | st{st} | wt={wt}')
		return ( mu * st * self.dt ) + ( np.power( rd * self.dt, 0.5) * st * wt ) 

	def _heston_returns(self, x0:float, T:int, mu:float, kappa:float, lamba:float, rho:float):
		"""
		"""
		# integrate a random correlation level
		dw1, dw2 = self._corr_noise(T=T, rho=rho)

		# integrating into the mean reverting volatility
		cir_vol = self._heston_vol_rates(kappa=kappa, lamba=lamba, rd=dw1).reshape(-1, 1)

		X = np.zeros(T)
		X[0] = x0

		# building the stochastic process for the discretized increments
		for t in range(1, T):
			X[t] = X[t-1] + self._heston_discrete(mu=mu, rd=cir_vol[t-1], st=X[t-1], wt=dw2[t-1])

		return X

	def heston_prices(self, mu:float, kappa:float, lamba:float, rho:float):
		"""
		"""
		_data = np.zeros((self.T, self.n))

		if self.n > 1:
			for i in range(self.n):
				_data[:, i] = self._heston_returns(x0=self.s0, T=self.T, mu=mu, kappa=kappa, lamba=lamba, rho=rho)

		else:
			_data = self._heston_returns(x0=self.s0, T=self.T, mu=mu, kappa=kappa, lamba=lamba, rho=rho)

		return _data

	# -------------------------------
	# Cox Ingersoll Ross (CIR) stochastic proces
	# -------------------------------

	def _cir_discrete(self, kappa:float, lamba:float, rt:float, rd:float, t:int):
		"""
		The Cox-Ingersoll-Ross model (CIR) is a mathematical 
		formula used to model interest rate movements and is 
		driven by a sole source of market risk. It is used as 
		a method to forecast interest rates and is based on a
		stochastic differential equation.
		KEY TAKEAWAYS
		The CIR is used to forecast interest rates.
		The CIR is a one-factor equilibrium model that uses a
		square-root diffusion process to ensure that the 
		calculated interest rates are always non-negative.
		source: https://en.wikipedia.org/wiki/Cox%E2%80%93Ingersoll%E2%80%93Ross_model
		https://www.jstor.org/stable/3532742?read-now=1&refreqid=excelsior%3A777958c5a8c04e6c69b3dfa464a1844a&seq=1#page_scan_tab_contents
		"""
		# consider the process Yt = √Xt
		dt = t / self.T
		sigma = self.rd_sigma / 100

		return kappa * ( lamba - rt ) * dt + sigma * np.power(rt*dt, 0.5) * rd

	def _cir_returns(self, x0:float, T:int, kappa:float, lamba:float, t:int, stochastic_volatility:bool):
		"""
		"""
		X = np.zeros(T)
		X[0] = x0

		for t in range(1, T): 
			_rd = self._random_disturbance(stochastic_volatility=stochastic_volatility)
			X[t] = X[t-1] + self._cir_discrete(kappa=kappa, lamba=lamba, rt=X[t-1], rd=_rd, t=t)

		return X

	def cir_rates(self, kappa:float, lamba:float, t:int=10, stochastic_volatility:bool=False):
		"""
		"""
		_data = np.zeros((self.T, self.n))

		if self.n > 1:
			for i in range(self.n):
				_data[:, i] = self._cir_returns(x0=self.r0, T=self.T, kappa=kappa, lamba=lamba, 
					t=t, stochastic_volatility=stochastic_volatility)

		else:
			_data = self._cir_returns(x0=self.r0, T=self.T, kappa=kappa, lamba=lamba, 
				t=t, stochastic_volatility=stochastic_volatility)

		return _data

	# -------------------------------
	# The Ornstein-Uhlenbech Mean Reverting Model
	# -------------------------------

	def _ou_discrete(self, st:float, rho:float, mu:float, rd:float):
		"""
		"""
		exp_lambda = np.exp(-rho * self.dt)
		sigma_lambda = ( ( 1 - np.exp(-2 * rho * self.dt) ) / ( 2 * rho ) ) ** 0.5

		return exp_lambda * st + ( 1 - exp_lambda ) * mu + sigma_lambda * rd

	def _ou_prices(self, rho:float, mu:float, stochastic_volatility:bool=False):
		"""
		"""
		# preallocate the data
		S = np.zeros(self.T)
		S[0] = self.s0
		for t in range(1, self.T):
			_rd = self._random_disturbance(stochastic_volatility=stochastic_volatility)
			_temp = self._ou_discrete(st=S[t-1], rho=rho, mu=mu, rd=_rd)
			#case handling for negative prices - needs external assessment!
			if _temp < 0: 
				S[t] = 0 #np.maximum(0, np.abs(_rd + np.abs(S[t-1])))
			S[t] = _temp

		return S

	def ou_prices(self, rho:float=0.2, mu:float=0.5, stochastic_volatility:bool=False):
		"""
		"""
		_data = np.zeros((self.T, self.n))

		if self.n > 1: 
			for i in range(self.n):
				_data[:, i] = self._ou_prices(rho=rho, mu=mu, stochastic_volatility=stochastic_volatility)

		else:
			_data = self._ou_prices(rho=rho, mu=mu, stochastic_volatility=stochastic_volatility)

		return _data

	# -------------------------------
	# Hull-White Model
	# -------------------------------

	# -------------------------------
	# Vasicek Interest Rate Model
	# -------------------------------

	def _vas_discrete(self, kappa:float, lamba:float, rt:float, rd:float, t:int):
		"""
		"""
		dt = t / self.T
		sigma = self.rd_sigma / 100

		return kappa * ( lamba - rt ) * dt +  sigma * np.power(dt, 0.5) * rd

	def _vas_returns (self, x0:float, T:int, kappa:float, lamba:float, t:int, stochastic_volatility:bool):
		"""
		"""
		X = np.zeros(T)
		X[0] = x0

		for t in range(1, T): 
			_rd = self._random_disturbance(stochastic_volatility=stochastic_volatility)
			X[t] = X[t-1] + self._vas_discrete(kappa=kappa, lamba=lamba, rt=X[t-1], rd=_rd, t=t)

		return X

	def vas_rates(self, kappa:float, lamba:float, t:int=10, stochastic_volatility:bool=False):
		"""
		"""
		_data = np.zeros((self.T, self.n))

		if self.n > 1:
			for i in range(self.n):
				_data[:, i] = self._vas_returns(x0=self.r0, T=self.T, kappa=kappa, lamba=lamba, 
					t=t, stochastic_volatility=stochastic_volatility)

		else:
			_data = self._vas_returns(x0=self.r0, T=self.T, kappa=kappa, lamba=lamba, 
						t=t, stochastic_volatility=stochastic_volatility)

		return _data

	# -------------------------------
	# Heath-Jarrow-Morton Model
	# -------------------------------