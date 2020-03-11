import numpy as np
import scipy.stats as st
import matplotlib.pyplot as plt
from matplotlib import style
import seaborn as sns
sns.set_style('whitegrid')

class EMH(object):

	def __init__(self):
		self.data = None

	# -----------------------------------
	# Helper functions
	# -----------------------------------	

	def _densities(self, ref_stats, z_stats):
		"""
		plot the densities for a series of data

		These ref_stats (normal distribuition) values are 
		colored red, because represents the light that 'stop' 
		the rejection of the null hypothesis. Analog representation
		is made for the z scores from the test.
		"""
		# plot the normal distribuition values
		sns.kdeplot(ref_stats, kernel='gau', shade=True, color="r")
		# plot the values for the statistic
		sns.kdeplot(z_stats, kernel='gau', shade=True, color="g")
		title_text = "Comparison of densities of a Normal Distribuited variable(red)\nagainst the Z* scores computed for the analyzed series of log prices"
		plt.title(title_text)
		plt.show()

	def _stat_plot(self, q_range:list =[2, 4], total_samples:int =500, stat_type:str ='mr'):
	    """
	    Generate the plots for the DIFFERENCES USING THE OVERLAPPING SAMPLES ESTIMATOR

	    """
	    # error handling 
	    if len(q_range) != 2:
	        print('Please select at most 2 ranges to analyze, e.g., [3,6]')
	        return None
	    
	    # setting the desired statistic
	    if  stat_type == 'md':
	        stat = self._md
	        statName = stat_type.capitalize() # capitalize the name
	    elif stat_type == 'mr':
	        stat = self._mr
	        statName = stat_type.capitalize() # capitalize the name
	    else:
	        print('not valid statistic, pelase try md or mr')
	        return None
	    
	    # generating the statistics for values without Stochastic Volatility
	    _statsQ1 = [stat(self.price_processes(1, T=sample*2), q_range[0]) for sample in range(q_range[0], total_samples)]
	    _statsQ2 = [stat(self.price_processes(1, T=sample*2), q_range[1]) for sample in range(q_range[1], total_samples)]
	    
	    # generating the statistics for values with Stochastic Volatility
	    _statsVolQ1 = [stat(self.price_processes(1, T=sample*2),q_range[0],
	                       unbiased=False) for sample in range(q_range[0], total_samples)]
	    _statsVolQ2 = [stat(self.price_processes(1, T=sample*2), q_range[1],
	                       unbiased=False) for sample in range(q_range[1], total_samples)]
	    
	    # Creating the plots        
	    fig, axes = plt.subplots(2, 2, figsize=(15,15))
	    # values of MD without Stochastic volatility
	    axes[0,0].plot(_statsQ1, marker='o', linestyle='None');
	    axes[0,0].set_title(f'Values of {statName} without Stochastic volatility')
	    axes[0,0].set_ylabel(f'Values of Md(q={q_range[0]})')
	    
	    axes[0,1].plot(_statsQ2, marker='o', linestyle='None');
	    axes[0,1].set_title(f'Values of {statName} without Stochastic Volatility')
	    axes[0,1].set_ylabel(f'Values of Md(q={q_range[1]})')
	    
	    # values of MD with Stochastic volatility
	    axes[1,0].plot(_statsVolQ1, marker='o', linestyle='None');
	    axes[1,0].set_title(f'Values of {statName} with Stochastic volatility')
	    axes[1,0].set_ylabel(f'Values of Md(q={q_range[0]})')
	    
	    axes[1,1].plot(_statsVolQ2, marker='o', linestyle='None')
	    axes[1,1].set_title(f'Values of {statName} with Stochastic volatility')
	    axes[1,1].set_ylabel(f'Values of Md(q={q_range[1]})')

	    plt.show()
	    
	    return 


	# -----------------------------------
	# Simulation area
	# -----------------------------------

	def random_disturbance(self, mu:float =0.0, sigma:float =0.45, size:int =1, stochastic_volatility:bool =True):
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
			size(int): size of the sample
			stochastic_volatility(boolean): choose if the disturbance is 
				Homoskedastic or Heteroskedastic variance. 

		Returns
			a float number, that represents the disturbance in a series of prices

		"""
		if not stochastic_volatility:
			return np.random.normal(mu, sigma, size)

		sigma_stochastic = max(np.random.normal(sigma, sigma/2, size), 0.01)

		return np.random.normal(mu, sigma_stochastic, size)

	def price_step(self, xt:float, mu:float, rd:float, dt:float =1/252):
		"""
		Step forward one unit of time in the log price process.
	    This step function is used to move from state t / t-1 in the log 
	    price process to t+1 / t. This involves adding a static drift component to the 
	    previous price and adding a random disturbance sampled from some distribution.
	    
	    Args:
	        xt(double): The current price in the log price process.
	        mu(double): The drift component of returns.
	        rd(double): A random disturbance in the force.
	        dt(double): This variable controls time. It is set to 1/252 by default
	    
	    Returns:
	        Xt(float): The next price in the log price process.
		"""
		return ( mu * dt ) + xt + ( rd * np.sqrt(dt) )

	def log_prices(self, mu:float =0.1062, rd_mu:float =0.0, rd_sigma:float =0.45, T:int =252, x0:float =1.0,  
					dt:float =1/252, stochastic_volatility:bool =True):
		"""
		This method is used to generate a log price process, which simulates the log 
		price of an asset, starting with the value of x0. The process is controlled by a 
		number of parameters including. The method works by iteratively sampling a 
		new log price using the price_step method/function.
    
    	Args:
    		mu(double): The drift component of returns.
    		rd_mu(double): The average disturbance to be added.
    		rd_sigma(double): The volatility of disturbances to be added.
        	T(int): The number of observations to simulate.
        	x0(double): The starting price of the asset. Set to 1.0.
        	dt(double): Time between each observation. set to 1/252 by default 
        				(simulates daily returns).
        	stochastic_volatility(boolean): Homoskedastic or Heteroskedastic variance.
        
    	Returns:
        	X(numpy array): A simulated log price process.
		"""
		# preallocate the data
		X = np.repeat(x0, T)
		for t in range(1, T):
			rd = self.random_disturbance(mu=rd_mu, sigma=rd_sigma, stochastic_volatility=stochastic_volatility)
			X[t] = self.price_step(xt=X[t-1], mu=mu, rd=rd, dt=dt)

		return X

	def _process(self, mu:float =0.1062, rd_mu:float =0.0, rd_sigma:float =0.45, T:int =252, x0:float =1.0,  
					dt:float =1/252, stochastic_volatility:bool =True):
		"""
		convert a log price process into a discrete price process.
		"""
		return np.exp(self.log_prices(mu=mu, rd_mu=rd_mu, rd_sigma=rd_sigma, T=T, 
			x0=np.log(x0), dt=dt, stochastic_volatility=stochastic_volatility))

	def price_processes(self, n:int, mu:float =0.1062, rd_mu:float =0.0, rd_sigma:float =0.45, T:int =252, x0:float =1.0,  
					dt:float =1/252, stochastic_volatility:bool =True):
	    """
	    This method constructs a numpy matrix, which contains multiple discrete 
	    price processes simulated using the _process helper method/function.
	    
	    Args:
	        n(int): The number of price processes to simulate.
	        
	    Returns:
	        a matrix with several simulated discrete price processes.
	    """
	    _data = np.empty((T, n))
	    if n > 1:
	        for i in range(n):
	            price_data_ = self._process(mu=mu, rd_mu=rd_mu, rd_sigma=rd_sigma, T=T, x0=x0, 
	            	dt=dt, stochastic_volatility=stochastic_volatility)
	            _data[:, i] = price_data_
	    else:
	        _data = np.array(self._process(mu=mu, rd_mu=rd_mu, rd_sigma=rd_sigma, T=T, x0=x0, 
	            	dt=dt, stochastic_volatility=stochastic_volatility))
	            
	    return _data

    # -----------------------------------
	# Homoskedastic Null Hypothesis
	# -----------------------------------

	def _mu(self, X, annualize:bool =True):
		"""
	    Given a log price process, this function estimates the value of mu_hat. Mu is the daily 
	    component of the returns which is attributable to upward, or downward drift. 
	    This estimator correspond to the maximum-likelihood estimator of mu_hat
	    This estimate can be annualized.
	    
	    Args:
	        X(array): A log price process.
	        annualize(boolean): Annualize the parameter estimate.
	        
	    Returns:
	        muEst(float): The estimated value of mu.
		"""
		n = X.shape[0]
		if annualize:
			return ( 1 / n ) * ( X[-1] - X[0] ) * 252 # python indexation
		else:
			return ( 1 / n ) * ( X[-1] - X[0] )		# python indexation, starts at 0

	def _vol_a(self, X, q:int =1, unbiased:bool =True, annualize:bool =True):
		"""
		This function calculates 'the sample variance of the first-difference of
		X; it corresponds to the maximum likelihood-estimator of the variance 
		parameter and therefore possesses the usal consistency, asymptotic normality
		and efficiency properties.' 
							Andrew Lo, The Size and Power of the Variance Ratio Test

		Args:
			X(numpy array): A log pricess process
			q(int): aggregation value, used to calculate the qth differences. 
			unbiased(boolean): the data represent a sample or the population?
			annualize(boolan): annualize the results

		Returns
			sigma_est(float): the sample variance of the first-difference of X

		"""
		# preallocating the data
		mu_est = self._mu(X=X, annualize=False) # MLE for the drift component
		sigma_est = 0.0
		# total number of observations, this is consistent with the paper notation
		n = int(np.floor( X.shape[0] / q ))

		for k in range(1, n * q):
			sigma_est += ( X[k] - X[k - 1] - mu_est ) ** 2
			# sigma_est += ( X[k * q] - X[k * q - q] - ( q * mu_est ) ) ** 2

		# unbiasing the estimator
		if unbiased:
			sigma_est =  sigma_est / ( n * q - 1 )
		else:
			sigma_est = sigma_est / ( n * q )

		# annualizing the results
		if annualize:
			return np.sqrt( sigma_est * 252 )
		else:
			return sigma_est

	def _vol_b(self, X, q:int =1, unbiased:bool =True, annualize:bool =True):
		"""
		This function calculates the variance of qth diferences of Xt, 
		which, under H1 (Homoskedastic variance increments), is q times 
		the variance of first-differences. By dividing by q, we obtain 
		the estimator sigma^2_b which also converges to sigma^2 under H1. 

		Args:
			X(numpy array): series of prices
			q(int): aggregation value, used to calculate the qth differences.
			annualize(bool): annualization of the results.

		Results:
			sigma^2_b(float): variance of the qth difference.
		"""
		# pre-allocating the data
		mu_est = self._mu(X=X, annualize=False)
		n = int(np.floor( X.shape[0] / q ))
		sigma_est = 0.0

		# differences of every qth observation
		for k in range(q, n * q): # q-1 BECAUSE python indexation
			sigma_est += ( X[k] - X[k - q] - (q * mu_est) ) ** 2

		# unbiasing the estimator
		if unbiased:
			m = q * (n * q - q + 1) * (1 - (q / ( n * q )))
			sigma_est = sigma_est / m
		else:
			sigma_est = sigma_est / ( n * ( q ** 2 ) )

		# annualizing the results
		if annualize:
			return np.sqrt( sigma_est * 252 )
		else:
			return sigma_est

	def _md(self, X, q:int, unbiased:bool =True, annualize:bool =True):
		"""
		differences in volatilities for both estimators. Under the null hypothesis
		of a Gaussian random walk, the tow estimators sigma^2_a and sigma^2_b should 
		be close; therefore a test of the random walk may be constructed by computing 
		the difference of both volatilities. 

		Args:
			X(numpy array): array with the series of prices.
			q(int): aggregation value, used to calculate the qth differences.
			unsiased(boolean): is the user using a sample or the universe of 
								observations? 
			annualize(boolean): the results should be annualized? 

		Results:
			Md(float): statistic for the volatilities differences.
		"""
		return self._vol_b(X=X, q=q, unbiased=unbiased, annualize=annualize) - self._vol_a(X=X, unbiased=unbiased, annualize=annualize)

	def _mr(self, X, q:int, unbiased:bool =True, annualize:bool =True):
		"""
		A test may also be based upon the dimensionless centered variance ratio
		Mr = sigma^2_b / sigma^2_a - 1. Which converges in probability to zero as well.
		So this method calculates this dimensionless ratio.

		Args:
			X(numpy array): array with the series of prices.
			q(int): aggregation value, used to calculate the qth differences.
			unsiased(boolean): is the user using a sample or the universe of 
								observations? 
			annualize(boolean): the results should be annualized? 

		Results:
			Mr(float): statistic for the dimentionless volatilities.

		"""
		return ( self._vol_b(X=X, q=q, unbiased=unbiased, annualize=annualize) / self._vol_a(X=X, unbiased=unbiased, annualize=annualize) ) - 1

	def h1(self, X, q:int, centered:bool =True, unbiased:bool =True, annualize:bool =True):
		"""
		This function calculates the IID Gaussian Null Hypothesis. 
		The essence of the random walk hypothesis is the restriction that the disturbances
		e_t are serially correlated or that innovations are unforecastable from past innovations. 
		Hence:
			H1: e_t IID N(0, sigma^2) / Homoskedasticity of variance increments

		Args: 
			X(numpy array): series of prices to test
			q(int): aggregation value, used to calculate the qth differences.
			centered(bool): want a dimentionless statistic? (this is preferred by the author)
			unbiased(bool): the data is a sample or the universe? 
			annualize(bool): annualization of the results. 

		Returns:
			Variance test stadistic for Homokedastic variance increments. 

		Notes:
		https://machinelearningmastery.com/critical-values-for-statistical-hypothesis-testing/
		https://stackoverflow.com/a/41338933/6509794
		"""
		n = np.floor( X.shape[0] / q )

		if centered:
			z1 =  np.sqrt( np.multiply(n, q) ) * self._mr(X=X, q=q, unbiased=unbiased, annualize=annualize)
			asymp = ( 2 * ( 2 * q - 1 ) * ( q - 1 ) ) / ( 3 * q )
			return z1 * np.power(asymp, -0.5)
		else:
			z1 = np.sqrt( np.multiply(n, q) ) * self._md(X=X, q=q, unbiased=unbiased, annualize=annualize)
			return z1

    # -----------------------------------
	# Heteroskedastic Null Hypothesis
	# -----------------------------------

	def _delta(self, X, q:int, j:int):
		"""
		 Helper method for the variance estimator v_hat

		"""
		# Get the estimate value for the drift component.
		mu_est = self._mu(X, False)
		
		# preallocating the data
		n = int(np.floor( X.shape[0] / q ))

		# Estimate the asymptotice variance given q and j.
		# Because the numerator and denominator are separate calculations
		# two separate loops will going to take place
		# this is much more clear than one big loop

		# NUMERATOR case
		numerator = 0.0
		for k in range(j + 1, n * q):
			numerator += ( ( X[k] - X[k - 1] - mu_est ) ** 2 ) * ( ( X[k - j] - X[k - j - 1] - mu_est) ** 2 ) 

		# DENOMINATOR case
		denominator = 0.0
		for k in range(1, n * q):
			denominator += ( X[k] - X[k - 1] - mu_est ) ** 2

		# Compute and return the statistic
		return ( n * q * numerator ) / ( denominator ** 2 )


	def _v_hat(self, X, q):
		"""
		Estimator for the value of the asymptotic variance of the _mr statistic. 
		This is equivalent to a weighted sum of the asymptotic variances for each of 
		the autocorrelation co-efficients under the null hypothesis.

		Given a log price process, X, and a sampling interval, q, this 
		method is used to estimate the asymptoticvariance of the _mr statistic in the 
		presence of stochastic volatility. 
		In other words, it is a heteroskedasticity consistent estimator of 
		the variance of the _mr statistic. This parameter is used to estimate the 
		probability that the given log price process was generated by a 
		Brownian Motion model with drift and stochastic volatility. 

		Args:
			X(numpy array): serie of log prices
			q(int): aggregation value, used to calculate the qth differences.

		Returns
			volatility(float): Estimator for the value of the asymptotic 
							  variance of the _mr statistic. 
		"""
		v_hat = 0.0
		for j in range(1, q - 1):
			delta = self._delta(X=X, q=q, j=j)
			v_hat += ((( 2 * ( q - j ) ) / q ) ** 2 ) * delta

		return v_hat

	def h2(self, X, q:int, centered:bool =True, unbiased:bool =True, annualize:bool =True):
		"""
		The Heteroskedastic Null Hypothesis
		"""
		n = np.floor( X.shape[0] / q )
		if centered:
			z2 = np.sqrt( np.multiply(n, q) ) * self._mr(X=X, q=q, unbiased=unbiased, annualize=annualize)
			return z2 * np.power(self._v_hat(X=X, q=q), -0.5)

		else:
			return None

    # -----------------------------------
	# The Variance Test Ratio Test
	# -----------------------------------

	def vrt(self, X, q:int, heteroskedastic:bool =True, centered:bool =True, unbiased:bool =True, annualize:bool =True):
		"""
		Calculate the z statistic and the p-value for the Varaciance Ratio test
		https://stackoverflow.com/a/20871775/6509794
		"""
		if heteroskedastic:
			z_score = self.h2(X=X, q=q, centered=centered, unbiased=unbiased, annualize=annualize)
			p_value = 1 - st.norm.cdf(z_score)
			return z_score, p_value
		else:
			z_score = self.h1(X=X, q=q, centered=centered, unbiased=unbiased, annualize=annualize)
			p_value = 1 - st.norm.cdf(z_score)
			return z_score, p_value