import numpy as np
import scipy.stats as st

class EMH(object):

    # -----------------------------------
	# Homoskedastic Null Hypothesis
	# -----------------------------------

	def __mu(self, X, annualize:bool =True):
		"""
	    Given a log price process, this function estimates the value of mu_hat. 
        Mu is the daily component of the returns which is attributable to upward, 
        or downward drift. 
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
			return ( 1 / n ) * ( X[-1] - X[0] ) * 252
		else:
			return ( 1 / n ) * ( X[-1] - X[0] )

	def __vol_a(self, X, q:int =1, unbiased:bool =True, annualize:bool =True):
		"""
		This function calculates 'the sample variance of the first-difference of
		X; it corresponds to the maximum likelihood-estimator of the variance 
		parameter and therefore possesses the usal consistency, asymptotic 
        normality and efficiency properties.'
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
		mu_est = self.__mu(X=X, annualize=False) # MLE for the drift component
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

	def __vol_b(self, X, q:int =1, unbiased:bool =True, annualize:bool =True):
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
		mu_est = self.__mu(X=X, annualize=False)
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

	def __md(self, X, q:int, unbiased:bool =True, annualize:bool =True):
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
		return self.__vol_b(X=X, q=q, unbiased=unbiased, annualize=annualize) - self.__vol_a(X=X, unbiased=unbiased, annualize=annualize)

	def __mr(self, X, q:int, unbiased:bool =True, annualize:bool =True):
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
		return ( self.__vol_b(X=X, q=q, unbiased=unbiased, annualize=annualize) / self.__vol_a(X=X, unbiased=unbiased, annualize=annualize) ) - 1

	def __h1(self, X, q:int, centered:bool =True, unbiased:bool =True, annualize:bool =True):
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
			z1 =  np.sqrt( np.multiply(n, q) ) * self.__mr(X=X, q=q, unbiased=unbiased, annualize=annualize)
			asymp = ( 2 * ( 2 * q - 1 ) * ( q - 1 ) ) / ( 3 * q )
			return z1 * np.power(asymp, -0.5)
		else:
			z1 = np.sqrt( np.multiply(n, q) ) * self.__md(X=X, q=q, unbiased=unbiased, annualize=annualize)
			return z1

    # -----------------------------------
	# Heteroskedastic Null Hypothesis
	# -----------------------------------

	def __delta(self, X, q:int, j:int):
		"""
		 Helper method for the variance estimator v_hat

		"""
		# Get the estimate value for the drift component.
		mu_est = self.__mu(X, False)
		
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


	def __v_hat(self, X, q):
		"""
		Estimator for the value of the asymptotic variance of the __mr statistic. 
		This is equivalent to a weighted sum of the asymptotic variances for each of 
		the autocorrelation co-efficients under the null hypothesis.

		Given a log price process, X, and a sampling interval, q, this 
		method is used to estimate the asymptoticvariance of the __mr statistic in the 
		presence of stochastic volatility. 
		In other words, it is a heteroskedasticity consistent estimator of 
		the variance of the __mr statistic. This parameter is used to estimate the 
		probability that the given log price process was generated by a 
		Brownian Motion model with drift and stochastic volatility. 

		Args:
			X(numpy array): serie of log prices
			q(int): aggregation value, used to calculate the qth differences.

		Returns
			volatility(float): Estimator for the value of the asymptotic 
							  variance of the __mr statistic. 
		"""
		v_hat = 0.0
		for j in range(1, q - 1):
			delta = self.__delta(X=X, q=q, j=j)
			v_hat += ((( 2 * ( q - j ) ) / q ) ** 2 ) * delta

		return v_hat

	def __h2(self, X, q:int, centered:bool =True, unbiased:bool =True, annualize:bool =True):
		"""
		The Heteroskedastic Null Hypothesis
		"""
		n = np.floor( X.shape[0] / q )
		if centered:
			z2 = np.sqrt( np.multiply(n, q) ) * self.__mr(X=X, q=q, unbiased=unbiased, annualize=annualize)
			return z2 * np.power(self.__v_hat(X=X, q=q), -0.5)

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
			z_score = self.__h2(X=X, q=q, centered=centered, unbiased=unbiased, annualize=annualize)
			p_value = 1 - st.norm.cdf(z_score)
			return z_score, p_value
		else:
			z_score = self.__h1(X=X, q=q, centered=centered, unbiased=unbiased, annualize=annualize)
			p_value = 1 - st.norm.cdf(z_score)
			return z_score, p_value
