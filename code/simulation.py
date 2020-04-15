# test the class
if __name__=='__main__':


	from statistics import mean
	import numpy as np
	import time
	from variance_test import *
	from price_paths import *
	from visuals import *

	# TESTING THE STATISTIC AGAINST GENERATED DATA
	# THIS DATA FOLLOWS A HESTON MODEL
	# FOR A BROWNIAN MODEL, PLEASE CHANGE THE PARAMETER
	# stochastic_volatility=False  (in price_processes and vrt methods)

	# number of variables to generate
	tickers = 1000
	# create the objects
	sims = SimPaths(n=tickers, T=252)
	emh = EMH()

	# # generate synthetic data
	start_fake_data = time.time()

	sims = SimPaths(n=500, T=252, s0=100)

	browmians = sims.brownian_prices()
	gbms = sims.gbm_prices()
	mertons = sims.merton_prices()
	ous = sims.ou_prices(mu=100)
	hestons = sims.heston_prices(mu=1, kappa=0.2, lamba=0.02, rho=0.85)

	# stacking all prices in just one matrix
	prices = np.hstack((browmians, gbms, mertons, ous, hestons))

	# Measuring time
	end_fake_data = time.time()
	print(f'It took {end_fake_data - start_fake_data} seconds generate synthetic data')

	# check the priori results for different ranges
	vrt_visuals = VRTVisuals()
	vrt_visuals.stat_plot(process='merton', total_samples=1000)
	
	# choosen lags to lookback
	# It's five because the aggregation represent a week of daily values
	q = 5
	print('\nMarket simulated prices')
	# plot the head of the numpy matrix
	print(prices.shape)
	# generate the asymtotic values - to compare the stadistics against the real
	# values of a distribuition
	dist_values = np.random.normal(0, 1, tickers)
	# calculate the stadistics
	start_vrt = time.time()

	z_values = []
	p_values = []
	for i in range(tickers):
		_z, _p = emh.vrt(prices[:, i], q, heteroskedastic=True)
		z_values.append(_z)
		p_values.append(_p)

	end_vrt = time.time()
	print('It took %.4f seconds to calculate the statistics for %s tickers' % (end_vrt - start_vrt, tickers))
	# check the cetral measures for the z-values
	print(f'The mean values for the p-values is {mean(p_values)}')

	# plot the results obtained on simulated asset prices
	vrt_visuals.densities(dist_values, z_values)

