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

	browmians = sims.brownian_prices()
	gbms = sims.gbm_prices()
	mertons = sims.merton_prices()

	_sim = np.hstack((browmians, gbms))
	sim_prices = np.hstack((_sim, mertons))

	# Measuring time
	end_fake_data = time.time()
	print(f'It took {end_fake_data - start_fake_data} seconds')

	# check the priori results for different ranges
	vrt_visuals = VRTVisuals()
	vrt_visuals.stat_plot(process='merton', total_samples=1000)
	
	# choosen lags to lookback
	# It's five because the aggregation represent a week of daily values
	q = 5
	print('\nMarket simulated prices')
	# plot the head of the numpy matrix
	print(sim_prices.shape)
	# generate the asymtotic values - to compare the stadistics against the real
	# values of a distribuition
	dist_values = np.random.normal(0, 1, tickers)
	# calculate the stadistics
	start_vrt = time.time()

	z_values = []
	p_values = []
	for i in range(tickers):
		_z, _p = emh.vrt(sim_prices[:, i], q, heteroskedastic=True)
		z_values.append(_z)
		p_values.append(_p)

	end_vrt = time.time()
	print('It took %.4f seconds to calculate the statistics for %s tickers' % (end_vrt - start_vrt, tickers))
	# check the cetral measures for the z-values
	print(f'The mean values for the z-values is {mean(z_values)}')

	# plot the results obtained on simulated asset prices
	vrt_visuals.densities(dist_values, z_values)

