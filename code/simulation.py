# test the class
if __name__=='__main__':


	from statistics import mean
	import numpy as np
	import time
	from main import EMH

	# TESTING THE STATISTIC AGAINST GENERATED DATA
	# THIS DATA FOLLOWS A BROWMIAN MODEL

	# number of variables to generate
	tickers = int(1e2)
	# create the object
	emh = EMH()
	# generate synthetic data
	start_fake_data = time.time()
	sim_prices = emh.price_processes(n=tickers)

	# Measuring time
	end_fake_data = time.time()
	print('It took %.4f seconds to generate %s tickers' % (end_fake_data - start_fake_data, tickers))
	# check the priori results for different ranges
	emh._stat_plot(q_range=[5, 10])
	# choosen lags to lookback
	# It's five because the aggregation represent a week of daily values
	q = 5
	print('\nMarket simulated prices')
	# plot the head of the numpy matrix
	print(sim_prices[:5, :2])
	# generate the asymtotic values - to compare the stadistics against the real
	# values of a distribuition
	dist_values = np.random.normal(0, 1, tickers)
	# calculate the stadistics
	start_vrt = time.time()

	z_values = []
	p_values = []
	for i in range(tickers):
		_z, _p = emh.vrt(sim_prices[:, i], q, heteroskedastic=False)
		z_values.append(_z)
		p_values.append(_p)

	end_vrt = time.time()
	print('It took %.4f seconds to calculate the statistics for %s tickers' % (end_vrt - start_vrt, tickers))
	# check the cetral measures for the z-values
	print(f'The mean values for the z-values is {mean(z_values)}')

	# plot the results obtained on simulated asset prices
	emh._densities(dist_values, z_values)

