import matplotlib.pyplot as plt
import seaborn as sns

sns.set_style('whitegrid')

from variance_test import EMH
from price_paths import PricePaths

class VRTVisuals(object):

	# -----------------------------------
	# Helper functions
	# -----------------------------------	

	def densities(self, ref_stats, z_stats):
		"""
		plot the densities for a series of data

		These ref_stats (normal distribuition) values are 
		colored red, because represents the light that 'stop' 
		the rejection of the null hypothesis. Analog representation
		is made for the z scores from the test.
		"""
		# plot the normal distribuition values
		sns.kdeplot(ref_stats, shade=True, color="r")
		# plot the values for the statistic
		sns.kdeplot(z_stats, shade=True, color="g")
		title_text = "Comparison of densities of a Normal Distribuited variable(red)\nagainst the Z* scores computed for the analyzed series of log prices"
		plt.title(title_text)
		plt.show()

	def stat_plot(self, process:str='brownian', q_range:list =[5, 10], 
                   total_samples:int =500, initial_price:float=1.0, 
                   stat_type:str ='mr', **kwargs):
	    """
	    Generate the plots for the DIFFERENCES USING THE OVERLAPPING SAMPLES ESTIMATOR

	    """
	    # creating the objecto for the price paths
	    sims = PricePaths(n=1, T=total_samples*2, s0=initial_price)

	    if process == 'brownian':
	    	_simulator = sims.brownian_prices
	    elif process == 'gmb':
	    	_simulator = sims.gbm_prices
	    elif process == 'merton':
	    	_simulator = sims.merton_prices
	    else:
	    	print(f'Your option is {process}. Please select one of these: brownian, gbm, merton, heston, vas or cir')
	    	return None

	    # error handling 
	    if len(q_range) != 2:
	        print('Please select at most 2 ranges to analyze, e.g., [3,6]')
	        return None
	    
	    # setting the desired statistic
	    if  stat_type == 'md':
	        stat = EMH()._md
	        statName = stat_type.capitalize() # capitalize the name
	    elif stat_type == 'mr':
	        stat = EMH()._mr
	        statName = stat_type.capitalize() # capitalize the name
	    else:
	        print('not valid statistic, pelase try md or mr')
	        return None
	    
	    # generating the statistics for values without Stochastic Volatility
	    _statsQ1 = [stat(X=_simulator(**kwargs), q=q_range[0]) for sample in range(q_range[0], total_samples)]
	    _statsQ2 = [stat(X=_simulator(**kwargs), q=q_range[1]) for sample in range(q_range[1], total_samples)]
	    
	    # generating the statistics for values with Stochastic Volatility
	    _statsVolQ1 = [stat(X=_simulator(**kwargs), q=q_range[0],
	                       unbiased=False) for sample in range(q_range[0], total_samples)]
	    _statsVolQ2 = [stat(X=_simulator(**kwargs), q=q_range[1],
	                       unbiased=False) for sample in range(q_range[1], total_samples)]
	    
	    # Creating the plots        
	    fig, axes = plt.subplots(2, 2, figsize=(15,15))
	    fig.suptitle(f' {statName} values for {process.capitalize()} prices paths', fontsize=16)
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
