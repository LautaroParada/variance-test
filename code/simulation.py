if __name__=='__main__':
    
    import numpy as np
    import time
    from statistics import mean
    import scipy.stats as stats
    import pylab
    
    from price_paths import PricePaths
    from variance_test import EMH
    from visuals import VRTVisuals
    
    n = 10                         # number of time series to simulate
    T = 100                         # number of steps
    s0 = 100.0                        # Initial short term rate
    
    sim = PricePaths(n, T, s0)      # Initialization of the simulation class
    emh = EMH()                     # Initialization of the test class
    vrt_visuals = VRTVisuals()
    
    # General parameters
    mu = 0.09                       # Long term mean return
    sigma = 0.1                    # Volatility
    
    # Particular parameters
    lam = T / 2                     # Intensity of the Jump (Merton process)
    k = 0.1                         # (Heston)
    theta = 0.06                    # (Heston)
    
    # generate synthetic data
    start_fake_data = time.time()
    
    # Simulate the random prices - Prices paths
    gbm = sim.gbm_prices(mu, sigma)                 # Geometric Brownian model 
    merton = sim.merton_prices(mu, sigma, lam)      # Merton model
    hes = sim.heston_prices(mu=mu,                  # Heston model
                            k=k, 
                            theta=theta, 
                            sigma=sigma)
    
    all_proc = np.hstack((gbm, merton, hes))
    
    # Measuring time
    end_fake_data = time.time()
    print(f"It took {round(end_fake_data - start_fake_data, 3)} seconds to simulate {all_proc.shape[1]} time series")
    
    #vrt_visuals.stat_plot(mu=mu, sigma=sigma)
    
    q = 5
    print('Market simulated prices')
    # generate the asymtotic values - to compare the stadistics against the real
    dist_values = np.random.normal(loc=0, scale=1, size=n)
    # calculate the statistics
    start_vrt = time.time()
    
    z_values = []
    p_values = []
    
    for i in range(n):
        _z, _p = emh.vrt(X=all_proc[:, i], q=q, heteroskedastic=True)
        z_values.append(_z)
        p_values.append(_p)
        
    end_vrt = time.time()
    print(f"It took {end_vrt - start_vrt} seconds to calculate the statistics for {all_proc.shape[1]} synthetic paths")
    
    # check the cetral measures for the z-values
    print(f"The mean values for the p-values is {mean(p_values)}")
    
    vrt_visuals.densities(dist_values, z_values)
    
    stats.probplot(z_values, dist='norm', plot=pylab);
    pylab.show()
    
    