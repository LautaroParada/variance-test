# Run the simulation

if __name__=='__main__':
    
    import numpy as np
    import matplotlib.pyplot as plt
    import seaborn as sns
    import pandas as pd
    from price_paths import PricePaths
    from variance_test import EMH
    from visuals import VRTVisuals
    
    n = 1                   # number of time series to simulate
    T = 1000                # number of steps
    s0 = 1                  # Initial price or initial short term rate
    
    sim = PricePaths(n, T, s0)
    
    mu = 0.05               # Long term mean return
    sigma = 0.05            # Volatility
    lam = 500               # Intensity of the Jump (Merton process)
    
    bro = sim.brownian_prices(mu, sigma)
    gbm = sim.gbm_prices(mu, sigma)
    merton = sim.merton_prices(mu, sigma, lam)
    hes = sim.heston_prices(rf=0.0, k=0.5, theta=1.0, sigma=sigma)
    
    all_proc = pd.DataFrame(np.vstack((bro, gbm, merton, hes)).T, columns=['Brownian', 'GBM', 'Merton', 'Heston'])
    heatmap  = sns.heatmap(all_proc.corr(), cmap="RdYlGn", cbar_kws={'label': 'Correlation across the models'}, annot=True)
    plt.title('Correlation across the simulated instruments')
    heatmap.set_yticklabels(heatmap.get_yticklabels(), rotation=0) 
    plt.show()
    
    # Price plots
    plt.plot(bro, label='Brownian')
    plt.plot(gbm, label='GBM')
    plt.plot(merton, label='Merton')
    plt.plot(hes, label='Heston')
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
    sigma = 0.001           # Volatility %
    lam = 0.7               # Reversion speed -> 0 <= lambda <= 1
    
    vas = sim.vas_rates(mu, sigma, lambda_=lam)
    cir = sim.cir_rates(mu, sigma, lambda_=lam)
    
    plt.plot(vas, label='Vasicek')
    plt.plot(cir, label='Cox-Ingersoll-Ross')
    plt.plot(np.ones(sim.T) * mu, color='red')
    plt.title('Simulated rate paths')
    plt.ylabel('rate')
    plt.xlabel('step')
    plt.legend()
    plt.show()