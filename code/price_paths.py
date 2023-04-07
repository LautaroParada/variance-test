import numpy as np
from scipy.stats import halfnorm

class PricePaths(object):
    
    def __init__(self, n:int, T:int, s0:float):

        if n <= 0 or T <= 0 or s0 <= 0:
            raise ValueError("Los parámetros n, T y s0 deben ser números positivos")

        self.n = n                      # number of paths to generate
        self.T = T                      # number of observations to generate
        self.s0 = s0                    # initial price
        
        self.h = self.n / self.T        # step to move in each step
        self.r0 = self.s0 / 100         # Initial rate, based on the price
        

    # -------------------------------------------
    # Brownian motion
    # -------------------------------------------


	# -------------------------------------------
	# Geometric Brownian motion
	# -------------------------------------------
    
    
    
	# -------------------------------------------
	# Merton Jump Diffusion Stochastic Process
	# -------------------------------------------
    

    
	# -------------------------------------------
	# Vasicek Interest Rate Model
	# -------------------------------------------
    

    
	# -------------------------------------------
	# Cox Ingersoll Ross (CIR) stochastic proces - RATES
	# -------------------------------------------
    

    # -------------------------------------------
	# Heston Stochastic Volatility Process
	# -------------------------------------------
    

    
    # -------------------------------------------
	# Ornstein–Uhlenbeck Process (Mean reverting)
	# -------------------------------------------
    
