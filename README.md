# The Variance Ratio Test

This statistical test proves that financial price series do have a predictability component (*at least to some degree*). The implicit logic behind the test is to reject the Random Walk Hypothesis (Efficient Markets model) via the comparison of variances estimators at different sampling intervals. 
The idea is to investigate the quality of the Efficiency of some financial instrument under the two most commonly know null hypothesis:

- The random walk with independently and identically distributed Gaussian increments (strong Efficiency). This hypothesis corresponds to the **Geometric Brownian Motion Model**, wherein volatility allows only for homoskedastic increments (constant). ***Under this hypothesis, the variance is a linear function of time.***

- The random walk with uncorrelated but heteroskedastic increments (semi-strong Efficiency). For this second hypothesis, the identically distributed assumption is relaxed and assumes that the random disturbance is independent and not identically distributed. ***This hypothesis corresponds to the Heston Model. Under this hypothesis, the variance is a non-linear function of time.***

If the test rejects both null hypotheses, we can conclude that the variance of the increments ***does not depend on Brownian motion***, and therefore asset prices have a component of predictability in their structure. That component may be exploited via a Machine Learning algorithm. 

Please be aware that the results do not necessarily imply that the stock market is inefficient in the stock price formation or that prices are not rational assessments of fundamental values.

The variance-ratio test is purely a descriptive tool for examing the stochastic evolution of prices trough time.

For a thoughtful explanation of the test, please visit the papers in which this test is based: 

- Lo, Andrew W. and MacKinlay, Archie Craig, Stock Market Prices Do Not Follow Random Walks: Evidence from a Simple Specification Test (February 1987). NBER Working Paper No. w2168. Available at SSRN: [https://ssrn.com/abstract=346975](https://ssrn.com/abstract=346975)

- Lo, Andrew W. and MacKinlay, Archie Craig, The Size and Power of the Variance Ratio Test in Finite Samples: a Monte Carlo Investigation (June 1988). NBER Working Paper No. t0066. Available at SSRN: [https://ssrn.com/abstract=396681](https://ssrn.com/abstract=396681)


## Future releases:

- Include a step by step explanation of the Variance Ratio test in QuantConnect.

- Implementation of the Long Term Memory in Stock Market Prices paper of Andrew Lo. More details in the paper 


	- Lo, Andrew W., Long-Term Memory in Stock Market Prices (May 1989). NBER Working Paper No. w2984. Available at SSRN: [https://ssrn.com/abstract=463442](https://ssrn.com/abstract=463442)

- Implementation of several market Efficiency tests.
