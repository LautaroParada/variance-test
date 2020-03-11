
# Variance Ratio Test

These statistical tests provide a descriptive tool for ***examing the stochastic evolution*** of prices trough the time of a financial log price series. The implicit logic behind the test is to reject the Random Walk model (i.e., Efficient Markets Hypothesis) via the *comparison of variances estimators at different sampling intervals.*

The idea is to investigate the quality of the Efficiency of some financial instrument log price series under the two most commonly know null hypothesis:

- **Homoskedastic Increments (strong market efficiency):** the disturbances/increments are IID normal random variables, wherein the variance of its increments is a linear function in the observation interval. This hypothesis corresponds to the Brownian Motion model.

- **Heteroskedastic Increments (semi-strong market efficiency):** the disturbances/increments are independents but not identically distributed (INID), wherein the variance of its increments is a non-linear function in the observation interval. This hypothesis corresponds to the Heston Model. 

- **Model dependant increments (weak market efficiency):** The third form disturbances relax the independence assumption, meaning that it allows for conditional heteroskedastic increments. Therefore, the volatility either has some non-linear structure (conditional on itself), or it is conditional on another random variable. Stochastic processes which employ ARCH (Autoregressive Conditional Heteroscedasticity) and GARCH (Generalized AutoRegressive Conditional Heteroscedasticity) models of volatility belong to this category.

If the test rejects the strong and the semi-strong forms of market efficiency, we can infer with enough statistical evidence **that the variance of the increments has some form of predictability in their structure.** Therefore the returns in the *price series are conditioned to the prior prices or by exogenous variable(s).*

Please be aware that the results of the Variance Ratio test, do not necessarily imply that the stock market is inefficient in the stock price formation or that prices are not rational assessments of fundamental values.

The Variance Ratio test is purely a descriptive tool for examing the stochastic evolution of prices trough time.

For a thoughtful explanation of the test, please visit the papers in which this test is based: 

- Lo, Andrew W. and MacKinlay, Archie Craig, Stock Market Prices Do Not Follow Random Walks: Evidence from a Simple Specification Test (February 1987). NBER Working Paper No. w2168. Available at SSRN: [https://ssrn.com/abstract=346975](https://ssrn.com/abstract=346975)

- Lo, Andrew W. and MacKinlay, Archie Craig, The Size and Power of the Variance Ratio Test in Finite Samples: a Monte Carlo Investigation (June 1988). NBER Working Paper No. t0066. Available at SSRN: [https://ssrn.com/abstract=396681](https://ssrn.com/abstract=396681)


## Future releases will include:

- Distribution of the code in PiPy as a package.

- Include a step by step explanation of the Variance Ratio test in [QuantConnect](https://www.quantconnect.com/).

- Implementation of the Long Term Memory in Stock Market Prices paper of Andrew Lo. More details in the paper 

    - Lo, Andrew W., Long-Term Memory in Stock Market Prices (May 1989). NBER Working Paper No. w2984. Available at SSRN: [https://ssrn.com/abstract=463442](https://ssrn.com/abstract=463442)

- Implementation of other market Efficiency tests.
