import numpy as np
import scipy.stats as st


class EMH(object):
    """Empirical tests for the Efficient Market Hypothesis."""

    # ------------------------------------------------------------------
    # Homoskedastic Null Hypothesis
    # ------------------------------------------------------------------

    def __mu(self, X, annualize: bool = True):
        """Estimate the drift component of a log price process."""
        prices = np.asarray(X, dtype=float)
        n_obs = prices.shape[0]
        if n_obs < 2:
            raise ValueError("At least two observations are required to estimate drift.")

        mu_est = (prices[-1] - prices[0]) / n_obs
        return mu_est * 252 if annualize else mu_est

    def __vol_a(self, X, q: int = 1, unbiased: bool = True, annualize: bool = True):
        """Sample variance of aggregated first differences."""
        if q <= 0:
            raise ValueError("Aggregation horizon q must be a positive integer.")

        prices = np.asarray(X, dtype=float)
        n_obs = prices.shape[0]
        if n_obs <= q:
            raise ValueError("Not enough observations for the requested aggregation horizon.")

        mu_est = self.__mu(X=prices, annualize=False)
        max_index = n_obs - 1
        num_increments = max_index // q
        if num_increments < 1:
            raise ValueError("Not enough data to compute aggregated first differences.")

        indices = np.arange(1, num_increments + 1, dtype=int)
        increments = prices[indices * q] - prices[indices * q - q] - (q * mu_est)
        sigma_est = float(np.dot(increments, increments))

        denominator = num_increments - 1 if unbiased else num_increments
        if denominator <= 0:
            raise ValueError("Degrees of freedom must be positive.")

        sigma_est /= denominator
        return float(np.sqrt(sigma_est * 252)) if annualize else float(sigma_est)

    def __vol_b(self, X, q: int = 1, unbiased: bool = True, annualize: bool = True):
        """Variance of the q-step differences of the price process."""
        if q <= 0:
            raise ValueError("Aggregation horizon q must be a positive integer.")

        prices = np.asarray(X, dtype=float)
        n_obs = prices.shape[0]
        if n_obs <= q:
            raise ValueError("Not enough observations for the requested aggregation horizon.")

        mu_est = self.__mu(X=prices, annualize=False)
        n = int(np.floor(n_obs / q))
        if n <= 1:
            raise ValueError("At least two aggregated periods are required to estimate variance.")

        upper_bound = n * q
        q_diffs = prices[q:upper_bound] - prices[: upper_bound - q] - (q * mu_est)
        sigma_est = float(np.dot(q_diffs, q_diffs))

        if unbiased:
            m = q * (upper_bound - q + 1) * (1 - (q / upper_bound))
            if m <= 0:
                raise ValueError("Unbiased adjustment resulted in a non-positive denominator.")
            sigma_est /= m
        else:
            sigma_est /= n * (q ** 2)

        return float(np.sqrt(sigma_est * 252)) if annualize else float(sigma_est)

    def __md(self, X, q: int, unbiased: bool = True, annualize: bool = True):
        """Difference between the two volatility estimators."""

        return self.__vol_b(
            X=X, q=q, unbiased=unbiased, annualize=annualize
        ) - self.__vol_a(X=X, q=q, unbiased=unbiased, annualize=annualize)

    def __mr(self, X, q: int, unbiased: bool = True, annualize: bool = True):
        """Centered variance ratio statistic."""

        numerator = self.__vol_b(X=X, q=q, unbiased=unbiased, annualize=annualize)
        denominator = self.__vol_a(X=X, q=q, unbiased=unbiased, annualize=annualize)
        return (numerator / denominator) - 1

    def __h1(self, X, q: int, centered: bool = True, unbiased: bool = True, annualize: bool = True):
        """IID Gaussian null hypothesis."""

        if q <= 0:
            raise ValueError("Aggregation horizon q must be a positive integer.")

        n = np.floor(np.asarray(X, dtype=float).shape[0] / q)
        if n <= 0:
            raise ValueError("Not enough observations for the requested aggregation horizon.")

        if centered:
            z1 = np.sqrt(n * q) * self.__mr(
                X=X, q=q, unbiased=unbiased, annualize=annualize
            )
            asymp = (2 * (2 * q - 1) * (q - 1)) / (3 * q)
            return z1 / np.sqrt(asymp)

        z1 = np.sqrt(n * q) * self.__md(
            X=X, q=q, unbiased=unbiased, annualize=annualize
        )
        return z1

    # ------------------------------------------------------------------
    # Heteroskedastic Null Hypothesis
    # ------------------------------------------------------------------

    def __delta(self, X, q: int, j: int):
        """Helper for the heteroskedastic variance estimator."""

        if q <= 0:
            raise ValueError("Aggregation horizon q must be a positive integer.")
        if j <= 0 or j >= q:
            raise ValueError("Lag j must satisfy 0 < j < q.")

        prices = np.asarray(X, dtype=float)
        mu_est = self.__mu(prices, False)
        n = int(np.floor(prices.shape[0] / q))
        upper_bound = n * q
        if upper_bound <= j + 1:
            raise ValueError("Not enough data to evaluate the heteroskedastic adjustment.")

        diffs = prices[1:upper_bound] - prices[: upper_bound - 1] - mu_est
        denominator = float(np.dot(diffs, diffs))
        if denominator <= 0:
            raise ValueError("Second moment of the differenced series is non-positive.")

        lead = diffs[j:]
        lagged = diffs[:-j]
        numerator = float(np.dot(lead ** 2, lagged ** 2))
        return (upper_bound * numerator) / (denominator ** 2)

    def __v_hat(self, X, q: int):
        """Asymptotic variance of the centered ratio under heteroskedasticity."""

        if q < 3:
            raise ValueError("q must be at least 3 for the heteroskedastic variance estimator.")

        v_hat = 0.0
        for j in range(1, q - 1):
            delta = self.__delta(X=X, q=q, j=j)
            weight = (2 * (q - j) / q) ** 2
            v_hat += weight * delta

        if v_hat <= 0:
            raise ValueError("Asymptotic variance estimate must be positive.")

        return float(v_hat)

    def __h2(self, X, q: int, centered: bool = True, unbiased: bool = True, annualize: bool = True):
        """Heteroskedastic null hypothesis."""

        if not centered:
            raise ValueError("Non-centered heteroskedastic statistic is not implemented.")

        n = np.floor(np.asarray(X, dtype=float).shape[0] / q)
        if n <= 0:
            raise ValueError("Not enough observations for the requested aggregation horizon.")

        z2 = np.sqrt(n * q) * self.__mr(
            X=X, q=q, unbiased=unbiased, annualize=annualize
        )
        v_hat = self.__v_hat(X=X, q=q)
        return z2 / np.sqrt(v_hat)

    # ------------------------------------------------------------------
    # Variance Ratio Test Interface
    # ------------------------------------------------------------------

    def vrt(
        self,
        X,
        q: int,
        heteroskedastic: bool = True,
        centered: bool = True,
        unbiased: bool = True,
        annualize: bool = True,
    ):
        """Compute the Variance Ratio test statistic and its p-value."""

        if heteroskedastic:
            z_score = self.__h2(
                X=X, q=q, centered=centered, unbiased=unbiased, annualize=annualize
            )
        else:
            z_score = self.__h1(
                X=X, q=q, centered=centered, unbiased=unbiased, annualize=annualize
            )

        p_value = 1 - st.norm.cdf(z_score)
        return z_score, p_value
