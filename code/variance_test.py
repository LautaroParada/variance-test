"""High-performance variance ratio test utilities."""

import numpy as np
import scipy.stats as st


class EMH(object):
    """Empirical tests for the Efficient Market Hypothesis."""

    # ------------------------------------------------------------------
    # Homoskedastic Null Hypothesis
    # ------------------------------------------------------------------

    def __prepare_prices(self, X: np.ndarray) -> np.ndarray:
        """Validate and standardize a price series into a 1-D NumPy array."""

        prices = np.asarray(X, dtype=float).reshape(-1)
        if prices.size < 2:
            raise ValueError("At least two observations are required to estimate drift.")

        return prices

    def __mu(self, X: np.ndarray, annualize: bool = True) -> float:
        """Estimate the drift component of a log price process."""

        prices = self.__prepare_prices(X)
        mu_est = (prices[-1] - prices[0]) / prices.shape[0]
        return float(mu_est * 252) if annualize else float(mu_est)

    def __variance_estimators(
        self,
        prices: np.ndarray,
        q: int = 1,
        unbiased: bool = True,
        annualize: bool = True,
    ) -> tuple[float, float]:
        """Compute both volatility estimators in a single pass for efficiency."""

        if q <= 0:
            raise ValueError("Aggregation horizon q must be a positive integer.")

        series = self.__prepare_prices(prices)
        n_obs = series.shape[0]
        if n_obs <= q:
            raise ValueError("Not enough observations for the requested aggregation horizon.")

        mu_est = (series[-1] - series[0]) / n_obs

        max_index = n_obs - 1
        num_increments = max_index // q
        if num_increments < 1:
            raise ValueError("Not enough data to compute aggregated first differences.")

        upper_bound = int(np.floor(n_obs / q)) * q
        if upper_bound <= q:
            raise ValueError("At least two aggregated periods are required to estimate variance.")

        increments = series[q : (num_increments + 1) * q : q] - series[: num_increments * q : q]
        increments = increments - (q * mu_est)
        sigma_a = float(np.dot(increments, increments))

        denom_a = num_increments - 1 if unbiased else num_increments
        if denom_a <= 0:
            raise ValueError("Degrees of freedom must be positive.")

        sigma_a /= denom_a

        q_diffs = series[q:upper_bound] - series[: upper_bound - q] - (q * mu_est)
        sigma_b = float(np.dot(q_diffs, q_diffs))

        if unbiased:
            m = q * (upper_bound - q + 1) * (1 - (q / upper_bound))
            if m <= 0:
                raise ValueError("Unbiased adjustment resulted in a non-positive denominator.")
            sigma_b /= m
        else:
            sigma_b /= (upper_bound / q) * (q ** 2)

        if annualize:
            sigma_a = float(np.sqrt(sigma_a * 252))
            sigma_b = float(np.sqrt(sigma_b * 252))

        return sigma_a, sigma_b

    def __vol_a(self, X, q: int = 1, unbiased: bool = True, annualize: bool = True):
        """Sample variance of aggregated first differences."""
        vol_a, _ = self.__variance_estimators(X, q=q, unbiased=unbiased, annualize=annualize)
        return vol_a

    def __vol_b(self, X, q: int = 1, unbiased: bool = True, annualize: bool = True):
        """Variance of the q-step differences of the price process."""
        _, vol_b = self.__variance_estimators(X, q=q, unbiased=unbiased, annualize=annualize)
        return vol_b

    def __md(self, X, q: int, unbiased: bool = True, annualize: bool = True):
        """Difference between the two volatility estimators."""
        vol_a, vol_b = self.__variance_estimators(
            X, q=q, unbiased=unbiased, annualize=annualize
        )
        return vol_b - vol_a

    def __mr(self, X, q: int, unbiased: bool = True, annualize: bool = True):
        """Centered variance ratio statistic."""

        vol_a, vol_b = self.__variance_estimators(
            X, q=q, unbiased=unbiased, annualize=annualize
        )
        return (vol_b / vol_a) - 1

    def __h1(self, X, q: int, centered: bool = True, unbiased: bool = True, annualize: bool = True):
        """IID Gaussian null hypothesis."""

        if q <= 0:
            raise ValueError("Aggregation horizon q must be a positive integer.")

        prices = self.__prepare_prices(X)
        n = np.floor(prices.shape[0] / q)
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

        prices = self.__prepare_prices(X)
        n = int(np.floor(prices.shape[0] / q))
        upper_bound = n * q
        if upper_bound <= j + 1:
            raise ValueError("Not enough data to evaluate the heteroskedastic adjustment.")

        mu_est = (prices[-1] - prices[0]) / prices.shape[0]
        diffs = prices[1:upper_bound] - prices[: upper_bound - 1] - mu_est
        denominator = float(np.dot(diffs, diffs))
        if denominator <= 0:
            raise ValueError("Second moment of the differenced series is non-positive.")

        lead = diffs[j:]
        lagged = diffs[:-j]
        numerator = float(np.dot(lead ** 2, lagged ** 2))
        return (upper_bound * numerator) / (denominator ** 2)

    @staticmethod
    def __delta_from_diffs(
        diffs: np.ndarray, denominator: float, upper_bound: int, j: int
    ) -> float:
        """Compute the delta statistic using precomputed differences."""

        if j <= 0 or j >= upper_bound:
            raise ValueError("Lag j must satisfy 0 < j < upper_bound.")

        lead = diffs[j:]
        lagged = diffs[:-j]
        numerator = float(np.dot(lead ** 2, lagged ** 2))
        return (upper_bound * numerator) / (denominator ** 2)

    def __v_hat(self, X, q: int):
        """Asymptotic variance of the centered ratio under heteroskedasticity."""

        if q < 3:
            raise ValueError("q must be at least 3 for the heteroskedastic variance estimator.")

        prices = self.__prepare_prices(X)
        n = int(np.floor(prices.shape[0] / q))
        upper_bound = n * q
        if upper_bound <= q:
            raise ValueError("Not enough observations to evaluate heteroskedastic variance.")

        mu_est = (prices[-1] - prices[0]) / prices.shape[0]
        diffs = prices[1:upper_bound] - prices[: upper_bound - 1] - mu_est
        denominator = float(np.dot(diffs, diffs))
        if denominator <= 0:
            raise ValueError("Second moment of the differenced series is non-positive.")

        v_hat = 0.0
        for j in range(1, q - 1):
            delta = self.__delta_from_diffs(diffs, denominator, upper_bound, j)
            weight = (2 * (q - j) / q) ** 2
            v_hat += weight * delta

        if v_hat <= 0:
            raise ValueError("Asymptotic variance estimate must be positive.")

        return float(v_hat)

    def __h2(self, X, q: int, centered: bool = True, unbiased: bool = True, annualize: bool = True):
        """Heteroskedastic null hypothesis."""

        if not centered:
            raise ValueError("Non-centered heteroskedastic statistic is not implemented.")

        prices = self.__prepare_prices(X)
        n = np.floor(prices.shape[0] / q)
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
