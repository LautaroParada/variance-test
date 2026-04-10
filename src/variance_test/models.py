"""Dataclass models for battery configuration and outcomes."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass
class RobustVRConfig:
    """Configuration contract for optional robust variance-ratio diagnostics."""

    enabled: bool = False
    enable_avr: bool = True
    enable_wbavr: bool = True
    enable_chow_denning: bool = True
    bootstrap_reps: int = 999
    bootstrap_seed: int | None = None
    wild_weights: str = "rademacher"
    chow_denning_calibration: str = "sidak_normal"
    q_list: tuple[int, ...] | None = None
    max_automatic_q: int | None = None

    def __post_init__(self) -> None:
        if self.bootstrap_reps < 199:
            raise ValueError("bootstrap_reps must be at least 199.")
        if self.wild_weights != "rademacher":
            raise ValueError("wild_weights currently supports only 'rademacher'.")
        if self.chow_denning_calibration != "sidak_normal":
            raise ValueError(
                "chow_denning_calibration currently supports only 'sidak_normal'."
            )
        if self.q_list is not None:
            if not self.q_list:
                raise ValueError("q_list cannot be empty when provided.")
            if any((not isinstance(q, int)) or q <= 0 for q in self.q_list):
                raise ValueError("All q_list values must be positive integers.")
            if any(curr <= prev for prev, curr in zip(self.q_list, self.q_list[1:])):
                raise ValueError("q_list must be strictly increasing.")
        if self.max_automatic_q is not None and self.max_automatic_q < 2:
            raise ValueError("max_automatic_q must be at least 2 when provided.")


@dataclass
class BatteryConfig:
    """Configuration contract for the weak-form battery orchestration layer."""

    input_kind: str = "log_prices"
    alpha: float = 0.05
    q_list: tuple[int, ...] = (2, 4, 8, 16)
    ljung_box_lags: tuple[int, ...] = (5, 10, 20)
    runs_test: bool = True
    arch_lm_lags: int = 5
    rolling_window: int | None = None
    rolling_step: int = 1
    seed: int | None = None
    robust_vr: RobustVRConfig | None = None

    def __post_init__(self) -> None:
        if self.input_kind not in {"log_prices", "returns"}:
            raise ValueError("input_kind must be 'log_prices' or 'returns'.")
        if not (0 < self.alpha < 1):
            raise ValueError("alpha must satisfy 0 < alpha < 1.")
        if not self.q_list:
            raise ValueError("q_list cannot be empty.")
        if any((not isinstance(q, int)) or q <= 0 for q in self.q_list):
            raise ValueError("All q_list values must be positive integers.")
        if any(curr <= prev for prev, curr in zip(self.q_list, self.q_list[1:])):
            raise ValueError("q_list must be strictly increasing.")

        if not self.ljung_box_lags:
            raise ValueError("ljung_box_lags cannot be empty.")
        if any((not isinstance(lag, int)) or lag <= 0 for lag in self.ljung_box_lags):
            raise ValueError("All ljung_box_lags values must be positive integers.")

        if self.arch_lm_lags < 1:
            raise ValueError("arch_lm_lags must be at least 1.")
        if self.rolling_window is not None and self.rolling_window < 2:
            raise ValueError("rolling_window must be None or at least 2.")
        if self.rolling_step < 1:
            raise ValueError("rolling_step must be at least 1.")
        if self.robust_vr is not None and not isinstance(self.robust_vr, RobustVRConfig):
            raise ValueError("robust_vr must be None or a RobustVRConfig instance.")


@dataclass
class TestOutcome:
    """Result contract for a single statistical test."""

    name: str
    null_hypothesis: str
    statistic: float | None
    p_value: float | None
    alpha: float
    reject_null: bool | None
    metadata: dict[str, object]
    warnings: list[str]

    def __post_init__(self) -> None:
        if not self.name:
            raise ValueError("name cannot be empty.")
        if not self.null_hypothesis:
            raise ValueError("null_hypothesis cannot be empty.")
        if not (0 < self.alpha < 1):
            raise ValueError("alpha must satisfy 0 < alpha < 1.")

        if self.p_value is not None and not (0 <= self.p_value <= 1):
            raise ValueError("p_value must be in [0, 1] when provided.")

        if self.reject_null is not None and self.p_value is not None:
            expected_reject = self.p_value < self.alpha
            if self.reject_null != expected_reject:
                raise ValueError("reject_null is inconsistent with p_value and alpha.")


@dataclass
class BatteryOutcome:
    """Result contract for a complete weak-form battery execution."""

    input_kind: str
    n_obs: int
    returns_n_obs: int
    tests: dict[str, TestOutcome]
    multiple_testing: dict[str, object]
    rolling: dict[str, object] | None
    warnings: list[str]

    def __post_init__(self) -> None:
        if self.input_kind not in {"log_prices", "returns"}:
            raise ValueError("input_kind must be 'log_prices' or 'returns'.")
        if self.n_obs < 1:
            raise ValueError("n_obs must be at least 1.")
        if self.returns_n_obs < 0:
            raise ValueError("returns_n_obs must be non-negative.")

        for key, value in self.tests.items():
            if not isinstance(key, str) or not key:
                raise ValueError("tests keys must be non-empty strings.")
            if not isinstance(value, TestOutcome):
                raise ValueError("tests values must be TestOutcome instances.")
