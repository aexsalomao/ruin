"""Tests for ruin.distribution module."""

from __future__ import annotations

import math

import numpy as np
import polars as pl
import pytest

from ruin.distribution import (
    JarqueBeraResult,
    autocorrelation,
    excess_kurtosis,
    jarque_bera,
    skewness,
)


class TestSkewness:
    def test_symmetric_near_zero(self) -> None:
        r = pl.Series([-0.02, -0.01, 0.0, 0.01, 0.02])
        assert math.isclose(skewness(r), 0.0, abs_tol=0.1)

    def test_right_skewed(self) -> None:
        r = pl.Series([0.0, 0.0, 0.0, 0.0, 1.0])
        assert skewness(r) > 0

    def test_left_skewed(self) -> None:
        r = pl.Series([-1.0, 0.0, 0.0, 0.0, 0.0])
        assert skewness(r) < 0

    def test_bias_option_differs(self) -> None:
        rng = np.random.default_rng(0)
        r = pl.Series(rng.normal(0, 1, 30).tolist())
        # Bias correction shifts value for small n
        assert skewness(r, bias=True) != skewness(r, bias=False)

    def test_constant_series_is_nan(self) -> None:
        assert math.isnan(skewness(pl.Series([0.01] * 10)))

    def test_too_few_raises(self) -> None:
        with pytest.raises(ValueError):
            skewness(pl.Series([0.01, 0.02]))


class TestExcessKurtosis:
    def test_normal_near_zero(self) -> None:
        rng = np.random.default_rng(0)
        r = pl.Series(rng.normal(0, 1, 10000).tolist())
        assert abs(excess_kurtosis(r)) < 0.2

    def test_leptokurtic_positive(self) -> None:
        # Series with an outlier is leptokurtic (fat tails)
        r = pl.Series([0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 10.0])
        assert excess_kurtosis(r) > 0.0

    def test_constant_series_is_nan(self) -> None:
        assert math.isnan(excess_kurtosis(pl.Series([0.01] * 10)))

    def test_too_few_raises(self) -> None:
        with pytest.raises(ValueError):
            excess_kurtosis(pl.Series([0.01, 0.02, 0.03]))


class TestJarqueBera:
    def test_returns_correct_type(self, daily_returns: pl.Series) -> None:
        result = jarque_bera(daily_returns)
        assert isinstance(result, JarqueBeraResult)
        assert result.statistic >= 0.0
        assert 0.0 <= result.p_value <= 1.0

    def test_normal_data_high_pvalue(self) -> None:
        rng = np.random.default_rng(42)
        r = pl.Series(rng.normal(0, 1, 5000).tolist())
        result = jarque_bera(r)
        assert result.p_value > 0.01

    def test_non_normal_low_pvalue(self) -> None:
        # Heavily skewed / fat-tailed distribution
        r = pl.Series([0.0] * 500 + [10.0] * 10)
        result = jarque_bera(r)
        assert result.p_value < 0.05

    def test_frozen_dataclass(self, daily_returns: pl.Series) -> None:
        result = jarque_bera(daily_returns)
        with pytest.raises((AttributeError, TypeError)):
            result.statistic = 0.0  # type: ignore[misc]

    def test_too_few_raises(self) -> None:
        with pytest.raises(ValueError):
            jarque_bera(pl.Series([0.01, 0.02, 0.03]))


class TestAutocorrelation:
    def test_white_noise_near_zero(self) -> None:
        rng = np.random.default_rng(42)
        r = pl.Series(rng.normal(0, 1, 1000).tolist())
        assert abs(autocorrelation(r)) < 0.1

    def test_ar1_positive_autocorrelation(self) -> None:
        rng = np.random.default_rng(0)
        n = 500
        e = rng.normal(0, 1, n)
        x = np.zeros(n)
        for i in range(1, n):
            x[i] = 0.5 * x[i - 1] + e[i]
        assert autocorrelation(pl.Series(x.tolist())) > 0.2

    def test_invalid_lag_zero(self) -> None:
        with pytest.raises(ValueError):
            autocorrelation(pl.Series([0.01] * 10), lag=0)

    def test_invalid_lag_negative(self) -> None:
        with pytest.raises(ValueError):
            autocorrelation(pl.Series([0.01] * 10), lag=-1)

    def test_lag_larger_than_series_raises(self) -> None:
        with pytest.raises(ValueError):
            autocorrelation(pl.Series([0.01] * 3), lag=5)
