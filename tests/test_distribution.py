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


def test_skewness_is_near_zero_for_symmetric_series() -> None:
    r = pl.Series([-0.02, -0.01, 0.0, 0.01, 0.02])
    assert skewness(r) == pytest.approx(0.0, abs=0.1)


def test_skewness_is_positive_for_right_skewed_series() -> None:
    r = pl.Series([0.0, 0.0, 0.0, 0.0, 1.0])
    assert skewness(r) > 0


def test_skewness_is_negative_for_left_skewed_series() -> None:
    r = pl.Series([-1.0, 0.0, 0.0, 0.0, 0.0])
    assert skewness(r) < 0


def test_skewness_bias_option_changes_result_for_small_n() -> None:
    rng = np.random.default_rng(0)
    r = pl.Series(rng.normal(0, 1, 30).tolist())
    assert skewness(r, bias=True) != skewness(r, bias=False)


def test_skewness_is_nan_for_constant_series() -> None:
    assert math.isnan(skewness(pl.Series([0.01] * 10)))


def test_skewness_rejects_too_few_observations() -> None:
    with pytest.raises(ValueError):
        skewness(pl.Series([0.01, 0.02]))


def test_excess_kurtosis_is_near_zero_for_normal_sample() -> None:
    rng = np.random.default_rng(0)
    r = pl.Series(rng.normal(0, 1, 10000).tolist())
    assert abs(excess_kurtosis(r)) < 0.2


def test_excess_kurtosis_is_positive_for_fat_tailed_sample() -> None:
    r = pl.Series([0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 10.0])
    assert excess_kurtosis(r) > 0.0


def test_excess_kurtosis_is_nan_for_constant_series() -> None:
    assert math.isnan(excess_kurtosis(pl.Series([0.01] * 10)))


def test_excess_kurtosis_rejects_too_few_observations() -> None:
    with pytest.raises(ValueError):
        excess_kurtosis(pl.Series([0.01, 0.02, 0.03]))


def test_jarque_bera_returns_statistic_and_pvalue_in_valid_range(daily_returns: pl.Series) -> None:
    result = jarque_bera(daily_returns)
    assert isinstance(result, JarqueBeraResult)
    assert result.statistic >= 0.0
    assert 0.0 <= result.p_value <= 1.0


def test_jarque_bera_gives_high_pvalue_for_normal_data() -> None:
    rng = np.random.default_rng(42)
    r = pl.Series(rng.normal(0, 1, 5000).tolist())
    result = jarque_bera(r)
    assert result.p_value > 0.01


def test_jarque_bera_gives_low_pvalue_for_non_normal_data() -> None:
    r = pl.Series([0.0] * 500 + [10.0] * 10)
    result = jarque_bera(r)
    assert result.p_value < 0.05


def test_jarque_bera_result_is_frozen(daily_returns: pl.Series) -> None:
    result = jarque_bera(daily_returns)
    with pytest.raises((AttributeError, TypeError)):
        result.statistic = 0.0  # type: ignore[misc]


def test_jarque_bera_rejects_too_few_observations() -> None:
    with pytest.raises(ValueError):
        jarque_bera(pl.Series([0.01, 0.02, 0.03]))


def test_autocorrelation_is_near_zero_for_white_noise() -> None:
    rng = np.random.default_rng(42)
    r = pl.Series(rng.normal(0, 1, 1000).tolist())
    assert abs(autocorrelation(r)) < 0.1


def test_autocorrelation_is_positive_for_ar1_process() -> None:
    rng = np.random.default_rng(0)
    n = 500
    e = rng.normal(0, 1, n)
    x = np.zeros(n)
    for i in range(1, n):
        x[i] = 0.5 * x[i - 1] + e[i]
    assert autocorrelation(pl.Series(x.tolist())) > 0.2


def test_autocorrelation_rejects_lag_zero() -> None:
    with pytest.raises(ValueError):
        autocorrelation(pl.Series([0.01] * 10), lag=0)


def test_autocorrelation_rejects_negative_lag() -> None:
    with pytest.raises(ValueError):
        autocorrelation(pl.Series([0.01] * 10), lag=-1)


def test_autocorrelation_rejects_lag_larger_than_series() -> None:
    with pytest.raises(ValueError):
        autocorrelation(pl.Series([0.01] * 3), lag=5)
