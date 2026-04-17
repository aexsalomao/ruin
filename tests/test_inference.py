"""Tests for ruin.inference module."""

from __future__ import annotations

import math

import numpy as np
import polars as pl
import pytest

from ruin.inference import (
    bootstrap_metric,
    sharpe_confidence_interval,
    sharpe_standard_error,
)
from ruin.ratios import sharpe_ratio


def _sharpe(r: pl.Series) -> float:
    return sharpe_ratio(r, periods_per_year=252)


def test_sharpe_standard_error_is_positive(daily_returns: pl.Series) -> None:
    se = sharpe_standard_error(daily_returns, periods_per_year=252)
    assert se > 0.0


def test_sharpe_standard_error_decreases_with_more_data() -> None:
    rng = np.random.default_rng(0)
    r_small = pl.Series(rng.normal(0.001, 0.01, 100).tolist())
    r_large = pl.Series(rng.normal(0.001, 0.01, 1000).tolist())
    se_small = sharpe_standard_error(r_small, periods_per_year=252)
    se_large = sharpe_standard_error(r_large, periods_per_year=252)
    assert se_small > se_large


def test_sharpe_standard_error_is_nan_for_constant_series() -> None:
    r = pl.Series([0.001] * 20)
    assert math.isnan(sharpe_standard_error(r, periods_per_year=252))


def test_sharpe_standard_error_rejects_too_few_observations() -> None:
    with pytest.raises(ValueError):
        sharpe_standard_error(pl.Series([0.01, 0.02]), periods_per_year=252)


def test_sharpe_confidence_interval_lower_less_than_upper(daily_returns: pl.Series) -> None:
    lo, hi = sharpe_confidence_interval(daily_returns, periods_per_year=252)
    assert lo < hi


def test_sharpe_confidence_interval_contains_point_estimate(daily_returns: pl.Series) -> None:
    lo, hi = sharpe_confidence_interval(daily_returns, periods_per_year=252)
    sr = sharpe_ratio(daily_returns, periods_per_year=252)
    assert lo <= sr <= hi


def test_sharpe_confidence_interval_widens_with_higher_confidence(
    daily_returns: pl.Series,
) -> None:
    lo95, hi95 = sharpe_confidence_interval(daily_returns, periods_per_year=252, confidence=0.95)
    lo80, hi80 = sharpe_confidence_interval(daily_returns, periods_per_year=252, confidence=0.80)
    assert (hi95 - lo95) > (hi80 - lo80)


def test_sharpe_confidence_interval_returns_two_tuple(daily_returns: pl.Series) -> None:
    ci = sharpe_confidence_interval(daily_returns, periods_per_year=252)
    assert isinstance(ci, tuple)
    assert len(ci) == 2


def test_bootstrap_metric_returns_three_tuple(daily_returns: pl.Series) -> None:
    result = bootstrap_metric(_sharpe, daily_returns, n_samples=100, seed=0)
    assert len(result) == 3


def test_bootstrap_metric_lower_less_than_upper(daily_returns: pl.Series) -> None:
    _, lo, hi = bootstrap_metric(_sharpe, daily_returns, n_samples=200, seed=42)
    assert lo < hi


def test_bootstrap_metric_point_matches_direct_estimate(daily_returns: pl.Series) -> None:
    sr_direct = sharpe_ratio(daily_returns, periods_per_year=252)
    point, _, _ = bootstrap_metric(_sharpe, daily_returns, n_samples=100, seed=0)
    assert point == pytest.approx(sr_direct, rel=1e-9)


def test_bootstrap_metric_is_deterministic_given_seed(daily_returns: pl.Series) -> None:
    r1 = bootstrap_metric(_sharpe, daily_returns, n_samples=100, seed=123)
    r2 = bootstrap_metric(_sharpe, daily_returns, n_samples=100, seed=123)
    assert r1 == r2


def test_bootstrap_metric_differs_across_seeds(daily_returns: pl.Series) -> None:
    _, lo1, hi1 = bootstrap_metric(_sharpe, daily_returns, n_samples=100, seed=1)
    _, lo2, hi2 = bootstrap_metric(_sharpe, daily_returns, n_samples=100, seed=2)
    assert (lo1, hi1) != (lo2, hi2)
