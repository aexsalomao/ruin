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


class TestSharpeStandardError:
    def test_positive(self, daily_returns: pl.Series) -> None:
        se = sharpe_standard_error(daily_returns, periods_per_year=252)
        assert se > 0.0

    def test_decreases_with_more_data(self) -> None:
        rng = np.random.default_rng(0)
        r_small = pl.Series(rng.normal(0.001, 0.01, 100).tolist())
        r_large = pl.Series(rng.normal(0.001, 0.01, 1000).tolist())
        se_small = sharpe_standard_error(r_small, periods_per_year=252)
        se_large = sharpe_standard_error(r_large, periods_per_year=252)
        assert se_small > se_large

    def test_constant_series_is_nan(self) -> None:
        r = pl.Series([0.001] * 20)
        assert math.isnan(sharpe_standard_error(r, periods_per_year=252))

    def test_too_few_raises(self) -> None:
        with pytest.raises(ValueError):
            sharpe_standard_error(pl.Series([0.01, 0.02]), periods_per_year=252)


class TestSharpeCI:
    def test_lower_lt_upper(self, daily_returns: pl.Series) -> None:
        lo, hi = sharpe_confidence_interval(daily_returns, periods_per_year=252)
        assert lo < hi

    def test_sr_within_ci(self, daily_returns: pl.Series) -> None:
        lo, hi = sharpe_confidence_interval(daily_returns, periods_per_year=252)
        sr = sharpe_ratio(daily_returns, periods_per_year=252)
        assert lo <= sr <= hi

    def test_wider_ci_for_lower_confidence_is_wrong_way(self, daily_returns: pl.Series) -> None:
        # Higher confidence requires wider CI.
        lo95, hi95 = sharpe_confidence_interval(
            daily_returns, periods_per_year=252, confidence=0.95
        )
        lo80, hi80 = sharpe_confidence_interval(
            daily_returns, periods_per_year=252, confidence=0.80
        )
        assert (hi95 - lo95) > (hi80 - lo80)

    def test_ci_returns_tuple(self, daily_returns: pl.Series) -> None:
        ci = sharpe_confidence_interval(daily_returns, periods_per_year=252)
        assert isinstance(ci, tuple)
        assert len(ci) == 2


class TestBootstrapMetric:
    def test_returns_three_tuple(self, daily_returns: pl.Series) -> None:
        result = bootstrap_metric(
            lambda r: sharpe_ratio(r, periods_per_year=252),
            daily_returns,
            n_samples=100,
            seed=0,
        )
        assert len(result) == 3

    def test_lower_lt_upper(self, daily_returns: pl.Series) -> None:
        _, lo, hi = bootstrap_metric(
            lambda r: sharpe_ratio(r, periods_per_year=252),
            daily_returns,
            n_samples=200,
            seed=42,
        )
        assert lo < hi

    def test_point_estimate_matches_scalar(self, daily_returns: pl.Series) -> None:
        sr_direct = sharpe_ratio(daily_returns, periods_per_year=252)
        point, _, _ = bootstrap_metric(
            lambda r: sharpe_ratio(r, periods_per_year=252),
            daily_returns,
            n_samples=100,
            seed=0,
        )
        assert math.isclose(point, sr_direct, rel_tol=1e-9)

    def test_reproducibility_with_seed(self, daily_returns: pl.Series) -> None:
        args = (lambda r: sharpe_ratio(r, periods_per_year=252), daily_returns)
        kwargs = {"n_samples": 100, "seed": 123}
        r1 = bootstrap_metric(*args, **kwargs)
        r2 = bootstrap_metric(*args, **kwargs)
        assert r1 == r2

    def test_different_seeds_differ(self, daily_returns: pl.Series) -> None:
        fn = lambda r: sharpe_ratio(r, periods_per_year=252)  # noqa: E731
        _, lo1, hi1 = bootstrap_metric(fn, daily_returns, n_samples=100, seed=1)
        _, lo2, hi2 = bootstrap_metric(fn, daily_returns, n_samples=100, seed=2)
        # Extremely unlikely for both endpoints to match byte-for-byte with different seeds.
        assert (lo1, hi1) != (lo2, hi2)
