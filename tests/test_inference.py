"""Tests for ruin.inference module."""

from __future__ import annotations

import math

import polars as pl
import pytest

from ruin.inference import bootstrap_metric, sharpe_confidence_interval, sharpe_standard_error
from ruin.ratios import sharpe_ratio


class TestSharpeStandardError:
    def test_positive(self, daily_returns):
        se = sharpe_standard_error(daily_returns, periods_per_year=252)
        assert se > 0.0

    def test_decreases_with_more_data(self):
        import numpy as np
        rng = np.random.default_rng(0)
        r_small = pl.Series(rng.normal(0.001, 0.01, 100).tolist())
        r_large = pl.Series(rng.normal(0.001, 0.01, 1000).tolist())
        se_small = sharpe_standard_error(r_small, periods_per_year=252)
        se_large = sharpe_standard_error(r_large, periods_per_year=252)
        assert se_small > se_large


class TestSharpeCI:
    def test_lower_lt_upper(self, daily_returns):
        lo, hi = sharpe_confidence_interval(daily_returns, periods_per_year=252)
        assert lo < hi

    def test_sr_within_ci(self, daily_returns):
        lo, hi = sharpe_confidence_interval(daily_returns, periods_per_year=252)
        sr = sharpe_ratio(daily_returns, periods_per_year=252)
        assert lo <= sr <= hi

    def test_wider_ci_for_lower_confidence(self, daily_returns):
        lo95, hi95 = sharpe_confidence_interval(daily_returns, periods_per_year=252, confidence=0.95)
        lo80, hi80 = sharpe_confidence_interval(daily_returns, periods_per_year=252, confidence=0.80)
        assert (hi95 - lo95) > (hi80 - lo80)


class TestBootstrapMetric:
    def test_returns_three_tuple(self, daily_returns):
        result = bootstrap_metric(
            lambda r: sharpe_ratio(r, periods_per_year=252),
            daily_returns,
            n_samples=100,
            seed=0,
        )
        assert len(result) == 3

    def test_lower_lt_upper(self, daily_returns):
        point, lo, hi = bootstrap_metric(
            lambda r: sharpe_ratio(r, periods_per_year=252),
            daily_returns,
            n_samples=200,
            seed=42,
        )
        assert lo < hi

    def test_point_estimate_matches_scalar(self, daily_returns):
        sr_direct = sharpe_ratio(daily_returns, periods_per_year=252)
        point, _, _ = bootstrap_metric(
            lambda r: sharpe_ratio(r, periods_per_year=252),
            daily_returns,
            n_samples=100,
            seed=0,
        )
        assert math.isclose(point, sr_direct, rel_tol=1e-9)
