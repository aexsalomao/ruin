"""Tests for ruin.market module."""

from __future__ import annotations

import math

import polars as pl
import pytest

from ruin.market import (
    alpha,
    beta,
    correlation,
    down_capture,
    downside_beta,
    tracking_error,
    up_capture,
    upside_beta,
)


class TestBeta:
    def test_beta_vs_self_is_one(self, daily_returns):
        b = beta(daily_returns, daily_returns)
        assert math.isclose(b, 1.0, rel_tol=1e-6)

    def test_beta_zero_benchmark(self):
        r = pl.Series([0.01, 0.02, 0.03])
        b_flat = pl.Series([0.0, 0.0, 0.0])
        assert math.isnan(beta(r, b_flat))

    def test_correlated_beta(self, daily_returns, benchmark_returns):
        b = beta(daily_returns, benchmark_returns)
        assert isinstance(b, float)
        assert not math.isnan(b)


class TestAlpha:
    def test_alpha_vs_self_is_zero(self, daily_returns):
        a = alpha(daily_returns, daily_returns, periods_per_year=252)
        assert math.isclose(a, 0.0, abs_tol=1e-10)

    def test_alpha_type(self, daily_returns, benchmark_returns):
        a = alpha(daily_returns, benchmark_returns, periods_per_year=252)
        assert isinstance(a, float)


class TestCorrelation:
    def test_self_correlation_is_one(self, daily_returns):
        c = correlation(daily_returns, daily_returns)
        assert math.isclose(c, 1.0, rel_tol=1e-6)

    def test_range(self, daily_returns, benchmark_returns):
        c = correlation(daily_returns, benchmark_returns)
        assert -1.0 <= c <= 1.0


class TestTrackingError:
    def test_self_te_is_zero(self, daily_returns):
        te = tracking_error(daily_returns, daily_returns, periods_per_year=252)
        assert math.isclose(te, 0.0, abs_tol=1e-10)


class TestCapture:
    def test_up_capture_self_is_one(self, daily_returns):
        uc = up_capture(daily_returns, daily_returns)
        assert math.isclose(uc, 1.0, rel_tol=1e-6)

    def test_down_capture_self_is_one(self, daily_returns):
        dc = down_capture(daily_returns, daily_returns)
        assert math.isclose(dc, 1.0, rel_tol=1e-6)
