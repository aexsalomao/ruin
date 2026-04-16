"""Tests for ruin.ratios module."""

from __future__ import annotations

import math

import polars as pl
import pytest

from ruin.ratios import (
    calmar_ratio,
    information_ratio,
    omega_ratio,
    sharpe_ratio,
    sortino_ratio,
    treynor_ratio,
)


class TestSharpeRatio:
    def test_zero_risk_free(self):
        # Two identical returns: std(ddof=1) is exactly 0 → NaN
        r = pl.Series([0.01, 0.01])
        sr = sharpe_ratio(r, periods_per_year=252)
        assert math.isnan(sr)

    def test_positive_sharpe(self, daily_returns):
        sr = sharpe_ratio(daily_returns, periods_per_year=252)
        assert isinstance(sr, float)

    def test_risk_free_reduces_sharpe(self, daily_returns):
        sr0 = sharpe_ratio(daily_returns, periods_per_year=252, risk_free=0.0)
        sr_rf = sharpe_ratio(daily_returns, periods_per_year=252, risk_free=0.0001)
        assert sr_rf < sr0


class TestSortinoRatio:
    def test_all_positive_returns(self):
        r = pl.Series([0.01, 0.02, 0.03])
        sr = sortino_ratio(r, periods_per_year=252)
        # No downside → NaN
        assert math.isnan(sr)

    def test_positive_sortino(self, daily_returns):
        sr = sortino_ratio(daily_returns, periods_per_year=252)
        assert isinstance(sr, float)


class TestCalmarRatio:
    def test_positive_cagr_negative_mdd(self, daily_returns):
        cr = calmar_ratio(daily_returns, periods_per_year=252)
        assert isinstance(cr, float)

    def test_no_drawdown_is_nan(self):
        r = pl.Series([0.01, 0.02, 0.03])
        assert math.isnan(calmar_ratio(r, periods_per_year=12))


class TestInformationRatio:
    def test_benchmark_equals_returns_is_nan(self, daily_returns):
        ir = information_ratio(daily_returns, daily_returns, periods_per_year=252)
        assert math.isnan(ir)

    def test_positive_ir(self, daily_returns, benchmark_returns):
        ir = information_ratio(daily_returns, benchmark_returns, periods_per_year=252)
        assert isinstance(ir, float)


class TestOmegaRatio:
    def test_all_positive(self):
        r = pl.Series([0.01, 0.02, 0.03])
        om = omega_ratio(r)
        assert math.isnan(om)  # no losses below threshold

    def test_above_one_when_positive_total(self):
        r = pl.Series([0.05, -0.01, 0.05, -0.01])
        # sum gains = 0.10, sum losses = 0.02 → omega = 5
        om = omega_ratio(r)
        assert math.isclose(om, 5.0, rel_tol=1e-9)
