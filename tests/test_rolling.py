"""Tests for ruin.rolling module."""

from __future__ import annotations

import math

import polars as pl
import pytest

import ruin


class TestRollingVolatility:
    def test_output_length(self, daily_returns):
        rv = ruin.rolling_volatility(daily_returns, window=20)
        assert len(rv) == len(daily_returns)

    def test_leading_nulls(self, daily_returns):
        rv = ruin.rolling_volatility(daily_returns, window=20)
        assert rv[:19].is_null().all()
        assert rv[19] is not None

    def test_full_window_matches_scalar(self, daily_returns):
        window = len(daily_returns)
        rv = ruin.rolling_volatility(daily_returns, window=window)
        scalar_vol = ruin.volatility(daily_returns)
        assert math.isclose(rv[-1], scalar_vol, rel_tol=1e-6)


class TestRollingSharpe:
    def test_output_aligned(self, daily_returns):
        rs = ruin.rolling_sharpe(daily_returns, window=60, periods_per_year=252)
        assert len(rs) == len(daily_returns)
        assert rs[:59].is_null().all()


class TestRollingBeta:
    def test_self_beta_is_one(self, daily_returns):
        rb = ruin.rolling_beta(daily_returns, daily_returns, window=50)
        non_null = rb.drop_nulls()
        assert all(math.isclose(v, 1.0, rel_tol=1e-4) for v in non_null.to_list())


class TestRollingMaxDrawdown:
    def test_non_positive(self, daily_returns):
        rmdd = ruin.rolling_max_drawdown(daily_returns, window=60)
        non_null = rmdd.drop_nulls()
        assert (non_null <= 1e-12).all()

    def test_output_length(self, daily_returns):
        rmdd = ruin.rolling_max_drawdown(daily_returns, window=60)
        assert len(rmdd) == len(daily_returns)


class TestRollingHitRate:
    def test_range(self, daily_returns):
        rhr = ruin.rolling_hit_rate(daily_returns, window=20)
        non_null = rhr.drop_nulls()
        assert (non_null >= 0.0).all() and (non_null <= 1.0).all()
