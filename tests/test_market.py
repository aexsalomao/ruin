"""Tests for ruin.market module."""

from __future__ import annotations

import math

import numpy as np
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
    def test_beta_vs_self_is_one(self, daily_returns: pl.Series) -> None:
        assert math.isclose(beta(daily_returns, daily_returns), 1.0, rel_tol=1e-6)

    def test_beta_zero_variance_benchmark_is_nan(self) -> None:
        r = pl.Series([0.01, 0.02, 0.03])
        b_flat = pl.Series([0.0, 0.0, 0.0])
        assert math.isnan(beta(r, b_flat))

    def test_correlated_beta(
        self, daily_returns: pl.Series, benchmark_returns: pl.Series
    ) -> None:
        b = beta(daily_returns, benchmark_returns)
        assert isinstance(b, float)
        assert not math.isnan(b)

    def test_beta_minus_self_is_minus_one(self, daily_returns: pl.Series) -> None:
        assert math.isclose(beta(daily_returns, -daily_returns), -1.0, rel_tol=1e-6)

    def test_length_mismatch_raises(self) -> None:
        with pytest.raises(ValueError):
            beta(pl.Series([0.01, 0.02]), pl.Series([0.01, 0.02, 0.03]))


class TestDirectionalBetas:
    def test_downside_beta_is_float(
        self, daily_returns: pl.Series, benchmark_returns: pl.Series
    ) -> None:
        db = downside_beta(daily_returns, benchmark_returns)
        assert isinstance(db, float)

    def test_upside_beta_is_float(
        self, daily_returns: pl.Series, benchmark_returns: pl.Series
    ) -> None:
        ub = upside_beta(daily_returns, benchmark_returns)
        assert isinstance(ub, float)

    def test_downside_beta_no_negative_periods_is_nan(self) -> None:
        r = pl.Series([0.01, 0.02, 0.03])
        b = pl.Series([0.01, 0.02, 0.03])
        assert math.isnan(downside_beta(r, b))

    def test_upside_beta_no_positive_periods_is_nan(self) -> None:
        r = pl.Series([-0.01, -0.02, -0.03])
        b = pl.Series([-0.01, -0.02, -0.03])
        assert math.isnan(upside_beta(r, b))


class TestAlpha:
    def test_alpha_vs_self_is_zero(self, daily_returns: pl.Series) -> None:
        a = alpha(daily_returns, daily_returns, periods_per_year=252)
        assert math.isclose(a, 0.0, abs_tol=1e-10)

    def test_alpha_type(
        self, daily_returns: pl.Series, benchmark_returns: pl.Series
    ) -> None:
        a = alpha(daily_returns, benchmark_returns, periods_per_year=252)
        assert isinstance(a, float)


class TestCorrelation:
    def test_self_correlation_is_one(self, daily_returns: pl.Series) -> None:
        assert math.isclose(
            correlation(daily_returns, daily_returns), 1.0, rel_tol=1e-6
        )

    def test_anticorrelation_is_minus_one(self, daily_returns: pl.Series) -> None:
        assert math.isclose(
            correlation(daily_returns, -daily_returns), -1.0, rel_tol=1e-6
        )

    def test_range(
        self, daily_returns: pl.Series, benchmark_returns: pl.Series
    ) -> None:
        c = correlation(daily_returns, benchmark_returns)
        assert -1.0 <= c <= 1.0


class TestTrackingError:
    def test_self_te_is_zero(self, daily_returns: pl.Series) -> None:
        te = tracking_error(daily_returns, daily_returns, periods_per_year=252)
        assert math.isclose(te, 0.0, abs_tol=1e-10)

    def test_te_is_non_negative(
        self, daily_returns: pl.Series, benchmark_returns: pl.Series
    ) -> None:
        te = tracking_error(daily_returns, benchmark_returns, periods_per_year=252)
        assert te >= 0.0


class TestCapture:
    def test_up_capture_self_is_one(self, daily_returns: pl.Series) -> None:
        assert math.isclose(up_capture(daily_returns, daily_returns), 1.0, rel_tol=1e-6)

    def test_down_capture_self_is_one(self, daily_returns: pl.Series) -> None:
        assert math.isclose(
            down_capture(daily_returns, daily_returns), 1.0, rel_tol=1e-6
        )

    def test_up_capture_no_up_periods_is_nan(self) -> None:
        r = pl.Series([-0.01, -0.02])
        b = pl.Series([-0.01, -0.02])
        assert math.isnan(up_capture(r, b))

    def test_down_capture_no_down_periods_is_nan(self) -> None:
        r = pl.Series([0.01, 0.02])
        b = pl.Series([0.01, 0.02])
        assert math.isnan(down_capture(r, b))
