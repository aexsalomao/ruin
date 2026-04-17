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


def test_beta_of_series_against_itself_is_one(daily_returns: pl.Series) -> None:
    assert beta(daily_returns, daily_returns) == pytest.approx(1.0, rel=1e-6)


def test_beta_is_nan_when_benchmark_has_zero_variance() -> None:
    r = pl.Series([0.01, 0.02, 0.03])
    b_flat = pl.Series([0.0, 0.0, 0.0])
    assert math.isnan(beta(r, b_flat))


def test_beta_against_correlated_benchmark_is_finite(
    daily_returns: pl.Series, benchmark_returns: pl.Series
) -> None:
    b = beta(daily_returns, benchmark_returns)
    assert isinstance(b, float)
    assert not math.isnan(b)


def test_beta_of_series_against_its_negation_is_minus_one(daily_returns: pl.Series) -> None:
    assert beta(daily_returns, -daily_returns) == pytest.approx(-1.0, rel=1e-6)


def test_beta_rejects_length_mismatch() -> None:
    with pytest.raises(ValueError):
        beta(pl.Series([0.01, 0.02]), pl.Series([0.01, 0.02, 0.03]))


def test_downside_beta_returns_float(
    daily_returns: pl.Series, benchmark_returns: pl.Series
) -> None:
    db = downside_beta(daily_returns, benchmark_returns)
    assert isinstance(db, float)


def test_upside_beta_returns_float(daily_returns: pl.Series, benchmark_returns: pl.Series) -> None:
    ub = upside_beta(daily_returns, benchmark_returns)
    assert isinstance(ub, float)


def test_downside_beta_is_nan_when_no_negative_periods() -> None:
    r = pl.Series([0.01, 0.02, 0.03])
    b = pl.Series([0.01, 0.02, 0.03])
    assert math.isnan(downside_beta(r, b))


def test_upside_beta_is_nan_when_no_positive_periods() -> None:
    r = pl.Series([-0.01, -0.02, -0.03])
    b = pl.Series([-0.01, -0.02, -0.03])
    assert math.isnan(upside_beta(r, b))


def test_alpha_of_series_against_itself_is_zero(daily_returns: pl.Series) -> None:
    a = alpha(daily_returns, daily_returns, periods_per_year=252)
    assert a == pytest.approx(0.0, abs=1e-10)


def test_alpha_returns_float(daily_returns: pl.Series, benchmark_returns: pl.Series) -> None:
    a = alpha(daily_returns, benchmark_returns, periods_per_year=252)
    assert isinstance(a, float)


def test_self_correlation_is_one(daily_returns: pl.Series) -> None:
    assert correlation(daily_returns, daily_returns) == pytest.approx(1.0, rel=1e-6)


def test_correlation_with_negation_is_minus_one(daily_returns: pl.Series) -> None:
    assert correlation(daily_returns, -daily_returns) == pytest.approx(-1.0, rel=1e-6)


def test_correlation_is_bounded_in_minus_one_one(
    daily_returns: pl.Series, benchmark_returns: pl.Series
) -> None:
    c = correlation(daily_returns, benchmark_returns)
    assert -1.0 <= c <= 1.0


def test_tracking_error_vs_self_is_zero(daily_returns: pl.Series) -> None:
    te = tracking_error(daily_returns, daily_returns, periods_per_year=252)
    assert te == pytest.approx(0.0, abs=1e-10)


def test_tracking_error_is_non_negative(
    daily_returns: pl.Series, benchmark_returns: pl.Series
) -> None:
    te = tracking_error(daily_returns, benchmark_returns, periods_per_year=252)
    assert te >= 0.0


def test_up_capture_vs_self_is_one(daily_returns: pl.Series) -> None:
    assert up_capture(daily_returns, daily_returns) == pytest.approx(1.0, rel=1e-6)


def test_down_capture_vs_self_is_one(daily_returns: pl.Series) -> None:
    assert down_capture(daily_returns, daily_returns) == pytest.approx(1.0, rel=1e-6)


def test_up_capture_is_nan_without_up_periods() -> None:
    r = pl.Series([-0.01, -0.02])
    b = pl.Series([-0.01, -0.02])
    assert math.isnan(up_capture(r, b))


def test_down_capture_is_nan_without_down_periods() -> None:
    r = pl.Series([0.01, 0.02])
    b = pl.Series([0.01, 0.02])
    assert math.isnan(down_capture(r, b))
