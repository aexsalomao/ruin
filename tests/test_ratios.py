"""Tests for ruin.ratios module."""

from __future__ import annotations

import math

import numpy as np
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


def test_sharpe_ratio_is_nan_for_zero_variance_series() -> None:
    r = pl.Series([0.01, 0.01])
    assert math.isnan(sharpe_ratio(r, periods_per_year=252))


def test_sharpe_ratio_returns_finite_float_for_normal_data(daily_returns: pl.Series) -> None:
    sr = sharpe_ratio(daily_returns, periods_per_year=252)
    assert isinstance(sr, float)
    assert not math.isnan(sr)


def test_sharpe_ratio_decreases_when_risk_free_increases(daily_returns: pl.Series) -> None:
    sr0 = sharpe_ratio(daily_returns, periods_per_year=252, risk_free=0.0)
    sr_rf = sharpe_ratio(daily_returns, periods_per_year=252, risk_free=0.0001)
    assert sr_rf < sr0


def test_sharpe_ratio_is_invariant_under_positive_scaling(daily_returns: pl.Series) -> None:
    sr = sharpe_ratio(daily_returns, periods_per_year=252)
    sr_scaled = sharpe_ratio(daily_returns * 3.0, periods_per_year=252)
    assert sr == pytest.approx(sr_scaled, rel=1e-6)


def test_sharpe_ratio_ddof_changes_result(daily_returns: pl.Series) -> None:
    sr0 = sharpe_ratio(daily_returns, periods_per_year=252, ddof=0)
    sr1 = sharpe_ratio(daily_returns, periods_per_year=252, ddof=1)
    # ddof=1 divides by n-1 → larger std → smaller |Sharpe|
    assert abs(sr0) > abs(sr1)


def test_sharpe_ratio_rejects_too_few_observations() -> None:
    with pytest.raises(ValueError):
        sharpe_ratio(pl.Series([0.01]), periods_per_year=252)


def test_sortino_ratio_is_nan_when_no_downside() -> None:
    r = pl.Series([0.01, 0.02, 0.03])
    assert math.isnan(sortino_ratio(r, periods_per_year=252))


def test_sortino_ratio_returns_float(daily_returns: pl.Series) -> None:
    sr = sortino_ratio(daily_returns, periods_per_year=252)
    assert isinstance(sr, float)


def test_sortino_ratio_is_at_least_sharpe_for_right_skewed_returns() -> None:
    r = pl.Series([0.05, 0.05, 0.05, -0.01, 0.05, -0.01, 0.05])
    sr = sharpe_ratio(r, periods_per_year=252)
    so = sortino_ratio(r, periods_per_year=252)
    assert so >= sr - 1e-9


def test_calmar_ratio_returns_float(daily_returns: pl.Series) -> None:
    cr = calmar_ratio(daily_returns, periods_per_year=252)
    assert isinstance(cr, float)


def test_calmar_ratio_is_nan_without_drawdown() -> None:
    r = pl.Series([0.01, 0.02, 0.03])
    assert math.isnan(calmar_ratio(r, periods_per_year=12))


def test_information_ratio_is_nan_when_benchmark_equals_returns(
    daily_returns: pl.Series,
) -> None:
    ir = information_ratio(daily_returns, daily_returns, periods_per_year=252)
    assert math.isnan(ir)


def test_information_ratio_is_finite_for_distinct_series(
    daily_returns: pl.Series, benchmark_returns: pl.Series
) -> None:
    ir = information_ratio(daily_returns, benchmark_returns, periods_per_year=252)
    assert isinstance(ir, float)
    assert not math.isnan(ir)


def test_information_ratio_rejects_length_mismatch() -> None:
    with pytest.raises(ValueError):
        information_ratio(
            pl.Series([0.01, 0.02]),
            pl.Series([0.01, 0.02, 0.03]),
            periods_per_year=252,
        )


def test_treynor_ratio_handles_near_zero_beta() -> None:
    r = pl.Series([0.01, 0.02, 0.03, 0.04])
    b = pl.Series([0.01, 0.0, 0.0, 0.0])
    tr = treynor_ratio(r, b, periods_per_year=252)
    assert isinstance(tr, float)


def test_treynor_ratio_returns_float(
    daily_returns: pl.Series, benchmark_returns: pl.Series
) -> None:
    tr = treynor_ratio(daily_returns, benchmark_returns, periods_per_year=252)
    assert isinstance(tr, float)


def test_omega_ratio_is_nan_when_all_positive() -> None:
    r = pl.Series([0.01, 0.02, 0.03])
    assert math.isnan(omega_ratio(r))


def test_omega_ratio_matches_hand_computed_value() -> None:
    r = pl.Series([0.05, -0.01, 0.05, -0.01])
    # sum gains = 0.10, sum losses = 0.02 → omega = 5
    assert omega_ratio(r) == pytest.approx(5.0, rel=1e-9)


def test_omega_ratio_decreases_with_higher_threshold() -> None:
    r = pl.Series([0.05, -0.01, 0.05, -0.01])
    om0 = omega_ratio(r, threshold=0.0)
    om_high = omega_ratio(r, threshold=0.02)
    assert om_high < om0


def test_omega_ratio_is_near_one_for_zero_mean_symmetric_returns() -> None:
    rng = np.random.default_rng(0)
    n = 10_000
    r = pl.Series(rng.normal(0.0, 0.01, n).tolist())
    om = omega_ratio(r)
    assert om == pytest.approx(1.0, abs=0.1)
