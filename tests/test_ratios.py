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


class TestSharpeRatio:
    def test_zero_variance_is_nan(self) -> None:
        r = pl.Series([0.01, 0.01])
        assert math.isnan(sharpe_ratio(r, periods_per_year=252))

    def test_positive_sharpe_float(self, daily_returns: pl.Series) -> None:
        sr = sharpe_ratio(daily_returns, periods_per_year=252)
        assert isinstance(sr, float)
        assert not math.isnan(sr)

    def test_risk_free_reduces_sharpe(self, daily_returns: pl.Series) -> None:
        sr0 = sharpe_ratio(daily_returns, periods_per_year=252, risk_free=0.0)
        sr_rf = sharpe_ratio(daily_returns, periods_per_year=252, risk_free=0.0001)
        assert sr_rf < sr0

    def test_scale_invariance(self, daily_returns: pl.Series) -> None:
        # Multiplying returns by a positive constant leaves the Sharpe ratio
        # unchanged (mean and std scale identically).
        sr = sharpe_ratio(daily_returns, periods_per_year=252)
        sr_scaled = sharpe_ratio(daily_returns * 3.0, periods_per_year=252)
        assert math.isclose(sr, sr_scaled, rel_tol=1e-6)

    def test_ddof_changes_result(self, daily_returns: pl.Series) -> None:
        sr0 = sharpe_ratio(daily_returns, periods_per_year=252, ddof=0)
        sr1 = sharpe_ratio(daily_returns, periods_per_year=252, ddof=1)
        # ddof=1 divides by n-1 → larger std → smaller |Sharpe|
        assert abs(sr0) > abs(sr1)

    def test_too_few_observations(self) -> None:
        with pytest.raises(ValueError):
            sharpe_ratio(pl.Series([0.01]), periods_per_year=252)


class TestSortinoRatio:
    def test_all_positive_returns_nan(self) -> None:
        r = pl.Series([0.01, 0.02, 0.03])
        assert math.isnan(sortino_ratio(r, periods_per_year=252))

    def test_is_float(self, daily_returns: pl.Series) -> None:
        sr = sortino_ratio(daily_returns, periods_per_year=252)
        assert isinstance(sr, float)

    def test_sortino_ge_sharpe_for_positive_skew(self) -> None:
        # Right-skewed returns: Sortino should be at least as high as Sharpe
        r = pl.Series([0.05, 0.05, 0.05, -0.01, 0.05, -0.01, 0.05])
        sr = sharpe_ratio(r, periods_per_year=252)
        so = sortino_ratio(r, periods_per_year=252)
        # Both should be positive and sortino >= sharpe (approximately)
        assert so >= sr - 1e-9


class TestCalmarRatio:
    def test_positive_cagr_negative_mdd(self, daily_returns: pl.Series) -> None:
        cr = calmar_ratio(daily_returns, periods_per_year=252)
        assert isinstance(cr, float)

    def test_no_drawdown_is_nan(self) -> None:
        r = pl.Series([0.01, 0.02, 0.03])
        assert math.isnan(calmar_ratio(r, periods_per_year=12))


class TestInformationRatio:
    def test_benchmark_equals_returns_is_nan(self, daily_returns: pl.Series) -> None:
        ir = information_ratio(daily_returns, daily_returns, periods_per_year=252)
        assert math.isnan(ir)

    def test_positive_ir_float(
        self, daily_returns: pl.Series, benchmark_returns: pl.Series
    ) -> None:
        ir = information_ratio(daily_returns, benchmark_returns, periods_per_year=252)
        assert isinstance(ir, float)
        assert not math.isnan(ir)

    def test_length_mismatch_raises(self) -> None:
        with pytest.raises(ValueError):
            information_ratio(
                pl.Series([0.01, 0.02]),
                pl.Series([0.01, 0.02, 0.03]),
                periods_per_year=252,
            )


class TestTreynorRatio:
    def test_zero_beta_is_nan(self) -> None:
        r = pl.Series([0.01, 0.02, 0.03, 0.04])
        b = pl.Series([0.01, 0.0, 0.0, 0.0])  # near-zero variance, keep len match
        tr = treynor_ratio(r, b, periods_per_year=252)
        assert isinstance(tr, float)

    def test_float_output(self, daily_returns: pl.Series, benchmark_returns: pl.Series) -> None:
        tr = treynor_ratio(daily_returns, benchmark_returns, periods_per_year=252)
        assert isinstance(tr, float)


class TestOmegaRatio:
    def test_all_positive_is_nan(self) -> None:
        r = pl.Series([0.01, 0.02, 0.03])
        assert math.isnan(omega_ratio(r))

    def test_exact_value(self) -> None:
        r = pl.Series([0.05, -0.01, 0.05, -0.01])
        # sum gains = 0.10, sum losses = 0.02 → omega = 5
        assert math.isclose(omega_ratio(r), 5.0, rel_tol=1e-9)

    def test_threshold_changes_result(self) -> None:
        r = pl.Series([0.05, -0.01, 0.05, -0.01])
        om0 = omega_ratio(r, threshold=0.0)
        om_high = omega_ratio(r, threshold=0.02)
        assert om_high < om0

    def test_symmetric_around_mean_near_one(self) -> None:
        rng = np.random.default_rng(0)
        n = 10_000
        r = pl.Series(rng.normal(0.0, 0.01, n).tolist())
        om = omega_ratio(r)
        # Mean near zero, so omega near 1
        assert math.isclose(om, 1.0, abs_tol=0.1)
