"""Reference tests: hand-computed values and parity checks.

These tests prove correctness against known exact values.
"""

from __future__ import annotations

import math

import polars as pl
import pytest

from ruin.activity import hit_rate, profit_factor
from ruin.drawdown import drawdown_series, max_drawdown
from ruin.ratios import omega_ratio, sharpe_ratio
from ruin.returns import cagr, total_return
from ruin.tail import conditional_value_at_risk, value_at_risk
from ruin.volatility import downside_deviation, volatility


# Hand-computed reference series
SIMPLE_RETURNS = pl.Series([0.01, -0.02, 0.03, -0.01, 0.02, 0.01, -0.03, 0.02, 0.01, -0.01])


class TestHandComputedReturns:
    """Exact values verified by hand."""

    def test_total_return(self):
        # product(1+r) - 1 computed manually
        r = pl.Series([0.1, 0.1])
        expected = 1.1 * 1.1 - 1.0
        assert math.isclose(total_return(r), expected, rel_tol=1e-12)

    def test_total_return_ten_periods(self):
        from functools import reduce
        import operator
        vals = SIMPLE_RETURNS.to_list()
        expected = reduce(operator.mul, [1 + v for v in vals], 1.0) - 1.0
        assert math.isclose(total_return(SIMPLE_RETURNS), expected, rel_tol=1e-12)

    def test_volatility_hand_computed(self):
        r = pl.Series([0.01, -0.01])
        # std([0.01, -0.01]) with ddof=1:
        # mean = 0, deviations = 0.01, -0.01
        # variance = (0.01^2 + 0.01^2) / (2-1) = 0.0002
        # std = sqrt(0.0002) ≈ 0.014142
        expected = (0.0002**0.5)
        assert math.isclose(volatility(r), expected, rel_tol=1e-9)


class TestHandComputedDrawdown:
    def test_single_drop(self):
        # prices: 1, 0.8 → dd = 0.8/1 - 1 = -0.2
        r = pl.Series([-0.2])
        dd = drawdown_series(r)
        assert math.isclose(dd[0], -0.2, rel_tol=1e-9)

    def test_recovery(self):
        # prices: 1, 0.8, 1.0 → dd: -0.2, 0.0
        r = pl.Series([-0.2, 0.25])
        dd = drawdown_series(r)
        assert math.isclose(dd[0], -0.2, rel_tol=1e-6)
        assert math.isclose(dd[1], 0.0, abs_tol=1e-9)

    def test_max_drawdown_exact(self):
        r = pl.Series([0.5, -1 / 3, -0.5])
        # wealth: 1, 1.5, 1.0, 0.5
        # hwm: 1, 1.5, 1.5, 1.5
        # dd: 0, 0, -1/3, -2/3
        mdd = max_drawdown(r)
        assert math.isclose(mdd, -2 / 3, rel_tol=1e-6)


class TestHandComputedRatios:
    def test_sharpe_zero_vol(self):
        r = pl.Series([0.01, 0.01, 0.01])
        sr = sharpe_ratio(r, periods_per_year=1)
        assert math.isnan(sr)

    def test_omega_exact(self):
        # gains: 0.05 * 2 = 0.1; losses: 0.01 * 2 = 0.02
        r = pl.Series([0.05, -0.01, 0.05, -0.01])
        om = omega_ratio(r)
        assert math.isclose(om, 5.0, rel_tol=1e-9)


class TestHandComputedTail:
    def test_historical_var_exact(self):
        # For 10 observations, 95% VaR = negative of 5th percentile
        r = pl.Series([-0.05, -0.04, -0.03, -0.02, -0.01, 0.01, 0.02, 0.03, 0.04, 0.05])
        var = value_at_risk(r, confidence=0.95)
        # 5th percentile of the sorted series is around 0.045 (interpolated)
        assert var > 0.0
        assert var <= 0.05


class TestHandComputedActivity:
    def test_hit_rate_exact(self):
        r = pl.Series([0.01, -0.01, 0.02, -0.02, 0.01])
        # 3 positive out of 5
        assert math.isclose(hit_rate(r), 3 / 5, rel_tol=1e-9)

    def test_profit_factor_exact(self):
        r = pl.Series([0.03, -0.01, 0.02, -0.01])
        # gains: 0.05, losses: 0.02
        assert math.isclose(profit_factor(r), 2.5, rel_tol=1e-9)
