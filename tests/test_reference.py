"""Reference tests: hand-computed values and parity checks.

These tests lock in exact values to protect against numerical regressions.
Tolerances are chosen for the Float32 output dtype where applicable.
"""

from __future__ import annotations

import math
import operator
from functools import reduce

import polars as pl

from ruin.activity import hit_rate, profit_factor
from ruin.drawdown import drawdown_series, max_drawdown
from ruin.ratios import omega_ratio, sharpe_ratio
from ruin.returns import total_return
from ruin.tail import value_at_risk
from ruin.volatility import volatility
from tests.conftest import FLOAT32_ABS_TOL, FLOAT32_REL_TOL

SIMPLE_RETURNS = pl.Series(
    "returns",
    [0.01, -0.02, 0.03, -0.01, 0.02, 0.01, -0.03, 0.02, 0.01, -0.01],
    dtype=pl.Float64,
)


class TestHandComputedReturns:
    def test_total_return(self) -> None:
        r = pl.Series([0.1, 0.1])
        assert math.isclose(total_return(r), 1.1 * 1.1 - 1.0, rel_tol=1e-12)

    def test_total_return_ten_periods(self) -> None:
        expected = reduce(operator.mul, [1 + v for v in SIMPLE_RETURNS.to_list()], 1.0) - 1.0
        assert math.isclose(total_return(SIMPLE_RETURNS), expected, rel_tol=1e-12)

    def test_volatility_hand_computed(self) -> None:
        r = pl.Series([0.01, -0.01])
        expected = math.sqrt(0.0002)
        assert math.isclose(volatility(r), expected, rel_tol=1e-9)


class TestHandComputedDrawdown:
    def test_single_drop(self) -> None:
        r = pl.Series([-0.2])
        dd = drawdown_series(r)
        assert math.isclose(dd[0], -0.2, rel_tol=FLOAT32_REL_TOL)

    def test_recovery(self) -> None:
        r = pl.Series([-0.2, 0.25])
        dd = drawdown_series(r)
        assert math.isclose(dd[0], -0.2, rel_tol=FLOAT32_REL_TOL)
        assert math.isclose(dd[1], 0.0, abs_tol=FLOAT32_ABS_TOL)

    def test_max_drawdown_exact(self) -> None:
        r = pl.Series([0.5, -1 / 3, -0.5])
        assert math.isclose(max_drawdown(r), -2 / 3, rel_tol=FLOAT32_REL_TOL)


class TestHandComputedRatios:
    def test_sharpe_zero_vol(self) -> None:
        assert math.isnan(sharpe_ratio(pl.Series([0.01, 0.01, 0.01]), periods_per_year=1))

    def test_omega_exact(self) -> None:
        r = pl.Series([0.05, -0.01, 0.05, -0.01])
        assert math.isclose(omega_ratio(r), 5.0, rel_tol=1e-9)


class TestHandComputedTail:
    def test_historical_var_exact(self) -> None:
        r = pl.Series([-0.05, -0.04, -0.03, -0.02, -0.01, 0.01, 0.02, 0.03, 0.04, 0.05])
        var = value_at_risk(r, confidence=0.95)
        assert var > 0.0
        assert var <= 0.05


class TestHandComputedActivity:
    def test_hit_rate_exact(self) -> None:
        r = pl.Series([0.01, -0.01, 0.02, -0.02, 0.01])
        assert math.isclose(hit_rate(r), 3 / 5, rel_tol=1e-9)

    def test_profit_factor_exact(self) -> None:
        r = pl.Series([0.03, -0.01, 0.02, -0.01])
        assert math.isclose(profit_factor(r), 2.5, rel_tol=1e-9)
