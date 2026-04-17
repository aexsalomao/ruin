"""Reference tests: hand-computed values lock in exact results for regression detection."""

from __future__ import annotations

import math
import operator
from functools import reduce

import polars as pl
import pytest

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


def test_total_return_matches_product_formula_for_two_periods() -> None:
    r = pl.Series([0.1, 0.1])
    assert total_return(r) == pytest.approx(1.1 * 1.1 - 1.0, rel=1e-12)


def test_total_return_matches_product_formula_for_ten_periods() -> None:
    expected = reduce(operator.mul, [1 + v for v in SIMPLE_RETURNS.to_list()], 1.0) - 1.0
    assert total_return(SIMPLE_RETURNS) == pytest.approx(expected, rel=1e-12)


def test_volatility_of_two_element_series_matches_hand_computed_value() -> None:
    r = pl.Series([0.01, -0.01])
    expected = math.sqrt(0.0002)
    assert volatility(r) == pytest.approx(expected, rel=1e-9)


def test_drawdown_series_for_single_drop_equals_loss() -> None:
    r = pl.Series([-0.2])
    dd = drawdown_series(r)
    assert dd[0] == pytest.approx(-0.2, rel=FLOAT32_REL_TOL)


def test_drawdown_series_returns_to_zero_on_recovery() -> None:
    r = pl.Series([-0.2, 0.25])
    dd = drawdown_series(r)
    assert dd[0] == pytest.approx(-0.2, rel=FLOAT32_REL_TOL)
    assert dd[1] == pytest.approx(0.0, abs=FLOAT32_ABS_TOL)


def test_max_drawdown_matches_hand_computed_two_thirds_loss() -> None:
    r = pl.Series([0.5, -1 / 3, -0.5])
    assert max_drawdown(r) == pytest.approx(-2 / 3, rel=FLOAT32_REL_TOL)


def test_sharpe_ratio_is_nan_for_constant_series() -> None:
    assert math.isnan(sharpe_ratio(pl.Series([0.01, 0.01, 0.01]), periods_per_year=1))


def test_omega_ratio_matches_hand_computed_five() -> None:
    r = pl.Series([0.05, -0.01, 0.05, -0.01])
    assert omega_ratio(r) == pytest.approx(5.0, rel=1e-9)


def test_historical_var_is_positive_and_below_worst_loss() -> None:
    r = pl.Series([-0.05, -0.04, -0.03, -0.02, -0.01, 0.01, 0.02, 0.03, 0.04, 0.05])
    var = value_at_risk(r, confidence=0.95)
    assert var > 0.0
    assert var <= 0.05


def test_hit_rate_matches_three_fifths() -> None:
    r = pl.Series([0.01, -0.01, 0.02, -0.02, 0.01])
    assert hit_rate(r) == pytest.approx(3 / 5, rel=1e-9)


def test_profit_factor_matches_hand_computed_two_point_five() -> None:
    r = pl.Series([0.03, -0.01, 0.02, -0.01])
    assert profit_factor(r) == pytest.approx(2.5, rel=1e-9)
