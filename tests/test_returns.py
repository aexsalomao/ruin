"""Tests for ruin.returns module."""

from __future__ import annotations

import math

import numpy as np
import polars as pl
import pytest

from ruin.returns import annualize_return, cagr, from_prices, total_return


class TestFromPrices:
    def test_simple_returns(self):
        prices = pl.Series([100.0, 110.0, 99.0, 108.9])
        r = from_prices(prices)
        assert len(r) == 3
        assert math.isclose(r[0], 0.1, rel_tol=1e-9)
        assert math.isclose(r[1], -0.1, rel_tol=1e-6)

    def test_log_returns(self):
        prices = pl.Series([100.0, 110.0])
        r = from_prices(prices, log=True)
        assert len(r) == 1
        assert math.isclose(r[0], math.log(1.1), rel_tol=1e-9)

    def test_numpy_input(self):
        prices = np.array([100.0, 105.0, 110.25])
        r = from_prices(prices)
        assert len(r) == 2
        assert math.isclose(r[0], 0.05, rel_tol=1e-9)
        assert math.isclose(r[1], 0.05, rel_tol=1e-9)

    def test_requires_at_least_2(self):
        with pytest.raises(ValueError):
            from_prices(pl.Series([100.0]))


class TestTotalReturn:
    def test_positive_series(self):
        r = pl.Series([0.1, 0.1, 0.1])
        # (1.1)^3 - 1 = 0.331
        assert math.isclose(total_return(r), 1.1**3 - 1, rel_tol=1e-9)

    def test_flat(self):
        r = pl.Series([0.0, 0.0, 0.0])
        assert total_return(r) == 0.0

    def test_drawdown_and_recovery(self):
        r = pl.Series([0.5, -1 / 3])
        # (1.5)(2/3) - 1 = 0
        assert math.isclose(total_return(r), 0.0, abs_tol=1e-9)

    def test_nan_dropped(self):
        r = pl.Series([0.1, float("nan"), 0.1])
        assert math.isclose(total_return(r), 1.1**2 - 1, rel_tol=1e-9)


class TestAnnualizeReturn:
    def test_geometric(self):
        r = pl.Series([0.01] * 12)
        ann = annualize_return(r, periods_per_year=12)
        expected = (1.01**12) - 1
        assert math.isclose(ann, expected, rel_tol=1e-9)

    def test_arithmetic(self):
        r = pl.Series([0.01] * 12)
        ann = annualize_return(r, periods_per_year=12, method="arithmetic")
        assert math.isclose(ann, 0.12, rel_tol=1e-9)

    def test_invalid_method(self):
        with pytest.raises(ValueError, match="Unknown method"):
            annualize_return(pl.Series([0.01]), periods_per_year=12, method="bad")

    def test_invalid_periods(self):
        with pytest.raises(ValueError):
            annualize_return(pl.Series([0.01]), periods_per_year=-1)


class TestCagr:
    def test_is_geometric_alias(self):
        r = pl.Series([0.01, 0.02, -0.01, 0.03])
        assert cagr(r, periods_per_year=252) == annualize_return(
            r, periods_per_year=252, method="geometric"
        )
