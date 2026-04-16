"""Tests for ruin.returns module."""

from __future__ import annotations

import math

import numpy as np
import polars as pl
import pytest

from ruin.returns import annualize_return, cagr, from_prices, total_return
from tests.conftest import FLOAT32_ABS_TOL, FLOAT32_REL_TOL


class TestFromPrices:
    def test_simple_returns(self) -> None:
        prices = pl.Series([100.0, 110.0, 99.0, 108.9])
        r = from_prices(prices)
        assert len(r) == 3
        assert math.isclose(r[0], 0.1, rel_tol=FLOAT32_REL_TOL)
        assert math.isclose(r[1], -0.1, rel_tol=FLOAT32_REL_TOL)

    def test_output_is_float32(self) -> None:
        prices = pl.Series([100.0, 110.0])
        r = from_prices(prices)
        assert r.dtype == pl.Float32

    def test_log_returns(self) -> None:
        prices = pl.Series([100.0, 110.0])
        r = from_prices(prices, log=True)
        assert len(r) == 1
        assert math.isclose(r[0], math.log(1.1), rel_tol=FLOAT32_REL_TOL)

    def test_log_vs_simple_for_small_moves(self) -> None:
        # For small moves, log ≈ simple returns (first-order approximation)
        prices = pl.Series([100.0, 100.01, 100.02])
        simple = from_prices(prices)
        log = from_prices(prices, log=True)
        for s, lg in zip(simple.to_list(), log.to_list(), strict=True):
            assert math.isclose(s, lg, rel_tol=1e-3)

    def test_numpy_input(self) -> None:
        prices = np.array([100.0, 105.0, 110.25])
        r = from_prices(prices)
        assert len(r) == 2
        assert math.isclose(r[0], 0.05, rel_tol=FLOAT32_REL_TOL)
        assert math.isclose(r[1], 0.05, rel_tol=FLOAT32_REL_TOL)

    def test_requires_at_least_2(self) -> None:
        with pytest.raises(ValueError, match="at least 2"):
            from_prices(pl.Series([100.0]))

    def test_rejects_zero_price(self) -> None:
        with pytest.raises(ValueError, match="strictly positive"):
            from_prices(pl.Series([100.0, 0.0, 50.0]))

    def test_rejects_negative_price(self) -> None:
        with pytest.raises(ValueError, match="strictly positive"):
            from_prices(pl.Series([100.0, -5.0, 50.0]))

    def test_flat_prices_yield_zeros(self) -> None:
        r = from_prices(pl.Series([100.0, 100.0, 100.0]))
        assert all(abs(v) < FLOAT32_ABS_TOL for v in r.to_list())


class TestTotalReturn:
    def test_positive_series(self) -> None:
        r = pl.Series([0.1, 0.1, 0.1])
        assert math.isclose(total_return(r), 1.1**3 - 1, rel_tol=1e-9)

    def test_flat(self) -> None:
        r = pl.Series([0.0, 0.0, 0.0])
        assert total_return(r) == 0.0

    def test_drawdown_and_recovery(self) -> None:
        r = pl.Series([0.5, -1 / 3])
        assert math.isclose(total_return(r), 0.0, abs_tol=1e-9)

    def test_nan_dropped(self) -> None:
        r = pl.Series([0.1, float("nan"), 0.1])
        assert math.isclose(total_return(r), 1.1**2 - 1, rel_tol=1e-9)

    def test_empty_raises(self) -> None:
        with pytest.raises(ValueError):
            total_return(pl.Series([], dtype=pl.Float64))

    def test_ruin_returns_negative_one(self) -> None:
        # -100% wipes out the portfolio
        r = pl.Series([-1.0])
        assert total_return(r) == -1.0

    def test_numpy_input(self) -> None:
        r = np.array([0.1, 0.1])
        assert math.isclose(total_return(r), 0.21, rel_tol=1e-9)


class TestAnnualizeReturn:
    def test_geometric(self) -> None:
        r = pl.Series([0.01] * 12)
        ann = annualize_return(r, periods_per_year=12)
        expected = (1.01**12) - 1
        assert math.isclose(ann, expected, rel_tol=1e-9)

    def test_arithmetic(self) -> None:
        r = pl.Series([0.01] * 12)
        ann = annualize_return(r, periods_per_year=12, method="arithmetic")
        assert math.isclose(ann, 0.12, rel_tol=1e-9)

    def test_geometric_after_ruin_is_nan(self) -> None:
        # Total return of -1 (ruin) cannot be raised to a fractional power.
        r = pl.Series([-1.0, 0.01])
        assert math.isnan(annualize_return(r, periods_per_year=12))

    def test_invalid_method(self) -> None:
        with pytest.raises(ValueError, match="Unknown method"):
            annualize_return(pl.Series([0.01]), periods_per_year=12, method="bad")

    def test_invalid_periods(self) -> None:
        with pytest.raises(ValueError):
            annualize_return(pl.Series([0.01]), periods_per_year=-1)


class TestCagr:
    def test_is_geometric_alias(self) -> None:
        r = pl.Series([0.01, 0.02, -0.01, 0.03])
        assert cagr(r, periods_per_year=252) == annualize_return(
            r, periods_per_year=252, method="geometric"
        )

    def test_cagr_on_constant_returns(self) -> None:
        r = pl.Series([0.01] * 252)
        # CAGR on 252 identical 1% daily returns is (1.01)^252 - 1
        expected = 1.01**252 - 1
        assert math.isclose(cagr(r, periods_per_year=252), expected, rel_tol=1e-9)
