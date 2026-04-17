"""Tests for ruin.returns module."""

from __future__ import annotations

import math

import numpy as np
import polars as pl
import pytest

from ruin.returns import annualize_return, cagr, from_prices, total_return
from tests.conftest import FLOAT32_ABS_TOL, FLOAT32_REL_TOL


def test_from_prices_computes_simple_returns() -> None:
    prices = pl.Series([100.0, 110.0, 99.0, 108.9])
    r = from_prices(prices)
    assert len(r) == 3
    assert r[0] == pytest.approx(0.1, rel=FLOAT32_REL_TOL)
    assert r[1] == pytest.approx(-0.1, rel=FLOAT32_REL_TOL)


def test_from_prices_output_is_float32() -> None:
    prices = pl.Series([100.0, 110.0])
    r = from_prices(prices)
    assert r.dtype == pl.Float32


def test_from_prices_computes_log_returns() -> None:
    prices = pl.Series([100.0, 110.0])
    r = from_prices(prices, log=True)
    assert len(r) == 1
    assert r[0] == pytest.approx(math.log(1.1), rel=FLOAT32_REL_TOL)


def test_from_prices_log_matches_simple_for_small_moves() -> None:
    prices = pl.Series([100.0, 100.01, 100.02])
    simple = from_prices(prices)
    log = from_prices(prices, log=True)
    for s, lg in zip(simple.to_list(), log.to_list(), strict=True):
        assert s == pytest.approx(lg, rel=1e-3)


def test_from_prices_accepts_numpy_input() -> None:
    prices = np.array([100.0, 105.0, 110.25])
    r = from_prices(prices)
    assert len(r) == 2
    assert r[0] == pytest.approx(0.05, rel=FLOAT32_REL_TOL)
    assert r[1] == pytest.approx(0.05, rel=FLOAT32_REL_TOL)


def test_from_prices_rejects_single_price() -> None:
    with pytest.raises(ValueError, match="at least 2"):
        from_prices(pl.Series([100.0]))


def test_from_prices_rejects_zero_price() -> None:
    with pytest.raises(ValueError, match="strictly positive"):
        from_prices(pl.Series([100.0, 0.0, 50.0]))


def test_from_prices_rejects_negative_price() -> None:
    with pytest.raises(ValueError, match="strictly positive"):
        from_prices(pl.Series([100.0, -5.0, 50.0]))


def test_from_prices_returns_zeros_for_flat_prices() -> None:
    r = from_prices(pl.Series([100.0, 100.0, 100.0]))
    assert all(abs(v) < FLOAT32_ABS_TOL for v in r.to_list())


def test_total_return_compounds_positive_returns() -> None:
    r = pl.Series([0.1, 0.1, 0.1])
    assert total_return(r) == pytest.approx(1.1**3 - 1, rel=1e-9)


def test_total_return_is_zero_for_flat_series() -> None:
    r = pl.Series([0.0, 0.0, 0.0])
    assert total_return(r) == 0.0


def test_total_return_is_zero_when_drawdown_fully_recovers() -> None:
    r = pl.Series([0.5, -1 / 3])
    assert total_return(r) == pytest.approx(0.0, abs=1e-9)


def test_total_return_drops_nan_values() -> None:
    r = pl.Series([0.1, float("nan"), 0.1])
    assert total_return(r) == pytest.approx(1.1**2 - 1, rel=1e-9)


def test_total_return_rejects_empty_series() -> None:
    with pytest.raises(ValueError):
        total_return(pl.Series([], dtype=pl.Float64))


def test_total_return_is_minus_one_for_wipeout() -> None:
    r = pl.Series([-1.0])
    assert total_return(r) == -1.0


def test_total_return_accepts_numpy_input() -> None:
    r = np.array([0.1, 0.1])
    assert total_return(r) == pytest.approx(0.21, rel=1e-9)


def test_annualize_return_geometric_compounds_monthly_returns() -> None:
    r = pl.Series([0.01] * 12)
    ann = annualize_return(r, periods_per_year=12)
    expected = (1.01**12) - 1
    assert ann == pytest.approx(expected, rel=1e-9)


def test_annualize_return_arithmetic_multiplies_mean_by_periods() -> None:
    r = pl.Series([0.01] * 12)
    ann = annualize_return(r, periods_per_year=12, method="arithmetic")
    assert ann == pytest.approx(0.12, rel=1e-9)


def test_annualize_return_is_nan_after_ruin() -> None:
    r = pl.Series([-1.0, 0.01])
    assert math.isnan(annualize_return(r, periods_per_year=12))


def test_annualize_return_rejects_unknown_method() -> None:
    with pytest.raises(ValueError, match="Unknown method"):
        annualize_return(pl.Series([0.01]), periods_per_year=12, method="bad")


def test_annualize_return_rejects_negative_periods() -> None:
    with pytest.raises(ValueError):
        annualize_return(pl.Series([0.01]), periods_per_year=-1)


def test_cagr_matches_geometric_annualize_return() -> None:
    r = pl.Series([0.01, 0.02, -0.01, 0.03])
    assert cagr(r, periods_per_year=252) == annualize_return(
        r, periods_per_year=252, method="geometric"
    )


def test_cagr_on_constant_daily_returns_matches_formula() -> None:
    r = pl.Series([0.01] * 252)
    expected = 1.01**252 - 1
    assert cagr(r, periods_per_year=252) == pytest.approx(expected, rel=1e-9)
