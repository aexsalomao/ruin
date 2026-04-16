"""Tests for ruin.volatility module."""

from __future__ import annotations

import math

import polars as pl
import pytest

from ruin.volatility import annualize_volatility, downside_deviation, semi_deviation, volatility


class TestVolatility:
    def test_known_value(self):
        r = pl.Series([0.01, -0.01, 0.02, -0.02])
        v = volatility(r)
        assert math.isclose(v, pl.Series([0.01, -0.01, 0.02, -0.02]).std(ddof=1), rel_tol=1e-9)

    def test_ddof0(self):
        r = pl.Series([1.0, 2.0, 3.0, 4.0])
        v0 = volatility(r, ddof=0)
        v1 = volatility(r, ddof=1)
        assert v0 < v1

    def test_single_element_raises(self):
        with pytest.raises(ValueError):
            volatility(pl.Series([0.01]))


class TestAnnualizeVolatility:
    def test_sqrt_of_time(self):
        r = pl.Series([0.01, -0.01, 0.02, -0.02])
        daily_vol = volatility(r)
        ann_vol = annualize_volatility(r, periods_per_year=252)
        assert math.isclose(ann_vol, daily_vol * math.sqrt(252), rel_tol=1e-9)

    def test_invalid_periods(self):
        with pytest.raises(ValueError):
            annualize_volatility(pl.Series([0.01, 0.02]), periods_per_year=0)


class TestDownsideDeviation:
    def test_no_downside(self):
        r = pl.Series([0.01, 0.02, 0.03])
        dd = downside_deviation(r, threshold=0.0)
        assert dd == 0.0

    def test_all_downside(self):
        r = pl.Series([-0.01, -0.02, -0.03])
        dd = downside_deviation(r, threshold=0.0)
        assert dd > 0.0

    def test_symmetric_around_zero(self):
        # With equal gains and losses, downside dev < total vol
        r = pl.Series([0.02, -0.02])
        dd = downside_deviation(r)
        vol = volatility(r)
        assert dd <= vol


class TestSemiDeviation:
    def test_all_positive_returns_zero(self):
        r = pl.Series([0.01, 0.02, 0.03])
        assert semi_deviation(r) == 0.0

    def test_negative_only(self):
        r = pl.Series([-0.01, -0.02])
        sd = semi_deviation(r)
        assert math.isclose(sd, pl.Series([-0.01, -0.02]).std(ddof=0), rel_tol=1e-9)
