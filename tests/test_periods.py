"""Tests for ruin.periods module."""

from __future__ import annotations

import datetime
import math

import polars as pl
import pytest

from ruin.periods import (
    annual_to_periodic,
    mtd,
    periodic_to_annual,
    periods_per_year_for,
    qtd,
    since_inception,
    trailing,
    ytd,
)


class TestPeriodSlicing:
    def setup_method(self):
        dates = [datetime.date(2024, 1, 1) + datetime.timedelta(days=i) for i in range(365)]
        self.df = pl.DataFrame({
            "date": dates,
            "returns": [0.001] * 365,
        })

    def test_ytd_within_year(self):
        result = ytd(self.df, date_col="date", as_of=datetime.date(2024, 6, 30))
        assert len(result) == 182  # Jan 1 to Jun 30 in 2024 (leap year: Feb has 29 days)

    def test_mtd_within_month(self):
        result = mtd(self.df, date_col="date", as_of=datetime.date(2024, 3, 15))
        assert len(result) == 15  # March 1 to March 15

    def test_qtd_q1(self):
        result = qtd(self.df, date_col="date", as_of=datetime.date(2024, 3, 31))
        assert len(result) == 91  # Jan 1 to Mar 31


class TestTrailing:
    def test_last_n(self):
        r = pl.Series(list(range(100)))
        t = trailing(r, n=10)
        assert len(t) == 10
        assert t[-1] == 99

    def test_invalid_n(self):
        with pytest.raises(ValueError):
            trailing(pl.Series([1, 2, 3]), n=0)


class TestSinceInception:
    def test_identity(self, daily_returns):
        result = since_inception(daily_returns)
        assert result.equals(daily_returns)


class TestPeriodsPerYear:
    def test_daily(self):
        assert periods_per_year_for("D") == 252

    def test_monthly(self):
        assert periods_per_year_for("M") == 12

    def test_annual(self):
        assert periods_per_year_for("A") == 1

    def test_invalid(self):
        with pytest.raises(ValueError):
            periods_per_year_for("X")


class TestRateConversion:
    def test_roundtrip(self):
        annual = 0.05
        periodic = annual_to_periodic(annual, periods_per_year=252)
        back = periodic_to_annual(periodic, periods_per_year=252)
        assert math.isclose(back, annual, rel_tol=1e-9)

    def test_annual_to_periodic_daily(self):
        p = annual_to_periodic(0.05, periods_per_year=252)
        assert p < 0.05

    def test_invalid_periods(self):
        with pytest.raises(ValueError):
            annual_to_periodic(0.05, periods_per_year=0)
