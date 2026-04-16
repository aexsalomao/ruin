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
    def setup_method(self) -> None:
        dates = [datetime.date(2024, 1, 1) + datetime.timedelta(days=i) for i in range(365)]
        self.df = pl.DataFrame({"date": dates, "returns": [0.001] * 365})

    def test_ytd_within_year(self) -> None:
        # 2024 is a leap year: Jan 1 to Jun 30 is 31 + 29 + 31 + 30 + 31 + 30 = 182 days.
        result = ytd(self.df, date_col="date", as_of=datetime.date(2024, 6, 30))
        assert len(result) == 182

    def test_mtd_within_month(self) -> None:
        result = mtd(self.df, date_col="date", as_of=datetime.date(2024, 3, 15))
        assert len(result) == 15

    def test_qtd_q1(self) -> None:
        result = qtd(self.df, date_col="date", as_of=datetime.date(2024, 3, 31))
        assert len(result) == 91

    def test_qtd_q2(self) -> None:
        # Q2: April 1 to May 15 → 30 + 15 = 45 days
        result = qtd(self.df, date_col="date", as_of=datetime.date(2024, 5, 15))
        assert len(result) == 45

    def test_qtd_q4(self) -> None:
        # Q4 starts October 1; fixture is 365 days from Jan 1, so last date is Dec 30.
        # Oct 1 to Dec 30 inclusive → 31 + 30 + 30 = 91 days
        result = qtd(self.df, date_col="date", as_of=datetime.date(2024, 12, 31))
        assert len(result) == 91


class TestTrailing:
    def test_last_n(self) -> None:
        r = pl.Series(list(range(100)))
        t = trailing(r, n=10)
        assert len(t) == 10
        assert t[-1] == 99

    def test_invalid_n_zero(self) -> None:
        with pytest.raises(ValueError):
            trailing(pl.Series([1, 2, 3]), n=0)

    def test_invalid_n_negative(self) -> None:
        with pytest.raises(ValueError):
            trailing(pl.Series([1, 2, 3]), n=-1)

    def test_n_exceeds_length_returns_full(self) -> None:
        r = pl.Series([1, 2, 3])
        assert len(trailing(r, n=100)) == 3


class TestSinceInception:
    def test_identity(self, daily_returns: pl.Series) -> None:
        result = since_inception(daily_returns)
        assert result.equals(daily_returns)


class TestPeriodsPerYear:
    def test_daily(self) -> None:
        assert periods_per_year_for("D") == 252

    def test_weekly(self) -> None:
        assert periods_per_year_for("W") == 52

    def test_monthly(self) -> None:
        assert periods_per_year_for("M") == 12

    def test_quarterly(self) -> None:
        assert periods_per_year_for("Q") == 4

    def test_annual(self) -> None:
        assert periods_per_year_for("A") == 1

    def test_year_alias(self) -> None:
        assert periods_per_year_for("Y") == 1

    def test_case_insensitive(self) -> None:
        assert periods_per_year_for("d") == periods_per_year_for("D")

    def test_invalid(self) -> None:
        with pytest.raises(ValueError):
            periods_per_year_for("X")


class TestRateConversion:
    def test_roundtrip(self) -> None:
        annual = 0.05
        periodic = annual_to_periodic(annual, periods_per_year=252)
        assert math.isclose(
            periodic_to_annual(periodic, periods_per_year=252), annual, rel_tol=1e-9
        )

    def test_annual_to_periodic_smaller(self) -> None:
        assert annual_to_periodic(0.05, periods_per_year=252) < 0.05

    def test_periodic_to_annual_larger(self) -> None:
        assert periodic_to_annual(0.0001, periods_per_year=252) > 0.0001

    def test_invalid_periods_annual_to_periodic(self) -> None:
        with pytest.raises(ValueError):
            annual_to_periodic(0.05, periods_per_year=0)

    def test_invalid_periods_periodic_to_annual(self) -> None:
        with pytest.raises(ValueError):
            periodic_to_annual(0.05, periods_per_year=-1)

    def test_zero_rate_roundtrip(self) -> None:
        assert annual_to_periodic(0.0, periods_per_year=252) == 0.0
        assert periodic_to_annual(0.0, periods_per_year=252) == 0.0
