"""Tests for ruin.periods module."""

from __future__ import annotations

import datetime

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


@pytest.fixture
def year_2024_daily() -> pl.DataFrame:
    dates = [datetime.date(2024, 1, 1) + datetime.timedelta(days=i) for i in range(365)]
    return pl.DataFrame({"date": dates, "returns": [0.001] * 365})


def test_ytd_returns_days_from_jan_1_to_as_of_date(year_2024_daily: pl.DataFrame) -> None:
    # 2024 leap year: Jan 1 to Jun 30 = 31+29+31+30+31+30 = 182 days.
    result = ytd(year_2024_daily, date_col="date", as_of=datetime.date(2024, 6, 30))
    assert len(result) == 182


def test_mtd_returns_days_from_month_start_to_as_of(year_2024_daily: pl.DataFrame) -> None:
    result = mtd(year_2024_daily, date_col="date", as_of=datetime.date(2024, 3, 15))
    assert len(result) == 15


def test_qtd_covers_full_q1(year_2024_daily: pl.DataFrame) -> None:
    result = qtd(year_2024_daily, date_col="date", as_of=datetime.date(2024, 3, 31))
    assert len(result) == 91


def test_qtd_covers_partial_q2(year_2024_daily: pl.DataFrame) -> None:
    # Q2: April 1 to May 15 → 30 + 15 = 45 days
    result = qtd(year_2024_daily, date_col="date", as_of=datetime.date(2024, 5, 15))
    assert len(result) == 45


def test_qtd_covers_q4_up_to_dec_30(year_2024_daily: pl.DataFrame) -> None:
    # Fixture spans Jan 1 to Dec 30. Q4 = Oct 1 to Dec 30 → 31 + 30 + 30 = 91 days.
    result = qtd(year_2024_daily, date_col="date", as_of=datetime.date(2024, 12, 31))
    assert len(result) == 91


def test_trailing_returns_last_n_values() -> None:
    r = pl.Series(list(range(100)))
    t = trailing(r, n=10)
    assert len(t) == 10
    assert t[-1] == 99


def test_trailing_rejects_zero_n() -> None:
    with pytest.raises(ValueError):
        trailing(pl.Series([1, 2, 3]), n=0)


def test_trailing_rejects_negative_n() -> None:
    with pytest.raises(ValueError):
        trailing(pl.Series([1, 2, 3]), n=-1)


def test_trailing_returns_full_series_when_n_exceeds_length() -> None:
    r = pl.Series([1, 2, 3])
    assert len(trailing(r, n=100)) == 3


def test_since_inception_returns_full_series(daily_returns: pl.Series) -> None:
    result = since_inception(daily_returns)
    assert result.equals(daily_returns)


def test_periods_per_year_for_daily_is_252() -> None:
    assert periods_per_year_for("D") == 252


def test_periods_per_year_for_weekly_is_52() -> None:
    assert periods_per_year_for("W") == 52


def test_periods_per_year_for_monthly_is_12() -> None:
    assert periods_per_year_for("M") == 12


def test_periods_per_year_for_quarterly_is_4() -> None:
    assert periods_per_year_for("Q") == 4


def test_periods_per_year_for_annual_is_1() -> None:
    assert periods_per_year_for("A") == 1


def test_periods_per_year_for_year_alias_is_1() -> None:
    assert periods_per_year_for("Y") == 1


def test_periods_per_year_for_is_case_insensitive() -> None:
    assert periods_per_year_for("d") == periods_per_year_for("D")


def test_periods_per_year_for_rejects_unknown_frequency() -> None:
    with pytest.raises(ValueError):
        periods_per_year_for("X")


def test_rate_conversion_roundtrip_preserves_annual_rate() -> None:
    annual = 0.05
    periodic = annual_to_periodic(annual, periods_per_year=252)
    assert periodic_to_annual(periodic, periods_per_year=252) == pytest.approx(annual, rel=1e-9)


def test_annual_to_periodic_is_smaller_than_annual_rate() -> None:
    assert annual_to_periodic(0.05, periods_per_year=252) < 0.05


def test_periodic_to_annual_is_larger_than_periodic_rate() -> None:
    assert periodic_to_annual(0.0001, periods_per_year=252) > 0.0001


def test_annual_to_periodic_rejects_zero_periods() -> None:
    with pytest.raises(ValueError):
        annual_to_periodic(0.05, periods_per_year=0)


def test_periodic_to_annual_rejects_negative_periods() -> None:
    with pytest.raises(ValueError):
        periodic_to_annual(0.05, periods_per_year=-1)


def test_rate_conversion_of_zero_is_zero() -> None:
    assert annual_to_periodic(0.0, periods_per_year=252) == 0.0
    assert periodic_to_annual(0.0, periods_per_year=252) == 0.0
