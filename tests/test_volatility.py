"""Tests for ruin.volatility module."""

from __future__ import annotations

import math

import numpy as np
import polars as pl
import pytest

from ruin.volatility import (
    annualize_volatility,
    downside_deviation,
    semi_deviation,
    volatility,
)


def test_volatility_matches_sample_std() -> None:
    r = pl.Series([0.01, -0.01, 0.02, -0.02])
    v = volatility(r)
    expected = float(r.std(ddof=1))
    assert v == pytest.approx(expected, rel=1e-9)


def test_volatility_ddof_zero_is_less_than_ddof_one() -> None:
    r = pl.Series([1.0, 2.0, 3.0, 4.0])
    assert volatility(r, ddof=0) < volatility(r, ddof=1)


def test_volatility_rejects_single_element_series() -> None:
    with pytest.raises(ValueError):
        volatility(pl.Series([0.01]))


def test_volatility_is_zero_for_constant_series() -> None:
    r = pl.Series([0.001] * 100)
    assert volatility(r) == 0.0


def test_volatility_accepts_numpy_input() -> None:
    arr = np.array([0.01, -0.01, 0.02, -0.02])
    expected = float(pl.Series(arr).std(ddof=1))
    assert volatility(arr) == pytest.approx(expected, rel=1e-9)


def test_volatility_scales_linearly() -> None:
    r = pl.Series([0.01, -0.01, 0.02, -0.02])
    assert volatility(r * 10) == pytest.approx(volatility(r) * 10, rel=1e-9)


def test_annualize_volatility_multiplies_by_sqrt_periods() -> None:
    r = pl.Series([0.01, -0.01, 0.02, -0.02])
    daily_vol = volatility(r)
    ann_vol = annualize_volatility(r, periods_per_year=252)
    assert ann_vol == pytest.approx(daily_vol * math.sqrt(252), rel=1e-9)


def test_annualize_volatility_rejects_zero_periods() -> None:
    with pytest.raises(ValueError):
        annualize_volatility(pl.Series([0.01, 0.02]), periods_per_year=0)


def test_annualize_volatility_rejects_negative_periods() -> None:
    with pytest.raises(ValueError):
        annualize_volatility(pl.Series([0.01, 0.02]), periods_per_year=-1)


def test_downside_deviation_is_zero_without_downside() -> None:
    r = pl.Series([0.01, 0.02, 0.03])
    assert downside_deviation(r, threshold=0.0) == 0.0


def test_downside_deviation_is_positive_when_all_returns_below_threshold() -> None:
    r = pl.Series([-0.01, -0.02, -0.03])
    assert downside_deviation(r, threshold=0.0) > 0.0


def test_downside_deviation_is_at_most_total_volatility() -> None:
    r = pl.Series([0.02, -0.02])
    assert downside_deviation(r) <= volatility(r)


def test_downside_deviation_includes_returns_below_threshold_above_zero() -> None:
    r = pl.Series([0.01, 0.02, 0.03])
    assert downside_deviation(r, threshold=0.10) > 0.0


def test_downside_deviation_rejects_invalid_ddof() -> None:
    with pytest.raises(ValueError):
        downside_deviation(pl.Series([-0.01]), ddof=2)


def test_semi_deviation_is_zero_when_all_returns_positive() -> None:
    r = pl.Series([0.01, 0.02, 0.03])
    assert semi_deviation(r) == 0.0


def test_semi_deviation_matches_std_of_negatives() -> None:
    r = pl.Series([-0.01, -0.02])
    expected = float(pl.Series([-0.01, -0.02]).std(ddof=0))
    assert semi_deviation(r) == pytest.approx(expected, rel=1e-9)


def test_semi_deviation_uses_only_negative_returns() -> None:
    r = pl.Series([0.05, 0.05, -0.01, -0.02])
    expected = float(pl.Series([-0.01, -0.02]).std(ddof=0))
    assert semi_deviation(r) == pytest.approx(expected, rel=1e-9)


def test_semi_deviation_rejects_invalid_ddof() -> None:
    with pytest.raises(ValueError):
        semi_deviation(pl.Series([-0.01]), ddof=2)
