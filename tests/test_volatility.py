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


class TestVolatility:
    def test_known_value(self) -> None:
        r = pl.Series([0.01, -0.01, 0.02, -0.02])
        v = volatility(r)
        expected = float(r.std(ddof=1))
        assert math.isclose(v, expected, rel_tol=1e-9)

    def test_ddof0_less_than_ddof1(self) -> None:
        r = pl.Series([1.0, 2.0, 3.0, 4.0])
        assert volatility(r, ddof=0) < volatility(r, ddof=1)

    def test_single_element_raises(self) -> None:
        with pytest.raises(ValueError):
            volatility(pl.Series([0.01]))

    def test_constant_series_is_zero(self) -> None:
        r = pl.Series([0.001] * 100)
        assert volatility(r) == 0.0

    def test_numpy_input(self) -> None:
        arr = np.array([0.01, -0.01, 0.02, -0.02])
        assert math.isclose(volatility(arr), float(pl.Series(arr).std(ddof=1)), rel_tol=1e-9)

    def test_scale_invariance(self) -> None:
        r = pl.Series([0.01, -0.01, 0.02, -0.02])
        assert math.isclose(volatility(r * 10), volatility(r) * 10, rel_tol=1e-9)


class TestAnnualizeVolatility:
    def test_sqrt_of_time(self) -> None:
        r = pl.Series([0.01, -0.01, 0.02, -0.02])
        daily_vol = volatility(r)
        ann_vol = annualize_volatility(r, periods_per_year=252)
        assert math.isclose(ann_vol, daily_vol * math.sqrt(252), rel_tol=1e-9)

    def test_invalid_periods_zero(self) -> None:
        with pytest.raises(ValueError):
            annualize_volatility(pl.Series([0.01, 0.02]), periods_per_year=0)

    def test_invalid_periods_negative(self) -> None:
        with pytest.raises(ValueError):
            annualize_volatility(pl.Series([0.01, 0.02]), periods_per_year=-1)


class TestDownsideDeviation:
    def test_no_downside(self) -> None:
        r = pl.Series([0.01, 0.02, 0.03])
        assert downside_deviation(r, threshold=0.0) == 0.0

    def test_all_downside(self) -> None:
        r = pl.Series([-0.01, -0.02, -0.03])
        assert downside_deviation(r, threshold=0.0) > 0.0

    def test_symmetric_le_total_vol(self) -> None:
        r = pl.Series([0.02, -0.02])
        assert downside_deviation(r) <= volatility(r)

    def test_threshold_shifts_downside(self) -> None:
        # With threshold above every return, all periods contribute.
        r = pl.Series([0.01, 0.02, 0.03])
        assert downside_deviation(r, threshold=0.10) > 0.0

    def test_invalid_ddof(self) -> None:
        with pytest.raises(ValueError):
            downside_deviation(pl.Series([-0.01]), ddof=2)


class TestSemiDeviation:
    def test_all_positive_returns_zero(self) -> None:
        r = pl.Series([0.01, 0.02, 0.03])
        assert semi_deviation(r) == 0.0

    def test_negative_only(self) -> None:
        r = pl.Series([-0.01, -0.02])
        sd = semi_deviation(r)
        expected = float(pl.Series([-0.01, -0.02]).std(ddof=0))
        assert math.isclose(sd, expected, rel_tol=1e-9)

    def test_mixed_uses_only_negatives(self) -> None:
        r = pl.Series([0.05, 0.05, -0.01, -0.02])
        sd = semi_deviation(r)
        expected = float(pl.Series([-0.01, -0.02]).std(ddof=0))
        assert math.isclose(sd, expected, rel_tol=1e-9)

    def test_invalid_ddof(self) -> None:
        with pytest.raises(ValueError):
            semi_deviation(pl.Series([-0.01]), ddof=2)
