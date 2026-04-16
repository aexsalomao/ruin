"""Tests for ruin.rolling module."""

from __future__ import annotations

import math

import numpy as np
import polars as pl
import pytest

import ruin
from tests.conftest import FLOAT32_ABS_TOL


class TestRollingVolatility:
    def test_output_length(self, daily_returns: pl.Series) -> None:
        rv = ruin.rolling_volatility(daily_returns, window=20)
        assert len(rv) == len(daily_returns)

    def test_output_is_float32(self, daily_returns: pl.Series) -> None:
        rv = ruin.rolling_volatility(daily_returns, window=20)
        assert rv.dtype == pl.Float32

    def test_leading_nulls(self, daily_returns: pl.Series) -> None:
        rv = ruin.rolling_volatility(daily_returns, window=20)
        assert rv[:19].is_null().all()
        assert rv[19] is not None

    def test_full_window_matches_scalar(self, daily_returns: pl.Series) -> None:
        window = len(daily_returns)
        rv = ruin.rolling_volatility(daily_returns, window=window)
        scalar_vol = ruin.volatility(daily_returns)
        assert math.isclose(rv[-1], scalar_vol, rel_tol=1e-4)

    def test_custom_min_periods(self, daily_returns: pl.Series) -> None:
        rv = ruin.rolling_volatility(daily_returns, window=20, min_periods=5)
        # With min_periods=5, the first non-null index is at position 4
        assert rv[:4].is_null().all()
        assert rv[4] is not None


class TestRollingSharpe:
    def test_output_aligned(self, daily_returns: pl.Series) -> None:
        rs = ruin.rolling_sharpe(daily_returns, window=60, periods_per_year=252)
        assert len(rs) == len(daily_returns)
        assert rs[:59].is_null().all()

    def test_output_is_float32(self, daily_returns: pl.Series) -> None:
        rs = ruin.rolling_sharpe(daily_returns, window=60, periods_per_year=252)
        assert rs.dtype == pl.Float32


class TestRollingBeta:
    def test_self_beta_is_one(self, daily_returns: pl.Series) -> None:
        rb = ruin.rolling_beta(daily_returns, daily_returns, window=50)
        non_null = rb.drop_nulls()
        assert all(math.isclose(v, 1.0, rel_tol=1e-3) for v in non_null.to_list())

    def test_output_is_float32(self, daily_returns: pl.Series) -> None:
        rb = ruin.rolling_beta(daily_returns, daily_returns, window=50)
        assert rb.dtype == pl.Float32

    def test_length_mismatch_raises(self) -> None:
        with pytest.raises(ValueError):
            ruin.rolling_beta(
                pl.Series([0.01, 0.02]), pl.Series([0.01, 0.02, 0.03]), window=2
            )

    def test_string_window_raises(self, daily_returns: pl.Series) -> None:
        with pytest.raises(TypeError):
            ruin.rolling_beta(daily_returns, daily_returns, window="30d")


class TestRollingCorrelation:
    def test_self_correlation_is_one(self, daily_returns: pl.Series) -> None:
        rc = ruin.rolling_correlation(daily_returns, daily_returns, window=50)
        non_null = rc.drop_nulls()
        assert all(math.isclose(v, 1.0, rel_tol=1e-3) for v in non_null.to_list())

    def test_output_is_float32(self, daily_returns: pl.Series) -> None:
        rc = ruin.rolling_correlation(daily_returns, daily_returns, window=50)
        assert rc.dtype == pl.Float32


class TestRollingMaxDrawdown:
    def test_non_positive(self, daily_returns: pl.Series) -> None:
        rmdd = ruin.rolling_max_drawdown(daily_returns, window=60)
        non_null = rmdd.drop_nulls()
        assert (non_null <= FLOAT32_ABS_TOL).all()

    def test_output_length(self, daily_returns: pl.Series) -> None:
        rmdd = ruin.rolling_max_drawdown(daily_returns, window=60)
        assert len(rmdd) == len(daily_returns)

    def test_output_is_float32(self, daily_returns: pl.Series) -> None:
        rmdd = ruin.rolling_max_drawdown(daily_returns, window=60)
        assert rmdd.dtype == pl.Float32

    def test_string_window_raises(self, daily_returns: pl.Series) -> None:
        with pytest.raises(TypeError):
            ruin.rolling_max_drawdown(daily_returns, window="30d")


class TestRollingHitRate:
    def test_range(self, daily_returns: pl.Series) -> None:
        rhr = ruin.rolling_hit_rate(daily_returns, window=20)
        non_null = rhr.drop_nulls()
        assert (non_null >= 0.0).all()
        assert (non_null <= 1.0).all()

    def test_output_is_float32(self, daily_returns: pl.Series) -> None:
        rhr = ruin.rolling_hit_rate(daily_returns, window=20)
        assert rhr.dtype == pl.Float32


class TestRollingMoments:
    def test_rolling_skewness_is_float32(self, daily_returns: pl.Series) -> None:
        rs = ruin.rolling_skewness(daily_returns, window=60)
        assert rs.dtype == pl.Float32
        assert len(rs) == len(daily_returns)

    def test_rolling_excess_kurtosis_is_float32(self, daily_returns: pl.Series) -> None:
        rk = ruin.rolling_excess_kurtosis(daily_returns, window=60)
        assert rk.dtype == pl.Float32

    def test_rolling_autocorrelation_is_float32(self, daily_returns: pl.Series) -> None:
        rac = ruin.rolling_autocorrelation(daily_returns, window=60)
        assert rac.dtype == pl.Float32


class TestRollingProfitFactor:
    def test_output_is_float32(self, daily_returns: pl.Series) -> None:
        rpf = ruin.rolling_profit_factor(daily_returns, window=60)
        assert rpf.dtype == pl.Float32

    def test_output_length(self, daily_returns: pl.Series) -> None:
        rpf = ruin.rolling_profit_factor(daily_returns, window=60)
        assert len(rpf) == len(daily_returns)


class TestRollingInvalidInput:
    def test_invalid_min_periods(self, daily_returns: pl.Series) -> None:
        with pytest.raises(ValueError):
            ruin.rolling_volatility(daily_returns, window=20, min_periods=0)

    def test_multi_column_dataframe_raises(self) -> None:
        df = pl.DataFrame({"a": [0.01, 0.02], "b": [0.03, 0.04]})
        with pytest.raises(ValueError):
            ruin.rolling_volatility(df, window=2)

    def test_2d_array_raises(self) -> None:
        arr = np.array([[0.01, 0.02], [0.03, 0.04]])
        with pytest.raises(ValueError):
            ruin.rolling_volatility(arr, window=2)
