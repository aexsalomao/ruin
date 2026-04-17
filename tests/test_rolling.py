"""Tests for ruin.rolling module."""

from __future__ import annotations

import numpy as np
import polars as pl
import pytest

import ruin
from tests.conftest import FLOAT32_ABS_TOL


def test_rolling_volatility_length_matches_input(daily_returns: pl.Series) -> None:
    rv = ruin.rolling_volatility(daily_returns, window=20)
    assert len(rv) == len(daily_returns)


def test_rolling_volatility_output_is_float32(daily_returns: pl.Series) -> None:
    rv = ruin.rolling_volatility(daily_returns, window=20)
    assert rv.dtype == pl.Float32


def test_rolling_volatility_has_leading_nulls(daily_returns: pl.Series) -> None:
    rv = ruin.rolling_volatility(daily_returns, window=20)
    assert rv[:19].is_null().all()
    assert rv[19] is not None


def test_rolling_volatility_full_window_matches_scalar(daily_returns: pl.Series) -> None:
    window = len(daily_returns)
    rv = ruin.rolling_volatility(daily_returns, window=window)
    scalar_vol = ruin.volatility(daily_returns)
    assert rv[-1] == pytest.approx(scalar_vol, rel=1e-4)


def test_rolling_volatility_respects_custom_min_periods(daily_returns: pl.Series) -> None:
    rv = ruin.rolling_volatility(daily_returns, window=20, min_periods=5)
    assert rv[:4].is_null().all()
    assert rv[4] is not None


def test_rolling_sharpe_length_matches_and_has_leading_nulls(daily_returns: pl.Series) -> None:
    rs = ruin.rolling_sharpe(daily_returns, window=60, periods_per_year=252)
    assert len(rs) == len(daily_returns)
    assert rs[:59].is_null().all()


def test_rolling_sharpe_output_is_float32(daily_returns: pl.Series) -> None:
    rs = ruin.rolling_sharpe(daily_returns, window=60, periods_per_year=252)
    assert rs.dtype == pl.Float32


def test_rolling_beta_of_series_against_itself_is_one(daily_returns: pl.Series) -> None:
    rb = ruin.rolling_beta(daily_returns, daily_returns, window=50)
    non_null = rb.drop_nulls()
    assert all(v == pytest.approx(1.0, rel=1e-3) for v in non_null.to_list())


def test_rolling_beta_output_is_float32(daily_returns: pl.Series) -> None:
    rb = ruin.rolling_beta(daily_returns, daily_returns, window=50)
    assert rb.dtype == pl.Float32


def test_rolling_beta_rejects_length_mismatch() -> None:
    with pytest.raises(ValueError):
        ruin.rolling_beta(pl.Series([0.01, 0.02]), pl.Series([0.01, 0.02, 0.03]), window=2)


def test_rolling_beta_rejects_string_window(daily_returns: pl.Series) -> None:
    with pytest.raises(TypeError):
        ruin.rolling_beta(daily_returns, daily_returns, window="30d")


def test_rolling_correlation_of_series_against_itself_is_one(daily_returns: pl.Series) -> None:
    rc = ruin.rolling_correlation(daily_returns, daily_returns, window=50)
    non_null = rc.drop_nulls()
    assert all(v == pytest.approx(1.0, rel=1e-3) for v in non_null.to_list())


def test_rolling_correlation_output_is_float32(daily_returns: pl.Series) -> None:
    rc = ruin.rolling_correlation(daily_returns, daily_returns, window=50)
    assert rc.dtype == pl.Float32


def test_rolling_max_drawdown_is_non_positive(daily_returns: pl.Series) -> None:
    rmdd = ruin.rolling_max_drawdown(daily_returns, window=60)
    non_null = rmdd.drop_nulls()
    assert (non_null <= FLOAT32_ABS_TOL).all()


def test_rolling_max_drawdown_length_matches_input(daily_returns: pl.Series) -> None:
    rmdd = ruin.rolling_max_drawdown(daily_returns, window=60)
    assert len(rmdd) == len(daily_returns)


def test_rolling_max_drawdown_output_is_float32(daily_returns: pl.Series) -> None:
    rmdd = ruin.rolling_max_drawdown(daily_returns, window=60)
    assert rmdd.dtype == pl.Float32


def test_rolling_max_drawdown_rejects_string_window(daily_returns: pl.Series) -> None:
    with pytest.raises(TypeError):
        ruin.rolling_max_drawdown(daily_returns, window="30d")


def test_rolling_hit_rate_stays_between_zero_and_one(daily_returns: pl.Series) -> None:
    rhr = ruin.rolling_hit_rate(daily_returns, window=20)
    non_null = rhr.drop_nulls()
    assert (non_null >= 0.0).all()
    assert (non_null <= 1.0).all()


def test_rolling_hit_rate_output_is_float32(daily_returns: pl.Series) -> None:
    rhr = ruin.rolling_hit_rate(daily_returns, window=20)
    assert rhr.dtype == pl.Float32


def test_rolling_skewness_output_is_float32_and_aligned(daily_returns: pl.Series) -> None:
    rs = ruin.rolling_skewness(daily_returns, window=60)
    assert rs.dtype == pl.Float32
    assert len(rs) == len(daily_returns)


def test_rolling_excess_kurtosis_output_is_float32(daily_returns: pl.Series) -> None:
    rk = ruin.rolling_excess_kurtosis(daily_returns, window=60)
    assert rk.dtype == pl.Float32


def test_rolling_autocorrelation_output_is_float32(daily_returns: pl.Series) -> None:
    rac = ruin.rolling_autocorrelation(daily_returns, window=60)
    assert rac.dtype == pl.Float32


def test_rolling_profit_factor_output_is_float32(daily_returns: pl.Series) -> None:
    rpf = ruin.rolling_profit_factor(daily_returns, window=60)
    assert rpf.dtype == pl.Float32


def test_rolling_profit_factor_length_matches_input(daily_returns: pl.Series) -> None:
    rpf = ruin.rolling_profit_factor(daily_returns, window=60)
    assert len(rpf) == len(daily_returns)


def test_rolling_rejects_zero_min_periods(daily_returns: pl.Series) -> None:
    with pytest.raises(ValueError):
        ruin.rolling_volatility(daily_returns, window=20, min_periods=0)


def test_rolling_rejects_multi_column_dataframe() -> None:
    df = pl.DataFrame({"a": [0.01, 0.02], "b": [0.03, 0.04]})
    with pytest.raises(ValueError):
        ruin.rolling_volatility(df, window=2)


def test_rolling_rejects_two_dimensional_array() -> None:
    arr = np.array([[0.01, 0.02], [0.03, 0.04]])
    with pytest.raises(ValueError):
        ruin.rolling_volatility(arr, window=2)
