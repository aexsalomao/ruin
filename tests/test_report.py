"""Tests for ruin.report module."""

from __future__ import annotations

import math

import polars as pl
import pytest

from ruin.report import summary


def test_summary_returns_single_row_dataframe(daily_returns: pl.Series) -> None:
    df = summary(daily_returns, periods_per_year=252)
    assert isinstance(df, pl.DataFrame)
    assert len(df) == 1


def test_summary_includes_key_columns(daily_returns: pl.Series) -> None:
    df = summary(daily_returns, periods_per_year=252)
    for col in ["sharpe_ratio", "max_drawdown", "cagr", "volatility"]:
        assert col in df.columns


def test_summary_float_columns_are_float32(daily_returns: pl.Series) -> None:
    df = summary(daily_returns, periods_per_year=252)
    assert df["sharpe_ratio"].dtype == pl.Float32
    assert df["volatility"].dtype == pl.Float32
    assert df["max_drawdown"].dtype == pl.Float32
    assert df["cagr"].dtype == pl.Float32


def test_summary_benchmark_columns_are_null_without_benchmark(daily_returns: pl.Series) -> None:
    df = summary(daily_returns, periods_per_year=252)
    assert df["beta"][0] is None


def test_summary_benchmark_columns_are_filled_with_benchmark(
    daily_returns: pl.Series, benchmark_returns: pl.Series
) -> None:
    df = summary(daily_returns, benchmark_returns, periods_per_year=252)
    assert df["beta"][0] is not None
    assert df["correlation"][0] is not None
    assert df["alpha"][0] is not None
    assert df["tracking_error"][0] is not None


def test_summary_returns_one_row_per_strategy(multi_strategy: pl.DataFrame) -> None:
    df = summary(multi_strategy, periods_per_year=252)
    assert len(df) == 5
    assert "name" in df.columns
    assert set(df["name"].to_list()) == {f"strat_{i}" for i in range(5)}


def test_summary_strict_mode_raises_on_nan_returns() -> None:
    r = pl.Series([0.01, float("nan"), 0.02])
    with pytest.raises(ValueError, match="strict"):
        summary(r, periods_per_year=252, strict=True)


def test_summary_strict_mode_raises_on_nan_benchmark(daily_returns: pl.Series) -> None:
    b = pl.Series([float("nan")] + [0.0] * (len(daily_returns) - 1))
    with pytest.raises(ValueError, match="strict"):
        summary(daily_returns, b, periods_per_year=252, strict=True)


def test_summary_non_strict_drops_nan() -> None:
    r = pl.Series([0.01, float("nan"), 0.02])
    df = summary(r, periods_per_year=252, strict=False)
    assert len(df) == 1
    assert not math.isnan(df["cagr"][0])


def test_summary_risk_free_reduces_sharpe(daily_returns: pl.Series) -> None:
    sr_zero = summary(daily_returns, periods_per_year=252, risk_free=0.0)["sharpe_ratio"][0]
    sr_high = summary(daily_returns, periods_per_year=252, risk_free=0.0005)["sharpe_ratio"][0]
    assert sr_high < sr_zero
