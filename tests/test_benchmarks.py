"""Performance benchmark tests using pytest-benchmark."""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
import polars as pl
import pytest

import ruin
from ruin.report import summary

if TYPE_CHECKING:
    from pytest_benchmark.fixture import BenchmarkFixture


@pytest.fixture(scope="module")
def ten_year_daily_benchmark_input() -> pl.Series:
    rng = np.random.default_rng(42)
    return pl.Series(rng.normal(0.0004, 0.01, 2520))


@pytest.fixture(scope="module")
def ten_year_daily_benchmark_reference() -> pl.Series:
    rng = np.random.default_rng(99)
    return pl.Series(rng.normal(0.0003, 0.01, 2520))


@pytest.fixture(scope="module")
def multi_strategy_large() -> pl.DataFrame:
    rng = np.random.default_rng(7)
    data = {f"s{i}": rng.normal(0.0003, 0.01, 2520) for i in range(500)}
    return pl.DataFrame(data)


def test_summary_completes_for_ten_year_daily_series(
    benchmark: BenchmarkFixture, ten_year_daily_benchmark_input: pl.Series
) -> None:
    result = benchmark(summary, ten_year_daily_benchmark_input, periods_per_year=252)
    assert result is not None


def test_summary_completes_for_500_strategy_dataframe(
    benchmark: BenchmarkFixture, multi_strategy_large: pl.DataFrame
) -> None:
    result = benchmark(summary, multi_strategy_large, periods_per_year=252)
    assert len(result) == 500


def test_rolling_volatility_completes_for_ten_year_daily_series(
    benchmark: BenchmarkFixture, ten_year_daily_benchmark_input: pl.Series
) -> None:
    result = benchmark(ruin.rolling_volatility, ten_year_daily_benchmark_input, window=60)
    assert result is not None


def test_rolling_sharpe_completes_for_ten_year_daily_series(
    benchmark: BenchmarkFixture, ten_year_daily_benchmark_input: pl.Series
) -> None:
    result = benchmark(
        ruin.rolling_sharpe, ten_year_daily_benchmark_input, window=60, periods_per_year=252
    )
    assert result is not None
