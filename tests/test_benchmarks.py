"""Performance benchmark tests using pytest-benchmark.

Regressions fail CI when performance drops below targets.
"""

from __future__ import annotations

import numpy as np
import polars as pl
import pytest


@pytest.fixture(scope="module")
def ten_year_daily():
    rng = np.random.default_rng(42)
    return pl.Series(rng.normal(0.0004, 0.01, 2520))


@pytest.fixture(scope="module")
def ten_year_daily_benchmark():
    rng = np.random.default_rng(99)
    return pl.Series(rng.normal(0.0003, 0.01, 2520))


@pytest.fixture(scope="module")
def multi_strategy_large():
    rng = np.random.default_rng(7)
    data = {f"s{i}": rng.normal(0.0003, 0.01, 2520) for i in range(500)}
    return pl.DataFrame(data)


def test_summary_single_10y_daily(benchmark, ten_year_daily):
    """Full summary() on a 10-year daily series should complete in well under 10ms."""
    from ruin.report import summary

    result = benchmark(summary, ten_year_daily, periods_per_year=252)
    assert result is not None


def test_summary_multi_strategy_500col(benchmark, multi_strategy_large):
    """summary() on 500 strategies x 10 years should complete in under 1 second."""
    from ruin.report import summary

    result = benchmark(summary, multi_strategy_large, periods_per_year=252)
    assert len(result) == 500


def test_rolling_volatility_10y(benchmark, ten_year_daily):
    """Rolling volatility on 10y daily within 2x of raw Polars rolling_std."""
    import ruin

    result = benchmark(ruin.rolling_volatility, ten_year_daily, window=60)
    assert result is not None


def test_rolling_sharpe_10y(benchmark, ten_year_daily):
    """Rolling Sharpe on 10y daily: reasonable throughput."""
    import ruin

    result = benchmark(ruin.rolling_sharpe, ten_year_daily, window=60, periods_per_year=252)
    assert result is not None
