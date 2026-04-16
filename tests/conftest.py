"""Shared fixtures for ruin test suite."""

from __future__ import annotations

import numpy as np
import polars as pl
import pytest


@pytest.fixture
def daily_returns() -> pl.Series:
    """252 daily returns drawn from N(0.0004, 0.01^2)."""
    rng = np.random.default_rng(42)
    return pl.Series("returns", rng.normal(0.0004, 0.01, 252), dtype=pl.Float64)


@pytest.fixture
def short_returns() -> pl.Series:
    """A short 10-period return series for exact hand-computed tests."""
    return pl.Series("returns", [0.01, -0.02, 0.03, -0.01, 0.02, 0.01, -0.03, 0.02, 0.01, -0.01])


@pytest.fixture
def benchmark_returns(daily_returns: pl.Series) -> pl.Series:
    """Benchmark correlated with daily_returns."""
    rng = np.random.default_rng(99)
    noise = pl.Series("benchmark", rng.normal(0.0003, 0.01, 252), dtype=pl.Float64)
    return daily_returns * 0.7 + noise * 0.3


@pytest.fixture
def multi_strategy() -> pl.DataFrame:
    """DataFrame with 5 strategy columns, 252 rows."""
    rng = np.random.default_rng(7)
    data = {f"strat_{i}": rng.normal(0.0003, 0.01, 252) for i in range(5)}
    return pl.DataFrame(data)
