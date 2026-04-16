"""Shared fixtures for ruin test suite."""

from __future__ import annotations

import numpy as np
import polars as pl
import pytest

# Tolerances chosen for Float32 output precision. Float32 gives ~7 decimal
# digits of precision; rel_tol=1e-5 leaves comfortable margin for single
# arithmetic operations, and FLOAT32_ABS_TOL avoids false negatives near 0.
FLOAT32_REL_TOL: float = 1e-5
FLOAT32_ABS_TOL: float = 1e-6


@pytest.fixture
def daily_returns() -> pl.Series:
    """252 daily returns drawn from N(0.0004, 0.01^2)."""
    rng = np.random.default_rng(42)
    return pl.Series("returns", rng.normal(0.0004, 0.01, 252), dtype=pl.Float64)


@pytest.fixture
def short_returns() -> pl.Series:
    """A short 10-period return series for exact hand-computed tests."""
    return pl.Series(
        "returns",
        [0.01, -0.02, 0.03, -0.01, 0.02, 0.01, -0.03, 0.02, 0.01, -0.01],
        dtype=pl.Float64,
    )


@pytest.fixture
def benchmark_returns(daily_returns: pl.Series) -> pl.Series:
    """Benchmark correlated with daily_returns."""
    rng = np.random.default_rng(99)
    noise = pl.Series("benchmark", rng.normal(0.0003, 0.01, 252), dtype=pl.Float64)
    return (daily_returns * 0.7 + noise * 0.3).rename("benchmark")


@pytest.fixture
def multi_strategy() -> pl.DataFrame:
    """DataFrame with 5 strategy columns, 252 rows."""
    rng = np.random.default_rng(7)
    data = {f"strat_{i}": rng.normal(0.0003, 0.01, 252) for i in range(5)}
    return pl.DataFrame(data)


@pytest.fixture
def ten_year_daily() -> pl.Series:
    """10 years of daily returns (2520 obs)."""
    rng = np.random.default_rng(42)
    return pl.Series("returns", rng.normal(0.0004, 0.01, 2520), dtype=pl.Float64)


@pytest.fixture
def constant_returns() -> pl.Series:
    """A series of identical non-zero returns — useful for degenerate-input tests."""
    return pl.Series("returns", [0.001] * 50, dtype=pl.Float64)


@pytest.fixture
def zero_returns() -> pl.Series:
    """A series of zero returns — maximally degenerate input."""
    return pl.Series("returns", [0.0] * 50, dtype=pl.Float64)


@pytest.fixture
def returns_with_nan() -> pl.Series:
    """Returns with interleaved NaN values — exercises the drop-NaN policy."""
    vals: list[float] = [0.01, float("nan"), -0.02, 0.03, float("nan"), -0.01, 0.02]
    return pl.Series("returns", vals, dtype=pl.Float64)
