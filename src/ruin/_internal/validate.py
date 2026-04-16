"""Input validation, schema enforcement, and NaN policy for ruin."""

from __future__ import annotations

import numpy as np
import polars as pl

# Public float dtype for all Polars outputs.
FLOAT_DTYPE: type[pl.DataType] = pl.Float32

# Internal float dtype used during computation. Float64 avoids catastrophic
# cancellation in cum_prod / variance-style accumulators; results are cast to
# FLOAT_DTYPE at the boundary of each public function.
INTERNAL_FLOAT_DTYPE: type[pl.DataType] = pl.Float64

# Type alias for inputs accepted by all public functions.
ReturnInput = pl.Series | pl.Expr | np.ndarray | pl.DataFrame


def to_series(returns: ReturnInput, name: str = "returns") -> pl.Series:
    """Coerce any supported input to a Float64 pl.Series with NaN/null dropped.

    pl.Expr raises — expressions must live inside `.select()` / `.with_columns()`. 1-D arrays
    and single-column DataFrames are accepted; multi-column DataFrames raise. `name` is used
    in error messages. Callers cast the *result* to Float32 before returning to users.
    """
    if isinstance(returns, pl.Expr):
        raise TypeError(
            f"'{name}' is a Polars Expr. Use it inside .select() / .with_columns(), "
            "or convert to a Series first."
        )
    if isinstance(returns, np.ndarray):
        if returns.ndim != 1:
            raise ValueError(f"'{name}' must be a 1-D array, got shape {returns.shape}")
        s = pl.Series(name, returns, dtype=INTERNAL_FLOAT_DTYPE)
    elif isinstance(returns, pl.DataFrame):
        if returns.width != 1:
            raise ValueError(
                f"'{name}' is a multi-column DataFrame; pass a single-column DataFrame "
                "or use the multi-column path explicitly."
            )
        s = returns.to_series(0).cast(INTERNAL_FLOAT_DTYPE)
    elif isinstance(returns, pl.Series):
        s = returns.cast(INTERNAL_FLOAT_DTYPE)
    else:
        raise TypeError(
            f"'{name}' must be a pl.Series, pl.Expr, np.ndarray, or pl.DataFrame; "
            f"got {type(returns).__name__}"
        )

    # Drop NaN (float NaN) and null (Polars null / None)
    s = s.drop_nans().drop_nulls()
    return s


def to_dataframe(returns: ReturnInput, name: str = "returns") -> pl.DataFrame:
    """Coerce any supported input to a Float64 pl.DataFrame with rows containing any NaN/null dropped.

    Scalar / Series inputs are wrapped in a single-column DataFrame named `name`.
    """
    if isinstance(returns, pl.DataFrame):
        df = returns.cast({col: INTERNAL_FLOAT_DTYPE for col in returns.columns})
        # Drop rows where ANY column is null/NaN
        return df.drop_nulls()
    s = to_series(returns, name=name)
    return s.to_frame()


def require_same_length(
    a: pl.Series,
    b: pl.Series,
    name_a: str = "returns",
    name_b: str = "benchmark",
) -> None:
    """Raise ValueError if two Series have different lengths."""
    if len(a) != len(b):
        raise ValueError(
            f"'{name_a}' (length {len(a)}) and '{name_b}' (length {len(b)}) "
            f"must have the same length."
        )


def require_minimum_length(s: pl.Series, min_len: int, metric_name: str = "metric") -> None:
    """Raise ValueError if Series has fewer than `min_len` observations."""
    if len(s) < min_len:
        raise ValueError(f"'{metric_name}' requires at least {min_len} observations; got {len(s)}.")


def require_strictly_positive(value: float, param: str) -> None:
    """Raise ValueError if a numeric parameter is not strictly positive."""
    if value <= 0:
        raise ValueError(f"'{param}' must be strictly positive; got {value}.")


def check_nan_strict(returns: ReturnInput, name: str = "returns") -> None:
    """Raise ValueError if input contains any NaN/null (used when `summary(strict=True)`)."""
    if isinstance(returns, pl.Series):
        n_nan = int(returns.is_nan().sum()) + int(returns.is_null().sum())
    elif isinstance(returns, np.ndarray):
        n_nan = int(np.isnan(returns).sum())
    elif isinstance(returns, pl.DataFrame):
        n_nan = sum(
            int(returns[col].is_nan().sum()) + int(returns[col].is_null().sum())
            for col in returns.columns
        )
    else:
        raise TypeError(f"Unsupported type for NaN check: {type(returns).__name__}")

    if n_nan > 0:
        raise ValueError(f"'{name}' contains {n_nan} NaN/null value(s) and strict=True was set.")


def align_benchmark(
    returns: ReturnInput,
    benchmark: ReturnInput,
) -> tuple[pl.Series, pl.Series]:
    """Return `(r, b)` as equal-length Float64 Series. Trusts caller for order; requires equal length."""
    r = to_series(returns, name="returns")
    b = to_series(benchmark, name="benchmark")
    require_same_length(r, b)
    return r, b
