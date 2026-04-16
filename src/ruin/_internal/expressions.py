"""Reusable Polars expression builders for internal use."""

from __future__ import annotations

import polars as pl


def excess_returns_expr(col: str = "returns", risk_free: float = 0.0) -> pl.Expr:
    """Expression that computes per-period excess returns over risk_free."""
    return pl.col(col) - risk_free


def compound_growth_expr(col: str = "returns") -> pl.Expr:
    """Expression: (1 + r).cumprod() — running compounded growth factor."""
    return (1 + pl.col(col)).cum_prod()


def running_max_expr(col: str = "returns") -> pl.Expr:
    """Cumulative maximum of a column (for drawdown computation)."""
    return pl.col(col).cum_max()
