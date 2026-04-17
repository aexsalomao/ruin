"""Tests for ruin.tail module."""

from __future__ import annotations

import numpy as np
import polars as pl
import pytest

from ruin.tail import conditional_value_at_risk, expected_shortfall, value_at_risk


def test_historical_value_at_risk_is_positive(daily_returns: pl.Series) -> None:
    assert value_at_risk(daily_returns, confidence=0.95) > 0.0


def test_parametric_value_at_risk_is_positive(daily_returns: pl.Series) -> None:
    assert value_at_risk(daily_returns, confidence=0.95, method="parametric") > 0.0


def test_value_at_risk_increases_with_confidence(daily_returns: pl.Series) -> None:
    assert value_at_risk(daily_returns, confidence=0.99) >= value_at_risk(
        daily_returns, confidence=0.95
    )


def test_value_at_risk_rejects_confidence_above_one() -> None:
    with pytest.raises(ValueError):
        value_at_risk(pl.Series([0.01]), confidence=1.5)


def test_value_at_risk_rejects_confidence_zero() -> None:
    with pytest.raises(ValueError):
        value_at_risk(pl.Series([0.01]), confidence=0.0)


def test_value_at_risk_rejects_unknown_method() -> None:
    with pytest.raises(ValueError):
        value_at_risk(pl.Series([0.01, -0.01]), method="bad")


def test_parametric_value_at_risk_matches_normal_theory() -> None:
    rng = np.random.default_rng(42)
    sigma = 0.01
    r = pl.Series(rng.normal(0.0, sigma, 100_000).tolist())
    var = value_at_risk(r, confidence=0.95, method="parametric")
    # z_{0.05} ≈ -1.645 → VaR ≈ 1.645 * sigma
    assert var == pytest.approx(1.645 * sigma, rel=0.05)


def test_historical_cvar_is_at_least_historical_var(daily_returns: pl.Series) -> None:
    var = value_at_risk(daily_returns)
    cvar = conditional_value_at_risk(daily_returns)
    assert cvar >= var


def test_parametric_cvar_is_at_least_parametric_var(daily_returns: pl.Series) -> None:
    var = value_at_risk(daily_returns, method="parametric")
    cvar = conditional_value_at_risk(daily_returns, method="parametric")
    assert cvar >= var - 1e-10


def test_expected_shortfall_equals_conditional_value_at_risk(daily_returns: pl.Series) -> None:
    cvar = conditional_value_at_risk(daily_returns)
    es = expected_shortfall(daily_returns)
    assert cvar == pytest.approx(es, rel=1e-9)


def test_conditional_value_at_risk_rejects_invalid_confidence() -> None:
    with pytest.raises(ValueError):
        conditional_value_at_risk(pl.Series([0.01, -0.01]), confidence=-0.1)


def test_conditional_value_at_risk_rejects_unknown_method() -> None:
    with pytest.raises(ValueError):
        conditional_value_at_risk(pl.Series([0.01, -0.01]), method="bad")
