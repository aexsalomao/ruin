"""Tests for ruin.tail module."""

from __future__ import annotations

import math

import numpy as np
import polars as pl
import pytest

from ruin.tail import conditional_value_at_risk, expected_shortfall, value_at_risk


class TestVaR:
    def test_historical_positive(self, daily_returns: pl.Series) -> None:
        assert value_at_risk(daily_returns, confidence=0.95) > 0.0

    def test_parametric_positive(self, daily_returns: pl.Series) -> None:
        assert value_at_risk(daily_returns, confidence=0.95, method="parametric") > 0.0

    def test_higher_confidence_higher_var(self, daily_returns: pl.Series) -> None:
        assert value_at_risk(daily_returns, confidence=0.99) >= value_at_risk(
            daily_returns, confidence=0.95
        )

    def test_invalid_confidence_upper(self) -> None:
        with pytest.raises(ValueError):
            value_at_risk(pl.Series([0.01]), confidence=1.5)

    def test_invalid_confidence_lower(self) -> None:
        with pytest.raises(ValueError):
            value_at_risk(pl.Series([0.01]), confidence=0.0)

    def test_invalid_method(self) -> None:
        with pytest.raises(ValueError):
            value_at_risk(pl.Series([0.01, -0.01]), method="bad")

    def test_parametric_matches_normal_theory(self) -> None:
        rng = np.random.default_rng(42)
        sigma = 0.01
        # 100k draws: empirical mean ≈ 0, std ≈ sigma
        r = pl.Series(rng.normal(0.0, sigma, 100_000).tolist())
        var = value_at_risk(r, confidence=0.95, method="parametric")
        # z_{0.05} ≈ -1.645 → VaR ≈ 1.645 * sigma
        assert math.isclose(var, 1.645 * sigma, rel_tol=0.05)


class TestCVaR:
    def test_historical_cvar_ge_var(self, daily_returns: pl.Series) -> None:
        var = value_at_risk(daily_returns)
        cvar = conditional_value_at_risk(daily_returns)
        assert cvar >= var

    def test_parametric_cvar_ge_parametric_var(self, daily_returns: pl.Series) -> None:
        var = value_at_risk(daily_returns, method="parametric")
        cvar = conditional_value_at_risk(daily_returns, method="parametric")
        assert cvar >= var - 1e-10

    def test_expected_shortfall_alias(self, daily_returns: pl.Series) -> None:
        cvar = conditional_value_at_risk(daily_returns)
        es = expected_shortfall(daily_returns)
        assert math.isclose(cvar, es, rel_tol=1e-9)

    def test_invalid_confidence(self) -> None:
        with pytest.raises(ValueError):
            conditional_value_at_risk(pl.Series([0.01, -0.01]), confidence=-0.1)

    def test_invalid_method(self) -> None:
        with pytest.raises(ValueError):
            conditional_value_at_risk(pl.Series([0.01, -0.01]), method="bad")
