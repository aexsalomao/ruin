"""Tests for ruin.tail module."""

from __future__ import annotations

import math

import polars as pl
import pytest

from ruin.tail import conditional_value_at_risk, expected_shortfall, value_at_risk


class TestVaR:
    def test_historical_positive(self, daily_returns):
        var = value_at_risk(daily_returns, confidence=0.95)
        assert var > 0.0

    def test_parametric_positive(self, daily_returns):
        var = value_at_risk(daily_returns, confidence=0.95, method="parametric")
        assert var > 0.0

    def test_higher_confidence_higher_var(self, daily_returns):
        var95 = value_at_risk(daily_returns, confidence=0.95)
        var99 = value_at_risk(daily_returns, confidence=0.99)
        assert var99 >= var95

    def test_invalid_confidence(self):
        with pytest.raises(ValueError):
            value_at_risk(pl.Series([0.01]), confidence=1.5)

    def test_invalid_method(self):
        with pytest.raises(ValueError):
            value_at_risk(pl.Series([0.01, -0.01]), method="bad")


class TestCVaR:
    def test_historical_cvar_ge_var(self, daily_returns):
        var = value_at_risk(daily_returns)
        cvar = conditional_value_at_risk(daily_returns)
        assert cvar >= var

    def test_parametric_cvar_ge_parametric_var(self, daily_returns):
        var = value_at_risk(daily_returns, method="parametric")
        cvar = conditional_value_at_risk(daily_returns, method="parametric")
        assert cvar >= var - 1e-10

    def test_expected_shortfall_alias(self, daily_returns):
        cvar = conditional_value_at_risk(daily_returns)
        es = expected_shortfall(daily_returns)
        assert math.isclose(cvar, es, rel_tol=1e-9)
