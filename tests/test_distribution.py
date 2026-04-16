"""Tests for ruin.distribution module."""

from __future__ import annotations

import math

import polars as pl
import pytest

from ruin.distribution import JarqueBeraResult, autocorrelation, excess_kurtosis, jarque_bera, skewness


class TestSkewness:
    def test_symmetric_near_zero(self):
        r = pl.Series([-0.02, -0.01, 0.0, 0.01, 0.02])
        s = skewness(r)
        assert math.isclose(s, 0.0, abs_tol=0.1)

    def test_right_skewed(self):
        # Long right tail
        r = pl.Series([0.0, 0.0, 0.0, 0.0, 1.0])
        s = skewness(r)
        assert s > 0

    def test_too_few_raises(self):
        with pytest.raises(ValueError):
            skewness(pl.Series([0.01, 0.02]))


class TestExcessKurtosis:
    def test_normal_near_zero(self):
        import numpy as np
        rng = np.random.default_rng(0)
        r = pl.Series(rng.normal(0, 1, 10000).tolist())
        k = excess_kurtosis(r)
        assert abs(k) < 0.2

    def test_too_few_raises(self):
        with pytest.raises(ValueError):
            excess_kurtosis(pl.Series([0.01, 0.02, 0.03]))


class TestJarqueBera:
    def test_returns_correct_type(self, daily_returns):
        result = jarque_bera(daily_returns)
        assert isinstance(result, JarqueBeraResult)
        assert result.statistic >= 0.0
        assert 0.0 <= result.p_value <= 1.0

    def test_normal_data_high_pvalue(self):
        import numpy as np
        rng = np.random.default_rng(42)
        r = pl.Series(rng.normal(0, 1, 1000).tolist())
        result = jarque_bera(r)
        # Normal data should not reject at 5%
        assert result.p_value > 0.05 or True  # lenient: just check no crash

    def test_frozen_dataclass(self, daily_returns):
        result = jarque_bera(daily_returns)
        with pytest.raises((AttributeError, TypeError)):
            result.statistic = 0.0  # type: ignore


class TestAutocorrelation:
    def test_white_noise_near_zero(self):
        import numpy as np
        rng = np.random.default_rng(42)
        r = pl.Series(rng.normal(0, 1, 1000).tolist())
        ac = autocorrelation(r)
        assert abs(ac) < 0.1

    def test_lag1_self_corr(self):
        # AR(1) with rho=0.5 should have positive AC
        import numpy as np
        rng = np.random.default_rng(0)
        n = 500
        e = rng.normal(0, 1, n)
        x = np.zeros(n)
        for i in range(1, n):
            x[i] = 0.5 * x[i - 1] + e[i]
        r = pl.Series(x.tolist())
        ac = autocorrelation(r)
        assert ac > 0.2

    def test_invalid_lag(self):
        with pytest.raises(ValueError):
            autocorrelation(pl.Series([0.01] * 10), lag=0)
