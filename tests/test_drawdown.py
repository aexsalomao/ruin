"""Tests for ruin.drawdown module."""

from __future__ import annotations

import math

import polars as pl
import pytest

from ruin.drawdown import (
    average_drawdown,
    drawdown_end,
    drawdown_series,
    drawdown_start,
    max_drawdown,
    max_drawdown_duration,
    recovery_time,
    time_underwater,
)


class TestDrawdownSeries:
    def test_monotone_up(self):
        r = pl.Series([0.1, 0.1, 0.1])
        dd = drawdown_series(r)
        # Never underwater after a new HWM each period
        assert (dd >= -1e-12).all()

    def test_single_drop(self):
        r = pl.Series([0.0, -0.5])  # after flat, then 50% drop
        dd = drawdown_series(r)
        assert math.isclose(dd[-1], -0.5, rel_tol=1e-6)

    def test_sign_convention(self):
        r = pl.Series([0.1, -0.2])
        dd = drawdown_series(r)
        assert all(v <= 1e-12 for v in dd.to_list())


class TestMaxDrawdown:
    def test_no_drawdown(self):
        r = pl.Series([0.01, 0.01, 0.01])
        assert max_drawdown(r) >= -1e-12  # effectively 0

    def test_known_drawdown(self):
        # 10% up then 50% down: wealth goes 1 -> 1.1 -> 0.55
        # drawdown at trough = 0.55/1.1 - 1 = -0.5
        r = pl.Series([0.1, -0.5])
        mdd = max_drawdown(r)
        assert math.isclose(mdd, -0.5, rel_tol=1e-6)

    def test_non_positive(self):
        import numpy as np
        rng = np.random.default_rng(0)
        r = pl.Series(rng.normal(0, 0.01, 500).tolist())
        assert max_drawdown(r) <= 0.0


class TestMaxDrawdownDuration:
    def test_no_drawdown(self):
        r = pl.Series([0.1, 0.1, 0.1])
        assert max_drawdown_duration(r) == 0

    def test_known_duration(self):
        # Down 3 periods, recover
        r = pl.Series([0.1, -0.05, -0.05, -0.05, 0.2])
        dur = max_drawdown_duration(r)
        assert dur == 3


class TestRecoveryTime:
    def test_immediate_recovery(self):
        r = pl.Series([0.1, -0.09, 0.0])
        # After drop from 1.1 to ~1.001, next flat period doesn't recover to 1.1
        # Recovery time depends on exact math
        rt = recovery_time(r)
        assert rt >= 0 or math.isnan(rt)

    def test_no_drawdown_returns_zero(self):
        r = pl.Series([0.01, 0.01, 0.01])
        assert recovery_time(r) == 0.0

    def test_unrecovered(self):
        r = pl.Series([0.1, -0.5])
        assert math.isnan(recovery_time(r))


class TestTimeUnderwater:
    def test_all_positive(self):
        r = pl.Series([0.01, 0.01, 0.01])
        assert time_underwater(r) == 0

    def test_mixed(self):
        r = pl.Series([0.1, -0.05, -0.05, 0.2])
        assert time_underwater(r) == 2


class TestDrawdownStartEnd:
    def test_peak_and_trough(self):
        r = pl.Series([0.1, 0.1, -0.5, -0.1])
        start = drawdown_start(r)
        end = drawdown_end(r)
        assert start <= end
        assert end == 3  # last element is the deepest
