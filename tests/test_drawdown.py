"""Tests for ruin.drawdown module."""

from __future__ import annotations

import math

import numpy as np
import polars as pl

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
from tests.conftest import FLOAT32_ABS_TOL, FLOAT32_REL_TOL


class TestDrawdownSeries:
    def test_monotone_up(self) -> None:
        r = pl.Series([0.1, 0.1, 0.1])
        dd = drawdown_series(r)
        assert (dd >= -FLOAT32_ABS_TOL).all()

    def test_single_drop(self) -> None:
        r = pl.Series([0.0, -0.5])
        dd = drawdown_series(r)
        assert math.isclose(dd[-1], -0.5, rel_tol=FLOAT32_REL_TOL)

    def test_sign_convention(self) -> None:
        r = pl.Series([0.1, -0.2])
        dd = drawdown_series(r)
        assert all(v <= FLOAT32_ABS_TOL for v in dd.to_list())

    def test_output_is_float32(self) -> None:
        dd = drawdown_series(pl.Series([0.01, -0.02, 0.03]))
        assert dd.dtype == pl.Float32

    def test_length_matches_input(self) -> None:
        r = pl.Series([0.01, -0.02, 0.03, -0.01])
        dd = drawdown_series(r)
        assert len(dd) == len(r)


class TestMaxDrawdown:
    def test_no_drawdown(self) -> None:
        r = pl.Series([0.01, 0.01, 0.01])
        assert max_drawdown(r) >= -FLOAT32_ABS_TOL

    def test_known_drawdown(self) -> None:
        # 10% up then 50% down: wealth goes 1 -> 1.1 -> 0.55
        # drawdown at trough = 0.55/1.1 - 1 = -0.5
        r = pl.Series([0.1, -0.5])
        assert math.isclose(max_drawdown(r), -0.5, rel_tol=FLOAT32_REL_TOL)

    def test_non_positive(self) -> None:
        rng = np.random.default_rng(0)
        r = pl.Series(rng.normal(0, 0.01, 500).tolist())
        assert max_drawdown(r) <= 0.0

    def test_max_drawdown_bounded_by_minus_one(self) -> None:
        # Wealth cannot go below zero so drawdown cannot exceed -100%.
        rng = np.random.default_rng(0)
        r = pl.Series(rng.normal(-0.05, 0.5, 1000).clip(-0.95, None).tolist())
        assert max_drawdown(r) >= -1.0


class TestAverageDrawdown:
    def test_no_drawdown(self) -> None:
        r = pl.Series([0.01, 0.01, 0.01])
        assert average_drawdown(r) == 0.0

    def test_single_episode(self) -> None:
        # One episode with trough -0.1 — average should equal -0.1
        r = pl.Series([-0.1, 0.2])
        assert math.isclose(average_drawdown(r), -0.1, rel_tol=FLOAT32_REL_TOL)

    def test_two_episodes_averaged(self) -> None:
        # Two symmetric episodes; average of two troughs
        r = pl.Series([-0.1, 0.2, -0.2, 0.3])
        avg = average_drawdown(r)
        assert avg < 0.0


class TestMaxDrawdownDuration:
    def test_no_drawdown(self) -> None:
        r = pl.Series([0.1, 0.1, 0.1])
        assert max_drawdown_duration(r) == 0

    def test_known_duration(self) -> None:
        r = pl.Series([0.1, -0.05, -0.05, -0.05, 0.2])
        assert max_drawdown_duration(r) == 3

    def test_returns_int(self) -> None:
        dur = max_drawdown_duration(pl.Series([0.1, -0.05, 0.2]))
        assert isinstance(dur, int)


class TestRecoveryTime:
    def test_immediate_recovery(self) -> None:
        r = pl.Series([0.1, -0.09, 0.0])
        rt = recovery_time(r)
        assert rt >= 0 or math.isnan(rt)

    def test_no_drawdown_returns_zero(self) -> None:
        r = pl.Series([0.01, 0.01, 0.01])
        assert recovery_time(r) == 0.0

    def test_unrecovered(self) -> None:
        r = pl.Series([0.1, -0.5])
        assert math.isnan(recovery_time(r))


class TestTimeUnderwater:
    def test_all_positive(self) -> None:
        r = pl.Series([0.01, 0.01, 0.01])
        assert time_underwater(r) == 0

    def test_mixed(self) -> None:
        r = pl.Series([0.1, -0.05, -0.05, 0.2])
        assert time_underwater(r) == 2

    def test_returns_int(self) -> None:
        tu = time_underwater(pl.Series([0.1, -0.05]))
        assert isinstance(tu, int)


class TestDrawdownStartEnd:
    def test_peak_and_trough(self) -> None:
        r = pl.Series([0.1, 0.1, -0.5, -0.1])
        start = drawdown_start(r)
        end = drawdown_end(r)
        assert start <= end
        assert end == 3

    def test_ordering_invariant(self) -> None:
        rng = np.random.default_rng(0)
        r = pl.Series(rng.normal(0.0, 0.02, 200).tolist())
        start = drawdown_start(r)
        end = drawdown_end(r)
        assert start <= end
