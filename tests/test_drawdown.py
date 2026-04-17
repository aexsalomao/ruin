"""Tests for ruin.drawdown module."""

from __future__ import annotations

import math

import numpy as np
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
from tests.conftest import FLOAT32_ABS_TOL, FLOAT32_REL_TOL


def test_drawdown_series_is_non_positive_on_monotone_up_returns() -> None:
    r = pl.Series([0.1, 0.1, 0.1])
    dd = drawdown_series(r)
    assert (dd >= -FLOAT32_ABS_TOL).all()


def test_drawdown_series_reflects_single_drop_exactly() -> None:
    r = pl.Series([0.0, -0.5])
    dd = drawdown_series(r)
    assert dd[-1] == pytest.approx(-0.5, rel=FLOAT32_REL_TOL)


def test_drawdown_series_values_are_always_non_positive() -> None:
    r = pl.Series([0.1, -0.2])
    dd = drawdown_series(r)
    assert all(v <= FLOAT32_ABS_TOL for v in dd.to_list())


def test_drawdown_series_output_is_float32() -> None:
    dd = drawdown_series(pl.Series([0.01, -0.02, 0.03]))
    assert dd.dtype == pl.Float32


def test_drawdown_series_length_matches_input() -> None:
    r = pl.Series([0.01, -0.02, 0.03, -0.01])
    dd = drawdown_series(r)
    assert len(dd) == len(r)


def test_max_drawdown_is_zero_when_no_losses() -> None:
    r = pl.Series([0.01, 0.01, 0.01])
    assert max_drawdown(r) >= -FLOAT32_ABS_TOL


def test_max_drawdown_matches_hand_computed_value() -> None:
    # 10% up then 50% down: wealth 1 -> 1.1 -> 0.55; drawdown = 0.55/1.1 - 1 = -0.5
    r = pl.Series([0.1, -0.5])
    assert max_drawdown(r) == pytest.approx(-0.5, rel=FLOAT32_REL_TOL)


def test_max_drawdown_is_non_positive() -> None:
    rng = np.random.default_rng(0)
    r = pl.Series(rng.normal(0, 0.01, 500).tolist())
    assert max_drawdown(r) <= 0.0


def test_max_drawdown_is_bounded_by_minus_one() -> None:
    rng = np.random.default_rng(0)
    r = pl.Series(rng.normal(-0.05, 0.5, 1000).clip(-0.95, None).tolist())
    assert max_drawdown(r) >= -1.0


def test_average_drawdown_is_zero_when_no_drawdown() -> None:
    r = pl.Series([0.01, 0.01, 0.01])
    assert average_drawdown(r) == 0.0


def test_average_drawdown_equals_single_episode_trough() -> None:
    r = pl.Series([-0.1, 0.2])
    assert average_drawdown(r) == pytest.approx(-0.1, rel=FLOAT32_REL_TOL)


def test_average_drawdown_is_negative_across_multiple_episodes() -> None:
    r = pl.Series([-0.1, 0.2, -0.2, 0.3])
    assert average_drawdown(r) < 0.0


def test_max_drawdown_duration_is_zero_without_drawdown() -> None:
    r = pl.Series([0.1, 0.1, 0.1])
    assert max_drawdown_duration(r) == 0


def test_max_drawdown_duration_matches_known_run_length() -> None:
    r = pl.Series([0.1, -0.05, -0.05, -0.05, 0.2])
    assert max_drawdown_duration(r) == 3


def test_max_drawdown_duration_returns_int() -> None:
    dur = max_drawdown_duration(pl.Series([0.1, -0.05, 0.2]))
    assert isinstance(dur, int)


def test_recovery_time_is_non_negative_or_nan_for_recovered_series() -> None:
    r = pl.Series([0.1, -0.09, 0.0])
    rt = recovery_time(r)
    assert rt >= 0 or math.isnan(rt)


def test_recovery_time_is_zero_without_drawdown() -> None:
    r = pl.Series([0.01, 0.01, 0.01])
    assert recovery_time(r) == 0.0


def test_recovery_time_is_nan_when_unrecovered() -> None:
    r = pl.Series([0.1, -0.5])
    assert math.isnan(recovery_time(r))


def test_time_underwater_is_zero_when_all_returns_positive() -> None:
    r = pl.Series([0.01, 0.01, 0.01])
    assert time_underwater(r) == 0


def test_time_underwater_counts_underwater_periods() -> None:
    r = pl.Series([0.1, -0.05, -0.05, 0.2])
    assert time_underwater(r) == 2


def test_time_underwater_returns_int() -> None:
    tu = time_underwater(pl.Series([0.1, -0.05]))
    assert isinstance(tu, int)


def test_drawdown_start_precedes_drawdown_end() -> None:
    r = pl.Series([0.1, 0.1, -0.5, -0.1])
    start = drawdown_start(r)
    end = drawdown_end(r)
    assert start <= end
    assert end == 3


def test_drawdown_start_is_always_before_or_equal_to_end() -> None:
    rng = np.random.default_rng(0)
    r = pl.Series(rng.normal(0.0, 0.02, 200).tolist())
    start = drawdown_start(r)
    end = drawdown_end(r)
    assert start <= end
