"""Tests for ruin.activity module."""

from __future__ import annotations

import math

import polars as pl
import pytest

from ruin.activity import (
    average_loss,
    average_win,
    best_period,
    hit_rate,
    longest_losing_streak,
    longest_winning_streak,
    profit_factor,
    win_loss_ratio,
    worst_period,
)


def test_hit_rate_is_one_when_all_returns_positive() -> None:
    r = pl.Series([0.01, 0.02, 0.03])
    assert hit_rate(r) == 1.0


def test_hit_rate_is_half_for_balanced_returns() -> None:
    r = pl.Series([0.01, -0.01, 0.02, -0.02])
    assert hit_rate(r) == pytest.approx(0.5, rel=1e-9)


def test_hit_rate_applies_custom_threshold() -> None:
    r = pl.Series([0.01, 0.02, 0.005])
    assert hit_rate(r, threshold=0.01) == pytest.approx(1 / 3, rel=1e-6)


def test_hit_rate_is_zero_when_all_losses() -> None:
    r = pl.Series([-0.01, -0.02, -0.03])
    assert hit_rate(r) == 0.0


def test_hit_rate_rejects_empty_series() -> None:
    with pytest.raises(ValueError):
        hit_rate(pl.Series([], dtype=pl.Float64))


def test_average_win_matches_hand_computed_value() -> None:
    r = pl.Series([0.02, -0.01, 0.04, -0.02])
    assert average_win(r) == pytest.approx(0.03, rel=1e-9)


def test_average_loss_matches_hand_computed_value() -> None:
    r = pl.Series([0.02, -0.01, 0.04, -0.02])
    assert average_loss(r) == pytest.approx(-0.015, rel=1e-9)


def test_average_win_is_nan_when_no_wins() -> None:
    assert math.isnan(average_win(pl.Series([-0.01, -0.02])))


def test_average_loss_is_nan_when_no_losses() -> None:
    assert math.isnan(average_loss(pl.Series([0.01, 0.02])))


def test_average_loss_is_non_positive() -> None:
    r = pl.Series([0.02, -0.01, 0.04, -0.02])
    assert average_loss(r) <= 0.0


def test_win_loss_ratio_matches_hand_computed_value() -> None:
    r = pl.Series([0.02, -0.01, 0.04, -0.02])
    assert win_loss_ratio(r) == pytest.approx(0.03 / 0.015, rel=1e-9)


def test_win_loss_ratio_is_nan_when_no_wins() -> None:
    assert math.isnan(win_loss_ratio(pl.Series([-0.01, -0.02])))


def test_win_loss_ratio_is_nan_when_no_losses() -> None:
    assert math.isnan(win_loss_ratio(pl.Series([0.01, 0.02])))


def test_profit_factor_matches_hand_computed_value() -> None:
    r = pl.Series([0.05, -0.01, 0.05, -0.01])
    assert profit_factor(r) == pytest.approx(5.0, rel=1e-9)


def test_profit_factor_is_nan_when_no_losses() -> None:
    assert math.isnan(profit_factor(pl.Series([0.01, 0.02])))


def test_profit_factor_rejects_empty_series() -> None:
    with pytest.raises(ValueError):
        profit_factor(pl.Series([], dtype=pl.Float64))


def test_best_period_returns_max_return() -> None:
    assert best_period(pl.Series([0.01, 0.05, -0.02])) == 0.05


def test_worst_period_returns_min_return() -> None:
    assert worst_period(pl.Series([0.01, 0.05, -0.02])) == -0.02


def test_worst_period_is_less_than_or_equal_to_best() -> None:
    r = pl.Series([0.01, 0.05, -0.02, 0.03])
    assert worst_period(r) <= best_period(r)


def test_longest_winning_streak_counts_consecutive_wins() -> None:
    r = pl.Series([0.01, 0.02, -0.01, 0.01, 0.02, 0.03])
    assert longest_winning_streak(r) == 3


def test_longest_losing_streak_counts_consecutive_losses() -> None:
    r = pl.Series([-0.01, -0.02, 0.01, -0.01, -0.02, -0.03])
    assert longest_losing_streak(r) == 3


def test_longest_winning_streak_is_zero_without_wins() -> None:
    assert longest_winning_streak(pl.Series([-0.01, -0.02])) == 0


def test_longest_losing_streak_is_zero_without_losses() -> None:
    assert longest_losing_streak(pl.Series([0.01, 0.02])) == 0


def test_streak_counts_return_ints() -> None:
    r = pl.Series([0.01, 0.02, -0.01])
    assert isinstance(longest_winning_streak(r), int)
    assert isinstance(longest_losing_streak(r), int)
