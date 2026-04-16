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


class TestHitRate:
    def test_all_positive(self) -> None:
        r = pl.Series([0.01, 0.02, 0.03])
        assert hit_rate(r) == 1.0

    def test_half_positive(self) -> None:
        r = pl.Series([0.01, -0.01, 0.02, -0.02])
        assert math.isclose(hit_rate(r), 0.5, rel_tol=1e-9)

    def test_threshold(self) -> None:
        r = pl.Series([0.01, 0.02, 0.005])
        assert math.isclose(hit_rate(r, threshold=0.01), 1 / 3, rel_tol=1e-6)

    def test_all_losses_is_zero(self) -> None:
        r = pl.Series([-0.01, -0.02, -0.03])
        assert hit_rate(r) == 0.0

    def test_empty_raises(self) -> None:
        with pytest.raises(ValueError):
            hit_rate(pl.Series([], dtype=pl.Float64))


class TestAverageWinLoss:
    def test_average_win(self) -> None:
        r = pl.Series([0.02, -0.01, 0.04, -0.02])
        assert math.isclose(average_win(r), 0.03, rel_tol=1e-9)

    def test_average_loss(self) -> None:
        r = pl.Series([0.02, -0.01, 0.04, -0.02])
        assert math.isclose(average_loss(r), -0.015, rel_tol=1e-9)

    def test_no_wins_nan(self) -> None:
        assert math.isnan(average_win(pl.Series([-0.01, -0.02])))

    def test_no_losses_nan(self) -> None:
        assert math.isnan(average_loss(pl.Series([0.01, 0.02])))

    def test_average_loss_is_non_positive(self) -> None:
        r = pl.Series([0.02, -0.01, 0.04, -0.02])
        assert average_loss(r) <= 0.0


class TestWinLossRatio:
    def test_known_value(self) -> None:
        r = pl.Series([0.02, -0.01, 0.04, -0.02])
        assert math.isclose(win_loss_ratio(r), 0.03 / 0.015, rel_tol=1e-9)

    def test_no_wins_is_nan(self) -> None:
        assert math.isnan(win_loss_ratio(pl.Series([-0.01, -0.02])))

    def test_no_losses_is_nan(self) -> None:
        assert math.isnan(win_loss_ratio(pl.Series([0.01, 0.02])))


class TestProfitFactor:
    def test_known_value(self) -> None:
        r = pl.Series([0.05, -0.01, 0.05, -0.01])
        assert math.isclose(profit_factor(r), 5.0, rel_tol=1e-9)

    def test_no_losses_nan(self) -> None:
        assert math.isnan(profit_factor(pl.Series([0.01, 0.02])))

    def test_empty_raises(self) -> None:
        with pytest.raises(ValueError):
            profit_factor(pl.Series([], dtype=pl.Float64))


class TestBestWorstPeriod:
    def test_best(self) -> None:
        assert best_period(pl.Series([0.01, 0.05, -0.02])) == 0.05

    def test_worst(self) -> None:
        assert worst_period(pl.Series([0.01, 0.05, -0.02])) == -0.02

    def test_worst_le_best(self) -> None:
        r = pl.Series([0.01, 0.05, -0.02, 0.03])
        assert worst_period(r) <= best_period(r)


class TestStreaks:
    def test_winning_streak(self) -> None:
        r = pl.Series([0.01, 0.02, -0.01, 0.01, 0.02, 0.03])
        assert longest_winning_streak(r) == 3

    def test_losing_streak(self) -> None:
        r = pl.Series([-0.01, -0.02, 0.01, -0.01, -0.02, -0.03])
        assert longest_losing_streak(r) == 3

    def test_no_wins(self) -> None:
        assert longest_winning_streak(pl.Series([-0.01, -0.02])) == 0

    def test_no_losses(self) -> None:
        assert longest_losing_streak(pl.Series([0.01, 0.02])) == 0

    def test_returns_int(self) -> None:
        r = pl.Series([0.01, 0.02, -0.01])
        assert isinstance(longest_winning_streak(r), int)
        assert isinstance(longest_losing_streak(r), int)
