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
    def test_all_positive(self):
        r = pl.Series([0.01, 0.02, 0.03])
        assert hit_rate(r) == 1.0

    def test_half_positive(self):
        r = pl.Series([0.01, -0.01, 0.02, -0.02])
        assert math.isclose(hit_rate(r), 0.5, rel_tol=1e-9)

    def test_threshold(self):
        r = pl.Series([0.01, 0.02, 0.005])
        hr = hit_rate(r, threshold=0.01)
        assert math.isclose(hr, 1 / 3, rel_tol=1e-6)


class TestAverageWinLoss:
    def test_average_win(self):
        r = pl.Series([0.02, -0.01, 0.04, -0.02])
        aw = average_win(r)
        assert math.isclose(aw, 0.03, rel_tol=1e-9)

    def test_average_loss(self):
        r = pl.Series([0.02, -0.01, 0.04, -0.02])
        al = average_loss(r)
        assert math.isclose(al, -0.015, rel_tol=1e-9)

    def test_no_wins_nan(self):
        r = pl.Series([-0.01, -0.02])
        assert math.isnan(average_win(r))

    def test_no_losses_nan(self):
        r = pl.Series([0.01, 0.02])
        assert math.isnan(average_loss(r))


class TestWinLossRatio:
    def test_known_value(self):
        r = pl.Series([0.02, -0.01, 0.04, -0.02])
        wlr = win_loss_ratio(r)
        assert math.isclose(wlr, 0.03 / 0.015, rel_tol=1e-9)


class TestProfitFactor:
    def test_known_value(self):
        r = pl.Series([0.05, -0.01, 0.05, -0.01])
        pf = profit_factor(r)
        assert math.isclose(pf, 5.0, rel_tol=1e-9)

    def test_no_losses_nan(self):
        r = pl.Series([0.01, 0.02])
        assert math.isnan(profit_factor(r))


class TestBestWorstPeriod:
    def test_best(self):
        r = pl.Series([0.01, 0.05, -0.02])
        assert best_period(r) == 0.05

    def test_worst(self):
        r = pl.Series([0.01, 0.05, -0.02])
        assert worst_period(r) == -0.02


class TestStreaks:
    def test_winning_streak(self):
        r = pl.Series([0.01, 0.02, -0.01, 0.01, 0.02, 0.03])
        assert longest_winning_streak(r) == 3

    def test_losing_streak(self):
        r = pl.Series([-0.01, -0.02, 0.01, -0.01, -0.02, -0.03])
        assert longest_losing_streak(r) == 3

    def test_no_wins(self):
        r = pl.Series([-0.01, -0.02])
        assert longest_winning_streak(r) == 0
