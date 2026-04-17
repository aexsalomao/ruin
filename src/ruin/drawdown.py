"""Drawdown metrics. Drawdowns are non-positive (`-0.23` = 23% drawdown). NaNs dropped."""

from __future__ import annotations

import polars as pl

from ruin._internal.validate import (
    FLOAT_DTYPE,
    ReturnInput,
    require_minimum_length,
    to_series,
)


def drawdown_series(returns: ReturnInput) -> pl.Series:
    """Drawdown at each period vs. prior high-water mark: `wealth[t] / cum_max(wealth) - 1`."""
    r = to_series(returns)
    require_minimum_length(r, 1, "drawdown_series")
    # Prepend initial wealth of 1.0 so the HWM starts at the investment date,
    # making first-period losses visible as drawdowns.
    wealth = pl.concat([pl.Series([1.0], dtype=r.dtype), (1.0 + r).cum_prod()])
    hwm = wealth.cum_max()
    dd = wealth / hwm - 1.0
    return dd.slice(1).rename("drawdown").cast(FLOAT_DTYPE)


def max_drawdown(returns: ReturnInput) -> float:
    """Maximum drawdown (non-positive fraction, e.g. -0.23 = 23%)."""
    dd = drawdown_series(returns)
    return float(dd.min())  # type: ignore[arg-type]


def average_drawdown(returns: ReturnInput) -> float:
    """Mean trough magnitude across distinct drawdown episodes. New episode at each new HWM."""
    dd = drawdown_series(returns)
    if len(dd) == 0:
        return 0.0

    # Identify episode troughs: dd goes from 0 -> negative -> recovers to 0
    # We scan for contiguous underwater segments and collect their minimum
    episodes: list[float] = []
    current_min = 0.0
    in_drawdown = False

    for v in dd.to_list():
        if v < 0.0:
            in_drawdown = True
            if v < current_min:
                current_min = v
        else:
            if in_drawdown:
                episodes.append(current_min)
                current_min = 0.0
                in_drawdown = False

    # Last episode may still be open
    if in_drawdown:
        episodes.append(current_min)

    if not episodes:
        return 0.0
    return sum(episodes) / len(episodes)


def max_drawdown_duration(returns: ReturnInput) -> int:
    """Longest consecutive run of periods spent underwater (in periods)."""
    dd = drawdown_series(returns)
    max_dur = 0
    current = 0
    for v in dd.to_list():
        if v < 0.0:
            current += 1
            if current > max_dur:
                max_dur = current
        else:
            current = 0
    return max_dur


def recovery_time(returns: ReturnInput) -> float:
    """Periods from the max-drawdown trough to the next new HWM. NaN if unrecovered."""
    r = to_series(returns)
    require_minimum_length(r, 1, "recovery_time")
    dd = drawdown_series(r)
    dd_list = dd.to_list()

    # Find index of max drawdown trough
    min_val = min(dd_list)
    if min_val >= 0.0:
        return 0.0

    trough_idx = dd_list.index(min_val)

    # Find the first subsequent index where dd == 0
    for i in range(trough_idx + 1, len(dd_list)):
        if dd_list[i] >= 0.0:
            return float(i - trough_idx)

    return float("nan")


def time_underwater(returns: ReturnInput) -> int:
    """Total number of periods spent below the HWM."""
    dd = drawdown_series(returns)
    return int((dd < 0.0).sum())


def drawdown_start(returns: ReturnInput) -> int:
    """0-based index of the HWM peak immediately preceding the maximum drawdown."""
    r = to_series(returns)
    require_minimum_length(r, 1, "drawdown_start")
    dd = drawdown_series(r)
    dd_list = dd.to_list()

    min_val = min(dd_list)
    if min_val >= 0.0:
        return 0

    trough_idx = dd_list.index(min_val)

    # Walk back from trough to find the last zero (HWM) before it
    for i in range(trough_idx, -1, -1):
        if dd_list[i] >= 0.0:
            return i

    return 0


def drawdown_end(returns: ReturnInput) -> int:
    """0-based index of the max-drawdown trough."""
    r = to_series(returns)
    require_minimum_length(r, 1, "drawdown_end")
    dd = drawdown_series(r)
    dd_list = dd.to_list()

    min_val = min(dd_list)
    return dd_list.index(min_val)
