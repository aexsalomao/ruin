"""Drawdown metrics.

Sign convention: drawdowns are non-positive floats. A 23% drawdown is ``-0.23``.
All functions accept ``pl.Series``, ``np.ndarray``, or ``pl.DataFrame``.
NaN values are dropped before computation.
"""

from __future__ import annotations

import math

import polars as pl

from ruin._internal.validate import ReturnInput, require_minimum_length, to_series


def drawdown_series(returns: ReturnInput) -> pl.Series:
    """Compute the drawdown at each period relative to the prior high-water mark.

    Parameters
    ----------
    returns:
        Periodic return series. NaNs are dropped.

    Returns
    -------
    pl.Series
        Drawdown series aligned to input (after NaN removal), values in (-inf, 0].
        Each value is ``wealth[t] / max(wealth[0..t]) - 1``.
    """
    r = to_series(returns)
    require_minimum_length(r, 1, "drawdown_series")
    # Prepend initial wealth of 1.0 so the HWM starts at the investment date,
    # making first-period losses visible as drawdowns.
    wealth = pl.concat([pl.Series([1.0]), (1.0 + r).cum_prod()])
    hwm = wealth.cum_max()
    dd = wealth / hwm - 1.0
    return dd.slice(1).rename("drawdown")


def max_drawdown(returns: ReturnInput) -> float:
    """Maximum drawdown over the full return series.

    Parameters
    ----------
    returns:
        Periodic return series. NaNs are dropped.

    Returns
    -------
    float
        Maximum drawdown as a non-positive fraction (e.g. -0.23 = 23% drawdown).
    """
    dd = drawdown_series(returns)
    return float(dd.min())  # type: ignore[arg-type]


def average_drawdown(returns: ReturnInput) -> float:
    """Mean magnitude of distinct drawdown episodes.

    A new episode begins each time the portfolio sets a new high-water mark.
    The magnitude is taken as the trough value within each episode.

    Parameters
    ----------
    returns:
        Periodic return series. NaNs are dropped.

    Returns
    -------
    float
        Average trough drawdown across episodes (non-positive).
    """
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
    """Longest consecutive run of periods spent underwater (below HWM).

    Parameters
    ----------
    returns:
        Periodic return series. NaNs are dropped.

    Returns
    -------
    int
        Length of the longest drawdown episode in periods.
    """
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
    """Periods from the max drawdown trough to the next new high-water mark.

    Parameters
    ----------
    returns:
        Periodic return series. NaNs are dropped.

    Returns
    -------
    float
        Number of periods from trough to recovery. ``float('nan')`` if the
        portfolio has not recovered by the end of the series.
    """
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
    """Total number of periods spent below the high-water mark.

    Parameters
    ----------
    returns:
        Periodic return series. NaNs are dropped.

    Returns
    -------
    int
        Count of periods where drawdown < 0.
    """
    dd = drawdown_series(returns)
    return int((dd < 0.0).sum())


def drawdown_start(returns: ReturnInput) -> int:
    """Index of the peak immediately preceding the maximum drawdown.

    Parameters
    ----------
    returns:
        Periodic return series. NaNs are dropped.

    Returns
    -------
    int
        0-based index of the HWM peak before the deepest drawdown.
    """
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
    """Index of the trough of the maximum drawdown.

    Parameters
    ----------
    returns:
        Periodic return series. NaNs are dropped.

    Returns
    -------
    int
        0-based index of the deepest drawdown point.
    """
    r = to_series(returns)
    require_minimum_length(r, 1, "drawdown_end")
    dd = drawdown_series(r)
    dd_list = dd.to_list()

    min_val = min(dd_list)
    return dd_list.index(min_val)
