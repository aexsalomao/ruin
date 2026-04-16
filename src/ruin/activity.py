"""Trade/period activity metrics.

All functions accept ``pl.Series``, ``np.ndarray``, or ``pl.DataFrame``.
NaN values are dropped before computation.
"""

from __future__ import annotations

from ruin._internal.validate import ReturnInput, require_minimum_length, to_series


def hit_rate(returns: ReturnInput, *, threshold: float = 0.0) -> float:
    """Fraction of periods with return strictly above threshold.

    Parameters
    ----------
    returns:
        Periodic return series. NaNs are dropped.
    threshold:
        Minimum acceptable return. Default 0.0.

    Returns
    -------
    float
        Hit rate in [0, 1].
    """
    r = to_series(returns)
    require_minimum_length(r, 1, "hit_rate")
    return float((r > threshold).mean())  # type: ignore[arg-type]


def average_win(returns: ReturnInput, *, threshold: float = 0.0) -> float:
    """Mean of returns strictly above threshold.

    Parameters
    ----------
    returns:
        Periodic return series. NaNs are dropped.
    threshold:
        Minimum acceptable return. Default 0.0.

    Returns
    -------
    float
        Mean excess return for winning periods. ``float('nan')`` if no wins.
    """
    r = to_series(returns)
    wins = r.filter(r > threshold)
    if len(wins) == 0:
        return float("nan")
    return float(wins.mean())  # type: ignore[arg-type]


def average_loss(returns: ReturnInput, *, threshold: float = 0.0) -> float:
    """Mean of returns strictly below threshold (non-positive result).

    Parameters
    ----------
    returns:
        Periodic return series. NaNs are dropped.
    threshold:
        Minimum acceptable return. Default 0.0.

    Returns
    -------
    float
        Mean return for losing periods (non-positive). ``float('nan')`` if no losses.
    """
    r = to_series(returns)
    losses = r.filter(r < threshold)
    if len(losses) == 0:
        return float("nan")
    return float(losses.mean())  # type: ignore[arg-type]


def win_loss_ratio(returns: ReturnInput, *, threshold: float = 0.0) -> float:
    """Ratio of average win to absolute value of average loss.

    Parameters
    ----------
    returns:
        Periodic return series. NaNs are dropped.
    threshold:
        Minimum acceptable return. Default 0.0.

    Returns
    -------
    float
        average_win / |average_loss|. ``float('nan')`` if no wins or no losses.
    """
    avg_w = average_win(returns, threshold=threshold)
    avg_l = average_loss(returns, threshold=threshold)
    if avg_w != avg_w or avg_l != avg_l:  # NaN check
        return float("nan")
    if avg_l == 0.0:
        return float("nan")
    return avg_w / abs(avg_l)


def profit_factor(returns: ReturnInput, *, threshold: float = 0.0) -> float:
    """Ratio of sum of gains to sum of absolute losses.

    Parameters
    ----------
    returns:
        Periodic return series. NaNs are dropped.
    threshold:
        Minimum acceptable return. Default 0.0.

    Returns
    -------
    float
        sum(r > threshold) / sum(|r < threshold|). ``float('nan')`` if no losses.
    """
    r = to_series(returns)
    require_minimum_length(r, 1, "profit_factor")
    gains = float(r.filter(r > threshold).sum())
    losses = float(abs(r.filter(r < threshold).sum()))
    if losses == 0.0:
        return float("nan")
    return gains / losses


def best_period(returns: ReturnInput) -> float:
    """Maximum single-period return.

    Parameters
    ----------
    returns:
        Periodic return series. NaNs are dropped.

    Returns
    -------
    float
        Maximum return in any single period.
    """
    r = to_series(returns)
    require_minimum_length(r, 1, "best_period")
    return float(r.max())  # type: ignore[arg-type]


def worst_period(returns: ReturnInput) -> float:
    """Minimum single-period return.

    Parameters
    ----------
    returns:
        Periodic return series. NaNs are dropped.

    Returns
    -------
    float
        Minimum return in any single period.
    """
    r = to_series(returns)
    require_minimum_length(r, 1, "worst_period")
    return float(r.min())  # type: ignore[arg-type]


def longest_winning_streak(returns: ReturnInput, *, threshold: float = 0.0) -> int:
    """Longest consecutive run of periods with return strictly above threshold.

    Parameters
    ----------
    returns:
        Periodic return series. NaNs are dropped.
    threshold:
        Minimum acceptable return. Default 0.0.

    Returns
    -------
    int
        Length of the longest consecutive winning run.
    """
    r = to_series(returns)
    max_streak = 0
    current = 0
    for v in r.to_list():
        if v > threshold:
            current += 1
            if current > max_streak:
                max_streak = current
        else:
            current = 0
    return max_streak


def longest_losing_streak(returns: ReturnInput, *, threshold: float = 0.0) -> int:
    """Longest consecutive run of periods with return strictly below threshold.

    Parameters
    ----------
    returns:
        Periodic return series. NaNs are dropped.
    threshold:
        Minimum acceptable return. Default 0.0.

    Returns
    -------
    int
        Length of the longest consecutive losing run.
    """
    r = to_series(returns)
    max_streak = 0
    current = 0
    for v in r.to_list():
        if v < threshold:
            current += 1
            if current > max_streak:
                max_streak = current
        else:
            current = 0
    return max_streak
