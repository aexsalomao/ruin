"""Trade/period activity metrics. NaNs dropped."""

from __future__ import annotations

from ruin._internal.validate import ReturnInput, require_minimum_length, to_series


def hit_rate(returns: ReturnInput, *, threshold: float = 0.0) -> float:
    """Fraction of periods with `return > threshold`, in [0, 1]."""
    r = to_series(returns)
    require_minimum_length(r, 1, "hit_rate")
    return float((r > threshold).mean())


def average_win(returns: ReturnInput, *, threshold: float = 0.0) -> float:
    """Mean of returns strictly above `threshold`. NaN if no wins."""
    r = to_series(returns)
    wins = r.filter(r > threshold)
    if len(wins) == 0:
        return float("nan")
    return float(wins.mean())


def average_loss(returns: ReturnInput, *, threshold: float = 0.0) -> float:
    """Mean of returns strictly below `threshold` (non-positive). NaN if no losses."""
    r = to_series(returns)
    losses = r.filter(r < threshold)
    if len(losses) == 0:
        return float("nan")
    return float(losses.mean())


def win_loss_ratio(returns: ReturnInput, *, threshold: float = 0.0) -> float:
    """`average_win / |average_loss|`. NaN if no wins or no losses."""
    avg_w = average_win(returns, threshold=threshold)
    avg_l = average_loss(returns, threshold=threshold)
    if avg_w != avg_w or avg_l != avg_l:  # NaN check
        return float("nan")
    if avg_l == 0.0:
        return float("nan")
    return avg_w / abs(avg_l)


def profit_factor(returns: ReturnInput, *, threshold: float = 0.0) -> float:
    """`sum(gains) / sum(|losses|)` relative to `threshold`. NaN if no losses."""
    r = to_series(returns)
    require_minimum_length(r, 1, "profit_factor")
    gains = float(r.filter(r > threshold).sum())
    losses = float(abs(r.filter(r < threshold).sum()))
    if losses == 0.0:
        return float("nan")
    return gains / losses


def best_period(returns: ReturnInput) -> float:
    """Maximum single-period return."""
    r = to_series(returns)
    require_minimum_length(r, 1, "best_period")
    return float(r.max())


def worst_period(returns: ReturnInput) -> float:
    """Minimum single-period return."""
    r = to_series(returns)
    require_minimum_length(r, 1, "worst_period")
    return float(r.min())


def longest_winning_streak(returns: ReturnInput, *, threshold: float = 0.0) -> int:
    """Longest consecutive run with `r > threshold`."""
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
    """Longest consecutive run with `r < threshold`."""
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
