"""Tail risk metrics: Value at Risk and Conditional Value at Risk.

Sign convention: VaR and CVaR are **positive loss magnitudes** (desk convention).
A VaR of 0.02 means the portfolio is expected to lose at most 2% with the given
confidence level.

All functions accept ``pl.Series``, ``np.ndarray``, or ``pl.DataFrame``.
NaN values are dropped before computation.
"""

from __future__ import annotations

import polars as pl

from ruin._internal.normal import norm_ppf
from ruin._internal.validate import ReturnInput, require_minimum_length, to_series


def value_at_risk(
    returns: ReturnInput,
    *,
    confidence: float = 0.95,
    method: str = "historical",
) -> float:
    """Value at Risk at a given confidence level.

    Parameters
    ----------
    returns:
        Periodic return series. NaNs are dropped.
    confidence:
        Confidence level in (0, 1). Default 0.95.
    method:
        ``"historical"`` (default): empirical quantile of the return distribution.
        ``"parametric"``: Gaussian approximation using mean and std.

    Returns
    -------
    float
        VaR as a **positive** loss magnitude (desk convention).
        E.g. 0.02 means "lose at most 2% with 95% confidence".

    Notes
    -----
    Historical VaR at confidence=0.95 is the negative of the 5th percentile of returns.
    Parametric VaR assumes normality and may underestimate fat-tailed losses.
    """
    if not 0.0 < confidence < 1.0:
        raise ValueError(f"'confidence' must be in (0, 1); got {confidence}")
    r = to_series(returns)
    require_minimum_length(r, 1, "value_at_risk")

    if method == "historical":
        q = float(r.quantile(1.0 - confidence, interpolation="linear"))  # type: ignore[call-arg]
        return -q
    elif method == "parametric":
        mu = float(r.mean())  # type: ignore[arg-type]
        sigma = float(r.std(ddof=1))  # type: ignore[arg-type]
        z = norm_ppf(1.0 - confidence)
        return -(mu + z * sigma)
    else:
        raise ValueError(f"Unknown method '{method}'; choose 'historical' or 'parametric'.")


def conditional_value_at_risk(
    returns: ReturnInput,
    *,
    confidence: float = 0.95,
    method: str = "historical",
) -> float:
    """Conditional Value at Risk (Expected Shortfall) at a given confidence level.

    Parameters
    ----------
    returns:
        Periodic return series. NaNs are dropped.
    confidence:
        Confidence level in (0, 1). Default 0.95.
    method:
        ``"historical"`` (default): mean of returns below the VaR threshold.
        ``"parametric"``: Gaussian approximation.

    Returns
    -------
    float
        CVaR as a **positive** loss magnitude (desk convention).
        E.g. 0.03 means "average loss in the worst (1 - confidence) scenarios is 3%".

    Notes
    -----
    CVaR is coherent (sub-additive) unlike VaR. For fat-tailed return series,
    historical CVaR is more conservative than the parametric version.
    """
    if not 0.0 < confidence < 1.0:
        raise ValueError(f"'confidence' must be in (0, 1); got {confidence}")
    r = to_series(returns)
    require_minimum_length(r, 1, "conditional_value_at_risk")

    if method == "historical":
        threshold = float(r.quantile(1.0 - confidence, interpolation="linear"))  # type: ignore[call-arg]
        tail = r.filter(r <= threshold)
        if len(tail) == 0:
            return -threshold
        return -float(tail.mean())  # type: ignore[arg-type]
    elif method == "parametric":
        from ruin._internal.normal import norm_pdf

        mu = float(r.mean())  # type: ignore[arg-type]
        sigma = float(r.std(ddof=1))  # type: ignore[arg-type]
        alpha = 1.0 - confidence
        z = norm_ppf(alpha)
        cvar = -(mu - sigma * norm_pdf(z) / alpha)
        return cvar
    else:
        raise ValueError(f"Unknown method '{method}'; choose 'historical' or 'parametric'.")


# Alias
expected_shortfall = conditional_value_at_risk
