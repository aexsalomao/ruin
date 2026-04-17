"""Tail risk: VaR and CVaR. Returned as positive loss magnitudes (0.02 = "lose up to 2%")."""

from __future__ import annotations

from ruin._internal.normal import norm_pdf, norm_ppf
from ruin._internal.validate import ReturnInput, require_minimum_length, to_series


def value_at_risk(
    returns: ReturnInput,
    *,
    confidence: float = 0.95,
    method: str = "historical",
) -> float:
    """Value at Risk at `confidence` in (0, 1), returned as a positive loss magnitude.

    `"historical"` (default): negative of the (1 - confidence) empirical quantile.
    `"parametric"`: Gaussian approximation using sample mean/std; may underestimate fat tails.
    """
    if not 0.0 < confidence < 1.0:
        raise ValueError(f"'confidence' must be in (0, 1); got {confidence}")
    r = to_series(returns)
    require_minimum_length(r, 1, "value_at_risk")

    if method == "historical":
        q = float(r.quantile(1.0 - confidence, interpolation="linear"))
        return -q
    elif method == "parametric":
        mu = float(r.mean())
        sigma = float(r.std(ddof=1))
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
    """CVaR (Expected Shortfall) at `confidence`, as a positive loss magnitude.

    `"historical"`: mean of returns at or below the VaR threshold.
    `"parametric"`: Gaussian approximation. Unlike VaR, CVaR is coherent (sub-additive).
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
        return -float(tail.mean())
    elif method == "parametric":
        mu = float(r.mean())
        sigma = float(r.std(ddof=1))
        alpha = 1.0 - confidence
        z = norm_ppf(alpha)
        cvar = -(mu - sigma * norm_pdf(z) / alpha)
        return cvar
    else:
        raise ValueError(f"Unknown method '{method}'; choose 'historical' or 'parametric'.")


# Alias
expected_shortfall = conditional_value_at_risk
