"""Property-based tests using hypothesis.

Invariants that must hold for any valid input. Failure here flags a conceptual
bug — the property should be recomputed before disabling the test.
"""

from __future__ import annotations

import math

import polars as pl
from hypothesis import given, settings
from hypothesis import strategies as st


def float_list(
    min_size: int = 10,
    max_size: int = 100,
    min_val: float = -0.5,
    max_val: float = 0.5,
) -> st.SearchStrategy[list[float]]:
    return st.lists(
        st.floats(
            min_value=min_val,
            max_value=max_val,
            allow_nan=False,
            allow_infinity=False,
        ),
        min_size=min_size,
        max_size=max_size,
    )


@given(returns=float_list(), scale=st.floats(min_value=0.01, max_value=10.0, allow_nan=False))
@settings(max_examples=100)
def test_sharpe_scale_invariant(returns: list[float], scale: float) -> None:
    """Sharpe ratio is invariant to positive scaling of returns."""
    from ruin.ratios import sharpe_ratio

    r = pl.Series(returns)
    sr1 = sharpe_ratio(r, periods_per_year=252)
    sr2 = sharpe_ratio(r * scale, periods_per_year=252)
    if math.isnan(sr1) or math.isnan(sr2):
        return
    assert math.isclose(sr1, sr2, rel_tol=1e-6)


@given(returns=float_list())
@settings(max_examples=100)
def test_max_drawdown_non_positive(returns: list[float]) -> None:
    """Max drawdown is always in [-1, 0]."""
    from ruin.drawdown import max_drawdown

    mdd = max_drawdown(pl.Series(returns))
    assert mdd <= 0.0
    assert mdd >= -1.0


@given(returns=float_list(min_val=-0.2, max_val=0.2))
@settings(max_examples=50)
def test_annualize_return_roundtrip(returns: list[float]) -> None:
    """Geometric annualization followed by de-annualization recovers the total return."""
    from ruin.returns import annualize_return, total_return

    r = pl.Series(returns)
    n = len(r)
    periods_per_year = 12
    ann = annualize_return(r, periods_per_year=periods_per_year)
    if math.isnan(ann):
        return
    reconstructed = (1 + ann) ** (n / periods_per_year) - 1
    assert math.isclose(reconstructed, total_return(r), rel_tol=1e-6)


@given(returns=float_list(min_size=60))
@settings(max_examples=50)
def test_rolling_sharpe_last_row_equals_scalar(returns: list[float]) -> None:
    """rolling_sharpe with window == len(returns) equals scalar sharpe_ratio at the last row."""
    from ruin.ratios import sharpe_ratio
    from ruin.rolling import rolling_sharpe

    r = pl.Series(returns)
    rs = rolling_sharpe(r, window=len(r), periods_per_year=252)
    scalar_sr = sharpe_ratio(r, periods_per_year=252)
    last = rs[-1]
    if math.isnan(scalar_sr) or last is None:
        return
    # Float32 precision for the rolling output.
    assert math.isclose(float(last), scalar_sr, rel_tol=1e-4)


@given(returns=float_list(min_size=20))
@settings(max_examples=100)
def test_omega_ratio_gt_one_iff_positive_mean(returns: list[float]) -> None:
    """omega_ratio > 1 iff mean(r) > 0 (for threshold=0)."""
    from ruin.ratios import omega_ratio

    r = pl.Series(returns)
    om = omega_ratio(r, threshold=0.0)
    mean_r = float(r.mean())
    if math.isnan(om):
        return
    if mean_r > 0:
        assert om > 1.0 - 1e-10
    elif mean_r < 0:
        assert om < 1.0 + 1e-10


@given(
    returns=float_list(),
    constant=st.floats(min_value=-0.05, max_value=0.05, allow_nan=False),
)
@settings(max_examples=50)
def test_hit_rate_invariant_to_common_shift(returns: list[float], constant: float) -> None:
    """Shifting both returns and threshold by the same constant preserves hit_rate."""
    from ruin.activity import hit_rate

    r = pl.Series(returns)
    threshold = 0.01
    hr1 = hit_rate(r, threshold=threshold)
    hr2 = hit_rate(r + constant, threshold=threshold + constant)
    assert math.isclose(hr1, hr2, abs_tol=1e-9)


@given(returns=float_list(min_size=10))
@settings(max_examples=100)
def test_volatility_non_negative(returns: list[float]) -> None:
    """Volatility is always non-negative."""
    from ruin.volatility import volatility

    assert volatility(pl.Series(returns)) >= 0.0


@given(returns=float_list(min_size=5))
@settings(max_examples=50)
def test_best_period_ge_worst_period(returns: list[float]) -> None:
    from ruin.activity import best_period, worst_period

    r = pl.Series(returns)
    assert best_period(r) >= worst_period(r)


@given(returns=float_list(min_size=30))
@settings(max_examples=50)
def test_historical_cvar_ge_var(returns: list[float]) -> None:
    """CVaR at a given confidence level is at least VaR at the same level."""
    from ruin.tail import conditional_value_at_risk, value_at_risk

    r = pl.Series(returns)
    var = value_at_risk(r, confidence=0.95)
    cvar = conditional_value_at_risk(r, confidence=0.95)
    assert cvar >= var - 1e-10
