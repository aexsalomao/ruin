"""Property-based tests using hypothesis.

Invariants that must hold for any valid input.
"""

from __future__ import annotations

import math

import polars as pl
import pytest
from hypothesis import assume, given, settings
from hypothesis import strategies as st


def float_list(min_size: int = 10, max_size: int = 100, min_val: float = -0.5, max_val: float = 0.5):
    return st.lists(
        st.floats(min_value=min_val, max_value=max_val, allow_nan=False, allow_infinity=False),
        min_size=min_size,
        max_size=max_size,
    )


@given(returns=float_list(), scale=st.floats(min_value=0.01, max_value=10.0, allow_nan=False))
@settings(max_examples=100)
def test_sharpe_scale_invariant(returns, scale):
    """Sharpe ratio is invariant to scaling returns by a positive constant."""
    from ruin.ratios import sharpe_ratio

    r = pl.Series(returns)
    sr1 = sharpe_ratio(r, periods_per_year=252)
    sr2 = sharpe_ratio(r * scale, periods_per_year=252)
    if math.isnan(sr1) or math.isnan(sr2):
        return
    assert math.isclose(sr1, sr2, rel_tol=1e-6)


@given(returns=float_list())
@settings(max_examples=100)
def test_max_drawdown_non_positive(returns):
    """Max drawdown is always in [-1, 0]."""
    from ruin.drawdown import max_drawdown

    r = pl.Series(returns)
    mdd = max_drawdown(r)
    assert mdd <= 0.0
    assert mdd >= -1.0


@given(returns=float_list(min_val=-0.2, max_val=0.2))
@settings(max_examples=50)
def test_annualize_return_roundtrip(returns):
    """Geometric annualization then back-conversion returns original total return."""
    from ruin.returns import annualize_return, total_return

    r = pl.Series(returns)
    n = len(r)
    periods_per_year = 12
    ann = annualize_return(r, periods_per_year=periods_per_year)
    # Reconstruct total: (1 + ann_return)^(n/ppy) - 1
    reconstructed = (1 + ann) ** (n / periods_per_year) - 1
    original_tr = total_return(r)
    assert math.isclose(reconstructed, original_tr, rel_tol=1e-6)


@given(returns=float_list(min_size=60))
@settings(max_examples=50)
def test_rolling_sharpe_last_row_equals_scalar(returns):
    """rolling_sharpe with window == len(returns) equals scalar sharpe_ratio on last row."""
    from ruin.ratios import sharpe_ratio
    from ruin.rolling import rolling_sharpe

    r = pl.Series(returns)
    window = len(r)
    rs = rolling_sharpe(r, window=window, periods_per_year=252)
    scalar_sr = sharpe_ratio(r, periods_per_year=252)
    last = rs[-1]
    if math.isnan(scalar_sr) or last is None:
        return
    assert math.isclose(float(last), scalar_sr, rel_tol=1e-4)


@given(returns=float_list(min_size=20))
@settings(max_examples=100)
def test_omega_ratio_gt_one_iff_positive_mean(returns):
    """omega_ratio > 1 iff mean(r) > 0 (for threshold=0)."""
    from ruin.ratios import omega_ratio

    r = pl.Series(returns)
    om = omega_ratio(r, threshold=0.0)
    mean_r = float(r.mean())
    if math.isnan(om):
        return  # no losses, inconclusive
    if mean_r > 0:
        assert om > 1.0 - 1e-10
    elif mean_r < 0:
        assert om < 1.0 + 1e-10


@given(
    returns=float_list(),
    constant=st.floats(min_value=-0.05, max_value=0.05, allow_nan=False),
)
@settings(max_examples=50)
def test_hit_rate_shifts_with_threshold(returns, constant):
    """Adding a constant to all returns and shifting threshold leaves hit_rate unchanged."""
    from ruin.activity import hit_rate

    r = pl.Series(returns)
    threshold = 0.01
    hr1 = hit_rate(r, threshold=threshold)
    hr2 = hit_rate(r + constant, threshold=threshold + constant)
    assert math.isclose(hr1, hr2, abs_tol=1e-9)
