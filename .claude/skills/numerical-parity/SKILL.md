---
name: numerical-parity
description: Cross-check a ruin metric's output against a reference implementation (empyrical, quantstats, or an Excel/hand calculation) before claiming the metric is correct. Use when adding a new metric, fixing a numerical bug, or refactoring the math in returns/volatility/drawdown/ratios/tail/market/distribution/activity.
---

# Numerical Parity Check

ruin is a library other people will trust to report risk numbers. "My implementation returns 0.742" is not evidence. "My implementation returns 0.742 and so does empyrical and my hand calc" is.

## When to invoke

- Adding a new metric (see `metric-add` skill).
- Fixing a numerical bug — before closing the fix, prove the new value is right.
- Refactoring: if the math touched changed shape (e.g., moving from a loop to a vectorized Polars expression), re-check parity.
- Reviewing a PR that changes a metric's computation.

## Reference sources (pick the best available)

In priority order:

1. **Closed-form hand calculation** on a short, deterministic series (3–5 observations). Best ground truth — no dependency, no hidden conventions.
2. **Spreadsheet** — Excel / Google Sheets formulas, saved as a test fixture with the expected numbers. Useful for Sharpe, drawdown, VaR.
3. **`empyrical` / `empyrical-reloaded`** — the canonical Python reference for most performance metrics. Add to dev deps if needed (`uv add --dev empyrical-reloaded`).
4. **`quantstats`** — broader coverage; less strict on conventions. Use when `empyrical` doesn't cover the metric.
5. **Academic source** — for metrics defined in a paper (Omega, Jensen's alpha, Jarque-Bera), cite the paper and reproduce its numerical example if one exists.

Never use "another random library" without checking its conventions.

## Convention traps (these cause most "parity failures")

- **Biased vs. unbiased variance.** `ddof=0` vs `ddof=1`. Polars' `std()` defaults to `ddof=1`; numpy's `std()` defaults to `ddof=0`.
- **Simple vs. log returns.** Many academic references use log returns; most practitioner libs use simple.
- **Arithmetic vs. geometric annualization.** `mean * N` vs `(1+mean)^N - 1`.
- **`periods_per_year`.** 252 (trading days), 260 (weekdays), 365 (calendar days), 12 (months), 4 (quarters). Always document ruin's default.
- **Sample vs. population** anything — skewness, kurtosis.
- **Excess vs. total kurtosis.** `empyrical` reports excess (subtract 3); some refs don't.
- **Drawdown sign.** ruin returns drawdowns as negative numbers (losses). Some libs return absolute values.
- **VaR sign.** 95% VaR is reported as a negative return in ruin; as a positive loss in some libs.
- **Risk-free rate units.** Annualized vs. per-period. Almost always the source of Sharpe mismatches.

If parity fails, walk this list **before** changing the implementation.

## How to run the check

1. Pick a deterministic test series (seed a `hypothesis` strategy, or hard-code a 10–50 point series).
2. Compute the metric in ruin.
3. Compute the reference value. For `empyrical`:
   ```python
   import empyrical
   ref = empyrical.sharpe_ratio(returns_np, period='daily')
   ```
   Note the `period` kwarg maps to `periods_per_year`.
4. Assert `pytest.approx(ruin_val, rel=1e-6) == ref_val`. Tighten the tolerance as much as the metric allows; loosen only with a comment explaining why.
5. Pin the reference value in a parametrized test — don't call the reference library at test time unless strictly necessary (flakiness, dep weight). Store the value as a constant with a comment saying where it came from.

## When they disagree

Do not "fix" ruin to match the reference until you understand the gap:

1. Identify which convention the reference uses.
2. Decide whether ruin's convention is deliberate or a bug.
3. If deliberate, add a docstring note + test comment documenting the divergence and *why*.
4. If a bug, fix the implementation and add a regression test with the reference value.

A "fix" that silently flips a convention will break every downstream user's dashboards.

## What not to do

- Don't use `==` on floats. `pytest.approx` always.
- Don't assert against a value you just got by running ruin — that's a snapshot test of today's implementation, not a parity test.
- Don't skip the hand calculation on trivial inputs. The 3-point series catches more bugs than the 10000-point random series.
- Don't silently swallow a mismatch "because the numbers are close." 1e-3 divergence on Sharpe is a real bug, not rounding.
