# Contributing

Thanks for your interest in contributing to ruin!

## Setup

```bash
git clone https://github.com/aexsalomao/ruin
cd ruin
uv sync --extra dev
pre-commit install
```

## Workflow

1. Fork the repo and create a branch: `{name}_fix_{description}` or `{name}_dev_{description}`
2. Make your changes
3. Run the full check suite locally before pushing:

```bash
uv run ruff check .
uv run ruff format .
uv run mypy src/
uv run pytest
```

4. Open a pull request against `master`

CI runs the same checks automatically on every PR.

## Scope (read this first)

ruin is **strictly scoped**. The composition principle — every function returns exactly one thing, with `summary()` the only bundled call — is load-bearing. Before opening an issue or PR, please re-read the *What This Library Is Not* section in [the README](https://github.com/aexsalomao/ruin#what-this-library-is-not).

Out of scope (will not be accepted):

- Money-weighted returns, IRR, fee crystallization
- Portfolio construction, optimization, factor models, Brinson attribution
- Backtesting engines, transaction costs, execution modeling
- Data fetching (no yfinance, no Bloomberg, no CSV readers)
- Time series hygiene (resampling, calendar alignment, gap filling)
- Forecasting (GARCH, EWMA, regime models)
- Plotting
- Pandas support
- Convenience wrappers that bundle multiple metrics

## Code Style

- Ruff for linting and formatting (enforced via pre-commit)
- mypy strict for static type checking — public functions must be fully annotated
- Polars-first; runtime deps are `polars` + `numpy` only (adding any runtime dep needs sign-off)
- Pure functions; no hidden state, no I/O
- See `.claude/rules/code-style.md` and `.claude/rules/testing.md` for the full conventions

## Numerical conventions (locked)

- **Returns in, numbers out.** No prices, no positions.
- **Public Polars outputs are `Float32`; internal math stays in `Float64`.** See `src/ruin/_internal/validate.py`.
- **Drawdowns are non-positive** (`-0.23` = 23% drawdown).
- **VaR / CVaR are positive loss magnitudes** (`0.02` = "lose at most 2%").
- `periods_per_year` and `risk_free` (per-period) are always explicit.

## Tests

Three tiers live under `tests/`:

1. **Unit tests** per module — edge cases, invalid-input branches, dtype assertions.
2. **`test_reference.py`** — hand-computed locked values. Changes here require deliberate review.
3. **`test_properties.py`** — Hypothesis invariants. Investigate failures before suppressing.

`test_benchmarks.py` is regression-only. Use `FLOAT32_REL_TOL` / `FLOAT32_ABS_TOL` from `tests/conftest.py` for `Float32` outputs.

## Future contributions

Areas where help is especially welcome:

- Cornish-Fisher VaR (lands under the `ruin[stats]` extra)
- Additional rolling-metric variants where the path-dependent algorithm is non-obvious
- More reference values in `test_reference.py` cross-checked against reputable sources
- Documentation: worked examples, citation links in `docs/metrics.md`
