# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

## [0.1.0] - 2026-04-17

### Added
- Initial release.
- `ruin.returns` — `from_prices`, `total_return`, `annualize_return`, `cagr`.
- `ruin.volatility` — `volatility`, `annualize_volatility`, `downside_deviation`, `semi_deviation`.
- `ruin.drawdown` — `drawdown_series`, `max_drawdown`, `average_drawdown`, `max_drawdown_duration`, `recovery_time`, `time_underwater`, `drawdown_start`, `drawdown_end`.
- `ruin.ratios` — `sharpe_ratio`, `sortino_ratio`, `calmar_ratio`, `information_ratio`, `treynor_ratio`, `omega_ratio`.
- `ruin.tail` — `value_at_risk`, `conditional_value_at_risk` / `expected_shortfall`.
- `ruin.market` — `beta`, `downside_beta`, `upside_beta`, `alpha`, `tracking_error`, `correlation`, `up_capture`, `down_capture`.
- `ruin.distribution` — `skewness`, `excess_kurtosis`, `jarque_bera`, `autocorrelation`.
- `ruin.activity` — `hit_rate`, `average_win`, `average_loss`, `win_loss_ratio`, `profit_factor`, `best_period`, `worst_period`, `longest_winning_streak`, `longest_losing_streak`.
- `ruin.rolling` — rolling variants of the major metrics, all returning length-aligned `Float32` `pl.Series`.
- `ruin.periods` — `mtd`, `qtd`, `ytd`, `trailing`, `since_inception`, `periods_per_year_for`, `annual_to_periodic`, `periodic_to_annual`.
- `ruin.inference` — `sharpe_standard_error`, `sharpe_confidence_interval`, `bootstrap_metric`.
- `ruin.report` — `summary` (the one bundled function).
- Documentation site at [aexsalomao.github.io/ruin](https://aexsalomao.github.io/ruin).
- Hand-computed reference tests, Hypothesis property tests, and `pytest-benchmark` regression suite.
- Ruff linting + formatting, mypy strict type checking, pre-commit hooks.
- GitHub Actions CI workflow (lint → format → typecheck → pytest with coverage).
- Automated PyPI publish workflow via GitHub Actions Trusted Publishing (OIDC).
