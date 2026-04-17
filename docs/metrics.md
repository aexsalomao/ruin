# Metrics Reference

Authoritative definitions for every metric exposed by `ruin`, paired with the
assumptions baked into each formula and the academic / industry sources that
motivate them.

The library follows the conventions set out in
[`docs/conventions.md`](conventions.md):

- Returns are dimensionless fractions (`0.01` = 1%).
- Drawdowns are non-positive (`-0.23` = 23% drawdown).
- VaR / CVaR are positive loss magnitudes (`0.02` = "lose at most 2%").
- `risk_free` is **per-period**, never annualized.
- NaN / null are dropped before computation; benchmark inputs must be
  pre-aligned in length.

Symbols used throughout this document:

- `r_t` — return in period `t` (a fraction)
- `n` — number of return observations after NaN drop
- `q` — `periods_per_year` (annualization factor; e.g. 252 for daily)
- `r_f` — per-period risk-free rate
- `b_t` — benchmark return in period `t`
- `1{·}` — indicator function

---

## 1. Returns (`ruin.returns`)

### `from_prices(prices, *, log=False)`

Convert a positive price series of length `N` into `N − 1` period returns.

- **Simple** (default): `r_t = P_t / P_{t-1} − 1`
- **Log**: `r_t = ln(P_t / P_{t-1})`

**Source.** Standard textbook definitions; e.g. Tsay, *Analysis of Financial
Time Series*, 3rd ed. (Wiley, 2010), §1.1.

**Assumptions.**

- Prices are strictly positive (dividends folded in via total-return prices, if
  desired). The function raises if any price ≤ 0.
- Equal time-spacing between observations — annualization downstream depends on
  it.
- Log returns are time-additive but not portfolio-additive; simple returns are
  portfolio-additive but not time-additive.

---

### `total_return(returns)`

Cumulative compounded return over the sample.

$$
R_{\text{tot}} = \prod_{t=1}^{n} (1 + r_t) - 1
$$

**Source.** Bodie, Kane & Marcus, *Investments*, 12th ed. (McGraw-Hill, 2021),
ch. 5.

**Assumptions.**

- Returns reinvest at the realized rate each period (compounding assumption).
- Path-dependent: a single trade quote at the boundaries can dominate the
  result. Carries no risk information.

---

### `annualize_return(returns, *, periods_per_year, method="geometric")`

Scale the realized return to a one-year horizon.

- **Geometric** (default):
  $\,R_{\text{ann}} = (1 + R_{\text{tot}})^{q / n} - 1$. Returns `NaN` when
  `R_tot ≤ −1` (ruin), avoiding fractional powers of negative numbers.
- **Arithmetic**: $\,R_{\text{ann}} = \bar r \cdot q$, with
  $\bar r = \tfrac{1}{n} \sum_t r_t$.

**Source.** CFA Institute, *Quantitative Investment Analysis*, 4th ed. (Wiley,
2020), ch. 1; Meucci, *Risk and Asset Allocation* (Springer, 2005), §3.2.

**Assumptions.**

- The realized period is representative of a typical year (clearly false for
  short series).
- Geometric assumes terminal-wealth equivalence under continued compounding.
- Arithmetic assumes i.i.d. returns and no compounding effects; it overstates
  expected terminal wealth for volatile strategies.

---

### `cagr(returns, *, periods_per_year)`

Alias for geometric `annualize_return`.

**Source.** Same as above. Industry term, no canonical academic origin.

---

## 2. Volatility (`ruin.volatility`)

### `volatility(returns, *, ddof=1)`

Periodic sample standard deviation:

$$
\sigma = \sqrt{\frac{1}{n - \text{ddof}} \sum_{t=1}^{n} (r_t - \bar r)^2}
$$

**Source.** Markowitz, "Portfolio Selection," *Journal of Finance* 7(1), 1952.

**Assumptions.**

- Returns are i.i.d. and second moment exists.
- Symmetric loss function: an upside move of size `x` contributes the same as a
  downside move of size `x` — questionable for skewed strategies.

---

### `annualize_volatility(returns, *, periods_per_year, ddof=1)`

$\sigma_{\text{ann}} = \sigma \cdot \sqrt{q}$ — the **square-root-of-time
rule**.

**Source.** Diebold, Hickman, Inoue & Schuermann, "Converting 1-day Volatility
to h-day Volatility: Scaling by √h is Worse than You Think," Wharton FIC
Working Paper 97-34, 1997.

**Assumptions.**

- Returns are i.i.d. (zero serial correlation).
- The relation breaks down under autocorrelation. The corrected formula is
  $\sigma_{\text{ann}} = \sigma \sqrt{q \cdot (1 + 2 \sum_{k=1}^{K} \rho_k)}$
  (see Lo (2002) below); use `autocorrelation` to diagnose before trusting the
  annualized value.

---

### `downside_deviation(returns, *, threshold=0.0, ddof=0)`

Sortino-convention semi-deviation: only periods with `r_t < threshold`
contribute to the numerator, but **all** `n` periods are in the denominator.

$$
\text{DD} = \sqrt{\frac{1}{n - \text{ddof}} \sum_{t=1}^{n} \min(r_t - \tau, 0)^2}
$$

**Source.** Sortino & van der Meer, "Downside Risk," *Journal of Portfolio
Management* 17(4), 1991. Sortino & Price, "Performance Measurement in a
Downside Risk Framework," *Journal of Investing* 3(3), 1994.

**Assumptions.**

- A **minimum acceptable return** (MAR) `τ` is the relevant reference point —
  not the mean.
- Investors care asymmetrically about losses below `τ`.
- Default `ddof=0` matches the Sortino convention; use `ddof=1` for
  sample-style.

---

### `semi_deviation(returns, *, ddof=0)`

Standard deviation computed only over strictly negative returns (both numerator
and denominator restricted). Returns 0 when there are no negative returns.

$$
\text{SD}_{-} = \sqrt{\frac{1}{n_{-} - \text{ddof}} \sum_{t : r_t < 0} (r_t - \bar r_{-})^2}
$$

where $n_{-}$ is the count of negative observations and $\bar r_{-}$ their
mean.

**Source.** Markowitz, *Portfolio Selection: Efficient Diversification of
Investments* (Wiley, 1959), ch. IX (semivariance).

**Assumptions.**

- Behaviour around zero matters more than around the mean — a stronger
  assumption than Sortino's.
- Sensitive to small samples once the negative-return count is small.

---

## 3. Drawdown (`ruin.drawdown`)

Common quantity used by all functions in this module:

$$
W_t = \prod_{s \le t} (1 + r_s), \quad
H_t = \max_{s \le t} W_s, \quad
D_t = W_t / H_t - 1 \;(\le 0).
$$

The implementation prepends an initial wealth of 1.0 so first-period losses are
visible as drawdowns.

### `drawdown_series(returns)`

Length-`n` Polars Series of `D_t`.

**Source.** Magdon-Ismail & Atiya, "Maximum Drawdown," *Risk Magazine* 17(10),
2004.

### `max_drawdown(returns)`

$D^{\star} = \min_t D_t$ (non-positive).

**Source.** Same as above; widely used since at least Grossman & Zhou,
"Optimal Investment Strategies for Controlling Drawdowns," *Mathematical
Finance* 3(3), 1993.

**Assumptions.** Path-dependent and sample-dependent — daily vs. month-end
data produce different numbers; short histories systematically understate the
worst drawdown.

### `average_drawdown(returns)`

Mean trough magnitude across distinct drawdown **episodes**, where an episode
is a contiguous run of `D_t < 0` separated by points where `D_t ≥ 0`. Returns 0
if there are no drawdowns.

**Source.** Industry convention; distinct from "average drawdown across all
underwater periods" used elsewhere. Compatible with the formulation in
Chekhlov, Uryasev & Zabarankin, "Drawdown Measure in Portfolio Optimization,"
*International Journal of Theoretical and Applied Finance* 8(1), 2005.

**Assumptions.** "Episode" is a definitional choice; alternatives (peak-to-peak
clustering, threshold filters) yield different numbers.

### `max_drawdown_duration(returns)`

Longest consecutive count of periods spent underwater (`D_t < 0`), in periods.

**Source.** Magdon-Ismail & Atiya (2004); see also Burghardt, Duncan & Liu,
"Deciphering Drawdowns," *Risk Magazine* 16(9), 2003.

### `recovery_time(returns)`

Number of periods from the maximum-drawdown trough to the first subsequent
return to a new HWM. Returns `NaN` if the strategy has not recovered by the end
of the sample (intentional — pretending recovery occurred would be dishonest).

**Source.** Burghardt, Duncan & Liu (2003).

### `time_underwater(returns)`

Total count of periods with `D_t < 0`.

### `drawdown_start(returns)` / `drawdown_end(returns)`

Zero-based index of the HWM peak immediately preceding the maximum drawdown,
and the index of the trough itself.

**Assumptions for the four index/duration metrics.** Equally-spaced periods;
the units are *periods*, not calendar days. Convert externally if you need
calendar-day durations.

---

## 4. Risk-Adjusted Ratios (`ruin.ratios`)

### `sharpe_ratio(returns, *, risk_free=0.0, periods_per_year, ddof=1)`

Annualized excess return per unit of annualized volatility:

$$
\text{SR} = \frac{(\bar r - r_f) \cdot q}{\sigma_{\text{excess}} \cdot \sqrt{q}}
= \frac{(\bar r - r_f)}{\sigma_{\text{excess}}} \sqrt{q}
$$

where $\sigma_{\text{excess}}$ is the sample std (`ddof=1`) of the excess
return series.

**Source.** Sharpe, "Mutual Fund Performance," *Journal of Business* 39(1),
1966; Sharpe, "The Sharpe Ratio," *Journal of Portfolio Management* 21(1),
1994.

**Assumptions.**

- Returns are i.i.d. and approximately normal — used to justify the
  square-root-of-time annualization.
- Symmetric loss function (penalises upside variability equally).
- Stable when `σ` is large relative to numerical noise; returns `NaN` when
  annualized volatility is zero.

---

### `sortino_ratio(returns, *, risk_free=0.0, threshold=None, periods_per_year)`

Annualized excess return per unit of annualized downside deviation. `threshold`
defaults to `risk_free`; downside deviation uses `ddof=0` and the
all-periods-in-denominator (Sortino) convention.

$$
\text{Sortino} = \frac{(\bar r - r_f) \cdot q}{\text{DD}_\tau \cdot \sqrt{q}}
$$

**Source.** Sortino & Price (1994); Sortino, *The Sortino Framework for
Constructing Portfolios* (Elsevier, 2010).

**Assumptions.**

- Investors care about returns below the MAR `τ`, not below the mean.
- Square-root-of-time annualization of downside deviation — same i.i.d.
  caveats as Sharpe.

---

### `calmar_ratio(returns, *, periods_per_year)`

$$
\text{Calmar} = \frac{\text{CAGR}}{|D^{\star}|}
$$

Returns `NaN` if the maximum drawdown is exactly zero.

**Source.** Young, "Calmar Ratio: A Smoother Tool," *Futures Magazine*, 1991.

**Assumptions.**

- Max drawdown is a meaningful risk measure for the strategy (path-dependent
  and small-sample-fragile).
- Originally defined on a 36-month rolling window; the implementation here
  uses the full sample. Short histories with no large drawdowns produce
  artificially high Calmar.

---

### `information_ratio(returns, benchmark, *, periods_per_year, ddof=1)`

Annualized active return per unit of annualized tracking error:

$$
\text{IR} = \frac{\overline{(r_t - b_t)} \cdot q}{\sigma_{r-b} \cdot \sqrt{q}}
$$

**Source.** Goodwin, "The Information Ratio," *Financial Analysts Journal*
54(4), 1998; Grinold & Kahn, *Active Portfolio Management*, 2nd ed.
(McGraw-Hill, 2000), ch. 5.

**Assumptions.**

- Active returns are i.i.d. (square-root-of-time again).
- Returns and benchmark are pre-aligned in length and time.

---

### `treynor_ratio(returns, benchmark, *, risk_free=0.0, periods_per_year)`

$$
\text{Treynor} = \frac{(\bar r - r_f) \cdot q}{\beta}
$$

**Source.** Treynor, "How to Rate Management of Investment Funds," *Harvard
Business Review* 43(1), 1965.

**Assumptions.**

- CAPM-style framework: market beta captures all systematic risk.
- Poorly defined when `β ≈ 0` or negative; returns `NaN` if `β = 0` exactly.
- Most informative for diversified, long-only portfolios.

---

### `omega_ratio(returns, *, threshold=0.0)`

Probability-weighted ratio of gains to losses around a threshold `τ`:

$$
\Omega(\tau) = \frac{\sum_t \max(r_t - \tau, 0)}{\sum_t \max(\tau - r_t, 0)}
$$

`Ω > 1` iff `mean(r) > τ` for any distribution with finite mean. At `τ = 0` it
coincides with the profit factor and the gain-to-pain ratio.

**Source.** Keating & Shadwick, "A Universal Performance Measure," *Journal of
Performance Measurement* 6(3), 2002.

**Assumptions.**

- Threshold choice is critical — using `r_f` per-period is more economically
  meaningful than `τ = 0`.
- Returns `NaN` when no observations fall strictly below `τ` (denominator
  zero).

---

## 5. Tail Risk (`ruin.tail`)

### `value_at_risk(returns, *, confidence=0.95, method="historical")`

The smallest loss `L` (positive number) such that
`P(loss ≤ L) ≥ confidence`.

- **Historical**: `−Q_{1−c}(r)` — the negative of the empirical
  `(1 − confidence)` quantile, with linear interpolation.
- **Parametric** (Gaussian): `−(μ + Φ⁻¹(1 − c) · σ)` using the sample mean and
  `ddof=1` sample std.

**Source.** Jorion, *Value at Risk: The New Benchmark for Managing Financial
Risk*, 3rd ed. (McGraw-Hill, 2007).

**Assumptions.**

- *Historical*: requires large samples; insensitive to tail shape beyond the
  cutoff.
- *Parametric*: returns are Normally distributed — systematically
  underestimates losses for fat-tailed distributions.
- Not coherent (fails sub-additivity in general) — see Artzner et al.,
  "Coherent Measures of Risk," *Mathematical Finance* 9(3), 1999.

---

### `conditional_value_at_risk(returns, *, confidence=0.95, method="historical")`

(Aliased as `expected_shortfall`.) Expected loss conditional on being in the
`(1 − c)` left tail.

- **Historical**: mean of returns `≤ Q_{1−c}(r)` (returned with sign flipped).
  If no observations meet that condition the function returns `−Q_{1−c}(r)`.
- **Parametric** (Gaussian): `−(μ − σ · φ(z) / α)` with `z = Φ⁻¹(α)`,
  `α = 1 − c`, where `φ` is the standard-Normal PDF.

**Source.** Rockafellar & Uryasev, "Optimization of Conditional Value-at-Risk,"
*Journal of Risk* 2(3), 2000; Acerbi & Tasche, "On the Coherence of Expected
Shortfall," *Journal of Banking & Finance* 26(7), 2002.

**Assumptions.**

- Coherent (sub-additive) — preferred over VaR for portfolio aggregation.
- *Parametric* version inherits the Normality assumption.
- Sample-size sensitive; consider bootstrapping with `bootstrap_metric`.

---

## 6. Market / Benchmark-Relative (`ruin.market`)

### `beta(returns, benchmark)`

OLS slope of `r_t` on `b_t`:

$$
\beta = \frac{\text{cov}(r, b)}{\text{var}(b)}
$$

with `ddof=1` throughout. Returns `NaN` when `var(b) = 0`.

**Source.** Sharpe, "Capital Asset Prices," *Journal of Finance* 19(3), 1964.

**Assumptions.**

- Linear relationship between strategy and benchmark.
- Stationarity — rolling beta is more informative than full-sample for live
  monitoring.

### `downside_beta(returns, benchmark)` / `upside_beta(returns, benchmark)`

Same definition as `beta` but conditioned on `b_t < 0` and `b_t > 0`
respectively. Both return `NaN` when fewer than 2 conditioning observations or
zero conditional variance.

**Source.** Bawa & Lindenberg, "Capital Market Equilibrium in a Mean-Lower
Partial Moment Framework," *Journal of Financial Economics* 5(2), 1977; Ang,
Chen & Xing, "Downside Risk," *Review of Financial Studies* 19(4), 2006.

### `alpha(returns, benchmark, *, risk_free=0.0, periods_per_year)`

Annualized Jensen's alpha:

$$
\alpha = q \cdot \overline{(r_t - r_f)} - \beta \cdot q \cdot \overline{(b_t - r_f)}
$$

**Source.** Jensen, "The Performance of Mutual Funds in the Period 1945–1964,"
*Journal of Finance* 23(2), 1968.

**Assumptions.**

- CAPM holds (a strong empirical assumption).
- Treat as a performance decomposition rather than a causal attribution.

### `tracking_error(returns, benchmark, *, periods_per_year, ddof=1)`

Annualized standard deviation of active returns:
$\sigma_{r-b} \cdot \sqrt{q}$.

**Source.** Roll, "A Mean/Variance Analysis of Tracking Error," *Journal of
Portfolio Management* 18(4), 1992.

**Assumptions.** Active returns are i.i.d.

### `correlation(returns, benchmark)`

Pearson correlation $\rho \in [-1, 1]$.

**Source.** Pearson, "Notes on the History of Correlation," *Biometrika* 13,
1920 (historical reference).

### `up_capture(returns, benchmark)` / `down_capture(returns, benchmark)`

Geometric capture ratios over benchmark up/down periods:

$$
\text{Up} = \frac{\prod_{t : b_t > 0}(1 + r_t) - 1}{\prod_{t : b_t > 0}(1 + b_t) - 1}
$$

(analogous formula for down-capture over `b_t < 0`).

**Source.** Morningstar, *Morningstar Methodology Paper: Upside/Downside
Capture Ratio*, 2011.

**Assumptions.**

- "Up market" = single-period benchmark > 0; alternative methodologies use
  monthly or quarterly aggregation.
- Returns `NaN` when there are no qualifying periods or the benchmark
  compounded return is exactly zero.

---

## 7. Distribution Shape (`ruin.distribution`)

Standardized residuals: $z_t = (r_t - \bar r) / \sigma$, with $\sigma$ the
sample std (`ddof=1`). All moment-based metrics return `NaN` when the input is
constant (`σ = 0` or `n_unique ≤ 1`).

### `skewness(returns, *, bias=False)`

- **Unbiased** (default, SAS / SPSS / Excel SKEW):
  $\,\hat{S} = \frac{n^2}{(n-1)(n-2)} \cdot \tfrac{1}{n} \sum_t z_t^3$.
- **Biased** (population): $\,\tfrac{1}{n} \sum_t z_t^3$.

**Source.** Joanes & Gill, "Comparing Measures of Sample Skewness and
Kurtosis," *Journal of the Royal Statistical Society Series D* 47(1), 1998.

### `excess_kurtosis(returns, *, bias=False)`

- **Unbiased** (default, Excel KURT):
  $\hat K = \frac{n(n+1)}{(n-1)(n-2)(n-3)} \sum_t z_t^4 - \frac{3(n-1)^2}{(n-2)(n-3)}$
- **Biased**: $\,\tfrac{1}{n}\sum_t z_t^4 - 3$.

**Source.** Joanes & Gill (1998).

**Assumptions for skewness / kurtosis.** Reliable only with reasonable sample
sizes (n > 100 for kurtosis is a useful rule of thumb); independence of
observations is assumed for the bias correction.

### `jarque_bera(returns)`

Returns a frozen `JarqueBeraResult(statistic, p_value)`:

$$
\text{JB} = \frac{n}{6}\left(S^2 + \frac{K^2}{4}\right)
$$

where `S` and `K` are the **biased** skewness and excess kurtosis. `p_value`
uses the exact χ²(2) survival function `exp(−JB / 2)` — no SciPy dependency.

**Source.** Jarque & Bera, "Efficient Tests for Normality, Homoscedasticity and
Serial Independence of Regression Residuals," *Economics Letters* 6(3), 1980.

**Assumptions.** Asymptotic — unreliable for n ≲ 100. Failing to reject does
not prove normality.

### `autocorrelation(returns, *, lag=1)`

$$
\rho_k = \frac{\text{cov}(r_t, r_{t-k})}{\sigma(r_t) \cdot \sigma(r_{t-k})}
$$

Implemented as the Pearson correlation between `r[lag:]` and `r[:n-lag]`.

**Source.** Box, Jenkins, Reinsel & Ljung, *Time Series Analysis: Forecasting
and Control*, 5th ed. (Wiley, 2015), ch. 2.

**Assumptions.** Stationarity. Positive lag-1 autocorrelation is a key
diagnostic for smoothed / illiquid pricing and invalidates the
square-root-of-time rule (see `annualize_volatility`).

---

## 8. Activity (`ruin.activity`)

All thresholds default to 0; pass the per-period `risk_free` for an
economically grounded reference.

### `hit_rate(returns, *, threshold=0.0)`

$\,\tfrac{1}{n} \sum_t 1\{r_t > \tau\}$, in `[0, 1]`.

### `average_win(returns, *, threshold=0.0)`

Mean of returns strictly above `τ`; `NaN` if there are none.

### `average_loss(returns, *, threshold=0.0)`

Mean of returns strictly below `τ` (typically non-positive); `NaN` if there are
none.

### `win_loss_ratio(returns, *, threshold=0.0)`

`average_win / |average_loss|`. `NaN` if either side is `NaN` or the
denominator is zero.

### `profit_factor(returns, *, threshold=0.0)`

$$
\text{PF} = \frac{\sum_{t : r_t > \tau} r_t}{\left|\sum_{t : r_t < \tau} r_t\right|}
$$

`NaN` when the denominator is zero.

**Source.** Pardo, *The Evaluation and Optimization of Trading Strategies*, 2nd
ed. (Wiley, 2008), ch. 12.

### `best_period(returns)` / `worst_period(returns)`

`max(r)` and `min(r)` respectively.

### `longest_winning_streak(returns, *, threshold=0.0)` / `longest_losing_streak(returns, *, threshold=0.0)`

Longest consecutive run with `r_t > τ` (`r_t < τ` for losing). Periods exactly
equal to `τ` break the streak.

**Assumptions for the activity module.**

- Strict inequality breaks streaks at `τ` — a deliberate convention; an
  observation precisely at the threshold is neither a win nor a loss.
- Statistics are sample-frequency-dependent (daily streaks ≠ monthly streaks).

---

## 9. Rolling Variants (`ruin.rolling`)

Every `rolling_*` function returns a Polars Series cast to **Float32**,
length-aligned to the input, with the leading `window − 1` values null.
Internally everything runs in Float64.

| Function | Underlying scalar metric | Window type |
|---|---|---|
| `rolling_volatility` | `volatility` | int or duration string |
| `rolling_downside_deviation` | `downside_deviation` | int or duration string |
| `rolling_sharpe` | `sharpe_ratio` | int or duration string |
| `rolling_sortino` | `sortino_ratio` | int or duration string |
| `rolling_beta` | `beta` | int only |
| `rolling_correlation` | `correlation` | int only |
| `rolling_tracking_error` | `tracking_error` | int or duration string |
| `rolling_alpha` | `alpha` | int or duration string |
| `rolling_skewness` | `skewness` (unbiased) | int only |
| `rolling_excess_kurtosis` | `excess_kurtosis` (unbiased) | int only |
| `rolling_autocorrelation` | `autocorrelation` | int only |
| `rolling_max_drawdown` | `max_drawdown` | int only |
| `rolling_hit_rate` | `hit_rate` | int or duration string |
| `rolling_profit_factor` | `profit_factor` | int only |

**Conventions.** `min_periods` defaults to `window` for integer windows. NaNs
inside a window are *not* dropped for the simple Polars-native rolls
(`rolling_volatility`, etc.); for path-dependent metrics implemented through
`_window_apply` (skew, kurtosis, max drawdown, profit factor,
autocorrelation), each window's NaN/null entries are dropped before the metric
runs and a window with fewer than `min_periods` valid observations emits
`null`.

**Source.** Standard rolling-window construction; see Tsay (2010) §2.7.

**Assumptions.** Within-window stationarity. Each window is a plug-in estimator
of the underlying metric — small windows produce noisy estimates of higher
moments.

---

## 10. Period Slicing & Rate Conversion (`ruin.periods`)

These are not statistical metrics, but they are the supported plumbing for
metric pipelines.

### `mtd / qtd / ytd(returns, *, date_col, as_of=None)`

Filter a DataFrame (or a date Series) to month-to-date, quarter-to-date, or
year-to-date relative to `as_of` (defaults to `datetime.date.today()`).

### `trailing(returns, *, n, date_col=None)`

Last `n` rows. Pass a sorted DataFrame; `date_col` is accepted for API
symmetry but unused. Raises if `n ≤ 0`.

### `since_inception(returns)`

Identity — provided for API symmetry.

### `periods_per_year_for(frequency)`

Conventional periods/year mapping: `D=252, W=52, M=12, Q=4, A=1, Y=1`. Raises
on unknown frequency.

**Source.** CFA Institute *Quantitative Investment Analysis* §2.

### `annual_to_periodic(rate, *, periods_per_year)`

$\,(1 + \text{rate})^{1/q} - 1$.

### `periodic_to_annual(rate, *, periods_per_year)`

$\,(1 + \text{rate})^{q} - 1$.

**Assumptions.** Geometric compounding; constant rate across all periods.

---

## 11. Inference (`ruin.inference`)

### `sharpe_standard_error(returns, *, periods_per_year)`

Lo (2002) autocorrelation-adjusted standard error of the **annualized** Sharpe
ratio:

$$
\text{SE}(\widehat{\text{SR}}_{\text{ann}}) \approx \sqrt{\frac{1 + 2 \rho_1 \cdot \widehat{\text{SR}}_q^2}{n}} \cdot \sqrt{q}
$$

where $\widehat{\text{SR}}_q = \bar r / \sigma$ is the per-period Sharpe and
$\rho_1$ is the lag-1 autocorrelation. Reduces to $\sqrt{q/n}$ under i.i.d.
returns.

**Source.** Lo, "The Statistics of Sharpe Ratios," *Financial Analysts Journal*
58(4), 2002.

**Assumptions.**

- Returns are stationary.
- Adjusts for first-order autocorrelation only — higher-order serial
  dependence still biases the estimator.

### `sharpe_confidence_interval(returns, *, periods_per_year, confidence=0.95)`

Asymptotic Wald interval $\widehat{\text{SR}} \pm z \cdot \text{SE}$ with
$z = \Phi^{-1}\!\left(\tfrac{1+c}{2}\right)$.

**Source.** Lo (2002).

**Assumptions.** Asymptotic Normality of the Sharpe estimator — best with
n ≥ ~60 observations.

### `bootstrap_metric(fn, returns, *, n_samples=1000, confidence=0.95, seed=None)`

Resampled-with-replacement bootstrap. Computes `fn(returns)` as the point
estimate, then resamples `n_samples` times to build percentile-method
confidence bounds.

**Source.** Efron & Tibshirani, *An Introduction to the Bootstrap* (Chapman &
Hall, 1993), ch. 13 (percentile method).

**Assumptions.**

- Observations are i.i.d. — i.i.d. resampling destroys serial dependence; for
  autocorrelated returns use a block bootstrap (not provided).
- `fn` accepts a `pl.Series`.
- Resamples that raise `ValueError` or `ZeroDivisionError` are silently
  skipped; if all fail, the bounds are returned as `NaN`.

---

## 12. Bundled Report (`ruin.report`)

### `summary(returns, benchmark=None, *, risk_free=0.0, periods_per_year, strict=False)`

Computes every scalar metric above for one return stream (or one row per
column when given a DataFrame). Benchmark-relative columns are `null` when no
benchmark is supplied. `strict=True` raises on NaN/null input instead of
dropping. Float64 output columns are downcast to Float32 for the public
`pl.DataFrame` return type, in line with the dtype policy in `.claude/CLAUDE.md`.

This is the **only** bundled function in `ruin` — everything else is a single
metric in, single number out.

---

## Definition ↔ Implementation Audit

The following table verifies that the formula stated above matches the actual
behaviour in source. Each row records: where the formula lives in this doc,
where the code lives, and any caveat that materially alters the textbook
formula.

| Metric | Source location | Match? | Notes |
|---|---|---|---|
| `from_prices` | `returns.py:16` | ✅ | Drops the leading null produced by `shift(1)` and any NaN; raises if any price ≤ 0. |
| `total_return` | `returns.py:26` | ✅ | Direct `(1 + r).product() − 1`. |
| `annualize_return` (geo) | `returns.py:33` | ✅ + edge case | Returns `NaN` when `total_return ≤ −1`, avoiding fractional powers of negatives — documented behaviour. |
| `annualize_return` (arith) | `returns.py:55` | ✅ | `mean(r) * q`. |
| `cagr` | `returns.py:60` | ✅ | Pure alias. |
| `volatility` | `volatility.py:8` | ✅ | `Series.std(ddof=ddof)`. |
| `annualize_volatility` | `volatility.py:15` | ✅ | `volatility · √q`. Validates `q > 0`. |
| `downside_deviation` | `volatility.py:27` | ✅ | All-periods denominator with `ddof=0`; matches Sortino convention. |
| `semi_deviation` | `volatility.py:48` | ✅ | Std restricted to negative returns (both numerator and denominator); returns 0 when there are none, as documented. |
| `drawdown_series` | `drawdown.py:15` | ✅ | Prepends initial wealth = 1 so first-period losses register as drawdowns. Documented in this doc + the source's docstring. |
| `max_drawdown` | `drawdown.py:27` | ✅ | `min(drawdown_series)`. |
| `average_drawdown` | `drawdown.py:33` | ✅ | Episode = contiguous underwater run, separated by `dd ≥ 0`; matches the doc. |
| `max_drawdown_duration` | `drawdown.py:61` | ✅ | Longest count of consecutive `dd < 0` periods. |
| `recovery_time` | `drawdown.py:76` | ✅ | First subsequent index with `dd ≥ 0` after the trough; `NaN` if unrecovered, `0.0` if no drawdown ever occurred. |
| `time_underwater` | `drawdown.py:98` | ✅ | Count of `dd < 0` periods. |
| `drawdown_start` | `drawdown.py:104` | ✅ | Walks back from trough to last `dd ≥ 0`. Returns `0` if no drawdown. |
| `drawdown_end` | `drawdown.py:125` | ✅ | Index of the trough (first occurrence of `min`). |
| `sharpe_ratio` | `ratios.py:17` | ✅ | Annualized; `NaN` when annualized vol = 0. |
| `sortino_ratio` | `ratios.py:35` | ✅ | Uses `downside_deviation` with `ddof=0` and the all-periods denominator. Threshold defaults to `risk_free`. |
| `calmar_ratio` | `ratios.py:57` | ✅ | `CAGR / |max_drawdown|`. The classical Calmar uses a 36-month window — this implementation uses the full sample (caveat noted above). |
| `information_ratio` | `ratios.py:68` | ✅ | Uses `align_benchmark`; annualized. |
| `treynor_ratio` | `ratios.py:85` | ✅ | `NaN` when `β = 0`; uses `align_benchmark`. |
| `omega_ratio` | `ratios.py:101` | ✅ | Sum-of-gains / sum-of-losses; `NaN` if no downside. |
| `value_at_risk` (hist) | `tail.py:9` | ✅ | `−Q_{1−c}` via `Series.quantile(..., interpolation="linear")`. |
| `value_at_risk` (param) | `tail.py:28` | ✅ | `−(μ + Φ⁻¹(α)·σ)` with sample `ddof=1`. |
| `conditional_value_at_risk` (hist) | `tail.py:53` | ✅ + edge case | Falls back to `−Q_{1−c}` if no observation lies at or below the quantile (documented above). |
| `conditional_value_at_risk` (param) | `tail.py:59` | ✅ | Closed-form Gaussian ES via local `norm_pdf` / `norm_ppf`. |
| `beta` | `market.py:10` | ✅ | Manual `cov / var` with `ddof=1`. |
| `downside_beta` | `market.py:30` | ✅ | Conditioned on `b < 0`; needs ≥ 2 observations. |
| `upside_beta` | `market.py:45` | ✅ | Conditioned on `b > 0`; needs ≥ 2 observations. |
| `alpha` | `market.py:60` | ✅ | Annualized Jensen's alpha using `align_benchmark`. |
| `tracking_error` | `market.py:77` | ✅ | Annualized std of active returns. |
| `correlation` | `market.py:90` | ✅ | Polars `pl.corr` over the aligned pair. |
| `up_capture` | `market.py:98` | ✅ | Geometric on `b > 0` periods; `NaN` on empty subset or zero benchmark. |
| `down_capture` | `market.py:113` | ✅ | Symmetric to up-capture on `b < 0`. |
| `skewness` | `distribution.py:29` | ✅ | Fisher-Pearson unbiased; biased option matches biased formula; `NaN` when constant. |
| `excess_kurtosis` | `distribution.py:53` | ✅ | Excel-KURT formula in unbiased path; biased subtracts 3. |
| `jarque_bera` | `distribution.py:78` | ✅ | Uses **biased** S and K (matches the original Jarque–Bera paper). χ²(2) p-value via `exp(−JB/2)`. Returns a frozen dataclass. |
| `autocorrelation` | `distribution.py:97` | ✅ | Pearson on `r[lag:]` vs. `r[:n−lag]`. Validates `lag ≥ 1`. |
| `hit_rate` | `activity.py:8` | ✅ | Strict inequality `r > τ`. |
| `average_win` | `activity.py:15` | ✅ | `r > τ`; `NaN` if none. |
| `average_loss` | `activity.py:24` | ✅ | `r < τ`; `NaN` if none. |
| `win_loss_ratio` | `activity.py:33` | ✅ | `NaN` propagation handled. |
| `profit_factor` | `activity.py:44` | ✅ | Ratio of gross gains to gross losses around `τ`; `NaN` on zero denominator. |
| `best_period` / `worst_period` | `activity.py:55–66` | ✅ | `Series.max()` / `Series.min()`. |
| `longest_winning_streak` / `losing_streak` | `activity.py:69–96` | ✅ | Strict comparison breaks streaks at `τ`, as documented. |
| `rolling_*` (native) | `rolling.py:90–216` | ✅ | Use Polars `rolling_*`; min-samples defaults to `window` for int windows; output cast to Float32. |
| `rolling_skewness / kurtosis / autocorr / max_drawdown / profit_factor` | `rolling.py:244–331` | ✅ | Use `_window_apply`, which drops within-window NaNs and emits null when valid count `< min_periods`. |
| `rolling_alpha` | `rolling.py:219` | ✅ | Composed from rolling means + rolling beta; matches the scalar formula. |
| `rolling_tracking_error` | `rolling.py:198` | ✅ | Annualized rolling std of active returns. |
| `mtd / qtd / ytd` | `periods.py:10–56` | ✅ | Inclusive lower and upper bounds; `as_of` defaults to `date.today()`. |
| `trailing` | `periods.py:59` | ✅ | `Series.tail(n)`; raises if `n ≤ 0`. |
| `since_inception` | `periods.py:71` | ✅ | Identity. |
| `periods_per_year_for` | `periods.py:86` | ✅ | Mapping matches the convention table; raises on unknown frequency. |
| `annual_to_periodic` / `periodic_to_annual` | `periods.py:96–107` | ✅ | Geometric, validates `q > 0`. |
| `sharpe_standard_error` | `inference.py:17` | ✅ | Implements Lo (2002) eq (12); `NaN` when σ = 0; needs `n ≥ 4`. |
| `sharpe_confidence_interval` | `inference.py:44` | ✅ | Wald CI using `Φ⁻¹((1+c)/2)`. |
| `bootstrap_metric` | `inference.py:58` | ✅ | i.i.d. percentile bootstrap; silently skips resamples that raise `(ValueError, ZeroDivisionError)`; returns `(NaN, NaN)` bounds if all fail. |
| `summary` | `report.py:51` | ✅ | Calls every scalar metric through `_safe`, which catches `(ValueError, ZeroDivisionError)` and converts to `NaN`. Float64 columns downcast to Float32 (dtype policy). |

### Observations from the audit

1. **Every metric in this document is implemented exactly as stated**, modulo
   four small but worth-flagging edge cases that all match documented
   behaviour:
   - `annualize_return` returns `NaN` on a wiped-out track record (`R_tot ≤ −1`)
     instead of raising.
   - `recovery_time` returns `NaN` on unrecovered series, `0.0` when no
     drawdown ever occurred.
   - Historical `conditional_value_at_risk` falls back to `−Q_{1−c}` when no
     return is at or below the quantile (rare with linear interpolation, but
     possible at extreme `confidence` on tiny samples).
   - `bootstrap_metric` silently drops resamples that raise
     `(ValueError, ZeroDivisionError)`; the point estimate is unaffected, but
     the CI is computed on a (possibly smaller) usable subset and degenerates
     to `(NaN, NaN)` if everything fails.

2. **Convention choices to keep in mind when comparing to other libraries.**
   - `downside_deviation` uses `ddof=0` and the all-periods denominator
     (Sortino convention). `pyfolio` and `empyrical` use the same; some
     in-house implementations divide by the count of downside obs only —
     that's `semi_deviation` here.
   - `jarque_bera` uses **biased** skew/kurtosis to match the original Jarque
     & Bera paper. SciPy's `jarque_bera` does the same, but a few tutorial
     implementations plug in the unbiased moments.
   - `skewness` / `excess_kurtosis` default to **unbiased** (SAS / SPSS /
     Excel). Pass `bias=True` for the biased estimators consistent with
     NumPy's `scipy.stats.skew(..., bias=True)`.
   - `calmar_ratio` uses the full sample, not the canonical 36-month window.
   - `sharpe_standard_error` is the Lo (2002) first-order correction only.

3. **Dtype contract.** All public `pl.Series` / `pl.DataFrame` outputs are
   `Float32`; scalar Python returns remain `float` (Float64 semantics).
   Internal accumulation runs in `Float64`. This matches the policy in
   `.claude/CLAUDE.md`.

4. **No undocumented metric is exposed by the package.** The functions in
   `_internal/` (`norm_ppf`, `norm_pdf`, `align_benchmark`, etc.) are
   helpers, not metrics, and live behind a leading-underscore namespace.
