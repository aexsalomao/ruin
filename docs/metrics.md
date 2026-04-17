# Metrics Reference

Authoritative definitions, mathematical formulas, modelling assumptions, and
citations for every metric exposed by `ruin`.

The library follows the conventions set out in
[`docs/conventions.md`](conventions.md):

- Returns are dimensionless fractions (`0.01` = 1%).
- Drawdowns are non-positive (`-0.23` = 23% drawdown).
- VaR / CVaR are positive loss magnitudes (`0.02` = "lose at most 2%").
- `risk_free` is **per-period**, never annualized.
- NaN / null values are dropped before computation; benchmark inputs must be
  pre-aligned in length.

### Notation

| Symbol | Meaning |
|---|---|
| $r_t$ | period-$t$ return (a fraction) |
| $\bar r$ | sample mean of $\{r_t\}$ |
| $n$ | number of return observations after NaN drop |
| $q$ | `periods_per_year` (annualization factor; e.g. 252 for daily) |
| $r_f$ | per-period risk-free rate |
| $\tau$ | per-period threshold or minimum acceptable return (MAR) |
| $b_t$ | benchmark return in period $t$ |
| $\sigma$ | sample standard deviation, `ddof=1` unless noted otherwise |
| $W_t$ | cumulative wealth $\prod_{s \le t}(1 + r_s)$ |
| $H_t$ | running high-water mark $\max_{s \le t} W_s$ |
| $D_t$ | period-$t$ drawdown $W_t / H_t - 1 \le 0$ |
| $\Phi(\cdot), \varphi(\cdot)$ | standard normal CDF / PDF |
| $\mathbf{1}\{\cdot\}$ | indicator function |

Each metric below follows the same template:

> **Definition.** What the function computes in one sentence.
> **Formula.** The mathematical expression.
> **Assumptions.** When the metric is reliable and how it can mislead.
> **Source(s).** Primary academic / industry references.

---

## 1. Returns (`ruin.returns`)

### `from_prices(prices, *, log=False)`

**Definition.** Convert a positive price series of length $N$ into $N - 1$
period returns; simple by default, log if `log=True`.

**Formula.**

$$
r_t^{\text{simple}} = \frac{P_t}{P_{t-1}} - 1, \qquad
r_t^{\text{log}} = \ln\!\left(\frac{P_t}{P_{t-1}}\right)
$$

**Assumptions.**

- Prices are strictly positive; the function raises if any $P_t \le 0$.
- Equal time-spacing between observations.
- Log returns are time-additive but not portfolio-additive; simple returns are
  portfolio-additive but not time-additive.

**Source.** Tsay, *Analysis of Financial Time Series*, 3rd ed. (Wiley, 2010),
§1.1.

---

### `total_return(returns)`

**Definition.** Cumulative compounded return over the full sample.

**Formula.**

$$
R_{\text{tot}} = \prod_{t=1}^{n} (1 + r_t) - 1
$$

**Assumptions.**

- Returns reinvest at the realized rate each period.
- Path-dependent and uninformative about risk.

**Source.** Bodie, Kane & Marcus, *Investments*, 12th ed. (McGraw-Hill, 2021),
ch. 5.

---

### `annualize_return(returns, *, periods_per_year, method="geometric")`

**Definition.** Scale the realized return to a one-year horizon, geometrically
(default) or arithmetically.

**Formula.**

$$
R_{\text{ann}}^{\text{geo}} = (1 + R_{\text{tot}})^{q / n} - 1, \qquad
R_{\text{ann}}^{\text{arith}} = \bar r \cdot q
$$

The geometric branch returns `NaN` when $R_{\text{tot}} \le -1$ (ruin), since
fractional powers of non-positive numbers are undefined.

**Assumptions.**

- The realized period is representative of a typical year (clearly violated on
  short series).
- Geometric assumes terminal-wealth equivalence under continued compounding.
- Arithmetic assumes i.i.d. returns; it overstates expected terminal wealth
  for volatile strategies.

**Source.** CFA Institute, *Quantitative Investment Analysis*, 4th ed. (Wiley,
2020), ch. 1; Meucci, *Risk and Asset Allocation* (Springer, 2005), §3.2.

---

### `cagr(returns, *, periods_per_year)`

**Definition.** Compound annual growth rate — alias for the geometric
`annualize_return`.

**Formula.**

$$
\text{CAGR} = (1 + R_{\text{tot}})^{q / n} - 1
$$

**Assumptions / Source.** As for geometric `annualize_return`. Industry term;
no canonical academic origin.

---

## 2. Volatility (`ruin.volatility`)

### `volatility(returns, *, ddof=1)`

**Definition.** Periodic sample standard deviation of returns (not
annualized).

**Formula.**

$$
\sigma = \sqrt{\frac{1}{n - \text{ddof}} \sum_{t=1}^{n} (r_t - \bar r)^2}
$$

**Assumptions.**

- Returns are i.i.d. and the second moment exists.
- Symmetric loss function — an upside move of size $x$ contributes the same
  as a downside move of size $x$.

**Source.** Markowitz, "Portfolio Selection," *Journal of Finance* 7(1), 1952.

---

### `annualize_volatility(returns, *, periods_per_year, ddof=1)`

**Definition.** Annualized volatility via the square-root-of-time rule.

**Formula.**

$$
\sigma_{\text{ann}} = \sigma \cdot \sqrt{q}
$$

**Assumptions.**

- Returns are i.i.d. (zero serial correlation).
- Under autocorrelation the rule under-/overstates risk; the corrected formula
  is $\sigma_{\text{ann}} = \sigma \sqrt{q\,(1 + 2\sum_{k=1}^{K} \rho_k)}$
  (see Lo (2002) below).

**Source.** Diebold, Hickman, Inoue & Schuermann, "Converting 1-day Volatility
to h-day Volatility: Scaling by √h is Worse than You Think," Wharton FIC
Working Paper 97-34, 1997.

---

### `downside_deviation(returns, *, threshold=0.0, ddof=0)`

**Definition.** Sortino-convention semi-deviation: only periods with
$r_t < \tau$ contribute to the numerator, but **all** $n$ periods are in the
denominator.

**Formula.**

$$
\text{DD}_\tau = \sqrt{\frac{1}{n - \text{ddof}} \sum_{t=1}^{n} \min(r_t - \tau,\,0)^2}
$$

**Assumptions.**

- The MAR $\tau$ is the relevant reference point — not the mean.
- Investors care asymmetrically about losses below $\tau$.
- `ddof=0` matches the Sortino convention; pass `ddof=1` for sample-style.

**Source.** Sortino & van der Meer, "Downside Risk," *Journal of Portfolio
Management* 17(4), 1991; Sortino & Price, "Performance Measurement in a
Downside Risk Framework," *Journal of Investing* 3(3), 1994.

---

### `semi_deviation(returns, *, ddof=0)`

**Definition.** Standard deviation computed only over strictly negative
returns (numerator and denominator both restricted). Returns 0 when there are
no negative returns.

**Formula.** With $n_{-} = |\{t : r_t < 0\}|$ and
$\bar r_{-} = \frac{1}{n_{-}} \sum_{t : r_t < 0} r_t$,

$$
\text{SD}_{-} = \sqrt{\frac{1}{n_{-} - \text{ddof}} \sum_{t \,:\, r_t < 0} (r_t - \bar r_{-})^2}
$$

**Assumptions.**

- Behaviour around zero matters more than around the mean.
- Sensitive to small samples once the negative-return count is small.

**Source.** Markowitz, *Portfolio Selection: Efficient Diversification of
Investments* (Wiley, 1959), ch. IX (semivariance).

---

## 3. Drawdown (`ruin.drawdown`)

All drawdown metrics derive from

$$
W_t = \prod_{s \le t}(1 + r_s), \qquad
H_t = \max_{s \le t} W_s, \qquad
D_t = \frac{W_t}{H_t} - 1 \le 0.
$$

The implementation prepends an initial wealth of $1$ so first-period losses
register as drawdowns.

### `drawdown_series(returns)`

**Definition.** The full drawdown time series $\{D_t\}_{t=1}^{n}$.

**Formula.** As above.

**Assumptions.** Path- and sample-dependent; daily vs. month-end data produce
different series.

**Source.** Magdon-Ismail & Atiya, "Maximum Drawdown," *Risk Magazine* 17(10),
2004.

---

### `max_drawdown(returns)`

**Definition.** The single worst drawdown observed over the sample.

**Formula.**

$$
D^{\star} = \min_{1 \le t \le n} D_t \le 0
$$

**Assumptions.** Short histories systematically understate $|D^{\star}|$.

**Source.** Magdon-Ismail & Atiya (2004); Grossman & Zhou, "Optimal Investment
Strategies for Controlling Drawdowns," *Mathematical Finance* 3(3), 1993.

---

### `average_drawdown(returns)`

**Definition.** Mean trough magnitude across distinct drawdown **episodes** —
contiguous runs of $D_t < 0$ separated by points where $D_t \ge 0$. Returns 0
when there are no drawdowns.

**Formula.** Let $E_1, \dots, E_K$ index the underwater episodes. Then

$$
\overline{D} = \frac{1}{K} \sum_{k=1}^{K} \min_{t \in E_k} D_t.
$$

**Assumptions.** "Episode" is a definitional choice; alternatives
(peak-to-peak clustering, threshold filters) yield different numbers.

**Source.** Industry convention; compatible with Chekhlov, Uryasev &
Zabarankin, "Drawdown Measure in Portfolio Optimization," *International
Journal of Theoretical and Applied Finance* 8(1), 2005.

---

### `max_drawdown_duration(returns)`

**Definition.** Longest consecutive count of periods spent underwater.

**Formula.**

$$
T^{\text{uw}}_{\max} = \max_{(a,b)} \{\, b - a + 1 \;:\; D_t < 0 \;\forall\, t \in [a, b]\,\}
$$

**Assumptions.** Equally-spaced periods; units are *periods*, not calendar
days.

**Source.** Magdon-Ismail & Atiya (2004); Burghardt, Duncan & Liu,
"Deciphering Drawdowns," *Risk Magazine* 16(9), 2003.

---

### `recovery_time(returns)`

**Definition.** Number of periods from the maximum-drawdown trough to the
first subsequent return to a new HWM. `NaN` if the strategy has not recovered
by the end of the sample, $0$ if no drawdown ever occurred.

**Formula.** With $t^{\star} = \arg\min_t D_t$,

$$
T^{\text{rec}} = \min\{\, t > t^{\star} \;:\; D_t \ge 0 \,\} - t^{\star},
$$

returning `NaN` if the set is empty.

**Source.** Burghardt, Duncan & Liu (2003).

---

### `time_underwater(returns)`

**Definition.** Total number of periods spent below the HWM.

**Formula.**

$$
T^{\text{uw}} = \sum_{t=1}^{n} \mathbf{1}\{D_t < 0\}
$$

---

### `drawdown_start(returns)` / `drawdown_end(returns)`

**Definition.** Zero-based indices of, respectively, the HWM peak immediately
preceding the maximum drawdown and the trough itself.

**Formula.** With $t^{\star} = \arg\min_t D_t$,

$$
t_{\text{start}} = \max\{\, t \le t^{\star} \;:\; D_t \ge 0 \,\}, \qquad
t_{\text{end}} = t^{\star}.
$$

**Assumptions.** Discrete period index; for the rare degenerate case of no
drawdown both functions return $0$.

**Source.** Standard construction; see Magdon-Ismail & Atiya (2004).

---

## 4. Risk-Adjusted Ratios (`ruin.ratios`)

### `sharpe_ratio(returns, *, risk_free=0.0, periods_per_year, ddof=1)`

**Definition.** Annualized excess return per unit of annualized volatility of
excess returns.

**Formula.** Letting $e_t = r_t - r_f$ and $\sigma_e$ its `ddof=1` std,

$$
\text{SR} = \frac{q \cdot \bar e}{\sqrt{q} \cdot \sigma_e} = \frac{\bar e}{\sigma_e} \sqrt{q}.
$$

Returns `NaN` when $\sqrt{q}\,\sigma_e = 0$.

**Assumptions.**

- Returns are i.i.d. and approximately Normal (justifies $\sqrt{q}$ scaling).
- Symmetric loss function (penalises upside variability equally).

**Source.** Sharpe, "Mutual Fund Performance," *Journal of Business* 39(1),
1966; Sharpe, "The Sharpe Ratio," *Journal of Portfolio Management* 21(1),
1994.

---

### `sortino_ratio(returns, *, risk_free=0.0, threshold=None, periods_per_year)`

**Definition.** Annualized excess return per unit of annualized downside
deviation. `threshold` defaults to `risk_free`; downside deviation uses
`ddof=0` and the all-periods denominator.

**Formula.**

$$
\text{Sortino} = \frac{q \cdot \overline{(r - r_f)}}{\sqrt{q} \cdot \text{DD}_\tau}
$$

**Assumptions.**

- Investors care about returns below the MAR $\tau$, not below the mean.
- $\sqrt{q}$ annualization of $\text{DD}_\tau$ inherits the i.i.d. caveats of
  Sharpe.

**Source.** Sortino & Price (1994); Sortino, *The Sortino Framework for
Constructing Portfolios* (Elsevier, 2010).

---

### `calmar_ratio(returns, *, periods_per_year)`

**Definition.** Annualized return per unit of maximum drawdown magnitude.

**Formula.**

$$
\text{Calmar} = \frac{\text{CAGR}}{|D^{\star}|}
$$

Returns `NaN` when $D^{\star} = 0$.

**Assumptions.**

- $D^{\star}$ is a meaningful summary of risk (path- and sample-dependent).
- The classical Calmar uses a 36-month rolling window; this implementation
  uses the full sample. Short histories with no large drawdowns yield
  artificially high Calmar.

**Source.** Young, "Calmar Ratio: A Smoother Tool," *Futures Magazine*, 1991.

---

### `information_ratio(returns, benchmark, *, periods_per_year, ddof=1)`

**Definition.** Annualized active return per unit of annualized tracking
error.

**Formula.** Letting $a_t = r_t - b_t$,

$$
\text{IR} = \frac{q \cdot \bar a}{\sqrt{q} \cdot \sigma_a}
$$

**Assumptions.**

- Active returns are i.i.d. ($\sqrt{q}$ scaling).
- Returns and benchmark are pre-aligned in length and time.

**Source.** Goodwin, "The Information Ratio," *Financial Analysts Journal*
54(4), 1998; Grinold & Kahn, *Active Portfolio Management*, 2nd ed.
(McGraw-Hill, 2000), ch. 5.

---

### `treynor_ratio(returns, benchmark, *, risk_free=0.0, periods_per_year)`

**Definition.** Annualized excess return per unit of market beta.

**Formula.**

$$
\text{Treynor} = \frac{q \cdot \overline{(r - r_f)}}{\beta}
$$

Returns `NaN` if $\beta = 0$.

**Assumptions.**

- CAPM-style framework: market beta captures all systematic risk.
- Poorly defined when $\beta \approx 0$ or negative.
- Most informative for diversified, long-only portfolios.

**Source.** Treynor, "How to Rate Management of Investment Funds," *Harvard
Business Review* 43(1), 1965.

---

### `omega_ratio(returns, *, threshold=0.0)`

**Definition.** Probability-weighted ratio of gains to losses around a
threshold $\tau$. $\Omega > 1$ iff $\bar r > \tau$ for any distribution with
finite mean. At $\tau = 0$ it coincides with the profit factor.

**Formula.**

$$
\Omega(\tau) = \frac{\sum_{t} \max(r_t - \tau,\,0)}{\sum_{t} \max(\tau - r_t,\,0)}
$$

equivalent to the integral form

$$
\Omega(\tau) = \frac{\int_\tau^{\infty} \bigl(1 - F(x)\bigr)\,dx}{\int_{-\infty}^{\tau} F(x)\,dx}.
$$

Returns `NaN` when no observations fall strictly below $\tau$.

**Assumptions.**

- Threshold choice is critical — using $r_f$ per-period is more economically
  meaningful than $\tau = 0$.

**Source.** Keating & Shadwick, "A Universal Performance Measure," *Journal of
Performance Measurement* 6(3), 2002.

---

## 5. Tail Risk (`ruin.tail`)

### `value_at_risk(returns, *, confidence=0.95, method="historical")`

**Definition.** The smallest positive loss $L$ such that
$\mathbb{P}(\text{loss} \le L) \ge c$ at confidence $c \in (0,1)$.

**Formula.** Let $\alpha = 1 - c$.

- *Historical*: $\;\text{VaR}_c = -\hat Q_\alpha(r)$, where $\hat Q_\alpha$ is
  the empirical $\alpha$-quantile with linear interpolation.
- *Parametric* (Gaussian):
  $\;\text{VaR}_c = -\bigl(\hat\mu + \Phi^{-1}(\alpha)\,\hat\sigma\bigr)$, with
  sample mean $\hat\mu$ and `ddof=1` sample std $\hat\sigma$.

**Assumptions.**

- *Historical*: requires large samples; insensitive to tail shape beyond the
  cutoff.
- *Parametric*: returns are Normally distributed — systematically
  underestimates losses for fat-tailed distributions.
- VaR is **not coherent** (fails sub-additivity in general).

**Source.** Jorion, *Value at Risk: The New Benchmark for Managing Financial
Risk*, 3rd ed. (McGraw-Hill, 2007); Artzner, Delbaen, Eber & Heath, "Coherent
Measures of Risk," *Mathematical Finance* 9(3), 1999.

---

### `conditional_value_at_risk(returns, *, confidence=0.95, method="historical")`

(Aliased as `expected_shortfall`.)

**Definition.** Expected loss conditional on being in the $\alpha = 1 - c$
left tail.

**Formula.** Let $z = \Phi^{-1}(\alpha)$.

- *Historical*:
  $\;\text{CVaR}_c = -\,\mathbb{E}\!\left[\,r \mid r \le \hat Q_\alpha(r)\right]$.
  If no observation is at or below $\hat Q_\alpha$, the implementation falls
  back to $-\hat Q_\alpha$.
- *Parametric* (Gaussian):
  $\;\text{CVaR}_c = -\!\left(\hat\mu - \hat\sigma \cdot \dfrac{\varphi(z)}{\alpha}\right)$.

**Assumptions.**

- CVaR is **coherent** (sub-additive) — preferred over VaR for portfolio
  aggregation.
- *Parametric* version inherits Normality.
- Sample-size sensitive; consider bootstrapping with `bootstrap_metric`.

**Source.** Rockafellar & Uryasev, "Optimization of Conditional Value-at-Risk,"
*Journal of Risk* 2(3), 2000; Acerbi & Tasche, "On the Coherence of Expected
Shortfall," *Journal of Banking & Finance* 26(7), 2002.

---

## 6. Market / Benchmark-Relative (`ruin.market`)

### `beta(returns, benchmark)`

**Definition.** OLS slope of strategy returns regressed on benchmark returns.

**Formula.**

$$
\beta = \frac{\widehat{\text{Cov}}(r, b)}{\widehat{\text{Var}}(b)}
$$

with `ddof=1` everywhere; `NaN` when $\widehat{\text{Var}}(b) = 0$.

**Assumptions.**

- Linear relationship between strategy and benchmark.
- Stationarity — rolling beta is more informative than full-sample beta for
  live monitoring.

**Source.** Sharpe, "Capital Asset Prices: A Theory of Market Equilibrium
Under Conditions of Risk," *Journal of Finance* 19(3), 1964.

---

### `downside_beta(returns, benchmark)` / `upside_beta(returns, benchmark)`

**Definition.** Beta computed only on periods where the benchmark is negative
(downside) or positive (upside). Both return `NaN` when fewer than 2
conditioning observations or zero conditional variance.

**Formula.** Let $\mathcal{T}_- = \{t : b_t < 0\}$ and
$\mathcal{T}_+ = \{t : b_t > 0\}$. Then

$$
\beta^{-} = \frac{\widehat{\text{Cov}}_{\mathcal{T}_-}(r, b)}{\widehat{\text{Var}}_{\mathcal{T}_-}(b)}, \qquad
\beta^{+} = \frac{\widehat{\text{Cov}}_{\mathcal{T}_+}(r, b)}{\widehat{\text{Var}}_{\mathcal{T}_+}(b)}.
$$

**Assumptions.** Conditional samples are large enough for stable second
moments; otherwise the metric is dominated by a few observations.

**Source.** Bawa & Lindenberg, "Capital Market Equilibrium in a Mean-Lower
Partial Moment Framework," *Journal of Financial Economics* 5(2), 1977; Ang,
Chen & Xing, "Downside Risk," *Review of Financial Studies* 19(4), 2006.

---

### `alpha(returns, benchmark, *, risk_free=0.0, periods_per_year)`

**Definition.** Annualized Jensen's alpha — the CAPM intercept on excess
returns.

**Formula.**

$$
\alpha = q \cdot \overline{(r - r_f)} - \beta \cdot q \cdot \overline{(b - r_f)}
$$

**Assumptions.**

- CAPM holds (a strong empirical assumption).
- Treat as a performance decomposition rather than a causal attribution.

**Source.** Jensen, "The Performance of Mutual Funds in the Period 1945–1964,"
*Journal of Finance* 23(2), 1968.

---

### `tracking_error(returns, benchmark, *, periods_per_year, ddof=1)`

**Definition.** Annualized standard deviation of active returns.

**Formula.** With $a_t = r_t - b_t$,

$$
\text{TE} = \sqrt{q} \cdot \sigma_a.
$$

**Assumptions.** Active returns are i.i.d.

**Source.** Roll, "A Mean/Variance Analysis of Tracking Error," *Journal of
Portfolio Management* 18(4), 1992.

---

### `correlation(returns, benchmark)`

**Definition.** Pearson product-moment correlation, $\rho \in [-1, 1]$.

**Formula.**

$$
\rho = \frac{\widehat{\text{Cov}}(r, b)}{\sigma_r \, \sigma_b}
$$

**Assumptions.** Joint distribution is well-approximated by a bivariate
distribution with finite second moments; nonlinear dependence is missed.

**Source.** Pearson, "Notes on the History of Correlation," *Biometrika* 13,
1920 (historical).

---

### `up_capture(returns, benchmark)` / `down_capture(returns, benchmark)`

**Definition.** Geometric capture ratios over benchmark-up / benchmark-down
periods.

**Formula.** With $\mathcal{T}_+ = \{t : b_t > 0\}$ and
$\mathcal{T}_- = \{t : b_t < 0\}$,

$$
\text{Up} = \frac{\prod_{t \in \mathcal{T}_+}(1 + r_t) - 1}{\prod_{t \in \mathcal{T}_+}(1 + b_t) - 1}, \qquad
\text{Down} = \frac{\prod_{t \in \mathcal{T}_-}(1 + r_t) - 1}{\prod_{t \in \mathcal{T}_-}(1 + b_t) - 1}.
$$

Returns `NaN` when there are no qualifying periods or the benchmark
compounded return is exactly zero.

**Assumptions.**

- "Up market" is defined as a single-period benchmark $> 0$; alternative
  methodologies use monthly or quarterly aggregation.
- Short series are unreliable.

**Source.** Morningstar, *Morningstar Methodology Paper: Upside/Downside
Capture Ratio*, 2011.

---

## 7. Distribution Shape (`ruin.distribution`)

Standardized residuals: $z_t = (r_t - \bar r) / \sigma$, with $\sigma$ the
sample std (`ddof=1`). All moment-based metrics return `NaN` when the input
is constant.

### `skewness(returns, *, bias=False)`

**Definition.** Third standardized moment; negative = left-skewed, zero =
symmetric.

**Formula.**

$$
\hat S_{\text{biased}} = \frac{1}{n} \sum_{t=1}^{n} z_t^3, \qquad
\hat S_{\text{unbiased}} = \frac{n^2}{(n-1)(n-2)} \cdot \hat S_{\text{biased}}.
$$

**Assumptions.** Reliable only with reasonable sample sizes; bias correction
assumes independent observations.

**Source.** Joanes & Gill, "Comparing Measures of Sample Skewness and
Kurtosis," *Journal of the Royal Statistical Society Series D* 47(1), 1998
(SAS / SPSS / Excel SKEW conventions).

---

### `excess_kurtosis(returns, *, bias=False)`

**Definition.** Fisher excess kurtosis; Normal $= 0$, positive $=$ fatter
tails.

**Formula.**

$$
\hat K_{\text{biased}} = \frac{1}{n} \sum_{t=1}^{n} z_t^4 - 3,
$$

$$
\hat K_{\text{unbiased}} = \frac{n(n+1)}{(n-1)(n-2)(n-3)} \sum_{t=1}^{n} z_t^4 \;-\; \frac{3(n-1)^2}{(n-2)(n-3)}.
$$

**Assumptions.** Large samples (n > 100 is a useful rule of thumb);
independence for the bias correction.

**Source.** Joanes & Gill (1998) (Excel KURT convention).

---

### `jarque_bera(returns)`

**Definition.** Jarque–Bera asymptotic test of Normality. Returns a frozen
`JarqueBeraResult(statistic, p_value)`.

**Formula.** Using **biased** $\hat S$ and $\hat K$,

$$
\text{JB} = \frac{n}{6}\left(\hat S^2 + \frac{\hat K^2}{4}\right) \;\stackrel{H_0}{\sim}\; \chi^2_2,
$$

$$
p = \exp\!\left(-\,\text{JB} / 2\right) \quad \text{(exact $\chi^2_2$ survival).}
$$

**Assumptions.** Asymptotic — unreliable for $n \lesssim 100$. Failing to
reject does not prove Normality.

**Source.** Jarque & Bera, "Efficient Tests for Normality, Homoscedasticity
and Serial Independence of Regression Residuals," *Economics Letters* 6(3),
1980.

---

### `autocorrelation(returns, *, lag=1)`

**Definition.** Lag-$k$ serial autocorrelation.

**Formula.**

$$
\rho_k = \frac{\widehat{\text{Cov}}(r_t,\, r_{t-k})}{\sigma(r_t)\,\sigma(r_{t-k})}
$$

implemented as the Pearson correlation between $r_{[k+1:\,n]}$ and
$r_{[1:\,n-k]}$.

**Assumptions.** Stationarity. Positive lag-1 autocorrelation is a key
diagnostic for smoothed / illiquid pricing and invalidates the
square-root-of-time rule.

**Source.** Box, Jenkins, Reinsel & Ljung, *Time Series Analysis: Forecasting
and Control*, 5th ed. (Wiley, 2015), ch. 2.

---

## 8. Activity (`ruin.activity`)

All thresholds default to $\tau = 0$; pass the per-period `risk_free` for an
economically grounded reference.

### `hit_rate(returns, *, threshold=0.0)`

**Definition.** Fraction of periods that beat the threshold; $\in [0, 1]$.

**Formula.**

$$
\text{HR}(\tau) = \frac{1}{n} \sum_{t=1}^{n} \mathbf{1}\{r_t > \tau\}
$$

---

### `average_win(returns, *, threshold=0.0)` / `average_loss(returns, *, threshold=0.0)`

**Definition.** Mean of returns strictly above (resp. below) $\tau$. `NaN` if
no observation meets the criterion.

**Formula.** Let $\mathcal{W} = \{t : r_t > \tau\}$ and
$\mathcal{L} = \{t : r_t < \tau\}$. Then

$$
\overline{W}(\tau) = \frac{1}{|\mathcal{W}|} \sum_{t \in \mathcal{W}} r_t, \qquad
\overline{L}(\tau) = \frac{1}{|\mathcal{L}|} \sum_{t \in \mathcal{L}} r_t.
$$

---

### `win_loss_ratio(returns, *, threshold=0.0)`

**Definition.** Average win over absolute average loss. `NaN` if either side
is `NaN` or the denominator is zero.

**Formula.**

$$
\text{WLR}(\tau) = \frac{\overline{W}(\tau)}{|\overline{L}(\tau)|}
$$

---

### `profit_factor(returns, *, threshold=0.0)`

**Definition.** Gross gains over gross losses around $\tau$. `NaN` when the
denominator is zero.

**Formula.**

$$
\text{PF}(\tau) = \frac{\sum_{t : r_t > \tau} r_t}{\left|\sum_{t : r_t < \tau} r_t\right|}
$$

**Source.** Pardo, *The Evaluation and Optimization of Trading Strategies*,
2nd ed. (Wiley, 2008), ch. 12.

---

### `best_period(returns)` / `worst_period(returns)`

**Definition.** Maximum / minimum single-period return.

**Formula.**

$$
r_{\max} = \max_t r_t, \qquad r_{\min} = \min_t r_t.
$$

---

### `longest_winning_streak(returns, *, threshold=0.0)` / `longest_losing_streak(returns, *, threshold=0.0)`

**Definition.** Longest run of consecutive periods strictly above (winning) or
strictly below (losing) $\tau$. Periods exactly equal to $\tau$ break the
streak.

**Formula.**

$$
S_{\text{win}}(\tau) = \max_{(a,b)} \{\, b - a + 1 \;:\; r_t > \tau \;\forall\, t \in [a, b]\,\},
$$

$$
S_{\text{lose}}(\tau) = \max_{(a,b)} \{\, b - a + 1 \;:\; r_t < \tau \;\forall\, t \in [a, b]\,\}.
$$

**Assumptions.**

- Strict inequality breaks streaks at $\tau$ — a deliberate convention.
- Sample-frequency-dependent (daily streaks ≠ monthly streaks).

---

## 9. Rolling Variants (`ruin.rolling`)

Every `rolling_*` function returns a Polars `Series` cast to **Float32**,
length-aligned to the input, with the leading $w - 1$ values null. Internally
all accumulation runs in Float64.

For a window of size $w$ and a scalar metric $f$ defined on a window's
returns, the rolling metric at time $t$ is

$$
f^{\text{roll}}_t = f\!\left(r_{t - w + 1},\, r_{t - w + 2},\, \dots,\, r_t\right) \quad \text{for } t \ge w,
$$

with `null` emitted whenever the valid (NaN-dropped) sample within the window
is below `min_periods`.

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
(`rolling_volatility`, etc.); for path-dependent metrics implemented via
`_window_apply` (skew, kurtosis, max-drawdown, profit factor,
autocorrelation), each window's NaN/null entries are dropped before the
metric runs.

**Assumptions.** Within-window stationarity. Each window is a plug-in
estimator of the underlying metric — small windows produce noisy estimates of
higher moments.

**Source.** Standard rolling-window construction; see Tsay (2010) §2.7.

---

## 10. Period Slicing & Rate Conversion (`ruin.periods`)

These are deterministic plumbing for metric pipelines, not statistical
metrics, but they appear in formulas above.

### `mtd / qtd / ytd(returns, *, date_col, as_of=None)`

**Definition.** Filter rows whose date lies in the current month / quarter /
year up to `as_of` (defaults to `datetime.date.today()`).

**Formula.** With $d_t$ the date of row $t$, reference date $d^{\star}$, and
$d_{\text{start}}(d^{\star})$ the first day of the corresponding month /
quarter / year,

$$
\{t \;:\; d_{\text{start}}(d^{\star}) \le d_t \le d^{\star}\}.
$$

---

### `trailing(returns, *, n, date_col=None)`

**Definition.** Last $n$ rows. Pass a sorted DataFrame; `date_col` is accepted
for API symmetry but unused. Raises if $n \le 0$.

**Formula.** Index slice $[N - n, N)$ where $N$ is the total row count.

---

### `since_inception(returns)`

**Definition / Formula.** Identity — returns the input unchanged. Provided
for API symmetry.

---

### `periods_per_year_for(frequency)`

**Definition.** Conventional periods/year mapping for common frequencies.
Raises on unknown input.

| Frequency | $q$ |
|---|---|
| D (daily) | 252 |
| W (weekly) | 52 |
| M (monthly) | 12 |
| Q (quarterly) | 4 |
| A / Y (annual) | 1 |

**Source.** CFA Institute, *Quantitative Investment Analysis* §2.

---

### `annual_to_periodic(rate, *, periods_per_year)` / `periodic_to_annual(rate, *, periods_per_year)`

**Definition.** Geometric conversion between annualized and per-period rates.

**Formula.**

$$
r_{\text{periodic}} = (1 + r_{\text{annual}})^{1/q} - 1, \qquad
r_{\text{annual}} = (1 + r_{\text{periodic}})^{q} - 1.
$$

**Assumptions.** Geometric compounding with a constant rate across all
periods. `periods_per_year > 0` is enforced.

---

## 11. Inference (`ruin.inference`)

### `sharpe_standard_error(returns, *, periods_per_year)`

**Definition.** Lo (2002) autocorrelation-adjusted standard error of the
**annualized** Sharpe ratio.

**Formula.** With per-period $\widehat{\text{SR}}_q = \bar r / \sigma$ and
lag-1 autocorrelation $\hat\rho_1$,

$$
\widehat{\text{SE}}\!\left(\widehat{\text{SR}}_{\text{ann}}\right)
\;\approx\;
\sqrt{\frac{1 + 2\,\hat\rho_1\,\widehat{\text{SR}}_q^{\,2}}{n}} \cdot \sqrt{q}.
$$

Reduces to $\sqrt{q / n}$ under i.i.d. returns. Returns `NaN` when
$\sigma = 0$; requires $n \ge 4$.

**Assumptions.**

- Returns are stationary.
- Adjusts for first-order autocorrelation only — higher-order serial
  dependence still biases the estimator.

**Source.** Lo, "The Statistics of Sharpe Ratios," *Financial Analysts
Journal* 58(4), 2002 (eq. 12).

---

### `sharpe_confidence_interval(returns, *, periods_per_year, confidence=0.95)`

**Definition.** Asymptotic Wald confidence interval for the annualized Sharpe
ratio.

**Formula.** With $z_{c} = \Phi^{-1}\!\left(\tfrac{1 + c}{2}\right)$,

$$
\text{CI}_c \;=\; \left(\widehat{\text{SR}}_{\text{ann}} - z_c \cdot \widehat{\text{SE}},\ \widehat{\text{SR}}_{\text{ann}} + z_c \cdot \widehat{\text{SE}}\right).
$$

**Assumptions.** Asymptotic Normality of the Sharpe estimator (best with
$n \ge \sim 60$).

**Source.** Lo (2002).

---

### `bootstrap_metric(fn, returns, *, n_samples=1000, confidence=0.95, seed=None)`

**Definition.** Percentile-method bootstrap CI for any scalar metric.
Returns `(point, lower, upper)`. The point estimate is `fn(returns)` on the
original sample.

**Formula.** Draw $B$ resamples $r^{(b)}_1, \dots, r^{(b)}_n$ i.i.d. with
replacement from $\{r_t\}$. Compute $\hat\theta^{(b)} = f(r^{(b)})$. Sort to
$\hat\theta^{(1)} \le \dots \le \hat\theta^{(B^{\star})}$ (using only $B^{\star}$
non-failing resamples). With $\alpha = 1 - c$,

$$
\text{CI}_c \;=\; \left(\hat\theta^{(\lfloor (\alpha/2) \cdot B^{\star} \rfloor)},\ \hat\theta^{(\lceil (1 - \alpha/2) \cdot B^{\star} \rceil - 1)}\right).
$$

If every resample raises (`ValueError` or `ZeroDivisionError`), the bounds
degenerate to `(NaN, NaN)`.

**Assumptions.**

- Observations are i.i.d. — i.i.d. resampling destroys serial dependence; for
  autocorrelated returns use a block bootstrap (not provided).
- `fn` accepts a `pl.Series` and returns a `float`.

**Source.** Efron & Tibshirani, *An Introduction to the Bootstrap* (Chapman &
Hall, 1993), ch. 13 (percentile method).

---

## 12. Bundled Report (`ruin.report`)

### `summary(returns, benchmark=None, *, risk_free=0.0, periods_per_year, strict=False)`

**Definition.** Computes every scalar metric above for one return stream — or
one row per column when given a DataFrame. Benchmark-relative columns are
`null` when no benchmark is supplied. `strict=True` raises on NaN/null input
instead of dropping. Float64 output columns are downcast to Float32 for the
public `pl.DataFrame` return type, in line with the dtype policy in
`.claude/CLAUDE.md`.

**Formula.** For each column $c$ and metric function $f \in \mathcal{F}$,

$$
\text{summary}[c, f] \;=\; \begin{cases} f(r_c) & \text{if no exception,} \\ \text{NaN} & \text{if } f \text{ raises } (\texttt{ValueError},\ \texttt{ZeroDivisionError}). \end{cases}
$$

This is the **only** bundled function in `ruin` — every other public symbol
is a single-metric building block.

**Source.** Composition pattern internal to `ruin`; see `report.py`.

---

## Definition ↔ Implementation Audit

The following table verifies that the formula stated above matches the actual
behaviour in source. Each row records the source location and any caveat that
materially alters the textbook formula.

| Metric | Source location | Match? | Notes |
|---|---|---|---|
| `from_prices` | `returns.py:16` | ✅ | Drops the leading null produced by `shift(1)` and any NaN; raises if any price ≤ 0. |
| `total_return` | `returns.py:26` | ✅ | Direct `(1 + r).product() − 1`. |
| `annualize_return` (geo) | `returns.py:33` | ✅ + edge case | Returns `NaN` when total return ≤ −1, avoiding fractional powers of negatives. |
| `annualize_return` (arith) | `returns.py:55` | ✅ | `mean(r) * q`. |
| `cagr` | `returns.py:60` | ✅ | Pure alias. |
| `volatility` | `volatility.py:8` | ✅ | `Series.std(ddof=ddof)`. |
| `annualize_volatility` | `volatility.py:15` | ✅ | `volatility · √q`; validates `q > 0`. |
| `downside_deviation` | `volatility.py:27` | ✅ | All-periods denominator with `ddof=0` — Sortino convention. |
| `semi_deviation` | `volatility.py:48` | ✅ | Restricted to negative returns; returns 0 when there are none. |
| `drawdown_series` | `drawdown.py:15` | ✅ | Prepends initial wealth = 1 so first-period losses register as drawdowns. |
| `max_drawdown` | `drawdown.py:27` | ✅ | `min(drawdown_series)`. |
| `average_drawdown` | `drawdown.py:33` | ✅ | Episode = contiguous underwater run, separated by `dd ≥ 0`. |
| `max_drawdown_duration` | `drawdown.py:61` | ✅ | Longest count of consecutive `dd < 0` periods. |
| `recovery_time` | `drawdown.py:76` | ✅ | First subsequent index with `dd ≥ 0` after the trough; `NaN` if unrecovered, `0.0` if no drawdown. |
| `time_underwater` | `drawdown.py:98` | ✅ | Count of `dd < 0` periods. |
| `drawdown_start` | `drawdown.py:104` | ✅ | Walks back from trough to the last `dd ≥ 0`; `0` if no drawdown. |
| `drawdown_end` | `drawdown.py:125` | ✅ | Index of the trough (first occurrence of `min`). |
| `sharpe_ratio` | `ratios.py:17` | ✅ | Annualized; `NaN` when annualized vol = 0. |
| `sortino_ratio` | `ratios.py:35` | ✅ | Uses `downside_deviation` (ddof=0, all-periods denominator); threshold defaults to `risk_free`. |
| `calmar_ratio` | `ratios.py:57` | ✅ | `CAGR / |max_drawdown|`. Full-sample window (not 36-month). |
| `information_ratio` | `ratios.py:68` | ✅ | Uses `align_benchmark`; annualized. |
| `treynor_ratio` | `ratios.py:85` | ✅ | `NaN` when `β = 0`. |
| `omega_ratio` | `ratios.py:101` | ✅ | Sum-of-gains over sum-of-losses; `NaN` if no downside. |
| `value_at_risk` (hist) | `tail.py:9` | ✅ | `−Q_{1−c}` via `Series.quantile(..., interpolation="linear")`. |
| `value_at_risk` (param) | `tail.py:28` | ✅ | `−(μ + Φ⁻¹(α)·σ)` with sample `ddof=1`. |
| `conditional_value_at_risk` (hist) | `tail.py:53` | ✅ + edge case | Falls back to `−Q_{1−c}` if no observation lies at or below the quantile. |
| `conditional_value_at_risk` (param) | `tail.py:59` | ✅ | Closed-form Gaussian ES via local `norm_pdf` / `norm_ppf`. |
| `beta` | `market.py:10` | ✅ | Manual `cov / var` with `ddof=1`. |
| `downside_beta` | `market.py:30` | ✅ | Conditioned on `b < 0`; needs ≥ 2 observations. |
| `upside_beta` | `market.py:45` | ✅ | Conditioned on `b > 0`; needs ≥ 2 observations. |
| `alpha` | `market.py:60` | ✅ | Annualized Jensen's alpha using `align_benchmark`. |
| `tracking_error` | `market.py:77` | ✅ | Annualized std of active returns. |
| `correlation` | `market.py:90` | ✅ | Polars `pl.corr` over the aligned pair. |
| `up_capture` | `market.py:98` | ✅ | Geometric on `b > 0` periods. |
| `down_capture` | `market.py:113` | ✅ | Symmetric to up-capture on `b < 0`. |
| `skewness` | `distribution.py:29` | ✅ | Fisher-Pearson unbiased; biased option matches biased formula; `NaN` when constant. |
| `excess_kurtosis` | `distribution.py:53` | ✅ | Excel KURT formula in unbiased path; biased subtracts 3. |
| `jarque_bera` | `distribution.py:78` | ✅ | Uses **biased** S and K (matches the original paper). χ²(2) p-value via `exp(−JB/2)`. |
| `autocorrelation` | `distribution.py:97` | ✅ | Pearson on `r[lag:]` vs. `r[:n−lag]`; validates `lag ≥ 1`. |
| `hit_rate` | `activity.py:8` | ✅ | Strict inequality `r > τ`. |
| `average_win` | `activity.py:15` | ✅ | `r > τ`; `NaN` if none. |
| `average_loss` | `activity.py:24` | ✅ | `r < τ`; `NaN` if none. |
| `win_loss_ratio` | `activity.py:33` | ✅ | `NaN` propagation handled. |
| `profit_factor` | `activity.py:44` | ✅ | Ratio of gross gains to gross losses around `τ`. |
| `best_period` / `worst_period` | `activity.py:55–66` | ✅ | `Series.max()` / `Series.min()`. |
| `longest_winning_streak` / `losing_streak` | `activity.py:69–96` | ✅ | Strict inequality; equality at `τ` breaks the streak. |
| `rolling_*` (native) | `rolling.py:90–216` | ✅ | Use Polars `rolling_*`; min-samples defaults to `window` for int windows; output cast to Float32. |
| `rolling_skewness / kurtosis / autocorr / max_drawdown / profit_factor` | `rolling.py:244–331` | ✅ | Use `_window_apply`; drop within-window NaNs; emit null when valid count `< min_periods`. |
| `rolling_alpha` | `rolling.py:219` | ✅ | Composed from rolling means + rolling beta. |
| `rolling_tracking_error` | `rolling.py:198` | ✅ | Annualized rolling std of active returns. |
| `mtd / qtd / ytd` | `periods.py:10–56` | ✅ | Inclusive lower and upper bounds; `as_of` defaults to today. |
| `trailing` | `periods.py:59` | ✅ | `Series.tail(n)`; raises if `n ≤ 0`. |
| `since_inception` | `periods.py:71` | ✅ | Identity. |
| `periods_per_year_for` | `periods.py:86` | ✅ | Mapping matches the convention table; raises on unknown frequency. |
| `annual_to_periodic` / `periodic_to_annual` | `periods.py:96–107` | ✅ | Geometric, validates `q > 0`. |
| `sharpe_standard_error` | `inference.py:17` | ✅ | Lo (2002) eq (12); `NaN` when σ = 0; needs `n ≥ 4`. |
| `sharpe_confidence_interval` | `inference.py:44` | ✅ | Wald CI using `Φ⁻¹((1+c)/2)`. |
| `bootstrap_metric` | `inference.py:58` | ✅ | i.i.d. percentile bootstrap; silently skips resamples raising `(ValueError, ZeroDivisionError)`; `(NaN, NaN)` if all fail. |
| `summary` | `report.py:51` | ✅ | Calls every scalar metric through `_safe`; Float64 columns downcast to Float32. |

### Observations from the audit

1. **Every metric in this document is implemented exactly as stated**, modulo
   four documented edge cases:
   - `annualize_return` returns `NaN` on a wiped-out track record
     ($R_{\text{tot}} \le -1$) instead of raising.
   - `recovery_time` returns `NaN` on unrecovered series and $0$ when no
     drawdown ever occurred.
   - Historical `conditional_value_at_risk` falls back to $-\hat Q_{1-c}$ when
     no return is at or below the quantile.
   - `bootstrap_metric` silently drops resamples raising
     `(ValueError, ZeroDivisionError)` and returns `(NaN, NaN)` bounds if
     everything fails.

2. **Convention choices to keep in mind when comparing to other libraries.**
   - `downside_deviation` uses the Sortino all-periods denominator with
     `ddof=0` (matches `pyfolio` / `empyrical`).
   - `jarque_bera` uses **biased** skew / kurtosis to match the original
     paper — same as `scipy.stats.jarque_bera`.
   - `skewness` / `excess_kurtosis` default to **unbiased** (SAS / SPSS /
     Excel). Pass `bias=True` for the population estimators.
   - `calmar_ratio` uses the full sample, not the canonical 36-month window.
   - `sharpe_standard_error` is Lo (2002)'s first-order correction only.

3. **Dtype contract.** All public `pl.Series` / `pl.DataFrame` outputs are
   `Float32`; scalar Python returns remain `float` (Float64 semantics).
   Internal accumulation runs in `Float64`. Matches the policy in
   `.claude/CLAUDE.md`.

4. **No undocumented metric is exposed by the package.** The functions in
   `_internal/` (`norm_ppf`, `norm_pdf`, `align_benchmark`, etc.) are
   helpers, not metrics, and live behind a leading-underscore namespace.
