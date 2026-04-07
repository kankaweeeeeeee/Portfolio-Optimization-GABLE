# Monte Carlo Portfolio Optimization

A Python/Jupyter project that uses Monte Carlo simulation and Markov regime-switching to optimize a ฿1 billion (THB) multi-asset portfolio across 15 stocks spanning Thai equities, US tech, gold, and cash.

---

## What This Project Does

1. **Simulates 1-year returns** across Bear / Base / Bull economic scenarios using 100,000 Monte Carlo paths per scenario.
2. **Stress-tests risk** with VaR (5%) and CVaR (95%) across all three scenarios.
3. **Optimizes portfolio weights** for both 1-year and 5-year horizons, subject to regulatory-style constraints.
4. **Models long-run regime shifts** using a 3-state Markov chain (Bear → Base → Bull) over a 5-year horizon with 100,000 simulation paths.
5. **Backtests** assumptions against real historical data downloaded from Yahoo Finance (2021–2026).
6. **Compares assumed vs real** correlation, return, and volatility statistics.
7. **Analyses the top 100 best simulations** by final value, Sharpe ratio, CVaR position, and annualised return.
8. **Computes rolling 1-year metrics** (10 metrics) over a 60-month simulation horizon: Total Return, CAGR, Sharpe, Max Drawdown, Alpha, Beta, Volatility, VaR, CVaR, P/E Proxy.

---

## Portfolio

| # | Ticker | Asset | Category |
|---|--------|-------|----------|
| 1 | KBANK | Kasikorn Bank | Thai Bank |
| 2 | GABLE | Gable & Frame | Thai Tech |
| 3 | GULF | Gulf Energy | Thai Energy |
| 4 | BDMS | Bangkok Dusit Medical | Healthcare |
| 5 | ICHI | Ichitan Group | Beverage |
| 6 | CASH | Cash | Cash |
| 7 | FSLR | First Solar | US Energy |
| 8 | PPH | VanEck Pharma ETF | Healthcare |
| 9 | GOOGL | Alphabet | US Tech |
| 10 | GOLD | SPDR Gold ETF | Gold |
| 11 | NVDA | NVIDIA | US Tech |
| 12 | MSFT | Microsoft | US Tech |
| 13 | SCB | Siam Commercial Bank | Thai Bank |
| 14 | TISCO | TISCO Financial | Thai Bank |
| 15 | NEE | NextEra Energy | US Energy |

**Initial capital:** ฿1,000,000,000 (1 billion THB)

---

## Key Results (from last run)

### 1-Year Simulation (Current Weights)

| Scenario | Mean Value | Loss Probability | Sharpe |
|----------|-----------|-----------------|--------|
| Base     | ฿1.163B   | 22.4%           | 0.68   |
| Bull     | ฿1.200B   | 19.7%           | 0.78   |
| Bear     | ฿1.107B   | 39.2%           | 0.31   |

### 5-Year Markov Simulation — 3-Way Comparison

| Portfolio | Mean (5Y) | Loss Prob | Sharpe |
|-----------|-----------|-----------|--------|
| Current weights    | ฿2.086B | 9.3%  | 1.11 |
| 1Y Optimal weights | ฿3.029B | 5.2%  | 1.41 |
| 5Y Optimal weights | ฿2.685B | 5.1%  | 1.41 |

### 1Y Optimized Weights (Sharpe = 0.84)

| Asset | Weight |
|-------|--------|
| KBANK | 5.84% |
| GABLE | 3.92% |
| GULF  | 8.82% |
| BDMS  | 8.92% |
| ICHI  | 5.85% |
| CASH  | 2.15% |
| FSLR  | 6.55% |
| PPH   | 9.61% |
| GOOGL | 5.37% |
| GOLD  | 9.61% |
| NVDA  | 9.61% |
| MSFT  | 4.94% |
| SCB   | 8.11% |
| TISCO | 8.54% |
| NEE   | 2.15% |

### 5Y Optimized Weights (Sharpe = 1.43)

| Asset | Weight |
|-------|--------|
| KBANK | 8.16% |
| GABLE | 4.82% |
| GULF  | 7.06% |
| BDMS  | 9.48% |
| ICHI  | 9.65% |
| CASH  | 9.10% |
| FSLR  | 2.14% |
| PPH   | 9.10% |
| GOOGL | 6.24% |
| GOLD  | 9.10% |
| NVDA  | 7.00% |
| MSFT  | 6.00% |
| SCB   | 6.56% |
| TISCO | 4.26% |
| NEE   | 2.14% |

### Rolling Metric Summary (30,000 paths, 60 months)

| Metric | Current | 5Y Optimized | Edge |
|--------|---------|-------------|------|
| Total Return (1Y) | 13.4% | 18.9% | 5Y Opt |
| CAGR (3Y) | 13.3% | 18.7% | 5Y Opt |
| Sharpe Ratio | 0.57 | 0.71 | 5Y Opt |
| Max Drawdown | -11.4% | -11.6% | 5Y Opt |
| Alpha | +12.7% | +17.9% | 5Y Opt |
| Volatility (Ann.) | 19.5% | 21.4% | Current |

---

## Portfolio Constraints

These constraints mimic real-world investment mandates:

- **Thai stocks ≥ 50%** of total portfolio (regulatory requirement)
- **Each sector ≤ 25%** (Tech, Thai Banks, Healthcare, Energy, Beverage, Gold, Cash)
- **No single asset > 10%**
- **All weights sum to 1**

---

## Files

```
├── Montecarlo_GABLE.ipynb   # Main notebook — clean, documented, ready to run
├── Montecarlo_GABLE.py              # Same code as a plain Python script
└── README.md                   # This file
```

---

## How to Run

### Requirements

```bash
pip install numpy pandas matplotlib yfinance jupyter
```

### Run as a Jupyter Notebook

```bash
jupyter notebook Montecarlo_GABLE.ipynb
```

Open in your browser, then **Run All Cells** (`Kernel → Restart & Run All`).

### Run as a Python script

```bash
python Montecarlo_GABLE.py
```

> **Note:** The script saves charts as image files and CSV exports to your `~/Downloads` folder.
> Plots will not display interactively when run as a script — use the notebook for interactive charts.

---

## About the Simulation Methods

### Monte Carlo (1-Year)
Each simulation draws 100,000 random portfolio returns from a multivariate log-normal distribution using the asset covariance matrix. Bear/Base/Bull scenarios shift the underlying parameters before sampling.
# Monte Carlo Portfolio Optimization

A Python/Jupyter project that uses Monte Carlo simulation and Markov regime-switching to optimize a ฿1 billion (THB) multi-asset portfolio across 15 stocks spanning Thai equities, US tech, gold, and cash.

---

## What This Project Does

1. **Simulates 1-year returns** across Bear / Base / Bull economic scenarios using 100,000 Monte Carlo paths per scenario.
2. **Stress-tests risk** with VaR (5%) and CVaR (95%) across all three scenarios.
3. **Optimizes portfolio weights** for both 1-year and 5-year horizons, subject to regulatory-style constraints.
4. **Models long-run regime shifts** using a 3-state Markov chain (Bear → Base → Bull) over a 5-year horizon with 100,000 simulation paths.
5. **Backtests** assumptions against real historical data downloaded from Yahoo Finance (2021–2026).
6. **Compares assumed vs real** correlation, return, and volatility statistics.
7. **Analyses the top 100 best simulations** by final value, Sharpe ratio, CVaR position, and annualised return.
8. **Computes rolling 1-year metrics** (10 metrics) over a 60-month simulation horizon: Total Return, CAGR, Sharpe, Max Drawdown, Alpha, Beta, Volatility, VaR, CVaR, P/E Proxy.

---

## Portfolio

| # | Ticker | Asset | Category |
|---|--------|-------|----------|
| 1 | KBANK | Kasikorn Bank | Thai Bank |
| 2 | GABLE | G-Able | Thai Tech |
| 3 | GULF | Gulf Energy | Thai Energy |
| 4 | BDMS | Bangkok Dusit Medical | Healthcare |
| 5 | ICHI | Ichitan Group | Beverage |
| 6 | CASH | Cash | Cash |
| 7 | FSLR | First Solar | US Energy |
| 8 | PPH | VanEck Pharma ETF | Healthcare |
| 9 | GOOGL | Alphabet | US Tech |
| 10 | GOLD | Gold (XAU/USD spot) | Gold |
| 11 | NVDA | NVIDIA | US Tech |
| 12 | MSFT | Microsoft | US Tech |
| 13 | SCB | SCB X (Siam Commercial Bank) | Thai Bank |
| 14 | TISCO | TISCO Financial | Thai Bank |
| 15 | NEE | NextEra Energy | US Energy |

**Initial capital:** ฿1,000,000,000 (1 billion THB)

---

## Historical Data: Expected Return (μ) and Volatility (σ)

### Methodology

- **μ (Expected Return):** Arithmetic mean of 5 annual total returns (2021–2025). Total return includes price appreciation plus reinvested dividends.
- **σ (Volatility):** Annualized standard deviation of daily returns, computed as σ_daily × √252. For US-listed assets, figures are sourced from financial data providers. For SET Thai stocks where daily series are not freely accessible, σ is estimated from sector-level beta and SET index historical data.
- **Period:** January 2021 – December 2025 (5 calendar years).
- **\* = Estimated** from year-end price data and dividend history. Verify with SET.or.th, Settrade, or Bloomberg before academic publication.
- **† = Partially estimated** from year-end price history.

### μ Calculation

```
μ = (r_2021 + r_2022 + r_2023 + r_2024 + r_2025) / 5
```

| # | Ticker | 2021 | 2022 | 2023 | 2024 | 2025 | Sum | μ (p.a.) |
|---|--------|------|------|------|------|------|-----|----------|
| 1 | KBANK  | +8.0%* | -25.0%* | +15.0%* | +18.0%* | +25.0%* | +41.0% | **+8.2%*** |
| 2 | GABLE  | +10.0%* | -8.0%* | +5.0%* | +4.0%* | -5.0%* | +6.0% | **+1.2%*** |
| 3 | GULF   | +30.0%* | -20.0%* | -10.0%* | -5.0%* | -8.0%* | -13.0% | **-2.6%*** |
| 4 | BDMS   | +5.0%* | -5.0%* | +8.0%* | +5.0%* | +2.0%* | +15.0% | **+3.0%*** |
| 5 | ICHI   | +15.0%* | -5.0%* | +10.0%* | +5.0%* | -5.0%* | +20.0% | **+4.0%*** |
| 6 | CASH   | +1.5% | +1.5% | +1.5% | +1.5% | +1.5% | +7.5% | **+1.5%** |
| 7 | FSLR   | -12.5%† | +71.9% | +15.0% | +2.3% | +48.2% | +124.9% | **+25.0%** |
| 8 | PPH    | +17.8% | +2.6% | +7.0% | +8.1% | +5.5%† | +41.0% | **+8.2%** |
| 9 | GOOGL  | +65.2% | -39.1% | +58.3% | +36.0% | +66.0% | +186.4% | **+37.3%** |
| 10 | GOLD  | -3.6% | -0.4% | +13.2% | +27.2% | +26.5% | +62.9% | **+12.6%** |
| 11 | NVDA  | +124.6% | -50.3% | +239.0% | +171.3% | +38.9% | +523.5% | **+104.7%** |
| 12 | MSFT  | +52.5% | -28.0% | +58.2% | +12.9% | +15.6% | +111.2% | **+22.2%** |
| 13 | SCB   | +10.0%* | -15.0%* | -5.0%* | +5.0%* | +8.0%* | +3.0% | **+0.6%*** |
| 14 | TISCO | +8.0%* | -5.0%* | +8.0%* | +5.0%* | +5.0%* | +21.0% | **+4.2%*** |
| 15 | NEE   | +23.0% | -8.5% | -25.3% | +21.5% | +15.5% | +26.2% | **+5.2%** |

### σ Summary

| # | Ticker | σ (p.a.) | Basis | Risk Level |
|---|--------|----------|-------|------------|
| 1 | KBANK | ~22%* | SET banking sector beta ~0.7–1.0; estimated | Low–Med |
| 2 | GABLE | ~32%* | SET IT mid-cap, limited liquidity; estimated | Medium |
| 3 | GULF | ~38%* | Energy holding, high capex sensitivity; estimated | Medium |
| 4 | BDMS | ~16%* | Defensive healthcare, low beta SET50 constituent; estimated | Low |
| 5 | ICHI | ~28%* | SET small-cap beverage; estimated | Low–Med |
| 6 | CASH | 0% | Fixed deposit — no price variance (definitional) | None |
| 7 | FSLR | ~50% | Beta ~1.02 [14]; 5-yr range −12.5% to +71.9% | High |
| 8 | PPH | ~15% | 30-day historical annualized: 14.7% [8] | Low |
| 9 | GOOGL | ~30% | ABG Analytics 1-yr historical: 30.67% [12] | Medium |
| 10 | GOLD | ~13% | Long-run annualized vol 13–15% [10] | Low |
| 11 | NVDA | ~65% | ABG Analytics 1-yr: 43.32% [13]; scaled for 5-yr extreme swings | Very High |
| 12 | MSFT | ~26% | ABG Analytics 1-yr historical: 26.01% [12] | Low–Med |
| 13 | SCB | ~22%* | SET banking sector; estimated | Low–Med |
| 14 | TISCO | ~18%* | SET banking sector, lower vol than peers; estimated | Low |
| 15 | NEE | ~22% | Beta 0.75 [15]; utilities sector σ ~22% | Low–Med |

### Risk Level Classification

| Level | σ Range |
|-------|---------|
| None | 0% |
| Low | < 18% |
| Low–Med | 18% – 27% |
| Medium | 28% – 39% |
| High | 40% – 54% |
| Very High | ≥ 55% |

---

## Key Results (from last run)

### 1-Year Simulation (Current Weights)

| Scenario | Mean Value | Loss Probability | Sharpe |
|----------|-----------|-----------------|--------|
| Base     | ฿1.163B   | 22.4%           | 0.68   |
| Bull     | ฿1.200B   | 19.7%           | 0.78   |
| Bear     | ฿1.107B   | 39.2%           | 0.31   |

### 5-Year Markov Simulation — 3-Way Comparison

| Portfolio | Mean (5Y) | Loss Prob | Sharpe |
|-----------|-----------|-----------|--------|
| Current weights    | ฿2.086B | 9.3%  | 1.11 |
| 1Y Optimal weights | ฿3.029B | 5.2%  | 1.41 |
| 5Y Optimal weights | ฿2.685B | 5.1%  | 1.41 |

### 1Y Optimized Weights (Sharpe = 0.84)

| Asset | Weight |
|-------|--------|
| KBANK | 5.84% |
| GABLE | 3.92% |
| GULF  | 8.82% |
| BDMS  | 8.92% |
| ICHI  | 5.85% |
| CASH  | 2.15% |
| FSLR  | 6.55% |
| PPH   | 9.61% |
| GOOGL | 5.37% |
| GOLD  | 9.61% |
| NVDA  | 9.61% |
| MSFT  | 4.94% |
| SCB   | 8.11% |
| TISCO | 8.54% |
| NEE   | 2.15% |

### 5Y Optimized Weights (Sharpe = 1.43)

| Asset | Weight |
|-------|--------|
| KBANK | 8.16% |
| GABLE | 4.82% |
| GULF  | 7.06% |
| BDMS  | 9.48% |
| ICHI  | 9.65% |
| CASH  | 9.10% |
| FSLR  | 2.14% |
| PPH   | 9.10% |
| GOOGL | 6.24% |
| GOLD  | 9.10% |
| NVDA  | 7.00% |
| MSFT  | 6.00% |
| SCB   | 6.56% |
| TISCO | 4.26% |
| NEE   | 2.14% |

### Rolling Metric Summary (30,000 paths, 60 months)

| Metric | Current | 5Y Optimized | Edge |
|--------|---------|-------------|------|
| Total Return (1Y) | 13.4% | 18.9% | 5Y Opt |
| CAGR (3Y) | 13.3% | 18.7% | 5Y Opt |
| Sharpe Ratio | 0.57 | 0.71 | 5Y Opt |
| Max Drawdown | -11.4% | -11.6% | 5Y Opt |
| Alpha | +12.7% | +17.9% | 5Y Opt |
| Volatility (Ann.) | 19.5% | 21.4% | Current |

---

## Portfolio Constraints

These constraints mimic real-world investment mandates:

- **Thai stocks ≥ 50%** of total portfolio (regulatory requirement)
- **Each sector ≤ 25%** (Tech, Thai Banks, Healthcare, Energy, Beverage, Gold, Cash)
- **No single asset > 10%**
- **All weights sum to 1**

---

## Files

```
├── Montecarlo_GABLE.ipynb   # Main notebook — clean, documented, ready to run
├── Montecarlo_GABLE.py              # Same code as a plain Python script
└── README.md                   # This file
```

---

## How to Run

### Requirements

```bash
pip install numpy pandas matplotlib yfinance jupyter
```

### Run as a Jupyter Notebook

```bash
jupyter notebook Montecarlo_GABLE_cleaned.ipynb
```

Open in your browser, then **Run All Cells** (`Kernel → Restart & Run All`).

### Run as a Python script

```bash
python Montecarlo_GABLE.py
```

> **Note:** The script saves charts as image files and CSV exports to your `~/Downloads` folder.
> Plots will not display interactively when run as a script — use the notebook for interactive charts.

---

## About the Simulation Methods

### Monte Carlo (1-Year)

Each simulation draws 100,000 random portfolio returns from a multivariate log-normal distribution using the asset covariance matrix. Bear/Base/Bull scenarios shift the underlying parameters before sampling.

### Markov Regime-Switching (5-Year)

Each simulation path transitions between Bear, Base, and Bull regimes each year according to a transition matrix:

|        | Bear | Base | Bull |
|--------|------|------|------|
| **Bear** | 60% | 30% | 10% |
| **Base** | 20% | 60% | 20% |
| **Bull** | 10% | 30% | 60% |

This produces more realistic long-run paths than a fixed-scenario assumption.

### Optimization

The optimizer randomly samples valid portfolios (subject to all constraints) and keeps the one with the highest Sharpe ratio. 2,000 portfolios are sampled for the 1Y optimizer; 1,000 for the 5Y optimizer (each requiring a full 5Y simulation, so it takes ~3–5 minutes).

### Backtest

Historical monthly prices are downloaded from Yahoo Finance (2021–2026) using `yfinance`. Real returns, volatilities, and correlations replace the assumed parameters to assess model accuracy. When live data is unavailable, assumed parameters are used as fallback.

---

## Glossary

| Term | Meaning |
|------|---------|
| **VaR (5%)** | Value at Risk — the loss exceeded in the worst 5% of scenarios |
| **CVaR (95%)** | Conditional VaR — average loss in the worst 5% tail |
| **Sharpe Ratio** | (Return − Risk-free rate) / Volatility; higher = better risk-adjusted return |
| **CAGR** | Compound Annual Growth Rate |
| **Alpha** | Return above what the benchmark (Thai bank index) explains |
| **Beta** | Sensitivity to the benchmark's moves |
| **Max Drawdown** | Largest peak-to-trough loss within a rolling window |
| **Regime** | Economic environment: Bear (downturn), Base (normal), Bull (strong growth) |
| **μ** | Expected annual return (arithmetic mean of 5-year historical annual returns) |
| **σ** | Annualized volatility (standard deviation of daily returns × √252) |

---

## References (Vancouver Format)

1. FinanceCharts.com. NVIDIA Corp (NVDA) Performance History & Total Returns [Internet]. [cited 2026 Apr 7]. Available from: https://www.financecharts.com/stocks/NVDA/performance/total-return

2. FinanceCharts.com. Alphabet Inc (GOOGL) Performance History & Total Returns [Internet]. [cited 2026 Apr 7]. Available from: https://www.financecharts.com/stocks/GOOGL/performance/total-return

3. FinanceCharts.com. Microsoft Corp (MSFT) Performance History & Total Returns [Internet]. [cited 2026 Apr 7]. Available from: https://www.financecharts.com/stocks/MSFT/performance/total-return

4. FinanceCharts.com. First Solar Inc (FSLR) Performance History & Total Returns [Internet]. [cited 2026 Apr 7]. Available from: https://www.financecharts.com/stocks/FSLR/performance

5. MacroTrends LLC. First Solar Inc (FSLR) Stock Price History [Internet]. [cited 2026 Apr 7]. Available from: https://www.macrotrends.net/stocks/charts/FSLR/first-solar/stock-price-history

6. FinanceCharts.com. NextEra Energy Inc (NEE) Performance History & Total Returns [Internet]. [cited 2026 Apr 7]. Available from: https://www.financecharts.com/stocks/NEE/performance/total-return

7. Yahoo Finance. VanEck Pharmaceutical ETF (PPH) Performance History [Internet]. [cited 2026 Apr 7]. Available from: https://finance.yahoo.com/quote/PPH/performance/

8. Logical Invest. VanEck Pharmaceutical ETF (PPH) Analysis [Internet]. [cited 2026 Apr 7]. Available from: https://logical-invest.com/app/etf/pph/vaneck-vectors-pharmaceutical-etf

9. World of Statistics. Gold's Annual Returns from 2000 to 2025 [Internet]. [cited 2026 Apr 7]. Available from: https://x.com/stats_feed/status/1946086023680733596

10. Voigt R, Balazsy S, Greschik H. In Gold We Trust Report: Performance Table for Gold and Silver Since 1971 [Internet]. Incrementum AG; 2024 [cited 2026 Apr 7]. Available from: https://ingoldwetrust.report/chart-performance-table-gold-silver/?lang=en

11. Bank of Thailand. Deposit Interest Rates [Internet]. Bangkok: Bank of Thailand; [cited 2026 Apr 7]. Available from: https://www.bot.or.th/en/financial-institutions/financial-institutions-statistics/deposit-interest-rate.html

12. ABG Analytics. Stock Volatility Estimates and Forecasts — Standard Deviations [Internet]. [cited 2026 Apr 7]. Available from: https://www.abg-analytics.com/stock-volatilities.shtml

13. ABG Analytics. NVIDIA Corporation (NVDA) Stock Report [Internet]. [cited 2026 Apr 7]. Available from: https://abg-analytics.com/Stocks/NVDA.shtml

14. Market Chameleon. First Solar Inc (FSLR) Summary [Internet]. [cited 2026 Apr 7]. Available from: https://marketchameleon.com/Overview/FSLR/Summary

15. StockAnalysis.com. NextEra Energy (NEE) Statistics & Valuation [Internet]. [cited 2026 Apr 7]. Available from: https://stockanalysis.com/stocks/nee/statistics/

16. StockAnalysis.com. Kasikornbank PCL (BKK:KBANK) Historical Stock Price Data [Internet]. [cited 2026 Apr 7]. Available from: https://stockanalysis.com/quote/bkk/KBANK/history/

17. TradingView. Kasikornbank Public Co Ltd (SET:KBANK) [Internet]. [cited 2026 Apr 7]. Available from: https://www.tradingview.com/symbols/SET-KBANK/

18. SCB X Public Company Limited. Dividend Policy and Payments [Internet]. Bangkok: SCB X; [cited 2026 Apr 7]. Available from: https://investor.scbx.com/en/shareholder-info/dividend-policy-and-payments

19. StockAnalysis.com. TISCO Financial Group PCL (BKK:TISCO) Dividend History [Internet]. [cited 2026 Apr 7]. Available from: https://stockanalysis.com/quote/bkk/TISCO/dividend/

20. Simply Wall St. TISCO Financial Group (SET:TISCO) Stock Analysis [Internet]. [cited 2026 Apr 7]. Available from: https://simplywall.st/stocks/th/banks/set-tisco/tisco-financial-group-shares

21. Wikipedia. Stock Exchange of Thailand [Internet]. Wikimedia Foundation; 2025 [cited 2026 Apr 7]. Available from: https://en.wikipedia.org/wiki/Stock_Exchange_of_Thailand

22. The Stock Exchange of Thailand. Official Factsheets — KBANK, SCB, GULF, BDMS, ICHI, GABLE, TISCO [Internet]. Bangkok: SET; [cited 2026 Apr 7]. Available from: https://www.set.or.th/en/market/product/stock/quote/kbank/factsheet

---

*\* = Estimated from sector benchmarks and available year-end price data. Verify with SET.or.th, Settrade, or Bloomberg terminal for academic publication.*

*† = 2021 return (FSLR) and 2025 return (PPH) are partially estimated from price history and trailing trend data respectively.*

