# Monte Carlo Portfolio Optimization

A Python/Jupyter project that uses Monte Carlo simulation and Markov regime-switching to optimize a ฿1 billion (THB) multi-asset portfolio across 15 stocks spanning Thai equities, US tech, gold, and cash.

---

## What This Project Does

1. **Simulates 1-year returns** across Bear / Base / Bull economic scenarios using 20,000 Monte Carlo paths per scenario.
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
├── Montecarlo9_cleaned.ipynb   # Main notebook — clean, documented, ready to run
├── Montecarlo9.py              # Same code as a plain Python script
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
jupyter notebook Montecarlo9_cleaned.ipynb
```

Open in your browser, then **Run All Cells** (`Kernel → Restart & Run All`).

### Run as a Python script

```bash
python Montecarlo9.py
```

> **Note:** The script saves charts as image files and CSV exports to your `~/Downloads` folder.
> Plots will not display interactively when run as a script — use the notebook for interactive charts.

---

## About the Simulation Methods

### Monte Carlo (1-Year)
Each simulation draws 20,000 random portfolio returns from a multivariate log-normal distribution using the asset covariance matrix. Bear/Base/Bull scenarios shift the underlying parameters before sampling.

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
