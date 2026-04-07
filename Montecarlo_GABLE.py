#!/usr/bin/env python
# coding: utf-8

# # Monte Carlo Simulation for Portfolio Optimization
# 
# **Portfolio:** 15 assets across Thai equities, US tech, gold, and cash.
# 
# | # | Ticker | Asset | Category |
# |---|--------|-------|----------|
# | 1 | KBANK | Kasikorn Bank | Thai Bank |
# | 2 | GABLE | Gable & Frame | Thai Tech |
# | 3 | GULF | Gulf Energy | Thai Energy |
# | 4 | BDMS | Bangkok Dusit Medical | Healthcare |
# | 5 | ICHI | Ichitan Group | Beverage |
# | 6 | CASH | Cash | Cash |
# | 7 | FSLR | First Solar | US Energy |
# | 8 | PPH | VanEck Pharma ETF | Healthcare |
# | 9 | GOOGL | Alphabet | US Tech |
# | 10 | GOLD | SPDR Gold ETF | Gold |
# | 11 | NVDA | NVIDIA | US Tech |
# | 12 | MSFT | Microsoft | US Tech |
# | 13 | SCB | Siam Commercial Bank | Thai Bank |
# | 14 | TISCO | TISCO Financial | Thai Bank |
# | 15 | NEE | NextEra Energy | US Energy |
# 

# ## 1. Imports


matplotlib.use('Agg')  # non-interactive backend for script execution

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.ticker import FuncFormatter
import warnings
warnings.filterwarnings('ignore')

np.random.seed(42)


# ## 2. Portfolio Parameters
# 
# Define assets, initial weights, assumed annual returns, and volatilities.
# Initial capital: **฿1,000,000,000 (1 billion THB)**.
# 


assets = [
    "KBANK", "GABLE", "GULF",  "BDMS", "ICHI", "CASH",
    "FSLR",  "PPH",   "GOOGL", "GOLD", "NVDA", "MSFT",
    "SCB",   "TISCO", "NEE"
]

weights = np.array([
    0.10, 0.05, 0.05, 0.10, 0.05, 0.10,
    0.02, 0.10, 0.05, 0.05, 0.05, 0.05,
    0.10, 0.05, 0.08
])

# Assumed annualised expected returns
returns = np.array([
    0.082, 0.012, 0.026, 0.030, 0.040, 0.015,
    0.250, 0.082, 0.373, 0.126, 1.047, 0.222,
    0.082, 0.082, 0.052
])

# Assumed annualised volatilities
vol = np.array([
    0.22, 0.32, 0.38, 0.16, 0.28, 0.01,
    0.50, 0.15, 0.30, 0.13, 0.65, 0.26,
    0.22, 0.18, 0.22
])

initial = 1_000_000_000  # 1 billion THB


# ## 3. Asset Groups & Base Correlation
# 
# Asset groups are used for constraint checking and scenario construction.
# The base correlation matrix assumes moderate correlation (0.6) between most assets,
# with gold and cash as diversifiers.
# 


# Asset group indices
Thai_stocks = [0, 1, 2, 3, 4, 12, 13]  # KBANK, GABLE, GULF, BDMS, ICHI, SCB, TISCO
Tech        = [1, 8, 10, 11]            # GABLE, GOOGL, NVDA, MSFT
Thai_banks  = [0, 12, 13]              # KBANK, SCB, TISCO
Healthcare  = [3, 7]                   # BDMS, PPH
Energy      = [2, 6, 14]               # GULF, FSLR, NEE
Beverage    = [4]                      # ICHI
Gold        = [9]
Cash        = [5]

# Base correlation: moderate (0.6) between all assets by default
base_corr = np.full((15, 15), 0.6)
np.fill_diagonal(base_corr, 1)

# Gold: low correlation (diversifier)
base_corr[9, :] = 0.2;  base_corr[:, 9] = 0.2;  base_corr[9, 9] = 1

# Cash: zero correlation
base_corr[5, :] = 0;    base_corr[:, 5] = 0;    base_corr[5, 5] = 1


# ## 4. Core Helper Functions
# 
# ### 4a. Portfolio Summary
# Prints key stats: mean, median, VaR (5%), and Sharpe ratio.
# 


def summarize(name, data, horizon_years=1):
    print(f"\n===== {name} =====")
    print(f"Mean:      {np.mean(data):,.0f}")
    print(f"Median:    {np.median(data):,.0f}")
    print(f"Min:       {np.min(data):,.0f}")
    print(f"Max:       {np.max(data):,.0f}")
    print(f"Loss Prob: {(data < initial).mean()*100:.2f}%")

    var_5 = np.percentile(data, 5)
    print(f"VaR (5%):  {initial - var_5:,.0f}")

    valid   = data > 0
    ann_ret = (data[valid] / initial) ** (1 / horizon_years) - 1
    sharpe  = (np.mean(ann_ret) - 0.02) / np.std(ann_ret)
    print(f"Sharpe:    {sharpe:.2f}")


# ### 4b. Scenario Builder
# 
# Applies bull/bear economic shocks to returns, volatilities, and correlations.
# 
# - **Bull:** tech and bank returns increase; correlations rise slightly
# - **Bear:** returns fall broadly; gold rises; correlations spike (contagion)
# - **Base:** no adjustment (returns/vol unchanged)
# 


def build_scenario(r_in, v_in, c_in, scenario="base"):
    r = r_in.copy()
    v = v_in.copy()
    c = c_in.copy()

    if scenario == "bull":
        r[Tech]       *= 1.3
        r[Thai_banks] *= 1.2
        r[Energy]     *= 1.15
        r[Healthcare] *= 1.05
        r[Gold]       *= 0.97
        v             *= 1.05
        c = np.clip(c + 0.03, -1, 1)

    elif scenario == "bear":
        r[Tech]       -= 0.12
        r[Thai_banks] -= 0.10
        r[Energy]     -= 0.10
        r[Beverage]   -= 0.08
        r[Healthcare] *= 0.95
        r[Gold]       += 0.05
        v             *= 1.3

        c = np.full_like(c, 0.75)
        np.fill_diagonal(c, 1)
        c[9, :] = 0.2;  c[:, 9] = 0.2;  c[9, 9] = 1
        c[5, :] = 0;    c[:, 5] = 0;    c[5, 5] = 1

    return r, v, c


# ### 4c. Simulation Engine
# 
# Runs `n_sim` Monte Carlo draws from a multivariate log-normal distribution.
# Returns final portfolio values after 1 year.
# 


def simulate(w, r, v, c, n_sim=20_000):
    """
    w : weights array
    r : expected returns (annual)
    v : volatilities (annual)
    c : correlation matrix
    Returns: array of final portfolio values (n_sim,)
    """
    cov          = np.outer(v, v) * c
    rand         = np.random.multivariate_normal(r, cov, n_sim)
    port_returns = np.exp(rand @ w) - 1
    port_returns = np.maximum(port_returns, -0.99)
    return initial * (1 + port_returns)


# ## 5. 1-Year Simulation — Bear / Base / Bull Scenarios
# 
# Run 20,000 simulations for each scenario and summarise the distribution.
# 


np.random.seed(42)

base_r, base_v, base_c = build_scenario(returns, vol, base_corr, "base")
bull_r, bull_v, bull_c = build_scenario(returns, vol, base_corr, "bull")
bear_r, bear_v, bear_c = build_scenario(returns, vol, base_corr, "bear")

base = simulate(weights, base_r, base_v, base_c)
bull = simulate(weights, bull_r, bull_v, bull_c)
bear = simulate(weights, bear_r, bear_v, bear_c)

summarize("Base (1Y)", base)
summarize("Bull (1Y)", bull)
summarize("Bear (1Y)", bear)


# ### 1-Year Distribution Plot


plt.figure(figsize=(12, 7))

plt.hist(base, bins=80, alpha=0.5, label="Base")
plt.hist(bull, bins=80, alpha=0.5, label="Bull")
plt.hist(bear, bins=80, alpha=0.5, label="Bear")

plt.axvline(initial,       color="black",  linestyle="--", linewidth=2, label="Initial")
plt.axvline(np.mean(base), linewidth=2, label="Mean Base")
plt.axvline(np.mean(bull), linewidth=2, label="Mean Bull")
plt.axvline(np.mean(bear), linewidth=2, label="Mean Bear")

plt.title("Monte Carlo Simulation (1 Year) — Bear / Base / Bull")
plt.xlabel("Portfolio Value (THB)")
plt.ylabel("Frequency")
plt.grid(alpha=0.3)
plt.legend()
plt.tight_layout()
plt.show()


# ## 6. CVaR Stress Test
# 
# **Conditional VaR (Expected Shortfall):** the average loss beyond the 95th percentile tail.
# A higher CVaR value means greater downside risk in the worst-case scenarios.
# 


def compute_cvar(data, confidence=0.95):
    threshold   = np.percentile(data, (1 - confidence) * 100)
    tail_losses = data[data <= threshold]
    return initial - np.mean(tail_losses)

scenarios_data = {"Base": base, "Bull": bull, "Bear": bear}

print("===== CVaR (95%) STRESS TEST =====")
for name, data in scenarios_data.items():
    var  = initial - np.percentile(data, 5)
    cvar = compute_cvar(data)
    prob = (data < initial).mean() * 100
    print(f"\n{name}")
    print(f"  VaR  (5%):  {var:>15,.0f} THB")
    print(f"  CVaR (95%): {cvar:>15,.0f} THB")
    print(f"  Loss Prob:  {prob:.2f}%")


# ## 7. Portfolio Constraints & Weight Generator
# 
# **Constraints enforced:**
# - Thai stocks ≥ 50% of portfolio
# - Each sector ≤ 25%
# - No single asset > 10%
# - All weights sum to 1
# 
# `generate_constrained_weights()` samples valid random portfolios for optimization.
# 


def check_constraints(w):
    if np.sum(w[Thai_stocks]) < 0.50:
        return False
    for group in [Tech, Thai_banks, Healthcare, Energy, Beverage, Gold, Cash]:
        if np.sum(w[group]) > 0.25 + 1e-6:
            return False
    if np.any(w > 0.10 + 1e-6):
        return False
    return True


def generate_constrained_weights():
    """
    Generates a random weight vector satisfying all portfolio constraints.
    Uses a Dirichlet sample + iterative correction strategy.
    Falls back to a hardcoded valid portfolio after 1000 failed attempts.
    """
    for _ in range(1000):
        w = np.random.dirichlet(np.ones(15))
        w = np.clip(w, 0.02, 0.10)
        w /= w.sum()

        thai_sum = w[Thai_stocks].sum()
        if thai_sum < 0.50:
            deficit  = 0.50 - thai_sum
            headroom = np.maximum(0.10 - w[Thai_stocks], 0)
            if headroom.sum() < deficit:
                continue
            w[Thai_stocks] += headroom * (deficit / headroom.sum())
            w[Thai_stocks]  = np.clip(w[Thai_stocks], 0.02, 0.10)

        total = w.sum()
        if total > 1.0 + 1e-9:
            excess     = total - 1.0
            non_thai   = [i for i in range(15) if i not in Thai_stocks]
            shrinkable = np.maximum(w[non_thai] - 0.02, 0)
            if shrinkable.sum() < excess:
                continue
            w[non_thai] -= shrinkable * (excess / shrinkable.sum())
            w[non_thai]  = np.clip(w[non_thai], 0.02, 0.10)

        w /= w.sum()

        for _ in range(3):  # 3 passes for convergence
            for group in [Tech, Thai_banks, Healthcare, Energy, Beverage, Gold, Cash]:
                gs = w[group].sum()
                if gs > 0.25 + 1e-9:
                    w[group] *= 0.25 / gs
            w /= w.sum()

        w = np.clip(w, 0.0, 0.10)
        w /= w.sum()

        if check_constraints(w) and abs(w.sum() - 1.0) < 1e-6:
            return w

    # Fallback: verified valid portfolio
    w = np.array([
        0.09, 0.07, 0.07, 0.08, 0.07, 0.05,
        0.03, 0.09, 0.04, 0.04, 0.04, 0.04,
        0.09, 0.08, 0.04
    ], dtype=float)
    w /= w.sum()
    return w


# Sanity check
print("Sanity checking constraint generator...")
fails, max_w_seen, min_thai = 0, 0.0, 1.0
for _ in range(1000):
    w = generate_constrained_weights()
    if not check_constraints(w):
        fails += 1
    max_w_seen = max(max_w_seen, w.max())
    min_thai   = min(min_thai, w[Thai_stocks].sum())

print(f"Failed        : {fails}/1000  (should be 0)")
print(f"Max weight    : {max_w_seen:.4%}  (should be ≤ 10%)")
print(f"Min Thai alloc: {min_thai:.4%}  (should be ≥ 50%)")


# ## 8. 1-Year Portfolio Optimization
# 
# ### 8a. Sharpe-Based Optimizer (1Y)
# 
# Randomly samples 2,000 valid portfolios and selects the one with the highest Sharpe ratio
# over a 1-year Monte Carlo base scenario.
# 


def evaluate_1y(w):
    data    = simulate(w, base_r, base_v, base_c)
    ann_ret = (data / initial) - 1
    return (np.mean(ann_ret) - 0.02) / np.std(ann_ret)

def optimize_1y(n_portfolios=2_000):
    best_sharpe, best_weights = -np.inf, None
    print(f"Running 1Y optimization — {n_portfolios} valid portfolios...")
    for i in range(n_portfolios):
        w      = generate_constrained_weights()
        sharpe = evaluate_1y(w)
        if sharpe > best_sharpe:
            best_sharpe  = sharpe
            best_weights = w.copy()
        if (i + 1) % 500 == 0:
            print(f"  {i+1}/{n_portfolios} done — best Sharpe: {best_sharpe:.3f}")
    print(f"\nDone.")
    return best_weights, best_sharpe

np.random.seed(42)
best_w_1y, best_sharpe_1y = optimize_1y()

print("\n===== 1-YEAR OPTIMAL WEIGHTS =====")
for asset, w in zip(assets, best_w_1y):
    print(f"{asset}: {w:.2%}")

current_sharpe = evaluate_1y(weights)
print(f"\nCurrent Sharpe:   {current_sharpe:.2f}")
print(f"Optimized Sharpe: {best_sharpe_1y:.2f}")


# ### 8b. Efficient Frontier
# 
# Plots the risk–return frontier across 2,000 random valid portfolios.
# The optimal portfolio (highest simulation-based Sharpe) is highlighted in red.
# 


def portfolio_performance(w, r, cov):
    return np.dot(w, r), np.sqrt(w @ cov @ w)

def optimize_portfolio(n_portfolios=2_000):
    cov = np.outer(vol, vol) * base_corr
    results, weight_list = [], []
    for _ in range(n_portfolios):
        w = generate_constrained_weights()
        data    = simulate(w, base_r, base_v, base_c)
        ann_ret = (data / initial) - 1
        sharpe  = (np.mean(ann_ret) - 0.02) / np.std(ann_ret)
        ret, vol_ = portfolio_performance(w, returns, cov)
        results.append([ret, vol_, sharpe])
        weight_list.append(w)
    results  = np.array(results)
    best_idx = np.argmax(results[:, 2])
    return results, weight_list, best_idx

np.random.seed(42)
results, weight_list, best_idx = optimize_portfolio()
best_weights                       = weight_list[best_idx]
best_return, best_vol, best_sharpe = results[best_idx]

print("\n===== OPTIMAL PORTFOLIO (Efficient Frontier) =====")
print(f"Return:     {best_return:.2%}")
print(f"Volatility: {best_vol:.2%}")
print(f"Sharpe:     {best_sharpe:.2f}")
print("\n===== OPTIMAL WEIGHTS =====")
for asset, w in zip(assets, best_weights):
    print(f"{asset}: {w:.2%}")

plt.figure(figsize=(10, 6))
plt.scatter(results[:, 1], results[:, 0], c=results[:, 2], alpha=0.5, cmap="RdYlGn")
plt.colorbar(label="Sharpe Ratio")
plt.scatter(best_vol, best_return, color="red", s=150, zorder=5, label="Optimal")
plt.xlabel("Volatility")
plt.ylabel("Return")
plt.title("Efficient Frontier (Simulation-based Sharpe)")
plt.legend()
plt.tight_layout()
plt.show()


# ### 8c. Current vs 1Y Optimized Portfolio — Distribution Comparison


np.random.seed(42)
opt_sim_1y = simulate(best_w_1y, base_r, base_v, base_c)

plt.figure(figsize=(12, 6))
plt.hist(base,       bins=80, alpha=0.5, label="Current Weights (Base)", color="steelblue")
plt.hist(opt_sim_1y, bins=80, alpha=0.5, label="Optimized Weights (Base)", color="green")
plt.axvline(initial,             color="black",  linestyle="--", linewidth=2, label="Initial Capital")
plt.axvline(np.mean(base),       color="blue",   linewidth=2, linestyle=":", label="Current Mean")
plt.axvline(np.mean(opt_sim_1y), color="green",  linewidth=2, linestyle=":", label="Optimized Mean")
plt.title("Current vs 1Y Optimized Portfolio — Monte Carlo Base Scenario")
plt.xlabel("Portfolio Value (THB)")
plt.ylabel("Frequency")
plt.legend()
plt.grid(alpha=0.3)
plt.tight_layout()
plt.show()


# ## 9. 5-Year Markov Regime-Switching Simulation
# 
# Uses a 3-state Markov chain (Bear / Base / Bull) to simulate realistic economic regime changes
# over a 5-year horizon. The transition matrix defines year-to-year regime probabilities.
# 
# | From \ To | Bear | Base | Bull |
# |-----------|------|------|------|
# | **Bear**  | 0.6  | 0.3  | 0.1  |
# | **Base**  | 0.2  | 0.6  | 0.2  |
# | **Bull**  | 0.1  | 0.3  | 0.6  |
# 


transition_matrix = np.array([
    [0.6, 0.3, 0.1],
    [0.2, 0.6, 0.2],
    [0.1, 0.3, 0.6]
])

def simulate_markov_5y(w=None, years=5, n_sim=100_000):
    """Run 5Y Markov regime-switching simulation. Uses portfolio weights by default."""
    if w is None:
        w = weights

    values    = np.full(n_sim, float(initial))
    scenarios = {
        0: build_scenario(returns, vol, base_corr, "bear"),
        1: build_scenario(returns, vol, base_corr, "base"),
        2: build_scenario(returns, vol, base_corr, "bull")
    }
    states = np.full(n_sim, 1, dtype=int)

    for _ in range(years):
        new_states = np.zeros(n_sim, dtype=int)
        for s in [0, 1, 2]:
            idx = states == s
            if idx.sum() == 0:
                continue
            new_states[idx] = np.random.choice([0, 1, 2], size=idx.sum(),
                                               p=transition_matrix[s])
        states = new_states

        port_returns = np.zeros(n_sim)
        for s in [0, 1, 2]:
            idx = states == s
            if idx.sum() == 0:
                continue
            r, v, c = scenarios[s]
            cov  = np.outer(v, v) * c
            rand = np.random.multivariate_normal(r, cov, idx.sum())
            port_returns[idx] = np.exp(rand @ w) - 1

        port_returns = np.maximum(port_returns, -0.99)
        values      *= (1 + port_returns)

    return values

np.random.seed(42)
markov_5y = simulate_markov_5y()
summarize("Markov Regime (5Y)", markov_5y, horizon_years=5)

plt.figure(figsize=(12, 7))
plt.hist(markov_5y, bins=80)
plt.axvline(initial,            color="black",  linestyle="--", linewidth=2, label="Initial")
plt.axvline(np.mean(markov_5y), color="orange", linewidth=2, label="Mean")
plt.title("Monte Carlo — 5Y Markov Regime Switching (Current Weights)")
plt.xlabel("Portfolio Value (THB)")
plt.ylabel("Frequency")
plt.grid(alpha=0.3)
plt.legend()
plt.tight_layout()
plt.show()


# ### 9a. Best 5Y Outcome Analysis — Year 1 Breakdown
# 
# Examines what Year 1 returns look like for the top 5% of 5-year simulations.
# This helps understand what early-year conditions lead to the best long-term outcomes.
# 


def analyze_best_5y_year1(top_n=0.05, n_sim=100_000):
    """
    Re-runs the 5Y Markov simulation, tracking Year 1 separately.
    Compares Year 1 return distribution of top 5% final outcomes vs all simulations.
    """
    values    = np.full(n_sim, float(initial))
    year1_end = np.zeros(n_sim)
    scenarios = {
        0: build_scenario(returns, vol, base_corr, "bear"),
        1: build_scenario(returns, vol, base_corr, "base"),
        2: build_scenario(returns, vol, base_corr, "bull")
    }
    states = np.full(n_sim, 1, dtype=int)

    for year in range(5):
        new_states = np.zeros(n_sim, dtype=int)
        for s in [0, 1, 2]:
            idx = states == s
            if idx.sum() == 0:
                continue
            new_states[idx] = np.random.choice([0, 1, 2], size=idx.sum(),
                                               p=transition_matrix[s])
        states = new_states

        port_returns = np.zeros(n_sim)
        for s in [0, 1, 2]:
            idx = states == s
            if idx.sum() == 0:
                continue
            r, v, c = scenarios[s]
            cov  = np.outer(v, v) * c
            rand = np.random.multivariate_normal(r, cov, idx.sum())
            port_returns[idx] = np.exp(rand @ weights) - 1

        port_returns = np.maximum(port_returns, -0.99)
        values      *= (1 + port_returns)
        if year == 0:
            year1_end = values.copy()

    threshold          = np.percentile(values, (1 - top_n) * 100)
    best_mask          = values >= threshold
    best_year1_returns = (year1_end[best_mask] / initial) - 1
    all_year1_returns  = (year1_end / initial) - 1

    print(f"\n===== YEAR 1 RETURNS — TOP {top_n*100:.0f}% BEST 5Y CASES =====")
    print(f"Simulations in group  : {best_mask.sum():,}")
    print(f"Mean  Year 1 return   : {np.mean(best_year1_returns):.2%}")
    print(f"Median Year 1 return  : {np.median(best_year1_returns):.2%}")
    print(f"% with positive Y1    : {(best_year1_returns > 0).mean()*100:.1f}%")

    plt.figure(figsize=(12, 6))
    plt.hist(all_year1_returns,  bins=80, alpha=0.4, label="All simulations",              color="steelblue")
    plt.hist(best_year1_returns, bins=80, alpha=0.6, label=f"Top {top_n*100:.0f}% (best 5Y)", color="green")
    plt.axvline(np.mean(all_year1_returns),  color="blue",  linewidth=2, linestyle="--", label="Mean (all)")
    plt.axvline(np.mean(best_year1_returns), color="green", linewidth=2, linestyle="--", label="Mean (top 5Y)")
    plt.axvline(0, color="black", linewidth=1.5, linestyle=":", label="Break-even")
    plt.title(f"Year 1 Return Distribution — Top {top_n*100:.0f}% Best 5Y Outcomes")
    plt.xlabel("Year 1 Return")
    plt.ylabel("Frequency")
    plt.legend()
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.show()
    return best_year1_returns

np.random.seed(42)
best_y1 = analyze_best_5y_year1(top_n=0.05)


# ## 10. 5-Year Portfolio Optimization
# 
# ### 10a. Sharpe-Based Optimizer (5Y)
# 
# Evaluates each candidate portfolio using the full 5Y Markov simulation.
# Scores by annualised Sharpe ratio over the 5-year horizon.
# 
# > ⏱ Expect ~3–5 minutes for 1,000 portfolios.
# 


def evaluate_5y(w, n_sim=20_000):
    """Score weights using full 5Y Markov simulation — annualised Sharpe."""
    values    = np.full(n_sim, float(initial))
    scenarios = {
        0: build_scenario(returns, vol, base_corr, "bear"),
        1: build_scenario(returns, vol, base_corr, "base"),
        2: build_scenario(returns, vol, base_corr, "bull")
    }
    states = np.full(n_sim, 1, dtype=int)

    for _ in range(5):
        new_states = np.zeros(n_sim, dtype=int)
        for s in [0, 1, 2]:
            idx = states == s
            if idx.sum() == 0:
                continue
            new_states[idx] = np.random.choice([0, 1, 2], size=idx.sum(),
                                               p=transition_matrix[s])
        states = new_states

        port_returns = np.zeros(n_sim)
        for s in [0, 1, 2]:
            idx = states == s
            if idx.sum() == 0:
                continue
            r, v, c = scenarios[s]
            cov  = np.outer(v, v) * c
            rand = np.random.multivariate_normal(r, cov, idx.sum())
            port_returns[idx] = np.exp(rand @ w) - 1

        port_returns = np.maximum(port_returns, -0.99)
        values      *= (1 + port_returns)

    ann_ret = (values / initial) ** (1/5) - 1
    return (np.mean(ann_ret) - 0.02) / np.std(ann_ret)


def optimize_5y(n_portfolios=1_000):
    best_sharpe, best_weights = -np.inf, None
    print(f"Running 5Y optimization — {n_portfolios} portfolios...")
    for i in range(n_portfolios):
        w      = generate_constrained_weights()
        sharpe = evaluate_5y(w)
        if sharpe > best_sharpe:
            best_sharpe  = sharpe
            best_weights = w.copy()
        if (i + 1) % 100 == 0:
            print(f"  {i+1}/{n_portfolios} done — best Sharpe: {best_sharpe:.3f}")
    print(f"\nDone — all {n_portfolios} portfolios were valid.")
    return best_weights, best_sharpe

np.random.seed(42)
best_w_5y, best_sharpe_5y = optimize_5y(n_portfolios=1_000)


# ### 10b. Results — 3-Way Weight & Performance Comparison
# 
# Compare current, 1Y optimal, and 5Y optimal weights side-by-side,
# then simulate all three over 5 years with the same random seed for a fair comparison.
# 


# Constraint check on 5Y weights
print("===== CONSTRAINT CHECK (5Y Optimal) =====")
print(f"Thai allocation : {np.sum(best_w_5y[Thai_stocks]):.2%}  (min 50%)")
for name, group in [("Tech", Tech), ("Thai banks", Thai_banks),
                     ("Healthcare", Healthcare), ("Energy", Energy),
                     ("Beverage (ICHI)", Beverage), ("Gold", Gold), ("Cash", Cash)]:
    print(f"{name:<16}: {np.sum(best_w_5y[group]):.2%}  (max 25%)")
print(f"Weights sum to  : {np.sum(best_w_5y):.4f}")

# Weight comparison table
print("\n===== WEIGHT COMPARISON =====")
print(f"{'Asset':<8} {'Current':>10} {'1Y Optimal':>12} {'5Y Optimal':>12} {'Diff(5Y-1Y)':>13}")
print("-" * 58)
for i, asset in enumerate(assets):
    diff = best_w_5y[i] - best_w_1y[i]
    flag = " ←" if abs(diff) > 0.05 else ""
    print(f"{asset:<8} {weights[i]:>10.2%} {best_w_1y[i]:>12.2%} {best_w_5y[i]:>12.2%} {diff:>+12.2%}{flag}")

# Simulate all three with same seed
np.random.seed(42); sim_current = simulate_markov_5y(weights)
np.random.seed(42); sim_1y_opt  = simulate_markov_5y(best_w_1y)
np.random.seed(42); sim_5y_opt  = simulate_markov_5y(best_w_5y)

print()
summarize("Current Weights   (5Y)", sim_current, horizon_years=5)
summarize("1Y Optimal Weights(5Y)", sim_1y_opt,  horizon_years=5)
summarize("5Y Optimal Weights(5Y)", sim_5y_opt,  horizon_years=5)

plt.figure(figsize=(12, 6))
plt.hist(sim_current, bins=80, alpha=0.4, label="Current Weights",    color="steelblue")
plt.hist(sim_1y_opt,  bins=80, alpha=0.4, label="1Y Optimal Weights", color="orange")
plt.hist(sim_5y_opt,  bins=80, alpha=0.5, label="5Y Optimal Weights", color="green")
plt.axvline(initial,              color="black",     linestyle="--", linewidth=2,  label="Initial Capital")
plt.axvline(np.mean(sim_current), color="steelblue", linewidth=2, linestyle=":",   label="Mean Current")
plt.axvline(np.mean(sim_1y_opt),  color="orange",    linewidth=2, linestyle=":",   label="Mean 1Y Opt")
plt.axvline(np.mean(sim_5y_opt),  color="green",     linewidth=2, linestyle=":",   label="Mean 5Y Opt")
plt.title("Current vs 1Y Optimal vs 5Y Optimal — 5Y Markov Simulation")
plt.xlabel("Portfolio Value (THB)")
plt.ylabel("Frequency")
plt.legend()
plt.grid(alpha=0.3)
plt.tight_layout()
plt.show()


# ## 11. Historical Backtest (2021–2026)
# 
# Download monthly price data from Yahoo Finance and compute real historical returns,
# volatilities, and correlations for all 15 assets.
# 
# > **Note:** CASH has no market ticker and uses assumed returns throughout.
# 


import yfinance as yf

ticker_map = {
    "KBANK" : "KBANK.BK",  "GABLE" : "GABLE.BK",  "GULF"  : "GULF.BK",
    "BDMS"  : "BDMS.BK",   "ICHI"  : "ICHI.BK",    "CASH"  : None,
    "FSLR"  : "FSLR",      "PPH"   : "PPH",         "GOOGL" : "GOOGL",
    "GOLD"  : "GLD",        "NVDA"  : "NVDA",        "MSFT"  : "MSFT",
    "SCB"   : "SCB.BK",    "TISCO" : "TISCO.BK",    "NEE"   : "NEE"
}

START = "2021-01-01"
END   = "2026-06-01"

price_data = {}
for name, ticker in ticker_map.items():
    if ticker is None:
        price_data[name] = None
        continue
    try:
        df = yf.download(ticker, start=START, end=END,
                         interval="1d", auto_adjust=True, progress=False)
        if len(df) == 0:
            price_data[name] = None
            continue
        if isinstance(df.columns, pd.MultiIndex):
            df.columns = df.columns.droplevel(1)
        monthly = df["Close"].resample("ME").last().dropna()
        if len(monthly) < 12:
            price_data[name] = None
            continue
        price_data[name] = monthly
        print(f"✓ {name:8s} ({ticker}) — {len(monthly)} months")
    except Exception:
        price_data[name] = None

downloaded = [k for k, v in price_data.items() if v is not None]
print(f"Downloaded: {len(downloaded)} assets | Using assumed data for the rest")


# ### 11a. Build Return Matrix
# 
# Compute monthly returns for successfully downloaded assets.
# Failed/missing assets use their assumed monthly return instead.
# 


import pandas as pd, numpy as np
from datetime import date
from dateutil.relativedelta import relativedelta

series_dict, failed_assets = {}, []

for name, series in price_data.items():
    if series is None:
        failed_assets.append(name)
    else:
        monthly_ret = series.pct_change()
        if isinstance(monthly_ret, pd.DataFrame):
            monthly_ret = monthly_ret.squeeze()
        series_dict[name] = monthly_ret

print(f"Successfully downloaded : {list(series_dict.keys())}")
print(f"Failed / using assumed  : {failed_assets}")

# Build return dataframe
if series_dict:
    ret_df = pd.DataFrame(series_dict)
else:
    # No live data — generate synthetic monthly dates (2021-01 to 2026-06)
    dates = pd.date_range("2021-01-31", periods=66, freq="ME")
    ret_df = pd.DataFrame(index=dates)

# Fill failed assets with assumed monthly return
for name in failed_assets:
    asset_idx       = assets.index(name)
    assumed_monthly = returns[asset_idx] / 12
    ret_df[name]    = assumed_monthly
    print(f"  {name}: using assumed monthly return {assumed_monthly:.4f}")

ret_df["CASH"] = 0.0
ret_df = ret_df.sort_index().ffill().fillna(0)[assets]

assert not ret_df.isna().any().any(), "Still contains NaN!"
assert not np.isinf(ret_df.values).any(), "Contains Inf!"

print(f"\nReturn matrix shape : {ret_df.shape}")
if len(ret_df) > 0:
    print(f"Date range          : {ret_df.index[0].date()} → {ret_df.index[-1].date()}")
    print(f"Total years         : {len(ret_df)/12:.2f}")

# Annualise real statistics
real_returns_series = ret_df.mean() * 12
real_vol_series     = ret_df.std()  * np.sqrt(12)

# Override volatility for assets without real data
for name in failed_assets + ["CASH"]:
    real_vol_series[name] = vol[assets.index(name)]

# Real correlation matrix (numerically cleaned)
real_corr_matrix = ret_df.corr().values
real_corr_matrix = (real_corr_matrix + real_corr_matrix.T) / 2
np.fill_diagonal(real_corr_matrix, 1)
real_corr_matrix = np.nan_to_num(real_corr_matrix)

print("\n===== REAL vs ASSUMED RETURNS =====")
print(f"{'Asset':<8} {'Assumed':>10} {'Real':>10} {'Diff':>10} {'Source':>10}")
print("-" * 52)
for i, asset in enumerate(assets):
    assumed = returns[i]
    real    = real_returns_series[asset]
    diff    = real - assumed
    source  = "assumed" if asset in failed_assets or asset == "CASH" else "real"
    flag    = " ⚠️" if abs(diff) > 0.15 and source == "real" else ""
    print(f"{asset:<8} {assumed:>10.2%} {real:>10.2%} {diff:>+10.2%} {source:>10}{flag}")


# ### 11b. Assumed vs Real Correlation Heatmap


fig, axes = plt.subplots(1, 2, figsize=(16, 6))
cmap = plt.cm.RdYlGn

im1 = axes[0].imshow(base_corr, cmap=cmap, vmin=-1, vmax=1)
axes[0].set_title("Assumed Correlation")
axes[0].set_xticks(range(len(assets))); axes[0].set_xticklabels(assets, rotation=45, ha="right")
axes[0].set_yticks(range(len(assets))); axes[0].set_yticklabels(assets)
plt.colorbar(im1, ax=axes[0])

im2 = axes[1].imshow(real_corr_matrix, cmap=cmap, vmin=-1, vmax=1)
axes[1].set_title("Real Correlation (Historical)")
axes[1].set_xticks(range(len(assets))); axes[1].set_xticklabels(assets, rotation=45, ha="right")
axes[1].set_yticks(range(len(assets))); axes[1].set_yticklabels(assets)
plt.colorbar(im2, ax=axes[1])

plt.suptitle("Assumed vs Real Correlation Matrix", fontsize=14)
plt.tight_layout()
plt.show()


# ### 11c. Historical Portfolio Performance


port_monthly    = ret_df[assets].values @ weights
port_cumulative = (1 + port_monthly).cumprod()

total_return      = port_cumulative[-1] - 1
n_years           = len(port_monthly) / 12
annualised_return = (1 + total_return) ** (1 / n_years) - 1
annualised_vol    = port_monthly.std() * np.sqrt(12)
sharpe            = (annualised_return - 0.02) / annualised_vol

print("===== HISTORICAL PORTFOLIO PERFORMANCE =====")
print(f"Period            : {ret_df.index[0].date()} → {ret_df.index[-1].date()}")
print(f"Total Return      : {total_return:.2%}")
print(f"Annualised Return : {annualised_return:.2%}")
print(f"Annualised Vol    : {annualised_vol:.2%}")
print(f"Sharpe Ratio      : {sharpe:.2f}")
print(f"Final Value (1B)  : {initial * port_cumulative[-1]:,.0f} THB")

plt.figure(figsize=(12, 5))
plt.plot(ret_df.index, port_cumulative * initial, color="steelblue", linewidth=2)
plt.axhline(initial, color="black", linestyle="--", linewidth=1.5, label="Initial Capital")
plt.title("Historical Portfolio Value (2021–2026)")
plt.xlabel("Date")
plt.ylabel("Portfolio Value (THB)")
plt.grid(alpha=0.3)
plt.legend()
plt.tight_layout()
plt.show()


# ## 12. Real vs Assumed Simulation
# 
# ### 12a. 1-Year Comparison
# 
# Re-run the 1-year Monte Carlo using real historical inputs (returns, vol, correlation)
# and compare the distribution against the assumed-input simulation.
# 


real_returns_arr = real_returns_series[assets].values
real_vol_arr     = real_vol_series[assets].values

np.random.seed(42)
real_base = simulate(weights, real_returns_arr, real_vol_arr, real_corr_matrix)

print("\n===== BACKTEST-BASED SIMULATION (1Y Base) =====")
summarize("Real Stats (1Y)",    real_base)
summarize("Assumed Stats (1Y)", base)

plt.figure(figsize=(12, 6))
plt.hist(base,      bins=80, alpha=0.5, label="Assumed inputs",        color="steelblue")
plt.hist(real_base, bins=80, alpha=0.5, label="Real historical inputs", color="darkorange")
plt.axvline(initial,            color="black",      linestyle="--", linewidth=2, label="Initial Capital")
plt.axvline(np.mean(base),      color="steelblue",  linewidth=2, linestyle=":", label="Mean (Assumed)")
plt.axvline(np.mean(real_base), color="darkorange", linewidth=2, linestyle=":", label="Mean (Real)")
plt.title("Assumed vs Real Historical Inputs — 1Y Monte Carlo")
plt.xlabel("Portfolio Value (THB)")
plt.ylabel("Frequency")
plt.legend()
plt.grid(alpha=0.3)
plt.tight_layout()
plt.show()


# ### 12b. 5-Year Markov Comparison
# 
# Uses a custom Markov simulator that accepts externally-built scenario parameters,
# so we can plug in real historical statistics for bear/base/bull shocks.
# 


def simulate_markov_5y_custom(w, bear_args, base_args, bull_args, years=5, n_sim=100_000):
    """5Y Markov simulator accepting custom scenario parameter tuples (r, v, c)."""
    values    = np.full(n_sim, float(initial))
    scenarios = {0: bear_args, 1: base_args, 2: bull_args}
    states    = np.full(n_sim, 1, dtype=int)

    for _ in range(years):
        new_states = np.zeros(n_sim, dtype=int)
        for s in [0, 1, 2]:
            idx = states == s
            if idx.sum() == 0: continue
            new_states[idx] = np.random.choice([0, 1, 2], size=idx.sum(),
                                               p=transition_matrix[s])
        states = new_states

        port_returns = np.zeros(n_sim)
        for s in [0, 1, 2]:
            idx = states == s
            if idx.sum() == 0: continue
            r, v, c = scenarios[s]
            cov  = np.outer(v, v) * c
            rand = np.random.multivariate_normal(r, cov, idx.sum())
            port_returns[idx] = np.exp(rand @ w) - 1

        port_returns = np.maximum(port_returns, -0.99)
        values      *= (1 + port_returns)

    return values


# Build real-input scenarios using the same shock logic
real_bear_args = build_scenario(real_returns_arr, real_vol_arr, real_corr_matrix, "bear")
real_base_args = build_scenario(real_returns_arr, real_vol_arr, real_corr_matrix, "base")
real_bull_args = build_scenario(real_returns_arr, real_vol_arr, real_corr_matrix, "bull")

np.random.seed(42); assumed_5y = simulate_markov_5y(weights)
np.random.seed(42); real_5y    = simulate_markov_5y_custom(weights, real_bear_args, real_base_args, real_bull_args)

print("\n===== BACKTEST-BASED SIMULATION (5Y Markov Regime) =====")
summarize("Real Stats (5Y)",    real_5y,    horizon_years=5)
summarize("Assumed Stats (5Y)", assumed_5y, horizon_years=5)

plt.figure(figsize=(12, 6))
plt.hist(assumed_5y, bins=80, alpha=0.5, label="Assumed inputs",        color="steelblue")
plt.hist(real_5y,    bins=80, alpha=0.5, label="Real historical inputs", color="darkorange")
plt.axvline(initial,             color="black",      linestyle="--", linewidth=2, label="Initial Capital")
plt.axvline(np.mean(assumed_5y), color="steelblue",  linewidth=2, linestyle=":", label="Mean (Assumed)")
plt.axvline(np.mean(real_5y),    color="darkorange", linewidth=2, linestyle=":", label="Mean (Real)")
plt.title("Assumed vs Real Historical Inputs — 5Y Markov Regime Monte Carlo")
plt.xlabel("Portfolio Value (THB)")
plt.ylabel("Frequency")
plt.legend()
plt.grid(alpha=0.3)
plt.tight_layout()
plt.show()


# ## 13. Top 100 Simulation Analysis
# 
# ### 13a. Markov Simulator with Return Tracking
# 
# Extended version of the 5Y simulator that records the final-year return matrix,
# used to analyse which assets drove the best outcomes.
# 


def simulate_markov_5y_tracked(w, bear_args, base_args, bull_args, years=5, n_sim=100_000):
    """
    5Y Markov simulator that also returns the final-year asset log-returns.
    Returns: (final_values, last_year_rand_matrix)
    """
    values    = np.full(n_sim, float(initial))
    scenarios = {0: bear_args, 1: base_args, 2: bull_args}
    states    = np.full(n_sim, 1, dtype=int)
    last_rand = None

    for yr in range(years):
        new_states = np.zeros(n_sim, dtype=int)
        for s in [0, 1, 2]:
            idx = states == s
            if idx.sum() == 0: continue
            new_states[idx] = np.random.choice([0, 1, 2], size=idx.sum(),
                                               p=transition_matrix[s])
        states = new_states

        rand_yr      = np.zeros((n_sim, len(w)))
        port_returns = np.zeros(n_sim)
        for s in [0, 1, 2]:
            idx = states == s
            if idx.sum() == 0: continue
            r, v, c = scenarios[s]
            cov  = np.outer(v, v) * c
            rand = np.random.multivariate_normal(r, cov, idx.sum())
            rand_yr[idx]      = rand
            port_returns[idx] = np.exp(rand @ w) - 1

        port_returns = np.maximum(port_returns, -0.99)
        values      *= (1 + port_returns)
        if yr == years - 1:
            last_rand = rand_yr

    return values, last_rand


# ### 13b. Top 100 by Final Value — 1Y and 5Y Optimized Weights
# 
# Run 100,000 simulations for each horizon and extract the top 100 by final portfolio value.
# 


real_returns_arr = real_returns_series[assets].values
real_vol_arr     = real_vol_series[assets].values
N_SIM, TOP_N     = 100_000, 100

# 1Y simulation — optimized weights, real inputs
np.random.seed(42)
cov_real        = np.outer(real_vol_arr, real_vol_arr) * real_corr_matrix
rand_1y         = np.random.multivariate_normal(real_returns_arr, cov_real, N_SIM)
port_ret_1y_opt = np.maximum(np.exp(rand_1y @ best_w_1y) - 1, -0.99)
final_val_1y    = initial * (1 + port_ret_1y_opt)

top100_idx_1y  = np.argsort(final_val_1y)[::-1][:TOP_N]
top100_vals_1y = final_val_1y[top100_idx_1y]
top100_rets_1y = rand_1y[top100_idx_1y]

print("===== TOP 100 — 1Y Optimized Weights (Real Inputs) =====")
print(f"Threshold (rank 100): {top100_vals_1y[-1]:,.0f} THB")
print(f"Mean of top 100     : {top100_vals_1y.mean():,.0f} THB")
print(f"Best case           : {top100_vals_1y[0]:,.0f} THB")

# 5Y Markov simulation — optimized weights, real inputs
np.random.seed(42)
final_val_5y, last_rand_5y = simulate_markov_5y_tracked(
    best_w_5y, real_bear_args, real_base_args, real_bull_args
)

top100_idx_5y  = np.argsort(final_val_5y)[::-1][:TOP_N]
top100_vals_5y = final_val_5y[top100_idx_5y]
top100_rets_5y = last_rand_5y[top100_idx_5y]

print("\n===== TOP 100 — 5Y Optimized Weights (Real Inputs, Markov) =====")
print(f"Threshold (rank 100): {top100_vals_5y[-1]:,.0f} THB")
print(f"Mean of top 100     : {top100_vals_5y.mean():,.0f} THB")
print(f"Best case           : {top100_vals_5y[0]:,.0f} THB")


# ### 13c. Top 100 vs All — Distribution & Asset Return Profile


fig, axes = plt.subplots(2, 2, figsize=(16, 10))

for row, (final_val, top_vals, top_rets, all_rand, color, label) in enumerate([
    (final_val_1y, top100_vals_1y, top100_rets_1y, rand_1y,      "gold",   "1Y Optimized"),
    (final_val_5y, top100_vals_5y, top100_rets_5y, last_rand_5y, "tomato", "5Y Optimized"),
]):
    # Distribution plot
    ax = axes[row, 0]
    ax.hist(final_val / 1e9,  bins=100, alpha=0.4, color="steelblue", label="All 100K sims")
    ax.hist(top_vals  / 1e9,  bins=20,  alpha=0.8, color=color,       label="Top 100")
    ax.axvline(initial / 1e9,          color="black",     linestyle="--", linewidth=2, label="Initial (1B)")
    ax.axvline(top_vals.mean() / 1e9,  color=color,       linestyle=":",  linewidth=2,
               label=f"Top 100 Mean ({top_vals.mean()/1e9:.2f}B)")
    ax.axvline(final_val.mean() / 1e9, color="steelblue", linestyle=":",  linewidth=2,
               label=f"All Mean ({final_val.mean()/1e9:.2f}B)")
    ax.set_title(f"{label} — Final Value Distribution")
    ax.set_xlabel("Portfolio Value (B THB)"); ax.set_ylabel("Frequency")
    ax.legend(fontsize=8); ax.grid(alpha=0.3)

    # Asset log-return comparison
    ax = axes[row, 1]
    x  = np.arange(len(assets))
    ax.bar(x - 0.2, all_rand.mean(axis=0),  0.4, label="All sims mean", color="steelblue", alpha=0.7)
    ax.bar(x + 0.2, top_rets.mean(axis=0),  0.4, label="Top 100 mean",  color=color,       alpha=0.9)
    ax.set_xticks(x); ax.set_xticklabels(assets, rotation=45, ha="right", fontsize=8)
    ax.set_title(f"{label} — Avg Asset Log-Return: Top 100 vs All")
    ax.set_ylabel("Log-Return"); ax.legend(); ax.grid(alpha=0.3, axis="y")

plt.suptitle("Top 100 Best Simulations vs All — 1Y & 5Y Optimized Weights", fontsize=13)
plt.tight_layout()
plt.show()


# ## 14. Top 100 by Sharpe, CVaR, and Annualised Return
# 
# For each horizon (1Y and 5Y), identify the top 100 simulations ranked by three criteria:
# - **Sharpe Ratio** — best risk-adjusted return
# - **CVaR (Safest)** — furthest from the tail (lowest downside risk)
# - **Annualised Return** — highest raw return
# 


RISK_FREE = 0.02

# ── 1Y metrics ──
ann_ret_1y    = (final_val_1y / initial) - 1
sharpe_1y     = ann_ret_1y - RISK_FREE
var5_1y       = np.percentile(final_val_1y, 5)
cvar_score_1y = final_val_1y - var5_1y

top_sharpe_idx_1y = np.argsort(sharpe_1y)[::-1][:TOP_N]
top_cvar_idx_1y   = np.argsort(cvar_score_1y)[::-1][:TOP_N]
top_ar_idx_1y     = np.argsort(ann_ret_1y)[::-1][:TOP_N]

# ── 5Y metrics ──
ann_ret_5y    = (final_val_5y / initial) ** (1/5) - 1
sharpe_5y     = ann_ret_5y - RISK_FREE
var5_5y       = np.percentile(final_val_5y, 5)
cvar_score_5y = final_val_5y - var5_5y

top_sharpe_idx_5y = np.argsort(sharpe_5y)[::-1][:TOP_N]
top_cvar_idx_5y   = np.argsort(cvar_score_5y)[::-1][:TOP_N]
top_ar_idx_5y     = np.argsort(ann_ret_5y)[::-1][:TOP_N]

for horizon, final_val, ann_ret, idx_s, idx_c, idx_a in [
    ("1Y", final_val_1y, ann_ret_1y, top_sharpe_idx_1y, top_cvar_idx_1y, top_ar_idx_1y),
    ("5Y", final_val_5y, ann_ret_5y, top_sharpe_idx_5y, top_cvar_idx_5y, top_ar_idx_5y),
]:
    print(f"\n===== TOP 100 — {horizon} Optimized Weights =====")
    for metric_name, idx in [("Sharpe", idx_s), ("CVaR (Safest)", idx_c), ("Ann. Return", idx_a)]:
        print(f"  {metric_name:<16}: mean value {final_val[idx].mean():,.0f} THB, "
              f"mean return {ann_ret[idx].mean():.2%}")


# ### 14a. Top 100 Distribution — By Sharpe, CVaR, Annualised Return


fig, axes = plt.subplots(2, 3, figsize=(18, 10))

metric_configs = [
    ("Sharpe Ratio",      top_sharpe_idx_1y, top_sharpe_idx_5y, "gold"),
    ("CVaR (Safest)",     top_cvar_idx_1y,   top_cvar_idx_5y,   "mediumseagreen"),
    ("Annualised Return", top_ar_idx_1y,     top_ar_idx_5y,     "tomato"),
]

for col, (metric_name, idx_1y, idx_5y, color) in enumerate(metric_configs):
    for row, (final_val, idx, horizon) in enumerate([
        (final_val_1y, idx_1y, "1Y Opt"),
        (final_val_5y, idx_5y, "5Y Opt"),
    ]):
        ax = axes[row, col]
        ax.hist(final_val / 1e9,          bins=100, alpha=0.35, color="steelblue", label="All 100K")
        ax.hist(final_val[idx] / 1e9,     bins=20,  alpha=0.85, color=color,       label=f"Top 100 ({metric_name})")
        ax.axvline(initial / 1e9,                   color="black", linestyle="--", linewidth=2, label="Initial (1B)")
        ax.axvline(final_val[idx].mean() / 1e9,     color=color,   linestyle=":",  linewidth=2,
                   label=f"Top Mean ({final_val[idx].mean()/1e9:.2f}B)")
        ax.axvline(final_val.mean() / 1e9,          color="steelblue", linestyle=":", linewidth=1.5,
                   label=f"All Mean ({final_val.mean()/1e9:.2f}B)")
        ax.set_title(f"{horizon} — Top 100 by {metric_name}")
        ax.set_xlabel("Portfolio Value (B THB)"); ax.set_ylabel("Frequency")
        ax.legend(fontsize=7); ax.grid(alpha=0.3)

plt.suptitle("Top 100 Simulations by Sharpe / CVaR / Annualised Return — 1Y & 5Y Optimized Weights",
             fontsize=13)
plt.tight_layout()
plt.show()


# ## 15. Export Results to CSV
# 
# Export top 100 simulation records — by final value and by metric — to your Downloads folder.
# Each row includes rank, final value, return, weights, and per-asset log-return contributions.
# 


import os
save_dir = os.path.join(os.path.expanduser("~"), "Downloads")
os.makedirs(save_dir, exist_ok=True)


def build_top100_df(top_vals, top_rets, opt_weights, horizon_label, horizon_years):
    rows = []
    for rank, (val, ret_row) in enumerate(zip(top_vals, top_rets), start=1):
        ann_ret   = (val / initial) ** (1 / horizon_years) - 1
        total_ret = (val / initial) - 1
        row = {
            "rank":               rank,
            "horizon":            horizon_label,
            "final_value_THB":    round(val, 0),
            "total_return_pct":   round(total_ret * 100, 4),
            "annualised_ret_pct": round(ann_ret   * 100, 4),
        }
        for asset, w in zip(assets, opt_weights):
            row[f"weight_{asset}"]  = round(w, 6)
        for asset, r in zip(assets, ret_row):
            row[f"logret_{asset}"]  = round(r, 6)
        for asset, r, w in zip(assets, ret_row, opt_weights):
            row[f"contrib_{asset}"] = round(r * w, 6)
        rows.append(row)
    return pd.DataFrame(rows)


def build_metric_df(final_vals, ann_rets, top_idx, rand_matrix,
                    opt_weights, horizon_label, horizon_years, metric_name):
    var5      = np.percentile(final_vals, 5)
    cvar_vals = final_vals - var5
    rows = []
    for rank, i in enumerate(top_idx, start=1):
        val      = final_vals[i]
        ann_ret  = ann_rets[i]
        tot_ret  = (val / initial) - 1
        excess   = ann_ret - RISK_FREE
        sharpe   = excess / np.std(ann_rets)
        ret_row  = rand_matrix[i]
        row = {
            "rank":               rank,
            "metric":             metric_name,
            "horizon":            horizon_label,
            "final_value_THB":    round(val, 0),
            "total_return_pct":   round(tot_ret * 100, 4),
            "annualised_ret_pct": round(ann_ret * 100, 4),
            "excess_ret_pct":     round(excess  * 100, 4),
            "sharpe_ratio":       round(sharpe,  4),
            "cvar_distance_THB":  round(cvar_vals[i], 0),
        }
        for asset, w in zip(assets, opt_weights):
            row[f"weight_{asset}"]  = round(w, 6)
        for asset, r in zip(assets, ret_row):
            row[f"logret_{asset}"]  = round(r, 6)
        for asset, r, w in zip(assets, ret_row, opt_weights):
            row[f"contrib_{asset}"] = round(r * w, 6)
        rows.append(row)
    return pd.DataFrame(rows)


# Build and save top-100-by-value CSVs
df_top100_1y = build_top100_df(top100_vals_1y, top100_rets_1y, best_w_1y, "1Y_optimized", 1)
df_top100_5y = build_top100_df(top100_vals_5y, top100_rets_5y, best_w_5y, "5Y_optimized", 5)
pd.concat([df_top100_1y, df_top100_5y]).to_csv(
    os.path.join(save_dir, "top100_optimized_simulations.csv"), index=False)

# Build and save top-100-by-metric CSVs
dfs = {}
for metric_name, idx_1y, idx_5y in [
    ("Sharpe", top_sharpe_idx_1y, top_sharpe_idx_5y),
    ("CVaR",   top_cvar_idx_1y,   top_cvar_idx_5y),
    ("AnnRet", top_ar_idx_1y,     top_ar_idx_5y),
]:
    dfs[f"1Y_{metric_name}"] = build_metric_df(
        final_val_1y, ann_ret_1y, idx_1y, rand_1y, best_w_1y, "1Y_optimized", 1, metric_name)
    dfs[f"5Y_{metric_name}"] = build_metric_df(
        final_val_5y, ann_ret_5y, idx_5y, last_rand_5y, best_w_5y, "5Y_optimized", 5, metric_name)

for label, df in dfs.items():
    df.to_csv(os.path.join(save_dir, f"top100_{label}.csv"), index=False)
    print(f"Saved: top100_{label}.csv  ({len(df)} rows)")

pd.concat(dfs.values()).to_csv(os.path.join(save_dir, "top100_all_metrics.csv"), index=False)
print(f"\nAll files saved to: {save_dir}")


# ## 16. 1-Year Scenarios Using 5Y Optimal Weights
# 
# Apply the 5Y optimized weights to the standard 1-year bear/base/bull simulation.
# Useful for comparing short-term risk/reward under a long-term-tuned portfolio.
# 


np.random.seed(42)
base_r1, base_v1, base_c1 = build_scenario(returns, vol, base_corr, 'base')
bull_r1, bull_v1, bull_c1 = build_scenario(returns, vol, base_corr, 'bull')
bear_r1, bear_v1, bear_c1 = build_scenario(returns, vol, base_corr, 'bear')

sim_base_5y = simulate(best_w_5y, base_r1, base_v1, base_c1)
sim_bull_5y = simulate(best_w_5y, bull_r1, bull_v1, bull_c1)
sim_bear_5y = simulate(best_w_5y, bear_r1, bear_v1, bear_c1)

summarize('Bear (1Y) — 5Y Optimal Weights', sim_bear_5y)
summarize('Base (1Y) — 5Y Optimal Weights', sim_base_5y)
summarize('Bull (1Y) — 5Y Optimal Weights', sim_bull_5y)

fig, ax = plt.subplots(figsize=(12, 6))
ax.hist(sim_base_5y, bins=80, alpha=0.4, color='steelblue', label='Base')
ax.hist(sim_bull_5y, bins=80, alpha=0.4, color='orange',    label='Bull')
ax.hist(sim_bear_5y, bins=80, alpha=0.4, color='green',     label='Bear')
ax.axvline(initial,              color='black',      linestyle='--', linewidth=2, label='Initial')
ax.axvline(np.mean(sim_base_5y), color='yellowgreen',linewidth=2,                 label='Mean Base')
ax.axvline(np.mean(sim_bull_5y), color='steelblue',  linewidth=2,                 label='Mean Bull')
ax.axvline(np.mean(sim_bear_5y), color='navy',       linewidth=2,                 label='Mean Bear')
ax.set_title('Monte Carlo Simulation (1 Year) — 5Y Optimal Weights')
ax.set_xlabel('Portfolio Value (THB)')
ax.set_ylabel('Frequency')
ax.set_xlim(0.5e9, 3.0e9)
ax.grid(alpha=0.3)
ax.legend()
plt.tight_layout()
plt.show()


# ## 17. Rolling 1-Year Metric Analysis (5-Year Horizon)
# 
# Simulates 30,000 monthly paths over 60 months using Markov regime switching,
# then computes rolling 12-month windows for 10 portfolio metrics.
# 
# **Metrics computed per window:**
# Total Return, CAGR (3Y), Rolling 1Y Avg, P/E Proxy, Sharpe Ratio,
# Max Drawdown, Alpha, Beta, Annualised Volatility, VaR (95%), CVaR (95%).
# 
# **Benchmark:** Equal-weight Thai banks (KBANK / SCB / TISCO).
# 


RF       = 0.02
N_SIM    = 30_000
N_MONTHS = 60  # 5-year horizon

def run_path_simulation(w, n_sim=N_SIM, n_months=N_MONTHS):
    """Returns (n_sim, n_months) array of monthly simple returns under Markov regime switching."""
    scenarios = {
        0: build_scenario(returns, vol, base_corr, 'bear'),
        1: build_scenario(returns, vol, base_corr, 'base'),
        2: build_scenario(returns, vol, base_corr, 'bull')
    }
    monthly_rets = np.zeros((n_sim, n_months))
    states = np.full(n_sim, 1, dtype=int)

    for m in range(n_months):
        new_states = np.zeros(n_sim, dtype=int)
        for s in [0, 1, 2]:
            idx = states == s
            if idx.sum() == 0: continue
            new_states[idx] = np.random.choice([0, 1, 2], size=idx.sum(),
                                               p=transition_matrix[s])
        states = new_states

        for s in [0, 1, 2]:
            idx = states == s
            if idx.sum() == 0: continue
            r2, v2, c2 = scenarios[s]
            r_m = r2 / 12;  v_m = v2 / np.sqrt(12)
            cov  = np.outer(v_m, v_m) * c2
            rand = np.random.multivariate_normal(r_m, cov, idx.sum())
            monthly_rets[idx, m] = np.exp(rand @ w) - 1

    return np.maximum(monthly_rets, -0.99)


np.random.seed(42)
print('Simulating Current weights...')
paths_set  = run_path_simulation(weights)
print('Simulating 5Y Optimized weights...')
paths_opt5 = run_path_simulation(best_w_5y)

bm_w = np.zeros(15);  bm_w[Thai_banks] = 1 / len(Thai_banks)
print('Simulating benchmark (Thai banks)...')
paths_bm = run_path_simulation(bm_w)
print('Done.')



def compute_rolling_metrics(paths, paths_bm, window=12):
    n_sim, n_months = paths.shape
    n_windows = n_months - window + 1
    records = []

    for t in range(n_windows):
        w_rets  = paths[:, t:t+window]
        bm_rets = paths_bm[:, t:t+window]
        cum     = np.prod(1 + w_rets, axis=1) - 1
        tot_ret = np.median(cum)

        cagr3 = np.median((1 + np.prod(1 + paths[:, t:t+36], axis=1) - 1) ** (1/3) - 1)                 if t + 36 <= n_months else np.nan

        rolling_1y_avg = w_rets.mean(axis=1).mean() * 12
        ann_ret = (1 + tot_ret) ** (12 / window) - 1 if tot_ret > -1 else np.nan
        pe = (1 / ann_ret) if (ann_ret and ann_ret > 0.001) else np.nan

        ann_rets_sim = (1 + cum) ** (12 / window) - 1
        sharpe       = (np.mean(ann_rets_sim) - RF) / (np.std(ann_rets_sim) + 1e-9)
        vol_ann      = w_rets.std(axis=1).mean() * np.sqrt(12)

        cum_val  = np.cumprod(1 + w_rets, axis=1)
        roll_max = np.maximum.accumulate(cum_val, axis=1)
        max_dd   = np.median(((cum_val - roll_max) / roll_max).min(axis=1))

        all_monthly = w_rets.flatten()
        var_95  = np.percentile(all_monthly, 5)
        tail    = all_monthly[all_monthly <= var_95]
        cvar_95 = tail.mean() if len(tail) > 0 else var_95

        pf_mean = w_rets.mean(axis=1);  bm_mean = bm_rets.mean(axis=1)
        cov_mat = np.cov(pf_mean, bm_mean)
        beta    = cov_mat[0, 1] / (cov_mat[1, 1] + 1e-9)
        alpha   = pf_mean.mean() * 12 - (RF + beta * (bm_mean.mean() * 12 - RF))

        records.append({
            'window': t, 'total_return': tot_ret, 'cagr_3y': cagr3,
            'rolling_1y_avg': rolling_1y_avg, 'pe_proxy': pe,
            'sharpe': sharpe, 'max_drawdown': max_dd, 'alpha': alpha,
            'beta': beta, 'volatility_ann': vol_ann,
            'var_95_monthly': var_95, 'cvar_95': cvar_95
        })

    return pd.DataFrame(records)

print('Computing rolling metrics...')
df_set  = compute_rolling_metrics(paths_set,  paths_bm)
df_opt5 = compute_rolling_metrics(paths_opt5, paths_bm)
print('Done.')


# ### 17a. Rolling Metrics — Summary Table


def fmt_metric(val, col):
    signed = col in ('max_drawdown', 'var_95_monthly', 'cvar_95', 'alpha')
    pct    = col in ('total_return', 'cagr_3y', 'rolling_1y_avg', 'max_drawdown',
                     'alpha', 'volatility_ann', 'var_95_monthly', 'cvar_95')
    if pct:          return f'{val*100:+.1f}%' if signed else f'{val*100:.1f}%'
    elif col == 'pe_proxy': return f'{np.clip(val, 0, 50):.1f}x'
    else:            return f'{val:.2f}'

metric_defs = [
    ('total_return',   'Total Return (1Y)',     True),
    ('cagr_3y',        'CAGR (3Y)',             True),
    ('rolling_1y_avg', 'Rolling 1Y Avg Return', True),
    ('pe_proxy',       'P/E Proxy',             False),
    ('sharpe',         'Sharpe Ratio',          True),
    ('max_drawdown',   'Max Drawdown',          False),
    ('alpha',          'Alpha',                 True),
    ('beta',           'Beta',                  False),
    ('volatility_ann', 'Volatility (Ann.)',     False),
    ('var_95_monthly', 'VaR (95%, Monthly)',    False),
    ('cvar_95',        'CVaR (95%)',            False),
]

print('\n' + '='*70)
print(f"{'Metric':<26} {'Current':>14} {'5Y Optimized':>14} {'Diff':>10} {'Edge':>12}")
print('='*70)
for col, label, higher_better in metric_defs:
    vs = df_set[col].dropna().mean()
    vo = df_opt5[col].dropna().mean()
    if col == 'pe_proxy':
        vs = np.clip(vs, 0, 50);  vo = np.clip(vo, 0, 50)
    diff = vo - vs
    opt_win = (diff > 0.0005 and higher_better) or (diff < -0.0005 and not higher_better)
    set_win = (diff < -0.0005 and higher_better) or (diff > 0.0005 and not higher_better)
    edge = '5Y Opt' if opt_win else ('Current' if set_win else 'Tied')
    print(f'{label:<26} {fmt_metric(vs,col):>14} {fmt_metric(vo,col):>14} '
          f'{fmt_metric(diff,col):>10} {edge:>12}')
print('='*70)


# ### 17b. Rolling Metric Charts — 10 Subplots (Dark Theme)


pct = FuncFormatter(lambda x, _: f'{x*100:.1f}%')
x   = df_set['window'].values

metric_plots = [
    ('Total Return (Rolling 1Y)',    'total_return',   pct,  None),
    ('CAGR (3Y)',                    'cagr_3y',        pct,  None),
    ('Rolling 1Y Avg Return',        'rolling_1y_avg', pct,  None),
    ('P/E Proxy',                    'pe_proxy',       None, (0, 50)),
    ('Sharpe Ratio (RF=2%)',         'sharpe',         None, None),
    ('Max Drawdown',                 'max_drawdown',   pct,  None),
    ('Alpha (vs Thai Bank Bench.)',  'alpha',          pct,  None),
    ('Beta (vs Thai Bank Bench.)',   'beta',           None, None),
    ('Annualised Volatility',        'volatility_ann', pct,  None),
    ('VaR & CVaR (95%, Monthly)',    'var_95_monthly', pct,  None),
]

fig, axes = plt.subplots(5, 2, figsize=(14, 20))
fig.suptitle('Rolling 1-Year Analysis: 5Y Optimized vs Current Portfolio\n'
             'Markov Regime-Switching · 30,000 Paths · 5-Year Horizon', fontsize=13)

for ax, (title, col, fmt, clip) in zip(axes.flat, metric_plots):
    if col == 'var_95_monthly':
        # combine VaR and CVaR in the last panel
        ax.plot(x, df_set['var_95_monthly'].values,  color='steelblue', linewidth=1.5, label='Current VaR')
        ax.plot(x, df_opt5['var_95_monthly'].values, color='orange',    linewidth=1.5, label='5Y Opt VaR')
        ax.plot(x, df_set['cvar_95'].values,         color='steelblue', linewidth=1.5, linestyle='--', label='Current CVaR')
        ax.plot(x, df_opt5['cvar_95'].values,        color='orange',    linewidth=1.5, linestyle='--', label='5Y Opt CVaR')
        ax.yaxis.set_major_formatter(pct)
        ax.legend(fontsize=7, ncol=2)
    else:
        ys = df_set[col].values
        yo = df_opt5[col].values
        if col == 'cagr_3y':
            m = ~np.isnan(ys) & ~np.isnan(yo)
            ys, yo, xp = ys[m], yo[m], x[m]
        else:
            xp = x
        if clip:
            ys = np.clip(ys, clip[0], clip[1])
            yo = np.clip(yo, clip[0], clip[1])
        ax.plot(xp, ys, color='steelblue', linewidth=1.5, label='Current (SET)')
        ax.plot(xp, yo, color='orange',    linewidth=1.5, label='5Y Optimized')
        if fmt:
            ax.yaxis.set_major_formatter(fmt)
        if col in ('sharpe', 'alpha'):
            ax.axhline(0, color='red', linewidth=1, linestyle='--', alpha=0.5)
        if col == 'beta':
            ax.axhline(1, color='gray', linewidth=1, linestyle=':', alpha=0.7)
        ax.legend(fontsize=8)

    ax.set_title(title)
    ax.set_xlabel('Rolling Window (month)')
    ax.grid(alpha=0.3)

plt.tight_layout()
plt.show()


# ### 17c. Rolling Metrics — Summary Table


print('\n' + '='*70)
print(f"{'Metric':<26} {'Current':>14} {'5Y Optimized':>14} {'Diff':>10} {'Edge':>12}")
print('='*70)
for col, label, higher_better in metric_defs:
    vs = df_set[col].dropna().mean()
    vo = df_opt5[col].dropna().mean()
    if col == 'pe_proxy':
        vs = np.clip(vs, 0, 50);  vo = np.clip(vo, 0, 50)
    diff = vo - vs
    opt_win = (diff > 0.0005 and higher_better) or (diff < -0.0005 and not higher_better)
    set_win = (diff < -0.0005 and higher_better) or (diff > 0.0005 and not higher_better)
    edge = '5Y Opt' if opt_win else ('Current' if set_win else 'Tied')
    print(f'{label:<26} {fmt_metric(vs,col):>14} {fmt_metric(vo,col):>14} '
          f'{fmt_metric(diff,col):>10} {edge:>12}')
print('='*70)

