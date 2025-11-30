import pandas as pd
import numpy as np
from scipy.stats import norm, t

print("\n" + "="*80)
print("VALUE AT RISK ENGINE (Historical, Parametric, Monte Carlo)")
print("="*80 + "\n")

# -------------------------------------------------------------
# Load portfolio returns
# -------------------------------------------------------------
df = pd.read_csv("data/portfolio_returns.csv")

df['Date'] = pd.to_datetime(df['Date'])
returns = df['Portfolio_Return'].values

print(f"Loaded {len(returns)} daily returns.")
print(f"Date range: {df['Date'].min().date()} to {df['Date'].max().date()}\n")

# -------------------------------------------------------------
# Historical VaR
# -------------------------------------------------------------
def historical_var(returns, confidence=0.95):
    return -np.percentile(returns, (1 - confidence) * 100)

# -------------------------------------------------------------
# Parametric (Normal) VaR
# -------------------------------------------------------------
def parametric_var_normal(returns, confidence=0.95):
    mu = np.mean(returns)
    sigma = np.std(returns)
    z = norm.ppf(1 - confidence)
    return -(mu + sigma * z)

# -------------------------------------------------------------
# Parametric (t-distribution) VaR
# -------------------------------------------------------------
def parametric_var_t(returns, confidence=0.95):
    df_t, loc, scale = t.fit(returns)
    t_score = t.ppf(1 - confidence, df_t)
    return -(loc + scale * t_score)

# -------------------------------------------------------------
# Monte Carlo VaR
# -------------------------------------------------------------
def monte_carlo_var(returns, confidence=0.95, simulations=10000):
    mu = np.mean(returns)
    sigma = np.std(returns)

    sim = np.random.normal(mu, sigma, simulations)
    return -np.percentile(sim, (1 - confidence) * 100)

# -------------------------------------------------------------
# Run all VaR methods
# -------------------------------------------------------------
conf = 0.95
var_hist = historical_var(returns, conf)
var_norm = parametric_var_normal(returns, conf)
var_t = parametric_var_t(returns, conf)
var_mc = monte_carlo_var(returns, conf)

print("="*80)
print(f"VaR Results (95% Daily)")
print("="*80)
print(f"Historical VaR      : {var_hist*100:.3f}%")
print(f"Parametric Normal   : {var_norm*100:.3f}%")
print(f"Parametric t-dist   : {var_t*100:.3f}%")
print(f"Monte Carlo         : {var_mc*100:.3f}%")
print("="*80 + "\n")
