import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os

from scipy.stats import norm

print("\n" + "="*80)
print("VISUALIZATIONS: PORTFOLIO, VaR, EXCEPTIONS, WEIGHTS")
print("="*80 + "\n")

os.makedirs("plots", exist_ok=True)


port_path = "data/portfolio_returns.csv"
comb_path = "data/combined_6stocks_5years.csv"

df_port = pd.read_csv(port_path)
df_port['Date'] = pd.to_datetime(df_port['Date'])
df_port = df_port.sort_values('Date')

df_comb = pd.read_csv(comb_path)
df_comb['Date'] = pd.to_datetime(df_comb['Date'])
df_comb = df_comb.sort_values(['Ticker', 'Date'])

print(f"Loaded portfolio returns: {len(df_port)} rows.")
print(f"Loaded combined price data: {len(df_comb)} rows.\n")


plt.figure(figsize=(12, 6))
plt.plot(df_port['Date'], df_port['Cumulative_Return'], linewidth=2)
plt.title("Cumulative Portfolio Return (Equal-Weight Portfolio)", fontsize=14, fontweight="bold")
plt.xlabel("Date")
plt.ylabel("Cumulative Value (Starting at 1.0)")
plt.grid(True, alpha=0.3)

cum_path = "plots/portfolio_cumulative_return.png"
plt.tight_layout()
plt.savefig(cum_path, dpi=300)
plt.close()
print(f"Saved: {cum_path}")


returns = df_port['Portfolio_Return'].values
hist_var_95 = -np.percentile(returns, 5)

plt.figure(figsize=(10, 6))
plt.hist(returns, bins=50, alpha=0.7, edgecolor='black')
plt.axvline(-hist_var_95, color='red', linestyle='--', linewidth=2,
            label=f"VaR 95% = {hist_var_95*100:.2f}%")
plt.title("Histogram of Daily Portfolio Returns with 95% Historical VaR", fontsize=14, fontweight="bold")
plt.xlabel("Daily Return")
plt.ylabel("Frequency")
plt.legend()
plt.grid(True, alpha=0.3)

hist_path = "plots/returns_histogram_var95.png"
plt.tight_layout()
plt.savefig(hist_path, dpi=300)
plt.close()
print(f"Saved: {hist_path}")


exceptions = returns < -hist_var_95

plt.figure(figsize=(14, 6))
plt.plot(df_port['Date'], returns * 100, label="Daily Return", linewidth=1)
plt.axhline(-hist_var_95 * 100, color='red', linestyle='--', label="VaR 95%")

# Mark exceptions
plt.scatter(
    df_port['Date'][exceptions],
    returns[exceptions] * 100,
    color='red',
    label="Exceptions",
    zorder=3
)

plt.title("Daily Returns vs 95% Historical VaR (Exceptions Highlighted)", fontsize=14, fontweight="bold")
plt.xlabel("Date")
plt.ylabel("Daily Return (%)")
plt.legend()
plt.grid(True, alpha=0.3)

exc_path = "plots/var_exceptions_timeseries.png"
plt.tight_layout()
plt.savefig(exc_path, dpi=300)
plt.close()
print(f"Saved: {exc_path}")

# Build returns matrix
df_comb = df_comb.sort_values(['Ticker', 'Date'])
df_comb['Log_Return'] = df_comb.groupby('Ticker')['Close'].transform(
    lambda x: np.log(x / x.shift(1))
)
df_comb = df_comb.dropna(subset=['Log_Return']).reset_index(drop=True)

returns_matrix = df_comb.pivot_table(
    index='Date',
    columns='Ticker',
    values='Log_Return'
).sort_index()

returns_matrix = returns_matrix.ffill().bfill()
tickers = list(returns_matrix.columns)
R = returns_matrix.values
n_assets = len(tickers)

def portfolio_returns(weights, R):
    return (R * weights).sum(axis=1)

def portfolio_var_historical(weights, R, confidence=0.95):
    pr = portfolio_returns(weights, R)
    return -np.percentile(pr, (1 - confidence) * 100)

# Equal weights
w_eq = np.array([1.0 / n_assets] * n_assets)

# Reuse optimized weights from optimization logic
from scipy.optimize import minimize

def annualized_return(weights, R):
    pr = portfolio_returns(weights, R)
    return pr.mean() * 252

def objective_neg_annual_return(weights, R):
    return -annualized_return(weights, R)

VAR_LIMIT = 0.02

def constraint_var(weights):
    return VAR_LIMIT - portfolio_var_historical(weights, R)

def constraint_weights_sum_to_one(weights):
    return np.sum(weights) - 1.0

bounds = tuple((0.0, 1.0) for _ in range(n_assets))
constraints = [
    {'type': 'eq', 'fun': constraint_weights_sum_to_one},
    {'type': 'ineq', 'fun': constraint_var},
]

res = minimize(
    objective_neg_annual_return,
    w_eq,
    args=(R,),
    method='SLSQP',
    bounds=bounds,
    constraints=constraints,
    options={'maxiter': 1000, 'disp': False}
)

if not res.success:
    print("Warning: optimization failed in visualization script:", res.message)
    w_opt = w_eq.copy()
else:
    w_opt = res.x

# Plot bar chart
x = np.arange(n_assets)
width = 0.35

plt.figure(figsize=(12, 6))
plt.bar(x - width/2, w_eq * 100, width, label="Equal-Weight")
plt.bar(x + width/2, w_opt * 100, width, label="Optimized")

plt.xticks(x, tickers, rotation=45)
plt.ylabel("Weight (%)")
plt.title("Equal-Weight vs Mean–VaR Optimized Weights", fontsize=14, fontweight="bold")
plt.legend()
plt.grid(True, alpha=0.3)

weights_path = "plots/weights_comparison.png"
plt.tight_layout()
plt.savefig(weights_path, dpi=300)
plt.show()
print(f"Saved: {weights_path}")

print("\nAll visualizations generated in 'plots/' folder.\n")
