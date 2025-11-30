import pandas as pd
import numpy as np
from scipy.optimize import minimize

print("\n" + "="*80)
print("MEAN–VaR PORTFOLIO OPTIMIZATION (6 STOCKS)")
print("="*80 + "\n")


# 1. Load combined price data (same CSV from data_collection)

combined_path = "data/combined_6stocks_5years.csv"
df = pd.read_csv(combined_path)

required_cols = {'Date', 'Ticker', 'Close'}
missing = required_cols - set(df.columns)
if missing:
    raise ValueError(f"Missing columns in {combined_path}: {missing}")

df['Date'] = pd.to_datetime(df['Date'])
df = df.sort_values(['Ticker', 'Date']).reset_index(drop=True)

print(f"Loaded {len(df)} rows for {df['Ticker'].nunique()} tickers.")
print(f"Tickers: {df['Ticker'].unique().tolist()}")
print(f"Date range: {df['Date'].min().date()} to {df['Date'].max().date()}\n")


# 2. Compute log returns per asset and pivot to matrix

df['Log_Return'] = df.groupby('Ticker')['Close'].transform(
    lambda x: np.log(x / x.shift(1))
)
df = df.dropna(subset=['Log_Return']).reset_index(drop=True)

returns_matrix = df.pivot_table(
    index='Date',
    columns='Ticker',
    values='Log_Return'
).sort_index()

# forward/backward fill tiny gaps
returns_matrix = returns_matrix.ffill().bfill()

tickers = list(returns_matrix.columns)
n_assets = len(tickers)

print("Returns matrix shape:", returns_matrix.shape)
print("Assets in optimization:", tickers, "\n")

# numpy array for speed
R = returns_matrix.values  # shape: (T, N)
T = R.shape[0]


# 3. Helper functions

def portfolio_returns(weights, R):
    """
    weights: (N,)
    R: (T, N)
    """
    return (R * weights).sum(axis=1)

def portfolio_var_historical(weights, R, confidence=0.95):
    """
    Historical VaR on portfolio daily returns.
    Returns positive number (e.g. 0.02 for 2%).
    """
    pr = portfolio_returns(weights, R)
    return -np.percentile(pr, (1 - confidence) * 100)

def annualized_return(weights, R):
    """
    Annualized expected return using mean daily return * 252
    """
    pr = portfolio_returns(weights, R)
    mean_daily = pr.mean()
    return mean_daily * 252

def annualized_volatility(weights, R):
    pr = portfolio_returns(weights, R)
    std_daily = pr.std()
    return std_daily * np.sqrt(252)


# 4. Define optimization problem

VAR_LIMIT = 0.02  # 2% daily VaR at 95%

def objective_neg_annual_return(weights, R):
    # We minimize negative annualized return = maximize return
    return -annualized_return(weights, R)

def constraint_var(weights):
    # VaR <= VAR_LIMIT  => VAR_LIMIT - VaR >= 0
    var_value = portfolio_var_historical(weights, R)
    return VAR_LIMIT - var_value

def constraint_weights_sum_to_one(weights):
    return np.sum(weights) - 1.0

# initial guess: equal weights
w0 = np.array([1.0 / n_assets] * n_assets)

bounds = tuple((0.0, 1.0) for _ in range(n_assets))
constraints = [
    {'type': 'eq', 'fun': constraint_weights_sum_to_one},
    {'type': 'ineq', 'fun': constraint_var},
]

print("Initial equal-weight portfolio stats:")
eq_ret = annualized_return(w0, R)
eq_vol = annualized_volatility(w0, R)
eq_var = portfolio_var_historical(w0, R)
eq_sharpe = eq_ret / eq_vol if eq_vol != 0 else np.nan

print(f"  Annualized Return   : {eq_ret*100:.2f}%")
print(f"  Annualized Volatility: {eq_vol*100:.2f}%")
print(f"  Daily VaR (95%)     : {eq_var*100:.2f}%")
print(f"  Sharpe Ratio        : {eq_sharpe:.3f}\n")

print("Running SLSQP optimization with VaR constraint (VaR ≤ 2%)...\n")

result = minimize(
    objective_neg_annual_return,
    w0,
    args=(R,),
    method='SLSQP',
    bounds=bounds,
    constraints=constraints,
    options={'maxiter': 1000, 'disp': False}
)

if not result.success:
    print("Optimization FAILED:", result.message)
else:
    print("Optimization successful.\n")

opt_w = result.x
opt_ret = annualized_return(opt_w, R)
opt_vol = annualized_volatility(opt_w, R)
opt_var = portfolio_var_historical(opt_w, R)
opt_sharpe = opt_ret / opt_vol if opt_vol != 0 else np.nan


# 5. Print results

print("="*80)
print("OPTIMAL PORTFOLIO WEIGHTS (Mean–VaR Optimization)")
print("="*80)
for t, w in zip(tickers, opt_w):
    print(f"{t:15s} : {w*100:6.2f}%")

print("\n" + "="*80)
print("PERFORMANCE COMPARISON: Equal-Weight vs Optimized")
print("="*80)
print(f"{'Metric':25s} {'Equal-Weight':>15s} {'Optimized':>15s}")
print("-"*60)
print(f"{'Annual Return':25s} {eq_ret*100:15.2f}% {opt_ret*100:15.2f}%")
print(f"{'Annual Volatility':25s} {eq_vol*100:15.2f}% {opt_vol*100:15.2f}%")
print(f"{'Daily VaR (95%)':25s} {eq_var*100:15.2f}% {opt_var*100:15.2f}%")
print(f"{'Sharpe Ratio':25s} {eq_sharpe:15.3f} {opt_sharpe:15.3f}")
print("="*80 + "\n")
