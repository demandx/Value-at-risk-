import pandas as pd
import numpy as np
import os

print("\n" + "="*80)
print("PORTFOLIO RETURN CONSTRUCTION - 6 STOCKS (USD)")
print("="*80 + "\n")

# ---------- 1. Load combined price data ----------
data_path = os.path.join("data", "combined_6stocks_5years.csv")

if not os.path.exists(data_path):
    raise FileNotFoundError(f"{data_path} not found. Run data_collection.py from project root first.")

df = pd.read_csv(data_path)

required_cols = {'Date', 'Ticker', 'Close'}
missing = required_cols - set(df.columns)
if missing:
    raise ValueError(f"CSV missing required columns: {missing}")

df['Date'] = pd.to_datetime(df['Date'])
df = df.sort_values(['Ticker', 'Date']).reset_index(drop=True)

print(f"Loaded {len(df)} rows for {df['Ticker'].nunique()} tickers.")
print(f"Date range: {df['Date'].min().date()} to {df['Date'].max().date()}\n")

# ---------- 2. Compute log returns per ticker ----------
df['Log_Return'] = df.groupby('Ticker')['Close'].transform(
    lambda x: np.log(x / x.shift(1))
)

df = df.dropna(subset=['Log_Return']).reset_index(drop=True)

print("Computed log returns per ticker.")
print(f"Rows after dropping NaNs: {len(df)}\n")

# ---------- 3. Pivot to returns matrix ----------
returns_pivot = df.pivot_table(
    index='Date',
    columns='Ticker',
    values='Log_Return'
).sort_index()

returns_pivot = returns_pivot.fillna(method='ffill').fillna(method='bfill')

print("Returns matrix shape:", returns_pivot.shape)
print("Tickers:", list(returns_pivot.columns), "\n")

# ---------- 4. Equal-weight portfolio ----------
num_stocks = len(returns_pivot.columns)
if num_stocks == 0:
    raise ValueError("No tickers found in returns_pivot.")

equal_weights = np.array([1 / num_stocks] * num_stocks)

portfolio_returns = (returns_pivot * equal_weights).sum(axis=1)

portfolio_df = pd.DataFrame({
    'Date': portfolio_returns.index,
    'Portfolio_Return': portfolio_returns.values
})
portfolio_df['Cumulative_Return'] = (1 + portfolio_df['Portfolio_Return']).cumprod()

# ---------- 5. Portfolio statistics ----------
trading_days = portfolio_df.shape[0]
mean_daily = portfolio_returns.mean()
std_daily = portfolio_returns.std()

annualized_return = mean_daily * 252
annualized_vol = std_daily * np.sqrt(252)
sharpe_ratio = annualized_return / annualized_vol if annualized_vol != 0 else np.nan
total_return = portfolio_df['Cumulative_Return'].iloc[-1] - 1

cum = portfolio_df['Cumulative_Return']
running_max = cum.cummax()
drawdown = (cum - running_max) / running_max
max_drawdown = drawdown.min()

print("="*80)
print("PORTFOLIO PERFORMANCE SUMMARY (Equal-Weight)")
print("="*80)
print(f"Trading Days         : {trading_days}")
print(f"Mean Daily Return    : {mean_daily*100:.4f}%")
print(f"Daily Volatility     : {std_daily*100:.4f}%")
print(f"Annualized Return    : {annualized_return*100:.2f}%")
print(f"Annualized Volatility: {annualized_vol*100:.2f}%")
print(f"Sharpe Ratio         : {sharpe_ratio:.3f}")
print(f"Total Return (5Y)    : {total_return*100:.2f}%")
print(f"Maximum Drawdown     : {max_drawdown*100:.2f}%")
print("="*80 + "\n")

# ---------- 6. Save output ----------
out_path = os.path.join("data", "portfolio_returns.csv")
portfolio_df.to_csv(out_path, index=False)
print(f"Saved portfolio returns to: {out_path}")
print(f"Rows: {len(portfolio_df)}")
print("Done.\n")
