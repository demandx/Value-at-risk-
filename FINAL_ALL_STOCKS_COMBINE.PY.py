
import yfinance as yf
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime, timedelta
import os
import time
import warnings
warnings.filterwarnings('ignore')

print("\n" + "="*80)
print("COMBINED STOCK DATA ANALYSIS - ROBUST DOWNLOAD")
print("="*80 + "\n")


# Define stocks
stocks_info = {
    # US Stocks
    'AAPL': {'name': 'Apple Inc.', 'market': 'NASDAQ'},
    'GOOGL': {'name': 'Alphabet Inc.', 'market': 'NASDAQ'},
    'TSLA': {'name': 'Tesla Inc.', 'market': 'NASDAQ'},
    # Indian Stocks
    'RELIANCE.NS': {'name': 'Reliance Industries', 'market': 'NSE'},
    'TCS.NS': {'name': 'Tata Consultancy Services', 'market': 'NSE'},
    'INFY.NS': {'name': 'Infosys Limited', 'market': 'NSE'},
}

tickers = list(stocks_info.keys())
print(f"Portfolio of {len(tickers)} stocks:\n")
for ticker, info in stocks_info.items():
    print(f"  {ticker:15s} - {info['name']:30s} ({info['market']})")
print()


# Download data one-by-one with delay
end_date = datetime.now()
start_date = end_date - timedelta(days=5*365)
print(f"Date range: {start_date.date()} to {end_date.date()}\n")
print("Downloading data with 2 seconds delay between tickers...")

all_data = {}
failed_tickers = []

for ticker in tickers:
    try:
        print(f"  Downloading {ticker}...", end=' ')
        df = yf.download(ticker, start=start_date.strftime('%Y-%m-%d'), end=end_date.strftime('%Y-%m-%d'), progress=False)
        if df.empty:
            print("No data")
            failed_tickers.append(ticker)
        else:
            all_data[ticker] = df
            print(f"Success ({len(df)} days)")
    except Exception as e:
        print(f"Failed ({e})")
        failed_tickers.append(ticker)
    time.sleep(2)
print()


# Download USD/INR exchange rates
print("Downloading USD/INR (INR=X) exchange rates...")
try:
    fx = yf.download('INR=X', start=start_date.strftime('%Y-%m-%d'), end=end_date.strftime('%Y-%m-%d'), progress=False)
    fx = fx[['Close']].copy()
    fx = fx.rename(columns={'Close': 'USDINR'})
    fx = fx.reset_index()
    fx['Date'] = pd.to_datetime(fx['Date'])
    fx['Date_dt'] = fx['Date']
    fx = fx[['Date_dt', 'USDINR']].copy()
    fx.columns = ['Date_dt', 'USDINR']
    print("✓ Exchange rates downloaded\n")
except Exception as e:
    print(f"Failed to download exchange rates: {e}")
    exit(1)

if failed_tickers:
    print("Warning: Failed to download for these tickers:")
    for t in failed_tickers:
        print(f" - {t}")
    print()

if not all_data:
    print("ERROR: No data downloaded for any ticker. Exiting.")
    exit(1)


# Prepare combined data
combined_data = []
for ticker, stock_data in all_data.items():
    company_name = stocks_info[ticker]['name']
    market = stocks_info[ticker]['market']

    for date in stock_data.index:
        row = {
            'Date': date.strftime('%Y-%m-%d'),
            'Ticker': ticker,
            'Company': company_name,
            'Market': market,
            'Open': round(float(stock_data.loc[date, 'Open']), 2),
            'High': round(float(stock_data.loc[date, 'High']), 2),
            'Low': round(float(stock_data.loc[date, 'Low']), 2),
            'Close': round(float(stock_data.loc[date, 'Close']), 2),
            'Volume': int(stock_data.loc[date, 'Volume']),
        }
        combined_data.append(row)

df_combined = pd.DataFrame(combined_data)

if df_combined.empty or 'Date' not in df_combined.columns:
    print("ERROR: No valid data was downloaded! Exiting.")
    exit(1)


# Sort by Date and Ticker
df_combined['Date_dt'] = pd.to_datetime(df_combined['Date'])
df_combined['Date'] = df_combined['Date_dt']
df_combined = df_combined.sort_values(['Date', 'Ticker']).reset_index(drop=True)
df_combined['Date'] = df_combined['Date'].dt.strftime('%Y-%m-%d')
print("Preparing for merge...")
print(f"df_combined shape before merge: {df_combined.shape}")
print(f"fx shape before merge: {fx.shape}")


# Ensure Date_dt exists and is datetime in df_combined
if 'Date_dt' not in df_combined.columns:
    df_combined['Date_dt'] = pd.to_datetime(df_combined['Date'])
else:
    df_combined['Date_dt'] = pd.to_datetime(df_combined['Date_dt'])


# Ensure Date_dt exists and is datetime in fx
if 'Date_dt' not in fx.columns:
    print("ERROR: fx does not have 'Date_dt' column!")
    print(f"fx columns: {fx.columns.tolist()}")
    exit(1)
else:
    fx['Date_dt'] = pd.to_datetime(fx['Date_dt'])


# Round down both dates to midnight to ensure match
df_combined['Date_dt'] = df_combined['Date_dt'].dt.normalize()
fx['Date_dt'] = fx['Date_dt'].dt.normalize()

print(f"df_combined Date_dt dtype: {df_combined['Date_dt'].dtype}")
print(f"fx Date_dt dtype: {fx['Date_dt'].dtype}")
print()


# Reset both to ensure no index issues
df_combined = df_combined.reset_index(drop=True)
fx = fx.reset_index(drop=True)


# Merge USD/INR exchange rates
print("Merging exchange rates...")
try:
    df_combined = pd.merge(df_combined, fx, on='Date_dt', how='left')
    print(f"✓ Merge successful. Combined data shape: {df_combined.shape}\n")
except Exception as e:
    print(f"ERROR during merge: {e}")
    print(f"df_combined columns: {df_combined.columns.tolist()}")
    print(f"fx columns: {fx.columns.tolist()}")
    print(f"df_combined['Date_dt'] sample:\n{df_combined['Date_dt'].head()}")
    print(f"fx['Date_dt'] sample:\n{fx['Date_dt'].head()}")
    exit(1)


# Convert Indian stock prices (INR) to USD
indian_tickers = ['RELIANCE.NS', 'TCS.NS', 'INFY.NS']
for col in ['Open', 'High', 'Low', 'Close']:
    mask = df_combined['Ticker'].isin(indian_tickers)
    df_combined.loc[mask, col] = df_combined.loc[mask, col] / df_combined.loc[mask, 'USDINR']

print("✓ Indian stocks converted from INR to USD\n")

print(f"Combined data prepared: {len(df_combined)} rows")
print(f"Date range: {df_combined['Date'].min()} to {df_combined['Date'].max()}")
print(f"Stocks included: {', '.join(df_combined['Ticker'].unique())}\n")

print("Sample of combined data (first 10 rows):")
print("-"*100)
print(df_combined.head(10).to_string(index=False))
print()


# Save combined CSV
if not os.path.exists('data'):
    os.makedirs('data')

csv_filename = 'data/combined_6stocks_5years.csv'
df_combined.to_csv(csv_filename, index=False)
print(f"Saved combined CSV: {csv_filename}")
print(f"File size: {os.path.getsize(csv_filename) / (1024*1024):.2f} MB")
print(f"Rows: {len(df_combined)}, Columns: {len(df_combined.columns)}\n")


# Prepare data for plotting
df_combined['Date_dt'] = pd.to_datetime(df_combined['Date'])
closing_prices = df_combined.pivot_table(index='Date_dt', columns='Ticker', values='Close')


# Plot combined graph
fig, ax = plt.subplots(figsize=(16, 8))
colors = ['#00008B', '#8B0000', '#006400', '#FF5C00', '#000000', '#4B0082']  # 6 hex codes

for idx, ticker in enumerate(closing_prices.columns):
    y = closing_prices[ticker].dropna()
    x = y.index
    if y.empty:
        continue
    ax.plot(
        x, y,
        label=f'{ticker} - {stocks_info[ticker]["name"]}',
        linewidth=2.5,
        color=colors[idx % len(colors)],
        # ... other args ...
    )


ax.set_title('Combined Stock Prices - 6 Stocks (5 Years, USD)', fontsize=18, fontweight='bold', pad=20)
ax.set_xlabel('Date', fontsize=13, fontweight='bold')
ax.set_ylabel('Price (USD)', fontsize=13, fontweight='bold')
ax.legend(loc='upper left', fontsize=11, framealpha=0.95, ncol=2)
ax.grid(True, alpha=0.3, linestyle='--')
ax.set_facecolor('#f8f9fa')
plt.xticks(rotation=45, ha='right')
fig.patch.set_facecolor('white')
plt.tight_layout()

plot_filename = 'data/combined_6stocks_plot.png'
plt.savefig(plot_filename, dpi=300, bbox_inches='tight')
plt.show()
plt.close()
print(f"Saved plot: {plot_filename}\n")

# Summary statistics
print("Summary statistics per stock:\n" + "="*100)
for ticker in closing_prices.columns:
    ticker_data = df_combined[df_combined['Ticker'] == ticker]
    close_prices = ticker_data['Close'].astype(float)

    print(f"Stock: {ticker} - {stocks_info[ticker]['name']} ({stocks_info[ticker]['market']})")
    print("-"*100)

    print(f"  Trading days: {len(ticker_data)}")
    print(f"  Date range: {ticker_data['Date'].min()} to {ticker_data['Date'].max()}\n")

    print(f"  Price statistics:")
    print(f"    Average:   ${close_prices.mean():.2f}")
    print(f"    Current:   ${close_prices.iloc[-1]:.2f}")
    print(f"    High (5y): ${ticker_data['High'].astype(float).max():.2f}")
    print(f"    Low (5y):  ${ticker_data['Low'].astype(float).min():.2f}\n")

    print(f"  Volume statistics:")
    print(f"    Average daily: {ticker_data['Volume'].astype(int).mean():,.0f} shares")
    print(f"    Max daily:     {ticker_data['Volume'].astype(int).max():,.0f} shares\n")

    first_close = close_prices.iloc[0]
    last_close = close_prices.iloc[-1]
    change = last_close - first_close
    change_pct = (change / first_close) * 100

    print(f"  Performance (5-year):")
    print(f"    Start price: ${first_close:.2f}")
    print(f"    End price:   ${last_close:.2f}")
    print(f"    Change:      ${change:.2f} ({change_pct:+.2f}%)\n")

print("="*100)
print("COMBINED DATA ANALYSIS COMPLETE!")
print(f"Completed at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
print("="*100)