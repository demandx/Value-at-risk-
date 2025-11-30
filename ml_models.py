import pandas as pd
import numpy as np

from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.pipeline import Pipeline

print("\n" + "="*80)
print("MACHINE LEARNING MODELS FOR NEXT-DAY PORTFOLIO RETURN")
print("="*80 + "\n")


# 1. Load portfolio returns

data_path = "data/portfolio_returns.csv"
df = pd.read_csv(data_path)

df['Date'] = pd.to_datetime(df['Date'])
df = df.sort_values('Date').reset_index(drop=True)

if 'Portfolio_Return' not in df.columns:
    raise ValueError("Column 'Portfolio_Return' not found in portfolio_returns.csv")

print(f"Loaded {len(df)} rows of portfolio returns.")
print(f"Date range: {df['Date'].min().date()} to {df['Date'].max().date()}\n")


df_features = df.copy()
r = df_features['Portfolio_Return']

# Lagged returns
df_features['ret_lag1'] = r.shift(1)
df_features['ret_lag2'] = r.shift(2)
df_features['ret_lag5'] = r.shift(5)

# Rolling volatility (std dev)
df_features['vol_5'] = r.rolling(window=5).std()
df_features['vol_20'] = r.rolling(window=20).std()

# Rolling historical VaR (95%) using last 100 days
window = 100
def rolling_var_95(x):
    return -np.percentile(x, 5)

df_features['var_95_100d'] = r.rolling(window=window).apply(rolling_var_95, raw=True)

# Target: next-day return
df_features['target_next_ret'] = r.shift(-1)

# Drop rows with NaNs created by shifting/rolling
df_features = df_features.dropna().reset_index(drop=True)

print("Feature engineering complete.")
print("Columns used as features:")
feature_cols = ['ret_lag1', 'ret_lag2', 'ret_lag5', 'vol_5', 'vol_20', 'var_95_100d']
print(feature_cols, "\n")

X = df_features[feature_cols].values
y = df_features['target_next_ret'].values
dates = df_features['Date'].values

print(f"Final dataset size after dropping NaNs: {X.shape[0]} samples\n")


# 3. Train-test split (time series style: no shuffling)

split_idx = int(0.8 * len(X))
X_train, X_test = X[:split_idx], X[split_idx:]
y_train, y_test = y[:split_idx], y[split_idx:]
dates_train, dates_test = dates[:split_idx], dates[split_idx:]

print(f"Train samples: {len(X_train)}, Test samples: {len(X_test)}\n")


# 4. Define models
#    - Linear Regression (baseline)
#    - Random Forest
#    - Gradient Boosting (or XGBoost-like)

models = {}

# Linear Regression with scaling
models['LinearRegression'] = Pipeline([
    ('scaler', StandardScaler()),
    ('model', LinearRegression())
])

# Random Forest (no scaling needed)
models['RandomForest'] = RandomForestRegressor(
    n_estimators=200,
    max_depth=8,
    random_state=42,
    n_jobs=-1
)

# Gradient Boosting
models['GradientBoosting'] = GradientBoostingRegressor(
    n_estimators=200,
    max_depth=3,
    learning_rate=0.05,
    random_state=42
)


try:
    from xgboost import XGBRegressor
    models['XGBoost'] = XGBRegressor(
        n_estimators=300,
        max_depth=3,
        learning_rate=0.05,
        subsample=0.8,
        colsample_bytree=0.8,
        objective='reg:squarederror',
        random_state=42,
        n_jobs=-1
    )
    xgb_available = True
    print("XGBoost detected. Including XGBoost in models.\n")
except ImportError:
    xgb_available = False
    print("XGBoost not installed. Skipping XGBoost model.\n")


results = []

def direction_accuracy(y_true, y_pred):
    sign_true = np.sign(y_true)
    sign_pred = np.sign(y_pred)
    return np.mean(sign_true == sign_pred)

for name, model in models.items():
    print("="*80)
    print(f"Training model: {name}")
    print("="*80)

    # Fit
    model.fit(X_train, y_train)

    # Predict
    y_pred = model.predict(X_test)

    # Metrics
    mae = mean_absolute_error(y_test, y_pred)
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    r2 = r2_score(y_test, y_pred)
    dir_acc = direction_accuracy(y_test, y_pred)

    print(f"MAE               : {mae:.6f}")
    print(f"RMSE              : {rmse:.6f}")
    print(f"R²                : {r2:.4f}")
    print(f"Direction Accuracy: {dir_acc*100:.2f}%")
    print()

    results.append({
        'Model': name,
        'MAE': mae,
        'RMSE': rmse,
        'R2': r2,
        'Direction_Acc': dir_acc
    })


results_df = pd.DataFrame(results)
results_df = results_df.sort_values('RMSE')

print("="*80)
print("MODEL COMPARISON SUMMARY (sorted by RMSE)")
print("="*80)
print(results_df.to_string(index=False))
print("="*80 + "\n")

# Save results
out_path = "data/ml_model_results.csv"
results_df.to_csv(out_path, index=False)
print(f"Saved ML model results to: {out_path}\n")
print("ML pipeline complete.\n")
