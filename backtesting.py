import pandas as pd
import numpy as np
from scipy.stats import norm, t, chi2

print("\n" + "="*80)
print("BACKTESTING VaR MODELS (Kupiec & Christoffersen)")
print("="*80 + "\n")


# 1. Load portfolio returns

df = pd.read_csv("data/portfolio_returns.csv")
df['Date'] = pd.to_datetime(df['Date'])
returns = df['Portfolio_Return'].values

print(f"Loaded {len(returns)} daily returns.")
print(f"Date range: {df['Date'].min().date()} to {df['Date'].max().date()}\n")

CONF_LEVEL = 0.95


# 2. VaR calculation functions (same logic as var_engine.py)

def historical_var(returns, confidence=0.95):
    return -np.percentile(returns, (1 - confidence) * 100)

def parametric_var_normal(returns, confidence=0.95):
    mu = np.mean(returns)
    sigma = np.std(returns)
    z = norm.ppf(1 - confidence)
    return -(mu + sigma * z)

def parametric_var_t(returns, confidence=0.95):
    df_t, loc, scale = t.fit(returns)
    t_score = t.ppf(1 - confidence, df_t)
    return -(loc + scale * t_score)

def monte_carlo_var(returns, confidence=0.95, simulations=10000, seed=42):
    np.random.seed(seed)
    mu = np.mean(returns)
    sigma = np.std(returns)
    sim = np.random.normal(mu, sigma, simulations)
    return -np.percentile(sim, (1 - confidence) * 100)

# Compute all VaRs once
var_methods = {
    "Historical": historical_var(returns, CONF_LEVEL),
    "Normal":     parametric_var_normal(returns, CONF_LEVEL),
    "t-dist":     parametric_var_t(returns, CONF_LEVEL),
    "MonteCarlo": monte_carlo_var(returns, CONF_LEVEL),
}

print("="*80)
print(f"VaR Levels (95% Daily)")
print("="*80)
for name, v in var_methods.items():
    print(f"{name:12s}: {v*100:.3f}%")
print("="*80 + "\n")


# 3. Kupiec Proportion of Failures Test

def kupiec_pof_test(returns, var_estimate, confidence_level):
    """
    Kupiec Proportion of Failures Test
    H0: observed exception rate = expected (1 - alpha)
    """
    # VaR is positive, returns are in decimal (e.g. -0.02)
    exceptions = returns < -var_estimate
    x = exceptions.sum()          # number of exceptions
    n = len(returns)
    alpha = 1 - confidence_level  # expected exception rate

    if x == 0 or x == n:
        # Degenerate case
        lr_stat = np.inf
        p_value = 0.0
    else:
        pi_hat = x / n
        # Likelihood ratio
        lr_stat = 2 * (
            x * np.log(pi_hat / alpha) +
            (n - x) * np.log((1 - pi_hat) / (1 - alpha))
        )
        p_value = 1 - chi2.cdf(lr_stat, df=1)

    return {
        "exceptions": x,
        "total": n,
        "obs_rate": x / n,
        "exp_rate": alpha,
        "LR_POF": lr_stat,
        "p_value": p_value,
        "pass": p_value > 0.05
    }


# 4. Christoffersen Conditional Coverage Test

def christoffersen_test(returns, var_estimate, confidence_level):
    """
    Christoffersen Conditional Coverage Test
    Combines:
      - Proportion of failures (Kupiec)
      - Independence of exceptions
    """
    # 0/1 exception series
    exc = (returns < -var_estimate).astype(int)

    n00 = ((exc[:-1] == 0) & (exc[1:] == 0)).sum()
    n01 = ((exc[:-1] == 0) & (exc[1:] == 1)).sum()
    n10 = ((exc[:-1] == 1) & (exc[1:] == 0)).sum()
    n11 = ((exc[:-1] == 1) & (exc[1:] == 1)).sum()

    # Transition probabilities
    if (n00 + n01) > 0:
        p01 = n01 / (n00 + n01)
    else:
        p01 = 0.0

    if (n10 + n11) > 0:
        p11 = n11 / (n10 + n11)
    else:
        p11 = 0.0

    # Unconditional probability
    pi = exc.mean()

    # LR for independence
    # Guard against zeros to avoid log(0)
    def safe_log(x):
        return np.log(x) if x > 0 else 0.0

    lr_ind = 2 * (
        n00 * safe_log(1 - p01) +
        n01 * safe_log(p01) +
        n10 * safe_log(1 - p11) +
        n11 * safe_log(p11) -
        (n00 + n10) * safe_log(1 - pi) -
        (n01 + n11) * safe_log(pi)
    )

    # Kupiec POF component
    kupiec_res = kupiec_pof_test(returns, var_estimate, confidence_level)
    lr_pof = kupiec_res["LR_POF"]

    # Combined
    lr_cc = lr_pof + lr_ind
    p_value_cc = 1 - chi2.cdf(lr_cc, df=2)

    return {
        "LR_POF": lr_pof,
        "LR_Ind": lr_ind,
        "LR_CC": lr_cc,
        "p_value_CC": p_value_cc,
        "pass": p_value_cc > 0.05,
        "p01": p01,
        "p11": p11,
    }


# 5. Run tests for all methods

print("="*80)
print("KUPIEC PROPORTION OF FAILURES TEST (95%)")
print("="*80)

for name, v in var_methods.items():
    res = kupiec_pof_test(returns, v, CONF_LEVEL)
    print(f"\n{name}:")
    print("-" * 40)
    print(f"Exceptions       : {res['exceptions']} / {res['total']}")
    print(f"Observed rate    : {res['obs_rate']*100:.2f}%")
    print(f"Expected rate    : {res['exp_rate']*100:.2f}%")
    print(f"LR_POF           : {res['LR_POF']:.4f}")
    print(f"p-value          : {res['p_value']:.4f}")
    print(f"Test pass (5%)   : {res['pass']}")

print("\n" + "="*80)
print("CHRISTOFFERSEN CONDITIONAL COVERAGE TEST (95%)")
print("="*80)

for name, v in var_methods.items():
    res = christoffersen_test(returns, v, CONF_LEVEL)
    print(f"\n{name}:")
    print("-" * 40)
    print(f"LR_POF           : {res['LR_POF']:.4f}")
    print(f"LR_Ind           : {res['LR_Ind']:.4f}")
    print(f"LR_CC            : {res['LR_CC']:.4f}")
    print(f"p-value (CC)     : {res['p_value_CC']:.4f}")
    print(f"p01 (0→1)        : {res['p01']:.4f}")
    print(f"p11 (1→1)        : {res['p11']:.4f}")
    print(f"Test pass (5%)   : {res['pass']}")

print("\n" + "="*80)
print("BACKTESTING COMPLETE")
print("="*80 + "\n")
