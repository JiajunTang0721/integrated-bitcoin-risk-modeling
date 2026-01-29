#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
import pandas as pd
from statistics import NormalDist
from scipy.stats import chi2, binom

DATA_PATH = "bitcoin.xlsx"


def find_col(candidates, cols):
    lower = {c.lower().strip(): c for c in cols}
    for cand in candidates:
        if cand in lower:
            return lower[cand]
    for c in cols:
        lc = c.lower()
        for cand in candidates:
            if cand in lc:
                return c
    return None


def kupiec_lr_uc(x, T, p):
    if x == 0:
        phat = 1e-12
    elif x == T:
        phat = 1 - 1e-12
    else:
        phat = x / T
    L0 = (T - x) * np.log(1 - p) + x * np.log(p)
    L1 = (T - x) * np.log(1 - phat) + x * np.log(phat)
    return -2 * (L0 - L1)


def acceptance_interval(T, p, alpha_test=0.05):
    lower_tail = alpha_test / 2
    upper_tail = 1 - alpha_test / 2

    x_low = int(binom.ppf(lower_tail, T, p))
    while binom.cdf(x_low, T, p) < lower_tail and x_low < T:
        x_low += 1

    x_high = int(binom.ppf(upper_tail, T, p))
    while x_high > 0 and binom.cdf(x_high - 1, T, p) > upper_tail:
        x_high -= 1

    return x_low, x_high


df = pd.read_excel(DATA_PATH)

date_col = find_col(["date", "dates"], df.columns)
price_col = find_col(["spot price", "price", "close", "closing price", "spot"], df.columns)
if date_col is None or price_col is None:
    raise ValueError(f"Cannot detect date/price columns. Columns={list(df.columns)}")

df = df[[date_col, price_col]].rename(columns={date_col: "Date", price_col: "Price"})
df["Date"] = pd.to_datetime(df["Date"])
df = df.sort_values("Date").reset_index(drop=True)

df["r"] = np.log(df["Price"] / df["Price"].shift(1))
ret_df = df.loc[df["r"].notna(), ["Date", "Price", "r"]].copy().reset_index(drop=True)

lam = 0.94
first_21 = ret_df.iloc[:21]["r"].to_numpy()
sigma2_0 = np.mean(first_21**2)

sigma2 = np.empty(len(ret_df), dtype=float)
sigma2[:] = np.nan
sigma2[0] = lam * sigma2_0 + (1 - lam) * (ret_df.loc[0, "r"] ** 2)
for t in range(1, len(ret_df)):
    sigma2[t] = lam * sigma2[t - 1] + (1 - lam) * (ret_df.loc[t, "r"] ** 2)

ret_df["sigma2"] = sigma2
ret_df["sigma"] = np.sqrt(sigma2)

start_bt = pd.Timestamp("2023-08-01")
end_bt = pd.Timestamp("2025-07-31")
bt = ret_df.loc[(ret_df["Date"] >= start_bt) & (ret_df["Date"] <= end_bt), ["Date", "r", "sigma"]].copy().reset_index(drop=True)

T = len(bt)
print("\n=== Section 6: VaR Backtesting (Unconditional Coverage / Kupiec) ===")
print(f"Backtest window: {start_bt.date()} to {end_bt.date()}")
print(f"Return count T = {T} (expected 731)")

bt["sigma_prev"] = bt["sigma"].shift(1)
bt = bt.dropna().reset_index(drop=True)
T = len(bt)
print(f"Effective T after aligning sigma_prev = {T}")

nd = NormalDist()
alphas = [0.95, 0.99, 0.995]

rows = []
for a in alphas:
    p = 1 - a
    z_low = nd.inv_cdf(1 - a)
    z_high = nd.inv_cdf(a)

    thr_long = z_low * bt["sigma_prev"]
    thr_short = z_high * bt["sigma_prev"]

    exc_long = (bt["r"] < thr_long).astype(int)
    exc_short = (bt["r"] > thr_short).astype(int)

    x_long = int(exc_long.sum())
    x_short = int(exc_short.sum())

    LR_long = kupiec_lr_uc(x_long, T, p)
    LR_short = kupiec_lr_uc(x_short, T, p)
    pval_long = 1 - chi2.cdf(LR_long, df=1)
    pval_short = 1 - chi2.cdf(LR_short, df=1)

    x_low, x_high = acceptance_interval(T, p, alpha_test=0.05)

    rows.append(
        {
            "alpha": a,
            "T": T,
            "p=1-alpha": p,
            "accept_x_low": x_low,
            "accept_x_high": x_high,
            "x_long": x_long,
            "LR_uc_long": LR_long,
            "pval_long": pval_long,
            "reject_long_5%": (pval_long < 0.05),
            "x_short": x_short,
            "LR_uc_short": LR_short,
            "pval_short": pval_short,
            "reject_short_5%": (pval_short < 0.05),
        }
    )

out = pd.DataFrame(rows)

print("\n=== Kupiec Unconditional Coverage Backtest (5% test) ===")
pd.set_option("display.float_format", lambda x: f"{x:,.6f}")
print(
    out[
        [
            "alpha",
            "T",
            "p=1-alpha",
            "accept_x_low",
            "accept_x_high",
            "x_long",
            "LR_uc_long",
            "pval_long",
            "reject_long_5%",
            "x_short",
            "LR_uc_short",
            "pval_short",
            "reject_short_5%",
        ]
    ]
)

alpha_show = 0.99
z_low = nd.inv_cdf(1 - alpha_show)
z_high = nd.inv_cdf(alpha_show)
thr_long = z_low * bt["sigma_prev"]
thr_short = z_high * bt["sigma_prev"]

exc_long = bt.loc[bt["r"] < thr_long, ["Date", "r"]].copy()
exc_short = bt.loc[bt["r"] > thr_short, ["Date", "r"]].copy()

print(f"\n--- Exceedance dates (alpha={alpha_show}) ---")
print(f"Long exceedances count = {len(exc_long)}")
print(exc_long.head(10))
print(f"Short exceedances count = {len(exc_short)}")
print(exc_short.head(10))


# In[ ]:




