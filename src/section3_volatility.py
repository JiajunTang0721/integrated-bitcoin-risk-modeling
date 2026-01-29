#!/usr/bin/env python
# coding: utf-8

# In[1]:


import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

DATA_PATH = "bitcoin.xlsx"  # fallback if derived/returns.csv is not found
RETURNS_CSV = os.path.join("derived", "returns.csv")
OUT_DIR = "derived"
OUT_EWMA = os.path.join(OUT_DIR, "df_ewma.csv")


def find_col(candidates, columns):
    lower_map = {c.lower().strip(): c for c in columns}
    for cand in candidates:
        if cand in lower_map:
            return lower_map[cand]
    for c in columns:
        lc = c.lower()
        for cand in candidates:
            if cand in lc:
                return c
    return None


def acf(x, max_lag=30):
    x = np.asarray(x)
    x = x - x.mean()
    denom = np.sum(x**2)
    out = np.empty(max_lag + 1, dtype=float)
    for k in range(max_lag + 1):
        out[k] = np.sum(x[k:] * x[: len(x) - k]) / denom
    return out


def load_returns():
    if os.path.exists(RETURNS_CSV):
        df_r = pd.read_csv(RETURNS_CSV)
        df_r["Date"] = pd.to_datetime(df_r["Date"])
        df_r = df_r.sort_values("Date").reset_index(drop=True)
        return df_r

    df = pd.read_excel(DATA_PATH)
    date_col = find_col(["date", "dates"], df.columns)
    price_col = find_col(["spot price", "price", "close", "closing price", "spot"], df.columns)

    if date_col is None:
        raise ValueError(f"Could not detect a date column. Available columns: {list(df.columns)}")
    if price_col is None:
        raise ValueError(f"Could not detect a price column. Available columns: {list(df.columns)}")

    df = df[[date_col, price_col]].rename(columns={date_col: "Date", price_col: "Price"})
    df["Date"] = pd.to_datetime(df["Date"])
    df = df.sort_values("Date").reset_index(drop=True)

    df["r"] = np.log(df["Price"] / df["Price"].shift(1))
    df_r = df.loc[df["r"].notna(), ["Date", "Price", "r"]].reset_index(drop=True)
    return df_r


def main():
    df_r = load_returns()

    rets = df_r["r"].to_numpy()
    dates_r = df_r["Date"].to_numpy()
    T = len(rets)

    print(f"Return sample size T = {T}")
    print(f"Date range (returns): {dates_r[0]} to {dates_r[-1]}")

    lam = 0.94

    sigma2 = np.empty(T, dtype=float)
    sigma2[:] = np.nan

    sigma2_1 = np.mean(rets**2)
    sigma2[0] = sigma2_1

    for t in range(1, T):
        sigma2[t] = lam * sigma2[t - 1] + (1 - lam) * (rets[t - 1] ** 2)

    sigma = np.sqrt(sigma2)

    df_ewma = pd.DataFrame({"Date": dates_r, "r": rets, "sigma2": sigma2, "sigma": sigma})

    print("\nEWMA summary:")
    print(df_ewma[["sigma", "sigma2"]].describe(percentiles=[0.01, 0.05, 0.5, 0.95, 0.99]))

    plt.figure()
    plt.plot(df_ewma["Date"], df_ewma["r"] * 100)
    plt.xlabel("Date")
    plt.ylabel("Daily log return (%)")
    plt.title("Bitcoin Daily Log Returns")
    plt.tight_layout()
    plt.show()

    plt.figure()
    plt.plot(df_ewma["Date"], df_ewma["sigma"] * 100)
    plt.xlabel("Date")
    plt.ylabel("EWMA volatility (%)")
    plt.title(f"EWMA Volatility (lambda={lam})")
    plt.tight_layout()
    plt.show()

    max_lag = 30
    acf_r2 = acf(df_ewma["r"].to_numpy() ** 2, max_lag=max_lag)
    acf_sigma2 = acf(df_ewma["sigma2"].to_numpy(), max_lag=max_lag)

    print("\nACF (squared returns) first 10 lags:")
    for k in range(1, 11):
        print(f"lag {k:2d}: {acf_r2[k]: .4f}")

    print("\nACF (EWMA variance) first 10 lags:")
    for k in range(1, 11):
        print(f"lag {k:2d}: {acf_sigma2[k]: .4f}")

    plt.figure()
    plt.bar(np.arange(max_lag + 1), acf_r2)
    plt.xlabel("Lag")
    plt.ylabel("ACF")
    plt.title("Autocorrelation of Squared Returns (Volatility Clustering Evidence)")
    plt.tight_layout()
    plt.show()

    plt.figure()
    plt.bar(np.arange(max_lag + 1), acf_sigma2)
    plt.xlabel("Lag")
    plt.ylabel("ACF")
    plt.title("Autocorrelation of EWMA Variance (Persistence Evidence)")
    plt.tight_layout()
    plt.show()

    r2_lag = (df_ewma["r"].shift(1) ** 2).to_numpy()
    sigma2_now = df_ewma["sigma2"].to_numpy()
    mask = ~np.isnan(r2_lag)

    corr = np.corrcoef(r2_lag[mask], sigma2_now[mask])[0, 1]
    print(f"\nCorr(r_(t-1)^2, sigma_t^2) = {corr:.4f}  (should be strongly positive)")

    r2 = df_ewma["r"].to_numpy() ** 2
    thr = np.quantile(r2, 0.99)
    shock_idx = np.where(r2 >= thr)[0]

    window = 5
    before = []
    after = []
    for t in shock_idx:
        if t - 1 >= 0:
            before.append(df_ewma.loc[t - 1, "sigma"])
        if t + window < T:
            after.append(df_ewma.loc[t + window, "sigma"])

    if len(before) > 0 and len(after) > 0:
        print(f"Avg sigma% one day before extreme shock: {np.mean(before) * 100:.3f}%")
        print(f"Avg sigma% {window} days after extreme shock: {np.mean(after) * 100:.3f}%")
    else:
        print("Not enough observations to compute shock window stats.")

    os.makedirs(OUT_DIR, exist_ok=True)
    df_ewma.to_csv(OUT_EWMA, index=False)

    print("\nDataFrame df_ewma ready for Section 4/5/6 (VaR/ES/Backtesting).")
    print(df_ewma.head())


if __name__ == "__main__":
    main()


# In[ ]:




