#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
import pandas as pd
from math import ceil
from statistics import NormalDist

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

V0 = 1_000_000
asof_date = pd.Timestamp("2025-07-31")
alphas = [0.95, 0.99, 0.995]
nd = NormalDist()

row = ret_df.loc[ret_df["Date"] == asof_date]
if row.empty:
    raise ValueError("31Jul2025 not found in return series. Check date range.")
sigma_asof = float(row["sigma"].iloc[0])

print("\n=== Section 5: Expected Shortfall (ES) ===")
print(f"EWMA sigma(asof 31Jul2025) = {sigma_asof:.7f} (decimal) = {100*sigma_asof:.5f}%")
print(f"Portfolio value V0 = ${V0:,.0f}")


def Phi(x):
    return nd.cdf(x)


def parametric_es_lognormal(s, alpha):
    z_low = nd.inv_cdf(1 - alpha)
    c_low = z_low * s
    p_low = Phi(c_low / s)

    num_low = np.exp(0.5 * s**2) * Phi((c_low - s**2) / s)
    Eexp_low = num_low / p_low
    Earith_low = Eexp_low - 1
    ES_long = -V0 * Earith_low

    z_high = nd.inv_cdf(alpha)
    c_high = z_high * s
    p_high = 1 - Phi(c_high / s)

    num_high = np.exp(0.5 * s**2) * (1 - Phi((c_high - s**2) / s))
    Eexp_high = num_high / p_high
    Earith_high = Eexp_high - 1
    ES_short = V0 * Earith_high

    return {"z_(1-alpha)": z_low, "z_alpha": z_high, "ES_long_$": ES_long, "ES_short_$": ES_short}


def parametric_var_exact_lognormal(s, alpha):
    z_low = nd.inv_cdf(1 - alpha)
    z_high = nd.inv_cdf(alpha)

    scen_long_log = z_low * s
    scen_short_log = z_high * s

    ar_long = np.exp(scen_long_log) - 1
    ar_short = np.exp(scen_short_log) - 1

    pnl_long = V0 * ar_long
    pnl_short = (-V0) * ar_short

    VaR_long = -pnl_long
    VaR_short = -pnl_short

    return {"VaR_long_$": VaR_long, "VaR_short_$": VaR_short, "Parametric_log_scenario_%": 100 * abs(scen_long_log)}


start_price = pd.Timestamp("2023-07-31")
end_price = pd.Timestamp("2025-07-31")

df_win = df.loc[(df["Date"] >= start_price) & (df["Date"] <= end_price), ["Date", "Price"]].copy()
df_win = df_win.sort_values("Date").reset_index(drop=True)
df_win["r"] = np.log(df_win["Price"] / df_win["Price"].shift(1))
r_win = df_win.loc[df_win["r"].notna(), ["Date", "Price", "r"]].copy().reset_index(drop=True)

m = len(r_win)
print(f"\nHistorical window return count m = {m} (expected 731)")

r_sorted = np.sort(r_win["r"].to_numpy())
arith_sorted = np.exp(r_sorted) - 1

param_rows = []
hist_rows = []
ratio_rows = []

for a in alphas:
    pv = parametric_var_exact_lognormal(sigma_asof, a)
    pe = parametric_es_lognormal(sigma_asof, a)

    k_low = ceil((1 - a) * m)
    k_high = ceil(a * m)

    r_q_long = r_sorted[k_low - 1]
    r_q_short = r_sorted[k_high - 1]

    ar_q_long = np.exp(r_q_long) - 1
    ar_q_short = np.exp(r_q_short) - 1

    VaR_long_hist = -V0 * ar_q_long
    VaR_short_hist = V0 * ar_q_short

    ES_long_hist = -V0 * np.mean(arith_sorted[:k_low])
    tail_short = arith_sorted[k_high - 1 :]
    ES_short_hist = V0 * np.mean(tail_short)

    param_rows.append(
        {
            "alpha": a,
            "ES_long_$": pe["ES_long_$"],
            "ES_short_$": pe["ES_short_$"],
            "VaR_long_$": pv["VaR_long_$"],
            "VaR_short_$": pv["VaR_short_$"],
            "ES_over_VaR_long": pe["ES_long_$"] / pv["VaR_long_$"],
            "ES_over_VaR_short": pe["ES_short_$"] / pv["VaR_short_$"],
        }
    )

    hist_rows.append(
        {
            "alpha": a,
            "rank_long": k_low,
            "VaR_long_$": VaR_long_hist,
            "ES_long_$": ES_long_hist,
            "rank_short": k_high,
            "VaR_short_$": VaR_short_hist,
            "ES_short_$": ES_short_hist,
            "ES_over_VaR_long": ES_long_hist / VaR_long_hist,
            "ES_over_VaR_short": ES_short_hist / VaR_short_hist,
        }
    )

    ratio_rows.append(
        {
            "alpha_%": 100 * a,
            "Parametric_log_scenario_%": pv["Parametric_log_scenario_%"],
            "Historical_log_q_long_%": 100 * r_q_long,
            "Historical_log_q_short_%": 100 * r_q_short,
        }
    )

param_tbl = pd.DataFrame(param_rows)
hist_tbl = pd.DataFrame(hist_rows)
scen_tbl = pd.DataFrame(ratio_rows)

pd.set_option("display.float_format", lambda x: f"{x:,.6f}")

print("\n=== Parametric ES (EWMA–Normal, exact lognormal P&L) ===")
print(
    param_tbl[
        [
            "alpha",
            "VaR_long_$",
            "ES_long_$",
            "ES_over_VaR_long",
            "VaR_short_$",
            "ES_short_$",
            "ES_over_VaR_short",
        ]
    ]
)

print("\n=== Historical Simulation ES (last 2 years, m=731) ===")
print(
    hist_tbl[
        [
            "alpha",
            "rank_long",
            "VaR_long_$",
            "ES_long_$",
            "ES_over_VaR_long",
            "rank_short",
            "VaR_short_$",
            "ES_short_$",
            "ES_over_VaR_short",
        ]
    ]
)

print("\n=== Scenario comparison in log-return space (%) ===")
print(scen_tbl)


def fmt_dollar(x):
    return f"{x:,.2f}"


print("\n--- Dollar ES (Parametric) ---")
for _, r in param_tbl.iterrows():
    print(
        f"alpha={r['alpha']:.3f}: long ES={fmt_dollar(r['ES_long_$'])} (ratio {r['ES_over_VaR_long']:.4f}), "
        f"short ES={fmt_dollar(r['ES_short_$'])} (ratio {r['ES_over_VaR_short']:.4f})"
    )

print("\n--- Dollar ES (Historical) ---")
for _, r in hist_tbl.iterrows():
    print(
        f"alpha={r['alpha']:.3f}: long ES={fmt_dollar(r['ES_long_$'])} (rank {int(r['rank_long'])}, ratio {r['ES_over_VaR_long']:.4f}), "
        f"short ES={fmt_dollar(r['ES_short_$'])} (rank {int(r['rank_short'])}, ratio {r['ES_over_VaR_short']:.4f})"
    )


# In[ ]:




