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


def fmt_dollar(x):
    return f"{x:,.2f}"


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

row = ret_df.loc[ret_df["Date"] == asof_date]
if row.empty:
    raise ValueError("31Jul2025 not found in return series. Check Date format/range in bitcoin.xlsx.")

sigma_asof = float(row["sigma"].iloc[0])

alphas = [0.95, 0.99, 0.995]
nd = NormalDist()

param_rows = []
for a in alphas:
    z_low = nd.inv_cdf(1 - a)
    z_high = nd.inv_cdf(a)

    scen_long_log = z_low * sigma_asof
    scen_short_log = z_high * sigma_asof

    scen_long_arith = np.exp(scen_long_log) - 1
    scen_short_arith = np.exp(scen_short_log) - 1

    pnl_long = V0 * scen_long_arith
    pnl_short = (-V0) * scen_short_arith

    var_long_exact = -pnl_long
    var_short_exact = -pnl_short

    var_approx = V0 * abs(z_low) * sigma_asof

    param_rows.append(
        {
            "alpha": a,
            "z_(1-alpha)": z_low,
            "z_alpha": z_high,
            "VaR_long_exact_$": var_long_exact,
            "VaR_short_exact_$": var_short_exact,
            "VaR_approx_$": var_approx,
            "Parametric_VaR_scenario_log_%": 100 * abs(scen_long_log),
        }
    )

param_tbl = pd.DataFrame(param_rows)

print("\n=== Parametric VaR (as of 31Jul2025, V0=$1,000,000) ===")
print(f"EWMA sigma(asof) = {sigma_asof:.7f} (decimal), = {100*sigma_asof:.5f}%")
print(
    param_tbl[
        [
            "alpha",
            "z_(1-alpha)",
            "z_alpha",
            "VaR_long_exact_$",
            "VaR_short_exact_$",
            "VaR_approx_$",
            "Parametric_VaR_scenario_log_%",
        ]
    ]
)

start_price = pd.Timestamp("2023-07-31")
end_price = pd.Timestamp("2025-07-31")

df_win = df.loc[(df["Date"] >= start_price) & (df["Date"] <= end_price), ["Date", "Price"]].copy()
df_win = df_win.sort_values("Date").reset_index(drop=True)
df_win["r"] = np.log(df_win["Price"] / df_win["Price"].shift(1))
r_win = df_win.loc[df_win["r"].notna(), ["Date", "Price", "r"]].copy().reset_index(drop=True)

m = len(r_win)
print(f"\nHistorical simulation return count m = {m} (expected 731)")

r_sorted = np.sort(r_win["r"].to_numpy())

hs_rows = []
for a in alphas:
    k_low = ceil((1 - a) * m)
    k_high = ceil(a * m)

    r_q_long = r_sorted[k_low - 1]
    r_q_short = r_sorted[k_high - 1]

    ar_q_long = np.exp(r_q_long) - 1
    ar_q_short = np.exp(r_q_short) - 1

    pnl_long = V0 * ar_q_long
    pnl_short = (-V0) * ar_q_short

    hs_rows.append(
        {
            "alpha": a,
            "(1-alpha)*m": (1 - a) * m,
            "rank_long": k_low,
            "log_q_long_%": 100 * r_q_long,
            "arith_q_long_%": 100 * ar_q_long,
            "VaR_long_$": -pnl_long,
            "rank_short": k_high,
            "log_q_short_%": 100 * r_q_short,
            "arith_q_short_%": 100 * ar_q_short,
            "VaR_short_$": -pnl_short,
        }
    )

hs_tbl = pd.DataFrame(hs_rows)

print("\n=== Historical Simulation VaR (as of 31Jul2025, last 2 years, V0=$1,000,000) ===")
print(
    hs_tbl[
        [
            "alpha",
            "(1-alpha)*m",
            "rank_long",
            "log_q_long_%",
            "arith_q_long_%",
            "VaR_long_$",
            "rank_short",
            "log_q_short_%",
            "arith_q_short_%",
            "VaR_short_$",
        ]
    ]
)

scenario_tbl = pd.DataFrame(
    {
        "alpha_%": [100 * a for a in alphas],
        "Historical_log_q_long_%": hs_tbl["log_q_long_%"].to_numpy(),
        "Parametric_log_scenario_%": param_tbl["Parametric_VaR_scenario_log_%"].to_numpy(),
        "Historical_log_q_short_%": hs_tbl["log_q_short_%"].to_numpy(),
        "Parametric_log_scenario_short_%": param_tbl["Parametric_VaR_scenario_log_%"].to_numpy(),
    }
)

print("\n=== VaR scenarios in log-return terms (percent) ===")
print(scenario_tbl)

print("\n--- Dollar VaR (Parametric exact) ---")
for _, r in param_tbl.iterrows():
    print(
        f"alpha={r['alpha']:.3f}: long={fmt_dollar(r['VaR_long_exact_$'])}, "
        f"short={fmt_dollar(r['VaR_short_exact_$'])}, approx={fmt_dollar(r['VaR_approx_$'])}"
    )

print("\n--- Dollar VaR (Historical simulation) ---")
for _, r in hs_tbl.iterrows():
    print(
        f"alpha={r['alpha']:.3f}: long={fmt_dollar(r['VaR_long_$'])} (rank {int(r['rank_long'])}), "
        f"short={fmt_dollar(r['VaR_short_$'])} (rank {int(r['rank_short'])})"
    )


# In[ ]:




