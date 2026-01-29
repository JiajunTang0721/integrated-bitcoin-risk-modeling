#!/usr/bin/env python
# coding: utf-8

# In[1]:


import os
import numpy as np
import pandas as pd

DATA_PATH = "bitcoin.xlsx"  # use "data/bitcoin.xlsx" if needed
OUT_DIR = "derived"
OUT_RETURNS = os.path.join(OUT_DIR, "returns.csv")


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


def main():
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

    rets = df_r["r"].to_numpy()
    dates_r = df_r["Date"].to_numpy()
    T = len(rets)

    print(f"Return sample size T = {T}")
    print(f"Date range (returns): {dates_r[0]} to {dates_r[-1]}")

    os.makedirs(OUT_DIR, exist_ok=True)
    df_r.to_csv(OUT_RETURNS, index=False)
    print(f"Saved returns to {OUT_RETURNS}")


if __name__ == "__main__":
    main()


# In[ ]:




