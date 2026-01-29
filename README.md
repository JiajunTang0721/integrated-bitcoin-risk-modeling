# Integrated Bitcoin Risk Modeling

This repository presents an integrated financial risk management framework for Bitcoin,
developed within the instructional framework of  
**IEOR 4745 – Applied Financial Risk Management (Columbia University)**.

The project combines volatility modeling, market risk measurement, tail risk assessment,
and regulatory-style backtesting into a unified analytical pipeline, with an empirical
application to Bitcoin daily returns.

---

## 📄 Paper (Direct Download)

👉 **[Download the full paper (PDF)](paper/main.pdf)**

The paper is written in LaTeX and compiled from:

- `paper/main.tex`  
- Figures stored in `paper/figures/`

---

## 🔍 Summary of Methods and Key Findings

### Methods

- **Returns:** Daily log returns constructed from BTC/USD prices  
- **Volatility modeling:** EWMA with λ = 0.94  
- **Market risk:**  
  - Parametric VaR and ES under EWMA–Normal assumptions  
  - Historical simulation VaR and ES (2-year rolling window, 731 observations)  
- **Tail risk:** Expected Shortfall computed using exact lognormal P&L  
- **Backtesting:** Kupiec unconditional coverage test at 95%, 99%, and 99.5% confidence levels  

---

### Key Empirical Findings

- Bitcoin returns exhibit **strong volatility clustering and persistence**, captured effectively by EWMA volatility.
- **Parametric VaR under normality systematically understates tail risk**, especially at extreme confidence levels.
- **Expected Shortfall consistently exceeds VaR**, confirming its superior sensitivity to extreme losses.
- **Historical simulation produces substantially larger VaR and ES estimates**, reflecting the heavy-tailed nature of Bitcoin returns.
- Kupiec backtests indicate:
  - Acceptable coverage at moderate confidence levels (95%)
  - **Systematic rejection at extreme confidence levels (99% and 99.5%)**, particularly for short positions

These results highlight the limitations of classical parametric risk models when applied
to cryptocurrencies and underscore the importance of tail-sensitive measures and empirical validation.

---

## 📊 Selected Quantitative Results

**Backtesting window:** 01 Aug 2023 – 31 Jul 2025 (T = 730)

| Confidence | Acceptance Interval | Long Exceedances | Reject | Short Exceedances | Reject |
|------------|---------------------|------------------|--------|-------------------|--------|
| 95%        | [25, 48]            | 34               | No     | 50                | Yes    |
| 99%        | [3, 13]             | 13               | No     | 20                | Yes    |
| 99.5%      | [0, 8]              | 9                | Yes    | 16                | Yes    |

(Full tables and derivations are provided in the paper.)

---

## 📂 Repository Structure

```text
integrated-bitcoin-risk-modeling/
├── paper/
│   ├── main.tex        # LaTeX source
│   ├── main.pdf        # Compiled paper (downloadable)
│   └── figures/        # Figures used in the paper
│
├── src/
│   ├── section3_ewma.py        # EWMA volatility modeling
│   ├── section4_var.py         # Parametric & historical VaR
│   ├── section5_es.py          # Expected Shortfall
│   └── section6_backtest.py    # Kupiec VaR backtesting
│
├── data/
│   └── bitcoin.xlsx            # BTC/USD daily price data
│
└── README.md
