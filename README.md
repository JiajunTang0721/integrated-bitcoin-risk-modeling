# Integrated Bitcoin Risk Modeling

This repository presents an integrated financial risk management framework for Bitcoin,
developed within the instructional framework of **IEOR 4745 Applied Financial Risk Management**
at Columbia University.

The project combines volatility modeling, market risk measurement, tail risk assessment,
and regulatory-style backtesting into a single, coherent analytical pipeline.

---

## Paper

The complete paper is available in PDF format:

- `paper/main.pdf`

The LaTeX source used to generate the paper is located at:

- `paper/main.tex`

All figures referenced in the paper are stored in:

- `paper/figures/`

---

## Overview

- **Asset:** Bitcoin (BTC/USD)  
- **Sample period:** 2015–2025  
- **Methods:**  
  - Log-return construction  
  - EWMA volatility modeling  
  - Parametric and historical Value at Risk (VaR)  
  - Expected Shortfall (ES)  
  - Kupiec unconditional coverage backtesting  

---

## Repository Structure

```text
integrated-bitcoin-risk-modeling/
├── paper/
│   ├── main.tex        # LaTeX source of the paper
│   ├── main.pdf        # Compiled paper
│   └── figures/        # Figures used in the paper
│
├── src/
│   ├── section3_ewma.py        # Volatility modeling (EWMA)
│   ├── section4_var.py         # Parametric & historical VaR
│   ├── section5_es.py          # Expected Shortfall analysis
│   └── section6_backtest.py    # Kupiec VaR backtesting
│
├── data/
│   └── bitcoin.xlsx            # Input price data (not included by default)
│
└── README.md
