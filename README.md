## 📄 Paper

👉 **[Download the full paper (PDF)](paper/main.pdf)**

This paper develops an integrated risk modeling framework for Bitcoin, covering:
- EWMA volatility modeling
- Parametric and historical VaR
- Expected Shortfall (ES)
- Regulatory backtesting (Kupiec test)

# Integrated Bitcoin Risk Modeling

This repository presents an integrated financial risk management framework for Bitcoin, developed within the instructional framework of IEOR 4745 Applied Financial Risk Management at Columbia University.

## Overview
- Asset: Bitcoin (BTC/USD)
- Horizon: 2015–2025
- Methods: EWMA volatility, Value at Risk (VaR), Expected Shortfall (ES), and regulatory backtesting

## Methodology
1. Log-return construction
2. EWMA volatility modeling
3. Parametric and historical VaR
4. Expected Shortfall
5. Kupiec unconditional coverage backtesting

## Key Findings
- Parametric VaR under normality assumptions understates tail risk
- Historical simulation captures heavy-tailed risk more effectively
- Expected Shortfall provides more informative tail risk measures
- EWMA-based VaR fails at extreme confidence levels in backtesting

## Repository Structure
```text
paper/        LaTeX source of the paper
src/          Python implementation
data/         Input data
