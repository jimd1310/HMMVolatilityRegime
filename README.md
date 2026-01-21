# Volatility Regimes in Equity Returns

This project investigates whether volatility clustering in equity returns is better explained by **discrete latent regimes** or **continuous autoregressive volatility dynamics**. Using daily SPY log-returns from 2000–2025, I compare a Gaussian Hidden Markov Model (HMM) against standard benchmarks, including i.i.d. Gaussian, i.i.d. Student-t, and GARCH(1,1), under a strict fixed-origin density forecasting framework spanning the COVID-19 structural break.

## What to Read First

- [**Full report (PDF)**](report/HMMVolatilityRegime.pdf) — complete methodology, results, and discussion.
- **[Main analysis notebook](analysis.ipynb)** — reproduces all tables and figures.
- **[Core model code](src/models/hmm.py)** — HMM estimation, filtering, and decoding logic.

## Key Findings

- A **4-state Gaussian HMM** is strongly favored by BIC and exhibits persistent, economically interpretable volatility regimes.
- Both **HMM and GARCH decisively outperform static return models**, confirming persistent time-varying volatility.
- The HMM delivers **superior in-sample fit and regime interpretability**, while **GARCH provides the strongest out-of-sample density forecasts** at short horizons.
- Results suggest **regime-switching and continuous volatility models capture complementary aspects** of volatility clustering rather than acting as substitutes.

## Repository Structure

- `src/` contains reusable model code (HMM, GARCH, likelihoods, forecasting logic).
- `analysis.ipynb` runs experiments, generates tables and figures, and calls `src/`.
- `report/` contains the full academic-style write-up and final PDF.

## Methods Overview

- **Models:**  
  - Gaussian Hidden Markov Models (2–5 states)  
  - i.i.d. Gaussian  
  - i.i.d. Student-$t$  
  - GARCH(1,1) with Gaussian innovations  

- **Estimation:**  
  - HMM parameters via Baum–Welch (EM), with multiple random initializations  
  - GARCH via maximum likelihood  

- **Model Selection:**  
  - Bayesian Information Criterion (BIC)  
  - Minimum expected regime duration to exclude degenerate states  

- **Forecasting:**  
  - Fixed-origin design: models trained on 2000–2019, evaluated on 2020–2025  
  - Recursive state updating (HMM filtering, GARCH variance recursion)  
  - One-step-ahead density forecasts evaluated using log predictive scores  

## Reproducing Results

1. Install dependencies:
   ```bash
   pip install -r requirements.txt

2. Run the analysis notebook:
   ```bash
   jupyter notebook analysis.ipynb
   ```

All tables and figures in the report are generated from this notebook.

## Notes on Limitations

- Gaussian emissions may partially conflate regime shifts with tail risk.
- Regime identification is sensitive to state count and distributional assumptions.
- Results focus on one-step-ahead forecasts for a single equity index.
- These issues are discussed explicitly in the report.

## Extensions

Natural extensions include Student-t HMM emissions, Markov-switching GARCH models, longer-horizon forecasting, and multi-asset validation.
