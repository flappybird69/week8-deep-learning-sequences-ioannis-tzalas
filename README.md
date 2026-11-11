# Week 8 Deep Learning for Sequences — Ioannis Tzalas

Continues [Week 1](https://github.com/flappybird69/week1-financial-analysis-ioannis-tzalas), [Week 2](https://github.com/flappybird69/week2-data-hygiene-ioannis-tzalas), [Week 3](https://github.com/flappybird69/week3-fintech-assignment-ioannis-tzalas), [Week 4](https://github.com/flappybird69/week4-time-series-forecasting-ioannis-tzalas), [Week 5](https://github.com/flappybird69/week5-feature-tournament-ioannis-tzalas), and [Week 6](https://github.com/flappybird69/week6-tree-ensembles-ioannis-tzalas), using Bitcoin (`BTC-USD`).

- [`Tzalas_W8_LSTM.ipynb`](Tzalas_W8_LSTM.ipynb): Executed assignment notebook containing the data contract and sequence generator, the three zero-cost contestants, the scalar by-hand toy step verification, the 10,451-parameter LSTM implementation, the seed gap audit and hyperparameter sensitivity tournament, the sealed holdout showdown, residual diagnostics across volatility regimes, the 8-rule Definition of Done (DoD) eval suite, and the behavioral coda.
- [`Tzalas_W8_Report.pdf`](Tzalas_W8_Report.pdf): The 1,000-word trading desk briefing paper providing executive interpretations, error diagnostics, and deployment recommendations.
- [`Tzalas_W8_Data.csv`](Tzalas_W8_Data.csv): The frozen raw Bitcoin daily closing prices (1,776 observations from 2022-01-01 to 2026-11-11).
- [`requirements.txt`](requirements.txt): Environment dependencies for exact offline reproduction.

## Methodology & Protocol
- **The Data Contract:** 1,776 daily closes of BTC-USD partitioned strictly in time into 70% Train (2022-01-01 to 2025-05-27, 1,243 rows), 15% Validation (2025-05-28 to 2026-02-17, 266 rows), and 15% Sealed Holdout Test (2026-02-18 to 2026-11-11, 267 rows). Zero random shuffling (`shuffle=False`).
- **Data Hygiene:** Feature normalization via `MinMaxScaler(0, 1)` fitted exclusively on training rows, guaranteeing zero forward look-ahead leakage into validation or test splits.
- **Look-back Window ($w$):** $w = 30$ trading days (~1 month) chosen to capture monthly liquidity and short-term crypto momentum cycles without creating high-dimensional sequence tensors.
- **Three Zero-Cost Contestants:** Naive persistence ($\hat{y}_t = y_{t-1}$), Moving Average MA(5), and Linear Regression on the flattened 30-day window.
- **LSTM Architecture:** Input $(w=30, 1) \to$ LSTM(50) $\to$ Dropout(0.2) $\to$ Dense(1), containing exactly **10,451 trainable parameters** ($4 \times [50 \times (1 + 50) + 50] + 51$). Parameter count is mathematically invariant to $w$. Trained with Adam (lr=0.001), batch 32, and EarlyStopping (patience 15 on validation loss).
- **The Seed Gap Audit:** Stochastic initialization variance measured via Seed 101 ($|\text{RMSE}_{42} - \text{RMSE}_{101}| = \$208.22$), establishing the noise floor for hyperparameter tuning.

## Key Empirical Findings & Showdown Verdict

| Model Architecture | Test MAE ($) | Test RMSE ($) | Test MAPE (%) | Test R² | Out-of-Sample Verdict |
|---|---|---|---|---|---|
| **Naive Persistence (Baseline)** | **$1,013.78** | **$1,428.80** | **1.40%** | **0.9660** | **Winning Benchmark** |
| Linear Regression ($w=30$) | $1,066.16 | $1,473.97 | 1.48% | 0.9638 | Optimal Linear Memory |
| Moving Average MA(5) | $1,580.98 | $2,216.38 | 2.19% | 0.9182 | Severe Phase Lag Tax |
| **LSTM(50 Units, Seed 42)** | $1,290.61 | $1,786.43 | 1.78% | 0.9469 | **Stay Simple (-25.03%)** |

- **Sprint Verdict:** Relative improvement over Naive is **-25.03%**. Under the course's decision bands ($<0\%$: Stay Simple; $0-5\%$: Modest; $>5\%$: Adds Value), the model unequivocally triggers **"STAY SIMPLE"**. The LSTM does **not** earn its complexity on daily Bitcoin price point-forecasts.
- **Error Anatomy:** Residuals exhibit strong lag-1 autocorrelation ($\rho_1 = 0.5897$), indicating that the network merely copies price persistence with an unavoidable phase delay.
- **Regime Breakdown:** The 20 worst error days coincided with extreme market turbulence (averaging 53.38% 7-day annualized volatility), whereas the 20 best error days occurred during quiet consolidation periods (33.82% volatility). LSTMs operate cleanly in calm markets, but break down during liquidation shocks.
- **Definition of Done (DoD):** All 8 automated assertions in the Eval Suite passed (8/8).

## Reproduction
Install dependencies in a Python 3.10+ / 3.14 environment:
```bash
pip install -r requirements.txt
```
Open `Tzalas_W8_LSTM.ipynb` and select **Kernel → Restart & Run All**. All tables, models, and audit assertions regenerate deterministically offline.
