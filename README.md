# Volatility Regime Detection with Machine Learning

## Overview

This project investigates whether **next-day high-volatility regimes** in the S&P 500 ETF (SPY) can be predicted using information available at the end of the current trading day.

Rather than predicting price direction, the goal is to to understand the **risk**:

> *Will the next trading day exhibit unusually high volatility relative to recent market conditions?*

---
## Data

- Asset: **SPY (S&P 500 ETF)**
- Frequency: Daily
- Period: 2015–2024
- Source: Yahoo Finance (`yfinance`)

All features and targets use **strictly past data**.

---

## Feature

The final feature set focuses on volatility level and regime dynamics, including:

- Rolling volatility (5-day, 20-day, 60-day)
- Absolute returns
- Intraday range (high–low scaled by close)
- Log change in trading volume
- **Volatility regime ratio** (short-term vs long-term volatility)

Feature experimentation showed that **performance gains came from better features, not more complex models**.  
Several intuitive features (e.g. explicit volatility persistence) were tested and removed when they failed to improve out-of-sample performance.

---

## Baselines

Two baselines are used for comparison:

### Baseline A — Always Calm
Always predicts “low volatility.”


### Baseline B — Rule-Based Volatility
Predicts high volatility when recent rolling volatility exceeds historical median.

---

## Models

### Logistic Regression (Final Model)

- Standardized features
- Class imbalance handled via `class_weight="balanced"`
- Outputs probabilistic risk estimates


### Gradient Boosting (Exploratory)

- Tested to capture nonlinear interactions
- Did not outperform logistic regression with the final feature set
- Found that the problem was **feature-limited, AND not model-limited**

---

## Evaluation Strategy

Because high-volatility days are relatively rare (23% of time), the **accuracy score is misleading**.
This is because the Baseline test always predicting low volatility will always show a higher accuracy
since their are more calm days.

Primary metrics:
- **Recall (high-volatility class)** — ability to catch risky days
- **Precision (high-volatility class)** — false-alarm rate
- **F1 score (high-volatility class)** — balance of recall and precision
- **ROC-AUC** — ranking quality independent of threshold

A **time-based train/test split** (80% / 20%) is used.

---


## Final Results (Out-of-Sample)

- **ROC-AUC ≈ 0.66**
- **Recall (high-volatility days) ≈ 0.62**
- **F1 score (high-volatility class) ≈ 0.45**
- Substantially outperformed the rest of the baselines

---

## How to Run

This project builds a daily volatility-regime dataset for SPY, trains a logistic regression model, and produces a probability forecast for next-day high volatility.

All commands should be run from the project root directory.

First, create and activate a virtual environment (recommended) and install dependencies:

```bash
python -m venv .venv
source .venv/bin/activate
# .venv\Scripts\activate        

pip install -r requirements.txt
