# Project 2: Multivariate Energy Load Forecasting with Weather Covariates

## Overview
An investigation into whether adding weather data as a covariate improves 
energy load forecasting accuracy. Two LSTM models are compared — one using 
load data only, and one incorporating temperature as an additional input feature.

## Results

| Model | MAPE | RMSE |
|-------|------|------|
| Univariate LSTM (load only) | 3.48% | 676.30 MW |
| **Multivariate LSTM (load + temperature)** | **3.32%** | **623.16 MW** |

**Key finding:** Adding temperature as a covariate improved forecast accuracy 
by 4.6%, reflecting the non-linear seasonal relationship between weather and 
energy demand. The modest improvement is consistent with the weak correlation 
(-0.246) observed between load and temperature in this dataset.

## Dataset
- **Energy:** [PJM Hourly Energy Consumption](https://www.kaggle.com/datasets/robikscube/hourly-energy-consumption) — AEP region (Kaggle)
- **Weather:** Historical daily temperature from [Open-Meteo](https://open-meteo.com) — Columbus, OH (39.96°N, 82.99°W)
- **Period:** 2016–2018
- **Granularity:** Daily average

## Methodology
1. Fetched historical temperature data via Open-Meteo API (no key required)
2. Merged energy and weather datasets on date index
3. Normalized each feature independently using training set statistics
4. Created 30-day sliding window sequences
5. Trained two identical 2-layer LSTMs — one with 1 input feature, one with 2
6. Evaluated on last 90 days held out as test set

## Metrics
- **MAPE** (Mean Absolute Percentage Error) — primary metric
- **RMSE** (Root Mean Squared Error) — secondary metric

## Project Structure
energy-load-forecasting-multivariate/

├── data/               # Raw CSV files (not tracked in Git)

├── notebooks/

│   └── project2_multivariate_lstm.ipynb

└── README.md
## Requirements
pandas

numpy

matplotlib

torch

requests

jupyter
## How to Run
1. Clone the repo
2. Install dependencies: `pip install -r requirements.txt`
3. Download AEP_hourly.csv from Kaggle and place in `data/`
4. Open and run `notebooks/project2_multivariate_lstm.ipynb`

## Context
Part of a series of energy forecasting projects exploring classical vs. 
deep learning approaches and the impact of weather covariates on forecast accuracy.
