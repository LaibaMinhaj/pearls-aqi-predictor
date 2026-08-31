# Pearls AQI Predictor — Karachi

An end-to-end machine learning system for 3-day Air Quality Index (AQI) forecasting in Karachi, Pakistan, using a cloud-based feature store and model registry.

## Live Demo
[https://pearls-aqi-predictor-khi.streamlit.app/](https://pearls-aqi-predictor-khi.streamlit.app/)

## Overview
This project predicts the **average AQI for the next 3 calendar days** in Karachi using historical weather and pollutant data, 92 engineered features, and an ensemble model (Ridge + SVR + CatBoost), served through an interactive dashboard with SHAP-based explainability and automated hourly/daily retraining.

## Tech Stack
- **Language**: Python
- **ML**: scikit-learn (Ridge Regression, Random Forest, HistGradientBoosting, SVR), LightGBM, XGBoost, CatBoost
- **Feature Store & Model Registry**: Hopsworks
- **Data Source**: Open-Meteo (Air Quality API, Historical Weather Archive API, Forecast API)
- **Dashboard**: Streamlit
- **Explainability**: SHAP
- **Automation**: GitHub Actions (hourly feature updates, daily retraining)

## Architecture

Open-Meteo APIs (historical + live)
│
▼
Feature Engineering (92 features: lags, rolling mean/std/min/max,
cyclical calendar encoding, current + forecasted weather/pollutants)
│
▼
Hopsworks Feature Store (features + targets)
│
▼
Training Pipeline (Ridge / RF / HGB / LightGBM / XGBoost / CatBoost / SVR comparison)
│
▼
Ensemble Model (Ridge + SVR + CatBoost, tuned per horizon)
│
▼
Hopsworks Model Registry (versioned models, with attached metrics)
│
▼
Live Prediction Pipeline → Streamlit Dashboard
│
GitHub Actions: hourly feature updates + daily retraining (automated)


## Model Performance (held-out test set)
| Horizon | Model | RMSE | MAE | R² |
|---|---|---|---|---|
| Day 1 | Ensemble (Ridge+SVR+CatBoost) | 9.17 | 6.46 | 0.636 |
| Day 2 | Ensemble (Ridge+SVR+CatBoost) | 12.44 | 9.16 | 0.320 |
| Day 3 | Ensemble (Ridge+SVR+CatBoost) | 13.11 | 9.65 | 0.239 |

The ensemble was selected after comparing **7 model families** (Ridge, Random Forest, HistGradientBoosting, LightGBM, XGBoost, CatBoost, SVR) across multiple feature-set experiments, hyperparameter tuning, and ensembling strategies. It substantially outperforms a naive persistence baseline (predict tomorrow = today) at every horizon:

| Horizon | Persistence RMSE | Persistence R² | Ensemble RMSE | Improvement |
|---|---|---|---|---|
| Day 1 | 14.30 | 0.11 | 9.17 | -36% |
| Day 2 | 16.60 | -0.21 | 12.44 | -25% |
| Day 3 | 18.10 | -0.45 | 13.11 | -28% |

## Key Design Decisions
- **Target definition**: Calendar-day average AQI (not point-in-time), per project requirements
- **Data source**: Open-Meteo used throughout (historical and live) instead of raw ground-station APIs (e.g., AQICN), avoiding staleness issues from multi-hour ground-station update cycles. This is a deliberate reliability/consistency tradeoff — Open-Meteo's modeled (CAMS-based) estimates can diverge from real-time sensor readings shown by consumer apps (e.g., AccuWeather), particularly during rapidly changing conditions.
- **Future weather**: Training uses actual historical observations as a proxy for forecast data (real historical forecasts don't exist); deployment uses genuine Open-Meteo forecasts in the same feature slots
- **Leakage prevention**: Only past/current observations and legitimately-forecastable weather are used as features; same-timestamp future pollutant values are never used
- **Ensembling**: Averaging Ridge, SVR, and CatBoost outperformed any single tuned model at Day 1/Day 3, and a default-parameter ensemble outperformed a fully-tuned one at Day 2 — individually tuning each model reduced the error diversity that makes ensembling effective
- **Rejected approaches** (tested, documented, not adopted): recency-weighted training, recursive horizon-chaining, and training only on recent data all underperformed using the full 3.5-year history — Karachi's AQI has seasonal patterns that shorter/weighted windows fail to capture
- **Error analysis**: The largest test-set errors clustered around a single hot/dry/low-humidity event (mid-March), consistent with dust-suspension conditions that lag-based features cannot anticipate before onset — a documented limitation rather than a bug

## Automation
- **Hourly** (GitHub Actions): fetches the latest data, computes features for any newly-complete hours, and incrementally upserts to the Hopsworks Feature Store (only new rows, not full reprocessing)
- **Daily** (GitHub Actions): retrains the ensemble on the full updated dataset and registers new model versions with attached performance metrics
- Both workflows include automatic retry logic for transient Hopsworks connectivity issues

## Setup (local development)
```bash
python -m venv venv
venv\Scripts\activate  # Windows
pip install -r requirements.txt
```

Create a `.env` file with:

HOPSWORKS_API_KEY=your_key_here


Run the dashboard:
```bash
streamlit run app.py
```

## Project Structure

aqi-predictor/
├── app.py # Streamlit dashboard
├── predict_utils.py # Live prediction pipeline + SHAP
├── predict.py # CLI test wrapper
├── feature_engineering.py # Feature/target engineering (initial backfill)
├── backfill_data.py # Historical data collection
├── train_final_models.py # Production training: ensemble, tuning, registry
├── push_to_hopsworks.py # Pushes features/targets to Feature Store
├── incremental_pipeline.py # Hourly automated feature updates
├── models/
│ └── feature_cols.joblib # Feature column order (training/inference consistency)
└── requirements.txt
├── experiments/                # exploratory scripts: model comparisons, tuning, error analysis
