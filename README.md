# Pearls AQI Predictor — Karachi

AI-powered 3-day Air Quality Index (AQI) forecasting for Karachi, Pakistan, built on a 100% serverless stack.

## Live Demo
[Add your Streamlit Cloud link here once deployed]

## Overview
This project predicts the **average AQI for the next 3 calendar days** in Karachi using historical weather and pollutant data, engineered features, and a Ridge Regression model, served through an interactive dashboard with SHAP-based explainability.

## Tech Stack
- **Language**: Python
- **ML**: scikit-learn (Ridge Regression, Random Forest, HistGradientBoosting), LightGBM
- **Feature Store & Model Registry**: Hopsworks
- **Data Sources**: Open-Meteo (Air Quality API, Historical Weather Archive API, Forecast API)
- **Dashboard**: Streamlit
- **Explainability**: SHAP
- **Automation**: GitHub Actions

## Architecture
Open-Meteo APIs (historical + live)
│
▼
Feature Engineering (76 features: lags, rolling averages,
calendar features, current + forecasted weather/pollutants)
│
▼
Hopsworks Feature Store (features + targets)
│
▼
Training Pipeline (Ridge / RF / HGB / LightGBM comparison)
│
▼
Hopsworks Model Registry (versioned models)
│
▼
Live Prediction Pipeline → Streamlit Dashboard
## Model Performance (test set)
| Horizon | Model | RMSE | MAE | R² |
|---|---|---|---|---|
| Day 1 | Ridge | 9.61 | 6.93 | 0.600 |
| Day 2 | Ridge | 12.77 | 9.61 | 0.283 |
| Day 3 | Ridge | 13.41 | 10.07 | 0.204 |

Ridge Regression was selected after comparing four model families (Ridge, Random Forest, HistGradientBoosting, LightGBM); Ridge consistently achieved the best performance across all three horizons, suggesting the underlying AQI relationship is predominantly linear given this feature set.

## Key Design Decisions
- **Target definition**: Calendar-day average AQI (not point-in-time), per project requirements
- **Data source**: Open-Meteo used throughout (historical and live) instead of raw ground-station APIs (e.g., AQICN), avoiding staleness issues from multi-hour ground-station update cycles
- **Future weather**: Training uses actual historical observations as a proxy for forecast data (real historical forecasts don't exist); deployment uses genuine Open-Meteo forecasts in the same feature slots
- **Leakage prevention**: Only past/current observations and legitimately-forecastable weather are used as features; same-timestamp future pollutant values are never used

## Setup (local development)
```bash
python -m venv venv
venv\Scripts\activate  # Windows
pip install -r requirements.txt
```

Create a `.env` file with:
```
HOPSWORKS_API_KEY=your_key_here
```


Run the dashboard:
```bash
streamlit run app.py
```

## Project Structure
aqi-predictor/
├── app.py # Streamlit dashboard
├── predict_utils.py # Live prediction pipeline + SHAP
├── feature_engineering.py # Feature/target engineering
├── train_from_hopsworks.py # Training pipeline (reads from Feature Store)
├── push_to_hopsworks.py # Pushes features/targets to Feature Store
├── push_models_to_registry.py # Pushes trained models to Model Registry
├── models/
│ └── feature_cols.joblib # Feature column order (training/inference consistency)
└── requirements.txt