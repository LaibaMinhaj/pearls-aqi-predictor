import os
import requests
import pandas as pd
import numpy as np
from datetime import datetime, timezone
from joblib import load
import hopsworks
from dotenv import load_dotenv

load_dotenv()

LAT, LON = 24.8607, 67.0011

_model_cache = {}


def aqi_category(aqi):
    if aqi <= 50:
        return "Good"
    elif aqi <= 100:
        return "Moderate"
    elif aqi <= 150:
        return "Unhealthy for Sensitive Groups"
    elif aqi <= 200:
        return "Unhealthy"
    elif aqi <= 300:
        return "Very Unhealthy"
    else:
        return "Hazardous"


_project = None

def get_project():
    global _project
    if _project is None:
        _project = hopsworks.login(api_key_value=os.getenv("HOPSWORKS_API_KEY"))
    return _project


def get_model(target_name):
    """Download (once, then cache) a model from the Hopsworks Model Registry."""
    if target_name in _model_cache:
        return _model_cache[target_name]

    project = get_project()
    mr = project.get_model_registry()

    model_meta = mr.get_model(name=f"ridge_{target_name}")  # no version = latest available
    model_dir = model_meta.download()

    model_path = os.path.join(model_dir, f"ridge_{target_name}_final.joblib")
    model = load(model_path)

    _model_cache[target_name] = model
    return model


def get_feature_cols():
    """Feature column order — pulled from the local file saved at training time."""
    return load("models/feature_cols.joblib")


def fetch_live_data():
    """Fetch recent + forecast pollutant and weather data from Open-Meteo."""
    aq_resp = requests.get(
        "https://air-quality-api.open-meteo.com/v1/air-quality",
        params={
            "latitude": LAT, "longitude": LON,
            "hourly": "pm2_5,pm10,carbon_monoxide,nitrogen_dioxide,sulphur_dioxide,ozone,us_aqi",
            "past_days": 5,
            "timezone": "UTC",
        },
    )
    aq_df = pd.DataFrame(aq_resp.json()["hourly"])
    aq_df["time"] = pd.to_datetime(aq_df["time"])

    weather_resp = requests.get(
        "https://api.open-meteo.com/v1/forecast",
        params={
            "latitude": LAT, "longitude": LON,
            "hourly": "temperature_2m,relative_humidity_2m,surface_pressure,wind_speed_10m,wind_direction_10m,precipitation",
            "past_days": 5,
            "forecast_days": 5,
            "timezone": "UTC",
        },
    )
    weather_df = pd.DataFrame(weather_resp.json()["hourly"])
    weather_df["time"] = pd.to_datetime(weather_df["time"])

    df = weather_df.merge(aq_df, on="time", how="left")
    return df.sort_values("time").reset_index(drop=True)


def build_feature_row(df):
    """Build the exact feature row used at training time, for the most recent valid 'now'."""
    current_hour = pd.Timestamp(datetime.now(timezone.utc)).floor("h").tz_localize(None)
    valid_times = df.loc[df["pm2_5"].notna(), "time"]
    now_time = valid_times[valid_times <= current_hour].max()
    now_idx = df.index[df["time"] == now_time][0]

    row = {}
    row["hour"] = now_time.hour
    row["day_of_week"] = now_time.dayofweek
    row["month"] = now_time.month
    row["is_weekend"] = int(now_time.dayofweek in [5, 6])

    lag_hours = [1, 3, 6, 12, 24, 48, 72]
    lag_columns = ["us_aqi", "pm2_5", "pm10", "temperature_2m", "relative_humidity_2m", "wind_speed_10m"]
    for col in lag_columns:
        for lag in lag_hours:
            idx = now_idx - lag
            row[f"{col}_lag{lag}h"] = df.loc[idx, col] if idx >= 0 else np.nan

    for col in ["us_aqi", "pm2_5"]:
        row[f"{col}_rollmean_24h"] = df.loc[now_idx - 24:now_idx - 1, col].mean()
        row[f"{col}_rollmean_72h"] = df.loc[now_idx - 72:now_idx - 1, col].mean()

    row["aqi_change_rate_24h"] = row["us_aqi_lag1h"] - row["us_aqi_lag24h"]

    for col in ["temperature_2m", "relative_humidity_2m", "surface_pressure", "wind_speed_10m", "wind_direction_10m", "precipitation"]:
        row[col] = df.loc[now_idx, col]
    for col in ["pm2_5", "pm10", "carbon_monoxide", "nitrogen_dioxide", "sulphur_dioxide", "ozone", "us_aqi"]:
        row[col] = df.loc[now_idx, col]

    future_weather_cols = ["temperature_2m", "relative_humidity_2m", "wind_speed_10m", "precipitation"]
    for col in future_weather_cols:
        for h in [24, 48, 72]:
            idx = now_idx + h
            row[f"{col}_future{h}h"] = df.loc[idx, col] if idx < len(df) else np.nan

    return row, now_time


# Test-set RMSE per horizon (from Experiment C — used to show a confidence band, not re-derived live)
MODEL_RMSE = {
    "target_day1": 9.61,
    "target_day2": 12.77,
    "target_day3": 13.41,
}


def predict_all():
    """Fetch live data, build features, and return day1/day2/day3 average-AQI predictions."""
    df = fetch_live_data()
    row, now_time = build_feature_row(df)

    feature_cols = get_feature_cols()
    X = pd.DataFrame([row])[feature_cols]

    predictions = {}
    for target_name in ["target_day1", "target_day2", "target_day3"]:
        model = get_model(target_name)
        pred = round(float(model.predict(X)[0]), 1)
        rmse = MODEL_RMSE[target_name]
        predictions[target_name] = {
            "aqi": pred,
            "category": aqi_category(pred),
            "rmse": rmse,
            "range_low": round(pred - rmse, 1),
            "range_high": round(pred + rmse, 1),
        }

    current_aqi = round(float(row["us_aqi"]), 1)
    return {
        "now_time": now_time,
        "current_aqi": current_aqi,
        "current_category": aqi_category(current_aqi),
        "predictions": predictions,
        "feature_row": X,  # needed for SHAP explanations
    }
import shap

_background_cache = {}


def get_background_data():
    """Small reference sample from training data, used as SHAP's baseline."""
    if "sample" in _background_cache:
        return _background_cache["sample"]

    project = get_project()
    fs = project.get_feature_store()
    fg = fs.get_feature_group(name="karachi_aqi_features", version=1)
    df = fg.read()
    df["time"] = pd.to_datetime(df["time"]).dt.tz_localize(None)

    feature_cols = get_feature_cols()
    sample = df[feature_cols].sample(n=200, random_state=42)

    _background_cache["sample"] = sample
    return sample


import re

BASE_LABELS = {
    "us_aqi": "AQI",
    "pm2_5": "PM2.5",
    "pm10": "PM10",
    "carbon_monoxide": "CO",
    "nitrogen_dioxide": "NO2",
    "sulphur_dioxide": "SO2",
    "ozone": "Ozone",
    "temperature_2m": "Temperature",
    "relative_humidity_2m": "Humidity",
    "surface_pressure": "Pressure",
    "wind_speed_10m": "Wind speed",
    "wind_direction_10m": "Wind direction",
    "precipitation": "Precipitation",
    "hour": "Hour of day",
    "day_of_week": "Day of week",
    "month": "Month",
    "is_weekend": "Weekend",
    "aqi_change_rate_24h": "Recent AQI trend",
}


def friendly_name(feature):
    m = re.match(r"^(.*)_lag(\d+)h$", feature)
    if m:
        base, hrs = m.groups()
        return f"{BASE_LABELS.get(base, base.replace('_', ' '))} ({hrs}h ago)"

    m = re.match(r"^(.*)_future(\d+)h$", feature)
    if m:
        base, hrs = m.groups()
        return f"Forecasted {BASE_LABELS.get(base, base.replace('_', ' '))} (+{hrs}h)"

    m = re.match(r"^(.*)_rollmean_(\d+)h$", feature)
    if m:
        base, hrs = m.groups()
        return f"{hrs}h average {BASE_LABELS.get(base, base.replace('_', ' '))}"

    return BASE_LABELS.get(feature, feature.replace("_", " ").title())


def explain_prediction(target_name, X):
    """Return top increasing/decreasing features for one prediction, in plain language."""
    model = get_model(target_name)
    background = get_background_data()

    explainer = shap.LinearExplainer(model, background)
    shap_values = explainer.shap_values(X)[0]

    contributions = list(zip(X.columns, shap_values))

    increasing = sorted([c for c in contributions if c[1] > 0], key=lambda x: x[1], reverse=True)
    decreasing = sorted([c for c in contributions if c[1] < 0], key=lambda x: x[1])

    top_increase = [(friendly_name(f), round(v, 1)) for f, v in increasing[:5]]
    top_decrease = [(friendly_name(f), round(v, 1)) for f, v in decreasing[:5]]

    return {"increase": top_increase, "decrease": top_decrease}