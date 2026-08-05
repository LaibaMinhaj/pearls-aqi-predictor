import os
import requests
import pandas as pd
import numpy as np
import hopsworks
from dotenv import load_dotenv

load_dotenv()

LAT, LON = 24.8607, 67.0011
WINDOW_DAYS = 10  # enough buffer for 72h lag + 72h future on both ends


def fetch_window():
    aq_resp = requests.get(
        "https://air-quality-api.open-meteo.com/v1/air-quality",
        params={
            "latitude": LAT, "longitude": LON,
            "hourly": "pm2_5,pm10,carbon_monoxide,nitrogen_dioxide,sulphur_dioxide,ozone,us_aqi",
            "past_days": WINDOW_DAYS,
            "timezone": "UTC",
        },
    )
    aq_df = pd.DataFrame(aq_resp.json()["hourly"])
    aq_df["time"] = pd.to_datetime(aq_df["time"])

    weather_resp = requests.get(
        "https://archive-api.open-meteo.com/v1/archive",
        params={
            "latitude": LAT, "longitude": LON,
            "start_date": (pd.Timestamp.utcnow() - pd.Timedelta(days=WINDOW_DAYS)).strftime("%Y-%m-%d"),
            "end_date": pd.Timestamp.utcnow().strftime("%Y-%m-%d"),
            "hourly": "temperature_2m,relative_humidity_2m,surface_pressure,wind_speed_10m,wind_direction_10m,precipitation",
            "timezone": "UTC",
        },
    )
    weather_df = pd.DataFrame(weather_resp.json()["hourly"])
    weather_df["time"] = pd.to_datetime(weather_df["time"])

    df = weather_df.merge(aq_df, on="time", how="inner")
    return df.sort_values("time").reset_index(drop=True)


def build_features(df):
    df["hour"] = df["time"].dt.hour
    df["day_of_week"] = df["time"].dt.dayofweek
    df["month"] = df["time"].dt.month
    df["is_weekend"] = df["day_of_week"].isin([5, 6]).astype(int)

    lag_hours = [1, 3, 6, 12, 24, 48, 72]
    lag_columns = ["us_aqi", "pm2_5", "pm10", "temperature_2m", "relative_humidity_2m", "wind_speed_10m"]
    for col in lag_columns:
        for lag in lag_hours:
            df[f"{col}_lag{lag}h"] = df[col].shift(lag)

    for col in ["us_aqi", "pm2_5"]:
        df[f"{col}_rollmean_24h"] = df[col].shift(1).rolling(window=24).mean()
        df[f"{col}_rollmean_72h"] = df[col].shift(1).rolling(window=72).mean()

    df["aqi_change_rate_24h"] = df["us_aqi_lag1h"] - df["us_aqi_lag24h"]

    future_weather_cols = ["temperature_2m", "relative_humidity_2m", "wind_speed_10m", "precipitation"]
    for col in future_weather_cols:
        for h in [24, 48, 72]:
            df[f"{col}_future{h}h"] = df[col].shift(-h)

    df["date"] = df["time"].dt.normalize()
    daily = (
        df.groupby("date", as_index=False)["us_aqi"]
          .mean()
          .rename(columns={"us_aqi": "daily_avg_aqi"})
    )
    daily["target_day1"] = daily["daily_avg_aqi"].shift(-1)
    daily["target_day2"] = daily["daily_avg_aqi"].shift(-2)
    daily["target_day3"] = daily["daily_avg_aqi"].shift(-3)
    df = df.merge(daily[["date", "target_day1", "target_day2", "target_day3"]], on="date", how="left")
    df = df.drop(columns=["date"])

    return df.dropna().reset_index(drop=True)


def main():
    print("Fetching recent data window...")
    raw_df = fetch_window()
    print(f"Fetched {len(raw_df)} raw hourly rows")

    features_df = build_features(raw_df)
    print(f"{len(features_df)} rows have complete features and are ready to push")

    if len(features_df) == 0:
        print("No complete rows yet — nothing to push this run.")
        return

    project = hopsworks.login(api_key_value=os.getenv("HOPSWORKS_API_KEY"))
    fs = project.get_feature_store()

    target_cols = ["target_day1", "target_day2", "target_day3"]
    feature_only_cols = [c for c in features_df.columns if c not in target_cols]

    aqi_fg = fs.get_feature_group(name="karachi_aqi_features", version=1)
    aqi_fg.insert(features_df[feature_only_cols], write_options={"wait_for_job": True})
    print(f"Upserted {len(features_df)} rows into karachi_aqi_features")

    targets_fg = fs.get_feature_group(name="karachi_aqi_targets", version=1)
    targets_fg.insert(features_df[["time"] + target_cols], write_options={"wait_for_job": True})
    print(f"Upserted {len(features_df)} rows into karachi_aqi_targets")


if __name__ == "__main__":
    main()