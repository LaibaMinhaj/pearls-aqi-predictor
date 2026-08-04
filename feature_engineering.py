import pandas as pd
import numpy as np

df = pd.read_csv("karachi_historical.csv", parse_dates=["time"])
df = df.sort_values("time").reset_index(drop=True)

# ---- Calendar features (known in advance, no leakage risk) ----
df["hour"] = df["time"].dt.hour
df["day_of_week"] = df["time"].dt.dayofweek       # 0=Monday
df["month"] = df["time"].dt.month
df["is_weekend"] = df["day_of_week"].isin([5, 6]).astype(int)

# ---- Lag features: only PAST values, safe to use for any future prediction ----
lag_hours = [1, 3, 6, 12, 24, 48, 72]
lag_columns = ["us_aqi", "pm2_5", "pm10", "temperature_2m", "relative_humidity_2m", "wind_speed_10m"]

for col in lag_columns:
    for lag in lag_hours:
        df[f"{col}_lag{lag}h"] = df[col].shift(lag)

# ---- Rolling averages: also only past data (shift ensures no leakage) ----
for col in ["us_aqi", "pm2_5"]:
    df[f"{col}_rollmean_24h"] = df[col].shift(1).rolling(window=24).mean()
    df[f"{col}_rollmean_72h"] = df[col].shift(1).rolling(window=72).mean()

# ---- AQI change rate (how fast it's rising/falling recently) ----
df["aqi_change_rate_24h"] = df["us_aqi_lag1h"] - df["us_aqi_lag24h"]

# ---- Future weather proxy: observed future weather standing in for forecast ----
# NOTE: uses actual historical observations at t+Nh (real historical forecasts
# don't exist for 2023-2026). At deployment, these columns get populated with
# real Open-Meteo forecast values instead — documented limitation.
future_weather_cols = ["temperature_2m", "relative_humidity_2m", "wind_speed_10m", "precipitation"]
future_horizons = [24, 48, 72]

for col in future_weather_cols:
    for h in future_horizons:
        df[f"{col}_future{h}h"] = df[col].shift(-h)

# ---- Targets: AVERAGE AQI per calendar day ----
# target_day1 = tomorrow's full-day average AQI
# target_day2 = the day after tomorrow's full-day average AQI
# target_day3 = three days from now's full-day average AQI
df["date"] = df["time"].dt.normalize()

daily = (
    df.groupby("date", as_index=False)["us_aqi"]
      .mean()
      .rename(columns={"us_aqi": "daily_avg_aqi"})
)

daily["target_day1"] = daily["daily_avg_aqi"].shift(-1)
daily["target_day2"] = daily["daily_avg_aqi"].shift(-2)
daily["target_day3"] = daily["daily_avg_aqi"].shift(-3)

df = df.merge(
    daily[["date", "target_day1", "target_day2", "target_day3"]],
    on="date",
    how="left"
)
df = df.drop(columns=["date"])

# ---- Drop rows with NaN from shifting (start and end of the dataset) ----
before = len(df)
df = df.dropna().reset_index(drop=True)
after = len(df)
print(f"Dropped {before - after} rows due to lag/target/future-weather shifting ({after} rows remain)")

df.to_csv("karachi_features.csv", index=False)
print("Saved karachi_features.csv")
print("\nColumns:", list(df.columns))