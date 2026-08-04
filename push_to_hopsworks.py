import os
import pandas as pd
import hopsworks
from dotenv import load_dotenv

load_dotenv()

project = hopsworks.login(api_key_value=os.getenv("HOPSWORKS_API_KEY"))
fs = project.get_feature_store()

df = pd.read_csv("karachi_features.csv", parse_dates=["time"])

target_cols = ["target_day1", "target_day2", "target_day3"]
feature_cols = [c for c in df.columns if c not in target_cols]

features_df = df[feature_cols].copy()

aqi_fg = fs.get_or_create_feature_group(
    name="karachi_aqi_features",
    version=1,
    primary_key=["time"],
    event_time="time",
    time_travel_format="HUDI",
    description="Karachi AQI features: calendar, lags, rolling means, current weather/pollutants, future weather proxy",
)
aqi_fg.insert(features_df)
print(f"Inserted {len(features_df)} rows into karachi_aqi_features")

# ---- Feature group 2: daily-average targets ----
targets_df = df[["time"] + target_cols].copy()

targets_fg = fs.get_or_create_feature_group(
    name="karachi_aqi_targets",
    version=1,
    primary_key=["time"],
    event_time="time",
    time_travel_format="HUDI",
    description="Karachi AQI daily-average prediction targets: Day1/Day2/Day3",
)
targets_fg.insert(targets_df)
print(f"Inserted {len(targets_df)} rows into karachi_aqi_targets")