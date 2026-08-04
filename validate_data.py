import pandas as pd

df = pd.read_csv("karachi_historical.csv", parse_dates=["time"])

print("Date range:", df["time"].min(), "to", df["time"].max())
print("Total rows:", len(df))

# Check for duplicate timestamps
duplicates = df["time"].duplicated().sum()
print("Duplicate timestamps:", duplicates)

# Check that timestamps are genuinely hourly with no gaps
expected_range = pd.date_range(start=df["time"].min(), end=df["time"].max(), freq="h")
missing_timestamps = expected_range.difference(df["time"])
print("Missing hourly timestamps:", len(missing_timestamps))
if len(missing_timestamps) > 0:
    print("First few missing:", missing_timestamps[:5].tolist())

# Quick sanity check on value ranges
print("\nValue ranges:")
print(df[["pm2_5", "pm10", "us_aqi", "temperature_2m", "relative_humidity_2m"]].describe())