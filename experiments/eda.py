import pandas as pd
import matplotlib.pyplot as plt

df = pd.read_csv("karachi_historical.csv", parse_dates=["time"])
df = df.set_index("time")

# 1. AQI over the full time range — spot trends and seasonality
plt.figure(figsize=(14, 5))
plt.plot(df.index, df["us_aqi"], linewidth=0.5)
plt.title("Karachi US AQI — Jan 2023 to Present")
plt.xlabel("Date")
plt.ylabel("US AQI")
plt.tight_layout()
plt.savefig("aqi_over_time.png")
print("Saved aqi_over_time.png")

# 2. Monthly average AQI — check for seasonal patterns
monthly_avg = df["us_aqi"].resample("ME").mean()
plt.figure(figsize=(12, 5))
monthly_avg.plot(kind="bar")
plt.title("Average Monthly AQI")
plt.xlabel("Month")
plt.ylabel("Average US AQI")
plt.tight_layout()
plt.savefig("monthly_aqi.png")
print("Saved monthly_aqi.png")

# 3. Hour-of-day pattern — does AQI spike at certain hours?
hourly_avg = df.groupby(df.index.hour)["us_aqi"].mean()
plt.figure(figsize=(10, 5))
hourly_avg.plot(kind="line", marker="o")
plt.title("Average AQI by Hour of Day")
plt.xlabel("Hour")
plt.ylabel("Average US AQI")
plt.tight_layout()
plt.savefig("hourly_pattern.png")
print("Saved hourly_pattern.png")

# 4. Correlation between pollutants/weather and AQI
correlations = df[["us_aqi", "pm2_5", "pm10", "carbon_monoxide", "nitrogen_dioxide",
                    "sulphur_dioxide", "ozone", "temperature_2m", "relative_humidity_2m",
                    "wind_speed_10m"]].corr()["us_aqi"].sort_values(ascending=False)
print("\nCorrelation with AQI:")
print(correlations)

# 5. Basic stats summary
print("\nAQI distribution:")
print(df["us_aqi"].describe())