import requests
import pandas as pd
import time
from datetime import datetime, timedelta

LAT, LON = 24.8607, 67.0011
START_DATE = datetime(2023, 1, 1)
END_DATE = datetime.now()
CHUNK_DAYS = 90  # pull ~3 months at a time to keep each request small

AIR_QUALITY_VARS = "pm2_5,pm10,carbon_monoxide,nitrogen_dioxide,sulphur_dioxide,ozone,us_aqi"
WEATHER_VARS = "temperature_2m,relative_humidity_2m,surface_pressure,wind_speed_10m,wind_direction_10m,precipitation"


def fetch_chunk(url, variables, start, end, max_retries=3):
    """Fetch one date-range chunk, retrying on failure."""
    params = {
        "latitude": LAT,
        "longitude": LON,
        "start_date": start.strftime("%Y-%m-%d"),
        "end_date": end.strftime("%Y-%m-%d"),
        "hourly": variables,
        "timezone": "UTC",
    }
    for attempt in range(1, max_retries + 1):
        try:
            resp = requests.get(url, params=params, timeout=30)
            if resp.status_code == 200:
                return resp.json()
            else:
                print(f"  Attempt {attempt}: status {resp.status_code}, retrying...")
        except requests.exceptions.RequestException as e:
            print(f"  Attempt {attempt}: request failed ({e}), retrying...")
        time.sleep(3)
    print(f"  FAILED after {max_retries} attempts for {start.date()} to {end.date()}")
    return None


def fetch_all_chunks(url, variables, label):
    """Loop over the full date range in non-overlapping chunks, collecting results."""
    all_rows = []
    current_start = START_DATE

    while current_start <= END_DATE:
        current_end = min(current_start + timedelta(days=CHUNK_DAYS), END_DATE)
        print(f"Fetching {label}: {current_start.date()} to {current_end.date()}")

        data = fetch_chunk(url, variables, current_start, current_end)

        if data and "hourly" in data:
            hourly = data["hourly"]
            chunk_df = pd.DataFrame(hourly)
            all_rows.append(chunk_df)

            # Save progress after every chunk, so a failure later doesn't lose everything
            partial = pd.concat(all_rows, ignore_index=True)
            partial.to_csv(f"_partial_{label}.csv", index=False)
        else:
            print(f"  Skipping this chunk — no data returned")

        current_start = current_end + timedelta(days=1)  # no overlap with next chunk
        time.sleep(1)

    if all_rows:
        return pd.concat(all_rows, ignore_index=True)
    return pd.DataFrame()


def main():
    print("=== Fetching air quality data ===")
    air_df = fetch_all_chunks(
        "https://air-quality-api.open-meteo.com/v1/air-quality", AIR_QUALITY_VARS, "airquality"
    )

    print("\n=== Fetching weather data ===")
    weather_df = fetch_all_chunks(
        "https://archive-api.open-meteo.com/v1/archive", WEATHER_VARS, "weather"
    )

    air_df = air_df.drop_duplicates(subset="time")
    weather_df = weather_df.drop_duplicates(subset="time")

    print(f"\nAir quality rows: {len(air_df)}")
    print(f"Weather rows: {len(weather_df)}")

    merged = pd.merge(air_df, weather_df, on="time", how="inner")
    merged = merged.sort_values("time").reset_index(drop=True)

    print(f"Merged rows: {len(merged)}")
    print(f"Missing values per column:\n{merged.isnull().sum()}")

    merged.to_csv("karachi_historical.csv", index=False)
    print("\nSaved to karachi_historical.csv")


if __name__ == "__main__":
    main()