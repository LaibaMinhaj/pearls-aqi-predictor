import requests

LAT, LON = 24.8607, 67.0011

test_ranges = [
    ("2023-01-01", "2023-01-02"),
    ("2022-01-01", "2022-01-02"),
    ("2021-01-01", "2021-01-02"),
    ("2020-01-01", "2020-01-02"),
]

for start, end in test_ranges:
    resp = requests.get(
        "https://air-quality-api.open-meteo.com/v1/air-quality",
        params={
            "latitude": LAT,
            "longitude": LON,
            "hourly": "pm2_5,us_aqi",
            "start_date": start,
            "end_date": end,
        },
    )
    data = resp.json()
    pm25_values = data.get("hourly", {}).get("pm2_5", [])
    non_null = [v for v in pm25_values if v is not None]
    print(f"{start} to {end}: status={resp.status_code}, records={len(pm25_values)}, non-null={len(non_null)}")