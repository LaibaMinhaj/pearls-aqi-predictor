import requests

LAT, LON = 24.8607, 67.0011

response = requests.get(
    "https://api.open-meteo.com/v1/forecast",
    params={
        "latitude": LAT,
        "longitude": LON,
        "hourly": "temperature_2m,relative_humidity_2m,surface_pressure,wind_speed_10m,wind_direction_10m,precipitation",
        "forecast_days": 3,
        "timezone": "UTC",
    },
)

print("Status:", response.status_code)
data = response.json()
print("Number of timestamps:", len(data.get("hourly", {}).get("time", [])))
print("First timestamp:", data.get("hourly", {}).get("time", [])[0])
print("Last timestamp:", data.get("hourly", {}).get("time", [])[-1])