import requests

LAT, LON = 24.8607, 67.0011

response = requests.get(
    "https://archive-api.open-meteo.com/v1/archive",
    params={
        "latitude": LAT,
        "longitude": LON,
        "start_date": "2023-01-01",
        "end_date": "2023-01-07",
        "hourly": (
            "temperature_2m,"
            "relative_humidity_2m,"
            "surface_pressure,"
            "wind_speed_10m,"
            "wind_direction_10m,"
            "precipitation"
        ),
        "timezone": "UTC",
    },
)

print("Status:", response.status_code)

data = response.json()

print("Coordinates:", data.get("latitude"), data.get("longitude"))
print("Timezone:", data.get("timezone"))
print("Units:", data.get("hourly_units"))
print("Number of timestamps:", len(data.get("hourly", {}).get("time", [])))

print("\nFirst 3 timestamps:")
print(data.get("hourly", {}).get("time", [])[:3])

print("\nFirst 3 values:")
hourly = data.get("hourly", {})
for key in hourly:
    if key != "time":
        print(key, hourly[key][:3])