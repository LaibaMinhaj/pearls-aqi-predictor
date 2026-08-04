import requests

LAT, LON = 24.8607, 67.0011

resp = requests.get(
    "https://air-quality-api.open-meteo.com/v1/air-quality",
    params={
        "latitude": LAT,
        "longitude": LON,
        "hourly": "pm2_5,pm10,carbon_monoxide,nitrogen_dioxide,sulphur_dioxide,ozone,us_aqi",
        "start_date": "2024-01-01",
        "end_date": "2024-01-07",
    },
)
print(resp.status_code)
print(resp.json())