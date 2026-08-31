import os
import requests
from dotenv import load_dotenv

# Load API key from .env
load_dotenv()

API_KEY = os.getenv("OPENWEATHER_API_KEY")

print("Key loaded:", API_KEY is not None)
print("Key length:", len(API_KEY) if API_KEY else 0)

# Karachi coordinates
LAT = 24.8607
LON = 67.0011

# OpenWeather current air pollution endpoint
url = "http://api.openweathermap.org/data/2.5/air_pollution"

params = {
    "lat": LAT,
    "lon": LON,
    "appid": API_KEY,
}

response = requests.get(url, params=params)

print("Status code:", response.status_code)
print("Response:", response.text)