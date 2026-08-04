import os
import hopsworks
from joblib import load
from dotenv import load_dotenv

load_dotenv()

project = hopsworks.login(api_key_value=os.getenv("HOPSWORKS_API_KEY"))
mr = project.get_model_registry()

targets = ["target_day1", "target_day2", "target_day3"]
algorithms = ["ridge", "rf"]

for target in targets:
    for algo in algorithms:
        local_path = f"models/{algo}_{target}_final.joblib"

        model = mr.python.create_model(
            name=f"{algo}_{target}",
            description=f"{algo.upper()} model predicting {target} (calendar-day average AQI) for Karachi",
        )
        model.save(local_path)
        print(f"Registered {algo}_{target} in Hopsworks Model Registry")