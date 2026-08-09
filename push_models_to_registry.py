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
        model_name = f"{algo}_{target}"
        local_path = f"models/{algo}_{target}_final.joblib"

        # Delete all existing versions before registering the new one,
        # so storage doesn't accumulate indefinitely with daily retraining
        try:
            existing_models = mr.get_models(name=model_name)
            for old_model in existing_models:
                old_model.delete()
                print(f"Deleted old version of {model_name}: v{old_model.version}")
        except Exception as e:
            print(f"No existing versions to clean up for {model_name} (or error: {e})")

        model = mr.python.create_model(
            name=model_name,
            description=f"{algo.upper()} model predicting {target} (calendar-day average AQI) for Karachi",
        )
        model.save(local_path)
        print(f"Registered {model_name} in Hopsworks Model Registry")