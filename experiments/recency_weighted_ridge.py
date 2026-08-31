import os
import pandas as pd
import numpy as np
import hopsworks
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from joblib import dump
from dotenv import load_dotenv

load_dotenv()
project = hopsworks.login(api_key_value=os.getenv("HOPSWORKS_API_KEY"))
fs = project.get_feature_store()

df = fs.get_feature_group(name="karachi_aqi_features", version=1).read()
targets_df = fs.get_feature_group(name="karachi_aqi_targets", version=1).read()
df["time"] = pd.to_datetime(df["time"]).dt.tz_localize(None)
targets_df["time"] = pd.to_datetime(targets_df["time"]).dt.tz_localize(None)
df = df.merge(targets_df, on="time", how="inner").sort_values("time").reset_index(drop=True)

exclude_cols = ["time", "target_day1", "target_day2", "target_day3"]
feature_cols = [c for c in df.columns if c not in exclude_cols]

n = len(df)
train_end = int(n * 0.70)
val_end = int(n * 0.85)

# Best alphas found from the widened search — update these once you rerun tuning
best_alphas = {"target_day1": 10000.0, "target_day2": 10000.0, "target_day3": 1000.0}

# Half-life in days: weight halves every HALF_LIFE_DAYS going back in time.
# Recent data gets weight close to 1.0; old data decays toward 0.
HALF_LIFE_DAYS = 180  # ~6 months — tune this if needed; shorter = more aggressive recency bias

def compute_recency_weights(times, reference_time):
    age_days = (reference_time - times).dt.total_seconds() / 86400
    decay_rate = np.log(2) / HALF_LIFE_DAYS
    return np.exp(-decay_rate * age_days)

for target in ["target_day1", "target_day2", "target_day3"]:
    X_train = df[feature_cols].iloc[:train_end]
    y_train = df[target].iloc[:train_end]
    X_val = df[feature_cols].iloc[train_end:val_end]
    y_val = df[target].iloc[train_end:val_end]
    times_train = df["time"].iloc[:train_end]

    weights = compute_recency_weights(times_train, times_train.max())

    model = Ridge(alpha=best_alphas[target])
    model.fit(X_train, y_train, sample_weight=weights)
    val_preds = model.predict(X_val)

    print(f"\n{target} (alpha={best_alphas[target]}, recency-weighted)")
    print("RMSE: {:.2f}, MAE: {:.2f}, R²: {:.4f}".format(
        np.sqrt(mean_squared_error(y_val, val_preds)),
        mean_absolute_error(y_val, val_preds),
        r2_score(y_val, val_preds)))