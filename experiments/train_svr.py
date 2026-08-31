import os
import pandas as pd
import numpy as np
import hopsworks
from sklearn.svm import SVR
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
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

# NOTE: SVR on the full ~31k row training set is slow (SVR scales poorly with
# data size, roughly O(n^2) to O(n^3)). This may take several minutes per target.
for target in ["target_day1", "target_day2", "target_day3"]:
    print(f"\n{'='*50}")
    print(f"Target: {target}")
    print('='*50)

    X_train = df[feature_cols].iloc[:train_end]
    y_train = df[target].iloc[:train_end]
    X_val = df[feature_cols].iloc[train_end:val_end]
    y_val = df[target].iloc[train_end:val_end]

    # Scaler fit ONLY on training data, then applied to validation — no leakage
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_val_scaled = scaler.transform(X_val)

    svr = SVR(kernel="rbf", C=10.0, epsilon=0.5)
    svr.fit(X_train_scaled, y_train)
    preds = svr.predict(X_val_scaled)

    print("SVR — RMSE: {:.2f}, MAE: {:.2f}, R²: {:.4f}".format(
        np.sqrt(mean_squared_error(y_val, preds)),
        mean_absolute_error(y_val, preds),
        r2_score(y_val, preds)))