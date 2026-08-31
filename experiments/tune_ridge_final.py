import os
import pandas as pd
import numpy as np
import hopsworks
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_squared_error, r2_score
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

alphas = [1000.0, 5000.0, 10000.0, 50000.0, 100000.0]

for target in ["target_day1", "target_day2", "target_day3"]:
    X_train = df[feature_cols].iloc[:train_end]
    y_train = df[target].iloc[:train_end]
    X_val = df[feature_cols].iloc[train_end:val_end]
    y_val = df[target].iloc[train_end:val_end]

    best_alpha, best_r2 = None, -np.inf
    for alpha in alphas:
        m = Ridge(alpha=alpha)
        m.fit(X_train, y_train)
        preds = m.predict(X_val)
        r2 = r2_score(y_val, preds)
        if r2 > best_r2:
            best_r2, best_alpha = r2, alpha

    print(f"{target}: best alpha = {best_alpha}, validation R² = {best_r2:.4f}")