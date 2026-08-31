import pandas as pd
import numpy as np
from catboost import CatBoostRegressor
from sklearn.metrics import mean_squared_error, r2_score

df = pd.read_csv("karachi_features.csv", parse_dates=["time"])
df = df.sort_values("time").reset_index(drop=True)

exclude_cols = ["time", "target_day1", "target_day2", "target_day3"]
feature_cols = [c for c in df.columns if c not in exclude_cols]

n = len(df)
train_end = int(n * 0.70)
val_end = int(n * 0.85)

cb_params = {
    "target_day1": {"depth": 6, "learning_rate": 0.03},
    "target_day2": {"depth": 6, "learning_rate": 0.06},
    "target_day3": {"depth": 4, "learning_rate": 0.01},
}

# Full history vs last 6 months of training data only
full_train = df.iloc[:train_end]
recent_cutoff = df["time"].iloc[train_end] - pd.Timedelta(days=180)
recent_train = full_train[full_train["time"] >= recent_cutoff]
print(f"Full training set: {len(full_train)} rows")
print(f"Recent-only (6mo) training set: {len(recent_train)} rows")

X_val = df[feature_cols].iloc[train_end:val_end]

for target in ["target_day1", "target_day2", "target_day3"]:
    y_val = df[target].iloc[train_end:val_end]

    m_full = CatBoostRegressor(iterations=1000, random_state=42, early_stopping_rounds=30,
                                verbose=False, **cb_params[target])
    m_full.fit(full_train[feature_cols], full_train[target], eval_set=(X_val, y_val))
    r2_full = r2_score(y_val, m_full.predict(X_val))

    m_recent = CatBoostRegressor(iterations=1000, random_state=42, early_stopping_rounds=30,
                                  verbose=False, **cb_params[target])
    m_recent.fit(recent_train[feature_cols], recent_train[target], eval_set=(X_val, y_val))
    r2_recent = r2_score(y_val, m_recent.predict(X_val))

    print(f"{target}: full-history R²={r2_full:.4f}, recent-only R²={r2_recent:.4f}")