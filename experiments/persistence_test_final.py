import pandas as pd
import numpy as np
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

df = pd.read_csv("karachi_features.csv", parse_dates=["time"])
df = df.sort_values("time").reset_index(drop=True)

n = len(df)
train_end = int(n * 0.70)
val_end = int(n * 0.85)

test_df = df.iloc[val_end:]
persistence_pred = test_df["us_aqi_rollmean_24h"]

for target in ["target_day1", "target_day2", "target_day3"]:
    y_true = test_df[target]
    rmse = np.sqrt(mean_squared_error(y_true, persistence_pred))
    mae = mean_absolute_error(y_true, persistence_pred)
    r2 = r2_score(y_true, persistence_pred)
    print(f"{target} — Persistence RMSE: {rmse:.2f}, MAE: {mae:.2f}, R²: {r2:.4f}")