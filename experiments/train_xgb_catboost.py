import os
import pandas as pd
import numpy as np
import hopsworks
from xgboost import XGBRegressor
from catboost import CatBoostRegressor
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

for target in ["target_day1", "target_day2", "target_day3"]:
    print(f"\n{'='*50}")
    print(f"Target: {target}")
    print('='*50)

    X_train = df[feature_cols].iloc[:train_end]
    y_train = df[target].iloc[:train_end]
    X_val = df[feature_cols].iloc[train_end:val_end]
    y_val = df[target].iloc[train_end:val_end]

    # ---- XGBoost with early stopping ----
    xgb = XGBRegressor(
        n_estimators=500,
        learning_rate=0.03,
        max_depth=5,
        subsample=0.8,
        colsample_bytree=0.8,
        random_state=42,
        early_stopping_rounds=30,
        eval_metric="rmse",
    )
    xgb.fit(X_train, y_train, eval_set=[(X_val, y_val)], verbose=False)
    xgb_preds = xgb.predict(X_val)
    print("XGBoost — RMSE: {:.2f}, MAE: {:.2f}, R²: {:.4f} (best_iter={})".format(
        np.sqrt(mean_squared_error(y_val, xgb_preds)),
        mean_absolute_error(y_val, xgb_preds),
        r2_score(y_val, xgb_preds),
        xgb.best_iteration))

    # ---- CatBoost with early stopping ----
    cb = CatBoostRegressor(
        iterations=1000,
        learning_rate=0.03,
        depth=6,
        random_state=42,
        early_stopping_rounds=30,
        verbose=False,
    )
    cb.fit(X_train, y_train, eval_set=(X_val, y_val))
    cb_preds = cb.predict(X_val)
    print("CatBoost — RMSE: {:.2f}, MAE: {:.2f}, R²: {:.4f} (best_iter={})".format(
        np.sqrt(mean_squared_error(y_val, cb_preds)),
        mean_absolute_error(y_val, cb_preds),
        r2_score(y_val, cb_preds),
        cb.get_best_iteration()))