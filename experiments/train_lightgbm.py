import os
import pandas as pd
import numpy as np
import hopsworks
import lightgbm
from lightgbm import LGBMRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from joblib import dump
from dotenv import load_dotenv

load_dotenv()

project = hopsworks.login(api_key_value=os.getenv("HOPSWORKS_API_KEY"))
fs = project.get_feature_store()

aqi_fg = fs.get_feature_group(name="karachi_aqi_features", version=1)
df = aqi_fg.read()
print(f"Read {len(df)} rows from Hopsworks feature group")

targets_fg = fs.get_feature_group(name="karachi_aqi_targets", version=1)
targets_df = targets_fg.read()
print(f"Read {len(targets_df)} rows from Hopsworks targets group")

df["time"] = pd.to_datetime(df["time"]).dt.tz_localize(None)
targets_df["time"] = pd.to_datetime(targets_df["time"]).dt.tz_localize(None)
df = df.merge(targets_df, on="time", how="inner")
df = df.sort_values("time").reset_index(drop=True)
print(f"After merge: {len(df)} rows")

exclude_cols = ["time", "target_day1", "target_day2", "target_day3"]
feature_cols = [c for c in df.columns if c not in exclude_cols]
print(f"Using {len(feature_cols)} features")
print("Experiment: E - LightGBM with early stopping (three independent models)")

n = len(df)
train_end = int(n * 0.70)
val_end = int(n * 0.85)

targets = ["target_day1", "target_day2", "target_day3"]

lgbm_params = dict(
    n_estimators=300,
    learning_rate=0.05,
    max_depth=-1,
    num_leaves=31,
    min_child_samples=20,
    random_state=42,
    verbose=-1,
)

for target in targets:
    print(f"\n{'='*50}")
    print(f"Target: {target}")
    print('='*50)

    X_train = df[feature_cols].iloc[:train_end]
    y_train = df[target].iloc[:train_end]
    X_val = df[feature_cols].iloc[train_end:val_end]
    y_val = df[target].iloc[train_end:val_end]
    X_trainval = df[feature_cols].iloc[:val_end]
    y_trainval = df[target].iloc[:val_end]
    X_test = df[feature_cols].iloc[val_end:]
    y_test = df[target].iloc[val_end:]

    # Stage 1: validation comparison, with early stopping
    lgbm = LGBMRegressor(**lgbm_params)
    lgbm.fit(
        X_train, y_train,
        eval_set=[(X_val, y_val)],
        eval_metric="l2",
        callbacks=[lightgbm.early_stopping(30), lightgbm.log_evaluation(0)],
    )
    val_preds = lgbm.predict(X_val)
    print(f"\n[Validation] (best iteration: {lgbm.best_iteration_})")
    print("LightGBM — RMSE: {:.2f}, MAE: {:.2f}, R²: {:.4f}".format(
        np.sqrt(mean_squared_error(y_val, val_preds)),
        mean_absolute_error(y_val, val_preds),
        r2_score(y_val, val_preds)))

    # Stage 2: final model on train+val (re-split a small internal validation
    # slice purely for early stopping, since we can't validate against test)
    inner_split = int(len(X_trainval) * 0.9)
    X_fit, X_earlystop = X_trainval.iloc[:inner_split], X_trainval.iloc[inner_split:]
    y_fit, y_earlystop = y_trainval.iloc[:inner_split], y_trainval.iloc[inner_split:]

    lgbm_final = LGBMRegressor(**lgbm_params)
    lgbm_final.fit(
        X_fit, y_fit,
        eval_set=[(X_earlystop, y_earlystop)],
        eval_metric="l2",
        callbacks=[lightgbm.early_stopping(30), lightgbm.log_evaluation(0)],
    )
    test_preds = lgbm_final.predict(X_test)
    print(f"\n[Final — Test] (best iteration: {lgbm_final.best_iteration_})")
    print("LightGBM — RMSE: {:.2f}, MAE: {:.2f}, R²: {:.4f}".format(
        np.sqrt(mean_squared_error(y_test, test_preds)),
        mean_absolute_error(y_test, test_preds),
        r2_score(y_test, test_preds)))

    dump(lgbm_final, f"models/lgbm_{target}_final.joblib")
    print(f"Saved models/lgbm_{target}_final.joblib")