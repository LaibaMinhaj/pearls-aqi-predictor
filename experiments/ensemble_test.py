import os
import pandas as pd
import numpy as np
import hopsworks
from sklearn.linear_model import Ridge
from sklearn.svm import SVR
from sklearn.preprocessing import StandardScaler
from catboost import CatBoostRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from dotenv import load_dotenv

load_dotenv()
project = hopsworks.login(api_key_value=os.getenv("HOPSWORKS_API_KEY"))
fs = project.get_feature_store()

df = fs.get_feature_group(name="karachi_aqi_features", version=2).read()
targets_df = fs.get_feature_group(name="karachi_aqi_targets", version=1).read()
df["time"] = pd.to_datetime(df["time"]).dt.tz_localize(None)
targets_df["time"] = pd.to_datetime(targets_df["time"]).dt.tz_localize(None)
df = df.merge(targets_df, on="time", how="inner").sort_values("time").reset_index(drop=True)

exclude_cols = ["time", "target_day1", "target_day2", "target_day3"]
feature_cols = [c for c in df.columns if c not in exclude_cols]

n = len(df)
train_end = int(n * 0.70)
val_end = int(n * 0.85)

final_alphas = {"target_day1": 10000.0, "target_day2": 10000.0, "target_day3": 1000.0}

for target in ["target_day2", "target_day3"]:
    print(f"\n{'='*50}\nTarget: {target}\n{'='*50}")

    X_train = df[feature_cols].iloc[:train_end]
    y_train = df[target].iloc[:train_end]
    X_val = df[feature_cols].iloc[train_end:val_end]
    y_val = df[target].iloc[train_end:val_end]

    ridge = Ridge(alpha=final_alphas[target])
    ridge.fit(X_train, y_train)
    ridge_preds = ridge.predict(X_val)

    scaler = StandardScaler()
    X_train_s = scaler.fit_transform(X_train)
    X_val_s = scaler.transform(X_val)
    svr = SVR(kernel="rbf", C=10.0, epsilon=0.5)
    svr.fit(X_train_s, y_train)
    svr_preds = svr.predict(X_val_s)

    cb = CatBoostRegressor(iterations=1000, learning_rate=0.03, depth=6,
                            random_state=42, early_stopping_rounds=30, verbose=False)
    cb.fit(X_train, y_train, eval_set=(X_val, y_val))
    cb_preds = cb.predict(X_val)

    ensemble_preds = (ridge_preds + svr_preds + cb_preds) / 3

    # ---- Weighted ensemble: weight each model inversely proportional to its RMSE ----
    rmse_ridge = np.sqrt(mean_squared_error(y_val, ridge_preds))
    rmse_svr = np.sqrt(mean_squared_error(y_val, svr_preds))
    rmse_cb = np.sqrt(mean_squared_error(y_val, cb_preds))

    inv_rmse = np.array([1/rmse_ridge, 1/rmse_svr, 1/rmse_cb])
    weights = inv_rmse / inv_rmse.sum()

    weighted_preds = weights[0]*ridge_preds + weights[1]*svr_preds + weights[2]*cb_preds
    print(f"Weights (Ridge/SVR/CatBoost): {weights.round(3)}")

    for name, preds in [("Ridge", ridge_preds), ("SVR", svr_preds), ("CatBoost", cb_preds),
                         ("Ensemble (avg)", ensemble_preds), ("Ensemble (weighted)", weighted_preds)]:
        print("{:<20} RMSE: {:.2f}, MAE: {:.2f}, R²: {:.4f}".format(
            name,
            np.sqrt(mean_squared_error(y_val, preds)),
            mean_absolute_error(y_val, preds),
            r2_score(y_val, preds)))