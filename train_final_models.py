import os
import pandas as pd
import numpy as np
import hopsworks
from sklearn.linear_model import Ridge
from sklearn.svm import SVR
from sklearn.preprocessing import StandardScaler
from catboost import CatBoostRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from joblib import dump
from dotenv import load_dotenv

load_dotenv()

project = hopsworks.login(api_key_value=os.getenv("HOPSWORKS_API_KEY"))
fs = project.get_feature_store()
mr = project.get_model_registry()

df = fs.get_feature_group(name="karachi_aqi_features", version=2).read()
targets_df = fs.get_feature_group(name="karachi_aqi_targets", version=1).read()
df["time"] = pd.to_datetime(df["time"]).dt.tz_localize(None)
targets_df["time"] = pd.to_datetime(targets_df["time"]).dt.tz_localize(None)
df = df.merge(targets_df, on="time", how="inner").sort_values("time").reset_index(drop=True)

exclude_cols = ["time", "target_day1", "target_day2", "target_day3"]
feature_cols = [c for c in df.columns if c not in exclude_cols]
dump(feature_cols, "models/feature_cols.joblib")

n = len(df)
train_end = int(n * 0.70)
val_end = int(n * 0.85)

CONFIG = {
    "target_day1": {"ridge_alpha": 10000.0, "svr_C": 1.0, "svr_epsilon": 0.1, "cb_depth": 6, "cb_lr": 0.03},
    "target_day2": {"ridge_alpha": 10000.0, "svr_C": 10.0, "svr_epsilon": 0.5, "cb_depth": 6, "cb_lr": 0.03},
    "target_day3": {"ridge_alpha": 1000.0, "svr_C": 1.0, "svr_epsilon": 0.1, "cb_depth": 4, "cb_lr": 0.01},
}

os.makedirs("models", exist_ok=True)

for target, cfg in CONFIG.items():
    print(f"\n{'='*50}\nTarget: {target}\n{'='*50}")

    X_trainval = df[feature_cols].iloc[:val_end]
    y_trainval = df[target].iloc[:val_end]
    X_test = df[feature_cols].iloc[val_end:]
    y_test = df[target].iloc[val_end:]

    # ---- Ridge ----
    ridge = Ridge(alpha=cfg["ridge_alpha"])
    ridge.fit(X_trainval, y_trainval)
    ridge_preds = ridge.predict(X_test)
    dump(ridge, f"models/ridge_{target}_final.joblib")

    # ---- SVR (+ scaler, saved together in a folder) ----
    scaler = StandardScaler()
    X_trainval_s = scaler.fit_transform(X_trainval)
    X_test_s = scaler.transform(X_test)
    svr = SVR(kernel="rbf", C=cfg["svr_C"], epsilon=cfg["svr_epsilon"])
    svr.fit(X_trainval_s, y_trainval)
    svr_preds = svr.predict(X_test_s)

    svr_dir = f"models/svr_{target}"
    os.makedirs(svr_dir, exist_ok=True)
    dump(svr, f"{svr_dir}/svr.joblib")
    dump(scaler, f"{svr_dir}/scaler.joblib")

    # ---- CatBoost ----
    cb = CatBoostRegressor(iterations=1000, depth=cfg["cb_depth"], learning_rate=cfg["cb_lr"],
                            random_state=42, early_stopping_rounds=30, verbose=False)
    cb.fit(X_trainval, y_trainval, eval_set=(X_test, y_test))
    cb_preds = cb.predict(X_test)
    dump(cb, f"models/catboost_{target}_final.joblib")

    # ---- Ensemble (what actually gets used at inference) ----
    ensemble_preds = (ridge_preds + svr_preds + cb_preds) / 3
    rmse = np.sqrt(mean_squared_error(y_test, ensemble_preds))
    mae = mean_absolute_error(y_test, ensemble_preds)
    r2 = r2_score(y_test, ensemble_preds)
    print(f"Ensemble (test) — RMSE: {rmse:.2f}, MAE: {mae:.2f}, R²: {r2:.4f}")

    metrics = {"rmse": float(rmse), "mae": float(mae), "r2": float(r2)}

    # ---- Register all three components, each tagged with the ENSEMBLE's metrics ----
    # (the ensemble is the actual production predictor; individual model metrics
    # aren't separately meaningful once combined, so we attach the same shared
    # metric to each so predict_utils can read it back regardless of which it loads)
    for name, local_path in [
        (f"ridge_{target}", f"models/ridge_{target}_final.joblib"),
        (f"svr_{target}", svr_dir),
        (f"catboost_{target}", f"models/catboost_{target}_final.joblib"),
    ]:
        try:
            existing = mr.get_models(name=name)
            for old in existing:
                old.delete()
        except Exception:
            pass

        m = mr.python.create_model(name=name, metrics=metrics,
                                    description=f"Component of {target} ensemble (RMSE={rmse:.2f})")
        m.save(local_path)
        print(f"Registered {name}")