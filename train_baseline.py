import pandas as pd
import numpy as np
from sklearn.linear_model import Ridge
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from joblib import dump

df = pd.read_csv("karachi_features.csv", parse_dates=["time"])
df = df.sort_values("time").reset_index(drop=True)

# Only exclude the columns that are genuinely unknowable at prediction time:
# the target columns themselves. Current pollutants/weather ARE known at
# prediction time (live sensor + forecast APIs), and future-weather-proxy
# columns are deliberately included as forecast substitutes.
exclude_cols = [
    "time", "target_1h", "target_24h", "target_48h", "target_72h",
]
feature_cols = [c for c in df.columns if c not in exclude_cols]

print(f"Using {len(feature_cols)} features")
print("Experiment: C - baseline + current weather + current pollutants + future weather proxy")

n = len(df)
train_end = int(n * 0.70)
val_end = int(n * 0.85)

print(f"Train samples: {train_end}")
print(f"Validation samples: {val_end - train_end}")
print(f"Test samples: {n - val_end}")
print(f"Train+Val samples (for final model): {val_end}")

targets = ["target_24h", "target_48h", "target_72h"]

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

    # ---- STAGE 1: train on 70%, compare models on validation ----
    ridge_baseline = Ridge(alpha=1.0)
    ridge_baseline.fit(X_train, y_train)
    ridge_val_preds = ridge_baseline.predict(X_val)

    rf_baseline = RandomForestRegressor(n_estimators=100, max_depth=15, random_state=42, n_jobs=-1)
    rf_baseline.fit(X_train, y_train)
    rf_val_preds = rf_baseline.predict(X_val)

    print("\n[Model comparison on Validation set]")
    print("Ridge  — RMSE: {:.2f}, MAE: {:.2f}, R²: {:.4f}".format(
        np.sqrt(mean_squared_error(y_val, ridge_val_preds)),
        mean_absolute_error(y_val, ridge_val_preds),
        r2_score(y_val, ridge_val_preds)))
    print("RF     — RMSE: {:.2f}, MAE: {:.2f}, R²: {:.4f}".format(
        np.sqrt(mean_squared_error(y_val, rf_val_preds)),
        mean_absolute_error(y_val, rf_val_preds),
        r2_score(y_val, rf_val_preds)))

    # ---- STAGE 2: retrain both on Train+Val (85%), evaluate once on Test ----
    ridge_final = Ridge(alpha=1.0)
    ridge_final.fit(X_trainval, y_trainval)
    ridge_test_preds = ridge_final.predict(X_test)

    rf_final = RandomForestRegressor(n_estimators=100, max_depth=15, random_state=42, n_jobs=-1)
    rf_final.fit(X_trainval, y_trainval)
    rf_test_preds = rf_final.predict(X_test)

    print("\n[Final models trained on Train+Val, evaluated on Test]")
    print("Ridge  — RMSE: {:.2f}, MAE: {:.2f}, R²: {:.4f}".format(
        np.sqrt(mean_squared_error(y_test, ridge_test_preds)),
        mean_absolute_error(y_test, ridge_test_preds),
        r2_score(y_test, ridge_test_preds)))
    print("RF     — RMSE: {:.2f}, MAE: {:.2f}, R²: {:.4f}".format(
        np.sqrt(mean_squared_error(y_test, rf_test_preds)),
        mean_absolute_error(y_test, rf_test_preds),
        r2_score(y_test, rf_test_preds)))

    # ---- Save only the final (Train+Val-trained) models ----
    dump(ridge_final, f"ridge_{target}_final.joblib")
    dump(rf_final, f"rf_{target}_final.joblib")
    print(f"\nSaved ridge_{target}_final.joblib and rf_{target}_final.joblib")