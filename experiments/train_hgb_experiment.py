import pandas as pd
import numpy as np
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from joblib import dump

df = pd.read_csv("karachi_features.csv", parse_dates=["time"])
df = df.sort_values("time").reset_index(drop=True)

exclude_cols = ["time", "target_1h", "target_24h", "target_48h", "target_72h"]
feature_cols = [c for c in df.columns if c not in exclude_cols]
print(f"Using {len(feature_cols)} features")
print("Experiment: D - HistGradientBoostingRegressor (same feature set as Experiment C)")

n = len(df)
train_end = int(n * 0.70)
val_end = int(n * 0.85)

targets = ["target_24h", "target_48h", "target_72h"]

hgb_params = dict(
    learning_rate=0.05,
    max_iter=300,
    max_leaf_nodes=31,
    min_samples_leaf=20,
    random_state=42,
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

    # Stage 1: compare on validation
    hgb = HistGradientBoostingRegressor(**hgb_params)
    hgb.fit(X_train, y_train)
    val_preds = hgb.predict(X_val)
    print("\n[Validation]")
    print("HGB — RMSE: {:.2f}, MAE: {:.2f}, R²: {:.4f}".format(
        np.sqrt(mean_squared_error(y_val, val_preds)),
        mean_absolute_error(y_val, val_preds),
        r2_score(y_val, val_preds)))

    # Stage 2: retrain on train+val, evaluate on test
    hgb_final = HistGradientBoostingRegressor(**hgb_params)
    hgb_final.fit(X_trainval, y_trainval)
    test_preds = hgb_final.predict(X_test)
    print("\n[Final — Test]")
    print("HGB — RMSE: {:.2f}, MAE: {:.2f}, R²: {:.4f}".format(
        np.sqrt(mean_squared_error(y_test, test_preds)),
        mean_absolute_error(y_test, test_preds),
        r2_score(y_test, test_preds)))

    dump(hgb_final, f"models/hgb_{target}_final.joblib")
    print(f"Saved models/hgb_{target}_final.joblib")