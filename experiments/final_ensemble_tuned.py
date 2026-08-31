import pandas as pd
import numpy as np
from sklearn.linear_model import Ridge
from sklearn.svm import SVR
from sklearn.preprocessing import StandardScaler
from catboost import CatBoostRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

df = pd.read_csv("karachi_features.csv", parse_dates=["time"])
df = df.sort_values("time").reset_index(drop=True)

exclude_cols = ["time", "target_day1", "target_day2", "target_day3"]
feature_cols = [c for c in df.columns if c not in exclude_cols]

n = len(df)
train_end = int(n * 0.70)
val_end = int(n * 0.85)

final_alphas = {"target_day1": 10000.0, "target_day2": 10000.0, "target_day3": 1000.0}
svr_params = {"C": 1.0, "epsilon": 0.1}  # from tuning
cb_params = {
    "target_day1": {"depth": 6, "learning_rate": 0.03},
    "target_day2": {"depth": 6, "learning_rate": 0.06},
    "target_day3": {"depth": 4, "learning_rate": 0.01},
}

for target in ["target_day1", "target_day2", "target_day3"]:
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
    svr = SVR(kernel="rbf", **svr_params)
    svr.fit(X_train_s, y_train)  # full data this time, not subsample
    svr_preds = svr.predict(X_val_s)

    cb = CatBoostRegressor(iterations=1000, random_state=42, early_stopping_rounds=30,
                            verbose=False, **cb_params[target])
    cb.fit(X_train, y_train, eval_set=(X_val, y_val))
    cb_preds = cb.predict(X_val)

    ensemble_preds = (ridge_preds + svr_preds + cb_preds) / 3

    for name, preds in [("Ridge", ridge_preds), ("SVR (tuned, full data)", svr_preds),
                         ("CatBoost (tuned)", cb_preds), ("Ensemble (avg)", ensemble_preds)]:
        print("{:<24} RMSE: {:.2f}, MAE: {:.2f}, R²: {:.4f}".format(
            name,
            np.sqrt(mean_squared_error(y_val, preds)),
            mean_absolute_error(y_val, preds),
            r2_score(y_val, preds)))