import pandas as pd
import numpy as np
from sklearn.linear_model import Ridge
from sklearn.svm import SVR
from sklearn.preprocessing import StandardScaler
from catboost import CatBoostRegressor

df = pd.read_csv("karachi_features.csv", parse_dates=["time"])
df = df.sort_values("time").reset_index(drop=True)

exclude_cols = ["time", "target_day1", "target_day2", "target_day3"]
feature_cols = [c for c in df.columns if c not in exclude_cols]

n = len(df)
val_end = int(n * 0.85)

CONFIG = {
    "target_day1": {"ridge_alpha": 10000.0, "svr_C": 1.0, "svr_epsilon": 0.1, "cb_depth": 6, "cb_lr": 0.03},
    "target_day2": {"ridge_alpha": 10000.0, "svr_C": 10.0, "svr_epsilon": 0.5, "cb_depth": 6, "cb_lr": 0.03},
    "target_day3": {"ridge_alpha": 1000.0, "svr_C": 1.0, "svr_epsilon": 0.1, "cb_depth": 4, "cb_lr": 0.01},
}

X_trainval = df[feature_cols].iloc[:val_end]
X_test = df[feature_cols].iloc[val_end:]
test_meta = df.iloc[val_end:][["time", "us_aqi", "temperature_2m", "relative_humidity_2m",
                                 "wind_speed_10m", "precipitation", "pm2_5"]].reset_index(drop=True)

for target, cfg in CONFIG.items():
    y_trainval = df[target].iloc[:val_end]
    y_test = df[target].iloc[val_end:].reset_index(drop=True)

    ridge = Ridge(alpha=cfg["ridge_alpha"]).fit(X_trainval, y_trainval)
    scaler = StandardScaler().fit(X_trainval)
    svr = SVR(kernel="rbf", C=cfg["svr_C"], epsilon=cfg["svr_epsilon"]).fit(scaler.transform(X_trainval), y_trainval)
    cb = CatBoostRegressor(iterations=1000, depth=cfg["cb_depth"], learning_rate=cfg["cb_lr"],
                            random_state=42, verbose=False).fit(X_trainval, y_trainval)

    preds = (ridge.predict(X_test) + svr.predict(scaler.transform(X_test)) + cb.predict(X_test)) / 3
    errors = np.abs(preds - y_test)

    result = test_meta.copy()
    result["actual"] = y_test
    result["predicted"] = preds.round(1)
    result["abs_error"] = errors.round(1)
    result = result.sort_values("abs_error", ascending=False)

    print(f"\n{'='*60}\n{target} — Top 15 worst predictions\n{'='*60}")
    print(result.head(15).to_string(index=False))