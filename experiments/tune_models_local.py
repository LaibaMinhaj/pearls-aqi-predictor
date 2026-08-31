import pandas as pd
import numpy as np
from sklearn.svm import SVR
from sklearn.preprocessing import StandardScaler
from catboost import CatBoostRegressor
from sklearn.metrics import mean_squared_error, r2_score

df = pd.read_csv("karachi_features.csv", parse_dates=["time"])
df = df.sort_values("time").reset_index(drop=True)

exclude_cols = ["time", "target_day1", "target_day2", "target_day3"]
feature_cols = [c for c in df.columns if c not in exclude_cols]

n = len(df)
train_end = int(n * 0.70)
val_end = int(n * 0.85)

for target in ["target_day1", "target_day2", "target_day3"]:
    print(f"\n{'='*50}\nTarget: {target}\n{'='*50}")

    X_train = df[feature_cols].iloc[:train_end]
    y_train = df[target].iloc[:train_end]
    X_val = df[feature_cols].iloc[train_end:val_end]
    y_val = df[target].iloc[train_end:val_end]

    scaler = StandardScaler()
    X_train_s = scaler.fit_transform(X_train)
    X_val_s = scaler.transform(X_val)

    # ---- SVR grid search (on a subsample for speed) ----
    sample_size = min(8000, len(X_train_s))
    X_train_sample = X_train_s[:sample_size]
    y_train_sample = y_train.iloc[:sample_size]

    best_svr = (None, -np.inf)
    for C in [1.0, 10.0, 50.0]:
        for epsilon in [0.1, 0.5]:
            m = SVR(kernel="rbf", C=C, epsilon=epsilon)
            m.fit(X_train_sample, y_train_sample)
            r2 = r2_score(y_val, m.predict(X_val_s))
            print(f"  SVR C={C}, epsilon={epsilon} -> R²={r2:.4f}")
            if r2 > best_svr[1]:
                best_svr = ((C, epsilon), r2)
    print(f"Best SVR: C={best_svr[0][0]}, epsilon={best_svr[0][1]}, R²={best_svr[1]:.4f}")

    # ---- CatBoost grid search ----
    best_cb = (None, -np.inf)
    for depth in [4, 6, 8]:
        for lr in [0.01, 0.03, 0.06]:
            m = CatBoostRegressor(iterations=1000, learning_rate=lr, depth=depth,
                                   random_state=42, early_stopping_rounds=30, verbose=False)
            m.fit(X_train, y_train, eval_set=(X_val, y_val))
            r2 = r2_score(y_val, m.predict(X_val))
            print(f"  CatBoost depth={depth}, lr={lr} -> R²={r2:.4f}")
            if r2 > best_cb[1]:
                best_cb = ((depth, lr), r2)
    print(f"Best CatBoost: depth={best_cb[0][0]}, lr={best_cb[0][1]}, R²={best_cb[1]:.4f}")