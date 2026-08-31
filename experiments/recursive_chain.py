import pandas as pd
import numpy as np
from sklearn.model_selection import KFold
from catboost import CatBoostRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

df = pd.read_csv("karachi_features.csv", parse_dates=["time"])
df = df.sort_values("time").reset_index(drop=True)

exclude_cols = ["time", "target_day1", "target_day2", "target_day3"]
feature_cols = [c for c in df.columns if c not in exclude_cols]

n = len(df)
train_end = int(n * 0.70)
val_end = int(n * 0.85)

cb_params = {
    "target_day1": {"depth": 6, "learning_rate": 0.03},
    "target_day2": {"depth": 6, "learning_rate": 0.06},
    "target_day3": {"depth": 4, "learning_rate": 0.01},
}

X_train_full = df[feature_cols].iloc[:train_end].reset_index(drop=True)
y_train_day1 = df["target_day1"].iloc[:train_end].reset_index(drop=True)
X_val = df[feature_cols].iloc[train_end:val_end].reset_index(drop=True)
y_val_day1 = df["target_day1"].iloc[train_end:val_end].reset_index(drop=True)
y_val_day2 = df["target_day2"].iloc[train_end:val_end].reset_index(drop=True)
y_val_day3 = df["target_day3"].iloc[train_end:val_end].reset_index(drop=True)

# ---- Step 1: generate out-of-fold Day1 predictions for the training set ----
print("Generating out-of-fold Day1 predictions...")
oof_day1_preds = np.zeros(len(X_train_full))
kf = KFold(n_splits=5, shuffle=False)  # shuffle=False keeps chronological folds

for fold_idx, (fit_idx, holdout_idx) in enumerate(kf.split(X_train_full)):
    m = CatBoostRegressor(iterations=500, random_state=42, verbose=False, **cb_params["target_day1"])
    m.fit(X_train_full.iloc[fit_idx], y_train_day1.iloc[fit_idx])
    oof_day1_preds[holdout_idx] = m.predict(X_train_full.iloc[holdout_idx])
    print(f"  Fold {fold_idx+1}/5 done")

# ---- Step 2: train the real Day1 model on ALL training data (for validation/inference use) ----
day1_model = CatBoostRegressor(iterations=1000, random_state=42, early_stopping_rounds=30,
                                verbose=False, **cb_params["target_day1"])
day1_model.fit(X_train_full, y_train_day1, eval_set=(X_val, y_val_day1))
val_day1_preds = day1_model.predict(X_val)

print(f"\nDay1 (baseline, for reference) — R²: {r2_score(y_val_day1, val_day1_preds):.4f}")

# ---- Step 3: build Day2/Day3 feature sets WITH the Day1 prediction added ----
X_train_chained = X_train_full.copy()
X_train_chained["day1_pred"] = oof_day1_preds  # out-of-fold, realistic

X_val_chained = X_val.copy()
X_val_chained["day1_pred"] = val_day1_preds  # real Day1 model's prediction, same as at inference

for target, y_val in [("target_day2", y_val_day2), ("target_day3", y_val_day3)]:
    y_train = df[target].iloc[:train_end].reset_index(drop=True)

    # Without chaining (baseline)
    m_plain = CatBoostRegressor(iterations=1000, random_state=42, early_stopping_rounds=30,
                                 verbose=False, **cb_params[target])
    m_plain.fit(X_train_full, y_train, eval_set=(X_val, y_val))
    preds_plain = m_plain.predict(X_val)

    # With chaining (day1_pred as extra feature)
    m_chained = CatBoostRegressor(iterations=1000, random_state=42, early_stopping_rounds=30,
                                   verbose=False, **cb_params[target])
    m_chained.fit(X_train_chained, y_train, eval_set=(X_val_chained, y_val))
    preds_chained = m_chained.predict(X_val_chained)

    print(f"\n{target}")
    print("  Without chaining — RMSE: {:.2f}, R²: {:.4f}".format(
        np.sqrt(mean_squared_error(y_val, preds_plain)), r2_score(y_val, preds_plain)))
    print("  With chaining    — RMSE: {:.2f}, R²: {:.4f}".format(
        np.sqrt(mean_squared_error(y_val, preds_chained)), r2_score(y_val, preds_chained)))