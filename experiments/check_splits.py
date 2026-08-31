import pandas as pd

df = pd.read_csv("karachi_features.csv", parse_dates=["time"])
df = df.sort_values("time").reset_index(drop=True)

n = len(df)
train_end = int(n * 0.70)
val_end = int(n * 0.85)

splits = {
    "Train": df.iloc[:train_end],
    "Validation": df.iloc[train_end:val_end],
    "Test": df.iloc[val_end:]
}

for name, split in splits.items():
    print(f"\n{name}")
    print(f"Date range: {split['time'].min()} -> {split['time'].max()}")
    print(f"Rows: {len(split)}")
    print(f"AQI mean: {split['target_24h'].mean():.2f}")
    print(f"AQI std: {split['target_24h'].std():.2f}")