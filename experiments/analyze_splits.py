import pandas as pd
import matplotlib.pyplot as plt

# Load data
df = pd.read_csv("karachi_features.csv", parse_dates=["time"])
df = df.sort_values("time").reset_index(drop=True)

n = len(df)
train_end = int(n * 0.70)
val_end = int(n * 0.85)

train = df.iloc[:train_end]
val = df.iloc[train_end:val_end]
test = df.iloc[val_end:]

print("=" * 60)
print("AQI Statistics")
print("=" * 60)

for name, split in [("Train", train), ("Validation", val), ("Test", test)]:
    print(f"\n{name}")
    print(f"Rows : {len(split)}")
    print(f"Mean : {split['us_aqi'].mean():.2f}")
    print(f"Std  : {split['us_aqi'].std():.2f}")
    print(f"Min  : {split['us_aqi'].min():.2f}")
    print(f"Max  : {split['us_aqi'].max():.2f}")

print("\nDate ranges")
print(f"Train      : {train['time'].min()} -> {train['time'].max()}")
print(f"Validation : {val['time'].min()} -> {val['time'].max()}")
print(f"Test       : {test['time'].min()} -> {test['time'].max()}")

plt.figure(figsize=(15,5))
plt.plot(df["time"], df["us_aqi"], linewidth=1)

plt.axvline(df.loc[train_end, "time"], linestyle="--", label="Train/Validation")
plt.axvline(df.loc[val_end, "time"], linestyle="--", label="Validation/Test")

plt.title("AQI over Time")
plt.xlabel("Time")
plt.ylabel("US AQI")
plt.legend()

plt.tight_layout()
plt.show()
targets = ["target_day1", "target_day2", "target_day3"]

for target in targets:
    print("\n" + "=" * 60)
    print(target)
    print("=" * 60)

    for name, split in [("Train", train), ("Validation", val), ("Test", test)]:
        print(f"\n{name}")
        print(f"Mean : {split[target].mean():.2f}")
        print(f"Std  : {split[target].std():.2f}")
        print(f"Min  : {split[target].min():.2f}")
        print(f"Max  : {split[target].max():.2f}")