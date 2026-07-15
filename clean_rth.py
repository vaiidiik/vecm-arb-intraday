import pandas as pd

RAW_CSV = "combined_banks_15min_2020_2025.csv"
OUTPUT_CSV = "combined_banks_15min_2020_2025_RTH_clean.csv"

df = pd.read_csv(RAW_CSV)
df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True)

df["et"] = df["timestamp"].dt.tz_convert("US/Eastern")
df["time"] = df["et"].dt.time

from datetime import time as dtime
rth_start = dtime(9, 30)
rth_end = dtime(16, 0)

df = df[(df["time"] >= rth_start) & (df["time"] <= rth_end)]
df = df.drop(columns=["et", "time"])
df = df.sort_values(by=["timestamp", "symbol"]).reset_index(drop=True)

df.to_csv(OUTPUT_CSV, index=False)
print(f"Saved {len(df)} RTH rows to {OUTPUT_CSV}")
print(f"Unique symbols: {sorted(df['symbol'].unique())}")
print(f"Date range: {df['timestamp'].min()} to {df['timestamp'].max()}")
bars_per_day = df.groupby([df['timestamp'].dt.date, 'symbol']).size()
print(f"Bars per symbol per day: min={bars_per_day.min()}, max={bars_per_day.max()}, median={bars_per_day.median()}")
