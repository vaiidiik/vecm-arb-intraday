import pandas as pd
import os

script_dir = os.path.dirname(os.path.abspath(__file__))
RTH_CSV = os.path.join(script_dir, "combined_banks_15min_2020_2025_RTH_clean.csv")
BARS_PER_DAY = 27
ADV_WINDOW = 20 * BARS_PER_DAY

df = pd.read_csv(RTH_CSV)
df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True)

prices = df.pivot(index="timestamp", columns="symbol", values="close").sort_index()
df["dollar_volume"] = df["close"] * df["volume"]
dollar_vol = df.pivot(index="timestamp", columns="symbol", values="dollar_volume").sort_index()

prices = prices.ffill().bfill()
dollar_vol = dollar_vol.reindex(prices.index).ffill().fillna(0)
dollar_adv = dollar_vol.rolling(window=ADV_WINDOW, min_periods=1).mean().shift(1).bfill()

out_prices = os.path.join(script_dir, "sample_prices_intraday.csv")
out_adv = os.path.join(script_dir, "sample_dollar_adv_intraday.csv")

prices.to_csv(out_prices, index_label="Date")
dollar_adv.to_csv(out_adv, index_label="Date")
print(f"Saved {out_prices} / {out_adv}: {prices.shape[0]} bars x {prices.shape[1]} symbols")
