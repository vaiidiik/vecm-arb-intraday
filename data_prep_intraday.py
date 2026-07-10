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

prices = prices.ffill()
# Fix: .bfill() here would fill any leading NaN (an asset with no price yet
# at the very start of the window) using a *later* price -- textbook
# lookahead, even though it's very unlikely to matter for continuously-
# traded mega-caps. Drop those leading rows instead of inventing values
# from the future; the strategy doesn't start trading until LOOKBACK bars
# in anyway, so this costs nothing in practice.
prices = prices.dropna(how="any")

dollar_vol = dollar_vol.reindex(prices.index).ffill().fillna(0)
# Fix: shift(1) necessarily leaves exactly one leading NaN (there's no
# "bar -1" to shift from). The previous .bfill() filled it from the *next*
# row's ADV -- again lookahead, again on a bar that's always before
# LOOKBACK and never read by any trading decision. Filling with 0 instead
# keeps the guarantee that nothing here is derived from a future bar.
dollar_adv = dollar_vol.rolling(window=ADV_WINDOW, min_periods=1).mean().shift(1).fillna(0.0)

out_prices = os.path.join(script_dir, "sample_prices_intraday.csv")
out_adv = os.path.join(script_dir, "sample_dollar_adv_intraday.csv")

prices.to_csv(out_prices, index_label="Date")
dollar_adv.to_csv(out_adv, index_label="Date")
print(f"Saved {out_prices} / {out_adv}: {prices.shape[0]} bars x {prices.shape[1]} symbols")