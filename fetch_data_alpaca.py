from alpaca.data.historical import StockHistoricalDataClient
from alpaca.data.requests import StockBarsRequest
from alpaca.data.timeframe import TimeFrame, TimeFrameUnit
from datetime import datetime
import pandas as pd


API_KEY = "your_api_key"
SECRET_KEY = "your_secret_key"


client = StockHistoricalDataClient(API_KEY, SECRET_KEY)


tickers = ["JPM", "BAC", "C", "WFC", "GS", "MS"]

start_time = datetime(2020, 1, 1)
end_time = datetime(2025, 12, 31)

print("Starting historical data download. Please wait, this may take a moment...")

try:
    request_params = StockBarsRequest(
        symbol_or_symbols=tickers,
        timeframe=TimeFrame(15, TimeFrameUnit.Minute),  
        start=start_time,
        end=end_time
    )

    bars = client.get_stock_bars(request_params)

    df = bars.df
    df = df.reset_index()

    df = df.sort_values(by=['timestamp', 'symbol']).reset_index(drop=True)

    output_filename = "combined_banks_15min_2020_2025.csv"
    df.to_csv(output_filename, index=False)

    print("\n" + "="*50)
    print("SUCCESS!")
    print(f"Total rows downloaded: {len(df)}")
    print(f"Saved to file: {output_filename}")
    print("="*50)

except Exception as e:
    print(f"\nAn error occurred: {e}")