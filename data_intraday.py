import pandas as pd
import numpy as np
import asyncio
import logging


class DataSimulator:
    def __init__(self, prices_path, adv_path, vol_window=20):
        self.logger = logging.getLogger("VECM_ARB.Simulator")
        self.logger.setLevel(logging.INFO)

        self.prices = pd.read_csv(prices_path, index_col='Date', parse_dates=True)
        self.adv = pd.read_csv(adv_path, index_col='Date', parse_dates=True)

        returns = self.prices.pct_change().fillna(0)
        self.volatility = returns.rolling(window=vol_window).std().fillna(0.01).shift(1).fillna(0.01)

    async def stream_to_publisher(self, publisher, delay=0.001):
        self.logger.info("Initiating historical data stream...")

        for timestamp, row in self.prices.iterrows():
            t_str = timestamp.strftime("%Y-%m-%d %H:%M:%S")

            for asset in self.prices.columns:
                price = row[asset]
                adv = self.adv.loc[timestamp, asset]
                vol = self.volatility.loc[timestamp, asset]

                publisher.publish_tick(t_str, asset, price, adv, vol)

                await asyncio.sleep(delay)

        self.logger.info("Data stream complete.")