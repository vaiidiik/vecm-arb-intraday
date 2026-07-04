import asyncio
import os
import logging

from data_intraday import DataSimulator
from broker_intraday import MarketPublisher

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(name)-22s  %(levelname)s  %(message)s",
)
logger = logging.getLogger("VECM_ARB.Publisher")

ASSETS = ["NVDA", "AMD", "TSM", "ASML", "AVGO", "QCOM"]
BARS_PER_DAY = 27
VOL_WINDOW = 20 * BARS_PER_DAY

TICK_DELAY_SECONDS = 0.001


async def main():
    script_dir = os.path.dirname(os.path.abspath(__file__))
    prices_path = os.path.join(script_dir, "sample_prices_intraday.csv")
    adv_path = os.path.join(script_dir, "sample_dollar_adv_intraday.csv")

    publisher = MarketPublisher(port=5555)
    simulator = DataSimulator(prices_path, adv_path, vol_window=VOL_WINDOW)

    available = simulator.prices.columns.tolist()
    assets = [a for a in ASSETS if a in available]
    if len(assets) < 2:
        raise ValueError(f"Need at least 2 assets. Found: {assets}")

    simulator.prices = simulator.prices[assets]
    simulator.adv = simulator.adv[assets]
    simulator.volatility = simulator.volatility[assets]

    n_bars = len(simulator.prices)
    logger.info(f"Publishing {n_bars} bars x {len(assets)} assets -> port 5555")
    logger.info(f"Assets: {assets}")
    logger.info("Make sure run_strategy_intraday.py is already running in another terminal.")

    await simulator.stream_to_publisher(publisher, delay=TICK_DELAY_SECONDS)

    publisher.send_end_signal()
    logger.info("All data sent. Publisher exiting.")


if __name__ == "__main__":
    asyncio.run(main())