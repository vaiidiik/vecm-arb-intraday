import asyncio
import os
import logging

from broker_intraday import StrategySubscriber
from strategy_engine_intraday import VECMStrategyEngine

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(name)-22s  %(levelname)s  %(message)s",
)
logger = logging.getLogger("VECM_ARB.Runner")

ASSETS = ["JPM", "BAC", "C", "WFC", "GS", "MS"]
INITIAL_CAPITAL = 1_000_000
IMPACT_GAMMA = 0.05


async def main():
    script_dir = os.path.dirname(os.path.abspath(__file__))

    engine = VECMStrategyEngine(
        assets=ASSETS,
        initial_capital=INITIAL_CAPITAL,
        impact_gamma=IMPACT_GAMMA,
    )
    subscriber = StrategySubscriber(required_assets=ASSETS, port=5555)

    logger.info(f"Strategy engine ready.  Assets: {ASSETS}")
    logger.info("Waiting for market data on port 5555 ...")
    logger.info("Start run_publisher_intraday.py in a second terminal when ready.")

    await subscriber.listen_and_buffer(engine)

    logger.info("Stream ended. Saving results ...")
    out_daily = os.path.join(script_dir, "backtest_intraday_live.csv")
    out_metrics = os.path.join(script_dir, "performance_metrics_intraday_live.csv")
    metrics = engine.save_results(out_daily, out_metrics)

    print("\n=== Performance Metrics (distributed intraday run) ===")
    for _, r in metrics.iterrows():
        print(f"  {r['metric']:20s}: {r['value']}")

    logger.info(f"Bar-level P&L saved  -> {out_daily}")
    logger.info(f"Metrics saved        -> {out_metrics}")


if __name__ == "__main__":
    asyncio.run(main())