import zmq
import zmq.asyncio
import json
import logging
from collections import defaultdict


class MarketPublisher:
    def __init__(self, port=5555):
        self.context = zmq.Context()
        self.socket = self.context.socket(zmq.PUB)
        self.socket.bind(f"tcp://*:{port}")
        self.logger = logging.getLogger("VECM_ARB.Publisher")

    def publish_tick(self, timestamp, asset_id, price, adv, volatility):
        payload = {
            "timestamp": str(timestamp),
            "asset": asset_id,
            "price": float(price),
            "adv": float(adv),
            "volatility": float(volatility),
        }
        self.socket.send_string(f"MARKET {json.dumps(payload)}")

    def send_end_signal(self):
        self.socket.send_string("END")
        self.logger.info("END signal sent to all subscribers.")


class StrategySubscriber:
    def __init__(self, required_assets, port=5555):
        self.context = zmq.asyncio.Context()
        self.socket = self.context.socket(zmq.SUB)
        self.socket.connect(f"tcp://localhost:{port}")
        self.socket.setsockopt_string(zmq.SUBSCRIBE, "MARKET")
        self.socket.setsockopt_string(zmq.SUBSCRIBE, "END")
        self.required_assets = list(required_assets)
        self.required_asset_set = set(required_assets)
        self.snapshot_buffer = defaultdict(dict)
        self.logger = logging.getLogger("VECM_ARB.Subscriber")

    async def listen_and_buffer(self, callback_engine):
        while True:
            raw_message = await self.socket.recv_string()

            if raw_message.startswith("END"):
                self.logger.info("END signal received — stopping subscriber.")
                return

            _, json_data = raw_message.split(" ", 1)
            data = json.loads(json_data)

            t_stamp = data["timestamp"]
            asset = data["asset"]

            if asset not in self.required_asset_set:
                continue

            self.snapshot_buffer[t_stamp][asset] = {
                "price": data["price"],
                "adv": data["adv"],
                "volatility": data["volatility"],
            }

            if self.required_asset_set.issubset(self.snapshot_buffer[t_stamp]):
                snapshot = {
                    a: self.snapshot_buffer[t_stamp][a]
                    for a in self.required_assets
                }
                await self._trigger_math_engine(t_stamp, snapshot, callback_engine)
                del self.snapshot_buffer[t_stamp]

    async def _trigger_math_engine(self, timestamp, snapshot, callback_engine):
        if hasattr(callback_engine, "run"):
            await callback_engine.run(snapshot, timestamp)
