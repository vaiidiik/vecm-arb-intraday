import asyncio
import logging
import numpy as np
from concurrent.futures import ThreadPoolExecutor

from vecm_model_intraday import vecm
from risk_intraday import dynamic_risk_engine
from backtest_intraday import PortfolioBacktester

BARS_PER_DAY = 27
TRADING_DAYS_PER_YEAR = 252
PERIODS_PER_YEAR = BARS_PER_DAY * TRADING_DAYS_PER_YEAR

LOOKBACK = 200 * BARS_PER_DAY
MAX_HOLDING_BARS = 25 * BARS_PER_DAY
TREND_WINDOW = 21 * BARS_PER_DAY

ENTRY_THRESHOLD = 1.5
ZERO_RANK_GRACE = 5


def _borrow_rates(assets, adv, vols, periods_per_year=PERIODS_PER_YEAR):
    base_annual = {
        "NVDA": 0.0035, "AMD": 0.0060, "TSM": 0.0120,
        "ASML": 0.0040, "AVGO": 0.0045, "QCOM": 0.0060,
    }
    annual = np.array([base_annual.get(a, 0.0080) for a in assets], dtype=float)
    vol_base = max(np.nanmedian(vols), 1e-4)
    adv_base = max(np.nanmedian(adv), 1.0)
    vol_stress = np.clip(vols / vol_base - 1.0, 0.0, 3.0)
    liquidity_stress = np.clip(adv_base / (adv + 1e-8) - 1.0, 0.0, 3.0)
    annual = annual * (1.0 + 0.25 * vol_stress + 0.15 * liquidity_stress)
    return annual / periods_per_year


class VECMStrategyEngine:
    def __init__(self, assets, initial_capital=1_000_000, impact_gamma=0.05):
        self.assets = list(assets)
        self.N = len(self.assets)
        self.logger = logging.getLogger("VECM_ARB.Engine")

        self._executor = ThreadPoolExecutor(
            max_workers=1, thread_name_prefix="vecm_worker"
        )

        self.math_engine = vecm(
            significance=0.01,
            entry_threshold=ENTRY_THRESHOLD,
            min_kappa=2.0,
            max_kappa=50.0,
            delay_span=5,
            k_ar_diff=7,
            periods_per_year=PERIODS_PER_YEAR,
        )
        self.risk_engine = dynamic_risk_engine(
            num_assets=self.N,
            aum=initial_capital,
            gamma=0.02,
            entry_threshold=ENTRY_THRESHOLD,
            exit_threshold=0.2,
            short_exit_threshold=0.2,
            long_exit_threshold=0.2,
            turnover_penalty=0.010,
            max_leverage=3.0,
            max_weight_per_asset=0.80,
            target_fraction=1.0,
            max_holding_days=MAX_HOLDING_BARS,
            stop_loss_pct=0.08,
            trailing_stop_pct=0.05,
            volatility_threshold=0.35,
            trend_threshold=0.50,
        )
        self.backtester = PortfolioBacktester(
            initial_capital=initial_capital, gamma=impact_gamma,
            periods_per_year=PERIODS_PER_YEAR,
        )

        self._historical_prices = []
        self._historical_returns = []
        self._w_prev = np.zeros(self.N)
        self._zero_rank_days = 0
        self._prev_z_score = 0.0

    def _process_tick(self, timestamp, current_prices, current_adv, current_vols):
        self._historical_prices.append(current_prices.copy())

        if len(self._historical_prices) > 1:
            prev = self._historical_prices[-2]
            ret = (current_prices - prev) / (prev + 1e-12)
            self._historical_returns.append(float(np.mean(ret)))

        if len(self._historical_prices) < LOOKBACK:
            return

        rolling_window = np.array(self._historical_prices[-LOOKBACK:], dtype=float)
        log_window = np.log(rolling_window)
        borrow_rates = _borrow_rates(self.assets, current_adv, current_vols)

        z_score = 0.0
        rank = 0
        w_new = np.zeros(self.N)

        log_window_hist = log_window[:-1]
        rank, beta_set = self.math_engine.cointegrate(log_window_hist)

        if rank > 0:
            self._zero_rank_days = 0
            z_score, kappa, beta, rsi_val = self.math_engine.calculate_sscore(
                log_window[-1], beta_set, log_window_hist
            )

            stable = self.math_engine.check_cointegration_stability(
                log_window, beta, window=10
            )
            if not stable:
                w_new = np.zeros(self.N)
            elif beta is not None and kappa > 0.0 and abs(z_score) <= 5.0:
                signal_returns = np.diff(log_window, axis=0)
                cov_matrix = np.cov(signal_returns.T)

                n_ret = len(self._historical_returns)
                market_vol = (
                    np.std(self._historical_returns[-TREND_WINDOW:-1]) if n_ret >= TREND_WINDOW else 0.0
                )
                trend_strength = (
                    np.mean(self._historical_returns[-TREND_WINDOW:-1]) * 20
                    if n_ret >= TREND_WINDOW else 0.0
                )
                position_holding_days = (
                    max(
                        self.backtester.days - d
                        for d in self.backtester.entry_day.values()
                    )
                    if self.backtester.entry_day else 0
                )

                current_exposure = float(np.sum(np.abs(self._w_prev)))
                pnl_since_entry_now = (
                    self.backtester.position_pnl_sum() / self.backtester.capital
                    if self.backtester.capital > 0 else 0.0
                )
                risk_breach = (
                    position_holding_days >= self.risk_engine.max_holding_days
                    or pnl_since_entry_now < -self.risk_engine.stop_loss_pct
                    or pnl_since_entry_now < -self.risk_engine.trailing_stop_pct
                )
                needs_rebalance = False

                if current_exposure < 1e-4 and abs(z_score) >= ENTRY_THRESHOLD:
                    needs_rebalance = True
                elif (
                    current_exposure > 1e-4
                    and (abs(z_score - self._prev_z_score) > 0.50 or risk_breach)
                ):
                    needs_rebalance = True
                elif current_exposure > 1e-4:
                    w_new = self._w_prev.copy()

                if needs_rebalance:
                    w_new = self.risk_engine.optimise_weights(
                        w_prev=self._w_prev,
                        cov_matrix=cov_matrix,
                        beta=beta,
                        z_score=z_score,
                        rsi=rsi_val,
                        adv=current_adv,
                        vols=current_vols,
                        borrow_rates=borrow_rates,
                        kappa=kappa,
                        holding_days=position_holding_days,
                        pnl_since_entry=(
                            self.backtester.position_pnl_sum()
                            / self.backtester.capital
                        ),
                        market_vol=market_vol,
                        trend_strength=trend_strength,
                        rank=rank,
                        capital=self.backtester.capital,
                    )

            self._prev_z_score = z_score

        else:
            self._zero_rank_days += 1
            if self._zero_rank_days <= ZERO_RANK_GRACE and np.sum(np.abs(self._w_prev)) > 1e-4:
                w_new = self._w_prev.copy()
            else:
                w_new = np.zeros(self.N)

        self.backtester.process_day(
            date=timestamp,
            w_prev=self._w_prev,
            w_new=w_new,
            prices_prev=self._historical_prices[-2],
            prices_new=self._historical_prices[-1],
            adv=current_adv,
            vols=current_vols,
            z_score=z_score,
            rank=rank,
            borrow_rates=borrow_rates,
        )

        self.logger.info(
            f"[{timestamp}] z={z_score:+.2f}  rank={rank}  "
            f"exposure={np.sum(np.abs(w_new)):.2f}  "
            f"capital=${self.backtester.capital:,.0f}"
        )

        self._w_prev = w_new

    async def run(self, snapshot, timestamp):
        prices = np.array([snapshot[a]["price"] for a in self.assets], dtype=float)
        adv = np.array([snapshot[a]["adv"] for a in self.assets], dtype=float)
        vols = np.array([snapshot[a]["volatility"] for a in self.assets], dtype=float)

        loop = asyncio.get_running_loop()
        await loop.run_in_executor(
            self._executor,
            self._process_tick,
            timestamp, prices, adv, vols,
        )

    def save_results(self, daily_path, metrics_path):
        self.backtester.save_results(daily_path, metrics_path)
        _, metrics = self.backtester.generate_metrics()
        return metrics