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
TREND_WINDOW = 21 * BARS_PER_DAY

ENTRY_THRESHOLD = 1.4
ZERO_RANK_GRACE = 5

EXIT_CRITICAL_IDX = 0
COINT_INVALID_STREAK_REQUIRED = 1
STABILITY_WINDOW = 10 * BARS_PER_DAY


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
            max_kappa=150.0,
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
            turnover_penalty=0.0005,
            max_leverage=3.0,
            max_weight_per_asset=0.80,
            target_fraction=1.0,
            volatility_threshold=0.35,
            trend_threshold=0.65,
            capital_per_trade_frac=1.0
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
        self._zscore_history = []
        self._active_pair = None
        self._active_beta = None
        self._invalid_streak = 0

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

        # Generate signals using the same flow as the backtester
        cached_signals = self.math_engine.generate_all_signals(log_window, self.N)

        SPREAD_WINDOW = 20 * BARS_PER_DAY
        spread_start = max(0, len(log_window) - SPREAD_WINDOW)
        log_recent = log_window[spread_start:]

        live_signals = []
        for sig in cached_signals:
            bv = sig["beta_full"]
            spread = log_recent @ bv
            if len(spread) < 20:
                continue
            z, mu, sigma = self.math_engine.compute_spread_z(spread, lookback=SPREAD_WINDOW)
            rsi_arr = self.math_engine.compute_rsi(spread)
            rsi = float(rsi_arr[-1]) if len(rsi_arr) > 0 else 50.0
            _, _, macd_hist = self.math_engine.compute_macd(spread)

            halflife_sig = sig.get("halflife", 50.0)
            live_signals.append({
                "z": z,
                "sigma": sigma,
                "halflife": halflife_sig,
                "beta_full": bv,
                "rsi": rsi,
                "macd_hist": macd_hist,
                "pair": sig.get("pair"),
                "rank": sig.get("rank", 1),
                "score": abs(z) / max(halflife_sig, 1.0),
            })

        live_signals.sort(key=lambda x: x["score"], reverse=True)

        if live_signals:
            z_score = live_signals[0]["z"]
            rank = live_signals[0]["rank"]
            self._zero_rank_days = 0
        else:
            z_score = 0.0
            rank = 0
            self._zero_rank_days += 1

        gross_exp = np.abs(self._w_prev).sum()

        if gross_exp > 0.01:
            pnl_since_entry = self.backtester.position_pnl_sum() / self.backtester.capital if self.backtester.capital > 0 else 0.0
            holding_bars = max([self.backtester.days - d for d in self.backtester.entry_day.values()], default=0)
        else:
            pnl_since_entry = 0.0
            holding_bars = 0

        coint_invalid = False
        if gross_exp > 0.01 and holding_bars > 0 and holding_bars % BARS_PER_DAY == 0:
            if self._active_pair is not None:
                pair_lp = log_window[:, list(self._active_pair)]
                pair_beta = self._active_beta[list(self._active_pair)] if self._active_beta is not None else None
                rank_check, _, _, _ = self.math_engine.cointegrate(pair_lp, critical_idx_override=EXIT_CRITICAL_IDX)
                rank_failed = (rank_check == 0)
                stability_ok = self.math_engine.check_cointegration_stability(pair_lp, pair_beta, window=STABILITY_WINDOW)
            else:
                full_rank, _, _, _ = self.math_engine.cointegrate(log_window, critical_idx_override=EXIT_CRITICAL_IDX)
                rank_failed = (full_rank == 0)
                stability_ok = self.math_engine.check_cointegration_stability(log_window, self._active_beta, window=STABILITY_WINDOW)

            if not stability_ok:
                self._invalid_streak = COINT_INVALID_STREAK_REQUIRED
            elif rank_failed:
                self._invalid_streak += 1
            else:
                self._invalid_streak = 0

            coint_invalid = self._invalid_streak >= COINT_INVALID_STREAK_REQUIRED
        elif gross_exp <= 0.01:
            self._invalid_streak = 0

        # Use the top signal's halflife for holding cap calculation if flat, or keep it 50
        active_halflife = live_signals[0].get("halflife", 50.0) if live_signals else 50.0

        force_exit = False
        if gross_exp > 0.01:
            force_exit = self.risk_engine.check_forced_exit(
                self._w_prev, holding_bars, halflife=active_halflife, coint_invalid=coint_invalid,
                pnl_since_entry=pnl_since_entry
            )

        w_new = self._w_prev.copy()

        if force_exit and gross_exp > 0.01:
            w_new = np.zeros(self.N)
        elif gross_exp > 0.01 and abs(z_score) < self.risk_engine.long_exit_threshold:
            w_new = np.zeros(self.N)
        elif self._zero_rank_days > ZERO_RANK_GRACE and gross_exp > 0.01:
            w_new = np.zeros(self.N)
        elif gross_exp < 0.01 and live_signals and abs(z_score) >= ENTRY_THRESHOLD:
            alpha = self.risk_engine.compute_alpha(live_signals, self.N, current_position=self._w_prev)

            if np.abs(alpha).max() > 1e-10:
                signal_returns = np.diff(log_window, axis=0)
                cov_matrix = np.cov(signal_returns.T)
                shrinkage = 0.2
                cov_matrix = (1 - shrinkage) * cov_matrix + shrinkage * np.diag(np.diag(cov_matrix))
                cov_matrix = (cov_matrix + cov_matrix.T) / 2.0
                cov_matrix += 1e-7 * np.eye(self.N)

                w_new = self.risk_engine.optimize(alpha, cov_matrix, self._w_prev, current_adv, current_vols)

        new_gross = np.abs(w_new).sum()
        if gross_exp < 0.01 and new_gross > 0.01:
            self._active_pair = live_signals[0].get("pair") if live_signals else None
            self._active_beta = live_signals[0].get("beta_full") if live_signals else None
            self._invalid_streak = 0
        elif new_gross < 0.01:
            self._active_pair = None
            self._active_beta = None
            self._invalid_streak = 0

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
