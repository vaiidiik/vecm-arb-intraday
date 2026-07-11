import asyncio
import logging
import numpy as np
import pandas as pd
from concurrent.futures import ThreadPoolExecutor

from vecm_model_intraday import vecm
from risk_intraday import dynamic_risk_engine
from backtest_intraday import PortfolioBacktester

BARS_PER_DAY = 27
TRADING_DAYS_PER_YEAR = 252
PERIODS_PER_YEAR = BARS_PER_DAY * TRADING_DAYS_PER_YEAR

# Matched to vecm_main_intraday.py -- was 200*BARS_PER_DAY here, which meant
# the "live" engine was fitting cointegration on a ~200-day window while the
# offline backtest used 60 days. Same strategy needs the same window.
LOOKBACK = 60 * BARS_PER_DAY
COV_WINDOW = 10 * BARS_PER_DAY
SPREAD_WINDOW = 20 * BARS_PER_DAY

ENTRY_THRESHOLD = 1.7
EXIT_THRESHOLD = 0.3

NUM_SLOTS = 2
SLOT_CAPITAL_FRAC = 0.5

ENABLE_RANK_DROP_EXIT = True
ENABLE_COINT_STABILITY_EXIT = True
# Matched to vecm_main_intraday.py -- was 2 here vs 5 there, meaning the live
# engine would force-liquidate on a rank drop 2.5x faster than the backtest
# it's supposed to mirror.
RANK_DROP_CONFIRMATIONS = 5
COINT_STABILITY_CONFIRMATIONS = 2

GAP_HOURS_THRESHOLD = 6.0

def _borrow_rates(assets, adv, vols, periods_per_year=PERIODS_PER_YEAR):
    base_annual = {
        "JPM": 0.0030, "BAC": 0.0035, "C": 0.0050,
        "WFC": 0.0040, "GS": 0.0035, "MS": 0.0045,
    }
    annual = np.array([base_annual.get(a, 0.0060) for a in assets], dtype=float)
    vol_base = max(np.nanmedian(vols), 1e-4)
    adv_base = max(np.nanmedian(adv), 1.0)
    vol_stress = np.clip(vols / vol_base - 1.0, 0.0, 3.0)
    liquidity_stress = np.clip(adv_base / (adv + 1e-8) - 1.0, 0.0, 3.0)
    annual = annual * (1.0 + 0.25 * vol_stress + 0.15 * liquidity_stress)
    return annual / periods_per_year

def _slot_z(beta, log_recent, math_engine, window):
    if beta is None:
        return 0.0
    spread = log_recent @ beta
    if len(spread) < 20:
        return 0.0
    z, _, _ = math_engine.compute_spread_z(spread, lookback=window)
    return z

def _new_slot(N):
    return {
        "w": np.zeros(N),
        "holding_start": None,
        "entry_prices": None,
        "entry_w": None,
        "entry_z": None,
        "active_halflife": 50.0,
        "active_pair": None,
        "active_beta": None,
        "_was_flat": True,
        "_coint_broken": False,
        "_rank_zero_streak": 0,
        "_coint_break_streak": 0,
    }

def _slot_pnl(slot, p_now):
    if slot["entry_prices"] is None:
        return 0.0
    safe_entry = np.where(slot["entry_prices"] > 0, slot["entry_prices"], 1.0)
    price_ret = (p_now - slot["entry_prices"]) / safe_entry
    return float(np.dot(slot["entry_w"], price_ret))

def _slot_bar_return(slot, p_prev, p_now):
    if slot["w"] is None:
        return 0.0
    safe_prev = np.where(p_prev > 0, p_prev, 1.0)
    return float(np.dot(slot["w"], (p_now - p_prev) / safe_prev))

def _gap_hours(ts_now, ts_prev):
    if ts_prev is None:
        return 0.0
    try:
        return (ts_now - ts_prev).total_seconds() / 3600.0
    except (TypeError, AttributeError):
        return 0.0

def _reset_slot(slot, N):
    slot["w"] = np.zeros(N)
    slot["holding_start"] = None
    slot["entry_prices"] = None
    slot["entry_w"] = None
    slot["entry_z"] = None
    slot["active_halflife"] = 50.0
    slot["active_pair"] = None
    slot["active_beta"] = None
    slot["_coint_broken"] = False
    slot["_rank_zero_streak"] = 0
    slot["_coint_break_streak"] = 0

def _enter_slot(slot, sig, w_new, t, p_now):
    slot["w"] = w_new
    slot["holding_start"] = t
    slot["entry_prices"] = p_now.copy()
    slot["entry_w"] = w_new.copy()
    slot["entry_z"] = sig["z"]
    slot["active_halflife"] = sig.get("halflife", 50.0)
    slot["active_pair"] = sig.get("pair")
    slot["active_beta"] = sig.get("beta_full")
    slot["_coint_broken"] = False
    slot["_rank_zero_streak"] = 0
    slot["_coint_break_streak"] = 0

class VECMStrategyEngine:
    def __init__(self, assets, initial_capital=1_000_000, impact_gamma=0.10):
        self.assets = list(assets)
        self.N = len(self.assets)
        self.logger = logging.getLogger("VECM_ARB.Engine")

        self._executor = ThreadPoolExecutor(
            max_workers=1, thread_name_prefix="vecm_worker"
        )

        # Matched to vecm_main_intraday.py's math_engine exactly. significance
        # (was 0.01, now 0.05) changes which Johansen critical-value column is
        # used, and k_ar_diff (was 7, now 3) changes the VAR lag order fed
        # into coint_johansen -- both actually change the test's output, not
        # just cosmetic. min_kappa/max_kappa/delay_span are accepted by `vecm`
        # but never referenced anywhere in vecm_model_intraday.py (dead
        # parameters); kept here only so the two constructor calls read
        # identically side by side.
        self.math_engine = vecm(
            significance=0.05,
            entry_threshold=ENTRY_THRESHOLD,
            exit_threshold=EXIT_THRESHOLD,
            min_kappa=1.0,
            max_kappa=100.0,
            k_ar_diff=3,
            periods_per_year=PERIODS_PER_YEAR,
            rsi_period=14,
            use_pairwise=True,
            zscore_lookback=SPREAD_WINDOW,
        )

        self.risk_engine = dynamic_risk_engine(
            num_assets=self.N,
            aum=initial_capital,
            gamma=0.05,
            entry_threshold=ENTRY_THRESHOLD,
            exit_threshold=EXIT_THRESHOLD,
            short_exit_threshold=EXIT_THRESHOLD,
            long_exit_threshold=EXIT_THRESHOLD,
            turnover_penalty=0.0005,
            max_leverage=1.0,
            max_weight_per_asset=0.8,
            target_fraction=0.8,
            volatility_threshold=0.50,
            periods_per_year=PERIODS_PER_YEAR,

            max_entry_halflife=120,
            min_beta_confirmations=2,

            preempt_z_ratio=2.0,
            preempt_quality_margin=1.4,
            preempt_min_holding_bars=10,
            gap_shock_threshold=-0.004,
            enable_gap_shock=False,
        )
        # `impact_gamma` here now defaults to 0.10 to match vecm_main_intraday.py's
        # backtester gamma. run_strategy_intraday.py currently passes an explicit
        # IMPACT_GAMMA=0.05, which would still override this default -- update
        # that constant too if you want the two to actually match at runtime.
        self.backtester = PortfolioBacktester(
            initial_capital=initial_capital,
            gamma=impact_gamma,
            maker_taker_fee=0.0,
            periods_per_year=PERIODS_PER_YEAR,
        )

        self._historical_prices = []
        self._historical_returns = []
        self._last_timestamp = None
        self._slots = [_new_slot(self.N) for _ in range(NUM_SLOTS)]
        self._cached_signals = []
        self._bar_count = 0

        self._trade_log = []

    def _process_tick(self, timestamp, current_prices, current_adv, current_vols):
        self._historical_prices.append(current_prices.copy())

        gap_hours = _gap_hours(timestamp, self._last_timestamp)
        is_gap_bar = gap_hours >= GAP_HOURS_THRESHOLD
        self._last_timestamp = timestamp

        if len(self._historical_prices) > 1:
            prev = self._historical_prices[-2]
            ret = (current_prices - prev) / (prev + 1e-12)
            self._historical_returns.append(float(np.mean(ret)))

        if len(self._historical_prices) < LOOKBACK:
            return

        self._bar_count += 1
        t = self._bar_count

        rolling_window = np.array(self._historical_prices[-LOOKBACK:], dtype=float)
        log_window = np.log(rolling_window)
        borrow_rates = _borrow_rates(self.assets, current_adv, current_vols)

        log_window_hist = log_window[:-1]

        if not self._cached_signals or t % BARS_PER_DAY == 0:
            self._cached_signals = self.math_engine.generate_all_signals(log_window_hist, self.N)

            for s in self._slots:
                if np.abs(s["w"]).sum() <= 0.01:
                    continue
                if s["active_pair"] is not None:
                    i, j = s["active_pair"]
                    pair_rank_now, _, _, _ = self.math_engine.cointegrate(log_window_hist[:, [i, j]])
                    rank_now = pair_rank_now
                else:
                    rank_full_now, _, _, _ = self.math_engine.cointegrate(log_window_hist)
                    rank_now = rank_full_now

                if rank_now == 0:
                    s["_rank_zero_streak"] = s.get("_rank_zero_streak", 0) + 1
                else:
                    s["_rank_zero_streak"] = 0
                s["_coint_broken"] = s["_rank_zero_streak"] >= RANK_DROP_CONFIRMATIONS

        spread_start = max(0, len(log_window) - SPREAD_WINDOW)
        log_recent = log_window[spread_start:]

        live_signals = []
        for sig in self._cached_signals:
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

                "confirmations": sig.get("confirmations", 1),
                "score": abs(z) / max(halflife_sig, 1.0),
            })

        live_signals.sort(key=lambda x: x["score"], reverse=True)

        if live_signals:
            z_score = live_signals[0]["z"]
            rank = live_signals[0]["rank"]
        else:
            z_score = 0.0
            rank = 0

        w_prev_total = sum((s["w"] for s in self._slots), np.zeros(self.N))

        for s in self._slots:
            s["_was_flat"] = np.abs(s["w"]).sum() < 0.01

        slot_quality = {}

        for slot_idx, s in enumerate(self._slots):
            gross_s = np.abs(s["w"]).sum()
            if gross_s <= 0.01:
                continue

            holding_bars = t - s["holding_start"]
            pnl_since_entry = _slot_pnl(s, current_prices)
            bar_ret = _slot_bar_return(s, self._historical_prices[-2], current_prices)
            z_s = _slot_z(s["active_beta"], log_recent, self.math_engine, SPREAD_WINDOW)

            if ENABLE_COINT_STABILITY_EXIT:
                if not self.math_engine.check_cointegration_stability(log_recent, s["active_beta"]):
                    s["_coint_break_streak"] = s.get("_coint_break_streak", 0) + 1
                else:
                    s["_coint_break_streak"] = 0
            else:
                s["_coint_break_streak"] = 0

            exit_reason = None
            if ENABLE_RANK_DROP_EXIT and s.get("_coint_broken"):
                exit_reason = "rank_dropped"
            elif ENABLE_COINT_STABILITY_EXIT and s.get("_coint_break_streak", 0) >= COINT_STABILITY_CONFIRMATIONS:
                exit_reason = "cointegration_breakdown"
            elif abs(z_s) < EXIT_THRESHOLD:
                exit_reason = "z_decay"
            elif self.risk_engine.check_gap_shock(bar_ret, is_gap_bar):
                exit_reason = "gap_shock"

            if exit_reason is not None:
                self._trade_log.append({
                    "timestamp": timestamp, "slot": slot_idx, "event": "exit",
                    "reason": exit_reason, "holding_bars": holding_bars,
                    "pnl_since_entry": pnl_since_entry,
                    "entry_z": s.get("entry_z"), "exit_z": z_s,
                })
                _reset_slot(s, self.N)
            else:
                slot_quality[slot_idx] = {
                    "score": abs(z_s) / max(s["active_halflife"], 1.0),
                    "z": z_s,
                    "holding_bars": holding_bars,
                }

        cov_start = max(0, len(self._historical_prices) - COV_WINDOW)
        cov_now = self.risk_engine.covariance_from_prices(
            np.array(self._historical_prices[cov_start:], dtype=float)
        )

        used_signatures = set()
        for s in self._slots:
            if np.abs(s["w"]).sum() > 0.01 and s["active_beta"] is not None:
                used_signatures.add(tuple(np.round(s["active_beta"], 6)))

        for slot_idx, s in enumerate(self._slots):
            if not s["_was_flat"] or np.abs(s["w"]).sum() > 0.01:
                continue

            chosen = None
            for sig in live_signals:
                sig_key = tuple(np.round(sig["beta_full"], 6))
                if sig_key in used_signatures:
                    continue
                alpha_candidate = self.risk_engine.compute_alpha([sig], self.N)
                if np.abs(alpha_candidate).max() < 1e-10:
                    continue

                other_gross = sum(
                    np.abs(self._slots[j]["w"]).sum() for j in range(len(self._slots)) if j != slot_idx
                )
                available_frac = max(0.0, self.risk_engine.max_leverage - other_gross)
                frac = self.risk_engine.conviction_capital_frac(sig["z"], SLOT_CAPITAL_FRAC, available_frac)
                if frac <= 0.0:
                    continue
                w_candidate = self.risk_engine.optimize(
                    alpha_candidate, cov_now, np.zeros(self.N),
                    adv=current_adv, vols=current_vols, capital_frac=frac,
                )
                if np.abs(w_candidate).sum() > 0.01:
                    chosen = (sig, w_candidate, frac)
                    break

            if chosen is not None:
                sig, w_new, frac = chosen
                _enter_slot(s, sig, w_new, t, current_prices)
                used_signatures.add(tuple(np.round(sig["beta_full"], 6)))
                self._trade_log.append({
                    "timestamp": timestamp, "slot": slot_idx, "event": "entry",
                    "reason": "signal_entry", "holding_bars": 0,
                    "pnl_since_entry": 0.0, "entry_z": sig["z"], "exit_z": None,
                    "capital_frac": round(frac, 4),
                })

        all_full = sum(np.abs(s["w"]).sum() > 0.01 for s in self._slots) == NUM_SLOTS
        if all_full and slot_quality:
            weakest_idx = min(slot_quality, key=lambda i: slot_quality[i]["score"])
            weakest = slot_quality[weakest_idx]

            for sig in live_signals:
                sig_key = tuple(np.round(sig["beta_full"], 6))
                if sig_key in used_signatures:
                    continue
                if not self.risk_engine.should_preempt(sig, weakest["score"], weakest["holding_bars"]):
                    continue
                alpha_candidate = self.risk_engine.compute_alpha([sig], self.N)
                if np.abs(alpha_candidate).max() < 1e-10:
                    continue
                other_gross = sum(
                    np.abs(self._slots[j]["w"]).sum() for j in range(len(self._slots)) if j != weakest_idx
                )
                available_frac = max(0.0, self.risk_engine.max_leverage - other_gross)
                frac = self.risk_engine.conviction_capital_frac(sig["z"], SLOT_CAPITAL_FRAC, available_frac)
                if frac <= 0.0:
                    continue

                w_candidate = self.risk_engine.optimize(
                    alpha_candidate, cov_now, self._slots[weakest_idx]["w"],
                    adv=current_adv, vols=current_vols, capital_frac=frac,
                )
                if np.abs(w_candidate).sum() <= 0.01:
                    continue

                s = self._slots[weakest_idx]
                evicted_pnl = _slot_pnl(s, current_prices)
                self._trade_log.append({
                    "timestamp": timestamp, "slot": weakest_idx, "event": "exit",
                    "reason": "preempted", "holding_bars": weakest["holding_bars"],
                    "pnl_since_entry": evicted_pnl,
                    "entry_z": s.get("entry_z"), "exit_z": weakest["z"],
                })

                _enter_slot(s, sig, w_candidate, t, current_prices)
                used_signatures.add(sig_key)
                self._trade_log.append({
                    "timestamp": timestamp, "slot": weakest_idx, "event": "entry",
                    "reason": "preempt_entry", "holding_bars": 0,
                    "pnl_since_entry": 0.0, "entry_z": sig["z"], "exit_z": None,
                    "capital_frac": round(frac, 4),
                })
                break

        w_new_total = sum((s["w"] for s in self._slots), np.zeros(self.N))

        self.backtester.process_day(
            date=timestamp,
            w_prev=w_prev_total,
            w_new=w_new_total,
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
            f"active_slots={sum(np.abs(s['w']).sum() > 0.01 for s in self._slots)}/{NUM_SLOTS}  "
            f"exposure={np.sum(np.abs(w_new_total)):.2f}  "
            f"capital=${self.backtester.capital:,.0f}"
        )

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

    def save_results(self, daily_path, metrics_path, trade_log_path=None):
        self.backtester.save_results(daily_path, metrics_path)
        if trade_log_path is not None:
            pd.DataFrame(self._trade_log).to_csv(trade_log_path, index=False)
        _, metrics = self.backtester.generate_metrics()
        return metrics