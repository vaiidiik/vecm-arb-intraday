import time
import os
import logging
import numpy as np
import pandas as pd
import pandas_market_calendars as mcal

from vecm_model_intraday import vecm
from risk_intraday import dynamic_risk_engine
from backtest_intraday import PortfolioBacktester

logging.basicConfig(
    level=logging.WARNING,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)

PREFERRED_ASSETS = ["JPM", "BAC", "C", "WFC", "GS", "MS"]
INITIAL_CAPITAL = 1_000_000
ENTRY_THRESHOLD = 1.7
EXIT_THRESHOLD = 0.3

BARS_PER_DAY = 27
TRADING_DAYS_PER_YEAR = 252
PERIODS_PER_YEAR = BARS_PER_DAY * TRADING_DAYS_PER_YEAR

LOOKBACK = 60 * BARS_PER_DAY
COV_WINDOW = 10 * BARS_PER_DAY
VOL_WINDOW = 10 * BARS_PER_DAY
SPREAD_WINDOW = 20 * BARS_PER_DAY
REFIT_INTERVAL = BARS_PER_DAY

NUM_SLOTS = 2
SLOT_CAPITAL_FRAC = 0.65

ENABLE_RANK_DROP_EXIT = False
ENABLE_COINT_STABILITY_EXIT = False
RANK_DROP_CONFIRMATIONS = 5
COINT_STABILITY_CONFIRMATIONS = 2

GAP_HOURS_THRESHOLD = 6.0
HALT_STALE_BARS = 2
MAKER_TAKER_FEE = 0.00025

EARLY_CLOSE_MAX_SESSION_HOURS = 6.4
EARLY_CLOSE_CALENDAR_START = "2015-01-01"
EARLY_CLOSE_CALENDAR_END = "2035-12-31"


def _build_early_close_cutoffs():
    nyse = mcal.get_calendar("NYSE")
    sched = nyse.schedule(start_date=EARLY_CLOSE_CALENDAR_START, end_date=EARLY_CLOSE_CALENDAR_END)
    session_hours = (sched["market_close"] - sched["market_open"]).dt.total_seconds() / 3600.0
    early = sched[session_hours < EARLY_CLOSE_MAX_SESSION_HOURS]
    close_et = early["market_close"].dt.tz_convert("US/Eastern")
    return {ts.date(): ts.time() for ts in close_et}


EARLY_CLOSE_CUTOFFS = _build_early_close_cutoffs()


def _is_post_early_close(timestamp):
    ts = pd.Timestamp(timestamp)
    ts_et = ts.tz_convert("US/Eastern") if ts.tzinfo is not None else ts.tz_localize("UTC").tz_convert("US/Eastern")
    cutoff = EARLY_CLOSE_CUTOFFS.get(ts_et.date())
    if cutoff is None:
        return False
    return ts_et.time() > cutoff


def _borrow_rates(assets, adv, vols):
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
    return annual / PERIODS_PER_YEAR


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


def _update_halt_state(stale_streak, p_prev, p_now, timestamp):
    if _is_post_early_close(timestamp):
        return np.zeros_like(stale_streak), np.zeros_like(stale_streak, dtype=bool)
    unchanged = np.isclose(p_now, p_prev, rtol=0.0, atol=1e-12)
    stale_streak = np.where(unchanged, stale_streak + 1, 0)
    halted_mask = stale_streak >= HALT_STALE_BARS
    return stale_streak, halted_mask


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


def generate_weight_trajectory(prices_df, adv_df, assets, exit_log=None):
    prices_df = prices_df[assets]
    adv_df = adv_df[assets]
    returns_df = prices_df.pct_change().fillna(0)
    vols_df = returns_df.rolling(window=VOL_WINDOW).std().fillna(0.01).shift(1).fillna(0.01)
    N = len(assets)
    T = len(prices_df)

    log_prices_all = np.log(prices_df.values.clip(min=1e-8))
    prices_all = prices_df.values
    adv_all = adv_df.values
    vols_all = vols_df.values
    dates = prices_df.index

    math_engine = vecm(
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

    risk_engine = dynamic_risk_engine(
        num_assets=N,
        aum=INITIAL_CAPITAL,
        gamma=0.025,
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

    slots = [_new_slot(N) for _ in range(NUM_SLOTS)]
    stale_streak = np.zeros(N)
    cached_signals = []

    for t in range(LOOKBACK, T):
        date = dates[t]
        p_now = prices_all[t]
        p_prev = prices_all[t - 1]
        adv_now = adv_all[t]
        vol_now = vols_all[t]

        bars_since_start = t - LOOKBACK
        gap_hours = _gap_hours(date, dates[t - 1] if t > 0 else None)
        is_gap_bar = gap_hours >= GAP_HOURS_THRESHOLD

        stale_streak, halted_mask = _update_halt_state(stale_streak, p_prev, p_now, date)

        if bars_since_start % REFIT_INTERVAL == 0 or not cached_signals:
            log_window = log_prices_all[t - LOOKBACK:t]
            cached_signals = math_engine.generate_all_signals(log_window, N)

            for slot in slots:
                if np.abs(slot["w"]).sum() <= 0.01:
                    continue
                if slot["active_pair"] is not None:
                    i, j = slot["active_pair"]
                    pair_rank_now, _, _, _ = math_engine.cointegrate(log_window[:, [i, j]])
                    rank_now = pair_rank_now
                else:
                    rank_full_now, _, _, _ = math_engine.cointegrate(log_window)
                    rank_now = rank_full_now

                if rank_now == 0:
                    slot["_rank_zero_streak"] = slot.get("_rank_zero_streak", 0) + 1
                else:
                    slot["_rank_zero_streak"] = 0
                slot["_coint_broken"] = slot["_rank_zero_streak"] >= RANK_DROP_CONFIRMATIONS

        spread_start = max(0, t - SPREAD_WINDOW)
        log_recent = log_prices_all[spread_start:t + 1]

        live_signals = []
        for sig in cached_signals:
            bv = sig["beta_full"]
            spread = log_recent @ bv
            if len(spread) < 20:
                continue
            z, mu, sigma = math_engine.compute_spread_z(spread, lookback=SPREAD_WINDOW)
            rsi_arr = math_engine.compute_rsi(spread)
            rsi = float(rsi_arr[-1]) if len(rsi_arr) > 0 else 50.0
            _, _, macd_hist = math_engine.compute_macd(spread)

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
            z_primary = live_signals[0]["z"]
            rank_primary = live_signals[0]["rank"]
        else:
            z_primary = 0.0
            rank_primary = 0

        total_w_prev = sum((slot["w"] for slot in slots), np.zeros(N))

        for slot in slots:
            slot["_was_flat"] = np.abs(slot["w"]).sum() < 0.01

        cov_start = max(0, t - COV_WINDOW)
        cov_now = risk_engine.covariance_from_prices(prices_all[cov_start:t + 1])

        slot_quality = {}

        for slot_idx, slot in enumerate(slots):
            gross_s = np.abs(slot["w"]).sum()
            if gross_s <= 0.01:
                continue

            slot_halted = halted_mask & (np.abs(slot["w"]) > 1e-6)
            if np.any(slot_halted):
                direction = -np.sign(slot["entry_z"])
                alpha_rehedge = direction * slot["active_beta"]
                frozen_values = np.where(slot_halted, slot["w"], 0.0)
                slot["w"] = risk_engine.optimize(
                    alpha_rehedge, cov_now, slot["w"],
                    adv=adv_now, vols=vol_now, capital_frac=gross_s,
                    frozen_mask=slot_halted, frozen_values=frozen_values,
                )
                if exit_log is not None:
                    exit_log.append({
                        "date": date, "slot": slot_idx, "event": "halt_rehedge", "reason": "asset_halted",
                        "holding_bars": t - slot["holding_start"], "pnl_since_entry": _slot_pnl(slot, p_now),
                        "entry_z": slot.get("entry_z"), "exit_z": None,
                    })
                continue

            holding_bars = t - slot["holding_start"]
            pnl_since_entry = _slot_pnl(slot, p_now)
            bar_ret = _slot_bar_return(slot, p_prev, p_now)
            z_slot = _slot_z(slot["active_beta"], log_recent, math_engine, SPREAD_WINDOW)

            if ENABLE_COINT_STABILITY_EXIT:
                if not math_engine.check_cointegration_stability(log_recent, slot["active_beta"]):
                    slot["_coint_break_streak"] = slot.get("_coint_break_streak", 0) + 1
                else:
                    slot["_coint_break_streak"] = 0
            else:
                slot["_coint_break_streak"] = 0

            exit_reason = None
            if ENABLE_RANK_DROP_EXIT and slot.get("_coint_broken"):
                exit_reason = "rank_dropped"
            elif ENABLE_COINT_STABILITY_EXIT and slot.get("_coint_break_streak", 0) >= COINT_STABILITY_CONFIRMATIONS:
                exit_reason = "cointegration_breakdown"
            elif abs(z_slot) < EXIT_THRESHOLD:
                exit_reason = "z_decay"
            elif risk_engine.check_gap_shock(bar_ret, is_gap_bar):
                exit_reason = "gap_shock"

            if exit_reason is not None:
                if exit_log is not None:
                    exit_log.append({
                        "date": date, "slot": slot_idx, "event": "exit", "reason": exit_reason,
                        "holding_bars": holding_bars, "pnl_since_entry": pnl_since_entry,
                        "entry_z": slot.get("entry_z"), "exit_z": z_slot,
                    })
                _reset_slot(slot, N)
            else:
                slot_quality[slot_idx] = {
                    "score": abs(z_slot) / max(slot["active_halflife"], 1.0),
                    "z": z_slot,
                    "holding_bars": holding_bars,
                }

        used_signatures = set()
        for slot in slots:
            if np.abs(slot["w"]).sum() > 0.01 and slot["active_beta"] is not None:
                used_signatures.add(tuple(np.round(slot["active_beta"], 6)))

        for slot_idx, slot in enumerate(slots):
            if not slot["_was_flat"] or np.abs(slot["w"]).sum() > 0.01:
                continue

            chosen = None
            for sig in live_signals:
                sig_key = tuple(np.round(sig["beta_full"], 6))
                if sig_key in used_signatures:
                    continue
                alpha_candidate = risk_engine.compute_alpha([sig], N)
                if np.abs(alpha_candidate).max() < 1e-10:
                    continue

                other_gross = sum(
                    np.abs(slots[j]["w"]).sum() for j in range(NUM_SLOTS) if j != slot_idx
                )
                available_frac = max(0.0, risk_engine.max_leverage - other_gross)
                frac = risk_engine.conviction_capital_frac(sig["z"], SLOT_CAPITAL_FRAC, available_frac)
                if frac <= 0.0:
                    continue
                w_candidate = risk_engine.optimize(
                    alpha_candidate, cov_now, np.zeros(N),
                    adv=adv_now, vols=vol_now, capital_frac=frac,
                )
                if np.abs(w_candidate).sum() > 0.01:
                    chosen = (sig, w_candidate, frac)
                    break

            if chosen is not None:
                sig, w_new, frac = chosen
                _enter_slot(slot, sig, w_new, t, p_now)
                used_signatures.add(tuple(np.round(sig["beta_full"], 6)))
                if exit_log is not None:
                    exit_log.append({
                        "date": date, "slot": slot_idx, "event": "entry", "reason": "signal_entry",
                        "holding_bars": 0, "pnl_since_entry": 0.0,
                        "entry_z": sig["z"], "exit_z": None, "capital_frac": round(frac, 4),
                    })

        all_full = sum(np.abs(slot["w"]).sum() > 0.01 for slot in slots) == NUM_SLOTS
        if all_full and slot_quality:
            weakest_idx = min(slot_quality, key=lambda i: slot_quality[i]["score"])
            weakest = slot_quality[weakest_idx]

            for sig in live_signals:
                sig_key = tuple(np.round(sig["beta_full"], 6))
                if sig_key in used_signatures:
                    continue
                if not risk_engine.should_preempt(sig, weakest["score"], weakest["holding_bars"]):
                    continue
                alpha_candidate = risk_engine.compute_alpha([sig], N)
                if np.abs(alpha_candidate).max() < 1e-10:
                    continue
                other_gross = sum(
                    np.abs(slots[j]["w"]).sum() for j in range(NUM_SLOTS) if j != weakest_idx
                )
                available_frac = max(0.0, risk_engine.max_leverage - other_gross)
                frac = risk_engine.conviction_capital_frac(sig["z"], SLOT_CAPITAL_FRAC, available_frac)
                if frac <= 0.0:
                    continue

                w_candidate = risk_engine.optimize(
                    alpha_candidate, cov_now, slots[weakest_idx]["w"],
                    adv=adv_now, vols=vol_now, capital_frac=frac,
                )
                if np.abs(w_candidate).sum() <= 0.01:
                    continue

                slot = slots[weakest_idx]
                evicted_pnl = _slot_pnl(slot, p_now)
                if exit_log is not None:
                    exit_log.append({
                        "date": date, "slot": weakest_idx, "event": "exit", "reason": "preempted",
                        "holding_bars": weakest["holding_bars"], "pnl_since_entry": evicted_pnl,
                        "entry_z": slot.get("entry_z"), "exit_z": weakest["z"],
                    })

                _enter_slot(slot, sig, w_candidate, t, p_now)
                used_signatures.add(sig_key)
                if exit_log is not None:
                    exit_log.append({
                        "date": date, "slot": weakest_idx, "event": "entry", "reason": "preempt_entry",
                        "holding_bars": 0, "pnl_since_entry": 0.0,
                        "entry_z": sig["z"], "exit_z": None, "capital_frac": round(frac, 4),
                    })
                break

        total_w_new = sum((slot["w"] for slot in slots), np.zeros(N))
        borrow = _borrow_rates(assets, adv_now, vol_now)

        yield {
            "date": date, "w_prev": total_w_prev, "w_new": total_w_new,
            "prices_prev": p_prev, "prices_new": p_now,
            "adv": adv_now, "vols": vol_now,
            "z_score": z_primary, "rank": rank_primary, "borrow": borrow,
        }


def main():
    script_dir = os.path.dirname(os.path.abspath(__file__))
    prices_path = os.path.join(script_dir, "sample_prices_intraday.csv")
    adv_path = os.path.join(script_dir, "sample_dollar_adv_intraday.csv")

    print("Loading data...")
    prices_df = pd.read_csv(prices_path, index_col="Date", parse_dates=True)
    adv_df = pd.read_csv(adv_path, index_col="Date", parse_dates=True)

    assets = [a for a in PREFERRED_ASSETS if a in prices_df.columns]
    if len(assets) < 2:
        raise ValueError("Need at least two available assets.")

    T = len(prices_df)
    print(f"Assets: {assets}")
    print(f"Total bars: {T}, Lookback: {LOOKBACK}, Trading bars: {T - LOOKBACK}")

    backtester = PortfolioBacktester(
        initial_capital=INITIAL_CAPITAL,
        gamma=0.10,
        maker_taker_fee=MAKER_TAKER_FEE,
        periods_per_year=PERIODS_PER_YEAR,
    )

    exit_log = []
    print("Starting backtest...")
    t0 = time.time()
    bars_done = 0

    for record in generate_weight_trajectory(prices_df, adv_df, assets, exit_log=exit_log):
        backtester.process_day(
            record["date"], record["w_prev"], record["w_new"],
            record["prices_prev"], record["prices_new"],
            record["adv"], record["vols"],
            record["z_score"], record["rank"], record["borrow"],
        )
        bars_done += 1
        if bars_done % (BARS_PER_DAY * 50) == 0:
            elapsed = time.time() - t0
            pct = bars_done / (T - LOOKBACK) * 100
            print(f"  {pct:.0f}% done ({bars_done}/{T - LOOKBACK}) | "
                  f"Capital: ${backtester.capital:,.0f} | "
                  f"Trades: {backtester.trade_count} | "
                  f"Time: {elapsed:.1f}s")

    elapsed = time.time() - t0
    print(f"\nBacktest complete in {elapsed:.1f}s")
    print(f"Refits: {(T - LOOKBACK) // REFIT_INTERVAL + 1}")

    out_daily = os.path.join(script_dir, "backtest_intraday.csv")
    out_metrics = os.path.join(script_dir, "performance_metrics_intraday.csv")
    out_exits = os.path.join(script_dir, "exit_reasons_intraday.csv")

    results_df = backtester.get_results_df()
    results_df.to_csv(out_daily, index=False)
    print(f"Saved {len(results_df)} rows to backtest_intraday.csv")

    exit_log_df = pd.DataFrame(exit_log)
    exit_log_df.to_csv(out_exits, index=False)
    print(f"Saved {len(exit_log_df)} entry/exit events to exit_reasons_intraday.csv")

    metrics = backtester.compute_metrics()
    metrics_list = [{"metric": k, "value": v} for k, v in metrics.items()]
    metrics_df = pd.DataFrame(metrics_list)
    metrics_df.to_csv(out_metrics, index=False)

    print("\n=== Performance Metrics ===")
    for k, v in metrics.items():
        print(f"  {k:20s}: {v}")


if __name__ == "__main__":
    main()