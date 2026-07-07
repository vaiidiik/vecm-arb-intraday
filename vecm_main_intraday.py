import time
import os
import logging
import numpy as np
import pandas as pd

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
ZERO_RANK_GRACE = 10

BARS_PER_DAY = 27
TRADING_DAYS_PER_YEAR = 252
PERIODS_PER_YEAR = BARS_PER_DAY * TRADING_DAYS_PER_YEAR

LOOKBACK = 60 * BARS_PER_DAY
COV_WINDOW = 10 * BARS_PER_DAY
VOL_WINDOW = 10 * BARS_PER_DAY
SPREAD_WINDOW = 20 * BARS_PER_DAY
REFIT_INTERVAL = BARS_PER_DAY

EXIT_CRITICAL_IDX = 0
COINT_INVALID_STREAK_REQUIRED = 1
STABILITY_WINDOW = 10 * BARS_PER_DAY


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
    returns_all = returns_df.values
    dates = prices_df.index

    print(f"Assets: {assets}")
    print(f"Total bars: {T}, Lookback: {LOOKBACK}, Trading bars: {T - LOOKBACK}")

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
        gamma=0.05,
        entry_threshold=ENTRY_THRESHOLD,
        exit_threshold=EXIT_THRESHOLD,
        short_exit_threshold=EXIT_THRESHOLD,
        long_exit_threshold=EXIT_THRESHOLD,
        turnover_penalty=0.0005,
        max_leverage=1.5,
        max_weight_per_asset=0.45,
        target_fraction=0.8,
        volatility_threshold=0.50,
        periods_per_year=PERIODS_PER_YEAR,
        capital_per_trade_frac=1.0,
    )

    backtester = PortfolioBacktester(
        initial_capital=INITIAL_CAPITAL,
        gamma=0.0,
        maker_taker_fee=0.0,
        periods_per_year=PERIODS_PER_YEAR,
        impact_gamma=0.0,
    )

    w = np.zeros(N)
    cached_signals = []
    zero_rank_streak = 0
    holding_start = None
    entry_capital = None
    active_halflife = 50.0
    active_pair = None
    active_beta = None
    invalid_streak = 0
    exit_log = []

    print("Starting backtest...")
    t0 = time.time()
    refit_count = 0

    for t in range(LOOKBACK, T):
        date = dates[t]
        p_now = prices_all[t]
        p_prev = prices_all[t - 1]
        adv_now = adv_all[t]
        vol_now = vols_all[t]

        bars_since_start = t - LOOKBACK

        coint_invalid = False
        coint_blowup = False

        if bars_since_start % REFIT_INTERVAL == 0 or not cached_signals:
            log_window = log_prices_all[t - LOOKBACK:t]
            cached_signals = math_engine.generate_all_signals(log_window, N)
            refit_count += 1

            if np.abs(w).sum() > 0.01:
                if active_pair is not None:
                    pair_lp = log_window[:, list(active_pair)]
                    pair_beta = active_beta[list(active_pair)] if active_beta is not None else None
                    pair_rank, _, _, _ = math_engine.cointegrate(pair_lp, critical_idx_override=EXIT_CRITICAL_IDX)
                    rank_failed = (pair_rank == 0)
                    stability_ok = math_engine.check_cointegration_stability(pair_lp, pair_beta, window=STABILITY_WINDOW)
                else:
                    full_rank, _, _, _ = math_engine.cointegrate(log_window, critical_idx_override=EXIT_CRITICAL_IDX)
                    rank_failed = (full_rank == 0)
                    stability_ok = math_engine.check_cointegration_stability(log_window, active_beta, window=STABILITY_WINDOW)

                coint_blowup = not stability_ok
                if coint_blowup:
                    invalid_streak = COINT_INVALID_STREAK_REQUIRED
                elif rank_failed:
                    invalid_streak += 1
                else:
                    invalid_streak = 0

                coint_invalid = invalid_streak >= COINT_INVALID_STREAK_REQUIRED

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
                "score": abs(z) / max(halflife_sig, 1.0),
            })

        live_signals.sort(key=lambda x: x["score"], reverse=True)

        if live_signals:
            z_primary = live_signals[0]["z"]
            rank_primary = live_signals[0]["rank"]
            zero_rank_streak = 0
        else:
            z_primary = 0.0
            rank_primary = 0
            zero_rank_streak += 1

        gross_exp = np.abs(w).sum()

        if gross_exp > 0.01:
            if entry_capital is None:
                entry_capital = backtester.capital
            pnl_since_entry = (backtester.capital - entry_capital) / entry_capital if entry_capital > 0 else 0.0
            holding_bars = t - holding_start if holding_start is not None else 0
        else:
            pnl_since_entry = 0.0
            holding_bars = 0

        force_exit = False
        if gross_exp > 0.01:
            force_exit = risk_engine.check_forced_exit(
                w, holding_bars, halflife=active_halflife, coint_invalid=coint_invalid,
                pnl_since_entry=pnl_since_entry
            )

        w_new = w.copy()
        exit_reason = None
        entry_reason = None

        if force_exit and gross_exp > 0.01:
            w_new = np.zeros(N)
            if coint_blowup:
                exit_reason = "coint_blowup"
            elif coint_invalid:
                exit_reason = "coint_persistent_invalid"
            elif (holding_bars >= risk_engine.stale_loss_holding_bars
                  and pnl_since_entry < risk_engine.stale_loss_threshold):
                exit_reason = "stale_loss_stop"
            else:
                exit_reason = "halflife_cap"

        elif gross_exp > 0.01 and abs(z_primary) < EXIT_THRESHOLD:
            w_new = np.zeros(N)
            exit_reason = "z_decay"

        elif zero_rank_streak > ZERO_RANK_GRACE and gross_exp > 0.01:
            w_new = np.zeros(N)
            exit_reason = "zero_rank_streak"

        elif gross_exp < 0.01 and live_signals and abs(z_primary) >= ENTRY_THRESHOLD:
            alpha = risk_engine.compute_alpha(live_signals, N, current_position=w)

            if np.abs(alpha).max() > 1e-10:
                cov_start = max(0, t - COV_WINDOW)
                ret_window = returns_all[cov_start:t]
                if ret_window.shape[0] < N + 5:
                    cov = np.eye(N) * 0.001
                else:
                    cov = np.cov(ret_window, rowvar=False)
                    shrinkage = 0.2
                    cov = (1 - shrinkage) * cov + shrinkage * np.diag(np.diag(cov))
                    cov = (cov + cov.T) / 2.0
                    cov += 1e-7 * np.eye(N)

                w_new = risk_engine.optimize(alpha, cov, w, adv_now, vol_now)
                if np.abs(w_new).sum() > 0.01:
                    entry_reason = "signal_entry"

        entry_capital_before = entry_capital

        new_gross = np.abs(w_new).sum()
        if gross_exp < 0.01 and new_gross > 0.01:
            holding_start = t
            entry_capital = backtester.capital
            active_halflife = live_signals[0].get("halflife", 50.0) if live_signals else 50.0
            active_pair = live_signals[0].get("pair") if live_signals else None
            active_beta = live_signals[0].get("beta_full") if live_signals else None
            invalid_streak = 0
        elif new_gross < 0.01:
            holding_start = None
            entry_capital = None
            active_halflife = 50.0
            active_pair = None
            active_beta = None
            invalid_streak = 0

        borrow = _borrow_rates(assets, adv_now, vol_now)

        backtester.process_day(
            date, w, w_new, p_prev, p_now, adv_now, vol_now,
            z_primary, rank_primary, borrow
        )

        if exit_reason is not None and entry_capital_before:
            realized_pnl = (backtester.capital - entry_capital_before) / entry_capital_before
            exit_log.append({
                "date": date, "event": "exit", "reason": exit_reason,
                "holding_bars": holding_bars, "pnl_since_entry": realized_pnl,
            })
        if entry_reason is not None:
            exit_log.append({
                "date": date, "event": "entry", "reason": entry_reason,
                "holding_bars": 0, "pnl_since_entry": 0.0,
            })

        w = w_new.copy()

        if bars_since_start > 0 and bars_since_start % (BARS_PER_DAY * 50) == 0:
            elapsed = time.time() - t0
            pct = bars_since_start / (T - LOOKBACK) * 100
            print(f"  {pct:.0f}% done ({bars_since_start}/{T - LOOKBACK}) | "
                  f"Capital: ${backtester.capital:,.0f} | "
                  f"Trades: {backtester.trade_count} | "
                  f"Time: {elapsed:.1f}s")

    elapsed = time.time() - t0
    print(f"\nBacktest complete in {elapsed:.1f}s")
    print(f"Refits: {refit_count}")

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
