import pandas as pd
import numpy as np
import os

from vecm_model_intraday import vecm
from risk_intraday import dynamic_risk_engine
from backtest_intraday import PortfolioBacktester

PREFERRED_ASSETS = ["NVDA", "AMD", "TSM", "ASML", "AVGO", "QCOM"]
ENTRY_THRESHOLD = 1.5
IMPACT_GAMMA = 0.05
ZERO_RANK_GRACE = 5

BARS_PER_DAY = 27
TRADING_DAYS_PER_YEAR = 252
PERIODS_PER_YEAR = BARS_PER_DAY * TRADING_DAYS_PER_YEAR

LOOKBACK = 200 * BARS_PER_DAY
VOL_WINDOW = 20 * BARS_PER_DAY
MAX_HOLDING_BARS = 25 * BARS_PER_DAY
TREND_WINDOW = 21 * BARS_PER_DAY


def estimate_borrow_rates(assets, adv, vols, periods_per_year=PERIODS_PER_YEAR):
    base_annual = {
        "NVDA": 0.0035, "AMD": 0.0060, "TSM": 0.0120,
        "ASML": 0.0040, "INTC": 0.0150, "TXN": 0.0045,
        "AVGO": 0.0045, "QCOM": 0.0060, "MU": 0.0100,
        "AMAT": 0.0050, "LRCX": 0.0050, "KLAC": 0.0050
    }
    annual = np.array([base_annual.get(asset, 0.0080) for asset in assets], dtype=float)
    vol_base = max(np.nanmedian(vols), 1e-4)
    adv_base = max(np.nanmedian(adv), 1.0)
    vol_stress = np.clip(vols / vol_base - 1.0, 0.0, 3.0)
    liquidity_stress = np.clip(adv_base / (adv + 1e-8) - 1.0, 0.0, 3.0)
    annual = annual * (1.0 + 0.25 * vol_stress + 0.15 * liquidity_stress)
    return annual / periods_per_year


def run_capacity_analysis():
    script_dir = os.path.dirname(os.path.abspath(__file__))
    prices_path = os.path.join(script_dir, "sample_prices_intraday.csv")
    adv_path = os.path.join(script_dir, "sample_dollar_adv_intraday.csv")
    prices = pd.read_csv(prices_path, index_col="Date", parse_dates=True)
    adv = pd.read_csv(adv_path, index_col="Date", parse_dates=True)
    available_assets = list(prices.columns)
    assets = [asset for asset in PREFERRED_ASSETS if asset in available_assets]

    if len(assets) < 2:
        assets = available_assets

    assets = assets[:12]
    prices = prices[assets]
    adv = adv[assets]
    n_assets = len(assets)
    returns = prices.pct_change().fillna(0)
    vols = returns.rolling(window=VOL_WINDOW).std().fillna(0.01).shift(1).fillna(0.01)
    aum_levels = [1000000, 10000000, 50000000, 100000000, 500000000]
    results = []

    for aum in aum_levels:
        math_engine = vecm(
            significance=0.01,
            entry_threshold=ENTRY_THRESHOLD,
            min_kappa=2.0,
            max_kappa=50.0,
            delay_span=5,
            k_ar_diff=7,
            periods_per_year=PERIODS_PER_YEAR
        )
        risk_engine = dynamic_risk_engine(
            num_assets=n_assets,
            aum=aum,
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
            trend_threshold=0.50
        )
        backtester = PortfolioBacktester(initial_capital=aum, gamma=IMPACT_GAMMA,
                                          periods_per_year=PERIODS_PER_YEAR)
        w_prev = np.zeros(n_assets)
        zero_rank_days = 0
        historical_prices = []
        historical_returns = []
        prev_z_score = 0.0

        for timestamp, row in prices.iterrows():
            current_prices = row.values.astype(float)
            current_adv = adv.loc[timestamp].values.astype(float)
            current_vols = vols.loc[timestamp].values.astype(float)
            historical_prices.append(current_prices)

            if len(historical_prices) > 1:
                ret = (current_prices - historical_prices[-2]) / historical_prices[-2]
                historical_returns.append(np.mean(ret))

            if len(historical_prices) < LOOKBACK:
                continue

            rolling_window = np.array(historical_prices[-LOOKBACK:], dtype=float)
            log_window = np.log(rolling_window)

            z_score = 0.0
            kappa = 0.0
            rsi_val = 50.0
            w_new = np.zeros(n_assets)
            borrow_rates = estimate_borrow_rates(assets, current_adv, current_vols)

            log_window_hist = log_window[:-1]
            rank, beta_set = math_engine.cointegrate(log_window_hist)

            if rank > 0:
                zero_rank_days = 0
                z_score, kappa, beta, rsi_val = math_engine.calculate_sscore(
                    log_window[-1], beta_set, log_window_hist
                )

                stable = math_engine.check_cointegration_stability(log_window, beta, window=10)
                if not stable:
                    w_new = np.zeros(n_assets)
                elif beta is not None and kappa > 0.0 and abs(z_score) <= 5.0:
                    signal_returns = np.diff(log_window, axis=0)
                    cov_matrix = np.cov(signal_returns.T)

                    market_vol = np.std(historical_returns[-TREND_WINDOW:-1]) if len(historical_returns) >= TREND_WINDOW else 0.0
                    trend_strength = np.mean(historical_returns[-TREND_WINDOW:-1]) * 20 if len(historical_returns) >= TREND_WINDOW else 0.0

                    if backtester.entry_day:
                        position_holding_days = max(backtester.days - d for d in backtester.entry_day.values())
                    else:
                        position_holding_days = 0

                    current_exposure = np.sum(np.abs(w_prev))
                    pnl_since_entry_now = backtester.position_pnl_sum() / backtester.capital if backtester.capital > 0 else 0.0
                    risk_breach = (
                        position_holding_days >= risk_engine.max_holding_days
                        or pnl_since_entry_now < -risk_engine.stop_loss_pct
                        or pnl_since_entry_now < -risk_engine.trailing_stop_pct
                    )
                    needs_rebalance = False
                    if current_exposure < 1e-4 and abs(z_score) >= ENTRY_THRESHOLD:
                        needs_rebalance = True
                    elif current_exposure > 1e-4 and (abs(z_score - prev_z_score) > 0.50 or risk_breach):
                        needs_rebalance = True
                    elif current_exposure > 1e-4:
                        w_new = w_prev

                    if needs_rebalance:
                        w_new = risk_engine.optimise_weights(
                            w_prev=w_prev,
                            cov_matrix=cov_matrix,
                            beta=beta,
                            z_score=z_score,
                            rsi=rsi_val,
                            adv=current_adv,
                            vols=current_vols,
                            borrow_rates=borrow_rates,
                            kappa=kappa,
                            holding_days=position_holding_days,
                            pnl_since_entry=backtester.position_pnl_sum() / backtester.capital,
                            market_vol=market_vol,
                            trend_strength=trend_strength,
                            rank=rank,
                            capital=backtester.capital
                        )

                    prev_z_score = z_score
            else:
                zero_rank_days += 1
                if zero_rank_days <= ZERO_RANK_GRACE and np.sum(np.abs(w_prev)) > 1e-4:
                    w_new = w_prev
                else:
                    w_new = np.zeros(n_assets)

            backtester.process_day(
                date=timestamp,
                w_prev=w_prev,
                w_new=w_new,
                prices_prev=historical_prices[-2],
                prices_new=historical_prices[-1],
                adv=current_adv,
                vols=current_vols,
                z_score=z_score,
                rank=rank,
                borrow_rates=borrow_rates
            )
            w_prev = w_new

        df, metrics = backtester.generate_metrics()
        cagr = metrics[metrics["metric"] == "cagr"]["value"].values[0] if not df.empty else 0.0
        sharpe = metrics[metrics["metric"] == "sharpe"]["value"].values[0] if not df.empty else 0.0
        results.append({
            "aum_millions": aum / 1000000,
            "cagr": cagr,
            "sharpe": sharpe
        })
        print(f"AUM ${aum/1e6:.0f}M  ->  cagr={cagr}  sharpe={sharpe}", flush=True)

    out_path = os.path.join(script_dir, "capacity_analysis_intraday.csv")
    pd.DataFrame(results).to_csv(out_path, index=False)
    print(f"saved -> {out_path}")


if __name__ == "__main__":
    run_capacity_analysis()