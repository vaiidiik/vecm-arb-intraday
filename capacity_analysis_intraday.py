import os
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from vecm_main_intraday import (
    generate_weight_trajectory,
    PREFERRED_ASSETS,
    PERIODS_PER_YEAR,
    MAKER_TAKER_FEE,
)
from backtest_intraday import PortfolioBacktester

IMPACT_GAMMA = 0.10

AUM_GRID = [
    1_000_000, 5_000_000, 10_000_000, 100_000_000,  1_000_000_000, 10_000_000_000,
]



def run_capacity_sweep(prices_df, adv_df, assets, aum_grid):
    trajectory = list(generate_weight_trajectory(prices_df, adv_df, assets))

    rows = []
    for aum in aum_grid:
        backtester = PortfolioBacktester(
            initial_capital=aum,
            gamma=IMPACT_GAMMA,
            maker_taker_fee=MAKER_TAKER_FEE,
            periods_per_year=PERIODS_PER_YEAR,
        )
        for record in trajectory:
            backtester.process_day(
                record["date"], record["w_prev"], record["w_new"],
                record["prices_prev"], record["prices_new"],
                record["adv"], record["vols"],
                record["z_score"], record["rank"], record["borrow"],
            )
        metrics = backtester.compute_metrics()
        cagr_pct = float(str(metrics.get("cagr", "0%")).rstrip("%"))
        max_dd_pct = float(str(metrics.get("max_dd", "0%")).rstrip("%"))
        rows.append({
            "aum": aum,
            "cagr_pct": cagr_pct,
            "sharpe": metrics.get("sharpe", 0.0),
            "max_dd_pct": max_dd_pct,
            "total_fees": metrics.get("total_fees", 0.0),
            "avg_turnover": metrics.get("avg_turnover", 0.0),
        })

    return pd.DataFrame(rows)


def find_capacity_ceiling(df):
    below_zero = df[df["cagr_pct"] <= 0.0]
    if below_zero.empty:
        return None
    return float(below_zero.iloc[0]["aum"])


def plot_capacity_curve(df, save_path):
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 8), sharex=True)

    ax1.plot(df["aum"], df["cagr_pct"], marker="o", color="#1f77b4")
    ax1.axhline(y=0, color="red", linestyle="--", alpha=0.7)
    ax1.set_xscale("log")
    ax1.set_ylabel("CAGR (%)")
    ax1.set_title("Capacity Analysis - CAGR vs AUM")
    ax1.grid(True, alpha=0.3)

    ax2.plot(df["aum"], df["sharpe"], marker="o", color="#2ca02c")
    ax2.axhline(y=0, color="red", linestyle="--", alpha=0.7)
    ax2.set_xscale("log")
    ax2.set_xlabel("AUM ($)")
    ax2.set_ylabel("Sharpe Ratio")
    ax2.set_title("Capacity Analysis - Sharpe vs AUM")
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(save_path, dpi=200, bbox_inches="tight")
    plt.close(fig)


def main():
    script_dir = os.path.dirname(os.path.abspath(__file__))
    prices_path = os.path.join(script_dir, "sample_prices_intraday.csv")
    adv_path = os.path.join(script_dir, "sample_dollar_adv_intraday.csv")

    prices_df = pd.read_csv(prices_path, index_col="Date", parse_dates=True)
    adv_df = pd.read_csv(adv_path, index_col="Date", parse_dates=True)

    assets = [a for a in PREFERRED_ASSETS if a in prices_df.columns]
    if len(assets) < 2:
        raise ValueError("Need at least two available assets.")

    print(f"Assets: {assets}")
    print(f"Running capacity sweep across {len(AUM_GRID)} AUM levels...")

    df = run_capacity_sweep(prices_df, adv_df, assets, AUM_GRID)

    out_csv = os.path.join(script_dir, "capacity_analysis_intraday.csv")
    df.to_csv(out_csv, index=False)
    print(f"Saved {out_csv}")

    ceiling = find_capacity_ceiling(df)
    if ceiling is not None:
        print(f"Capacity ceiling (CAGR crosses zero): ${ceiling:,.0f}")
    else:
        print("CAGR stays positive across the full AUM grid tested.")

    out_png = os.path.join(script_dir, "capacity_analysis_intraday.png")
    plot_capacity_curve(df, out_png)
    print(f"Saved {out_png}")

    print("\n=== Capacity Sweep ===")
    for _, row in df.iterrows():
        print(f"  AUM ${row['aum']:>15,.0f} | CAGR {row['cagr_pct']:>7.2f}% | "
              f"Sharpe {row['sharpe']:>6.2f} | MaxDD {row['max_dd_pct']:>6.2f}% | "
              f"Fees ${row['total_fees']:>12,.2f}")


if __name__ == "__main__":
    main()
