import os
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from matplotlib.ticker import FuncFormatter

def currency_formatter(x, pos):
    return f'${x:,.0f}'

def percent_formatter(x, pos):
    return f'{x:.1f}%'

def generate_tear_sheet():
    script_dir = os.path.dirname(os.path.abspath(__file__))
    daily_path = os.path.join(script_dir, "backtest_intraday.csv")

    if not os.path.exists(daily_path):
        print(f"Error: File missing at {daily_path}")
        return

    if os.path.getsize(daily_path) < 5:
        print("Error: The CSV file exists but is essentially empty.")
        return

    try:
        df = pd.read_csv(daily_path)
        cols = df.columns.tolist()
        if "Date" in cols:
            df.set_index("Date", inplace=True)
        elif "date" in cols:
            df.set_index("date", inplace=True)
        else:
            print(f"Error: No date column found. Actual columns: {cols}")
            return
        df.index = pd.to_datetime(df.index, utc=True)
    except Exception as e:
        print(f"Error reading file: {e}")
        return

    sns.set_theme(style="darkgrid", palette="Set2")
    fig = plt.figure(figsize=(16, 20), constrained_layout=True)
    gs = fig.add_gridspec(5, 3, height_ratios=[2, 1, 1.5, 1, 1])

    ax0 = fig.add_subplot(gs[0, :])
    ax0.plot(df.index, df['net_equity'], label='Net Equity', color='#1f77b4', linewidth=2)
    ax0.plot(df.index, df['gross_equity'], label='Gross Equity', color='#ff7f0e', alpha=0.6, linewidth=1.5)
    ax0.yaxis.set_major_formatter(FuncFormatter(currency_formatter))
    ax0.set_title('Equity Curve', fontsize=14, fontweight='bold')
    ax0.set_ylabel('Capital ($)')
    ax0.legend(loc='upper left')
    ax0.grid(True, alpha=0.3)

    ax1 = fig.add_subplot(gs[1, :])
    ax1.fill_between(df.index, df['drawdown'] * -100, 0, color='#d62728', alpha=0.6, step='post')
    ax1.yaxis.set_major_formatter(FuncFormatter(percent_formatter))
    ax1.set_title('Drawdown (%)', fontsize=14, fontweight='bold')
    ax1.set_ylabel('Drawdown')
    ax1.grid(True, alpha=0.3)

    ax2 = fig.add_subplot(gs[2, :])
    ax2.plot(df.index, df['z_score'], color='#9467bd', linewidth=1.5, label='Z-Score')
    ax2.axhline(y=1.5, color='red', linestyle='--', alpha=0.7, linewidth=1)
    ax2.axhline(y=-1.5, color='red', linestyle='--', alpha=0.7, linewidth=1)
    ax2.axhline(y=0, color='black', linestyle='-', alpha=0.5)
    ax2.set_title('VECM Spread Z-Score', fontsize=14, fontweight='bold')
    ax2.set_ylabel('Z-Score')
    ax2.legend(loc='upper right')
    ax2.grid(True, alpha=0.3)

    ax3 = fig.add_subplot(gs[3, 0])
    ax3.plot(df.index, df['rank'], color='#2ca02c', drawstyle='steps-post', linewidth=1.5)
    ax3.set_title('Cointegration Rank', fontsize=12, fontweight='bold')
    ax3.set_ylabel('Rank')
    ax3.set_yticks([0, 1, 2, 3])
    ax3.grid(True, alpha=0.3)

    ax4 = fig.add_subplot(gs[3, 1])
    ax4.plot(df.index, df['Gross_Exposure'], color='#e377c2', linewidth=1.5)
    ax4.set_title('Gross Exposure (Leverage)', fontsize=12, fontweight='bold')
    ax4.set_ylabel('Exposure')
    ax4.grid(True, alpha=0.3)

    ax5 = fig.add_subplot(gs[3, 2])
    ax5.plot(df.index, df['Daily_Turnover'], color='#8c564b', linewidth=1)
    ax5.set_title('Per-Bar Turnover', fontsize=12, fontweight='bold')
    ax5.set_ylabel('Turnover')
    ax5.grid(True, alpha=0.3)

    ax6 = fig.add_subplot(gs[4, 0])
    daily_ret = df['net_equity'].pct_change().dropna()
    ax6.hist(daily_ret, bins=50, color='#1f77b4', alpha=0.7, edgecolor='black')
    ax6.axvline(x=0, color='red', linestyle='--', alpha=0.7)
    ax6.set_title('15-Min Bar Returns Distribution', fontsize=12, fontweight='bold')
    ax6.set_xlabel('Return')
    ax6.set_ylabel('Frequency')
    ax6.grid(True, alpha=0.3)

    ax7 = fig.add_subplot(gs[4, 1])
    rolling_sharpe = daily_ret.rolling(520).mean() / daily_ret.rolling(520).std() * np.sqrt(6804)
    ax7.plot(df.index[1:], rolling_sharpe, color='#17becf', linewidth=1.5)
    ax7.axhline(y=0, color='black', linestyle='-', alpha=0.5)
    ax7.set_title('520-Bar (~20-Day) Rolling Sharpe Ratio', fontsize=12, fontweight='bold')
    ax7.set_ylabel('Sharpe')
    ax7.grid(True, alpha=0.3)

    ax8 = fig.add_subplot(gs[4, 2])
    cum_ret = (1 + daily_ret).cumprod()
    ax8.plot(cum_ret.index, cum_ret, color='#bcbd22', linewidth=1.5)
    ax8.set_title('Cumulative Return', fontsize=12, fontweight='bold')
    ax8.set_ylabel('Cumulative PnL')
    ax8.yaxis.set_major_formatter(FuncFormatter(lambda x, _: f'{x:.2f}x'))
    ax8.grid(True, alpha=0.3)

    plt.suptitle('VECM Statistical Arbitrage - Performance Tear Sheet', fontsize=18, fontweight='bold')
    save_path = os.path.join(script_dir, "performance_tear_sheet_intraday.png")
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"Tear sheet saved to {save_path}")

if __name__ == "__main__":
    generate_tear_sheet()
