import numpy as np
import pandas as pd
import logging

class PortfolioBacktester:
    def __init__(self, initial_capital=1000000.0, gamma=0.10, maker_taker_fee=0.0004, periods_per_year=252.0):
        self.periods_per_year = periods_per_year
        self.initial_capital = initial_capital
        self.capital = initial_capital
        self.gross_capital = initial_capital
        self.gamma = gamma
        self.fee_rate = maker_taker_fee
        self.daily_results = []
        self.peak_capital = initial_capital
        self.logger = logging.getLogger("VECM_ARB.Backtest")
        self.days = 0
        self.entry_prices = {}
        self.entry_weights = {}
        self.entry_day = {}
        self.position_pnl = {}

    def position_pnl_sum(self):
        return sum(self.position_pnl.values())

    def process_day(self, date, w_prev, w_new, prices_prev, prices_new, adv, vols, z_score, rank, borrow_rates):
        w_prev = np.asarray(w_prev, dtype=float)
        w_new = np.asarray(w_new, dtype=float)
        prices_prev = np.asarray(prices_prev, dtype=float)
        prices_new = np.asarray(prices_new, dtype=float)

        asset_returns = (prices_new - prices_prev) / prices_prev
        gross_pnl = self.capital * np.dot(w_prev, asset_returns)

        self.days += 1

        for i in range(len(w_prev)):
            if abs(w_prev[i]) > 1e-6:
                if i not in self.entry_prices:
                    self.entry_prices[i] = prices_prev[i]
                    self.entry_weights[i] = w_prev[i]
                    self.entry_day[i] = self.days
                    self.position_pnl[i] = 0.0
                else:
                    position_return = (prices_new[i] - self.entry_prices[i]) / self.entry_prices[i]
                    self.position_pnl[i] = self.entry_weights[i] * position_return * self.capital
            else:
                if i in self.entry_prices:
                    del self.entry_prices[i]
                    del self.entry_weights[i]
                    del self.entry_day[i]
                    del self.position_pnl[i]

        holding_days = max([self.days - d for d in self.entry_day.values()], default=0)
        pnl_since_entry = self.position_pnl_sum() / self.capital if self.capital > 0 else 0.0

        delta_w = w_new - w_prev
        trade_value = self.capital * np.sum(np.abs(delta_w))
        base_fee = trade_value * self.fee_rate

        adv_safe = np.maximum(adv, 1.0)
        eta = self.gamma * vols * np.sqrt(self.capital / adv_safe)
        market_impact = self.capital * np.sum(eta * np.power(np.abs(delta_w), 1.5))

        borrow_cost = self.capital * np.dot(np.maximum(-w_prev, 0.0), borrow_rates)

        total_friction = base_fee + market_impact + borrow_cost
        net_pnl = gross_pnl - total_friction

        self.capital += net_pnl
        self.gross_capital += gross_pnl
        self.peak_capital = max(self.peak_capital, self.capital)
        drawdown = (self.peak_capital - self.capital) / self.peak_capital

        self.daily_results.append({
            "Date": date, "date": date,
            "Capital": self.capital, "capital": self.capital,
            "net_equity": self.capital, "gross_equity": self.gross_capital,
            "Gross_PnL": gross_pnl, "Net_PnL": net_pnl,
            "Total_Friction": total_friction, "Base_Fees": base_fee,
            "Market_Impact": market_impact, "Borrow_Cost": borrow_cost,
            "Drawdown": drawdown, "drawdown": drawdown,
            "Z_Score": z_score, "z_score": z_score,
            "Cointegration_Rank": rank, "rank": rank,
            "Gross_Exposure": np.sum(np.abs(w_new)),
            "gross_exposure": np.sum(np.abs(w_new)),
            "Daily_Turnover": np.sum(np.abs(delta_w)),
            "daily_turnover": np.sum(np.abs(delta_w)),
            "holding_days": holding_days,
            "pnl_since_entry": pnl_since_entry
        })

    def generate_metrics(self):
        if not self.daily_results:
            return pd.DataFrame(), pd.DataFrame(columns=["metric", "value"])

        df = pd.DataFrame(self.daily_results)
        days = len(df)
        years = days / self.periods_per_year

        ending_capital = df["Capital"].iloc[-1]
        cagr = (ending_capital / self.initial_capital) ** (1 / years) - 1.0 if years > 0 else 0.0
        max_dd = df["Drawdown"].max()

        daily_returns = df["Capital"].pct_change().dropna()
        volatility = daily_returns.std() * np.sqrt(self.periods_per_year)
        sharpe = (cagr - 0.04) / volatility if volatility > 0 else 0.0

        avg_turnover = df["Daily_Turnover"].mean()
        annual_turnover = avg_turnover * self.periods_per_year

        metrics_list = [
            {"metric": "total_days", "value": days},
            {"metric": "ending_capital", "value": round(ending_capital, 2)},
            {"metric": "cagr", "value": f"{cagr * 100:.2f}%"},
            {"metric": "max_dd", "value": f"{max_dd * 100:.2f}%"},
            {"metric": "sharpe", "value": round(sharpe, 2)},
            {"metric": "volatility", "value": round(volatility, 4)},
            {"metric": "turnover", "value": round(annual_turnover, 2)},
            {"metric": "exposure", "value": round(df["Gross_Exposure"].mean(), 4)}
        ]

        metrics = pd.DataFrame(metrics_list)
        return df, metrics

    def save_results(self, daily_path, metrics_path):
        df, metrics = self.generate_metrics()
        if df.empty:
            self.logger.warning("No data to save.")
            return
        df.to_csv(daily_path, index=False)
        metrics.to_csv(metrics_path, index=False)