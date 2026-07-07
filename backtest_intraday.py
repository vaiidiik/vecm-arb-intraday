import numpy as np
import pandas as pd
import logging


class PortfolioBacktester:
    def __init__(self, initial_capital=1000000.0, gamma=0.10, maker_taker_fee=0.0,
                 periods_per_year=6804.0, impact_gamma=0.0):
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
        self.trade_count = 0
        self.trade_log = []

    def position_pnl_sum(self):
        return sum(self.position_pnl.values())

    def process_day(self, date, w_prev, w_new, prices_prev, prices_new, adv, vols,
                    z_score, rank, borrow_rates):
        w_prev = np.asarray(w_prev, dtype=float)
        w_new = np.asarray(w_new, dtype=float)
        prices_prev = np.asarray(prices_prev, dtype=float)
        prices_new = np.asarray(prices_new, dtype=float)
        adv = np.asarray(adv, dtype=float)
        vols = np.asarray(vols, dtype=float)
        borrow_rates = np.asarray(borrow_rates, dtype=float)

        safe_prev = np.where(prices_prev > 0, prices_prev, 1.0)
        asset_returns = (prices_new - prices_prev) / safe_prev
        gross_pnl = self.capital * np.dot(w_prev, asset_returns)

        self.days += 1

        for i in range(len(w_new)):
            was_active = abs(w_prev[i]) > 1e-6
            is_active = abs(w_new[i]) > 1e-6

            if is_active and not was_active:
                self.entry_prices[i] = prices_new[i]
                self.entry_weights[i] = w_new[i]
                self.entry_day[i] = self.days
                self.position_pnl[i] = 0.0
                self.trade_count += 1
            elif is_active and was_active:
                if i in self.entry_prices and self.entry_prices[i] > 0:
                    position_return = (prices_new[i] - self.entry_prices[i]) / self.entry_prices[i]
                    self.position_pnl[i] = self.entry_weights[i] * position_return * self.capital
            elif not is_active and was_active:
                if i in self.entry_prices:
                    del self.entry_prices[i]
                    del self.entry_weights[i]
                    del self.entry_day[i]
                    del self.position_pnl[i]

        holding_days = max([self.days - d for d in self.entry_day.values()], default=0)
        pnl_since_entry = self.position_pnl_sum() / self.capital if self.capital > 0 else 0.0

        delta_w = w_new - w_prev
        turnover = float(np.abs(delta_w).sum())

        notional_traded = self.capital * np.sum(np.abs(delta_w))
        base_fees = float(notional_traded * self.fee_rate) if turnover > 1e-6 else 0.0

        short_weights = np.minimum(w_prev, 0.0)
        borrow_cost = float(self.capital * np.dot(-short_weights, borrow_rates)) if np.any(short_weights < -1e-6) else 0.0

        total_friction = base_fees + borrow_cost
        net_pnl = gross_pnl - total_friction

        self.capital += net_pnl
        self.gross_capital += gross_pnl
        self.peak_capital = max(self.peak_capital, self.capital)
        drawdown = (self.peak_capital - self.capital) / self.peak_capital if self.peak_capital > 0 else 0.0

        gross_exposure = float(np.abs(w_new).sum())

        result = {
            "Date": str(date),
            "date": str(date),
            "Capital": round(self.capital, 2),
            "capital": round(self.capital, 2),
            "net_equity": round(self.capital, 2),
            "gross_equity": round(self.gross_capital, 2),
            "Gross_PnL": round(gross_pnl, 2),
            "Net_PnL": round(net_pnl, 2),
            "Total_Friction": round(total_friction, 4),
            "Base_Fees": round(base_fees, 4),
            "Market_Impact": 0.0,
            "Borrow_Cost": round(borrow_cost, 4),
            "Drawdown": round(drawdown, 6),
            "drawdown": round(drawdown, 6),
            "Z_Score": round(float(z_score), 4),
            "z_score": round(float(z_score), 4),
            "Cointegration_Rank": int(rank),
            "rank": int(rank),
            "Gross_Exposure": round(gross_exposure, 4),
            "gross_exposure": round(gross_exposure, 4),
            "Daily_Turnover": round(turnover, 4),
            "daily_turnover": round(turnover, 4),
            "holding_days": holding_days,
            "pnl_since_entry": round(pnl_since_entry, 6),
        }
        self.daily_results.append(result)
        return result

    def get_results_df(self):
        return pd.DataFrame(self.daily_results)

    def compute_metrics(self):
        if not self.daily_results:
            return {}

        df = pd.DataFrame(self.daily_results)
        equities = df["net_equity"].values.astype(float)

        total_bars = len(equities)
        years = total_bars / self.periods_per_year

        ending = equities[-1]
        starting = self.initial_capital

        if years > 0 and ending > 0 and starting > 0:
            cagr = (ending / starting) ** (1.0 / years) - 1.0
        else:
            cagr = 0.0

        returns = np.diff(equities) / np.where(equities[:-1] > 0, equities[:-1], 1.0)
        returns = returns[np.isfinite(returns)]

        if len(returns) > 1:
            vol = float(np.std(returns) * np.sqrt(self.periods_per_year))
            mean_ret = float(np.mean(returns) * self.periods_per_year)
            sharpe = (mean_ret - 0.04) / vol if vol > 1e-10 else 0.0
        else:
            vol = 0.0
            sharpe = 0.0

        max_dd = float(df["drawdown"].max()) if "drawdown" in df else 0.0
        avg_turnover = float(df["daily_turnover"].mean()) if "daily_turnover" in df else 0.0
        avg_exposure = float(df["gross_exposure"].mean()) if "gross_exposure" in df else 0.0
        total_fees = float(df["Total_Friction"].sum()) if "Total_Friction" in df else 0.0

        return {
            "total_bars": total_bars,
            "ending_capital": round(ending, 2),
            "cagr": f"{cagr * 100:.2f}%",
            "max_dd": f"{max_dd * 100:.2f}%",
            "sharpe": round(sharpe, 2),
            "volatility": round(vol, 4),
            "avg_turnover": round(avg_turnover, 4),
            "avg_exposure": round(avg_exposure, 4),
            "total_trades": self.trade_count,
            "total_fees": round(total_fees, 2),
        }

    def generate_metrics(self):
        if not self.daily_results:
            return pd.DataFrame(), pd.DataFrame(columns=["metric", "value"])
        df = pd.DataFrame(self.daily_results)
        metrics_dict = self.compute_metrics()
        metrics_list = [{"metric": k, "value": v} for k, v in metrics_dict.items()]
        metrics = pd.DataFrame(metrics_list)
        return df, metrics

    def save_results(self, daily_path, metrics_path):
        df, metrics = self.generate_metrics()
        if df.empty:
            self.logger.warning("No data to save.")
            return
        df.to_csv(daily_path, index=False)
        metrics.to_csv(metrics_path, index=False)
