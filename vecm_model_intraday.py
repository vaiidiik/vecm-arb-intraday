import numpy as np
import pandas as pd
from statsmodels.tsa.vector_ar.vecm import coint_johansen


class vecm:
    def __init__(self, significance=0.05, entry_threshold=1.5, exit_threshold=0.0,
                 short_exit_threshold=0.0, long_exit_threshold=0.0,
                 min_kappa=2.0, max_kappa=50.0,
                 volume_window=20, volume_clip=(0.5, 2.0),
                 trend_lookback=120, trend_threshold=0.50,
                 delay_span=5, k_ar_diff=5, periods_per_year=252):
        self.sig_level = significance
        self.critical_idx = 2 if significance <= 0.01 else 1 if significance <= 0.05 else 0
        self.entry_threshold = entry_threshold
        self.exit_threshold = exit_threshold
        self.short_exit_threshold = short_exit_threshold
        self.long_exit_threshold = long_exit_threshold
        self.min_kappa = min_kappa
        self.max_kappa = max_kappa
        self.volume_window = volume_window
        self.volume_clip = volume_clip
        self.trend_lookback = trend_lookback
        self.trend_threshold = trend_threshold
        self.delay_span = delay_span
        self.k_ar_diff = k_ar_diff
        self.periods_per_year = periods_per_year

    def is_trending(self, log_prices):
        log_prices = np.asarray(log_prices, dtype=float)
        if log_prices.shape[0] < 2:
            return False
        avg_return = float(np.mean(log_prices[-1] - log_prices[0]))
        return abs(avg_return) > self.trend_threshold

    def compute_rsi(self, series, period=14):
        delta = np.diff(series)
        gain = np.where(delta > 0, delta, 0.0)
        loss = np.where(delta < 0, -delta, 0.0)
        if len(gain) == 0:
            return np.array([50.0])
        avg_gain = pd.Series(gain).ewm(alpha=1.0 / period, adjust=False).mean().values
        avg_loss = pd.Series(loss).ewm(alpha=1.0 / period, adjust=False).mean().values
        rs = np.divide(avg_gain, avg_loss, out=np.zeros_like(avg_gain), where=avg_loss != 0)
        rsi = 100.0 - (100.0 / (1.0 + rs))
        return np.concatenate(([50.0], rsi))

    def cointegrate(self, log_prices):
        log_prices = np.asarray(log_prices, dtype=float)
        if log_prices.ndim != 2 or log_prices.shape[0] < 10 or log_prices.shape[1] < 2:
            return 0, None
        if not np.isfinite(log_prices).all():
            return 0, None
        try:
            result = coint_johansen(log_prices, det_order=0, k_ar_diff=self.k_ar_diff)
        except Exception:
            return 0, None
        trace_stats = result.lr1
        critical_values = result.cvt[:, self.critical_idx]
        rank = 0
        for i in range(len(trace_stats)):
            if np.isfinite(trace_stats[i]) and trace_stats[i] > critical_values[i]:
                rank += 1
            else:
                break
        rank = min(rank, log_prices.shape[1])
        if rank == 0:
            return 0, None
        beta_set = self._normalize_beta_matrix(result.evec[:, :rank])
        if beta_set is None:
            return 0, None
        return rank, beta_set

    def _normalize_beta_matrix(self, beta_set):
        beta_set = np.asarray(beta_set, dtype=float)
        if beta_set.ndim == 1:
            beta_set = beta_set.reshape(-1, 1)
        cols = []
        for idx in range(beta_set.shape[1]):
            beta = beta_set[:, idx]
            if not np.isfinite(beta).all():
                continue
            scale = np.sum(np.abs(beta))
            if scale < 1e-12:
                continue
            beta = beta / scale
            anchor = np.argmax(np.abs(beta))
            if beta[anchor] < 0:
                beta = -beta
            cols.append(beta)
        if not cols:
            return None
        return np.column_stack(cols)

    def _fit_ou(self, spread_series):
        spread_series = np.asarray(spread_series, dtype=float)
        if len(spread_series) < 10 or not np.isfinite(spread_series).all():
            return None
        x = spread_series[:-1]
        y = spread_series[1:]
        n = len(x)
        sx = np.sum(x); sy = np.sum(y)
        sxx = np.sum(x * x); sxy = np.sum(x * y)
        denom = n * sxx - sx * sx
        if abs(denom) < 1e-12:
            return None
        b = (n * sxy - sx * sy) / denom
        a = (sy - b * sx) / n
        if not np.isfinite(a) or not np.isfinite(b) or b <= 0.0 or b >= 1.0:
            return None
        residuals = y - a - b * x
        if len(residuals) < 3:
            return None
        var_zeta = np.var(residuals, ddof=2)
        if not np.isfinite(var_zeta) or var_zeta <= 1e-16:
            return None
        kappa = -np.log(b) * self.periods_per_year
        if not np.isfinite(kappa) or kappa < self.min_kappa or kappa > self.max_kappa:
            return None
        m = a / (1.0 - b)
        sigma_eq = np.sqrt(var_zeta / (1.0 - b * b))
        if not np.isfinite(m) or not np.isfinite(sigma_eq) or sigma_eq < 1e-10:
            return None
        return {"a": a, "b": b, "kappa": kappa, "m": m,
                "sigma_eq": sigma_eq, "var_zeta": var_zeta}

    def _score_candidates(self, current_log_prices, beta_set, historical_log_prices):
        beta_set = self._normalize_beta_matrix(beta_set)
        if beta_set is None:
            return []
        historical_log_prices = np.asarray(historical_log_prices, dtype=float)
        current_log_prices = np.asarray(current_log_prices, dtype=float)
        scores = []
        for idx in range(beta_set.shape[1]):
            beta = beta_set[:, idx]
            historical_spread = historical_log_prices @ beta
            spread_rsi = self.compute_rsi(historical_spread, period=14)
            ou_params = self._fit_ou(historical_spread)
            if ou_params is None:
                continue
            current_spread = float(current_log_prices @ beta)
            current_rsi = spread_rsi[-1]
            s_score = (current_spread - ou_params["m"]) / ou_params["sigma_eq"]
            if np.isfinite(s_score) and abs(s_score) < 10.0:
                scores.append({
                    "s_score": float(s_score),
                    "kappa": float(ou_params["kappa"]),
                    "beta": beta,
                    "rsi": current_rsi,
                    "selection_score": abs(float(s_score)),
                })
        return scores

    def calculate_sscore(self, current_log_prices, beta, historical_log_prices):
        scores = self._score_candidates(current_log_prices, beta, historical_log_prices)
        if not scores:
            return 0.0, 0.0, None, 50.0
        selected = max(scores, key=lambda item: item["selection_score"])
        return selected["s_score"], selected["kappa"], selected["beta"], selected["rsi"]

    def volume_adjust_returns(self, log_prices_window, adv_window):
        log_prices_window = np.asarray(log_prices_window, dtype=float)
        raw_returns = np.diff(log_prices_window, axis=0)
        if adv_window is None:
            return raw_returns
        adv_window = np.asarray(adv_window, dtype=float)
        if adv_window.shape[0] != log_prices_window.shape[0]:
            return raw_returns
        adjusted_returns = raw_returns.copy()
        for idx in range(raw_returns.shape[0]):
            end = idx + 2
            start = max(0, end - self.volume_window)
            typical_adv = np.mean(adv_window[start:end], axis=0)
            current_adv = adv_window[idx + 1]
            volume_factor = np.divide(
                typical_adv, current_adv + 1e-8,
                out=np.ones_like(typical_adv),
                where=(current_adv + 1e-8) > 0
            )
            volume_factor = np.clip(volume_factor, self.volume_clip[0], self.volume_clip[1])
            adjusted_returns[idx] = raw_returns[idx] * volume_factor
        return adjusted_returns

    def volume_adjust_log_prices(self, log_prices_window, adv_window):
        adjusted_returns = self.volume_adjust_returns(log_prices_window, adv_window)
        adjusted_log_prices = np.empty_like(log_prices_window, dtype=float)
        adjusted_log_prices[0] = log_prices_window[0]
        adjusted_log_prices[1:] = adjusted_log_prices[0] + np.cumsum(adjusted_returns, axis=0)
        return adjusted_log_prices

    def check_cointegration_stability(self, log_prices, beta, window=10):
        if beta is None:
            return True
        beta = np.asarray(beta)
        if beta.ndim == 0:
            return True
        if beta.ndim == 1:
            beta = beta.reshape(-1, 1)
        if beta.shape[0] != log_prices.shape[1]:
            return True
        spread = log_prices @ beta
        if len(spread) < window + 1:
            return True
        past = spread[-(window + 1):-1]
        mean = np.mean(past)
        std = np.std(past)
        if std < 1e-8:
            return True
        if abs(spread[-1] - mean) > 5.0 * std:
            return False
        return True