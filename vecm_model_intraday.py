import numpy as np
import pandas as pd
from itertools import combinations
from statsmodels.tsa.vector_ar.vecm import coint_johansen


class vecm:
    def __init__(self, significance=0.05, entry_threshold=1.5, exit_threshold=0.3,
                 short_exit_threshold=0.3, long_exit_threshold=0.3,
                 min_kappa=1.0, max_kappa=100.0,
                 volume_window=20, volume_clip=(0.5, 2.0),
                 trend_lookback=135, trend_threshold=0.30,
                 delay_span=3, k_ar_diff=3, periods_per_year=6804,
                 rsi_period=14, rsi_overbought=65, rsi_oversold=35,
                 macd_fast=12, macd_slow=26, macd_signal=9,
                 use_pairwise=True, zscore_lookback=540):
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
        self.rsi_period = rsi_period
        self.rsi_overbought = rsi_overbought
        self.rsi_oversold = rsi_oversold
        self.macd_fast = macd_fast
        self.macd_slow = macd_slow
        self.macd_signal = macd_signal
        self.use_pairwise = use_pairwise
        self.zscore_lookback = zscore_lookback

    def compute_rsi(self, series, period=None):
        if period is None:
            period = self.rsi_period
        series = np.asarray(series, dtype=float)
        if len(series) < 2:
            return np.array([50.0])
        delta = np.diff(series)
        gain = np.where(delta > 0, delta, 0.0)
        loss = np.where(delta < 0, -delta, 0.0)
        avg_gain = pd.Series(gain).ewm(alpha=1.0 / period, adjust=False).mean().values
        avg_loss = pd.Series(loss).ewm(alpha=1.0 / period, adjust=False).mean().values
        rs = np.divide(avg_gain, avg_loss, out=np.ones_like(avg_gain), where=avg_loss > 1e-10)
        rsi = 100.0 - (100.0 / (1.0 + rs))
        return np.concatenate(([50.0], rsi))

    def compute_macd(self, series):
        series = np.asarray(series, dtype=float)
        if len(series) < self.macd_slow + self.macd_signal:
            return 0.0, 0.0, 0.0
        s = pd.Series(series)
        ema_fast = s.ewm(span=self.macd_fast, adjust=False).mean()
        ema_slow = s.ewm(span=self.macd_slow, adjust=False).mean()
        macd_line = ema_fast - ema_slow
        signal_line = macd_line.ewm(span=self.macd_signal, adjust=False).mean()
        histogram = macd_line - signal_line
        return float(macd_line.iloc[-1]), float(signal_line.iloc[-1]), float(histogram.iloc[-1])

    def cointegrate(self, log_prices, critical_idx_override=None):
        log_prices = np.asarray(log_prices, dtype=float)
        if log_prices.ndim != 2 or log_prices.shape[0] < 50 or log_prices.shape[1] < 2:
            return 0, None, None, None
        if not np.isfinite(log_prices).all():
            mask = np.isfinite(log_prices).all(axis=1)
            log_prices = log_prices[mask]
            if log_prices.shape[0] < 50:
                return 0, None, None, None
        critical_idx = self.critical_idx if critical_idx_override is None else critical_idx_override
        try:
            result = coint_johansen(log_prices, det_order=0, k_ar_diff=self.k_ar_diff)
            trace_stats = result.lr1
            critical_values = result.cvt[:, critical_idx]
            rank = 0
            for i in range(len(trace_stats)):
                if np.isfinite(trace_stats[i]) and trace_stats[i] > critical_values[i]:
                    rank += 1
                else:
                    break
            rank = min(rank, log_prices.shape[1])
            if rank == 0:
                return 0, None, trace_stats, critical_values
            beta = result.evec[:, :rank]
            beta = self._normalize_beta(beta)
            if beta is None:
                return 0, None, trace_stats, critical_values
            return rank, beta, trace_stats, critical_values
        except Exception:
            return 0, None, None, None

    def _normalize_beta(self, beta_set):
        beta_set = np.asarray(beta_set, dtype=float)
        if beta_set.ndim == 1:
            beta_set = beta_set.reshape(-1, 1)
        cols = []
        for idx in range(beta_set.shape[1]):
            b = beta_set[:, idx].copy()
            if not np.isfinite(b).all():
                continue
            scale = np.sum(np.abs(b))
            if scale < 1e-12:
                continue
            b = b / scale
            anchor = np.argmax(np.abs(b))
            if b[anchor] < 0:
                b = -b
            cols.append(b)
        if not cols:
            return None
        return np.column_stack(cols)

    def estimate_halflife(self, spread):
        spread = np.asarray(spread, dtype=float)
        if len(spread) < 20:
            return 999.0
        spread_dm = spread - np.mean(spread)
        y = spread_dm[1:]
        x = spread_dm[:-1]
        denom = np.dot(x, x)
        if denom < 1e-10:
            return 999.0
        phi = np.dot(x, y) / denom
        if phi >= 1.0 or phi <= 0.0:
            return 999.0
        halflife = -np.log(2) / np.log(phi)
        return max(halflife, 1.0)

    def compute_spread_z(self, spread, lookback=None):
        if lookback is None:
            lookback = self.zscore_lookback
        spread = np.asarray(spread, dtype=float)
        if len(spread) < 20:
            return 0.0, 0.0, 1.0
        window = spread[-lookback:] if len(spread) > lookback else spread
        mu = np.mean(window)
        sigma = np.std(window)
        if sigma < 1e-10:
            return 0.0, mu, 1.0
        z = (spread[-1] - mu) / sigma
        return float(z), float(mu), float(sigma)

    def generate_all_signals(self, log_prices, n_assets):
        candidates = []

        rank_full, beta_full, trace_full, crit_full = self.cointegrate(log_prices)
        if rank_full > 0 and beta_full is not None:
            for v in range(min(rank_full, 3)):
                bv = beta_full[:, v]
                spread = log_prices @ bv
                hl = self.estimate_halflife(spread)
                if hl > 500:
                    continue
                z, mu, sigma = self.compute_spread_z(spread)
                rsi_arr = self.compute_rsi(spread)
                rsi = float(rsi_arr[-1])
                _, _, macd_hist = self.compute_macd(spread)
                score = abs(z) * (1.0 / max(hl, 1.0))
                candidates.append({
                    "type": "multivariate",
                    "rank": rank_full,
                    "beta_full": bv.copy(),
                    "z": z,
                    "halflife": hl,
                    "mu": mu,
                    "sigma": sigma,
                    "rsi": rsi,
                    "macd_hist": macd_hist,
                    "score": score,
                    "pair": None,
                    "trace_stat": float(trace_full[v]) if trace_full is not None else 0.0,
                    "critical_value": float(crit_full[v]) if crit_full is not None else 0.0,
                })

        if self.use_pairwise:
            for i, j in combinations(range(n_assets), 2):
                pair_lp = log_prices[:, [i, j]]
                pr, pb, ptr, pcr = self.cointegrate(pair_lp)
                if pr > 0 and pb is not None:
                    bv_pair = pb[:, 0]
                    spread = pair_lp @ bv_pair
                    hl = self.estimate_halflife(spread)
                    if hl > 200:
                        continue
                    z, mu, sigma = self.compute_spread_z(spread)
                    if abs(z) < 0.5:
                        continue
                    rsi_arr = self.compute_rsi(spread)
                    rsi = float(rsi_arr[-1])
                    _, _, macd_hist = self.compute_macd(spread)
                    beta_full_vec = np.zeros(n_assets)
                    beta_full_vec[i] = bv_pair[0]
                    beta_full_vec[j] = bv_pair[1]
                    score = abs(z) * (1.0 / max(hl, 1.0))
                    candidates.append({
                        "type": "pairwise",
                        "rank": pr,
                        "beta_full": beta_full_vec,
                        "z": z,
                        "halflife": hl,
                        "mu": mu,
                        "sigma": sigma,
                        "rsi": rsi,
                        "macd_hist": macd_hist,
                        "score": score,
                        "pair": (i, j),
                        "trace_stat": float(ptr[0]) if ptr is not None else 0.0,
                        "critical_value": float(pcr[0]) if pcr is not None else 0.0,
                    })

        if not candidates:
            return []

        candidates.sort(key=lambda x: x["score"], reverse=True)
        return candidates[:3]

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
        mean_val = np.mean(past)
        std_val = np.std(past)
        if std_val < 1e-8:
            return True
        if abs(spread[-1] - mean_val) > 5.0 * std_val:
            return False
        return True
