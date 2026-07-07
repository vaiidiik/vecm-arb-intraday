import numpy as np
import cvxpy as cp
import logging


class dynamic_risk_engine:
    def __init__(self, num_assets, aum, gamma=0.05,
                 entry_threshold=1.5,
                 exit_threshold=0.3,
                 short_exit_threshold=0.3,
                 long_exit_threshold=None,
                 max_leverage=1.5,
                 target_fraction=0.8,
                 max_weight_per_asset=0.45,
                 turnover_penalty=0.0005,
                 volatility_threshold=0.50,
                 trend_threshold=0.30,
                 periods_per_year=6804,
                 capital_per_trade_frac=0.20,
                 stale_loss_holding_bars=55,
                 stale_loss_threshold=-0.01):
        self.N = num_assets
        self.aum = aum
        self.gamma = gamma
        self.entry_threshold = entry_threshold
        self.long_exit_threshold = exit_threshold if long_exit_threshold is None else long_exit_threshold
        self.short_exit_threshold = short_exit_threshold
        self.max_leverage = max_leverage
        self.target_fraction = target_fraction
        self.max_weight_per_asset = max_weight_per_asset
        self.turnover_penalty = turnover_penalty
        self.volatility_threshold = volatility_threshold
        self.trend_threshold = trend_threshold
        self.periods_per_year = periods_per_year
        self.capital_per_trade_frac = capital_per_trade_frac
        self.stale_loss_holding_bars = stale_loss_holding_bars
        self.stale_loss_threshold = stale_loss_threshold
        self.logger = logging.getLogger("VECM_ARB.Risk")

    def compute_alpha(self, signals, n_assets, current_position=None):
        alpha = np.zeros(n_assets)
        if not signals:
            return alpha

        best = signals[0]
        z = best["z"]
        beta = best["beta_full"]
        rsi = best.get("rsi", 50.0)
        macd_hist = best.get("macd_hist", 0.0)
        halflife = best.get("halflife", 50.0)

        if abs(z) < self.long_exit_threshold:
            return alpha

        if abs(z) < self.entry_threshold:
            return alpha

        macd_confirmed = False
        if z > self.entry_threshold and macd_hist < 0:
            macd_confirmed = True
        elif z < -self.entry_threshold and macd_hist > 0:
            macd_confirmed = True

        if not macd_confirmed:
            return alpha

        hl_scale = min(2.0, max(0.3, 50.0 / max(halflife, 1.0)))

        rsi_boost = 1.0
        if z > 0 and rsi > 60:
            rsi_boost = 1.0 + min((rsi - 60) / 80.0, 0.3)
        elif z < 0 and rsi < 40:
            rsi_boost = 1.0 + min((40 - rsi) / 80.0, 0.3)

        direction = -np.sign(z)
        magnitude = min(abs(z) / 3.0, 1.0) * hl_scale * rsi_boost

        # NOTE: previously this also multiplied by self.capital_per_trade_frac.
        # That double-counted the capital allocation: eff_leverage/eff_max_wt
        # in optimize() (and _fallback_allocation()) ALREADY scale the position
        # limits by capital_per_trade_frac. Scaling alpha down by the same
        # factor here shrank the optimizer's expected-return term relative to
        # the risk penalty, so the solver never pushed weights anywhere near
        # the (already-reduced) leverage cap. Result: realized gross exposure
        # averaged ~8% of capital instead of the ~30%+ the config intended.
        # capital_per_trade_frac is applied exactly once now, in optimize().
        alpha = direction * magnitude * beta

        return alpha

    def optimize(self, alpha, cov, w_prev, adv=None, vols=None):
        N = self.N
        alpha = np.asarray(alpha, dtype=float)
        w_prev = np.asarray(w_prev, dtype=float)

        if np.abs(alpha).max() < 1e-10:
            return np.zeros(N)

        cov_reg = np.array(cov, dtype=float)
        cov_reg = (cov_reg + cov_reg.T) / 2.0
        eigvals = np.linalg.eigvalsh(cov_reg)
        if eigvals.min() < 1e-8:
            cov_reg += (1e-7 - min(eigvals.min(), 0)) * np.eye(N)

        eff_leverage = self.max_leverage * self.capital_per_trade_frac
        eff_max_wt = self.max_weight_per_asset * self.capital_per_trade_frac

        try:
            w_var = cp.Variable(N)
            ret = alpha @ w_var
            risk = cp.quad_form(w_var, cov_reg)
            turnover = cp.norm1(w_var - w_prev)

            objective = cp.Maximize(ret - self.gamma * risk - self.turnover_penalty * turnover)

            constraints = [
                cp.norm1(w_var) <= eff_leverage,
                w_var >= -eff_max_wt,
                w_var <= eff_max_wt,
            ]

            prob = cp.Problem(objective, constraints)

            for solver in [cp.CLARABEL, cp.SCS]:
                try:
                    prob.solve(solver=solver, max_iters=500, warm_start=True, verbose=False)
                    if prob.status in [cp.OPTIMAL, cp.OPTIMAL_INACCURATE] and w_var.value is not None:
                        result = np.array(w_var.value).flatten()
                        result[np.abs(result) < 1e-4] = 0.0
                        return result
                except Exception:
                    continue

        except Exception:
            pass

        return self._fallback_allocation(alpha, w_prev)

    def _fallback_allocation(self, alpha, w_prev):
        alpha_norm = np.abs(alpha).sum()
        if alpha_norm < 1e-10:
            return np.zeros(self.N)

        eff_leverage = self.max_leverage * self.capital_per_trade_frac
        eff_max_wt = self.max_weight_per_asset * self.capital_per_trade_frac

        target = alpha / alpha_norm * eff_leverage * self.target_fraction
        target = np.clip(target, -eff_max_wt, eff_max_wt)

        gross = np.abs(target).sum()
        if gross > eff_leverage:
            target *= eff_leverage / gross

        max_step = 0.15
        delta = target - w_prev
        delta = np.clip(delta, -max_step, max_step)
        w_new = w_prev + delta

        gross = np.abs(w_new).sum()
        if gross > eff_leverage:
            w_new *= eff_leverage / gross

        w_new[np.abs(w_new) < 1e-4] = 0.0
        return w_new

    def check_forced_exit(self, w, holding_bars, halflife=50.0, coint_invalid=False,
                           pnl_since_entry=None):
        if coint_invalid:
            return True

        # Data-driven stale-loss stop: trades still held past stale_loss_holding_bars
        # AND underwater beyond stale_loss_threshold rarely recover (empirically,
        # only ~6% of such trades ended up profitable, averaging ~-1.6% by the
        # time they were eventually closed by the halflife cap). Cutting them
        # here instead of waiting out the full halflife cap removes the worst
        # tail of losing trades without touching the trades that are fine.
        if (pnl_since_entry is not None
                and holding_bars >= self.stale_loss_holding_bars
                and pnl_since_entry < self.stale_loss_threshold):
            return True

        max_bars = int(3.0 * halflife)
        absolute_cap = 40 * 27
        max_bars = min(max_bars, absolute_cap)
        max_bars = max(max_bars, 27)
        if holding_bars > max_bars:
            return True
        return False
