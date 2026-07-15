import numpy as np
import cvxpy as cp
import logging


class dynamic_risk_engine:
    def __init__(self, num_assets, aum, gamma=0.05,
                 entry_threshold=1.5,
                 exit_threshold=0.3,
                 short_exit_threshold=0.3,
                 long_exit_threshold=None,
                 max_leverage=1.0,
                 target_fraction=0.8,
                 max_weight_per_asset=0.45,
                 turnover_penalty=0.0005,
                 volatility_threshold=0.50,
                 trend_threshold=0.30,
                 periods_per_year=6804,
                 capital_per_trade_frac=0.20,
                 max_entry_halflife=120,
                 min_beta_confirmations=2,
                 preempt_z_ratio=2.0,
                 preempt_quality_margin=1.4,
                 preempt_min_holding_bars=10,
                 gap_shock_threshold=-0.004,
                 enable_gap_shock=False,
                 conviction_max_scale=2.0):
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

                                                                             
                                                                          
                                                                           
                                                                       
                                                                         
                                                                      
                                                                
                                                          
        self.max_entry_halflife = max_entry_halflife
        self.min_beta_confirmations = min_beta_confirmations

                                                                     
                                                                         
                                                                         
                                                                      
                                                                     
                                                                              
        self.preempt_z_ratio = preempt_z_ratio
        self.preempt_quality_margin = preempt_quality_margin
        self.preempt_min_holding_bars = preempt_min_holding_bars

                                                       
                                                                          
                                                                          
                                                                          
                                                                        
                                                                        
                                                                
         
                                                                          
                             
                                                                          
                                        
                                                                          
                                                                         
                                                                          
                                                                    
                                                                           
                                                                           
                                                                     
                                                                            
                                                                     
                                                                        
                                                                           
                                                                 
        self.gap_shock_threshold = gap_shock_threshold
        self.enable_gap_shock = enable_gap_shock

                                                                                
                                                                            
                                                                          
                                                                        
                                                                           
                                                                           
                                                                           
                                                                         
                                                                        
                                                                          
                                                                           
                                                                             
        self.conviction_max_scale = conviction_max_scale

        self.logger = logging.getLogger("VECM_ARB.Risk")

    def size_from_beta(self, signal, capital_frac):
        z = signal["z"]
        beta = signal["beta_full"]
        macd_hist = signal.get("macd_hist", 0.0)
        halflife = signal.get("halflife", 50.0)
        confirmations = signal.get("confirmations", 1)

        if abs(z) < self.entry_threshold:
            return None

                                                                            
        if self.max_entry_halflife is not None and halflife > self.max_entry_halflife:
            return None
        if self.min_beta_confirmations is not None and confirmations < self.min_beta_confirmations:
            return None

        macd_confirmed = False
        if z > self.entry_threshold and macd_hist < 0:
            macd_confirmed = True
        elif z < -self.entry_threshold and macd_hist > 0:
            macd_confirmed = True

        if not macd_confirmed:
            return None

        beta_sum = np.sum(np.abs(beta))
        if beta_sum < 1e-10:
            return None

        beta_norm = beta / beta_sum
        direction = -np.sign(z)
        gross_budget = self.max_leverage * capital_frac
        w = direction * beta_norm * gross_budget

        max_wt = self.max_weight_per_asset * capital_frac
        w = np.clip(w, -max_wt, max_wt)
        return w

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
        confirmations = best.get("confirmations", 1)

        if abs(z) < self.long_exit_threshold:
            return alpha

        if abs(z) < self.entry_threshold:
            return alpha

                                                                            
        if self.max_entry_halflife is not None and halflife > self.max_entry_halflife:
            return alpha
        if self.min_beta_confirmations is not None and confirmations < self.min_beta_confirmations:
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

        alpha = direction * magnitude * beta

        return alpha

    def conviction_capital_frac(self, z, base_frac, available_frac, max_scale=None):
        """
        Idle-capital fix: instead of handing every qualifying entry the
        same flat `base_frac` (e.g. SLOT_CAPITAL_FRAC), scale it up when
        the signal is a standout -- so a really strong opportunity can
        actually use the capital an empty second slot would otherwise
        leave idle, rather than being capped at its own slot's share.

        Scaling ramps linearly from 1x at |z| == entry_threshold to
        `max_scale`x at |z| == preempt_z_ratio * entry_threshold (the same
        bar should_preempt already treats as "standout"), then holds flat.
        A marginal entry right at threshold still gets exactly base_frac,
        same as before this change.

        `available_frac` is the leverage headroom actually free this bar
        (max_leverage minus gross exposure already committed to other
        slots) -- the final result is clipped to it, so this can never
        push total gross exposure over max_leverage no matter how strong
        the signal is.
        """
        if base_frac <= 0 or available_frac <= 0:
            return 0.0

        scale_cap = self.conviction_max_scale if max_scale is None else max_scale
        lo = self.entry_threshold
        hi = self.preempt_z_ratio * self.entry_threshold
        z_abs = abs(z)

        if hi <= lo:
            scale = 1.0
        else:
            t = float(np.clip((z_abs - lo) / (hi - lo), 0.0, 1.0))
            scale = 1.0 + t * (scale_cap - 1.0)

        desired = base_frac * scale
        return float(np.clip(desired, 0.0, available_frac))

    def optimize(self, alpha, cov, w_prev, adv=None, vols=None, capital_frac=None):
        """
        Mean-variance sizing: maximize alpha@w - gamma*w'cov*w -
        turnover_penalty*|w-w_prev|, under the same gross-leverage and
        per-asset caps `size_from_beta` used -- but splitting the budget
        by the *actual* covariance structure instead of a flat
        |beta|-proportional split. `capital_frac` lets a caller size
        against a specific budget (e.g. a slot's SLOT_CAPITAL_FRAC)
        instead of the engine-wide capital_per_trade_frac default.
        """
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

        frac = self.capital_per_trade_frac if capital_frac is None else capital_frac
        eff_leverage = self.max_leverage * frac
        eff_max_wt = self.max_weight_per_asset * frac

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

        return self._fallback_allocation(alpha, w_prev, capital_frac=frac)

    def _fallback_allocation(self, alpha, w_prev, capital_frac=None):
        alpha_norm = np.abs(alpha).sum()
        if alpha_norm < 1e-10:
            return np.zeros(self.N)

        frac = self.capital_per_trade_frac if capital_frac is None else capital_frac
        eff_leverage = self.max_leverage * frac
        eff_max_wt = self.max_weight_per_asset * frac

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

    @staticmethod
    def covariance_from_prices(price_window):
        """
        Rolling per-bar return covariance from a (T, N) price window --
        the real risk input `optimize()` needed and that was previously
        going unused. Falls back to a small diagonal (independent-asset)
        matrix if the window is too short to estimate covariance
        reliably, so callers never have to special-case the cold start.
        """
        price_window = np.asarray(price_window, dtype=float)
        if price_window.ndim != 2 or price_window.shape[0] < 3:
            n = price_window.shape[1] if price_window.ndim == 2 else 1
            return np.eye(n) * 1e-6

        safe_prev = np.where(price_window[:-1] > 0, price_window[:-1], 1.0)
        returns = (price_window[1:] - price_window[:-1]) / safe_prev
        finite_mask = np.isfinite(returns).all(axis=1)
        returns = returns[finite_mask]

        n = price_window.shape[1]
        if returns.shape[0] < 3:
            return np.eye(n) * 1e-6

        cov = np.cov(returns, rowvar=False)
        return np.atleast_2d(cov)

    def should_preempt(self, candidate_sig, weakest_score, weakest_holding_bars):
        """
        Recommendation #3 ("the z=8 problem"): instead of adding a 3rd
        slot, let a much stronger opportunity evict the weakest
        currently-held slot rather than being skipped outright because
        both slots happen to already be filled.

        Guarded by:
          - a minimum holding period on the incumbent (no instant churn)
          - an absolute z bar well above the entry threshold (only
            standout opportunities qualify, not marginally-better ones)
          - a quality margin over the incumbent's current score
        """
        if weakest_holding_bars is None or weakest_holding_bars < self.preempt_min_holding_bars:
            return False
        if abs(candidate_sig["z"]) < self.preempt_z_ratio * self.entry_threshold:
            return False

        halflife = candidate_sig.get("halflife", 50.0)
        candidate_score = candidate_sig.get("score", abs(candidate_sig["z"]) / max(halflife, 1.0))
        if weakest_score <= 1e-10:
            return True
        return candidate_score >= self.preempt_quality_margin * weakest_score

    def check_gap_shock(self, bar_ret, is_gap_bar):
        """
        Evaluated only on the bar immediately following a >=6h time gap
        (see GAP_HOURS_THRESHOLD in the caller). `bar_ret` is the slot's
        return attributable to *just that gap* (current weight x price
        change from the last bar to this one, as a fraction of NAV). If
        the gap alone already cost more than gap_shock_threshold, exit now
        rather than letting a likely-broken relationship keep bleeding
        until z-decay eventually catches it. Never fires on intraday bars
        and never fires on gaps that moved in the position's favor.

        Disabled by default -- see the __init__ comment for why (tested
        twice, net-negative both times).
        """
        if not self.enable_gap_shock:
            return False
        if not is_gap_bar or bar_ret is None:
            return False
        return bar_ret < self.gap_shock_threshold
