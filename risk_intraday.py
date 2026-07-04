import numpy as np
import cvxpy as cp
import logging

class dynamic_risk_engine:
    def __init__(self, num_assets, aum, maker_taker_fee=0.0004, gamma=0.10,
                 entry_threshold=1.5,
                 exit_threshold=0.0,
                 short_exit_threshold=0.0,
                 long_exit_threshold=None,
                 max_leverage=3.0,
                 target_fraction=0.8,
                 max_weight_per_asset=0.50,
                 turnover_penalty=0.010,
                 max_holding_days=15,
                 stop_loss_pct=0.08,
                 trailing_stop_pct=0.05,
                 volatility_threshold=0.35,
                 trend_threshold=0.50):
        self.N = num_assets
        self.aum = aum
        self.fee_rate = maker_taker_fee
        self.gamma = gamma
        self.entry_threshold = entry_threshold
        self.long_exit_threshold = exit_threshold if long_exit_threshold is None else long_exit_threshold
        self.short_exit_threshold = short_exit_threshold
        self.max_leverage = max_leverage
        self.target_fraction = target_fraction
        self.max_weight_per_asset = max_weight_per_asset
        self.turnover_penalty = turnover_penalty
        self.max_holding_days = max_holding_days
        self.stop_loss_pct = stop_loss_pct
        self.trailing_stop_pct = trailing_stop_pct
        self.volatility_threshold = volatility_threshold
        self.trend_threshold = trend_threshold
        self.logger = logging.getLogger("VECM_ARB.Risk")
        self.rank_position_scale = {1: 1.0, 2: 0.8, 3: 0.6}

    def _side_capacity(self):
        return min(self.max_leverage / 2.0, self.max_weight_per_asset * self.N / 2.0)

    def _allocate_side(self, beta, descending=True):
        order = np.argsort(beta)
        if descending:
            order = order[::-1]
        remaining = self._side_capacity()
        value = 0.0
        for idx in order:
            amount = min(self.max_weight_per_asset, remaining)
            value += amount * beta[idx]
            remaining -= amount
            if remaining <= 1e-12:
                break
        return value

    def _exposure_cap(self, beta):
        beta = np.asarray(beta, dtype=float)
        long_value = self._allocate_side(beta, descending=True)
        short_value = self._allocate_side(beta, descending=False)
        return max(long_value - short_value, 0.0)

    def _current_state(self, w_prev, beta, cap):
        if cap <= 1e-12: return 0
        exposure = float(np.dot(w_prev, beta))
        if exposure > 0.25 * cap: return 1
        if exposure < -0.25 * cap: return -1
        return 0

    def _compute_target_exposure(self, s_score, w_prev, beta, rsi, holding_days, pnl_since_entry):
        cap = self._exposure_cap(beta)
        if cap <= 1e-12: return 0.0

        if holding_days >= self.max_holding_days:
            return 0.0

        if pnl_since_entry < -self.stop_loss_pct:
            return 0.0

        if pnl_since_entry < -self.trailing_stop_pct:
            return 0.0

        target = self.target_fraction * cap
        state = self._current_state(w_prev, beta, cap)

        if state == 0:
            if s_score <= -self.entry_threshold:
                return target
            if s_score >= self.entry_threshold:
                return -target
            return 0.0

        exit_long = self.long_exit_threshold
        exit_short = self.short_exit_threshold

        if state > 0:
            if s_score >= -exit_long:
                return 0.0
            return target

        if s_score <= exit_short:
            return 0.0
        return -target

    def _regime_filter(self, market_vol, trend_strength):
        if market_vol > self.volatility_threshold:
            return False
        if abs(trend_strength) > self.trend_threshold:
            return False
        return True

    def optimise_weights(self, w_prev, cov_matrix, beta, z_score, rsi, adv, vols,
                         borrow_rates, halted_indices=None, kappa=0.0,
                         holding_days=0, pnl_since_entry=0.0,
                         market_vol=0.0, trend_strength=0.0, rank=1, capital=None):
        if halted_indices is None: halted_indices = []
        w_prev = np.asarray(w_prev, dtype=float)
        beta = np.asarray(beta, dtype=float)
        adv = np.asarray(adv, dtype=float)
        vols = np.asarray(vols, dtype=float)
        borrow_rates = np.asarray(borrow_rates, dtype=float)

        if kappa <= 0.0 or beta.shape[0] != self.N:
            return np.zeros(self.N)

        if not self._regime_filter(market_vol, trend_strength):
            return np.zeros(self.N)

        rank_scale = self.rank_position_scale.get(rank, 0.5)

        target_exposure = self._compute_target_exposure(z_score, w_prev, beta, rsi,
                                                        holding_days, pnl_since_entry)
        if abs(target_exposure) < 1e-12:
            return np.zeros(self.N)

        norm_sq = np.sum(beta ** 2)
        if norm_sq < 1e-12: return np.zeros(self.N)
        w_ideal = (target_exposure / norm_sq) * beta
        w_ideal = w_ideal * rank_scale

        w = cp.Variable(self.N)
        delta_w = w - w_prev

        tracking_error = cp.sum_squares(w - w_ideal)
        total_friction = self.fee_rate + self.turnover_penalty
        fee_penalty = total_friction * cp.norm(delta_w, 1)
        borrow_penalty = borrow_rates @ cp.pos(-w)

        adv_safe = np.maximum(adv, 1.0)
        eta = self.gamma * vols * np.sqrt((capital if capital is not None else self.aum) / adv_safe)
        impact_penalty = cp.sum(cp.multiply(eta, cp.power(cp.abs(delta_w), 1.5)))

        objective = cp.Minimize(20.0 * tracking_error + fee_penalty + borrow_penalty + impact_penalty)

        constraints = [
            cp.sum(w) == 0,
            cp.norm(w, 1) <= self.max_leverage * rank_scale,
            cp.abs(w) <= self.max_weight_per_asset,
        ]
        for idx in halted_indices:
            constraints.append(w[idx] == 0)

        problem = cp.Problem(objective, constraints)

        for solver in [cp.CLARABEL, cp.SCS]:
            try:
                problem.solve(solver=solver)
                if problem.status in ["optimal", "optimal_inaccurate"] and w.value is not None:
                    return np.asarray(w.value, dtype=float)
            except Exception:
                continue

        self.logger.warning(f"Solver failed. Status: {problem.status}. Using fallback.")
        return self._fallback_weights(w_ideal)

    def _fallback_weights(self, w_ideal):
        w = np.clip(w_ideal, -self.max_weight_per_asset, self.max_weight_per_asset)
        w = w - np.mean(w)
        if np.sum(np.abs(w)) > self.max_leverage:
            w = w / (np.sum(np.abs(w)) / self.max_leverage)
        return w

    def _graceful_unwind(self, w_prev):
        return w_prev * 0.9