"""
Scheduling policies.

Baselines
  LB        : load balancing, all nodes on, proportional dispatch (aging-spreading, energy-hungry)
  CONS      : energy-aware consolidation (FFD on energy efficiency) -- FAMASO / ETFC class
  TAC       : thermal-aware consolidation (coolest nodes first)
  WL        : wear-levelling consolidation (least-aged nodes first)
  AgingUtil : Lyapunov with a utilisation-exponential aging proxy (Lin et al. 2026 style),
              no temperature, no recovery, no safety rule
Proposed
  RACO      : Recovery-Aware, thermally-coupled Aging/energy Co-Optimisation (Lyapunov DPP)
Ablations
  RACO-noRec, RACO-noTherm, RACO-noSafety, EnergyOpt (= RACO without the aging term)
"""
import numpy as np
from . import physics as ph


class Policy:
    name = "base"

    def reset(self, sim):
        pass

    def decide(self, sim, k, lam_f, amb, cap):
        raise NotImplementedError

    def dispatch(self, sim, k, s, lam, amb, cap):
        return sim.prop_dispatch(s, lam, cap)

    def post(self, sim, k, s, lam_i, theta, V, dtau, amb):
        pass


# ============================================================================ baselines
class LB(Policy):
    name = "LB"

    def decide(self, sim, k, lam_f, amb, cap):
        return sim.alive.copy()


class CONS(Policy):
    """Energy-aware consolidation: most energy-efficient nodes first, first-fit dispatch."""
    name = "CONS"

    def reset(self, sim):
        cfg = sim.cfg
        P, _ = ph.power_temperature(cfg, sim.cap_nom, np.full(sim.N, cfg.V_nom), np.zeros(sim.N),
                                    np.full(sim.N, 25.0), sim.hw)
        self.order_all = np.argsort(P / np.maximum(sim.cap_nom, 1e-9))

    def _order(self, sim):
        return np.array([i for i in self.order_all if sim.alive[i]], dtype=int)

    def decide(self, sim, k, lam_f, amb, cap):
        return sim.activate_until(self._order(sim), cap, (1 + sim.cfg.delta) * lam_f)

    def dispatch(self, sim, k, s, lam, amb, cap):
        return sim.ff_dispatch(self._order(sim), s, lam, cap)


class TAC(Policy):
    """Thermal-aware consolidation: coolest nodes first (3 degC hysteresis for running nodes)."""
    name = "TAC"

    def decide(self, sim, k, lam_f, amb, cap):
        temp = np.where(sim.s_prev, sim.theta - 3.0, amb + 0.0)
        idx = np.where(sim.alive)[0]
        order = idx[np.argsort(temp[idx])]
        return sim.activate_until(order, cap, (1 + sim.cfg.delta) * lam_f)


class WL(Policy):
    """Wear-levelling consolidation: least-aged nodes first (0.5 mV hysteresis)."""
    name = "WL"

    def decide(self, sim, k, lam_f, amb, cap):
        score = sim.a - 0.5e-3 * sim.s_prev
        idx = np.where(sim.alive)[0]
        order = idx[np.argsort(score[idx])]
        return sim.activate_until(order, cap, (1 + sim.cfg.delta) * lam_f)


# ============================================================================ Lyapunov family
class LyapunovPolicy(Policy):
    """
    Drift-plus-penalty:   min  V*[energy] + sum_i Z_i * dtau_i      per frame
    Z_i(k+1) = max(Z_i + dtau_i - b_i, 0)   (virtual queue on equivalent stress time)

    Flags
      use_aging      : include the Z*dtau term (False -> EnergyOpt)
      aging_model    : 'physical' (Arrhenius equivalent stress time) | 'util_exp' (proxy)
      thermal_aware  : planner sees real ambient temperature (False -> assumes 25 degC)
      recovery_aware : planner uses the actual recoverable state, recovery margin and rest-to-recover
      safety         : Theorem-3 safety rule (force sleep when Z >= Z_th)
    """

    def __init__(self, name, use_aging=True, aging_model="physical", thermal_aware=True,
                 recovery_aware=True, safety=True, kappa_u=3.0, u_ref=0.7):
        self.name = name
        self.use_aging, self.aging_model = use_aging, aging_model
        self.thermal_aware, self.recovery_aware, self.safety = thermal_aware, recovery_aware, safety
        self.kappa_u, self.u_ref = kappa_u, u_ref

    # ------------------------------------------------------------------ setup
    def reset(self, sim):
        cfg = sim.cfg
        self.Z = np.zeros(sim.N)
        self.stat = dict(forced=0.0, forced_queue=0.0, alive=0.0, frames=0, frames_forced=0, zmax=0.0)   # safety-rule statistics
        guard = cfg.guard_mV * 1e-3
        if self.aging_model == "physical":
            # reference recoverable level r0 for the budget; recovery-aware planner re-weights
            # every active frame by w(r) = tau_fail(r0)/tau_fail(r)  (exact for constant r)
            self.r_margin = 0.5 * cfg.r_sat_ref if self.recovery_aware else cfg.r_sat_ref
            self.tau_crit = float(ph.tau_of_p(cfg, cfg.a_crit - self.r_margin - guard))
        else:
            self.r_margin = cfg.r_sat_ref
            self.tau_crit = cfg.T_ref_years * 8760.0
        self.b = self.tau_crit / (cfg.L_target_years * 8760.0) * cfg.frame_h
        self.Z_th = cfg.Z_th_frac * self.tau_crit
        self.lv = np.linspace(0.0, 1.0, cfg.M_levels + 1)
        self.hw2 = {k: v[:, None] for k, v in sim.hw.items()}
        self.sleep_cost = cfg.V * sim.hw["P_sleep"] * cfg.frame_h
        self.wake_E = cfg.V * sim.hw["E_wake"] / 3600.0

    # ------------------------------------------------------------------ planner model
    def _w(self, sim, r_mid):
        """Recovery weight: lifetime consumed per unit stress time relative to the reference r0."""
        if not (self.recovery_aware and self.aging_model == "physical"):
            return 1.0
        cfg = sim.cfg
        g = cfg.guard_mV * 1e-3
        num = cfg.a_crit - self.r_margin - g
        den = np.maximum(cfg.a_crit - r_mid - g, 0.15 * num)
        return (num / den) ** (1.0 / cfg.n_exp)

    def _plan_a(self, sim):
        cfg = sim.cfg
        if self.recovery_aware:
            return sim.a
        return sim.p + cfg.r_sat_ref

    def _amb_plan(self, amb):
        return amb if self.thermal_aware else np.full_like(amb, 25.0)

    def _dtau_plan(self, sim, lam, theta, V, mu):
        fh = sim.cfg.frame_h
        if self.aging_model == "physical":
            return fh * ph.accel_factor(sim.cfg, theta, V)
        u = lam / mu
        return fh * np.exp(self.kappa_u * (u - self.u_ref))

    def _grid(self, sim, amb, cap):
        cfg = sim.cfg
        a_pl = self._plan_a(sim)
        Vp = ph.supply_voltage(cfg, a_pl)
        lam = cap[:, None] * self.lv[None, :]
        P, th = ph.power_temperature(cfg, lam, Vp[:, None], a_pl[:, None], self._amb_plan(amb)[:, None], self.hw2)
        dtau = self._dtau_plan(sim, lam, th, Vp[:, None], self.hw2["mu"])
        self._raw_dtau = dtau
        if self.recovery_aware and self.aging_model == "physical":
            AF = dtau / cfg.frame_h
            r_on = ph.r_next(cfg, sim.r[:, None], True, AF)
            dtau = dtau * self._w(sim, 0.5 * (sim.r[:, None] + r_on))
        cost = cfg.V * P * cfg.frame_h
        if self.use_aging:
            cost = cost + self.Z[:, None] * dtau
        return cost, dtau

    # ------------------------------------------------------------------ convex dispatch (water-filling)
    def _waterfill(self, s, lam, cap, grid):
        ids = np.where(s & (cap > 0))[0]
        out = np.zeros(len(cap))
        if len(ids) == 0:
            return out, 0.0
        M = grid.shape[1] - 1
        cg = grid[ids]
        seg = cap[ids] / M
        mc = np.diff(cg, axis=1) / seg[:, None]
        mc = np.maximum.accumulate(mc, axis=1)
        flat = mc.ravel()
        sizes = np.repeat(seg, M)
        rows = np.repeat(np.arange(len(ids)), M)
        order = np.argsort(flat, kind="stable")
        cs = np.cumsum(sizes[order])
        take = np.zeros(len(flat))
        if lam >= cs[-1]:
            take[:] = sizes
        else:
            j = int(np.searchsorted(cs, lam))
            take[order[:j]] = sizes[order[:j]]
            take[order[j]] = lam - (cs[j - 1] if j > 0 else 0.0)
        out[ids] = np.bincount(rows, weights=take, minlength=len(ids))
        cost = cg[:, 0].sum() + float((flat * take).sum())
        return out, cost

    def _J(self, s, lam_f, cap, grid, wake, alive):
        _, c = self._waterfill(s, lam_f, cap, grid)
        return c + wake[s].sum() + self.sleep_cost[alive & ~s].sum()

    # ------------------------------------------------------------------ decide
    def decide(self, sim, k, lam_f, amb, cap):
        cfg = sim.cfg
        alive = sim.alive
        grid, dtau = self._grid(sim, amb, cap)
        self._last_grid = grid
        need = (1 + cfg.delta) * lam_f
        tot = cap[alive].sum()
        q = float(np.clip(need / max(tot, 1e-9), 0.2, 1.0))
        x = q * cfg.M_levels
        lo = int(np.floor(x)); hi = min(lo + 1, cfg.M_levels); w = x - lo
        cost_q = (1 - w) * grid[:, lo] + w * grid[:, hi]
        wake = self.wake_E * (~sim.s_prev)
        score = (cost_q + wake - self.sleep_cost) / np.maximum(q * cap, 1e-9)
        score = np.where(cap > 0, score, np.inf)

        forced = np.zeros(sim.N, bool)
        if self.safety and self.use_aging:
            forced |= self.Z >= self.Z_th
            if self.recovery_aware and self.aging_model == "physical":
                dt_q = (1 - w) * self._raw_dtau[:, lo] + w * self._raw_dtau[:, hi]
                a_pred = ph.p_of_tau(cfg, sim.tau + dt_q) + ph.r_next(cfg, sim.r, True, dt_q / cfg.frame_h)
                forced |= a_pred >= cfg.a_crit - cfg.guard_mV * 1e-3
        forced &= alive
        st = self.stat; st["forced"] += forced.sum(); st["alive"] += alive.sum(); st["frames"] += 1
        st["forced_queue"] += float((alive & (self.Z >= self.Z_th)).sum()) if (self.safety and self.use_aging) else 0.0
        st["frames_forced"] += int(forced.any()); st["zmax"] = max(st["zmax"], float(self.Z.max() / self.Z_th))
        elig = np.where(alive & ~forced)[0]
        # forced nodes are never activated; demand beyond the capacity of the
        # eligible nodes overflows to the cloud (lifetime has priority over SLA)
        order = elig[np.argsort(score[elig])].astype(int)
        s = sim.activate_until(order, cap, need)

        if cfg.local_search and s.sum() > 1:
            best = self._J(s, lam_f, cap, grid, wake, alive)
            act = np.where(s & ~forced)[0]
            # (i) try to drop expensive active nodes
            for i in act[np.argsort(-score[act])][:3]:
                t = s.copy(); t[i] = False
                if cap[t].sum() >= need:
                    J = self._J(t, lam_f, cap, grid, wake, alive)
                    if J < best:
                        s, best = t, J
            # (ii) swap the marginal active node with the next eligible candidates
            act = np.where(s & ~forced)[0]
            if len(act):
                worst = act[np.argmax(score[act])]
                cand = [i for i in elig[np.argsort(score[elig])] if not s[i]][:3]
                for c in cand:
                    t = s.copy(); t[worst] = False; t[c] = True
                    if cap[t].sum() >= need:
                        J = self._J(t, lam_f, cap, grid, wake, alive)
                        if J < best:
                            s, best = t, J
        return s

    def dispatch(self, sim, k, s, lam, amb, cap):
        out, _ = self._waterfill(s, lam, cap, self._last_grid)
        return out

    # ------------------------------------------------------------------ virtual-queue update
    def post(self, sim, k, s, lam_i, theta, V, dtau, amb):
        if self.aging_model == "physical" and self.thermal_aware:
            d = dtau * self._w(sim, 0.5 * (sim.r_before + sim.r))
        elif self.aging_model == "physical":
            a_pl = self._plan_a(sim)
            Vp = ph.supply_voltage(sim.cfg, a_pl)
            _, th = ph.power_temperature(sim.cfg, lam_i, Vp, a_pl, np.full(sim.N, 25.0), sim.hw)
            d = np.where(s, self._dtau_plan(sim, lam_i, th, Vp, sim.hw["mu"]), 0.0)
            d = d * self._w(sim, 0.5 * (sim.r_before + sim.r))
        else:
            d = np.where(s, self._dtau_plan(sim, lam_i, None, None, sim.hw["mu"]), 0.0)
        self.Z = np.maximum(self.Z + d - self.b, 0.0)


def make_policy(name):
    if name == "LB":
        return LB()
    if name == "CONS":
        return CONS()
    if name == "TAC":
        return TAC()
    if name == "WL":
        return WL()
    if name == "AgingUtil":
        return LyapunovPolicy("AgingUtil", aging_model="util_exp", recovery_aware=False, safety=False)
    if name == "AgingUtil-S":
        return LyapunovPolicy("AgingUtil-S", aging_model="util_exp", recovery_aware=False, safety=True)
    if name == "RACO":
        return LyapunovPolicy("RACO")
    if name == "RACO-noRec":
        return LyapunovPolicy("RACO-noRec", recovery_aware=False)
    if name == "RACO-noTherm":
        return LyapunovPolicy("RACO-noTherm", thermal_aware=False)
    if name == "RACO-noSafety":
        return LyapunovPolicy("RACO-noSafety", safety=False)
    if name == "EnergyOpt":
        return LyapunovPolicy("EnergyOpt", use_aging=False)
    raise ValueError(name)


BASELINES = ["LB", "CONS", "TAC", "WL", "AgingUtil"]
PROPOSED = "RACO"
ABLATIONS = ["RACO", "RACO-noRec", "RACO-noTherm", "RACO-noSafety", "EnergyOpt"]
