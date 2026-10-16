"""
Frame-level discrete-time simulator of an aging fog network.

Each frame k:
  1. policy chooses the active set s(k)  (using a demand forecast, see Sim.forecast)
  2. policy dispatches the *actual* demand lam(k) to active nodes (overflow -> cloud = SLA violation)
  3. physics: power, junction temperature, NBTI aging (tau, p, r), failures
  4. policy post-update (Lyapunov virtual queues)
"""
import time
import numpy as np

from .config import NODE_TYPES
from . import physics as ph
from .inputs import build_inputs

HW_KEYS = ["mu", "P_base", "P_leak0", "P_dyn", "P_sleep", "R_th", "E_wake", "t_wake", "embodied"]


def gini(x):
    x = np.sort(np.asarray(x, dtype=float))
    if x.sum() <= 0:
        return 0.0
    n = len(x)
    return float((2 * np.arange(1, n + 1) - n - 1).dot(x) / (n * x.sum()))


class Sim:
    def __init__(self, cfg, inputs=None):
        self.cfg = cfg
        self.layout = cfg.layout()
        self.N = len(self.layout)
        self.types = np.array([t for t, _ in self.layout])
        self.sites = np.array([s for _, s in self.layout])
        self.hw = {k: np.array([NODE_TYPES[t][k] for t in self.types], dtype=float) for k in HW_KEYS}
        if cfg.seed_diversity:
            # per-seed network delays and manufacturing variation of the hardware
            lay_rng = np.random.default_rng(7 + 101 * cfg.seed)
            self.dnet = lay_rng.uniform(cfg.dnet_min_s, cfg.dnet_max_s, self.N)
            self.hw["R_th"] = self.hw["R_th"] * lay_rng.uniform(0.9, 1.1, self.N)       # packaging / airflow
            self.hw["P_leak0"] = self.hw["P_leak0"] * lay_rng.lognormal(0.0, 0.15, self.N)  # leakage spread
        else:
            lay_rng = np.random.default_rng(7)                  # fixed topology
            self.dnet = lay_rng.uniform(cfg.dnet_min_s, cfg.dnet_max_s, self.N)
        if inputs is None:
            inputs = build_inputs(cfg, self.layout)
        self.shape, self.amb, self.T_out = inputs
        self.cap_nom = self.capacity(np.zeros(self.N))
        lam = self.shape * cfg.load_level * self.cap_nom.sum()
        self.lam = np.minimum(lam, 0.97 * self.cap_nom.sum())

    # ------------------------------------------------------------------ helpers
    def capacity(self, a):
        """SLA-feasible request rate per node (M/M/1 deadline + utilisation cap)."""
        mu = self.hw["mu"] * ph.freq_scale(self.cfg, a)
        slack = np.maximum(self.cfg.deadline_s - self.dnet, 1e-3)
        return np.clip(np.minimum(self.cfg.rho_max * mu, mu - 1.0 / slack), 0.0, None)

    def forecast(self, k):
        """Demand forecast for frame k, using only past frames.
        last   : value of the previous frame (persistence)
        max3   : maximum of the last 3 frames (covers short ramps)
        hybrid : max( max3 , same hour yesterday x today's trend ), the default
                 -> anticipates the daily ramp-up before it happens
        """
        lam = self.lam
        if k == 0:
            return lam[0]
        mode = self.cfg.forecast
        if mode == "last":
            return lam[k - 1]
        m3 = lam[max(0, k - 3):k].max()
        if mode == "max3":
            return m3
        day = int(round(24 / self.cfg.frame_h))
        if k <= day:
            return m3
        trend = lam[k - 1] / max(lam[k - 1 - day], 1e-9)          # today vs yesterday, same hour
        seasonal = lam[k - day] * min(max(trend, 0.8), 1.25)        # same hour yesterday, trend-corrected
        return max(m3, seasonal)

    @staticmethod
    def activate_until(order, cap, need):
        s = np.zeros(len(cap), bool)
        if len(order) == 0:
            return s
        cs = np.cumsum(cap[order])
        j = int(np.searchsorted(cs, need))
        s[order[: min(j + 1, len(order))]] = True
        return s

    @staticmethod
    def prop_dispatch(s, lam, cap):
        c = np.where(s, cap, 0.0)
        tot = c.sum()
        if tot <= 0:
            return np.zeros_like(cap)
        return c * min(1.0, lam / tot)

    @staticmethod
    def ff_dispatch(order, s, lam, cap):
        out = np.zeros_like(cap)
        rem = lam
        for i in order:
            if not s[i] or rem <= 0:
                continue
            x = min(cap[i], rem)
            out[i] = x
            rem -= x
        return out

    # ------------------------------------------------------------------ main loop
    def run(self, policy):
        cfg, N, hw = self.cfg, self.N, self.hw
        fh, n = cfg.frame_h, cfg.n_frames
        a_crit, guard = cfg.a_crit, cfg.guard_mV * 1e-3
        per_year = int(round(8760 / fh))
        per_day = int(round(24 / fh))

        self.alive = np.ones(N, bool)
        self.s_prev = np.ones(N, bool)
        self.tau = np.zeros(N)
        self.p = np.zeros(N)
        self.r = np.zeros(N)
        self.a = np.zeros(N)
        self.r_before = np.zeros(N)
        self.theta = self.amb[0].copy()
        self.k = 0
        fail_t = np.full(N, np.nan)
        policy.reset(self)

        # accumulators
        E_node = np.zeros(N)
        E_comp = dict(base=0.0, leak=0.0, dyn=0.0, sleep=0.0, wake=0.0)   # energy breakdown [Wh]
        act_frames = np.zeros(N)
        theta_act_sum = np.zeros(N)
        theta_max = np.zeros(N)
        r_act_sum_last = np.zeros(N)
        act_last = np.zeros(N)
        wakeups = np.zeros(N)
        dem_req = over_req = wake_req = 0.0
        tau_1y_before = None
        n_active_sum = 0.0

        log = None
        if cfg.detailed_log:
            nd = n // per_day
            log = dict(a=np.zeros((nd, N)), p=np.zeros((nd, N)), tau=np.zeros((nd, N)), Z=np.zeros((nd, N)),
                       theta=np.zeros((nd, N)), act=np.zeros((nd, N)), E=np.zeros(nd), dem=np.zeros(nd),
                       over=np.zeros(nd), nact=np.zeros(nd), alive=np.zeros(nd))
            wk0 = int((365 + 196) * per_day)                       # a summer week in year 2
            wk0 = min(wk0, max(0, n - 7 * per_day))
            log["week_start_day"] = wk0 // per_day
            log["week_s"] = np.zeros((7 * per_day, N))
            log["week_theta"] = np.zeros((7 * per_day, N))
            log["week_lam"] = np.zeros(7 * per_day)
            d_th = np.zeros(N); d_act = np.zeros(N); d_E = d_dem = d_over = d_nact = 0.0

        t0 = time.perf_counter()
        for k in range(n):
            self.k = k
            amb_k = self.amb[k]
            lam_k = self.lam[k]
            lam_f = self.forecast(k)
            cap = np.where(self.alive, self.capacity(self.a), 0.0)

            s = policy.decide(self, k, lam_f, amb_k, cap) & self.alive
            lam_i = policy.dispatch(self, k, s, lam_k, amb_k, cap)
            lam_i = np.where(s, np.minimum(lam_i, cap), 0.0)
            served = lam_i.sum()
            over = max(lam_k - served, 0.0)

            woke = s & ~self.s_prev
            wreq = 0.0
            if woke.any():
                wakeups += woke
                cap_on = cap[s & self.s_prev].sum()
                tw = hw["t_wake"][woke].max()
                wreq = max(0.0, min(lam_k, served) - cap_on) * tw     # requests lost while resuming

            V = ph.supply_voltage(cfg, self.a)
            P, th = ph.power_temperature(cfg, lam_i, V, self.a, amb_k, hw, active=s)
            P = np.where(self.alive, P, 0.0)
            E = P * fh + woke * hw["E_wake"] / 3600.0                  # Wh
            on = s & self.alive
            e_base = np.where(on, hw["P_base"], 0.0) * fh
            e_dyn = np.where(on, hw["P_dyn"] * (lam_i / hw["mu"]) * (V / cfg.V_nom) ** 2, 0.0) * fh
            E_comp["base"] += e_base.sum(); E_comp["dyn"] += e_dyn.sum()
            E_comp["leak"] += (np.where(on, P, 0.0) * fh - e_base - e_dyn).sum()
            E_comp["sleep"] += (np.where(self.alive & ~s, P, 0.0) * fh).sum()
            E_comp["wake"] += (woke * hw["E_wake"] / 3600.0).sum()
            AF = ph.accel_factor(cfg, th, V)
            dtau = np.where(s, fh * AF, 0.0)
            self.r_before = self.r.copy()
            self.tau = self.tau + dtau
            self.p = ph.p_of_tau(cfg, self.tau)
            self.r = np.where(self.alive, ph.r_next(cfg, self.r, s, AF), self.r)
            self.a = self.p + self.r
            self.theta = th

            newly = self.alive & ((s & (self.a >= a_crit)) | (self.p >= a_crit - guard))
            if newly.any():
                fail_t[newly] = (k + 1) * fh / 8760.0
                self.alive &= ~newly

            policy.post(self, k, s, lam_i, th, V, dtau, amb_k)

            # ---- accumulate
            E_node += E
            act_frames += s
            theta_act_sum += np.where(s, th, 0.0)
            theta_max = np.maximum(theta_max, np.where(s, th, 0.0))
            dem_req += lam_k * 3600 * fh
            over_req += over * 3600 * fh
            wake_req += wreq
            n_active_sum += s.sum()
            if k >= n - per_year:
                r_act_sum_last += np.where(s, self.r, 0.0)
                act_last += s
            if k == n - per_year:
                tau_1y_before = self.tau.copy()
            self.s_prev = s

            if log is not None:
                d_th += np.where(s, th, amb_k); d_act += s; d_E += E.sum(); d_dem += lam_k
                d_over += over + wreq / (3600 * fh); d_nact += s.sum()
                if (k + 1) % per_day == 0:
                    d = (k + 1) // per_day - 1
                    if d < log["a"].shape[0]:
                        log["a"][d] = self.a; log["p"][d] = self.p; log["tau"][d] = self.tau
                        log["Z"][d] = getattr(policy, "Z", np.zeros(N))
                        log["theta"][d] = d_th / per_day; log["act"][d] = d_act / per_day
                        log["E"][d] = d_E; log["dem"][d] = d_dem / per_day; log["over"][d] = d_over / per_day
                        log["nact"][d] = d_nact / per_day; log["alive"][d] = self.alive.sum()
                    d_th[:] = 0; d_act[:] = 0; d_E = d_dem = d_over = d_nact = 0.0
                w = k - log["week_start_day"] * per_day
                if 0 <= w < 7 * per_day:
                    log["week_s"][w] = s; log["week_theta"][w] = np.where(s, th, np.nan); log["week_lam"][w] = lam_k
        runtime = time.perf_counter() - t0

        # ---------------------------------------------------------- lifetime projection
        H = n * fh / 8760.0
        life = fail_t.copy()
        projected = np.zeros(N, bool)
        if tau_1y_before is None:
            tau_1y_before = np.zeros(N)
        yrs = per_year * fh / 8760.0
        for i in np.where(self.alive)[0]:
            rate = (self.tau[i] - tau_1y_before[i]) / yrs
            rbar = r_act_sum_last[i] / act_last[i] if act_last[i] > 0 else 0.5 * cfg.r_sat_ref
            tau_fail = ph.tau_of_p(cfg, max(a_crit - max(rbar, guard), 1e-6))
            if self.tau[i] >= tau_fail:
                life[i] = H
            elif rate <= 1e-9:
                life[i] = 30.0
            else:
                life[i] = min(30.0, H + (tau_fail - self.tau[i]) / rate)
            projected[i] = True

        E_tot_kWh = E_node.sum() / 1000.0
        carbon_op = E_tot_kWh * cfg.grid_kg_per_kWh / 1000.0
        life_eff = np.clip(life, 0.25, cfg.max_service_years)
        carbon_emb = float((hw["embodied"] * H / life_eff).sum()) / 1000.0
        srt = np.sort(life)
        res = dict(
            energy_MWh=E_tot_kWh / 1000.0,
            energy_per_year_MWh=E_tot_kWh / 1000.0 / H,
            sla_violation_pct=100.0 * (over_req + wake_req) / dem_req,
            overflow_pct=100.0 * over_req / dem_req,
            wake_loss_pct=100.0 * wake_req / dem_req,
            first_failure_obs_years=float(np.nanmin(fail_t)) if np.isfinite(fail_t).any() else np.nan,
            network_lifetime_years=float(srt[0]),
            life_10pct_years=float(np.quantile(srt, 0.10)),
            median_node_life_years=float(np.median(srt)),
            n_failed_by_L=int(np.sum(fail_t <= cfg.L_target_years)),
            n_failed_by_H=int(np.isfinite(fail_t).sum()),
            aging_gini=gini(np.minimum(self.a, a_crit)),
            aging_std_mV=float(np.std(np.minimum(self.a, a_crit)) * 1e3),
            mean_active_frac=float(n_active_sum / (n * N)),
            wakeups_per_node_day=float(wakeups.sum() / N / (n * fh / 24)),
            mean_theta_active=float(theta_act_sum.sum() / max(act_frames.sum(), 1)),
            max_theta=float(theta_max.max()),
            carbon_op_t=carbon_op, carbon_emb_t=carbon_emb, carbon_total_t=carbon_op + carbon_emb,
            E_base_MWh=E_comp["base"] / 1e6, E_leak_MWh=E_comp["leak"] / 1e6, E_dyn_MWh=E_comp["dyn"] / 1e6,
            E_sleep_MWh=E_comp["sleep"] / 1e6, E_wake_MWh=E_comp["wake"] / 1e6,
            forced_node_pct=(100.0 * policy.stat["forced"] / max(policy.stat["alive"], 1)) if hasattr(policy, "stat") else np.nan,
            forced_queue_node_pct=(100.0 * policy.stat["forced_queue"] / max(policy.stat["alive"], 1)) if hasattr(policy, "stat") else np.nan,
            forced_frame_pct=(100.0 * policy.stat["frames_forced"] / max(policy.stat["frames"], 1)) if hasattr(policy, "stat") else np.nan,
            zmax_over_zth=policy.stat["zmax"] if hasattr(policy, "stat") else np.nan,
            runtime_s=runtime, ms_per_frame=1000.0 * runtime / n,
        )
        nodes = dict(type=self.types, site=self.sites, life_years=life, projected=projected,
                     fail_time=fail_t, a_end_mV=self.a * 1e3, p_end_mV=self.p * 1e3, tau_end_h=self.tau,
                     active_frac=act_frames / n,
                     mean_theta_active=np.where(act_frames > 0, theta_act_sum / np.maximum(act_frames, 1), np.nan),
                     mean_ambient=self.amb.mean(axis=0), energy_kWh=E_node / 1000.0,
                     aging_rate_mV_per_year=self.p * 1e3 / H)
        return res, nodes, log
