"""
Experiment suites.  Every run is cached in results/raw/ so an interrupted run can be resumed.

  E1 main         : 5 baselines + RACO                       (seeds_main)
  E2 ablation     : RACO-noRec, -noTherm, -noSafety, EnergyOpt (seeds_main)
  E3 pareto       : L_target sweep for RACO and RACO-noRec    -> energy-lifetime Pareto front
  E4 V            : Lyapunov V sweep                          -> [O(1/V), O(V)] trade-off
  E5 tau_r        : recovery time-constant (incl. no recovery)
  E6 climate      : Tabriz / Rasht / Tehran / Bandar Abbas
  E7 load         : mean load level
  E8 compensation : fixed-voltage (frequency-degrading) silicon instead of AVS
  E9 n_exp        : NBTI time exponent
  E9b delta       : capacity head-room over the forecast (SLA-energy trade-off)
  E10 scalability : ms per frame vs number of nodes
  E11 solver gap  : greedy+local-search vs exhaustive per-frame optimum (N = 10)
"""
import itertools
import os
import pickle
import time
from dataclasses import replace
from multiprocessing import Pool

import numpy as np

from .config import Config
from .policies import make_policy, BASELINES, ABLATIONS, LyapunovPolicy
from .simulator import Sim

MAIN_METHODS = BASELINES + ["RACO"]

MODES = {
    "quick": dict(seeds_main=[0], seeds_sweep=[0],
                  L=[4, 6, 8, 12, 16], V=[1, 10, 30, 100, 1000],
                  tau_r=[0.5, 2.0, 24.0, float("inf")], tau_r_methods=["RACO", "RACO-noRec", "CONS", "WL"],
                  climate_methods=["LB", "CONS", "WL", "RACO"], load=[0.25, 0.40, 0.55],
                  load_methods=["LB", "CONS", "WL", "RACO"], comp_methods=["CONS", "WL", "AgingUtil", "RACO"],
                  n_exp=[0.16, 0.25], n_methods=["CONS", "RACO"],
                  delta=[0.0, 0.05, 0.10, 0.20], delta_methods=["CONS", "RACO"],
                  scal_N=[10, 20, 50, 100, 200], gap_frames=120,
                  lref=[7.0], lref_methods=["CONS", "RACO"]),
    "full": dict(seeds_main=[0, 1, 2, 3, 4, 5, 6, 7], seeds_sweep=[0, 1, 2],
                 L=[4, 5, 6, 7, 8, 10, 12, 14, 16], V=[1, 3, 10, 30, 100, 300, 1000],
                 tau_r=[0.5, 2.0, 6.0, 24.0, float("inf")], tau_r_methods=["RACO", "RACO-noRec", "CONS", "WL"],
                 climate_methods=["LB", "CONS", "WL", "AgingUtil", "RACO"], load=[0.25, 0.40, 0.55, 0.65],
                 load_methods=["LB", "CONS", "WL", "AgingUtil", "RACO"], comp_methods=MAIN_METHODS,
                 n_exp=[0.16, 0.20, 0.25], n_methods=["CONS", "WL", "RACO"],
                 delta=[0.0, 0.05, 0.10, 0.15, 0.25], delta_methods=["CONS", "AgingUtil", "RACO"],
                 scal_N=[10, 20, 50, 100, 200, 500], gap_frames=300,
                 lref=[3.5, 7.0, 10.0], lref_methods=["LB", "CONS", "WL", "AgingUtil", "RACO"]),
}


def build_jobs(mode, base: Config):
    m = MODES[mode]
    J = []

    def add(exp, method, param, value, seed, log=False, **ov):
        J.append(dict(exp=exp, method=method, param=param, value=value, seed=seed, log=log, ov=ov))

    for s in m["seeds_main"]:
        for meth in MAIN_METHODS:
            add("main", meth, "-", "-", s, log=(s == m["seeds_main"][0]))
        for meth in ABLATIONS[1:]:
            add("ablation", meth, "-", "-", s, log=(s == m["seeds_main"][0]))
        add("safety_baseline", "AgingUtil-S", "-", "-", s)
    for s in m["seeds_sweep"]:
        for L in m["L"]:
            for meth in ["RACO", "RACO-noRec"]:
                add("pareto", meth, "L_target_years", L, s, L_target_years=L)
        for V in m["V"]:
            add("V", "RACO", "V", V, s, V=V)
        for tr in m["tau_r"]:
            for meth in m["tau_r_methods"]:
                add("tau_r", meth, "tau_r_h", tr, s, tau_r_h=tr)
        for c in ["Tabriz", "Rasht", "Tehran", "BandarAbbas"]:
            for meth in m["climate_methods"]:
                add("climate", meth, "climate", c, s, climate=c)
        for ld in m["load"]:
            for meth in m["load_methods"]:
                add("load", meth, "load_level", ld, s, load_level=ld)
        for meth in m["comp_methods"]:
            add("compensation", meth, "compensation", "fixedV", s, compensation="fixedV")
        for dl in m["delta"]:
            for meth in m["delta_methods"]:
                add("delta", meth, "delta", dl, s, delta=dl)
        for ne in m["n_exp"]:
            for meth in m["n_methods"]:
                add("n_exp", meth, "n_exp", ne, s, n_exp=ne)
        for lr in m["lref"]:
            for meth in m["lref_methods"]:
                add("lref", meth, "T_ref_years", lr, s, T_ref_years=lr)
    return J


# revision of the safety rule; runs of the methods that use it are cached under a new name
SAFETY_REV = 2
SAFETY_METHODS = ("RACO", "RACO-noRec", "RACO-noTherm")


def _key(j):
    v = j["value"]
    v = "inf" if isinstance(v, float) and np.isinf(v) else v
    k = f"{j['exp']}__{j['method']}__{j['param']}={v}__s{j['seed']}"
    if j["method"] in SAFETY_METHODS:
        k += f"_rev{SAFETY_REV}"
    return k


def load_cached(mode, base, out_dir):
    """Cached results of every job of the suite (used by --plots-only)."""
    raw = os.path.join(out_dir, "raw")
    h = _cfg_hash(base)
    res = []
    for j in build_jobs(mode, base):
        p = os.path.join(raw, _key(j) + "__" + h + ".pkl")
        if os.path.exists(p):
            with open(p, "rb") as f:
                res.append(pickle.load(f))
    return res


def _cfg_hash(base):
    """Fingerprint of the base configuration: changing any setting gives new cache files."""
    import hashlib
    from dataclasses import asdict
    d = {k: v for k, v in asdict(base).items() if k not in ("seed", "detailed_log")}
    return hashlib.md5(repr(sorted(d.items())).encode()).hexdigest()[:8]


def run_job(args):
    j, base, raw_dir = args
    path = os.path.join(raw_dir, _key(j) + "__" + _cfg_hash(base) + ".pkl")
    if os.path.exists(path):
        with open(path, "rb") as f:
            return pickle.load(f)
    cfg = replace(base, seed=j["seed"], detailed_log=j["log"], **j["ov"])
    res, nodes, log = Sim(cfg).run(make_policy(j["method"]))
    out = dict(job=j, res=res, nodes=nodes, log=log, cfg=cfg.to_dict())
    with open(path + ".tmp", "wb") as f:
        pickle.dump(out, f)
    os.replace(path + ".tmp", path)
    return out


def run_all_jobs(mode, base, out_dir, jobs_n=None, verbose=True):
    raw = os.path.join(out_dir, "raw")
    os.makedirs(raw, exist_ok=True)
    J = build_jobs(mode, base)
    # longest first (Lyapunov policies are slower)
    J.sort(key=lambda j: (j["method"] in ("LB", "CONS", "TAC", "WL")))
    n = jobs_n or os.cpu_count() or 1
    results = []
    t0 = time.time()
    with Pool(n) as p:
        for i, r in enumerate(p.imap_unordered(run_job, [(j, base, raw) for j in J])):
            results.append(r)
            if verbose:
                jj = r["job"]
                print(f"[{i + 1:4d}/{len(J)}] {time.time() - t0:7.0f}s  {jj['exp']:12s} {jj['method']:13s} "
                      f"{jj['param']}={jj['value']} seed={jj['seed']}  life={r['res']['network_lifetime_years']:.2f}y "
                      f"E={r['res']['energy_MWh']:.2f}MWh SLA={r['res']['sla_violation_pct']:.2f}%", flush=True)
    return results


# --------------------------------------------------------------------------- E10 scalability
def scalability(mode, base, days=30):
    rows = []
    for N in MODES[mode]["scal_N"]:
        for meth in ["CONS", "RACO"]:
            cfg = replace(base, n_nodes=N, horizon_years=days / 365.0)
            sim = Sim(cfg)
            t0 = time.perf_counter()
            sim.run(make_policy(meth))
            dt = time.perf_counter() - t0
            rows.append(dict(method=meth, n_nodes=N, frames=cfg.n_frames, total_s=dt, ms_per_frame=1000 * dt / cfg.n_frames))
            print(f"scalability N={N:4d} {meth:5s} {1000 * dt / cfg.n_frames:.3f} ms/frame", flush=True)
    return rows


# --------------------------------------------------------------------------- E11 solver gap
class _GapProbe(LyapunovPolicy):
    def __init__(self, frames):
        super().__init__("RACO")
        self.frames = set(frames)
        self.rows = []

    def decide(self, sim, k, lam_f, amb, cap):
        s = super().decide(sim, k, lam_f, amb, cap)
        if k not in self.frames:
            return s
        cfg = sim.cfg
        grid = self._last_grid
        wake = self.wake_E * (~sim.s_prev)
        alive = sim.alive
        need = (1 + cfg.delta) * lam_f
        forced = (self.Z >= self.Z_th) & alive
        elig = np.where(alive & ~forced)[0]
        if cap[elig].sum() < need:
            return s
        J_g = self._J(s, lam_f, cap, grid, wake, alive)
        best = np.inf
        for m in range(1, len(elig) + 1):
            for comb in itertools.combinations(elig, m):
                t = np.zeros(sim.N, bool)
                t[list(comb)] = True
                if cap[t].sum() < need:
                    continue
                Jt = self._J(t, lam_f, cap, grid, wake, alive)
                best = min(best, Jt)
        self.rows.append(dict(frame=k, day=k * cfg.frame_h / 24, J_greedy_ls=J_g, J_opt=best,
                              rel_gap_pct=100 * (J_g - best) / abs(best), n_active=int(s.sum())))
        return s


def solver_gap(mode, base, days=120):
    cfg = replace(base, n_nodes=10, horizon_years=days / 365.0)
    rng = np.random.default_rng(3)
    lo = int(30 * 24 / cfg.frame_h)
    frames = rng.choice(np.arange(lo, cfg.n_frames), MODES[mode]["gap_frames"], replace=False)
    pol = _GapProbe(frames)
    Sim(cfg).run(pol)
    # greedy without local search on the same instances
    cfg2 = replace(cfg, local_search=False)
    pol2 = _GapProbe(frames)
    Sim(cfg2).run(pol2)
    g2 = {r["frame"]: r for r in pol2.rows}
    for r in pol.rows:
        if r["frame"] in g2:
            r["rel_gap_greedy_only_pct"] = g2[r["frame"]]["rel_gap_pct"]
    print(f"solver gap: {len(pol.rows)} frames, mean gap {np.mean([r['rel_gap_pct'] for r in pol.rows]):.3f}%", flush=True)
    return pol.rows
