"""Publication figures (PNG 300 dpi + vector PDF)."""
import os
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

COL = {"LB": "#8C8C8C", "CONS": "#E69F00", "TAC": "#56B4E9", "WL": "#009E73", "AgingUtil": "#CC79A7",
       "RACO": "#D55E00", "RACO-noRec": "#0072B2", "RACO-noTherm": "#882255", "RACO-noSafety": "#44AA99",
       "EnergyOpt": "#117733"}
LABEL = {"LB": "LB", "CONS": "CONS (energy-aware)", "TAC": "TAC (thermal-aware)", "WL": "WL (wear-levelling)",
         "AgingUtil": "AgingUtil (Lin et al.-style)", "RACO": "RACO (proposed)", "RACO-noRec": "w/o recovery",
         "RACO-noTherm": "w/o thermal", "RACO-noSafety": "w/o safety rule", "EnergyOpt": "w/o aging term"}
TYPE_COL = {"SBC": "#E69F00", "MiniPC": "#0072B2", "EdgeSrv": "#009E73"}
MAIN = ["LB", "CONS", "TAC", "WL", "AgingUtil", "RACO"]
ABL = ["RACO", "RACO-noRec", "RACO-noTherm", "RACO-noSafety", "EnergyOpt"]

plt.rcParams.update({
    "font.family": "DejaVu Sans", "font.size": 9, "axes.titlesize": 9.5, "axes.labelsize": 9,
    "legend.fontsize": 7.5, "xtick.labelsize": 8, "ytick.labelsize": 8, "axes.spines.top": False,
    "axes.spines.right": False, "axes.grid": True, "grid.alpha": 0.25, "grid.linewidth": 0.5,
    "savefig.bbox": "tight", "figure.dpi": 110,
})


def _save(fig, out, name):
    os.makedirs(out, exist_ok=True)
    fig.savefig(os.path.join(out, name + ".png"), dpi=300)
    fig.savefig(os.path.join(out, name + ".pdf"))
    plt.close(fig)


def _agg(df, by, metric):
    g = df.groupby(by)[metric]
    return g.mean(), g.std().fillna(0.0)


def _bars(ax, methods, mean, std, ylabel, title=None, fmt="{:.2f}"):
    x = np.arange(len(methods))
    vals = [mean.get(m, np.nan) for m in methods]
    errs = [std.get(m, 0.0) for m in methods]
    ax.bar(x, vals, yerr=errs, color=[COL[m] for m in methods], capsize=2.5, width=0.7,
           error_kw=dict(lw=0.8), edgecolor="none")
    for xi, v in zip(x, vals):
        if np.isfinite(v):
            ax.text(xi, v, fmt.format(v), ha="center", va="bottom", fontsize=6.5)
    ax.set_xticks(x)
    ax.set_xticklabels([m.replace("RACO-", "") for m in methods], rotation=35, ha="right")
    ax.set_ylabel(ylabel)
    if title:
        ax.set_title(title)
    ax.grid(axis="x", visible=False)


# =============================================================================
def fig_inputs(sim, out):
    fh = sim.cfg.frame_h
    per_day = int(24 / fh)
    fig, ax = plt.subplots(1, 3, figsize=(10, 2.6))
    w0 = max(0, min(200 * per_day, len(sim.lam) - 7 * per_day))
    t = np.arange(7 * per_day) * fh / 24
    ax[0].plot(t, sim.lam[w0:w0 + 7 * per_day], color="#333333", lw=1)
    ax[0].axhline(sim.cap_nom.sum(), color="#D55E00", ls="--", lw=0.8, label="SLA-feasible capacity")
    ax[0].set_xlabel("Day"); ax[0].set_ylabel("Demand [req/s]"); ax[0].set_title("(a) Workload, one week"); ax[0].legend()
    nd = sim.amb.shape[0] // per_day
    daily = sim.amb[: nd * per_day].reshape(nd, per_day, -1).mean(axis=1)
    days = np.arange(nd) / 365
    for site, c in [("indoor", "#0072B2"), ("factory", "#E69F00"), ("outdoor", "#D55E00")]:
        cols = sim.sites == site
        ax[1].plot(days[:730], daily[:730, cols].mean(axis=1), color=c, lw=0.9, label=site)
    ax[1].set_xlabel("Year"); ax[1].set_ylabel("Enclosure air [°C]"); ax[1].set_title(f"(b) Ambient by site ({sim.cfg.climate})"); ax[1].legend()
    s0 = max(0, min(196 * per_day, len(sim.lam) - 7 * per_day))
    for site, c in [("indoor", "#0072B2"), ("factory", "#E69F00"), ("outdoor", "#D55E00")]:
        cols = sim.sites == site
        ax[2].plot(t, sim.amb[s0:s0 + 7 * per_day, cols].mean(axis=1), color=c, lw=0.9, label=site)
    ax[2].set_xlabel("Day (summer week)"); ax[2].set_ylabel("Enclosure air [°C]"); ax[2].set_title("(c) Diurnal ambient")
    _save(fig, out, "fig01_inputs")


def fig_aging_traj(logs, nodes, cfg, out, methods=("LB", "CONS", "WL", "RACO")):
    methods = [m for m in methods if m in logs]
    fig, ax = plt.subplots(1, len(methods), figsize=(2.7 * len(methods), 2.6), sharey=True)
    ax = np.atleast_1d(ax)
    for a_, m in zip(ax, methods):
        lg = logs[m]
        yrs = np.arange(lg["a"].shape[0]) / 365
        for i, t in enumerate(nodes[m]["type"]):
            a_.plot(yrs, lg["a"][:, i] * 1e3, color=TYPE_COL[t], lw=0.6, alpha=0.8)
        a_.axhline(cfg["a_crit_mV"], color="k", ls="--", lw=0.8)
        a_.axvline(cfg["L_target_years"], color="#D55E00", ls=":", lw=0.8)
        a_.set_title(LABEL[m]); a_.set_xlabel("Year")
    ax[0].set_ylabel("ΔVth = p + r [mV]")
    h = [Line2D([], [], color=c, lw=1.2, label=t) for t, c in TYPE_COL.items()]
    h += [Line2D([], [], color="k", ls="--", lw=0.8, label="a_crit"), Line2D([], [], color="#D55E00", ls=":", lw=0.8, label="L_target")]
    ax[-1].legend(handles=h, loc="lower right")
    _save(fig, out, "fig02_aging_trajectories")


def fig_survival(logs, cfg, out):
    fig, ax = plt.subplots(figsize=(4.2, 2.8))
    for m in MAIN:
        if m in logs:
            al = logs[m]["alive"]
            N = al.max() if al.max() > 0 else 1
            ax.plot(np.arange(len(al)) / 365, al / N, color=COL[m], lw=2.0 if m == "RACO" else 1.1, label=LABEL[m])
    ax.axvline(cfg["L_target_years"], color="k", ls=":", lw=0.8)
    ax.set_xlabel("Year"); ax.set_ylabel("Fraction of nodes alive"); ax.set_ylim(0, 1.03)
    ax.legend(loc="lower left")
    _save(fig, out, "fig03_survival")


def fig_main_bars(df, out):
    d = df[df.exp == "main"]
    mets = [("energy_MWh", "Energy over horizon [MWh]"), ("network_lifetime_years", "Network lifetime [yr]"),
            ("median_node_life_years", "Median node lifetime [yr]"), ("sla_violation_pct", "SLA violation [%]"),
            ("aging_gini", "Aging Gini index"), ("carbon_total_t", "Lifecycle carbon [t CO2e]")]
    fig, ax = plt.subplots(2, 3, figsize=(9.5, 5.2))
    for a_, (met, lab) in zip(ax.ravel(), mets):
        mu, sd = _agg(d, "method", met)
        _bars(a_, [m for m in MAIN if m in mu.index], mu, sd, lab, fmt="{:.2f}" if met != "aging_gini" else "{:.3f}")
    fig.tight_layout()
    _save(fig, out, "fig04_main_comparison")


def fig_pareto(df, out, sla_max=1.0):
    """Only operating points that meet the service level (SLA violation <= sla_max %) are drawn;
    longer lifetimes obtained by shedding demand to the cloud are not a valid trade-off."""
    fig, ax = plt.subplots(figsize=(4.6, 3.3))
    p = df[df.exp == "pareto"]
    for m, mk in [("RACO", "o"), ("RACO-noRec", "s")]:
        q = p[p.method == m].groupby("value")[["network_lifetime_years", "energy_MWh", "sla_violation_pct"]].mean()
        q.index = q.index.astype(float); q = q.sort_index()
        if not len(q):
            continue
        q = q[q.sla_violation_pct <= sla_max]
        ax.plot(q.network_lifetime_years, q.energy_MWh, marker=mk, ms=4, color=COL[m], lw=1.6 if m == "RACO" else 1.0,
                label=LABEL[m] + f" (L_target sweep, SLA ≤ {sla_max:g}%)")
        for L, row in q.iterrows():
            if L not in (4, 6, 8, 12, 16):
                continue
            ax.annotate(f"L={L:g}", (row.network_lifetime_years, row.energy_MWh), fontsize=6, xytext=(3, -8 if m == "RACO" else 3),
                        textcoords="offset points", color=COL[m])
    d = df[df.exp == "main"].groupby("method")[["network_lifetime_years", "energy_MWh"]].mean()
    for m in MAIN[:-1]:
        if m in d.index:
            ax.scatter(d.loc[m, "network_lifetime_years"], d.loc[m, "energy_MWh"], color=COL[m], marker="D", s=30,
                       zorder=5, label=LABEL[m])
    ax.set_xlabel("Network lifetime (first node failure) [yr]"); ax.set_ylabel("Energy over horizon [MWh]")
    ax.set_title("Energy–lifetime trade-off (lower-right is better)")
    ax.legend(fontsize=6.5)
    _save(fig, out, "fig05_pareto_energy_lifetime")


def fig_schedule(logs, nodes, out, methods=("CONS", "WL", "RACO")):
    methods = [m for m in methods if m in logs]
    fig, ax = plt.subplots(len(methods) + 1, 1, figsize=(8, 1.1 + 1.6 * len(methods)), sharex=True,
                           gridspec_kw=dict(height_ratios=[0.6] + [1] * len(methods)))
    lg0 = logs[methods[0]]
    T = len(lg0["week_lam"])
    t = np.arange(T) * 7 / T
    ax[0].plot(t, lg0["week_lam"], color="#333333", lw=0.9); ax[0].set_ylabel("Demand\n[req/s]")
    ax[0].set_title(f"Active (dark) / deep-sleep (light) schedule, week starting day {lg0['week_start_day']}")
    for a_, m in zip(ax[1:], methods):
        order = np.argsort([["SBC", "MiniPC", "EdgeSrv"].index(x) for x in nodes[m]["type"]], kind="stable")
        a_.imshow(logs[m]["week_s"][:, order].T, aspect="auto", cmap="Greys", vmin=-0.3, vmax=1.0,
                  extent=[0, 7, len(order) - 0.5, -0.5], interpolation="nearest")
        a_.set_ylabel(f"{m}\nnode"); a_.grid(False)
        types = np.array(nodes[m]["type"])[order]
        for tt in ["MiniPC", "EdgeSrv"]:
            idx = np.where(types == tt)[0]
            if len(idx):
                a_.axhline(idx[0] - 0.5, color="#D55E00", lw=0.6)
    ax[-1].set_xlabel("Day")
    _save(fig, out, "fig06_schedule_heatmap")


def fig_ablation(df, out):
    d = pd.concat([df[(df.exp == "main") & (df.method == "RACO")], df[df.exp == "ablation"]])
    fig, ax = plt.subplots(1, 4, figsize=(10, 2.7))
    for a_, (met, lab) in zip(ax, [("network_lifetime_years", "Network lifetime [yr]"), ("energy_MWh", "Energy [MWh]"),
                                   ("sla_violation_pct", "SLA violation [%]"), ("n_failed_by_L", "Nodes failed before L_target")]):
        mu, sd = _agg(d, "method", met)
        _bars(a_, [m for m in ABL if m in mu.index], mu, sd, lab)
    fig.tight_layout()
    _save(fig, out, "fig07_ablation")


def _line_sweep(ax, d, xcol, ycol, methods, xlog=False, xticks=None):
    for m in methods:
        q = d[d.method == m]
        if not len(q):
            continue
        g = q.groupby("value")[ycol]
        mu, sd = g.mean(), g.std().fillna(0)
        xs = np.arange(len(mu)) if xticks == "cat" else np.array(mu.index, dtype=float)
        ax.errorbar(xs, mu.values, yerr=sd.values, color=COL[m], marker="o", ms=3, lw=1.8 if m == "RACO" else 1.0,
                    capsize=2, label=LABEL[m])
        if xticks == "cat":
            ax.set_xticks(xs); ax.set_xticklabels([("∞" if (isinstance(v, float) and np.isinf(v)) else f"{v:g}") for v in mu.index])
    if xlog:
        ax.set_xscale("log")


def fig_V(df, out):
    d = df[df.exp == "V"]
    if not len(d):
        return
    fig, ax = plt.subplots(1, 3, figsize=(9, 2.6))
    for a_, (met, lab) in zip(ax, [("energy_MWh", "Energy [MWh]"), ("network_lifetime_years", "Network lifetime [yr]"),
                                   ("wakeups_per_node_day", "Wake-ups / node / day")]):
        _line_sweep(a_, d, "value", met, ["RACO"], xlog=True)
        a_.set_xlabel("Lyapunov weight V"); a_.set_ylabel(lab)
    fig.tight_layout()
    _save(fig, out, "fig08_V_tradeoff")


def fig_sweep(df, exp, xlabel, name, methods, out, cat=False, mets=None):
    d = df[df.exp == exp]
    if not len(d):
        return
    mets = mets or [("network_lifetime_years", "Network lifetime [yr]"), ("energy_MWh", "Energy [MWh]"),
                    ("sla_violation_pct", "SLA violation [%]")]
    fig, ax = plt.subplots(1, len(mets), figsize=(3.1 * len(mets), 2.7))
    for a_, (met, lab) in zip(ax, mets):
        _line_sweep(a_, d, "value", met, [m for m in methods if m in d.method.unique()], xticks="cat" if cat else None)
        a_.set_xlabel(xlabel); a_.set_ylabel(lab)
    ax[0].legend(fontsize=6.5)
    fig.tight_layout()
    _save(fig, out, name)


def fig_climate(df, out):
    d = df[df.exp == "climate"]
    if not len(d):
        return
    cities = ["Tabriz", "Rasht", "Tehran", "BandarAbbas"]
    meths = [m for m in MAIN if m in d.method.unique()]
    fig, ax = plt.subplots(1, 2, figsize=(9, 2.9))
    w = 0.8 / len(meths)
    for a_, (met, lab) in zip(ax, [("network_lifetime_years", "Network lifetime [yr]"), ("energy_MWh", "Energy [MWh]")]):
        for j, m in enumerate(meths):
            g = d[d.method == m].groupby("value")[met]
            mu, sd = g.mean().reindex(cities), g.std().reindex(cities).fillna(0)
            a_.bar(np.arange(4) + (j - (len(meths) - 1) / 2) * w, mu.values, w, yerr=sd.values, color=COL[m],
                   label=LABEL[m], capsize=1.5, error_kw=dict(lw=0.6))
        a_.set_xticks(np.arange(4)); a_.set_xticklabels(cities); a_.set_ylabel(lab); a_.grid(axis="x", visible=False)
    ax[0].legend(fontsize=6.5, ncol=5, loc="upper center", bbox_to_anchor=(1.1, -0.12), frameon=False)
    fig.tight_layout()
    _save(fig, out, "fig10_sensitivity_climate")


def fig_temp_aging(nodes, out, methods=("CONS", "WL", "RACO")):
    fig, ax = plt.subplots(figsize=(4.2, 3))
    mk = {"SBC": "o", "MiniPC": "s", "EdgeSrv": "^"}
    for m in methods:
        if m not in nodes:
            continue
        n = nodes[m]
        for t in mk:
            sel = np.array(n["type"]) == t
            ax.scatter(np.array(n["mean_theta_active"])[sel], np.array(n["life_years"])[sel], color=COL[m], marker=mk[t],
                       s=18, alpha=0.85, edgecolor="none")
    h = [Line2D([], [], color=COL[m], marker="o", ls="", label=LABEL[m]) for m in methods if m in nodes]
    h += [Line2D([], [], color="grey", marker=v, ls="", label=k) for k, v in mk.items()]
    ax.set_xlabel("Mean junction temperature while active [°C]"); ax.set_ylabel("Node lifetime [yr]")
    ax.legend(handles=h, fontsize=6.5)
    _save(fig, out, "fig12_temperature_vs_lifetime")


def fig_gap(gap, out):
    if not gap:
        return
    g = pd.DataFrame(gap)
    fig, ax = plt.subplots(figsize=(4, 2.7))
    for col, c, lab in [("rel_gap_pct", "#D55E00", "greedy + local search"), ("rel_gap_greedy_only_pct", "#0072B2", "greedy only")]:
        if col in g:
            x = np.sort(g[col].dropna().values)
            ax.plot(x, np.arange(1, len(x) + 1) / len(x), color=c, label=lab)
    ax.set_xlabel("Per-frame optimality gap vs exhaustive [%]"); ax.set_ylabel("CDF"); ax.legend()
    _save(fig, out, "fig13_solver_gap_cdf")


def fig_scal(scal, out):
    if not scal:
        return
    s = pd.DataFrame(scal)
    fig, ax = plt.subplots(figsize=(4, 2.7))
    for m in s.method.unique():
        q = s[s.method == m]
        ax.plot(q.n_nodes, q.ms_per_frame, marker="o", color=COL[m], label=LABEL[m])
    ax.set_xscale("log"); ax.set_yscale("log")
    ax.set_xlabel("Number of fog nodes N"); ax.set_ylabel("Decision time per frame [ms]"); ax.legend()
    _save(fig, out, "fig14_scalability")


def fig_carbon(df, out):
    d = df[df.exp == "main"].groupby("method")[["carbon_op_t", "carbon_emb_t"]].mean()
    meths = [m for m in MAIN if m in d.index]
    fig, ax = plt.subplots(figsize=(4.4, 2.8))
    x = np.arange(len(meths))
    ax.bar(x, d.loc[meths, "carbon_op_t"], color="#56B4E9", label="operational")
    ax.bar(x, d.loc[meths, "carbon_emb_t"], bottom=d.loc[meths, "carbon_op_t"], color="#CC79A7", label="embodied (amortised)")
    ax.set_xticks(x); ax.set_xticklabels(meths, rotation=30, ha="right"); ax.set_ylabel("t CO2e over horizon")
    ax.legend(); ax.grid(axis="x", visible=False)
    _save(fig, out, "fig15_lifecycle_carbon")


def fig_comp(df, out):
    a = df[df.exp == "main"].copy(); a["mode"] = "AVS"
    b = df[df.exp == "compensation"].copy(); b["mode"] = "fixed V"
    if not len(b):
        return
    meths = [m for m in MAIN if m in b.method.unique()]
    fig, ax = plt.subplots(1, 3, figsize=(9.5, 2.7))
    for a_, (met, lab) in zip(ax, [("network_lifetime_years", "Network lifetime [yr]"), ("energy_MWh", "Energy [MWh]"),
                                   ("sla_violation_pct", "SLA violation [%]")]):
        for j, (dd, hatch, lb) in enumerate([(a, "", "AVS"), (b, "///", "fixed V")]):
            mu = dd.groupby("method")[met].mean().reindex(meths)
            a_.bar(np.arange(len(meths)) + (j - 0.5) * 0.38, mu.values, 0.38, color=[COL[m] for m in meths], hatch=hatch,
                   edgecolor="white", label=lb)
        a_.set_xticks(np.arange(len(meths))); a_.set_xticklabels(meths, rotation=30, ha="right"); a_.set_ylabel(lab)
        a_.grid(axis="x", visible=False)
    ax[0].legend(handles=[plt.Rectangle((0, 0), 1, 1, fc="grey"), plt.Rectangle((0, 0), 1, 1, fc="grey", hatch="///", ec="white")],
                 labels=["AVS", "fixed V"])
    fig.tight_layout()
    _save(fig, out, "fig16_compensation_mode")


def fig_queues(logs, nodes, cfg, out):
    if "RACO" not in logs:
        return
    lg, n = logs["RACO"], nodes["RACO"]
    from .physics import tau_of_p
    from .config import Config
    c = Config(**{k: v for k, v in cfg.items() if k in Config.__dataclass_fields__})
    tau_crit = float(tau_of_p(c, c.a_crit - 0.5 * c.r_sat_ref - c.guard_mV * 1e-3))
    Zth = c.Z_th_frac * tau_crit
    yrs = np.arange(lg["Z"].shape[0]) / 365
    fig, ax = plt.subplots(1, 2, figsize=(8.5, 2.8))
    for i, t in enumerate(n["type"]):
        ax[0].plot(yrs, lg["Z"][:, i] / Zth, color=TYPE_COL[t], lw=0.6, alpha=0.8)
        ax[1].plot(yrs, lg["tau"][:, i] / tau_crit, color=TYPE_COL[t], lw=0.6, alpha=0.8)
    ax[0].axhline(1.0, color="k", ls="--", lw=0.8); ax[0].set_ylabel("Z_i / Z_th"); ax[0].set_xlabel("Year")
    ax[0].set_title("(a) Aging virtual queues stay bounded")
    ax[1].plot(yrs, yrs / c.L_target_years, color="k", ls="--", lw=1.0, label="budget b·t")
    ax[1].set_ylabel("τ_i / τ_crit (permanent stress)"); ax[1].set_xlabel("Year"); ax[1].legend()
    ax[1].set_title("(b) Equivalent stress time vs budget")
    h = [Line2D([], [], color=cc, lw=1.2, label=t) for t, cc in TYPE_COL.items()]
    ax[0].legend(handles=h, fontsize=6.5)
    fig.tight_layout()
    _save(fig, out, "fig17_virtual_queues_theorem3")


def make_all(df, logs, nodes, cfg, gap, scal, sim_inputs, out):
    fig_inputs(sim_inputs, out)
    if logs:
        fig_aging_traj(logs, nodes, cfg, out)
        fig_survival(logs, cfg, out)
        fig_schedule(logs, nodes, out)
        fig_queues(logs, nodes, cfg, out)
        fig_temp_aging(nodes, out)
    fig_main_bars(df, out)
    fig_pareto(df, out)
    fig_ablation(df, out)
    fig_V(df, out)
    fig_sweep(df, "tau_r", "Recovery time-constant τ_r [h]", "fig09_sensitivity_recovery", ["RACO", "RACO-noRec", "CONS", "WL"], out, cat=True)
    fig_climate(df, out)
    fig_sweep(df, "load", "Mean load level", "fig11_sensitivity_load", MAIN, out)
    fig_gap(gap, out)
    fig_scal(scal, out)
    fig_carbon(df, out)
    fig_comp(df, out)
    fig_sweep(df, "n_exp", "NBTI time exponent n", "fig18_sensitivity_n_exp", ["CONS", "WL", "RACO"], out)
    fig_sweep(df, "delta", "Capacity head-room δ over the forecast", "fig19_sensitivity_headroom",
              ["CONS", "AgingUtil", "RACO"], out)
