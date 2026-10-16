"""Figures 2-10 of the paper, drawn from results/<scenario>/results_RACO.xlsx.

Reference scenario: Azure Functions workload with the measured weather of Tehran (results/azure_weather).
Usage:  python paper_figures/make_figures.py
"""
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap

import os
HERE = os.path.dirname(os.path.abspath(__file__))
OUT = HERE + os.sep
WB = {k: os.path.join(HERE, "..", "results", k, "results_RACO.xlsx") for k in ["azure", "alibaba", "azure_weather", "alibaba_weather"]}
X = pd.read_excel(WB["azure_weather"], sheet_name=None)

plt.rcParams.update({
    "font.family": ["Times New Roman", "Liberation Serif", "DejaVu Serif"], "font.size": 9, "axes.labelsize": 9, "legend.fontsize": 7.5,
    "xtick.labelsize": 8, "ytick.labelsize": 8, "axes.spines.top": False, "axes.spines.right": False,
    "axes.grid": True, "grid.alpha": 0.25, "grid.linewidth": 0.5, "savefig.bbox": "tight", "savefig.dpi": 300,
    "mathtext.fontset": "stix",
})
COL = {"LB": "#8C8C8C", "CONS": "#E69F00", "TAC": "#56B4E9", "WL": "#009E73", "AgingUtil": "#CC79A7",
       "RACO": "#D55E00", "RACO-noRec": "#0072B2"}
LAB = {"RACO-noRec": "RACO w/o recovery"}
MAIN = ["LB", "CONS", "TAC", "WL", "AgingUtil", "RACO"]
SENS = ["LB", "CONS", "WL", "AgingUtil", "RACO"]


def lab(m):
    return LAB.get(m, m)


def save(fig, name):
    fig.savefig(OUT + name + ".png")
    plt.close(fig)


# ---------------------------------------------------------------- Fig. 2 main comparison
def fig_main():
    m = X["Main_Summary"].set_index("method").loc[MAIN]
    fig, ax = plt.subplots(1, 4, figsize=(7.2, 2.2))
    x = np.arange(len(MAIN))
    c = [COL[k] for k in MAIN]
    specs = [("energy_MWh", "Energy [MWh]", "(a)"), ("network_lifetime_years", "Network lifetime [years]", "(b)"),
             ("sla_violation_pct", "SLA violations [%]", "(c)")]
    for a, (k, yl, p) in zip(ax[:3], specs):
        a.bar(x, m[k + "_mean"], yerr=m[k + "_std"], color=c, width=0.7, capsize=2, error_kw=dict(lw=0.7))
        a.set_ylabel(yl); a.set_xlabel(p)
    ax[0].set_ylim(15, 27.5)
    ax[1].axhline(5, color="k", ls="--", lw=0.8)
    ax[1].text(-0.4, 5.6, "$L$ = 5", fontsize=7.5)
    a = ax[3]
    a.bar(x, m["carbon_op_t_mean"], color=c, width=0.7)
    a.bar(x, m["carbon_emb_t_mean"], bottom=m["carbon_op_t_mean"], color=c, width=0.7, alpha=0.4)
    a.set_ylabel("Lifecycle carbon [t CO$_2$e]"); a.set_xlabel("(d)")
    h1 = plt.Rectangle((0, 0), 1, 1, color="#777777"); h2 = plt.Rectangle((0, 0), 1, 1, color="#777777", alpha=0.4)
    a.legend([h1, h2], ["operational", "embodied"], fontsize=6.5, loc="upper right", framealpha=0.9)
    a.set_ylim(0, 28)
    for a in ax:
        a.set_xticks(x); a.set_xticklabels(MAIN, rotation=55, ha="right", fontsize=7.5)
        a.grid(axis="x", visible=False)
    fig.tight_layout(w_pad=0.8)
    save(fig, "fig2")


# ---------------------------------------------------------------- Fig. 3 aging and survival over time
def fig_time():
    a_crit = 42.817
    fig, ax = plt.subplots(1, 2, figsize=(7.0, 2.5))
    for m in MAIN:
        d = X["Daily_" + m]
        yr = d["day"] / 365.0
        ax[0].plot(yr, d["max_dVth_mV"], color=COL[m], lw=1.5 if m == "RACO" else 0.9, label=lab(m))
        ax[1].plot(yr, d["alive_nodes"] / 20.0, color=COL[m], lw=1.5 if m == "RACO" else 0.9, label=lab(m))
    ax[0].axhline(a_crit, color="k", ls="--", lw=0.8); ax[0].text(0.05, a_crit + 0.6, "$a^{\\mathrm{crit}}$", fontsize=8)
    ax[0].set_ylabel("Maximum $\\Delta V_{\\mathrm{th}}$ [mV]"); ax[0].set_xlabel("Time [years]\n(a)")
    ax[0].set_ylim(0, 47)
    ax[1].axvline(5, color="k", ls="--", lw=0.8)
    ax[1].set_ylabel("Fraction of operational nodes"); ax[1].set_xlabel("Time [years]\n(b)")
    ax[1].set_ylim(0, 1.05)
    ax[1].legend(loc="lower left", fontsize=7, ncol=2)
    fig.tight_layout(w_pad=1.5)
    save(fig, "fig3")


# ---------------------------------------------------------------- Fig. 4 Pareto
def fig_pareto(sla_max=1.0):
    fig, ax = plt.subplots(figsize=(4.2, 3.0))
    p = X["Pareto_Ltarget"].copy(); p["v"] = p.value.astype(float)
    for m, mk in [("RACO", "o"), ("RACO-noRec", "s")]:
        q = p[(p.method == m) & (p.sla_violation_pct_mean <= sla_max)].sort_values("v")
        ax.plot(q.network_lifetime_years_mean, q.energy_MWh_mean, marker=mk, ms=4, color=COL[m],
                lw=1.6 if m == "RACO" else 1.0, label=lab(m))
        for _, r in q.iterrows():
            if r.v in (4, 8, 12, 16) or (m != "RACO" and r.v in (4, 7)):
                ax.annotate(f"{r.v:g}", (r.network_lifetime_years_mean, r.energy_MWh_mean), fontsize=7,
                            xytext=(2, -9) if m == "RACO" else (2, 4), textcoords="offset points", color=COL[m])
    d = X["Main_Summary"].set_index("method")
    for m in MAIN[:-1]:
        ax.scatter(d.loc[m, "network_lifetime_years_mean"], d.loc[m, "energy_MWh_mean"], color=COL[m], marker="D", s=26, zorder=5, label=m)
    ax.set_xlabel("Network lifetime [years]"); ax.set_ylabel("Energy [MWh]")
    ax.legend(fontsize=7, loc="upper right")
    fig.tight_layout()
    save(fig, "fig4")


# ---------------------------------------------------------------- Fig. 5 weekly schedule
def fig_schedule():
    w = X["Week_Schedule"]
    nodes = [c for c in w.columns if c.startswith("n")]
    fig, ax = plt.subplots(2, 1, figsize=(7.0, 3.6), sharex=True)
    cm = ListedColormap(["#F2F2F2", "#D55E00"])
    for a, m, p in zip(ax, ["RACO", "CONS"], ["(a)", "(b)"]):
        q = w[w.method == m].sort_values("frame")
        a.imshow(q[nodes].values.T, aspect="auto", cmap=cm, interpolation="nearest", vmin=0, vmax=1,
                 extent=[0, len(q) / 24.0, len(nodes) - 0.5, -0.5])
        a.set_yticks([0, 8, 16, 19]); a.set_yticklabels(["SBC 1", "Mini-PC 1", "Server 1", "Server 4"], fontsize=7)
        a.grid(False)
        a.set_xlabel(p) if m == "RACO" else a.set_xlabel("Day of the week\n" + p)
        for b in (8, 16):
            a.axhline(b - 0.5, color="white", lw=1.0)
    ax[0].tick_params(labelbottom=True)
    fig.tight_layout(h_pad=0.6)
    save(fig, "fig5")


# ---------------------------------------------------------------- Fig. 6 load level
def fig_load():
    s = X["Sens_Load"]
    fig, ax = plt.subplots(1, 3, figsize=(7.2, 2.3))
    specs = [("network_lifetime_years", "Network lifetime [years]", "(a)"), ("energy_MWh", "Energy [MWh]", "(b)"),
             ("sla_violation_pct", "SLA violations [%]", "(c)")]
    for a, (k, yl, p) in zip(ax, specs):
        for m in SENS:
            q = s[s.method == m].sort_values("value")
            a.errorbar(q.value, q[k + "_mean"], yerr=q[k + "_std"], color=COL[m], marker="o", ms=3,
                       lw=1.5 if m == "RACO" else 0.9, capsize=2, label=lab(m))
        a.set_ylabel(yl); a.set_xlabel("Load level\n" + p); a.set_xticks([0.25, 0.4, 0.55, 0.65])
    ax[0].axhline(5, color="k", ls="--", lw=0.8)
    ax[0].legend(fontsize=6.5, loc="upper right")
    fig.tight_layout(w_pad=1.0)
    save(fig, "fig6")


# ---------------------------------------------------------------- Fig. 7 climate
def fig_climate():
    s = X["Sens_Climate"]
    cities = ["Tabriz", "Rasht", "Tehran", "BandarAbbas"]
    names = ["Tabriz", "Rasht", "Tehran", "Bandar Abbas"]
    fig, ax = plt.subplots(1, 2, figsize=(7.0, 2.4))
    wdt = 0.16
    for j, m in enumerate(SENS):
        q = s[s.method == m].set_index("value").loc[cities]
        xx = np.arange(4) + (j - 2) * wdt
        ax[0].bar(xx, q.network_lifetime_years_mean, wdt, yerr=q.network_lifetime_years_std, color=COL[m], label=lab(m), capsize=1.5, error_kw=dict(lw=0.6))
        ax[1].bar(xx, q.energy_MWh_mean, wdt, yerr=q.energy_MWh_std, color=COL[m], label=lab(m), capsize=1.5, error_kw=dict(lw=0.6))
    ax[0].axhline(5, color="k", ls="--", lw=0.8)
    ax[0].set_ylabel("Network lifetime [years]"); ax[1].set_ylabel("Energy [MWh]"); ax[1].set_ylim(15, 27.5)
    for a, p in zip(ax, ["(a)", "(b)"]):
        a.set_xticks(range(4)); a.set_xticklabels(names); a.set_xlabel(p); a.grid(axis="x", visible=False)
    h, l = ax[0].get_legend_handles_labels()
    fig.legend(h, l, fontsize=7, ncol=5, loc="upper center", bbox_to_anchor=(0.5, 1.07), frameon=False)
    fig.tight_layout(w_pad=1.5)
    save(fig, "fig7")


# ---------------------------------------------------------------- Fig. 8 recovery time constant
def fig_taur():
    s = X["Sens_TauR"].copy()
    s["v"] = s.value.astype(str).replace({"inf": "∞"})
    order = ["0.5", "2.0", "6.0", "24.0", "∞"]; ticks = ["0.5", "2", "6", "24", "∞"]
    fig, ax = plt.subplots(1, 2, figsize=(7.0, 2.4))
    for m in ["CONS", "WL", "RACO-noRec", "RACO"]:
        q = s[s.method == m].set_index("v").loc[order]
        for a, k in zip(ax, ["network_lifetime_years", "energy_MWh"]):
            a.errorbar(range(5), q[k + "_mean"], yerr=q[k + "_std"], color=COL[m], marker="o", ms=3,
                       lw=1.5 if m == "RACO" else 0.9, capsize=2, label=lab(m))
    ax[0].axhline(5, color="k", ls="--", lw=0.8)
    ax[0].set_ylabel("Network lifetime [years]"); ax[1].set_ylabel("Energy [MWh]")
    for a, p in zip(ax, ["(a)", "(b)"]):
        a.set_xticks(range(5)); a.set_xticklabels(ticks); a.set_xlabel("Recovery time constant $\\tau_r$ [h]\n" + p)
    ax[0].legend(fontsize=6.5, loc="upper right")
    fig.tight_layout(w_pad=1.5)
    save(fig, "fig8")


# ---------------------------------------------------------------- Fig. 9 control parameters
def fig_knobs():
    fig, ax = plt.subplots(1, 2, figsize=(7.0, 2.4))
    d = X["Sens_Delta"]; d = d[d.method == "RACO"].sort_values("value")
    a = ax[0]; b = a.twinx()
    a.plot(d.value, d.energy_MWh_mean, "o-", color=COL["RACO"], ms=3, label="Energy")
    b.plot(d.value, d.sla_violation_pct_mean, "s--", color="#0072B2", ms=3, label="SLA violations")
    a.set_ylabel("Energy [MWh]", color=COL["RACO"]); b.set_ylabel("SLA violations [%]", color="#0072B2")
    a.set_xlabel("Capacity head-room $\\delta$\n(a)"); b.grid(False); b.spines["right"].set_visible(True)
    v = X["Sens_V"].copy(); v["v"] = v.value.astype(float); v = v[v.method == "RACO"].sort_values("v")
    a = ax[1]; b = a.twinx()
    a.plot(v.v, v.energy_MWh_mean, "o-", color=COL["RACO"], ms=3)
    b.plot(v.v, v.network_lifetime_years_mean, "s--", color="#009E73", ms=3)
    a.set_xscale("log"); a.set_ylim(17, 21); b.set_ylim(4, 8)
    a.set_ylabel("Energy [MWh]", color=COL["RACO"]); b.set_ylabel("Network lifetime [years]", color="#009E73")
    a.set_xlabel("Lyapunov parameter $V$\n(b)"); b.grid(False); b.spines["right"].set_visible(True)
    fig.tight_layout(w_pad=2.0)
    save(fig, "fig9")


# ---------------------------------------------------------------- Fig. 10 solver gap and scalability
def fig_solver():
    g = X["SolverGap"]; sc = X["Scalability"]
    fig, ax = plt.subplots(1, 2, figsize=(7.0, 2.4))
    for k, lb, c, ls in [("rel_gap_greedy_only_pct", "Greedy", "#0072B2", "--"), ("rel_gap_pct", "Greedy + local search", COL["RACO"], "-")]:
        v = np.sort(g[k].dropna().values)
        ax[0].step(v, np.arange(1, len(v) + 1) / len(v), where="post", color=c, ls=ls, label=lb)
    ax[0].set_xlabel("Relative optimality gap [%]\n(a)"); ax[0].set_ylabel("Empirical CDF"); ax[0].legend(fontsize=7, loc="lower right")
    for m, c in [("CONS", COL["CONS"]), ("RACO", COL["RACO"])]:
        q = sc[sc.method == m].sort_values("n_nodes")
        ax[1].plot(q.n_nodes, q.ms_per_frame, "o-", color=c, ms=3, label=m)
    ax[1].set_xscale("log"); ax[1].set_xlabel("Number of fog nodes $N$\n(b)"); ax[1].set_ylabel("Decision time per frame [ms]")
    ax[1].legend(fontsize=7)
    fig.tight_layout(w_pad=1.5)
    save(fig, "fig10")


if __name__ == "__main__":
    for f in [fig_main, fig_time, fig_pareto, fig_schedule, fig_load, fig_climate, fig_taur, fig_knobs, fig_solver]:
        f(); print(f.__name__)
