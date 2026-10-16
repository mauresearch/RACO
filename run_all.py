#!/usr/bin/env python3
"""
RACO: Recovery-Aware, thermally-coupled Aging/energy Co-Optimisation for fog scheduling.
Runs all experiments, draws all figures and writes one Excel workbook.

Examples
  python run_all.py --mode quick                 # 1 seed, reduced sweeps
  python run_all.py --mode full --jobs 8         # 5 seeds main / 3 seeds sweeps (results reported in the paper)
  python run_all.py --mode full --workload-csv data/azure_functions_iot.csv \
                    --weather-csv data/weather_Tehran.csv
  python run_all.py --plots-only                 # re-draw from cached results/raw
"""
import argparse
import os
import pickle
import sys
import time

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from raco.config import Config                       # noqa: E402
from raco.simulator import Sim                       # noqa: E402
from raco import experiments as ex                   # noqa: E402
from raco import plots, excel                        # noqa: E402


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", default="quick", choices=list(ex.MODES))
    ap.add_argument("--jobs", type=int, default=None, help="parallel worker processes (default: all cores)")
    ap.add_argument("--out", default="results")
    ap.add_argument("--years", type=float, default=None, help="simulated horizon (default 6)")
    ap.add_argument("--frame-h", type=float, default=None, help="frame length in hours (default 1)")
    ap.add_argument("--climate", default=None, choices=["Tehran", "BandarAbbas", "Tabriz", "Rasht"])
    ap.add_argument("--workload-csv", default=None, help="prepared workload CSV (timestamp,value)")
    ap.add_argument("--weather-csv", default=None, help="Open-Meteo hourly CSV for the base city")
    ap.add_argument("--weather-dir", default=None, help="folder with weather_<City>.csv for all four cities")
    ap.add_argument("--forecast", default=None, choices=["last", "max3", "hybrid"])
    ap.add_argument("--delta", type=float, default=None, help="capacity head-room over the forecast")
    ap.add_argument("--plots-only", action="store_true")
    ap.add_argument("--skip-scal", action="store_true")
    ap.add_argument("--skip-gap", action="store_true")
    a = ap.parse_args(argv)

    kw = {}
    if a.years: kw["horizon_years"] = a.years
    if a.frame_h: kw["frame_h"] = a.frame_h
    if a.climate: kw["climate"] = a.climate
    if a.forecast: kw["forecast"] = a.forecast
    if a.delta is not None: kw["delta"] = a.delta
    if a.workload_csv: kw.update(workload_source="csv", workload_csv=os.path.abspath(a.workload_csv))
    if a.weather_dir: kw["weather_dir"] = os.path.abspath(a.weather_dir)
    if a.weather_csv: kw.update(weather_source="csv", weather_csv=os.path.abspath(a.weather_csv))
    base = Config(**kw)
    out = os.path.abspath(a.out)
    raw = os.path.join(out, "raw")
    figs = os.path.join(out, "figures")
    os.makedirs(raw, exist_ok=True)
    t0 = time.time()

    print(f"a_crit = {base.a_crit * 1e3:.2f} mV, horizon = {base.horizon_years} y, frames = {base.n_frames}, mode = {a.mode}")
    if a.plots_only:
        results = ex.load_cached(a.mode, base, out)
    else:
        results = ex.run_all_jobs(a.mode, base, out, a.jobs)

    # ------------------------------------------------------------- scalability / solver gap (cached)
    def cached(name, fn):
        p = os.path.join(raw, f"{name}_rev{ex.SAFETY_REV}__{ex._cfg_hash(base)}.pkl")
        if os.path.exists(p):
            with open(p, "rb") as fh:
                return pickle.load(fh)
        if a.plots_only:
            return []
        v = fn()
        with open(p, "wb") as fh:
            pickle.dump(v, fh)
        return v

    scal = [] if a.skip_scal else cached("scalability", lambda: ex.scalability(a.mode, base))
    gap = [] if a.skip_gap else cached("solver_gap", lambda: ex.solver_gap(a.mode, base))

    # ------------------------------------------------------------- tables
    rows, logs, nodes = [], {}, {}
    for r in results:
        j = r["job"]
        rows.append(dict(exp=j["exp"], method=j["method"], param=j["param"], value=j["value"], seed=j["seed"], **r["res"]))
        if r["log"] is not None and j["exp"] == "main":
            logs[j["method"]] = r["log"]
            nodes[j["method"]] = r["nodes"]
    df = pd.DataFrame(rows)
    order = {m: i for i, m in enumerate(plots.MAIN + plots.ABL)}
    df["_o"] = df.method.map(order)
    df = df.sort_values(["exp", "_o", "seed"], kind="stable").drop(columns="_o").reset_index(drop=True)
    cfg0 = base.to_dict()

    # ------------------------------------------------------------- figures + excel
    sim0 = Sim(base)
    plots.make_all(df, logs, nodes, cfg0, gap, scal, sim0, figs)
    xlsx = os.path.join(out, "results_RACO.xlsx")
    excel.write_excel(xlsx, df, cfg0, nodes, logs, gap, scal)

    # ------------------------------------------------------------- console summary
    m = df[df.exp == "main"].groupby("method")[["energy_MWh", "network_lifetime_years", "median_node_life_years",
                                                 "sla_violation_pct", "carbon_total_t"]].mean()
    print("\n=== Main comparison (mean over seeds) ===")
    print(m.reindex([x for x in plots.MAIN if x in m.index]).round(3).to_string())
    print(f"\nFigures : {figs}\nExcel   : {xlsx}\nTotal time: {(time.time() - t0) / 60:.1f} min")


if __name__ == "__main__":
    main()
