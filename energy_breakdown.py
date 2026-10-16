# -*- coding: utf-8 -*-
"""
Energy breakdown (board base, leakage, dynamic, deep sleep, wake-up) of the main methods.

Usage (same inputs as one scenario of run_all.py):
    python energy_breakdown.py --workload-csv data/azure_functions_iot.csv --weather-dir data --out results/azure_weather
Writes <out>/energy_breakdown.xlsx (per replication and mean over replications).
"""
import argparse
import os
import sys
from dataclasses import replace
from multiprocessing import Pool

import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from raco.config import Config            # noqa: E402
from raco.simulator import Sim            # noqa: E402
from raco.policies import make_policy     # noqa: E402

METHODS = ["LB", "CONS", "TAC", "WL", "AgingUtil", "RACO"]
COMP = ["E_base_MWh", "E_leak_MWh", "E_dyn_MWh", "E_sleep_MWh", "E_wake_MWh", "energy_MWh"]


def _run(job):
    base, method, seed = job
    res, _, _ = Sim(replace(base, seed=seed)).run(make_policy(method))
    return dict(method=method, seed=seed, **{k: res[k] for k in COMP})


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--workload-csv", default="")
    ap.add_argument("--weather-dir", default="")
    ap.add_argument("--seeds", type=int, default=8)
    ap.add_argument("--jobs", type=int, default=max(1, (os.cpu_count() or 2) - 1))
    ap.add_argument("--out", default="results")
    a = ap.parse_args(argv)
    kw = {}
    if a.workload_csv:
        kw.update(workload_source="csv", workload_csv=os.path.abspath(a.workload_csv))
    if a.weather_dir:
        kw["weather_dir"] = os.path.abspath(a.weather_dir)
    base = Config(**kw)
    jobs = [(base, m, s) for s in range(a.seeds) for m in METHODS]
    rows = []
    with Pool(a.jobs) as p:
        for i, r in enumerate(p.imap_unordered(_run, jobs), 1):
            rows.append(r)
            print(f"[{i}/{len(jobs)}] {r['method']:10s} seed={r['seed']}  E={r['energy_MWh']:.2f} MWh", flush=True)
    df = pd.DataFrame(rows).sort_values(["method", "seed"])
    mean = df.groupby("method")[COMP].mean().loc[METHODS].reset_index()
    os.makedirs(a.out, exist_ok=True)
    path = os.path.join(a.out, "energy_breakdown.xlsx")
    with pd.ExcelWriter(path) as xw:
        mean.to_excel(xw, sheet_name="Mean", index=False)
        df.to_excel(xw, sheet_name="PerSeed", index=False)
    print(mean.round(3).to_string(index=False))
    print("written", path)


if __name__ == "__main__":
    main()
