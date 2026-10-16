# -*- coding: utf-8 -*-
"""
Safety-rule statistics of RACO versus the Lyapunov parameter V: share of node-frames forced into deep
sleep (rest-to-recover and queue threshold) and largest virtual queue relative to the threshold.

Usage:
    python safety_statistics.py --workload-csv data/azure_functions_iot.csv --out results/azure
Writes <out>/safety_statistics_V.csv.
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

V_VALUES = [1, 3, 10, 30, 100, 300, 1000]
KEYS = ["energy_MWh", "network_lifetime_years", "sla_violation_pct", "forced_node_pct",
        "forced_queue_node_pct", "forced_frame_pct", "zmax_over_zth"]


def _run(job):
    base, V, seed = job
    res, _, _ = Sim(replace(base, V=V, seed=seed)).run(make_policy("RACO"))
    return dict(V=V, seed=seed, **{k: res[k] for k in KEYS})


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--workload-csv", default="")
    ap.add_argument("--weather-dir", default="")
    ap.add_argument("--seeds", type=int, default=3)
    ap.add_argument("--jobs", type=int, default=max(1, (os.cpu_count() or 2) - 1))
    ap.add_argument("--out", default="results")
    a = ap.parse_args(argv)
    kw = {}
    if a.workload_csv:
        kw.update(workload_source="csv", workload_csv=os.path.abspath(a.workload_csv))
    if a.weather_dir:
        kw["weather_dir"] = os.path.abspath(a.weather_dir)
    base = Config(**kw)
    jobs = [(base, V, s) for s in range(a.seeds) for V in V_VALUES]
    with Pool(a.jobs) as p:
        rows = list(p.imap_unordered(_run, jobs))
    df = pd.DataFrame(rows).sort_values(["V", "seed"])
    os.makedirs(a.out, exist_ok=True)
    path = os.path.join(a.out, "safety_statistics_V.csv")
    df.to_csv(path, index=False)
    print(df.groupby("V")[KEYS].mean().round(3).to_string())
    print("written", path)


if __name__ == "__main__":
    main()
