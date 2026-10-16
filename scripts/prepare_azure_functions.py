#!/usr/bin/env python3
"""
Convert the Azure Functions Trace 2019 into a workload CSV (timestamp,value) for run_all.py.

Download (≈ 140 MB):
  https://github.com/Azure/AzurePublicDataset/releases/download/dataset-functions-2019/azurefunctions_dataset2019_azurefunctions-dataset2019.tar.xz

The archive can be given directly (no need to extract it):
  python scripts/prepare_azure_functions.py --src azurefunctions_dataset2019_azurefunctions-dataset2019.tar.xz --iot-only
or a folder that contains the extracted invocations_per_function_md.anon.dXX.csv files:
  python scripts/prepare_azure_functions.py --src path/to/folder --iot-only

Total invocations of all functions per minute (14 days); --iot-only keeps event / queue / timer
triggers, which resemble IoT traffic.
"""
import argparse
import glob
import io
import os
import re
import tarfile

import numpy as np
import pandas as pd

PAT = re.compile(r"invocations_per_function_md\.anon\.d(\d+)\.csv$")


def _iter_tables(src):
    if os.path.isdir(src):
        files = sorted(f for f in glob.glob(os.path.join(src, "**", "*.csv"), recursive=True) if PAT.search(f))
        for f in files:
            yield os.path.basename(f), pd.read_csv(f)
    else:
        with tarfile.open(src, "r:*") as tf:
            members = sorted((m for m in tf.getmembers() if PAT.search(m.name)), key=lambda m: m.name)
            for m in members:
                yield os.path.basename(m.name), pd.read_csv(io.BytesIO(tf.extractfile(m).read()))


def prepare(src, out="data/azure_functions_iot.csv", iot_only=True):
    series = []
    minute_cols = [str(i) for i in range(1, 1441)]
    for name, df in _iter_tables(src):
        if iot_only:
            df = df[df["Trigger"].isin(["event", "queue", "timer"])]
        series.append(df[minute_cols].sum(axis=0).values.astype(float))
        print(f"{name}: {len(df)} functions, {series[-1].sum():.3e} invocations", flush=True)
    if not series:
        raise SystemExit("no invocations_per_function_md.anon.dXX.csv found in " + src)
    v = np.concatenate(series)
    ts = pd.date_range("2019-07-15", periods=len(v), freq="1min")
    os.makedirs(os.path.dirname(os.path.abspath(out)), exist_ok=True)
    pd.DataFrame({"timestamp": ts, "value": v}).to_csv(out, index=False)
    print(f"wrote {out}: {len(v)} minutes, peak/mean = {v.max() / v.mean():.2f}")
    return out


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", required=True, help=".tar.xz archive or extracted folder")
    ap.add_argument("--out", default="data/azure_functions_iot.csv")
    ap.add_argument("--iot-only", action="store_true")
    a = ap.parse_args()
    prepare(a.src, a.out, a.iot_only)
