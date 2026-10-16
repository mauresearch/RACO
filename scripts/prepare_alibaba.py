#!/usr/bin/env python3
"""
Convert Alibaba cluster-trace-v2018 machine_usage into a workload CSV (timestamp,value).

Download (1.7 GB, 8 days, ~4000 machines):
  http://aliopentrace.oss-cn-beijing.aliyuncs.com/v2018Traces/machine_usage.tar.gz
  (docs: https://github.com/alibaba/clusterdata/blob/master/cluster-trace-v2018/trace_2018.md)

The .tar.gz can be given directly (it is streamed, never fully extracted), or the extracted CSV:
  python scripts/prepare_alibaba.py --src machine_usage.tar.gz
  python scripts/prepare_alibaba.py --src machine_usage.csv

Columns (schema.txt, no header row): machine_id, time_stamp[s], cpu_util_percent, mem_util_percent,
mem_gps, mkpi, net_in, net_out, disk_io_percent.  Output = cluster-mean CPU utilisation per minute.
Invalid values (<0 or >100) are dropped.  Reading is chunked so it fits in laptop memory.
"""
import argparse
import os
import tarfile

import pandas as pd

COLS = ["machine_id", "time_stamp", "cpu", "mem", "mem_gps", "mkpi", "net_in", "net_out", "disk"]


def _open(src):
    if src.endswith((".tar.gz", ".tgz", ".tar")):
        tf = tarfile.open(src, "r:*")
        m = next(m for m in tf.getmembers() if m.name.endswith(".csv"))
        return tf.extractfile(m)
    return open(src, "rb")


def prepare(src, out="data/alibaba_cpu.csv", chunksize=5_000_000):
    acc_s = acc_n = None
    with _open(src) as fh:
        reader = pd.read_csv(fh, header=None, names=COLS, usecols=["time_stamp", "cpu"], chunksize=chunksize)
        for i, ch in enumerate(reader):
            ch = ch.dropna()
            ch = ch[(ch.cpu >= 0) & (ch.cpu <= 100)]
            ch["minute"] = (ch["time_stamp"] // 60).astype(int)
            g = ch.groupby("minute")["cpu"]
            s, n = g.sum(), g.count()
            acc_s = s if acc_s is None else acc_s.add(s, fill_value=0)
            acc_n = n if acc_n is None else acc_n.add(n, fill_value=0)
            print(f"chunk {i}: {len(ch)} rows", flush=True)
    mean = (acc_s / acc_n).sort_index()
    mean = mean[acc_n.reindex(mean.index) > 0.5 * acc_n.median()]     # drop sparse edge minutes
    ts = pd.Timestamp("2018-01-01") + pd.to_timedelta(mean.index * 60, unit="s")
    os.makedirs(os.path.dirname(os.path.abspath(out)), exist_ok=True)
    pd.DataFrame({"timestamp": ts, "value": mean.values}).to_csv(out, index=False)
    print(f"wrote {out}: {len(mean)} minutes, peak/mean = {mean.max() / mean.mean():.2f}")
    return out


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", required=True, help="machine_usage.tar.gz or machine_usage.csv")
    ap.add_argument("--out", default="data/alibaba_cpu.csv")
    ap.add_argument("--chunksize", type=int, default=5_000_000)
    a = ap.parse_args()
    prepare(a.src, a.out, a.chunksize)
