"""
Exogenous inputs: workload demand and ambient (enclosure-air) temperature per node.

* Synthetic generators are used by default so the code runs out of the box.
* Real data can be plugged in through CSV files (see scripts/ and README):
    - workload : prepared CSV  timestamp,value  (any resolution)  -> shape is tiled over the horizon
    - weather  : Open-Meteo hourly CSV (time,temperature_2m)      -> tiled over the horizon
"""
import os

import numpy as np
import pandas as pd

# Approximate monthly climatology (mean daily temperature, mean diurnal range) [degC].
# Used when no measured weather file is supplied (see scripts/download_weather.py).
CLIMATES = {
    "Tehran":      dict(mean=[4, 6, 11, 17, 22, 28, 31, 30, 25, 19, 11, 6],
                        rng=[8, 9, 10, 11, 12, 13, 13, 13, 13, 12, 9, 8]),
    "BandarAbbas": dict(mean=[19, 20, 23, 27, 31, 33, 35, 35, 33, 30, 25, 21],
                        rng=[9, 9, 8, 8, 8, 7, 6, 6, 7, 8, 9, 9]),
    "Tabriz":      dict(mean=[-2, 0, 6, 12, 17, 22, 26, 25, 20, 14, 6, 1],
                        rng=[9, 10, 11, 12, 13, 14, 14, 14, 14, 13, 10, 9]),
    "Rasht":       dict(mean=[7, 8, 10, 15, 20, 24, 26, 26, 23, 18, 13, 9],
                        rng=[6, 6, 6, 7, 8, 8, 8, 8, 7, 7, 6, 6]),
}


def _ar1(n, phi, sigma, rng):
    e = rng.normal(0.0, sigma * np.sqrt(1 - phi ** 2), n)
    x = np.empty(n)
    x[0] = rng.normal(0, sigma)
    for i in range(1, n):
        x[i] = phi * x[i - 1] + e[i]
    return x


# --------------------------------------------------------------------------- workload
def synthetic_workload_shape(n_frames, frame_h, rng):
    """IoT-like demand: two daily peaks, weekend dip, mild seasonality, AR(1) noise, bursts. Mean = 1."""
    t = np.arange(n_frames) * frame_h
    hod, dow, doy = t % 24, (t // 24) % 7, (t / 24.0) % 365
    diurnal = 0.35 + 0.65 * (np.exp(-0.5 * ((hod - 11) / 3.0) ** 2) + 0.8 * np.exp(-0.5 * ((hod - 20) / 2.5) ** 2))
    weekly = np.where(dow >= 5, 0.82, 1.0)
    seasonal = 1.0 + 0.06 * np.sin(2 * np.pi * (doy - 100) / 365.0)
    noise = np.exp(_ar1(n_frames, 0.7 ** frame_h, 0.07, rng))
    bursts = np.ones(n_frames)
    n_b = rng.poisson(2.0 * n_frames * frame_h / (30 * 24))
    for s in rng.integers(0, n_frames, n_b):
        L = max(1, int(rng.integers(1, 4) / frame_h))
        bursts[s:s + L] *= rng.uniform(1.15, 1.35)
    x = diurnal * weekly * seasonal * noise * bursts
    return x / x.mean()


def csv_workload_shape(path, n_frames, frame_h, rng, diversity=True):
    """Read a prepared CSV (timestamp,value), resample to the frame length and build the horizon.

    diversity=True  (default): day-block bootstrap.  The trace is cut into whole days; every
        simulated day is a randomly drawn real day (same weekday when the trace has >= 14 days),
        scaled by a random day-level factor.  Each seed therefore sees a different but realistic
        multi-year sequence -> independent replications.
    diversity=False: the trace is tiled as is, with small seed-dependent noise.
    """
    df = pd.read_csv(path)
    ts = pd.to_datetime(df.iloc[:, 0], errors="coerce", utc=True)
    if ts.isna().mean() > 0.5:          # numeric seconds
        ts = pd.to_datetime(pd.to_numeric(df.iloc[:, 0]), unit="s", utc=True)
    s = pd.Series(pd.to_numeric(df.iloc[:, 1], errors="coerce").values, index=ts).dropna().sort_index()
    s = s.resample(f"{int(round(frame_h * 60))}min").mean().interpolate()
    base = s.values / s.values.mean()
    per_day = int(round(24 / frame_h))
    n_days_trace = len(base) // per_day
    if not diversity or n_days_trace < 2:
        reps = int(np.ceil(n_frames / len(base)))
        x = np.tile(base, reps)[:n_frames]
        x = x * np.exp(_ar1(n_frames, 0.95, 0.05, rng))
        return x / x.mean()
    days = base[: n_days_trace * per_day].reshape(n_days_trace, per_day)
    dow0 = s.index[0].dayofweek
    trace_dow = (dow0 + np.arange(n_days_trace)) % 7
    n_days = int(np.ceil(n_frames / per_day))
    same_weekday = n_days_trace >= 14
    out = np.empty((n_days, per_day))
    day_factor = np.exp(_ar1(n_days, 0.8, 0.06, rng))          # slowly varying day-level intensity
    for d in range(n_days):
        pool = np.where(trace_dow == (d % 7))[0] if same_weekday else np.arange(n_days_trace)
        out[d] = days[rng.choice(pool)] * day_factor[d]
    x = out.ravel()[:n_frames] * np.exp(_ar1(n_frames, 0.7, 0.03, rng))
    return x / x.mean()


# --------------------------------------------------------------------------- climate
def synthetic_outdoor_temperature(city, n_frames, frame_h, rng):
    c = CLIMATES[city]
    t = np.arange(n_frames) * frame_h
    doy = (t / 24.0) % 365
    month_pos = doy / 365.0 * 12 - 0.5
    mids = np.arange(12)
    mean = np.interp(month_pos, mids, c["mean"], period=12)
    dr = np.interp(month_pos, mids, c["rng"], period=12)
    hod = t % 24
    diurnal = 0.5 * dr * np.cos(2 * np.pi * (hod - 15) / 24.0)
    synoptic = _ar1(n_frames, 0.97 ** frame_h, 2.5, rng)     # multi-day weather systems
    return mean + diurnal + synoptic


def csv_outdoor_temperature(path, n_frames, frame_h, rng, diversity=True):
    """Open-Meteo hourly CSV (metadata block, blank line, then 'time,temperature_2m (°C)')."""
    with open(path, encoding="utf-8") as f:
        lines = f.readlines()
    hdr = next(i for i, l in enumerate(lines) if l.strip().lower().startswith("time"))
    df = pd.read_csv(path, skiprows=hdr)
    ts = pd.to_datetime(df.iloc[:, 0])
    col = [c for c in df.columns if c.lower().startswith("temperature")][0]
    s = pd.Series(pd.to_numeric(df[col], errors="coerce").values, index=ts).dropna()
    s = s.resample(f"{int(round(frame_h * 60))}min").mean().interpolate()
    v = s.values
    if diversity:
        # start the simulation in a seed-dependent year of the weather record
        per_year = int(round(8760 / frame_h))
        n_years = max(1, len(v) // per_year)
        v = np.roll(v, -int(rng.integers(0, n_years)) * per_year)
    reps = int(np.ceil(n_frames / len(v)))
    return np.tile(v, reps)[:n_frames] + _ar1(n_frames, 0.97, 0.5, rng)


def installation_ambient(T_out, site, n_frames, frame_h, rng):
    """Air temperature around the node enclosure for a given installation site."""
    t = np.arange(n_frames) * frame_h
    hod, dow, doy = t % 24, (t // 24) % 7, (t / 24.0) % 365
    if site == "indoor":          # HVAC-controlled room
        return 23.0 + 1.5 * np.sin(2 * np.pi * (doy - 100) / 365) + _ar1(n_frames, 0.9, 0.7, rng)
    if site == "outdoor":         # street cabinet: enclosure rise + solar gain
        sun = np.clip(np.sin(np.pi * (hod - 6) / 12.0), 0, None)
        cloud = np.clip(0.85 + _ar1(n_frames, 0.95, 0.15, rng), 0.3, 1.0)
        return T_out + 6.0 + 10.0 * sun * cloud
    if site == "factory":         # shop floor, warmer during work hours
        work = ((hod >= 7) & (hod < 19) & (dow < 5)).astype(float)
        return 0.3 * np.maximum(T_out, 15.0) + 21.0 + 6.0 * work + _ar1(n_frames, 0.9, 1.0, rng)
    raise ValueError(site)


def build_inputs(cfg, layout):
    """Return (demand shape [n_frames], ambient [n_frames, N])."""
    rng = np.random.default_rng(1000 + cfg.seed)
    n, fh = cfg.n_frames, cfg.frame_h
    if cfg.workload_source == "csv":
        shape = csv_workload_shape(cfg.workload_csv, n, fh, rng, cfg.seed_diversity)
    else:
        shape = synthetic_workload_shape(n, fh, rng)
    city_csv = os.path.join(cfg.weather_dir, f"weather_{cfg.climate}.csv") if cfg.weather_dir else ""
    if city_csv and os.path.exists(city_csv):
        T_out = csv_outdoor_temperature(city_csv, n, fh, rng, cfg.seed_diversity)
    elif cfg.weather_source == "csv":
        T_out = csv_outdoor_temperature(cfg.weather_csv, n, fh, rng, cfg.seed_diversity)
    else:
        T_out = synthetic_outdoor_temperature(cfg.climate, n, fh, rng)
    amb = np.empty((n, len(layout)))
    for i, (_, site) in enumerate(layout):
        amb[:, i] = installation_ambient(T_out, site, n, fh, rng)
    return shape, amb, T_out
