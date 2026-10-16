# -*- coding: utf-8 -*-
r"""
Run the four evaluation scenarios of the paper one after another.

  1. Azure                     -> results/azure
  2. Alibaba                   -> results/alibaba
  3. Azure   + measured weather -> results/azure_weather
  4. Alibaba + measured weather -> results/alibaba_weather

Needs in data/:
  azure_functions_iot.csv           (built from the Azure Functions 2019 archive on first use)
  alibaba_cpu.csv                   (built from machine_usage.tar.gz on first use)
  weather_Tehran.csv, weather_BandarAbbas.csv, weather_Tabriz.csv, weather_Rasht.csv
Usage:
    python run_scenarios.py
Full mode, 4 scenarios: ~3.5 h on 4 cores, ~2 h on 8 cores (~13 h on 1 core).
A scenario whose results already exist is skipped automatically (cached), so the script can be
restarted after an interruption.
"""
import os
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
os.chdir(HERE)
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.join(HERE, "scripts"))

# ------------------------------------------------------------------ settings
MODE = "full"
JOBS = max(1, (os.cpu_count() or 2) - 1)   # parallel worker processes
AZURE_ARCHIVE = os.path.join("data", "azurefunctions_dataset2019_azurefunctions-dataset2019.tar.xz")
ALIBABA_ARCHIVE = os.path.join("data", "machine_usage.tar.gz")
SCENARIOS = ["azure", "alibaba", "azure_weather", "alibaba_weather"]   # remove any you don't need
# ---------------------------------------------------------------------------


def main():
    from run_all import main as run_main
    azure_csv = os.path.join("data", "azure_functions_iot.csv")
    ali_csv = os.path.join("data", "alibaba_cpu.csv")
    if any(s.startswith("azure") for s in SCENARIOS) and not os.path.exists(azure_csv):
        from prepare_azure_functions import prepare as prepare_azure
        prepare_azure(AZURE_ARCHIVE, azure_csv, iot_only=True)
    if any(s.startswith("alibaba") for s in SCENARIOS) and not os.path.exists(ali_csv):
        from prepare_alibaba import prepare
        prepare(ALIBABA_ARCHIVE, ali_csv)
    cities = ["Tehran", "BandarAbbas", "Tabriz", "Rasht"]
    missing = [c for c in cities if not os.path.exists(os.path.join("data", f"weather_{c}.csv"))]
    if missing and any(s.endswith("weather") for s in SCENARIOS):
        raise SystemExit(f"missing weather files for {missing}: run scripts/download_weather.py first")

    for sc in SCENARIOS:
        t0 = time.time()
        csv = azure_csv if sc.startswith("azure") else ali_csv
        args = ["--mode", MODE, "--jobs", str(JOBS), "--workload-csv", csv, "--out", os.path.join("results", sc)]
        if sc.endswith("weather"):
            args += ["--weather-dir", "data"]
        print("\n" + "=" * 70 + f"\nSCENARIO {sc}:  run_all.py " + " ".join(args) + "\n" + "=" * 70, flush=True)
        run_main(args)
        print(f"scenario {sc} finished in {(time.time() - t0) / 3600:.2f} h", flush=True)


if __name__ == "__main__":
    main()
