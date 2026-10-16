#!/usr/bin/env python3
"""
Download real hourly 2-m air temperature from the free Open-Meteo archive API (no key needed)
for the four cities used in the evaluation (Tehran, Bandar Abbas, Tabriz, Rasht).

  python scripts/download_weather.py --start 2019-01-01 --end 2023-12-31
  python run_all.py --mode full --weather-dir data

If the script cannot reach the API, open the printed URL in a browser and save the CSV
into data/ with the same name.
"""
import argparse
import os
import urllib.request

CITIES = {"Tehran": (35.69, 51.39), "BandarAbbas": (27.18, 56.27), "Tabriz": (38.08, 46.29), "Rasht": (37.28, 49.58)}


def download(start="2019-01-01", end="2023-12-31", out="data"):
    os.makedirs(out, exist_ok=True)
    for city, (lat, lon) in CITIES.items():
        url = (f"https://archive-api.open-meteo.com/v1/archive?latitude={lat}&longitude={lon}"
               f"&start_date={start}&end_date={end}&hourly=temperature_2m&timezone=Asia%2FTehran&format=csv")
        path = os.path.join(out, f"weather_{city}.csv")
        print(f"{city}: {url}")
        try:
            urllib.request.urlretrieve(url, path)
            print(f"  -> {path}")
        except Exception as e:  # noqa: BLE001
            print(f"  failed ({e}); download the URL manually and save as {path}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--start", default="2019-01-01")
    ap.add_argument("--end", default="2023-12-31")
    ap.add_argument("--out", default="data")
    x = ap.parse_args()
    download(x.start, x.end, x.out)
