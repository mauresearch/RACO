# Input data

Place the input traces in this folder. They are not distributed with the repository.

| File | Created by |
|---|---|
| `azure_functions_iot.csv` | `python scripts/prepare_azure_functions.py --src <archive> --out data/azure_functions_iot.csv --iot-only` |
| `alibaba_cpu.csv` | `python scripts/prepare_alibaba.py --src <machine_usage.tar.gz> --out data/alibaba_cpu.csv` |
| `weather_Tehran.csv`, `weather_BandarAbbas.csv`, `weather_Tabriz.csv`, `weather_Rasht.csv` | `python scripts/download_weather.py --out data` |

`run_scenarios.py` builds the two workload files automatically if the original archives are placed here.
