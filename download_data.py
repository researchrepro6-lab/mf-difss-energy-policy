#!/usr/bin/env python3
"""Re-download the public raw files into data/raw_fresh/ (the analysis itself
uses the checksummed snapshot in data/raw/). Upstream data are revised over
time, so fresh files may differ from the snapshot used in the manuscript."""
import hashlib
import urllib.request
from pathlib import Path

OUT = Path(__file__).resolve().parent / "data" / "raw_fresh"
FILES = {
    "WB_RISE.csv": "https://data360files.worldbank.org/data360-data/data/WB_RISE/WB_RISE.csv",
    "owid-energy-data.csv": "https://raw.githubusercontent.com/owid/energy-data/master/owid-energy-data.csv",
    "owid-energy-codebook.csv": "https://raw.githubusercontent.com/owid/energy-data/master/owid-energy-codebook.csv",
    "API_EN.GHG.CO2.PC.CE.AR5_excel.zip":   # zip archive containing the .xls file
        "https://api.worldbank.org/v2/en/indicator/EN.GHG.CO2.PC.CE.AR5?downloadformat=excel",
}
MANUAL = ["rise-2016-database-for-download-1.xlsx (https://rise.esmap.org, legacy 2016 workbook)",
          "Renewable_Energy.csv (IMF Climate Change Indicators Dashboard, https://climatedata.imf.org)",
          "ccpi_2023_2024_ranks.csv (CCPI results, https://ccpi.org)"]

if __name__ == "__main__":
    OUT.mkdir(parents=True, exist_ok=True)
    for name, url in FILES.items():
        dest = OUT / name
        print(f"Downloading {url}")
        urllib.request.urlretrieve(url, dest)
        print(f"  {name}: sha256={hashlib.sha256(dest.read_bytes()).hexdigest()}")
    print("Files that must be obtained manually:\n  - " + "\n  - ".join(MANUAL))
