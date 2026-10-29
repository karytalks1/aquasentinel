"""Download and extract the LeakDB Hanoi benchmark into data/leakdb/.

Usage:  .venv/Scripts/python.exe src/download_leakdb.py

LeakDB (Leakage Diagnosis Benchmark) is published by the KIOS Center of
Excellence, University of Cyprus. This fetches the Hanoi benchmark used here -
10 scenarios of one simulated year each, 79 MB - rather than the full 13.9 GB
LeakDB archive.

  Data  : CC BY 4.0,  https://zenodo.org/records/1313116
  Code  : EUPL-1.2,   https://github.com/KIOS-Research/LeakDB
  Cite  : Vrachimis, Kyriakou, Eliades and Polycarpou (2018), "LeakDB: A
          benchmark dataset for leakage diagnosis in water distribution
          networks", WDSA/CCWI Joint Conference.

Safe to re-run: an already-extracted dataset is left alone.
"""
from __future__ import annotations

import sys
import urllib.request
import zipfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
RAW = ROOT / "data" / "leakdb_raw"
OUT = ROOT / "data" / "leakdb"

URL = ("https://github.com/KIOS-Research/LeakDB/raw/master/"
       "CCWI-WDSA2018/Benchmarks/Hanoi_CMH.zip")
ARCHIVE = RAW / "Hanoi_CMH.zip"
EXPECTED_SCENARIOS = 10


def download() -> None:
    RAW.mkdir(parents=True, exist_ok=True)
    if ARCHIVE.exists() and ARCHIVE.stat().st_size > 1_000_000:
        print(f"  archive already present ({ARCHIVE.stat().st_size / 1e6:.0f} MB)")
        return

    print(f"  downloading {URL}")
    req = urllib.request.Request(URL, headers={"User-Agent": "python"})
    with urllib.request.urlopen(req, timeout=300) as r, open(ARCHIVE, "wb") as f:
        done = 0
        while chunk := r.read(1 << 18):
            f.write(chunk)
            done += len(chunk)
            sys.stdout.write(f"\r  {done / 1e6:6.1f} MB")
            sys.stdout.flush()
    print()


def extract() -> None:
    existing = sorted(OUT.glob("Scenario-*"))
    if len(existing) >= EXPECTED_SCENARIOS:
        print(f"  already extracted ({len(existing)} scenarios)")
        return
    OUT.mkdir(parents=True, exist_ok=True)
    print("  extracting")
    with zipfile.ZipFile(ARCHIVE) as z:
        z.extractall(OUT)


def main() -> None:
    download()
    extract()
    scenarios = sorted(OUT.glob("Scenario-*"))
    print(f"\n{len(scenarios)} scenarios in {OUT}")
    if len(scenarios) < EXPECTED_SCENARIOS:
        raise SystemExit(
            f"expected {EXPECTED_SCENARIOS} scenarios, found {len(scenarios)}"
        )
    print("next:  .venv/Scripts/python.exe run_hanoi.py")


if __name__ == "__main__":
    main()
