"""Download the BattLeDIM 2020 (L-Town) dataset from Zenodo into data/raw/.

BattLeDIM = 'Battle of the Leakage Detection and Isolation Methods', the benchmark
released with the 2020 competition. It provides two years of SCADA data from the
L-Town network together with ground-truth labels for every leak event, which is
what lets us train and honestly evaluate a supervised model.

Safe to re-run: files already present at the right size are skipped, and partial
downloads resume where they stopped.
"""
import json
import time
import urllib.request
from pathlib import Path

RECORD_URL = "https://zenodo.org/api/records/4017659"
RAW = Path(__file__).resolve().parents[1] / "data" / "raw"

# Only the files we actually need. The two 92 MB .xlsx files duplicate the CSVs,
# and L-TOWN_Real.inp (177 MB) is a detailed variant we do not use.
WANTED = {
    "L-TOWN.inp",
    "README.txt",
    "dataset_configuration.yaml",
    "2018_SCADA_Pressures.csv",
    "2018_SCADA_Flows.csv",
    "2018_SCADA_Levels.csv",
    "2018_SCADA_Demands.csv",
    "2018_Leakages.csv",
    "2018_Fixed_Leakages_Report.txt",
    "2019_SCADA_Pressures.csv",
    "2019_SCADA_Flows.csv",
    "2019_SCADA_Levels.csv",
    "2019_SCADA_Demands.csv",
    "2019_Leakages.csv",
    "2019_Leakages_Report.txt",
}


def fetch(url: str, dest: Path, size: int, attempts: int = 8) -> None:
    """Download `url` to `dest`, resuming after a dropped connection.

    Zenodo intermittently returns 502/504 on the larger files, so each attempt
    asks only for the bytes we are still missing via an HTTP Range request.
    """
    for attempt in range(1, attempts + 1):
        have = dest.stat().st_size if dest.exists() else 0
        if have >= size:
            break
        req = urllib.request.Request(url)
        if have:
            req.add_header("Range", f"bytes={have}-")
        try:
            with urllib.request.urlopen(req, timeout=180) as r:
                # If the server ignored our Range header, restart the file.
                mode = "ab" if r.status == 206 else "wb"
                with open(dest, mode) as f:
                    while chunk := r.read(1 << 18):
                        f.write(chunk)
        except Exception as e:  # noqa: BLE001 - any network error is retryable
            got = dest.stat().st_size if dest.exists() else 0
            print(f"    attempt {attempt}/{attempts} stopped at "
                  f"{got / 1e6:.1f}/{size / 1e6:.1f} MB "
                  f"({type(e).__name__}), retrying")
            time.sleep(3 * attempt)

    final = dest.stat().st_size if dest.exists() else 0
    if final != size:
        raise RuntimeError(f"{dest.name}: got {final} bytes, expected {size}")
    print(f"  {dest.name:<32} {size / 1e6:7.1f} MB  ok")


def main() -> None:
    RAW.mkdir(parents=True, exist_ok=True)
    with urllib.request.urlopen(RECORD_URL, timeout=60) as r:
        record = json.load(r)

    files = {f["key"]: f for f in record["files"]}
    if missing := WANTED - files.keys():
        print(f"note: not present in the Zenodo record: {sorted(missing)}")

    for name in sorted(WANTED & files.keys()):
        dest = RAW / name
        size = files[name]["size"]
        if dest.exists() and dest.stat().st_size == size:
            print(f"  {name:<32} {size / 1e6:7.1f} MB  cached")
            continue
        fetch(files[name]["links"]["self"], dest, size)

    print(f"\nAll files in {RAW}")


if __name__ == "__main__":
    main()
