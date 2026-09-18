# AquaSentinel — Leak Detection and Localisation in a Water Distribution Network

Machine-learning detection of **new pipe leaks**, and localisation of each leak to a
repair district, using only the pressure and flow sensors a water utility already has.

Evaluated on **BattLeDIM 2020**, a public benchmark built from the L-Town network:
two years of 5-minute SCADA data with ground-truth labels for 33 real leak events.

## Headline results (2019 — held-out, never used for tuning)

| | This work | Minimum night flow (what utilities use today) |
|---|---|---|
| Leaks detected | **17 / 19 (89%)** | 1 / 19 (5%) |
| False alarms | **0** | 0 |
| Median time to detection | **14 days** | 30 days |

Localisation to 1 of 12 zones: **37% top-1** (random: 8%) and **53% top-3**
(random: 25%) — narrowing the search from 43 km of pipe to 3.6 km.
Localisation is scored on each leak's true start day, so it measures the
fingerprint matching on its own, separately from how quickly the detector fires.

## Quick start

```bash
python -m venv .venv
```

```bash
.venv/Scripts/python.exe -m pip install -r requirements.txt
```

```bash
.venv/Scripts/python.exe src/download_data.py
```

```bash
.venv/Scripts/python.exe run_all.py
```

Then launch the dashboard:

```bash
.venv/Scripts/streamlit run app.py
```

The first run computes residuals and simulates a leak on all 902 pipes
(~15 minutes total); everything is cached to `data/processed/` afterwards.

## How it works

**1. Expected-pressure model.** A Ridge regression predicts each of the 33 sensor
pressures from total inflow, tank level and time-of-day/week. It is refitted on a
*trailing 21-day window*, so leaks already running are absorbed into the reference
and stay quiet, while a genuinely new leak breaks the relationship.

This matters more than it sounds: only 2.2% of 2018 is leak-free and **none** of 2019
is, so "is anything leaking right now?" has no useful answer. Reframing the task as
*onset* detection is what makes the problem well-posed.

**2. CUSUM detection.** The residuals are converted to robust z-scores (trailing
median/MAD) and accumulated by a two-sided CUSUM, which builds evidence across days
so slow incipient leaks eventually trip the threshold. Thresholds are tuned on 2018
under a false-alarm budget and frozen before 2019 is touched.

**3. Fingerprint localisation.** A leak shifts the 33 sensors by different amounts;
the *direction* of that 33-dimensional shift depends on where the leak is, and only
its magnitude depends on leak size. Since 14 training leaks is far too few to learn
from, the fingerprint library is generated hydraulically instead: EPANET (via WNTR)
simulates a leak on every one of the 902 pipes. The observed shift is matched to the
library by cosine similarity, and pipe scores are aggregated into zone scores.

## Layout

```
src/config.py         paths, sensor lists, ground-truth leak table
src/download_data.py  fetches the dataset from Zenodo (resumable)
src/data_loader.py    CSV parsing, labels, time features
src/network.py        topology, k-means zoning, geometry
src/residuals.py      rolling-reference expected-pressure model
src/detect.py         CUSUM detector, tuning, evaluation
src/simulate.py       EPANET leak sensitivity matrix
src/localize.py       fingerprint matching to zones
src/experiments.py    full study -> reports/tables/*.csv
src/figures.py        all report figures
app.py                Streamlit dashboard
reports/REPORT.md     write-up with results and limitations
```

## Data

BattLeDIM 2020 dataset, Zenodo record 4017659 (CC BY 4.0), from the KIOS Center of
Excellence. `src/download_data.py` fetches the 14 files used here (~200 MB) and skips
the 92 MB spreadsheet duplicates and the 177 MB detailed network variant.
