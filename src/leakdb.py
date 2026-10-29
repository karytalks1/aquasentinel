"""Loader for the LeakDB Hanoi benchmark (KIOS, CCWI-WDSA 2018).

Layout of the extracted archive, one folder per scenario:

    Scenario-N/
        Hanoi_CMH_Scenario-N.inp     the model that generated THIS scenario
        Labels.csv                   per-timestep 0/1, is anything leaking
        Scenario-N_info.csv          duration, timestep, parameter uncertainty
        Pressures/Node_*.csv         one file per junction
        Flows/Link_*.csv             one file per pipe
        Demands/Node_*.csv           one file per junction
        Leaks/Leak_<node>_info.csv   node, area, diameter, type, start, end, peak
        Leaks/Leak_<node>_demand.csv the leak discharge time series

Each scenario is one simulated year at 30-minute resolution (17,520 rows).

Why this dataset: unlike BattLeDIM it provides completely leak-free scenarios,
so a clean baseline can be learned directly, and it ships the true model for
each scenario, so the simulated leak fingerprints can be verified rather than
assumed.
"""
from __future__ import annotations

import functools
import glob
import re
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd

import config as C

ROOT = C.ROOT / "data" / "leakdb"
PROCESSED = C.PROCESSED / "leakdb"
PROCESSED.mkdir(parents=True, exist_ok=True)

SAMPLE_MINUTES = 30
STEPS_PER_DAY = 24 * 60 // SAMPLE_MINUTES

# Scenarios with no leak at any timestep, used to learn "healthy" behaviour.
CLEAN_SCENARIOS = (1, 2, 4, 9)
LEAKY_SCENARIOS = (3, 5, 6, 7, 8, 10)

# Development / held-out split over the leaky scenarios. Every threshold is
# chosen on DEV and frozen before TEST is evaluated.
DEV_SCENARIOS = (3, 5, 6)
TEST_SCENARIOS = (7, 8, 10)


@dataclass(frozen=True)
class LeakEvent:
    """One ground-truth leak from a Leaks/Leak_<node>_info.csv file."""

    scenario: int
    node: str
    kind: str              # 'abrupt' or 'incipient'
    diameter_m: float
    area_m2: float
    start: datetime
    end: datetime
    peak: datetime

    @property
    def diameter_mm(self) -> float:
        return self.diameter_m * 1000


def scenario_dir(n: int) -> Path:
    return ROOT / f"Scenario-{n}"


def inp_file(n: int) -> Path:
    return scenario_dir(n) / f"Hanoi_CMH_Scenario-{n}.inp"


def available_scenarios() -> list[int]:
    return sorted(int(re.search(r"Scenario-(\d+)", p).group(1))
                  for p in glob.glob(str(ROOT / "Scenario-*")))


def _read_series(path: Path) -> pd.Series:
    df = pd.read_csv(path, parse_dates=["Timestamp"])
    return df.set_index("Timestamp").iloc[:, 0].astype("float64")


def _node_key(path: str) -> int:
    return int(re.search(r"_(\d+)\.csv$", path).group(1))


@functools.lru_cache(maxsize=16)
def pressures(n: int) -> pd.DataFrame:
    """(timesteps x junctions) pressure in metres. Columns are node names."""
    cache = PROCESSED / f"pressures_{n}.csv"
    if cache.exists():
        return pd.read_csv(cache, index_col=0, parse_dates=True)

    files = sorted(glob.glob(str(scenario_dir(n) / "Pressures" / "Node_*.csv")),
                   key=_node_key)
    data = {Path(f).stem.replace("Node_", ""): _read_series(Path(f)) for f in files}
    df = pd.DataFrame(data).sort_index()
    # The reservoir is reported with a constant zero pressure; it carries no
    # information and would break the per-sensor normalisation.
    df = df.loc[:, df.std() > 0]
    df.to_csv(cache)
    return df


@functools.lru_cache(maxsize=16)
def flows(n: int) -> pd.DataFrame:
    """(timesteps x pipes) flow in m^3/h."""
    cache = PROCESSED / f"flows_{n}.csv"
    if cache.exists():
        return pd.read_csv(cache, index_col=0, parse_dates=True)

    files = sorted(glob.glob(str(scenario_dir(n) / "Flows" / "Link_*.csv")),
                   key=_node_key)
    data = {Path(f).stem.replace("Link_", ""): _read_series(Path(f)) for f in files}
    df = pd.DataFrame(data).sort_index()
    df.to_csv(cache)
    return df


@functools.lru_cache(maxsize=16)
def labels(n: int) -> pd.Series:
    """Per-timestep 0/1: is any leak discharging."""
    s = _read_series(scenario_dir(n) / "Labels.csv").sort_index()
    return (s > 0).rename("leaking")


@functools.lru_cache(maxsize=16)
def leaks(n: int) -> tuple[LeakEvent, ...]:
    """Ground-truth leak events for one scenario."""
    out = []
    for info in sorted(glob.glob(str(scenario_dir(n) / "Leaks" / "*_info.csv"))):
        raw = pd.read_csv(info, header=0, names=["key", "value"], dtype=str)
        kv = {str(k).strip(): str(v).strip() for k, v in
              zip(raw["key"], raw["value"])}
        out.append(LeakEvent(
            scenario=n,
            node=kv["Leak Node"],
            kind=kv["Leak Type"].lower(),
            diameter_m=float(kv["Leak Diameter"]),
            area_m2=float(kv["Leak Area"]),
            start=pd.Timestamp(kv["Leak Start"]).to_pydatetime(),
            end=pd.Timestamp(kv["Leak End"]).to_pydatetime(),
            peak=pd.Timestamp(kv["Peak Time"]).to_pydatetime(),
        ))
    return tuple(sorted(out, key=lambda e: e.start))


def all_leaks(scenarios=LEAKY_SCENARIOS) -> list[LeakEvent]:
    return [e for n in scenarios for e in leaks(n)]


def inflow(n: int) -> pd.Series:
    """Total water entering the network, from the reservoir pipe (m^3/h).

    Hanoi has a single reservoir, so one pipe carries the whole supply. Its
    flow is the natural stand-in for the inlet meter a utility would have.
    """
    import warnings
    warnings.filterwarnings("ignore")
    import wntr

    wn = wntr.network.WaterNetworkModel(str(inp_file(n)))
    res = wn.reservoir_name_list[0]
    supply = [p for p in wn.pipe_name_list
              if res in (wn.get_link(p).start_node_name,
                         wn.get_link(p).end_node_name)]
    f = flows(n)
    cols = [p for p in supply if p in f.columns]
    return f[cols].abs().sum(axis=1).rename("inflow")


@functools.lru_cache(maxsize=1)
def node_coords() -> pd.DataFrame:
    """Junction coordinates from the network model."""
    import warnings
    warnings.filterwarnings("ignore")
    import wntr

    wn = wntr.network.WaterNetworkModel(str(inp_file(CLEAN_SCENARIOS[0])))
    names = wn.junction_name_list
    xy = np.array([wn.get_node(nm).coordinates for nm in names])
    return pd.DataFrame(xy, index=names, columns=["x", "y"])


@functools.lru_cache(maxsize=8)
def sensor_nodes(k: int = 8) -> tuple[str, ...]:
    """Pick k junctions to act as pressure sensors, spread across the network.

    A utility instruments a handful of points, not every junction. Choosing
    them by spatial spread (k-means, then the junction nearest each centre)
    is deterministic and avoids hand-picking nodes that flatter the method.
    """
    from sklearn.cluster import KMeans

    coords = node_coords()
    measured = [c for c in coords.index if c in pressures(CLEAN_SCENARIOS[0]).columns]
    pts = coords.loc[measured]
    km = KMeans(n_clusters=k, n_init=10, random_state=0).fit(pts.to_numpy())
    chosen = []
    for centre in km.cluster_centers_:
        d = np.hypot(pts["x"] - centre[0], pts["y"] - centre[1])
        for cand in d.sort_values().index:
            if cand not in chosen:
                chosen.append(cand)
                break
    return tuple(sorted(chosen, key=int))


def time_features(idx: pd.DatetimeIndex) -> pd.DataFrame:
    """Cyclical calendar terms: demand is strongly daily and weekly."""
    tod = idx.hour * 60 + idx.minute
    dow = idx.dayofweek.to_numpy()
    return pd.DataFrame(
        {
            "tod_sin": np.sin(2 * np.pi * tod / 1440),
            "tod_cos": np.cos(2 * np.pi * tod / 1440),
            "dow_sin": np.sin(2 * np.pi * dow / 7),
            "dow_cos": np.cos(2 * np.pi * dow / 7),
            "is_weekend": (dow >= 5).astype(float),
        },
        index=idx,
    )


def summary() -> pd.DataFrame:
    """One row per scenario: size, leak count and how much of it is leaking."""
    rows = []
    for n in available_scenarios():
        lab = labels(n)
        ev = leaks(n)
        rows.append({
            "scenario": n,
            "role": ("clean" if n in CLEAN_SCENARIOS else
                     "development" if n in DEV_SCENARIOS else "held-out test"),
            "rows": len(lab),
            "leaking_share": round(float(lab.mean()), 3),
            "n_leaks": len(ev),
            "leaks": ", ".join(f"node {e.node} ({e.kind}, {e.diameter_mm:.0f} mm)"
                               for e in ev) or "-",
        })
    return pd.DataFrame(rows)
