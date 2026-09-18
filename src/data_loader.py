"""Load the BattLeDIM SCADA CSVs into tidy pandas frames.

The raw files are European-formatted: ';' column separator and ',' decimal mark.
Everything here returns a DatetimeIndex at 5-minute resolution.
"""
from __future__ import annotations

import functools

import numpy as np
import pandas as pd

import config as C


def _read(path) -> pd.DataFrame:
    df = pd.read_csv(path, sep=";", decimal=",", parse_dates=["Timestamp"])
    return df.set_index("Timestamp").sort_index().astype("float64")


@functools.lru_cache(maxsize=8)
def load_year(year: int) -> dict[str, pd.DataFrame]:
    """Return {'pressure', 'flow', 'level', 'demand', 'leakflow'} for one year."""
    out = {
        "pressure": _read(C.RAW / f"{year}_SCADA_Pressures.csv"),
        "flow": _read(C.RAW / f"{year}_SCADA_Flows.csv"),
        "level": _read(C.RAW / f"{year}_SCADA_Levels.csv"),
        "demand": _read(C.RAW / f"{year}_SCADA_Demands.csv"),
        "leakflow": _read(C.RAW / f"{year}_Leakages.csv"),
    }
    idx = out["pressure"].index
    for k, v in out.items():
        if not v.index.equals(idx):
            raise ValueError(f"{year} {k}: timestamps do not align with pressures")
    return out


def sensor_frame(year: int) -> pd.DataFrame:
    """The signals a utility would actually have: 33 pressures, 3 flows, 1 level.

    Deliberately excludes the 82 AMR demand meters and the leak flows - AMR data
    in this dataset is only available with a long reporting delay, and leak flows
    are the labels.
    """
    d = load_year(year)
    df = pd.concat(
        [
            d["pressure"][C.PRESSURE_SENSORS].add_prefix("p_"),
            d["flow"][C.FLOW_SENSORS].add_prefix("f_"),
            d["level"][C.LEVEL_SENSORS].add_prefix("l_"),
        ],
        axis=1,
    )
    return df


def leak_labels(year: int) -> pd.DataFrame:
    """Per-timestep labels derived from the ground-truth leak flow time series.

    Columns
    -------
    any_leak     : bool  - at least one leak is discharging
    n_active     : int   - how many leaks are discharging
    total_lps    : float - summed leak discharge (m^3/h)
    dominant     : str   - pipe ID of the largest active leak ('none' if dry)
    """
    lf = load_year(year)["leakflow"]
    active = lf > 0.0
    dominant = lf.idxmax(axis=1).where(active.any(axis=1), "none")
    return pd.DataFrame(
        {
            "any_leak": active.any(axis=1),
            "n_active": active.sum(axis=1).astype("int16"),
            "total_lps": lf.sum(axis=1),
            "dominant": dominant,
        },
        index=lf.index,
    )


def leak_free_mask(year: int) -> pd.Series:
    """True where no leak is discharging at all - our clean reference data."""
    return ~leak_labels(year)["any_leak"]


def total_inflow(year: int) -> pd.Series:
    """Total water entering the network (m^3/h) across the three flow meters."""
    f = load_year(year)["flow"]
    return f[C.FLOW_SENSORS].sum(axis=1).rename("inflow")


def time_features(idx: pd.DatetimeIndex) -> pd.DataFrame:
    """Cyclical calendar features - demand is strongly daily and weekly."""
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
