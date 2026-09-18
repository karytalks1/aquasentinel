"""L-Town network topology: zone partitioning and pipe/sensor geometry.

Localising a leak to an individual pipe is hopeless with only 33 pressure
sensors across 905 pipes, and it is not what a utility needs anyway - a repair
crew is dispatched to a district. We therefore partition the network into K
spatial zones and localise to a zone.
"""
from __future__ import annotations

import functools
import warnings

import numpy as np
import pandas as pd

import config as C

warnings.filterwarnings("ignore", category=UserWarning)

N_ZONES = 12


@functools.lru_cache(maxsize=1)
def load_model():
    import wntr
    return wntr.network.WaterNetworkModel(str(C.INP_FILE))


@functools.lru_cache(maxsize=1)
def node_coords() -> pd.DataFrame:
    wn = load_model()
    names = wn.junction_name_list
    xy = np.array([wn.get_node(n).coordinates for n in names])
    return pd.DataFrame(xy, index=names, columns=["x", "y"])


@functools.lru_cache(maxsize=1)
def pipe_table() -> pd.DataFrame:
    """One row per pipe with its endpoints, length and midpoint coordinates."""
    wn = load_model()
    coords = node_coords()
    rows = []
    for name in wn.pipe_name_list:
        link = wn.get_link(name)
        a, b = link.start_node_name, link.end_node_name
        if a not in coords.index or b not in coords.index:
            continue  # pipe touching a tank/reservoir - excluded from zoning
        rows.append(
            {
                "pipe": name,
                "start": a,
                "end": b,
                "length_m": float(link.length),
                "diameter_m": float(link.diameter),
                "x": (coords.at[a, "x"] + coords.at[b, "x"]) / 2,
                "y": (coords.at[a, "y"] + coords.at[b, "y"]) / 2,
            }
        )
    return pd.DataFrame(rows).set_index("pipe")


@functools.lru_cache(maxsize=8)
def zone_assignment(n_zones: int = N_ZONES) -> pd.DataFrame:
    """Assign every pipe to one of `n_zones` spatial districts via k-means.

    Clustering is on pipe midpoints, so a zone is a contiguous geographic
    district - which is what a repair crew can actually be sent to.
    """
    from sklearn.cluster import KMeans

    pt = pipe_table()
    km = KMeans(n_clusters=n_zones, n_init=10, random_state=0)
    labels = km.fit_predict(pt[["x", "y"]].to_numpy())
    out = pt.copy()
    out["zone"] = labels
    return out


def pipe_zone(pipe: str, n_zones: int = N_ZONES) -> int:
    return int(zone_assignment(n_zones).at[pipe, "zone"])


def zone_centroids(n_zones: int = N_ZONES) -> pd.DataFrame:
    z = zone_assignment(n_zones)
    return z.groupby("zone")[["x", "y"]].mean()


def sensor_coords() -> pd.DataFrame:
    return node_coords().loc[C.PRESSURE_SENSORS]


def zone_stats(n_zones: int = N_ZONES) -> pd.DataFrame:
    """Per-zone pipe count and total length - used to size the search effort."""
    z = zone_assignment(n_zones)
    return pd.DataFrame(
        {
            "n_pipes": z.groupby("zone").size(),
            "length_km": z.groupby("zone")["length_m"].sum() / 1000,
        }
    )
