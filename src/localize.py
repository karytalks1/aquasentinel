"""Localise a detected leak by matching its pressure fingerprint.

When a leak starts, the 33 sensors each shift by a different amount. That
33-dimensional shift vector points in a direction that depends mainly on *where*
the leak is, and only its length depends on how big the leak is. So we normalise
the observed shift to unit length and compare its direction against the
simulated fingerprint of every candidate pipe (see simulate.py), using cosine
similarity.

Predicting an exact pipe out of 902 from 33 sensors is not realistic, so the
reported answer is a *zone* - the district a repair crew would be sent to.
Zone scores aggregate the similarity of all pipes in that zone.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

import config as C
import network as N
import residuals as res
import simulate as sim


@dataclass
class Localisation:
    pipe_pred: str
    zone_pred: int
    zone_scores: pd.Series      # similarity per zone, descending
    pipe_scores: pd.Series      # similarity per pipe, descending


def _unit(v: np.ndarray) -> np.ndarray:
    n = np.linalg.norm(v)
    return v / n if n > 0 else v


def localise(observed: pd.Series, n_zones: int = N.N_ZONES,
             top_frac: float = 0.05) -> Localisation:
    """Match one observed 33-sensor shift vector against the simulated library.

    Zone score = mean cosine similarity of the best `top_frac` of pipes in that
    zone. Taking the top slice rather than the zone average stops a large zone
    with many irrelevant pipes from diluting a genuinely good match.
    """
    lib = sim.unit_signatures()
    cols = [c for c in lib.columns if c in observed.index]
    lib = lib[cols]
    obs = _unit(observed[cols].to_numpy(dtype=float))

    sims = pd.Series(lib.to_numpy() @ obs, index=lib.index).sort_values(ascending=False)

    zones = N.zone_assignment(n_zones)["zone"]
    df = pd.DataFrame({"sim": sims, "zone": zones.reindex(sims.index)}).dropna()

    def _score(g: pd.DataFrame) -> float:
        k = max(1, int(round(top_frac * len(g))))
        return float(g["sim"].nlargest(k).mean())

    zone_scores = df.groupby("zone").apply(_score, include_groups=False)
    zone_scores = zone_scores.sort_values(ascending=False)

    return Localisation(
        pipe_pred=str(sims.index[0]),
        zone_pred=int(zone_scores.index[0]),
        zone_scores=zone_scores,
        pipe_scores=sims,
    )


def observed_signature(year: int, day: pd.Timestamp) -> pd.Series:
    """The residual shift fingerprint around a given day.

    Uses the adaptive window, which was selected on the 2018 development year
    (58% top-1 there, against 50% for the best fixed window).
    """
    R = res.daily_residuals(year)
    return res.adaptive_signature(R, pd.Timestamp(day))


def evaluate(year: int, alarm_days: dict[str, pd.Timestamp] | None = None,
             n_zones: int = N.N_ZONES) -> pd.DataFrame:
    """Localise every leak of `year` and score against the true pipe/zone.

    `alarm_days` maps pipe -> the day our detector actually fired. When omitted
    the true onset day is used, which isolates localisation accuracy from
    detection latency.
    """
    zones = N.zone_assignment(n_zones)
    pt = N.pipe_table()
    rows = []

    for lk in sorted(C.leaks_for(year), key=lambda l: l.start):
        day = (alarm_days or {}).get(lk.pipe, pd.Timestamp(lk.start.date()))
        obs = observed_signature(year, day)
        if obs.isna().any():
            continue
        loc = localise(obs, n_zones=n_zones)
        true_zone = int(zones.at[lk.pipe, "zone"])
        order = list(loc.zone_scores.index)
        rank = order.index(true_zone) + 1
        dist = float(
            np.hypot(
                pt.at[loc.pipe_pred, "x"] - pt.at[lk.pipe, "x"],
                pt.at[loc.pipe_pred, "y"] - pt.at[lk.pipe, "y"],
            )
        )
        rows.append(
            {
                "pipe": lk.pipe,
                "kind": lk.kind,
                "diameter_mm": round(lk.diameter_m * 1000, 1),
                "day_used": day.date(),
                "true_zone": true_zone,
                "pred_zone": loc.zone_pred,
                "zone_rank": rank,
                "top1": rank == 1,
                "top3": rank <= 3,
                "pred_pipe": loc.pipe_pred,
                "dist_m": round(dist, 1),
            }
        )

    return pd.DataFrame(rows)
