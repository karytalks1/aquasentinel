"""Leak onset detection by CUSUM change detection on pressure residual z-scores.

The detector watches the 33 normalised residual streams. A new leak produces a
small but *persistent* shift, which is exactly what a CUSUM is built to catch:
it accumulates evidence over days, so a slow incipient leak eventually trips the
threshold even though no single day looks abnormal.

Hyper-parameters (k, h) are tuned on 2018 and then frozen; 2019 is touched only
for the final evaluation.
"""
from __future__ import annotations

from dataclasses import dataclass, asdict

import numpy as np
import pandas as pd

import config as C
import residuals as res


@dataclass
class Alarm:
    day: pd.Timestamp
    sensor: str
    score: float


@dataclass
class DetectorParams:
    k: float = 0.5           # CUSUM slack: shifts smaller than k*sigma are ignored
    h: float = 6.0           # alarm threshold on the accumulated statistic
    refractory_days: int = 14  # silence after an alarm, so one leak = one alarm


def cusum(Z: pd.DataFrame, k: float) -> pd.DataFrame:
    """Two-sided CUSUM per sensor. Returns the per-day, per-sensor statistic."""
    Zf = Z.fillna(0.0).to_numpy()
    n, m = Zf.shape
    sp = np.zeros(m)
    sn = np.zeros(m)
    out = np.zeros((n, m))
    for t in range(n):
        sp = np.maximum(0.0, sp + Zf[t] - k)
        sn = np.maximum(0.0, sn - Zf[t] - k)
        out[t] = np.maximum(sp, sn)
    return pd.DataFrame(out, index=Z.index, columns=Z.columns)


def run_detector(Z: pd.DataFrame, p: DetectorParams) -> tuple[list[Alarm], pd.Series]:
    """Sequentially scan the z-score stream and emit alarms.

    The CUSUM state is reset after every alarm; without this one leak would
    raise an alarm every single day until it was repaired.
    """
    Zf = Z.fillna(0.0)
    cols = list(Zf.columns)
    arr = Zf.to_numpy()
    sp = np.zeros(len(cols))
    sn = np.zeros(len(cols))
    alarms: list[Alarm] = []
    trace = np.zeros(len(arr))
    cooldown = 0

    for t, day in enumerate(Zf.index):
        sp = np.maximum(0.0, sp + arr[t] - p.k)
        sn = np.maximum(0.0, sn - arr[t] - p.k)
        stat = np.maximum(sp, sn)
        trace[t] = stat.max()
        if cooldown > 0:
            cooldown -= 1
            continue
        j = int(stat.argmax())
        if stat[j] >= p.h:
            alarms.append(Alarm(day, cols[j], float(stat[j])))
            sp[:] = 0.0
            sn[:] = 0.0
            cooldown = p.refractory_days

    return alarms, pd.Series(trace, index=Zf.index, name="cusum")


def evaluate(alarms: list[Alarm], year: int, max_delay_days: int = 45) -> dict:
    """Match alarms to ground-truth onsets, greedily and earliest-first.

    An alarm counts as a detection if it falls within `max_delay_days` after an
    as-yet-unmatched leak onset. Every other alarm is a false alarm.
    """
    leaks = sorted(C.leaks_for(year), key=lambda l: l.start)
    unmatched = {lk.pipe: lk for lk in leaks}
    matches: list[dict] = []
    false_alarms: list[Alarm] = []

    for a in sorted(alarms, key=lambda x: x.day):
        cand = [
            lk for lk in unmatched.values()
            if 0 <= (a.day.date() - lk.start.date()).days <= max_delay_days
        ]
        if not cand:
            false_alarms.append(a)
            continue
        lk = min(cand, key=lambda l: l.start)  # attribute to the oldest open leak
        matches.append(
            {
                "pipe": lk.pipe,
                "kind": lk.kind,
                "diameter_mm": lk.diameter_m * 1000,
                "onset": lk.start,
                "alarm_day": a.day,
                "delay_days": (a.day.date() - lk.start.date()).days,
                "sensor": a.sensor,
                "score": a.score,
            }
        )
        del unmatched[lk.pipe]

    detected = len(matches)
    delays = [m["delay_days"] for m in matches]
    return {
        "n_leaks": len(leaks),
        "n_alarms": len(alarms),
        "detected": detected,
        "detection_rate": detected / len(leaks) if leaks else float("nan"),
        "false_alarms": len(false_alarms),
        "missed": [lk.pipe for lk in unmatched.values()],
        "median_delay_days": float(np.median(delays)) if delays else float("nan"),
        "mean_delay_days": float(np.mean(delays)) if delays else float("nan"),
        "matches": matches,
        "false_alarm_days": [a.day for a in false_alarms],
    }


def tune(year: int = C.TRAIN_YEAR, fa_budget: int = 6) -> DetectorParams:
    """Grid-search (k, h) on the training year's full sensor set."""
    return tune_on(res.daily_residuals(year), year, fa_budget)


def tune_on(R: pd.DataFrame, year: int, fa_budget: int = 6) -> DetectorParams:
    """Grid-search (k, h) against a given residual frame.

    Objective: maximise detections, break ties by lower median delay, subject to
    staying inside a false-alarm budget. A utility cannot chase 50 phantom leaks
    a year, so the budget - not raw accuracy - is the binding constraint.

    Taking the residual frame as an argument (rather than a year) lets the
    sensor-count ablation re-tune honestly for each reduced sensor set.
    """
    Z = res.normalised(R)
    best, best_key = None, None
    for k in (0.25, 0.5, 0.75, 1.0, 1.5):
        for h in (3, 4, 5, 6, 8, 10, 12, 15):
            p = DetectorParams(k=k, h=h)
            alarms, _ = run_detector(Z, p)
            m = evaluate(alarms, year)
            if m["false_alarms"] > fa_budget:
                continue
            key = (m["detected"], -m["median_delay_days"] if m["detected"] else -999,
                   -m["false_alarms"])
            if best_key is None or key > best_key:
                best_key, best = key, p
    return best or DetectorParams()
