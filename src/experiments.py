"""Run the full study end to end and write every result table to reports/tables.

Protocol
--------
2018 is the development year: detector thresholds and the signature window are
chosen here. 2019 is touched only to produce the final numbers. Nothing in the
2019 data influences any hyper-parameter.
"""
from __future__ import annotations

import json
import warnings

import numpy as np
import pandas as pd

import config as C
import data_loader as dl
import detect as D
import localize as L
import network as N
import residuals as res

warnings.filterwarnings("ignore")


def mnf_baseline(year: int, z_thresh: float = 3.0,
                 refractory_days: int = 14) -> dict:
    """The method a utility uses today: watch minimum night flow for a step.

    Included so the ML result has an honest point of comparison rather than
    being reported against nothing.
    """
    inflow = dl.total_inflow(year)
    night = inflow.between_time("02:00", "04:00").resample("D").mean()
    diff = night.diff()
    scale = diff.rolling(28, min_periods=10).std().shift(1)
    z = (diff / scale).fillna(0.0)

    alarms, cooldown = [], 0
    for day, val in z.items():
        if cooldown > 0:
            cooldown -= 1
            continue
        if abs(val) >= z_thresh:
            alarms.append(D.Alarm(pd.Timestamp(day), "MNF", float(abs(val))))
            cooldown = refractory_days
    return D.evaluate(alarms, year)


def run() -> dict:
    out: dict = {}

    # ---- 1. Detection -----------------------------------------------------
    params = D.tune(C.TRAIN_YEAR, fa_budget=6)
    out["detector_params"] = {"k": params.k, "h": params.h,
                              "refractory_days": params.refractory_days}

    det_rows, alarm_map = [], {}
    for year in (C.TRAIN_YEAR, C.TEST_YEAR):
        Z = res.normalised(res.daily_residuals(year))
        alarms, trace = D.run_detector(Z, params)
        m = D.evaluate(alarms, year)
        alarm_map[year] = {mm["pipe"]: pd.Timestamp(mm["alarm_day"]) for mm in m["matches"]}
        base = mnf_baseline(year)
        det_rows.append(
            {
                "year": year,
                "role": "development" if year == C.TRAIN_YEAR else "held-out test",
                "leaks": m["n_leaks"],
                "detected": m["detected"],
                "detection_rate": round(m["detection_rate"], 4),
                "false_alarms": m["false_alarms"],
                "median_delay_days": m["median_delay_days"],
                "mnf_detected": base["detected"],
                "mnf_detection_rate": round(base["detection_rate"], 4),
                "mnf_false_alarms": base["false_alarms"],
                "mnf_median_delay_days": base["median_delay_days"],
            }
        )
        pd.DataFrame(m["matches"]).to_csv(C.TABLES / f"detections_{year}.csv", index=False)

    det = pd.DataFrame(det_rows)
    det.to_csv(C.TABLES / "detection_summary.csv", index=False)
    out["detection"] = det.to_dict("records")

    # ---- 2. Localisation --------------------------------------------------
    loc_rows = []
    for year in (C.TRAIN_YEAR, C.TEST_YEAR):
        df = L.evaluate(year)
        df.to_csv(C.TABLES / f"localisation_{year}.csv", index=False)
        for kind, sub in [("all", df), *df.groupby("kind")]:
            if len(sub) == 0:
                continue
            loc_rows.append(
                {
                    "year": year,
                    "subset": kind,
                    "n": len(sub),
                    "top1": round(sub["top1"].mean(), 4),
                    "top3": round(sub["top3"].mean(), 4),
                    "median_zone_rank": float(sub["zone_rank"].median()),
                    "median_dist_m": float(sub["dist_m"].median()),
                    "random_top1": round(1 / N.N_ZONES, 4),
                }
            )
    loc = pd.DataFrame(loc_rows)
    loc.to_csv(C.TABLES / "localisation_summary.csv", index=False)
    out["localisation"] = loc.to_dict("records")

    # ---- 3. Sensor-count ablation ----------------------------------------
    sa = sensor_ablation()
    sa.to_csv(C.TABLES / "sensor_ablation.csv", index=False)
    out["sensor_ablation"] = sa.to_dict("records")

    # ---- 4. Zone-count ablation ------------------------------------------
    za = zone_ablation()
    za.to_csv(C.TABLES / "zone_ablation.csv", index=False)
    out["zone_ablation"] = za.to_dict("records")

    (C.ROOT / "reports" / "results.json").write_text(json.dumps(out, indent=2, default=str))
    return out


def sensor_ablation(counts=(5, 10, 15, 20, 25, 33), seed: int = 0) -> pd.DataFrame:
    """How much does performance depend on how many pressure sensors we have?

    This is the question a utility actually asks, because every extra sensor
    costs money to install and maintain.
    """
    rng = np.random.default_rng(seed)
    R_tr = res.daily_residuals(C.TRAIN_YEAR)
    R_te = res.daily_residuals(C.TEST_YEAR)
    zones = N.zone_assignment()["zone"]
    rows = []

    for n_sensors in counts:
        det_rates, top1s, fas, delays = [], [], [], []
        for _ in range(5 if n_sensors < 33 else 1):
            cols = (list(R_te.columns) if n_sensors >= 33
                    else list(rng.choice(R_te.columns, n_sensors, replace=False)))
            p = D.tune_on(R_tr[cols], C.TRAIN_YEAR)
            Z = res.normalised(R_te[cols])
            alarms, _ = D.run_detector(Z, p)
            m = D.evaluate(alarms, C.TEST_YEAR)
            det_rates.append(m["detection_rate"])
            # Detection rate alone is gameable - a lower threshold always finds
            # more leaks - so the false-alarm count is reported alongside it.
            fas.append(m["false_alarms"])
            delays.append(m["median_delay_days"])

            hits = []
            for lk in C.leaks_for(C.TEST_YEAR):
                obs = res.adaptive_signature(R_te[cols], pd.Timestamp(lk.start.date()))
                if obs.isna().any():
                    continue
                loc = L.localise(obs)
                hits.append(loc.zone_pred == int(zones.at[lk.pipe]))
            top1s.append(float(np.mean(hits)) if hits else np.nan)

        rows.append(
            {
                "n_sensors": n_sensors,
                "detection_rate": round(float(np.mean(det_rates)), 4),
                "false_alarms": round(float(np.mean(fas)), 1),
                "median_delay_days": round(float(np.nanmean(delays)), 1),
                "localisation_top1": round(float(np.nanmean(top1s)), 4),
            }
        )
    return pd.DataFrame(rows)


def zone_ablation(zone_counts=(6, 8, 10, 12, 16, 20)) -> pd.DataFrame:
    """Accuracy versus how finely we try to localise.

    Fewer zones are easier to hit but send the crew over more pipe; the table
    reports both so the trade-off is explicit.
    """
    rows = []
    for k in zone_counts:
        df = L.evaluate(C.TEST_YEAR, n_zones=k)
        stats = N.zone_stats(k)
        rows.append(
            {
                "n_zones": k,
                "top1": round(df["top1"].mean(), 4),
                "top3": round(df["top3"].mean(), 4),
                "random_top1": round(1 / k, 4),
                "mean_zone_length_km": round(float(stats["length_km"].mean()), 2),
            }
        )
    return pd.DataFrame(rows)


if __name__ == "__main__":
    r = run()
    print(json.dumps(r["detection"], indent=2, default=str))
    print(pd.DataFrame(r["localisation"]).to_string(index=False))
