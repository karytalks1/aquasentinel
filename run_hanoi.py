"""Run the full LeakDB Hanoi study and write every table to reports/tables.

Usage:  .venv/Scripts/python.exe run_hanoi.py

Protocol: scenarios 3, 5 and 6 are the development set and fix every threshold.
Scenarios 7, 8 and 10 are held out and evaluated once. Scenarios 1, 2, 4 and 9
contain no leak at all and measure false alarms on a healthy network.
"""
from __future__ import annotations

import sys
import time
import warnings
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent / "src"))
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd


def main() -> None:
    t0 = time.time()

    import config as C
    import hanoi as H
    import leakdb as LK

    out = C.TABLES
    sensors = LK.sensor_nodes(H.N_SENSORS)

    print("[1/5] dataset")
    summary = LK.summary()
    summary.to_csv(out / "hanoi_scenarios.csv", index=False)
    print(f"      {len(summary)} scenarios, {len(LK.all_leaks())} leak events, "
          f"{len(LK.CLEAN_SCENARIOS)} of them leak-free")
    print(f"      sensors used: {', '.join(sensors)}")

    print("[2/5] residuals")
    for n in LK.available_scenarios():
        H.daily_residuals(n, sensors)
    print(f"      cached for {len(LK.available_scenarios())} scenarios")

    print("[3/5] detection")
    params = H.tune(fa_budget=4, sensors=sensors)
    print(f"      tuned on development scenarios: k={params.k}, h={params.h}, "
          f"refractory={params.refractory_days} d")

    rows, per_leak = [], []
    for label, scen in [("development", LK.DEV_SCENARIOS),
                        ("held-out test", LK.TEST_SCENARIOS)]:
        m = H.evaluate_set(scen, params, sensors)
        rows.append({
            "set": label,
            "scenarios": " ".join(str(s) for s in scen),
            "leaks": m["n_leaks"],
            "detected": m["detected"],
            "detection_rate": round(m["detection_rate"], 4),
            "false_alarms": m["false_alarms"],
            "repair_alarms": m["repair_alarms"],
            "median_delay_days": m["median_delay_days"],
        })
        for ps in m["per_scenario"]:
            for d in ps["detections"]:
                per_leak.append({
                    "set": label, "scenario": d.scenario, "node": d.node,
                    "kind": d.kind, "diameter_mm": round(d.diameter_mm, 1),
                    "onset": d.onset.date(),
                    "alarm_day": d.alarm_day.date() if d.alarm_day is not None else "",
                    "delay_days": d.delay_days if d.delay_days is not None else "",
                    "detected": d.alarm_day is not None,
                })

    clean_fa = H.false_alarms_on_clean(params, sensors)
    rows.append({
        "set": "leak-free control", "scenarios": " ".join(str(s) for s in LK.CLEAN_SCENARIOS),
        "leaks": 0, "detected": 0, "detection_rate": float("nan"),
        "false_alarms": clean_fa, "repair_alarms": 0,
        "median_delay_days": float("nan"),
    })
    det = pd.DataFrame(rows)
    det.to_csv(out / "hanoi_detection_summary.csv", index=False)
    pd.DataFrame(per_leak).to_csv(out / "hanoi_detections.csv", index=False)
    print(det.to_string(index=False))

    print("[4/5] localisation")
    loc = H.evaluate_localisation(LK.DEV_SCENARIOS + LK.TEST_SCENARIOS,
                                  sensors=sensors)
    loc.to_csv(out / "hanoi_localisation.csv", index=False)

    loc_rows = []
    for label, scen in [("development", LK.DEV_SCENARIOS),
                        ("held-out test", LK.TEST_SCENARIOS),
                        ("all", LK.DEV_SCENARIOS + LK.TEST_SCENARIOS)]:
        sub = loc[loc["scenario"].isin(scen)]
        if not len(sub):
            continue
        loc_rows.append({
            "set": label, "n": len(sub),
            "top1": round(float(sub["top1"].mean()), 4),
            "top3": round(float(sub["top3"].mean()), 4),
            "median_rank": float(sub["rank"].median()),
            "median_dist_m": float(sub["dist_m"].median()),
            "random_top1": round(1 / 31, 4),
        })
    loc_sum = pd.DataFrame(loc_rows)
    loc_sum.to_csv(out / "hanoi_localisation_summary.csv", index=False)
    print(loc_sum.to_string(index=False))

    print("[5/5] sensitivity to model error")
    mism = model_mismatch(H, LK, sensors)
    mism.to_csv(out / "hanoi_model_mismatch.csv", index=False)
    print(mism.to_string(index=False))

    print(f"\ndone in {time.time() - t0:.0f}s   tables -> {out}")


def model_mismatch(H, LK, sensors) -> pd.DataFrame:
    """Localise using a different scenario's model, to quantify model error.

    LeakDB perturbs pipe diameters and roughness independently per scenario, so
    borrowing another scenario's .inp is a direct test of how much the method
    depends on owning an accurate model of the network.
    """
    coords = LK.node_coords()
    rows = []
    for own in (True, False):
        hits1, hits3, dists = [], [], []
        for n in LK.TEST_SCENARIOS:
            lib_scenario = n if own else LK.CLEAN_SCENARIOS[0]
            for e in LK.leaks(n):
                obs = H.observed_signature(n, pd.Timestamp(e.start.date()),
                                           sensors=sensors)
                if obs.isna().any():
                    continue
                ranking = H.localise(obs, lib_scenario, sensors)
                order = list(ranking.index)
                rank = order.index(e.node) + 1 if e.node in order else len(order)
                hits1.append(rank == 1)
                hits3.append(rank <= 3)
                dists.append(float(np.hypot(
                    coords.at[order[0], "x"] - coords.at[e.node, "x"],
                    coords.at[order[0], "y"] - coords.at[e.node, "y"])))
        rows.append({
            "library_model": "this scenario's own model" if own
                             else f"another scenario's model (Scenario-{LK.CLEAN_SCENARIOS[0]})",
            "n": len(hits1),
            "top1": round(float(np.mean(hits1)), 4),
            "top3": round(float(np.mean(hits3)), 4),
            "median_dist_m": round(float(np.median(dists)), 1),
        })
    return pd.DataFrame(rows)


if __name__ == "__main__":
    main()
