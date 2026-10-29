"""Walk through one Hanoi leak simulation, printing every intermediate number.

Usage:  .venv/Scripts/python.exe src/demo_hanoi.py [scenario] [node]
        .venv/Scripts/python.exe src/demo_hanoi.py 8 21

Everything the localisation step does to all 31 junctions is done here to one
junction, slowly, so each number can be checked by hand.
"""
from __future__ import annotations

import sys
import warnings
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd

import hanoi as H
import leakdb as LK

SCENARIO = int(sys.argv[1]) if len(sys.argv) > 1 else 8
NODE = sys.argv[2] if len(sys.argv) > 2 else "21"
RULE = "=" * 72


def main() -> None:
    import wntr

    sensors = LK.sensor_nodes(H.N_SENSORS)
    print(RULE)
    print(f"LEAK SIMULATION WALKTHROUGH - Scenario {SCENARIO}, leak at node {NODE}")
    print(RULE)

    # ---- 1. the model ----------------------------------------------------
    inp = LK.inp_file(SCENARIO)
    wn = wntr.network.WaterNetworkModel(str(inp))
    total_km = sum(wn.get_link(p).length for p in wn.pipe_name_list) / 1000
    print(f"\n1. THE NETWORK MODEL  ({inp.name})")
    print(f"   {len(wn.junction_name_list)} junctions, {len(wn.pipe_name_list)} pipes,"
          f" {total_km:.1f} km, {len(wn.reservoir_name_list)} reservoir")
    print(f"   LeakDB ships this scenario's OWN model, so the simulation can be")
    print(f"   compared with the data it generated.")
    print(f"   Sensors we are allowed to read: {', '.join(sensors)}")

    # ---- 2. healthy run --------------------------------------------------
    base = H._simulate_pressures(
        wntr.network.WaterNetworkModel(str(inp)), sensors)
    print("\n2. BASELINE RUN - healthy network, 24 h, hourly steps")
    for s in sensors:
        print(f"      sensor {s:>3}: {base[s]:7.2f} m")

    # ---- 3. how big is the hole? ----------------------------------------
    d = H.LIB_DIAMETER_M
    area = np.pi * (d / 2) ** 2
    coeff = H.emitter_coefficient()
    print("\n3. SIZE OF THE SIMULATED HOLE")
    print(f"   diameter  d  = {d * 1000:.0f} mm")
    print(f"   area      A  = pi*(d/2)^2 = {area:.5f} m2")
    print("   Orifice equation:  Q = Cd * A * sqrt(2*g*P)")
    print("   EPANET emitter:    Q = C  * sqrt(P)")
    print(f"   so  C = Cd * A * sqrt(2g) = {H.DISCHARGE_COEFF} * {area:.5f}"
          f" * {np.sqrt(2 * H.G):.3f} = {coeff:.5f}")
    print(f"   At a typical {50:.0f} m head that is"
          f" {H.reference_leak_flow():.0f} m3/h of water lost.")

    # ---- 4. leak run -----------------------------------------------------
    wn2 = wntr.network.WaterNetworkModel(str(inp))
    wn2.get_node(NODE).emitter_coefficient = coeff
    leaked = H._simulate_pressures(wn2, sensors)
    print(f"\n4. LEAK RUN - same network, emitter switched on at node {NODE}")
    drop = leaked - base
    print("      sensor   healthy     leaking     change")
    for s in sensors:
        print(f"      {s:>6} {base[s]:9.2f} {leaked[s]:11.2f} {drop[s]:+10.3f} m")

    # ---- 5. the fingerprint ---------------------------------------------
    print("\n5. FINGERPRINT = leaking - healthy")
    nearest = drop.abs().idxmax()
    print(f"   Sensor {nearest} reacts most ({drop[nearest]:+.2f} m); the leak is"
          f" hydraulically closest to it.")
    print(f"   Weakest reaction: sensor {drop.abs().idxmin()}"
          f" ({drop[drop.abs().idxmin()]:+.2f} m).")

    unit = drop / np.linalg.norm(drop.to_numpy())
    print("\n6. NORMALISE TO UNIT LENGTH")
    print("   The size of a real leak is unknown, so only the SHAPE of the")
    print("   response can be used to locate it.")
    print(f"   length before = {np.linalg.norm(drop.to_numpy()):.3f}"
          f"   after = {np.linalg.norm(unit.to_numpy()):.3f}")

    # ---- 7. repeat for every junction -----------------------------------
    lib = H.node_library(SCENARIO, sensors)
    print(f"\n7. REPEAT FOR ALL {len(lib)} JUNCTIONS  ->  the library")
    print(f"   stored as data/processed/leakdb/library_{SCENARIO}_{len(sensors)}.csv")

    # ---- 8. match the measured leak -------------------------------------
    event = next((e for e in LK.leaks(SCENARIO) if e.node == NODE), None)
    if event is None:
        print("\n(no real leak at this node in this scenario, so no comparison)")
        return

    obs = H.observed_signature(SCENARIO, pd.Timestamp(event.start.date()),
                               sensors=sensors)
    print(f"\n8. THE REAL MEASURED LEAK")
    print(f"   Node {event.node}, {event.kind}, {event.diameter_mm:.0f} mm,"
          f" started {event.start:%d %b %Y}")
    print("   Measured change in each sensor's residual around that date:")
    for s in sensors:
        print(f"      sensor {s:>3}: {obs[s]:+.3f} m")

    ranking = H.localise(obs, SCENARIO, sensors)
    order = list(ranking.index)
    rank = order.index(NODE) + 1
    print("\n9. MATCH MEASURED AGAINST SIMULATED (cosine similarity)")
    for i, (nd, sc) in enumerate(ranking.head(5).items(), 1):
        mark = "  <-- the true leak node" if nd == NODE else ""
        print(f"      {i}. node {nd:>3}   similarity {sc:+.3f}{mark}")
    print(f"\n   True node ranked {rank} of {len(order)}"
          f"   (random guess would be {len(order) // 2} on average)")

    print("\n" + RULE)


if __name__ == "__main__":
    main()
