"""Walk through one leak simulation, printing every intermediate number.

Usage:  .venv/Scripts/python.exe src/demo_simulation.py [pipe]

Everything the localisation step does to all 902 pipes is done here to a single
pipe, slowly and with the numbers shown, so the method can be checked by hand.
"""
from __future__ import annotations

import sys
import warnings
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd

import config as C
import network as N
import simulate as sim

PIPE = sys.argv[1] if len(sys.argv) > 1 else "p523"
RULE = "-" * 68


def main() -> None:
    import wntr

    print(RULE)
    print(f"LEAK SIMULATION WALKTHROUGH - pipe {PIPE}")
    print(RULE)

    # ---- 1. the network model -------------------------------------------
    wn = wntr.network.WaterNetworkModel(str(C.INP_FILE))
    print("\n1. NETWORK MODEL (L-TOWN.inp, published with the dataset)")
    print(f"   junctions {len(wn.junction_name_list)}, pipes {len(wn.pipe_name_list)},"
          f" tanks {len(wn.tank_name_list)}, reservoirs {len(wn.reservoir_name_list)}")
    print(f"   pressure sensors we are allowed to read: {len(C.PRESSURE_SENSORS)}")

    # ---- 2. healthy network ---------------------------------------------
    print("\n2. BASELINE RUN - no leak, 24 h, hourly steps")
    base = sim.baseline_pressures()
    print(f"   mean pressure over the {len(base)} sensors: {base.mean():.2f} m")
    print("   example sensors:", ", ".join(f"{s}={base[s]:.2f} m"
                                           for s in base.index[:4]))

    # ---- 3. how big a leak are we simulating? ---------------------------
    q = sim.target_leak_flow()
    area = np.pi * (sim.LEAK_DIAMETER_M / 2) ** 2
    print("\n3. SIZE OF THE SIMULATED LEAK (orifice equation)")
    print(f"   hole diameter   d = {sim.LEAK_DIAMETER_M * 1000:.0f} mm"
          "   (median of the 33 real leaks)")
    print(f"   hole area       A = pi*(d/2)^2 = {area:.3e} m2")
    print(f"   velocity        v = sqrt(2*g*h) with h = {sim.TYPICAL_HEAD_M:.0f} m"
          f"  ->  {np.sqrt(2 * sim.G * sim.TYPICAL_HEAD_M):.2f} m/s")
    print(f"   discharge       Q = Cd*A*v with Cd = {sim.DISCHARGE_COEFF}"
          f"  ->  {q:.5f} m3/s = {q * 3600:.1f} m3/h")

    # ---- 4. the emitter --------------------------------------------------
    coeff = sim._calibrate_emitter()
    print("\n4. PUTTING THAT LEAK INTO EPANET (as an emitter)")
    print("   An emitter is a pressure-dependent hole: Q = C * sqrt(P).")
    print("   C is calibrated once so the hole discharges the flow above:")
    print(f"   emitter coefficient C = {coeff:.5f}")

    node_name = wn.get_link(PIPE).start_node_name
    print(f"   pipe {PIPE} runs {node_name} -> {wn.get_link(PIPE).end_node_name};"
          f" the leak is placed at {node_name}")

    # ---- 5. leak run -----------------------------------------------------
    wn.get_node(node_name).emitter_coefficient = coeff
    wn.options.time.duration = sim.SIM_HOURS * 3600
    wn.options.time.hydraulic_timestep = 3600
    wn.options.time.report_timestep = 3600
    res = wntr.sim.EpanetSimulator(wn).run_sim()
    leaked = res.node["pressure"][C.PRESSURE_SENSORS].mean()
    actual_q = float(res.node["demand"][node_name].mean())

    print("\n5. LEAK RUN - identical network, emitter switched on")
    print(f"   water escaping at {node_name}: {actual_q * 3600:.1f} m3/h")
    print(f"   mean sensor pressure falls {base.mean():.2f} -> {leaked.mean():.2f} m")

    # ---- 6. the fingerprint ---------------------------------------------
    drop = leaked - base
    print("\n6. FINGERPRINT = leak pressures - baseline pressures")
    print("   the five sensors that react most:")
    for s, v in drop.abs().nlargest(5).items():
        print(f"     {s:>5}  {base[s]:7.2f} -> {leaked[s]:7.2f} m   "
              f"change {drop[s]:+.3f} m")
    print(f"   sensors barely affected: "
          + ", ".join(f"{s} ({drop[s]:+.3f})" for s in drop.abs().nsmallest(3).index))

    # ---- 7. normalisation ------------------------------------------------
    unit = drop / np.linalg.norm(drop.to_numpy())
    print("\n7. NORMALISE TO UNIT LENGTH")
    print("   The real leak's size is unknown, so only the SHAPE of the response")
    print("   can be used. Dividing by the vector length removes the size.")
    print(f"   length before = {np.linalg.norm(drop.to_numpy()):.4f}"
          f"   after = {np.linalg.norm(unit.to_numpy()):.4f}")

    # ---- 8. compare with the stored library and with reality -------------
    stored = sim.unit_signatures().loc[PIPE].reindex(unit.index)
    print("\n8. CHECK AGAINST THE STORED LIBRARY")
    print("   cosine similarity with the cached row for this pipe: "
          f"{float(np.dot(unit.to_numpy(), stored.to_numpy())):.4f}  (1.0 = identical)")

    leak = next((lk for lk in C.LEAKS if lk.pipe == PIPE), None)
    if leak is not None:
        import residuals as res_mod
        R = res_mod.daily_residuals(leak.year)
        obs = res_mod.adaptive_signature(R, pd.Timestamp(leak.start.date()))
        if not obs.isna().any():
            obs_u = obs / np.linalg.norm(obs.to_numpy())
            cos = float(np.dot(obs_u.to_numpy(), stored.to_numpy()))
            print("\n9. COMPARE SIMULATION WITH THE REAL MEASURED LEAK")
            print(f"   {PIPE} really leaked on {leak.start:%d %b %Y} "
                  f"({leak.kind}, {leak.diameter_m * 1000:.1f} mm)")
            print(f"   cosine similarity simulated vs measured: {cos:.3f}")
            zone = N.zone_assignment()
            print(f"   that pipe sits in district {int(zone.at[PIPE, 'zone'])}")

    print("\n" + RULE)
    print("The localisation step repeats steps 5-7 for all 902 pipes and stores")
    print("the result as data/processed/sensitivity_matrix.csv.")
    print(RULE)


if __name__ == "__main__":
    main()
