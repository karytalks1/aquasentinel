"""Build a leak sensitivity matrix for L-Town with EPANET (via WNTR).

Only 14 real leaks exist in the training year - nowhere near enough to learn
where a leak is from its pressure fingerprint. So we generate the fingerprints
hydraulically instead: put a leak on every one of the ~900 pipes in turn, and
record how the 33 sensor pressures respond.

The leak is modelled as an EPANET *emitter* at the pipe's upstream node, which
gives the physically correct pressure-dependent discharge Q = C*sqrt(P) rather
than a fixed withdrawal. Because the true leak size is unknown at detection
time, every signature is later normalised to unit length - only the *direction*
of the response across sensors carries location information.
"""
from __future__ import annotations

import warnings

import numpy as np
import pandas as pd

import config as C
import network as N

warnings.filterwarnings("ignore")

SIM_HOURS = 24

# Equivalent orifice for the emitter, chosen as the median of the 33 real
# BattLeDIM leaks (diameters span 8.8 - 22.9 mm).
LEAK_DIAMETER_M = 0.015
DISCHARGE_COEFF = 0.75
TYPICAL_HEAD_M = 45.0
G = 9.81

# WNTR reports flows in SI (m^3/s) whatever the .inp file's own unit setting,
# and its emitter coefficient follows the same convention. Rather than guess the
# conversion we calibrate it numerically once, in _calibrate_emitter().
_EMITTER_CACHE: dict[str, float] = {}


def target_leak_flow() -> float:
    """Discharge of the reference leak in m^3/s, from the orifice equation."""
    area = np.pi * (LEAK_DIAMETER_M / 2) ** 2
    return DISCHARGE_COEFF * area * np.sqrt(2 * G * TYPICAL_HEAD_M)


def _calibrate_emitter(probe_pipes: int = 5) -> float:
    """Find the emitter coefficient giving `target_leak_flow()` on average.

    Emitter discharge is linear in the coefficient at fixed pressure, so a
    single probe run per pipe is enough to rescale.
    """
    if "c" in _EMITTER_CACHE:
        return _EMITTER_CACHE["c"]

    import wntr

    probe_c = 0.05
    pipes = list(N.pipe_table().index)
    step = max(1, len(pipes) // probe_pipes)
    flows = []
    for pipe in pipes[::step][:probe_pipes]:
        wn = wntr.network.WaterNetworkModel(str(C.INP_FILE))
        node_name = wn.get_link(pipe).start_node_name
        node = wn.get_node(node_name)
        if not hasattr(node, "emitter_coefficient"):
            continue
        node.emitter_coefficient = probe_c
        wn.options.time.duration = SIM_HOURS * 3600
        wn.options.time.hydraulic_timestep = 3600
        wn.options.time.report_timestep = 3600
        try:
            r = wntr.sim.EpanetSimulator(wn).run_sim()
        except Exception:
            continue
        flows.append(float(r.node["demand"][node_name].mean()))

    if not flows:
        raise RuntimeError("emitter calibration failed on every probe pipe")
    coeff = probe_c * target_leak_flow() / float(np.mean(flows))
    _EMITTER_CACHE["c"] = coeff
    return coeff


def _simulate(wn) -> pd.Series:
    import wntr
    wn.options.time.duration = SIM_HOURS * 3600
    wn.options.time.hydraulic_timestep = 3600
    wn.options.time.report_timestep = 3600
    results = wntr.sim.EpanetSimulator(wn).run_sim()
    return results.node["pressure"][C.PRESSURE_SENSORS].mean()


def baseline_pressures() -> pd.Series:
    import wntr
    wn = wntr.network.WaterNetworkModel(str(C.INP_FILE))
    return _simulate(wn)


def sensitivity_matrix(use_cache: bool = True) -> pd.DataFrame:
    """(pipes x 33 sensors) matrix of pressure change caused by a leak.

    Row p, column s = mean drop in pressure at sensor s when pipe p leaks.
    """
    cache = C.PROCESSED / "sensitivity_matrix.csv"
    if use_cache and cache.exists():
        return pd.read_csv(cache, index_col=0)

    import wntr

    base = baseline_pressures()
    emitter_c = _calibrate_emitter()
    print(f"  calibrated emitter coefficient = {emitter_c:.5f} "
          f"(target leak {target_leak_flow() * 3600:.1f} m3/h)")
    pipes = list(N.pipe_table().index)
    rows: dict[str, pd.Series] = {}

    for i, pipe in enumerate(pipes, 1):
        wn = wntr.network.WaterNetworkModel(str(C.INP_FILE))
        node_name = wn.get_link(pipe).start_node_name
        node = wn.get_node(node_name)
        if not hasattr(node, "emitter_coefficient"):
            continue  # tank or reservoir endpoint
        node.emitter_coefficient = emitter_c
        try:
            rows[pipe] = _simulate(wn) - base
        except Exception:
            continue  # a handful of pipes make the hydraulics fail to converge
        if i % 100 == 0:
            print(f"  simulated {i}/{len(pipes)} pipes")

    S = pd.DataFrame(rows).T
    S.index.name = "pipe"
    S.to_csv(cache)
    return S


def unit_signatures(S: pd.DataFrame | None = None) -> pd.DataFrame:
    """Row-normalise the sensitivity matrix to unit vectors (direction only)."""
    if S is None:
        S = sensitivity_matrix()
    norm = np.linalg.norm(S.to_numpy(), axis=1, keepdims=True)
    norm[norm == 0] = 1.0
    return pd.DataFrame(S.to_numpy() / norm, index=S.index, columns=S.columns)


if __name__ == "__main__":
    S = sensitivity_matrix(use_cache=False)
    print("sensitivity matrix", S.shape)
    print("mean |dP| per pipe: %.4f m" % S.abs().mean(axis=1).mean())
    print("largest single response: %.3f m" % S.abs().to_numpy().max())
