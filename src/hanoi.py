"""Leak detection and localisation on the LeakDB Hanoi benchmark.

Method, in four steps:

1. Every scenario begins with at least 63 leak-free days (verified against the
   published labels), so the first REF_DAYS of each scenario are used as its own
   healthy reference. This matters because LeakDB gives each scenario its own
   perturbed pipe diameters and roughness, so a baseline learned on one scenario
   does not transfer to another.
2. A Ridge regression per sensor predicts pressure from inflow, inflow squared
   and calendar terms. The residual (measured minus predicted) is near zero
   while the network is healthy.
3. A two-sided CUSUM on the robust z-scores of those residuals raises an alarm
   when accumulated evidence crosses a threshold. (k, h) are tuned on the
   development scenarios only.
4. To localise, a leak is simulated at each of the 31 junctions using that
   scenario's own model, giving a library of pressure fingerprints. The observed
   fingerprint is matched to the library by cosine similarity.

Unlike BattLeDIM this dataset ships the true model for each scenario, so the
simulated fingerprints can be checked against the data they are matched to.
"""
from __future__ import annotations

import warnings
from dataclasses import dataclass

import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge

import config as C
import detect as D
import leakdb as LK

warnings.filterwarnings("ignore")

WINDOW_DAYS = 21       # trailing window the baseline is refitted on each day
GAP_DAYS = 2           # days skipped before "today", so a leak that started
                       # yesterday cannot contaminate its own reference
REPAIR_WINDOW_DAYS = 14  # an alarm this soon after a leak is repaired is
                         # reporting the repair, not a phantom leak
REF_DAYS = 60          # leak-free opening period (every scenario has >= 63 days);
                       # used only to set the alarm scale and to exclude the
                       # warm-up from evaluation
RIDGE_ALPHA = 1.0
N_SENSORS = 8

# Reference leak used to build the fingerprint library. Only the direction of
# the response is used, so the exact size does not matter; 100 mm sits in the
# middle of the 21-200 mm leaks present in the benchmark.
LIB_DIAMETER_M = 0.100
DISCHARGE_COEFF = 0.75
G = 9.81


def emitter_coefficient(diameter_m: float = LIB_DIAMETER_M) -> float:
    """EPANET emitter coefficient for a round orifice, in SI units.

    EPANET models an emitter as Q = C * sqrt(P). The orifice equation gives
    Q = Cd * A * sqrt(2 * g * P), so by inspection C = Cd * A * sqrt(2 * g),
    with Q in m^3/s and P in metres (WNTR works in SI regardless of the units
    declared in the .inp file). No numerical calibration is needed.
    """
    area = np.pi * (diameter_m / 2) ** 2
    return DISCHARGE_COEFF * area * np.sqrt(2 * G)


def reference_leak_flow(diameter_m: float = LIB_DIAMETER_M,
                        head_m: float = 50.0) -> float:
    """Discharge of that orifice at a typical head, m^3/h (for reporting)."""
    return emitter_coefficient(diameter_m) * np.sqrt(head_m) * 3600


# ---------------------------------------------------------------------------
# Residuals
# ---------------------------------------------------------------------------
def _design(scenario: int, sensors: tuple[str, ...]) -> tuple[pd.DataFrame, pd.DataFrame]:
    P = LK.pressures(scenario)[list(sensors)]
    inflow = LK.inflow(scenario).reindex(P.index)
    X = pd.concat(
        [inflow.rename("inflow"), (inflow ** 2).rename("inflow_sq"),
         LK.time_features(P.index)],
        axis=1,
    )
    return X, P


def daily_residuals(scenario: int, sensors: tuple[str, ...] | None = None,
                    use_cache: bool = True) -> pd.DataFrame:
    """Mean residual per sensor per day, in metres.

    The baseline is refitted every day on a trailing window rather than once at
    the start of the year. A single fixed baseline was tried first and fails
    here: demand follows a yearly cycle (inflow swings from about 6160 m3/h in
    Q3 to 5020 in Q4), and a window short enough to be certainly leak-free is
    far too short to learn that cycle, so the residual drifts and the detector
    alarms on a perfectly healthy network.

    A trailing window sidesteps the problem entirely - it is never more than a
    few weeks old, so seasonal change is absorbed - and it is the same scheme
    used on the BattLeDIM network, so one method covers both benchmarks.
    """
    sensors = sensors or LK.sensor_nodes(N_SENSORS)
    tag = f"{scenario}_{len(sensors)}"
    cache = LK.PROCESSED / f"residuals_{tag}.csv"
    if use_cache and cache.exists():
        return pd.read_csv(cache, index_col=0, parse_dates=True)

    X, P = _design(scenario, sensors)
    days = pd.date_range(X.index[0].normalize(), X.index[-1].normalize(), freq="D")
    rows: dict[pd.Timestamp, pd.Series] = {}

    for day in days[WINDOW_DAYS + GAP_DAYS:]:
        tr0 = day - pd.Timedelta(days=WINDOW_DAYS + GAP_DAYS)
        tr1 = day - pd.Timedelta(days=GAP_DAYS)
        m_tr = (X.index >= tr0) & (X.index < tr1)
        m_te = (X.index >= day) & (X.index < day + pd.Timedelta(days=1))
        if m_tr.sum() < 200 or m_te.sum() < 12:
            continue
        model = Ridge(alpha=RIDGE_ALPHA).fit(X[m_tr], P[m_tr])
        rows[day] = (P[m_te] - model.predict(X[m_te])).mean()

    daily = pd.DataFrame(rows).T
    daily.index.name = "day"
    daily.to_csv(cache)
    return daily


def zscores(R: pd.DataFrame) -> pd.DataFrame:
    """Standardise each sensor by the spread of its healthy reference period.

    The scale comes from the scenario's leak-free opening period, which the
    published labels confirm is healthy in every scenario (at least 63 days).
    Each day's residual there is produced by a model fitted on earlier days, so
    it is an honest out-of-sample error.
    """
    start = R.index[0]
    ref = R.loc[: start + pd.Timedelta(days=REF_DAYS)]
    centre = ref.median()
    scale = (1.4826 * (ref - centre).abs().median()).clip(lower=1e-4)
    return (R - centre) / scale


# ---------------------------------------------------------------------------
# Detection
# ---------------------------------------------------------------------------
@dataclass
class Detection:
    scenario: int
    node: str
    kind: str
    diameter_mm: float
    onset: pd.Timestamp
    alarm_day: pd.Timestamp | None
    delay_days: float | None


def detect(scenario: int, params: D.DetectorParams,
           sensors: tuple[str, ...] | None = None):
    """Run the CUSUM detector over one scenario. Returns (alarms, trace)."""
    Z = zscores(daily_residuals(scenario, sensors))
    # The reference period defines "normal"; alarms there would be circular.
    Z = Z.loc[Z.index[0] + pd.Timedelta(days=REF_DAYS):]
    return D.run_detector(Z, params)


def evaluate_scenario(scenario: int, params: D.DetectorParams,
                      max_delay_days: int = 60,
                      sensors: tuple[str, ...] | None = None) -> dict:
    """Match alarms to this scenario's leak onsets, earliest first."""
    alarms, trace = detect(scenario, params, sensors)
    events = list(LK.leaks(scenario))
    unmatched = {e.node: e for e in events}
    rows, false_alarms = [], []

    # A leak that is still running keeps the residual shifted, so the detector
    # re-alarms while it remains unrepaired. Those repeats are not false: they
    # report a leak that genuinely is there. An alarm only counts as false when
    # the published labels say nothing was leaking that day.
    leaking_day = LK.labels(scenario).resample("1D").max()

    ends = [pd.Timestamp(e.end) for e in events]

    def after_repair(day: pd.Timestamp) -> bool:
        """True if a leak was repaired in the days just before this alarm.

        The detector measures change, so it reacts when a leak stops just as it
        reacts when one starts: the pressure returns to its healthy level and
        the trailing reference, which had absorbed the leak, is briefly wrong.
        Counting those as phantom leaks would misrepresent the method.
        """
        return any(0 <= (day.date() - e.date()).days <= REPAIR_WINDOW_DAYS
                   for e in ends)

    repair_alarms = []
    for a in sorted(alarms, key=lambda x: x.day):
        cand = [e for e in unmatched.values()
                if 0 <= (a.day.date() - e.start.date()).days <= max_delay_days]
        if not cand:
            if after_repair(a.day):
                repair_alarms.append(a.day)
            elif not bool(leaking_day.get(a.day, False)):
                false_alarms.append(a.day)
            continue
        e = min(cand, key=lambda x: x.start)
        rows.append(Detection(scenario, e.node, e.kind, e.diameter_mm,
                              pd.Timestamp(e.start), a.day,
                              (a.day.date() - e.start.date()).days))
        del unmatched[e.node]

    for e in unmatched.values():
        rows.append(Detection(scenario, e.node, e.kind, e.diameter_mm,
                              pd.Timestamp(e.start), None, None))

    delays = [r.delay_days for r in rows if r.delay_days is not None]
    return {
        "scenario": scenario,
        "n_leaks": len(events),
        "detected": len(delays),
        "false_alarms": len(false_alarms),
        "repair_alarms": len(repair_alarms),
        "median_delay_days": float(np.median(delays)) if delays else float("nan"),
        "detections": rows,
        "false_alarm_days": false_alarms,
        "alarms": alarms,
        "trace": trace,
    }


def evaluate_set(scenarios, params: D.DetectorParams,
                 sensors: tuple[str, ...] | None = None) -> dict:
    """Aggregate detection results over several scenarios."""
    per = [evaluate_scenario(n, params, sensors=sensors) for n in scenarios]
    leaks = sum(p["n_leaks"] for p in per)
    det = sum(p["detected"] for p in per)
    fa = sum(p["false_alarms"] for p in per)
    repairs = sum(p["repair_alarms"] for p in per)
    delays = [d.delay_days for p in per for d in p["detections"]
              if d.delay_days is not None]
    return {
        "scenarios": list(scenarios),
        "n_leaks": leaks,
        "detected": det,
        "detection_rate": det / leaks if leaks else float("nan"),
        "false_alarms": fa,
        "repair_alarms": repairs,
        "median_delay_days": float(np.median(delays)) if delays else float("nan"),
        "per_scenario": per,
    }


def false_alarms_on_clean(params: D.DetectorParams,
                          sensors: tuple[str, ...] | None = None) -> int:
    """Alarms raised on the four scenarios that never leak: all are false."""
    total = 0
    for n in LK.CLEAN_SCENARIOS:
        alarms, _ = detect(n, params, sensors)
        total += len(alarms)
    return total


REFRACTORY_DAYS = 30


def tune(dev_scenarios=LK.DEV_SCENARIOS, fa_budget: int = 2,
         sensors: tuple[str, ...] | None = None) -> D.DetectorParams:
    """Grid-search (k, h) on the development scenarios only.

    False alarms are counted on the development scenarios and on the clean
    scenarios, and the budget is the binding constraint: a detector that cries
    wolf is switched off by the people who have to respond to it.
    """
    best, best_key = None, None
    # k must exceed any small systematic bias in the residual, or the CUSUM
    # accumulates it indefinitely and alarms on a perfectly healthy network,
    # so the grid reaches well above 1 sigma.
    for k in (0.5, 1.0, 1.5, 2.0, 2.5, 3.0, 4.0):
        for h in (4, 6, 8, 10, 12, 15, 20, 25):
            p = D.DetectorParams(k=k, h=h, refractory_days=REFRACTORY_DAYS)
            m = evaluate_set(dev_scenarios, p, sensors)
            fa = m["false_alarms"] + false_alarms_on_clean(p, sensors)
            if fa > fa_budget:
                continue
            key = (m["detected"],
                   -m["median_delay_days"] if m["detected"] else -1e9, -fa)
            if best_key is None or key > best_key:
                best_key, best = key, p
    if best is None:
        raise RuntimeError(
            f"no (k, h) on the grid stayed within {fa_budget} false alarms; "
            "raise fa_budget or widen the grid rather than accepting defaults"
        )
    return best


# ---------------------------------------------------------------------------
# Localisation
# ---------------------------------------------------------------------------
SIM_HOURS = 24


def _simulate_pressures(wn, sensors) -> pd.Series:
    import wntr
    wn.options.time.duration = SIM_HOURS * 3600
    wn.options.time.hydraulic_timestep = 3600
    wn.options.time.report_timestep = 3600
    res = wntr.sim.EpanetSimulator(wn).run_sim()
    return res.node["pressure"][list(sensors)].mean()


def node_library(scenario: int, sensors: tuple[str, ...] | None = None,
                 use_cache: bool = True) -> pd.DataFrame:
    """Simulated pressure response at the sensors for a leak at each junction.

    Row = the junction carrying the leak, column = sensor, value = change in
    mean pressure. The simulation uses this scenario's own .inp, which LeakDB
    perturbs per scenario - so the model matches the data it will be compared
    against, unlike BattLeDIM where the true model is withheld.
    """
    sensors = sensors or LK.sensor_nodes(N_SENSORS)
    cache = LK.PROCESSED / f"library_{scenario}_{len(sensors)}.csv"
    if use_cache and cache.exists():
        return pd.read_csv(cache, index_col=0, dtype={0: str})

    import wntr

    coeff = emitter_coefficient()
    base = _simulate_pressures(
        wntr.network.WaterNetworkModel(str(LK.inp_file(scenario))), sensors)

    rows: dict[str, pd.Series] = {}
    wn0 = wntr.network.WaterNetworkModel(str(LK.inp_file(scenario)))
    for node in wn0.junction_name_list:
        wn = wntr.network.WaterNetworkModel(str(LK.inp_file(scenario)))
        wn.get_node(node).emitter_coefficient = coeff
        try:
            rows[node] = _simulate_pressures(wn, sensors) - base
        except Exception:
            continue

    lib = pd.DataFrame(rows).T
    lib.index.name = "node"
    lib.to_csv(cache)
    return lib


def unit_rows(df: pd.DataFrame) -> pd.DataFrame:
    n = np.linalg.norm(df.to_numpy(), axis=1, keepdims=True)
    n[n == 0] = 1.0
    return pd.DataFrame(df.to_numpy() / n, index=df.index, columns=df.columns)


def observed_signature(scenario: int, day: pd.Timestamp, pre: int = 10,
                       post: int = 10,
                       sensors: tuple[str, ...] | None = None) -> pd.Series:
    """Change in each sensor's residual around `day`: the leak's fingerprint."""
    R = daily_residuals(scenario, sensors)
    day = pd.Timestamp(day)
    before = R.loc[day - pd.Timedelta(days=pre): day - pd.Timedelta(days=1)]
    after = R.loc[day: day + pd.Timedelta(days=post)]
    if len(before) < 3 or len(after) < 3:
        return pd.Series(np.nan, index=R.columns)
    return after.mean() - before.mean()


def localise(observed: pd.Series, scenario: int,
             sensors: tuple[str, ...] | None = None) -> pd.Series:
    """Rank every junction by how well its simulated response matches.

    Only the direction of the response is compared, because the size of the
    real leak is unknown at detection time.
    """
    lib = unit_rows(node_library(scenario, sensors))
    cols = [c for c in lib.columns if c in observed.index]
    v = observed[cols].to_numpy(dtype=float)
    norm = np.linalg.norm(v)
    if norm == 0 or not np.isfinite(norm):
        return pd.Series(dtype=float)
    return pd.Series(lib[cols].to_numpy() @ (v / norm),
                     index=lib.index.astype(str)).sort_values(ascending=False)


def evaluate_localisation(scenarios, params: D.DetectorParams | None = None,
                          sensors: tuple[str, ...] | None = None,
                          use_alarm_day: bool = False) -> pd.DataFrame:
    """Localise every leak in `scenarios` and score against the true junction.

    With `use_alarm_day` the fingerprint is taken around the day the detector
    actually fired, which is the operational case; otherwise the true onset is
    used, which separates localisation accuracy from detection latency.
    """
    coords = LK.node_coords()
    rows = []
    for n in scenarios:
        alarm_for = {}
        if use_alarm_day and params is not None:
            for d in evaluate_scenario(n, params, sensors=sensors)["detections"]:
                if d.alarm_day is not None:
                    alarm_for[d.node] = d.alarm_day
        for e in LK.leaks(n):
            day = alarm_for.get(e.node, pd.Timestamp(e.start.date()))
            obs = observed_signature(n, day, sensors=sensors)
            if obs.isna().any():
                continue
            ranking = localise(obs, n, sensors)
            if ranking.empty:
                continue
            order = list(ranking.index)
            rank = order.index(e.node) + 1 if e.node in order else len(order)
            pred = order[0]
            dist = float(np.hypot(coords.at[pred, "x"] - coords.at[e.node, "x"],
                                  coords.at[pred, "y"] - coords.at[e.node, "y"]))
            rows.append({
                "scenario": n, "true_node": e.node, "kind": e.kind,
                "diameter_mm": round(e.diameter_mm, 1), "pred_node": pred,
                "rank": rank, "top1": rank == 1, "top3": rank <= 3,
                "dist_m": round(dist, 1), "n_candidates": len(order),
            })
    return pd.DataFrame(rows)
