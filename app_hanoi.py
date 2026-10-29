"""AquaSentinel - Hanoi dashboard (LeakDB benchmark).

Run with:  .venv/Scripts/streamlit run app_hanoi.py
"""
from __future__ import annotations

import sys
import warnings
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent / "src"))
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.colors import sample_colorscale
import streamlit as st

import config as C
import detect as D
import hanoi as H
import leakdb as LK

st.set_page_config(page_title="AquaSentinel - Hanoi", page_icon="💧", layout="wide")

CYAN, GREEN, RED, AMBER, GREY = "#22b8cf", "#34d399", "#f87171", "#fbbf24", "#8b9bb4"
PLOT_CFG = {"displaylogo": False}

st.markdown(
    """
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800&display=swap');
html, body, [class*="css"], .stMarkdown { font-family: 'Inter', system-ui, sans-serif; }
.block-container { padding-top: 1.6rem; padding-bottom: 3rem; max-width: 1400px; }
header[data-testid="stHeader"] { background: transparent; }
.hero { border-radius: 20px; padding: 30px 36px 26px; margin-bottom: 20px;
  background: radial-gradient(1100px 380px at 85% -10%, rgba(34,184,207,.33), transparent 60%),
              linear-gradient(135deg, #0a2540 0%, #0d3b66 55%, #0e5a7a 100%);
  border: 1px solid rgba(255,255,255,.08); box-shadow: 0 18px 44px rgba(0,0,0,.33); }
.hero h1 { font-size: 2.3rem; font-weight: 800; margin: 0 0 6px; color: #fff; letter-spacing: -.02em; }
.hero p { font-size: 1rem; color: #cfe3f3; margin: 0; max-width: 840px; line-height: 1.55; }
.badges { margin-top: 14px; display: flex; flex-wrap: wrap; gap: 8px; }
.badge { font-size: .76rem; font-weight: 600; padding: 5px 11px; border-radius: 999px;
  background: rgba(255,255,255,.10); color: #e6f4fb; border: 1px solid rgba(255,255,255,.14); }
.kpi { border-radius: 16px; padding: 17px 19px 15px; height: 100%;
  background: linear-gradient(180deg, rgba(255,255,255,.055), rgba(255,255,255,.02));
  border: 1px solid rgba(255,255,255,.08); }
.kpi .label { font-size: .72rem; font-weight: 600; letter-spacing: .08em; text-transform: uppercase; color: #8b9bb4; }
.kpi .value { font-size: 2.1rem; font-weight: 800; line-height: 1.15; margin-top: 4px; color: #fff; }
.kpi .value small { font-size: .95rem; font-weight: 600; color: #8b9bb4; margin-left: 4px; }
.kpi .cmp { font-size: .8rem; margin-top: 6px; color: #9fb1c8; }
.kpi .cmp b { color: #34d399; }
.section-title { font-size: 1.2rem; font-weight: 700; margin: 24px 0 4px; color: #fff; }
.section-sub { font-size: .88rem; color: #8b9bb4; margin-bottom: 12px; }
.step { border-radius: 14px; padding: 15px; height: 100%;
  background: rgba(255,255,255,.035); border: 1px solid rgba(255,255,255,.07); }
.step .num { display:inline-flex; align-items:center; justify-content:center; width:25px; height:25px;
  border-radius:8px; font-size:.78rem; font-weight:700; color:#0b1320; background:#22b8cf; margin-bottom:7px; }
.step h4 { margin: 0 0 4px; font-size: .95rem; font-weight: 700; color: #fff; }
.step p { margin: 0; font-size: .82rem; line-height: 1.5; color: #a9b8cc; }
.verdict { border-radius: 14px; padding: 13px 17px; margin: 6px 0 14px; font-size: .96rem;
  border: 1px solid; display: flex; gap: 12px; align-items: center; }
.verdict.ok { background: rgba(52,211,153,.10); border-color: rgba(52,211,153,.40); color: #d1fae5; }
.verdict.bad { background: rgba(251,191,36,.10); border-color: rgba(251,191,36,.40); color: #fef3c7; }
.verdict .icon { font-size: 1.4rem; }
.mono { font-family: ui-monospace, Menlo, Consolas, monospace; font-size: .86rem; color: #cfe3f3; }
</style>
""",
    unsafe_allow_html=True,
)


def html(markup: str) -> None:
    st.markdown(markup, unsafe_allow_html=True)


def kpi(col, label, value, cmp=""):
    col.markdown(f'<div class="kpi"><div class="label">{label}</div>'
                 f'<div class="value">{value}</div><div class="cmp">{cmp}</div></div>',
                 unsafe_allow_html=True)


def style_fig(fig, height, legend=True):
    fig.update_layout(template="plotly_dark", height=height,
                      paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
                      font=dict(family="Inter, system-ui", color="#cdd9e5", size=12),
                      margin=dict(l=8, r=8, t=14, b=8), showlegend=legend,
                      legend=dict(bgcolor="rgba(11,19,32,.6)",
                                  bordercolor="rgba(255,255,255,.08)", borderwidth=1,
                                  font=dict(size=11)),
                      hoverlabel=dict(bgcolor="#111c2e", bordercolor="#22b8cf"))
    fig.update_xaxes(gridcolor="rgba(255,255,255,.06)", zeroline=False)
    fig.update_yaxes(gridcolor="rgba(255,255,255,.06)", zeroline=False)
    return fig


def as_map(fig):
    fig.update_xaxes(visible=False)
    fig.update_yaxes(visible=False, scaleanchor="x", scaleratio=1)
    return fig


# ---------------------------------------------------------------------------
# Cached work
# ---------------------------------------------------------------------------
@st.cache_data(show_spinner="Loading network...")
def geometry():
    return LK.node_coords(), LK.pipe_table()


@st.cache_data(show_spinner="Computing residuals...")
def residuals(scenario: int, n_sensors: int) -> pd.DataFrame:
    return H.daily_residuals(scenario, LK.sensor_nodes(n_sensors))


@st.cache_data(show_spinner="Tuning detector on development scenarios...")
def tuned(n_sensors: int):
    p = H.tune(fa_budget=4, sensors=LK.sensor_nodes(n_sensors))
    return float(p.k), float(p.h), int(p.refractory_days)


@st.cache_data(show_spinner="Running detector...")
def detection(scenario: int, n_sensors: int, k: float, h: float, r: int):
    params = D.DetectorParams(k=k, h=h, refractory_days=r)
    res = H.evaluate_scenario(scenario, params, sensors=LK.sensor_nodes(n_sensors))
    alarms = pd.DataFrame([{"day": a.day, "sensor": a.sensor, "score": a.score}
                           for a in res["alarms"]])
    rows = [{"node": d.node, "kind": d.kind, "diameter_mm": d.diameter_mm,
             "onset": d.onset, "alarm_day": d.alarm_day,
             "delay_days": d.delay_days} for d in res["detections"]]
    return alarms, res["trace"], pd.DataFrame(rows), res["false_alarms"], res["repair_alarms"]


@st.cache_data(show_spinner="Simulating leaks at every junction...")
def library(scenario: int, n_sensors: int) -> pd.DataFrame:
    return H.node_library(scenario, LK.sensor_nodes(n_sensors))


@st.cache_data
def results_table(name: str):
    path = C.TABLES / name
    return pd.read_csv(path) if path.exists() else None


@st.cache_data(show_spinner="Running EPANET...")
def simulate_node(scenario: int, node: str, diameter_mm: float, n_sensors: int):
    """One live EPANET run: healthy vs a leak of the chosen size at `node`."""
    import wntr
    sensors = LK.sensor_nodes(n_sensors)
    inp = str(LK.inp_file(scenario))
    base = H._simulate_pressures(wntr.network.WaterNetworkModel(inp), sensors)
    coeff = H.emitter_coefficient(diameter_mm / 1000)
    wn = wntr.network.WaterNetworkModel(inp)
    wn.get_node(node).emitter_coefficient = coeff
    leaked = H._simulate_pressures(wn, sensors)
    return base, leaked, coeff


def network_traces(coords, pipes, line_color="rgba(120,160,200,.45)", width=1.6):
    xs, ys = [], []
    for a, b in zip(pipes["start"].astype(str), pipes["end"].astype(str)):
        xs += [coords.at[a, "x"], coords.at[b, "x"], None]
        ys += [coords.at[a, "y"], coords.at[b, "y"], None]
    return go.Scatter(x=xs, y=ys, mode="lines", line=dict(color=line_color, width=width),
                      hoverinfo="skip", showlegend=False)


# ---------------------------------------------------------------------------
# Sidebar
# ---------------------------------------------------------------------------
with st.sidebar:
    st.markdown("### 💧 AquaSentinel · Hanoi")
    st.caption("LeakDB benchmark · KIOS, University of Cyprus")

    n_sensors = st.select_slider("Pressure sensors", options=[4, 6, 8, 10, 12],
                                 value=H.N_SENSORS)
    k_def, h_def, r_def = tuned(n_sensors)

    scenario = st.selectbox(
        "Scenario", LK.available_scenarios(),
        index=LK.available_scenarios().index(8),
        format_func=lambda n: (f"{n} · " + ("leak-free" if n in LK.CLEAN_SCENARIOS
                                            else "development" if n in LK.DEV_SCENARIOS
                                            else "held-out test")),
    )

    st.markdown("##### Detector")
    st.caption(f"Tuned on scenarios {', '.join(map(str, LK.DEV_SCENARIOS))}: "
               f"k = {k_def:g}, h = {h_def:g}, refractory = {r_def} d")
    k = st.slider("CUSUM slack k", 0.5, 5.0, k_def, 0.5)
    h = st.slider("Alarm threshold h", 2.0, 25.0, h_def, 1.0)
    refr = st.slider("Refractory (days)", 7, 60, r_def)
    if (k, h, refr) == (k_def, h_def, r_def):
        st.success("Settings match the report", icon="✅")
    else:
        st.warning("Settings differ from the report", icon="⚠️")

coords, pipes = geometry()
sensors = LK.sensor_nodes(n_sensors)
events = LK.leaks(scenario)
alarms, trace, det_rows, n_fa, n_repair = detection(scenario, n_sensors, k, h, refr)

html("""
<div class="hero">
  <h1>AquaSentinel · Hanoi</h1>
  <p>Detecting when a <b>new</b> leak starts from a handful of pressure sensors, then
     naming the junction it is at — by matching the measured pressure response against
     a leak simulated at every junction in EPANET.</p>
  <div class="badges">
    <span class="badge">31 junctions · 39.6 km</span>
    <span class="badge">10 scenario-years</span>
    <span class="badge">4 leak-free controls</span>
    <span class="badge">9 labelled leaks</span>
    <span class="badge">LeakDB · CC BY 4.0</span>
  </div>
</div>
""")

tab_overview, tab_detect, tab_local, tab_sim = st.tabs(
    ["📊  Results", "📡  Detection", "🎯  Localisation", "🔬  Simulation (live)"]
)

# ---------------------------------------------------------------------------
# Results
# ---------------------------------------------------------------------------
with tab_overview:
    det = results_table("hanoi_detection_summary.csv")
    loc = results_table("hanoi_localisation_summary.csv")
    cols = st.columns(4)
    if det is not None:
        test = det[det["set"] == "held-out test"].iloc[0]
        clean = det[det["set"] == "leak-free control"].iloc[0]
        kpi(cols[0], "Leaks detected",
            f"{int(test['detected'])}<small>/ {int(test['leaks'])}</small>",
            f"<b>{test['detection_rate']:.0%}</b> on held-out scenarios")
        kpi(cols[1], "Median delay", f"{test['median_delay_days']:.0f}<small>days</small>",
            "from leak start to alarm")
        kpi(cols[2], "False alarms (healthy years)",
            f"{int(clean['false_alarms'])}", "across 4 leak-free scenarios")
    if loc is not None:
        lt = loc[loc["set"] == "held-out test"].iloc[0]
        kpi(cols[3], "Exact junction found", f"{lt['top1']:.0%}",
            f"<b>{lt['top1'] / lt['random_top1']:.0f}×</b> better than chance "
            f"({lt['random_top1']:.1%})")

    html('<div class="section-title">How it works</div>'
         '<div class="section-sub">Four stages, from sensor readings to a junction</div>')
    steps = [
        ("Predict", "A regression learns what each sensor should read from inflow and "
                    "time of day, refitted daily on the previous 21 days."),
        ("Compare", "Residual = measured − predicted. About 1.5 cm when healthy; a new "
                    "leak steps it away and keeps it there."),
        ("Detect", "CUSUM accumulates the daily deviations like a running balance and "
                   "alarms when they cross a threshold."),
        ("Locate", "A leak is simulated at each of the 31 junctions; the measured "
                   "fingerprint is matched to the closest simulated one."),
    ]
    for i, (col, (title, text)) in enumerate(zip(st.columns(4), steps), 1):
        col.markdown(f'<div class="step"><div class="num">{i}</div><h4>{title}</h4>'
                     f'<p>{text}</p></div>', unsafe_allow_html=True)

    for title, name, note in [
        ("Detection", "hanoi_detection_summary.csv",
         "Thresholds fixed on the development scenarios, then the held-out set "
         "evaluated once. Repair alarms are counted separately: the detector measures "
         "change, and a repair is a change."),
        ("Localisation", "hanoi_localisation_summary.csv",
         "Rank of the true junction among 31 candidates. Random choice scores 3.2%."),
        ("Sensitivity to model error", "hanoi_model_mismatch.csv",
         "Repeating localisation with another scenario's model, whose pipe diameters "
         "differ by up to 30%."),
        ("Per-leak detection", "hanoi_detections.csv", ""),
        ("Scenarios", "hanoi_scenarios.csv", ""),
    ]:
        df = results_table(name)
        if df is not None:
            with st.expander(title):
                st.dataframe(df, width="stretch", hide_index=True)
                if note:
                    st.caption(note)

# ---------------------------------------------------------------------------
# Detection
# ---------------------------------------------------------------------------
with tab_detect:
    role = ("leak-free control" if scenario in LK.CLEAN_SCENARIOS
            else "development" if scenario in LK.DEV_SCENARIOS else "held-out test")
    html(f'<div class="section-title">Scenario {scenario} · {role}</div>'
         f'<div class="section-sub">One simulated year at 30-minute resolution, '
         f'{len(sensors)} pressure sensors</div>')

    c = st.columns(4)
    found = int(det_rows["alarm_day"].notna().sum()) if len(det_rows) else 0
    kpi(c[0], "Leaks in scenario", str(len(events)))
    kpi(c[1], "Detected", f"{found}<small>/ {len(events)}</small>")
    kpi(c[2], "False alarms", str(n_fa), "excluding repair transients")
    kpi(c[3], "Repair alarms", str(n_repair), "raised when a leak is fixed")

    fig = go.Figure()
    fig.add_trace(go.Scatter(x=trace.index, y=trace.to_numpy(), mode="lines",
                             name="Evidence (CUSUM)", fill="tozeroy",
                             line=dict(color=CYAN, width=2),
                             fillcolor="rgba(34,184,207,.12)"))
    fig.add_hline(y=float(h), line_dash="dash", line_color=AMBER,
                  annotation_text="alarm line", annotation_font_color=AMBER)
    for e in events:
        fig.add_vline(x=pd.Timestamp(e.start), line_color="rgba(52,211,153,.55)",
                      line_width=2)
        fig.add_vline(x=pd.Timestamp(e.end), line_color="rgba(139,155,180,.45)",
                      line_width=1, line_dash="dot")
    if len(alarms):
        fig.add_trace(go.Scatter(
            x=alarms["day"], y=[trace.loc[d] for d in alarms["day"]], mode="markers",
            name="Alarm", marker=dict(size=12, color=RED, symbol="triangle-down"),
            hovertext=[f"{d:%d %b} · sensor {s}" for d, s in
                       zip(alarms["day"], alarms["sensor"])], hoverinfo="text"))
    style_fig(fig, 380)
    fig.update_layout(yaxis_title="accumulated evidence",
                      legend=dict(orientation="h", y=1.1))
    st.plotly_chart(fig, width="stretch", config=PLOT_CFG)
    st.caption("Green lines are leak starts, dotted grey lines are repairs. "
               "In the leak-free scenarios there are no green lines and the trace "
               "should never reach the alarm line.")

    if len(det_rows):
        show = det_rows.copy()
        show["onset"] = pd.to_datetime(show["onset"]).dt.strftime("%d %b %Y")
        show["alarm_day"] = show["alarm_day"].apply(
            lambda d: pd.Timestamp(d).strftime("%d %b %Y") if pd.notna(d) else "not detected")
        st.dataframe(show, width="stretch", hide_index=True)

    with st.expander("Residual z-scores, every sensor"):
        R = residuals(scenario, n_sensors)
        Z = H.zscores(R).clip(-8, 8)
        heat = go.Figure(go.Heatmap(z=Z.T.to_numpy(), x=Z.index, y=Z.columns,
                                    colorscale="RdBu_r", zmid=0,
                                    colorbar=dict(title=dict(text="z"))))
        style_fig(heat, 380, legend=False)
        st.plotly_chart(heat, width="stretch", config=PLOT_CFG)

# ---------------------------------------------------------------------------
# Localisation
# ---------------------------------------------------------------------------
with tab_local:
    if not events:
        st.info(f"Scenario {scenario} contains no leak. "
                "Pick a development or held-out scenario to localise one.")
    else:
        idx = st.selectbox(
            "Leak", range(len(events)),
            format_func=lambda i: (f"node {events[i].node} · {events[i].start:%d %b %Y} · "
                                   f"{events[i].kind} · {events[i].diameter_mm:.0f} mm"),
            label_visibility="collapsed")
        e = events[idx]
        obs = H.observed_signature(scenario, pd.Timestamp(e.start.date()),
                                   sensors=sensors)
        if obs.isna().any():
            st.warning("Not enough residual history around this date.")
        else:
            ranking = H.localise(obs, scenario, sensors)
            order = list(ranking.index)
            rank = order.index(e.node) + 1
            pred = order[0]
            dist = float(np.hypot(coords.at[pred, "x"] - coords.at[e.node, "x"],
                                  coords.at[pred, "y"] - coords.at[e.node, "y"]))
            if rank == 1:
                html(f'<div class="verdict ok"><span class="icon">🎯</span><div>'
                     f'<b>Correct junction, first guess.</b> Predicted node {pred}, '
                     f'ranked 1 of {len(order)}.</div></div>')
            else:
                html(f'<div class="verdict bad"><span class="icon">🧭</span><div>'
                     f'<b>True junction ranked {rank} of {len(order)}.</b> '
                     f'Best guess was node {pred}, {dist:.0f} m away.</div></div>')

            cc = st.columns(4)
            kpi(cc[0], "Predicted junction", pred, f"true: node {e.node}")
            kpi(cc[1], "Rank of true junction", f"{rank}<small>/ {len(order)}</small>")
            kpi(cc[2], "Distance error", f"{dist:.0f}<small>m</small>")
            kpi(cc[3], "Leak size", f"{e.diameter_mm:.0f}<small>mm</small>", e.kind)

            left, right = st.columns([1.3, 1])
            with left:
                lo, hi = float(ranking.min()), float(ranking.max())
                fig = go.Figure()
                fig.add_trace(network_traces(coords, pipes))
                score_col = [sample_colorscale(
                    "Turbo", [0.05 + 0.9 * ((float(ranking.get(n, lo)) - lo) /
                                            (hi - lo) if hi > lo else 0)])[0]
                    for n in coords.index]
                fig.add_trace(go.Scatter(
                    x=coords["x"], y=coords["y"], mode="markers", name="junction",
                    marker=dict(size=14, color=score_col,
                                line=dict(width=1, color="#0b1320")),
                    hovertext=[f"node {n} · score {ranking.get(n, float('nan')):+.3f}"
                               for n in coords.index], hoverinfo="text"))
                sc = coords.loc[list(sensors)]
                fig.add_trace(go.Scatter(x=sc["x"], y=sc["y"], mode="markers",
                                         name="sensor",
                                         marker=dict(size=9, color="#fff",
                                                     symbol="triangle-up")))
                fig.add_trace(go.Scatter(
                    x=[coords.at[e.node, "x"]], y=[coords.at[e.node, "y"]],
                    mode="markers", name=f"true leak (node {e.node})",
                    marker=dict(size=22, color=GREEN, symbol="star",
                                line=dict(width=2, color="#0b1320"))))
                style_fig(fig, 460)
                as_map(fig)
                fig.update_layout(legend=dict(orientation="h", y=-0.02))
                st.plotly_chart(fig, width="stretch", config=PLOT_CFG)
                st.caption("Junctions shaded by how well their simulated response "
                           "matches what the sensors measured.")

            with right:
                top = ranking.head(8)[::-1]
                bar = go.Figure(go.Bar(
                    x=top.to_numpy(), y=[f"node {n}" for n in top.index],
                    orientation="h",
                    marker=dict(color=[GREEN if n == e.node else "rgba(139,155,180,.45)"
                                       for n in top.index]),
                    text=[f"{v:.2f}" for v in top.to_numpy()], textposition="outside"))
                style_fig(bar, 380, legend=False)
                bar.update_traces(cliponaxis=False)
                bar.update_xaxes(range=[min(0, float(top.min())) * 1.1,
                                        float(top.max()) * 1.3])
                bar.update_layout(xaxis_title="similarity to simulated response")
                st.plotly_chart(bar, width="stretch", config=PLOT_CFG)

            lib = H.unit_rows(library(scenario, n_sensors))
            simd = lib.loc[e.node].reindex(obs.index)
            obs_u = obs / np.linalg.norm(obs.to_numpy())
            cos = float(np.dot(obs_u.to_numpy(), simd.to_numpy()))
            html(f'<div class="section-title">Measured against simulated</div>'
                 f'<div class="section-sub">The simulated leak is 100 mm, the real one '
                 f'{e.diameter_mm:.0f} mm — only the shape is compared · cosine '
                 f'similarity <b style="color:#22b8cf">{cos:.2f}</b></div>')
            cmp = go.Figure()
            cmp.add_trace(go.Bar(x=obs.index, y=obs_u.to_numpy(),
                                 name="measured", marker_color=CYAN))
            cmp.add_trace(go.Bar(x=obs.index, y=simd.to_numpy(),
                                 name="simulated (EPANET)", marker_color=AMBER))
            style_fig(cmp, 300)
            cmp.update_layout(barmode="group", bargap=.25,
                              xaxis_title="sensor", yaxis_title="normalised change",
                              legend=dict(orientation="h", y=1.12))
            st.plotly_chart(cmp, width="stretch", config=PLOT_CFG)

# ---------------------------------------------------------------------------
# Live simulation
# ---------------------------------------------------------------------------
with tab_sim:
    html('<div class="section-title">Open a hole anywhere and watch the pressures</div>'
         '<div class="section-sub">Each run solves the hydraulics twice with EPANET — '
         'once healthy, once leaking — and subtracts. This is exactly how the '
         'fingerprint library is built, one junction at a time.</div>')

    c1, c2 = st.columns([1, 1])
    node = c1.selectbox("Leak at junction", list(coords.index),
                        index=list(coords.index).index(
                            events[0].node if events else list(coords.index)[0]))
    dia = c2.slider("Hole diameter (mm)", 20, 300, int(H.LIB_DIAMETER_M * 1000), 10)

    base, leaked, coeff = simulate_node(scenario, node, float(dia), n_sensors)
    drop = leaked - base
    area = np.pi * (dia / 2000) ** 2
    q = coeff * np.sqrt(50.0) * 3600

    html(f'<div class="mono">A = π(d/2)² = {area:.5f} m² &nbsp;·&nbsp; '
         f'C = Cd·A·√(2g) = {H.DISCHARGE_COEFF} × {area:.5f} × {np.sqrt(2 * H.G):.3f} '
         f'= <b>{coeff:.5f}</b> &nbsp;·&nbsp; at 50 m head that is '
         f'<b>{q:.0f} m³/h</b> escaping</div>')
    st.caption("The emitter coefficient comes straight from the orifice equation. "
               "Nothing here is fitted or tuned.")

    cc = st.columns(3)
    kpi(cc[0], "Biggest pressure drop", f"{drop.min():.2f}<small>m</small>",
        f"at sensor {drop.idxmin()}")
    kpi(cc[1], "Smallest", f"{drop.max():.2f}<small>m</small>",
        f"at sensor {drop.idxmax()}")
    kpi(cc[2], "Water escaping", f"{q:.0f}<small>m³/h</small>", f"{dia} mm hole")

    left, right = st.columns([1, 1.1])
    with left:
        fig = go.Figure()
        fig.add_trace(network_traces(coords, pipes))
        fig.add_trace(go.Scatter(x=coords["x"], y=coords["y"], mode="markers",
                                 name="junction",
                                 marker=dict(size=9, color="rgba(160,180,210,.6)"),
                                 hovertext=[f"node {n}" for n in coords.index],
                                 hoverinfo="text"))
        sc = coords.loc[list(sensors)]
        mag = drop.abs()
        fig.add_trace(go.Scatter(
            x=sc["x"], y=sc["y"], mode="markers+text", name="sensor",
            marker=dict(size=10 + 34 * (mag / mag.max()).to_numpy(), color=RED,
                        opacity=.75, line=dict(width=1, color="#0b1320")),
            text=[f"{drop[s]:.2f}" for s in sensors], textposition="top center",
            textfont=dict(size=10, color="#fff"),
            hovertext=[f"sensor {s}: {drop[s]:+.3f} m" for s in sensors],
            hoverinfo="text"))
        fig.add_trace(go.Scatter(x=[coords.at[node, "x"]], y=[coords.at[node, "y"]],
                                 mode="markers", name=f"leak (node {node})",
                                 marker=dict(size=20, color=GREEN, symbol="star",
                                             line=dict(width=2, color="#0b1320"))))
        style_fig(fig, 470)
        as_map(fig)
        fig.update_layout(legend=dict(orientation="h", y=-0.02))
        st.plotly_chart(fig, width="stretch", config=PLOT_CFG)
        st.caption("Circle size is how hard each sensor reacts. The nearest sensor "
                   "always moves most — that is what makes the leak locatable.")

    with right:
        fig = go.Figure()
        fig.add_trace(go.Bar(x=list(sensors), y=base[list(sensors)].to_numpy(),
                             name="healthy", marker_color="rgba(139,155,180,.55)"))
        fig.add_trace(go.Bar(x=list(sensors), y=leaked[list(sensors)].to_numpy(),
                             name="leaking", marker_color=CYAN))
        style_fig(fig, 230)
        fig.update_layout(barmode="group", yaxis_title="pressure (m)",
                          xaxis_title="sensor", legend=dict(orientation="h", y=1.18),
                          margin=dict(t=36))
        st.plotly_chart(fig, width="stretch", config=PLOT_CFG)

        fig = go.Figure(go.Bar(x=list(sensors), y=drop[list(sensors)].to_numpy(),
                               marker_color=RED))
        style_fig(fig, 230, legend=False)
        fig.update_layout(yaxis_title="change (m)", xaxis_title="sensor",
                          title=dict(text="Fingerprint = leaking − healthy",
                                     font=dict(size=13), x=0), margin=dict(t=36))
        st.plotly_chart(fig, width="stretch", config=PLOT_CFG)

    with st.expander("Where this leak would be ranked, if it were real"):
        ranking = H.localise(drop, scenario, sensors)
        order = list(ranking.index)
        st.write(f"Matching this simulated fingerprint back against the library ranks "
                 f"node **{order[0]}** first; the true node **{node}** is ranked "
                 f"**{order.index(node) + 1} of {len(order)}**.")
        st.caption("A sanity check on the method: a perfect, noise-free fingerprint "
                   "should recover its own junction.")
