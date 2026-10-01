"""AquaSentinel - interactive leak detection and localisation dashboard.

Run with:  .venv/Scripts/streamlit run app.py
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
from plotly.colors import qualitative, sample_colorscale
import streamlit as st

import config as C
import detect as D
import localize as L
import network as N
import residuals as res

st.set_page_config(page_title="AquaSentinel", page_icon="💧", layout="wide")

# --------------------------------------------------------------------------
# Visual language
# --------------------------------------------------------------------------
CYAN, GREEN, RED, AMBER = "#22b8cf", "#34d399", "#f87171", "#fbbf24"
ZONE_COLORS = qualitative.Light24
PLOT_CFG = {"displaylogo": False}

st.markdown(
    """
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800&display=swap');
html, body, [class*="css"], .stMarkdown, .stText { font-family: 'Inter', system-ui, sans-serif; }
.block-container { padding-top: 1.6rem; padding-bottom: 3rem; max-width: 1400px; }
header[data-testid="stHeader"] { background: transparent; }

.hero {
  position: relative; overflow: hidden; border-radius: 20px;
  padding: 34px 38px 30px;
  background: radial-gradient(1200px 400px at 85% -10%, rgba(34,184,207,.35), transparent 60%),
              linear-gradient(135deg, #0a2540 0%, #0d3b66 55%, #0e5a7a 100%);
  border: 1px solid rgba(255,255,255,.08);
  box-shadow: 0 20px 50px rgba(0,0,0,.35);
  margin-bottom: 22px;
}
.hero h1 { font-size: 2.55rem; font-weight: 800; margin: 0 0 6px; color: #fff; letter-spacing: -0.02em; }
.hero p  { font-size: 1.06rem; color: #cfe3f3; margin: 0; max-width: 820px; line-height: 1.55; }
.badges { margin-top: 16px; display: flex; flex-wrap: wrap; gap: 8px; }
.badge {
  font-size: .76rem; font-weight: 600; letter-spacing: .02em; padding: 5px 11px; border-radius: 999px;
  background: rgba(255,255,255,.10); color: #e6f4fb; border: 1px solid rgba(255,255,255,.14);
}

.kpi {
  border-radius: 16px; padding: 18px 20px 16px; height: 100%;
  background: linear-gradient(180deg, rgba(255,255,255,.055), rgba(255,255,255,.02));
  border: 1px solid rgba(255,255,255,.08);
}
.kpi .label { font-size: .74rem; font-weight: 600; letter-spacing: .08em; text-transform: uppercase; color: #8b9bb4; }
.kpi .value { font-size: 2.35rem; font-weight: 800; line-height: 1.15; margin-top: 4px; color: #fff; }
.kpi .value small { font-size: 1rem; font-weight: 600; color: #8b9bb4; margin-left: 4px; }
.kpi .cmp { font-size: .82rem; margin-top: 6px; color: #9fb1c8; }
.kpi .cmp b { color: #34d399; }
.kpi.accent { border-color: rgba(34,184,207,.45); box-shadow: inset 0 0 0 1px rgba(34,184,207,.15); }

.section-title { font-size: 1.25rem; font-weight: 700; margin: 26px 0 4px; color: #fff; }
.section-sub { font-size: .9rem; color: #8b9bb4; margin-bottom: 12px; }

.step {
  border-radius: 14px; padding: 16px 16px 14px; height: 100%;
  background: rgba(255,255,255,.035); border: 1px solid rgba(255,255,255,.07);
}
.step .num {
  display: inline-flex; align-items: center; justify-content: center; width: 26px; height: 26px;
  border-radius: 8px; font-size: .8rem; font-weight: 700; color: #0b1320; background: #22b8cf; margin-bottom: 8px;
}
.step h4 { margin: 0 0 4px; font-size: .98rem; font-weight: 700; color: #fff; }
.step p  { margin: 0; font-size: .84rem; line-height: 1.5; color: #a9b8cc; }

.verdict {
  border-radius: 14px; padding: 14px 18px; margin: 6px 0 14px; font-size: .98rem;
  border: 1px solid; display: flex; gap: 12px; align-items: center;
}
.verdict.ok  { background: rgba(52,211,153,.10); border-color: rgba(52,211,153,.40); color: #d1fae5; }
.verdict.bad { background: rgba(251,191,36,.10); border-color: rgba(251,191,36,.40); color: #fef3c7; }
.verdict .icon { font-size: 1.5rem; }

section[data-testid="stSidebar"] { border-right: 1px solid rgba(255,255,255,.06); }
.side-brand { font-size: 1.35rem; font-weight: 800; color: #fff; margin-bottom: 2px; }
.side-tag { font-size: .8rem; color: #8b9bb4; margin-bottom: 14px; line-height: 1.45; }

div[data-baseweb="tab-list"] { gap: 6px; }
button[data-baseweb="tab"] { font-weight: 600; }
</style>
""",
    unsafe_allow_html=True,
)


def html(markup: str) -> None:
    st.markdown(markup, unsafe_allow_html=True)


def style_fig(fig: go.Figure, height: int, legend: bool = True) -> go.Figure:
    fig.update_layout(
        template="plotly_dark",
        height=height,
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        font=dict(family="Inter, system-ui, sans-serif", color="#cdd9e5", size=12),
        margin=dict(l=8, r=8, t=12, b=8),
        showlegend=legend,
        legend=dict(bgcolor="rgba(11,19,32,.6)", bordercolor="rgba(255,255,255,.08)",
                    borderwidth=1, font=dict(size=11)),
        hoverlabel=dict(bgcolor="#111c2e", bordercolor="#22b8cf",
                        font=dict(family="Inter", color="#e6edf5")),
    )
    fig.update_xaxes(gridcolor="rgba(255,255,255,.06)", zeroline=False)
    fig.update_yaxes(gridcolor="rgba(255,255,255,.06)", zeroline=False)
    return fig


def map_axes(fig: go.Figure) -> go.Figure:
    fig.update_xaxes(visible=False)
    fig.update_yaxes(visible=False, scaleanchor="x", scaleratio=1)
    return fig


# --------------------------------------------------------------------------
# Cached computations
# --------------------------------------------------------------------------
@st.cache_data(show_spinner="Computing pressure residuals...")
def get_residuals(year: int) -> pd.DataFrame:
    return res.daily_residuals(year)


@st.cache_data(show_spinner="Tuning detector on 2018...")
def get_params() -> tuple[float, float, int]:
    p = D.tune(C.TRAIN_YEAR)
    return float(p.k), float(p.h), int(p.refractory_days)


@st.cache_data(show_spinner="Running detector...")
def get_detection(year: int, k: float, h: float, refractory: int):
    Z = res.normalised(get_residuals(year))
    params = D.DetectorParams(k=k, h=h, refractory_days=refractory)
    alarms, trace = D.run_detector(Z, params)
    metrics = D.evaluate(alarms, year)
    alarm_df = pd.DataFrame([{"day": a.day, "sensor": a.sensor, "score": a.score}
                             for a in alarms])
    return alarm_df, trace, metrics


@st.cache_data(show_spinner="Loading network...")
def get_network(n_zones: int):
    return (N.zone_assignment(n_zones), N.sensor_coords(), N.zone_stats(n_zones),
            N.node_coords(), N.pipe_table())


@st.cache_data
def read_table(name: str) -> pd.DataFrame | None:
    path = C.TABLES / name
    return pd.read_csv(path) if path.exists() else None


def segments(pipes: pd.DataFrame, coords: pd.DataFrame) -> tuple[list, list]:
    """Pipe start/end coordinates as one polyline with None breaks."""
    xs, ys = [], []
    for a, b in zip(pipes["start"], pipes["end"]):
        xs += [coords.at[a, "x"], coords.at[b, "x"], None]
        ys += [coords.at[a, "y"], coords.at[b, "y"], None]
    return xs, ys


# --------------------------------------------------------------------------
# Sidebar
# --------------------------------------------------------------------------
k_def, h_def, r_def = get_params()
DEFAULTS = {"year": C.TEST_YEAR, "n_zones": N.N_ZONES, "k": k_def, "h": h_def,
            "refractory": r_def}
for key, val in DEFAULTS.items():
    st.session_state.setdefault(key, val)


def reset_defaults() -> None:
    for key, val in DEFAULTS.items():
        st.session_state[key] = val


with st.sidebar:
    html('<div class="side-brand">💧 AquaSentinel</div>'
         '<div class="side-tag">Leak detection &amp; localisation on the L-Town '
         'network · BattLeDIM 2020 benchmark</div>')
    year = st.selectbox(
        "Evaluation year", [C.TEST_YEAR, C.TRAIN_YEAR], key="year",
        format_func=lambda y: f"{y} · {'held-out test' if y == C.TEST_YEAR else 'development'}",
    )
    n_zones = st.slider("Search districts", 6, 20, step=2, key="n_zones")

    st.markdown("##### Detector")
    st.caption(f"Tuned on {C.TRAIN_YEAR}: k = {k_def:g}, h = {h_def:g}, "
               f"refractory = {r_def} d")
    k = st.slider("CUSUM slack  k", 0.25, 2.0, step=0.25, key="k",
                  help="Daily abnormality below k·σ is ignored.")
    h = st.slider("Alarm threshold  h", 2.0, 20.0, step=1.0, key="h",
                  help="Accumulated evidence needed to raise an alarm.")
    refractory = st.slider("Refractory period (days)", 5, 30, key="refractory",
                           help="Silence after an alarm, so one leak = one alarm.")

    tuned = (float(k) == k_def and float(h) == h_def and int(refractory) == r_def
             and int(n_zones) == N.N_ZONES)
    if tuned:
        st.success("Using the tuned settings reported in the study", icon="✅")
    else:
        st.warning("Settings differ from the reported configuration", icon="⚠️")
    st.button("↺  Reset to reported settings", on_click=reset_defaults,
              width="stretch")

alarms, trace, metrics = get_detection(year, float(k), float(h), int(refractory))
zones_df, sensors, zstats, coords, ptable = get_network(int(n_zones))
R = get_residuals(year)
leaks = sorted(C.leaks_for(year), key=lambda x: x.start)
baseline = read_table("detection_summary.csv")

# --------------------------------------------------------------------------
# Hero
# --------------------------------------------------------------------------
html(f"""
<div class="hero">
  <h1>AquaSentinel</h1>
  <p>Finding hidden water leaks from the pressure sensors a utility already owns —
     detecting when a <b>new</b> leak starts, and pointing the repair crew to the
     right district.</p>
  <div class="badges">
    <span class="badge">🗺️ 43 km · 905 pipes</span>
    <span class="badge">📡 33 pressure sensors</span>
    <span class="badge">🧪 33 labelled real leaks</span>
    <span class="badge">🔒 {C.TEST_YEAR} held out from tuning</span>
    <span class="badge">⚙️ EPANET physics + ML</span>
  </div>
</div>
""")

tab_overview, tab_replay, tab_explore, tab_evidence = st.tabs(
    ["📊  Overview", "▶️  Year replay", "🎯  Leak explorer", "🔬  Evidence & limits"]
)

# --------------------------------------------------------------------------
# Overview
# --------------------------------------------------------------------------
with tab_overview:
    mnf_rate = mnf_delay = None
    if baseline is not None and (baseline["year"] == year).any():
        row = baseline[baseline["year"] == year].iloc[0]
        mnf_rate, mnf_delay = row["mnf_detection_rate"], row["mnf_median_delay_days"]

    top1 = None
    ls_ = read_table("localisation_summary.csv")
    if ls_ is not None:
        m = ls_[(ls_["year"] == year) & (ls_["subset"] == "all")]
        if len(m):
            top1 = float(m.iloc[0]["top1"])

    delay = metrics["median_delay_days"]
    cards = [
        ("Leaks detected",
         f"{metrics['detected']}<small>/ {metrics['n_leaks']}</small>",
         f"<b>{metrics['detection_rate']:.0%}</b> vs {mnf_rate:.0%} for night-flow method"
         if mnf_rate is not None else f"{metrics['detection_rate']:.0%} detection rate",
         True),
        ("False alarms", f"{metrics['false_alarms']}",
         "across the entire year", False),
        ("Median time to detect",
         f"{delay:.0f}<small>days</small>" if np.isfinite(delay) else "–",
         f"<b>{mnf_delay / delay:.1f}× faster</b> than current practice"
         if mnf_delay and np.isfinite(delay) and delay > 0 else "from leak start to alarm",
         False),
        ("Correct district, 1st guess",
         f"{top1:.0%}" if top1 is not None else "–",
         f"<b>{top1 / (1 / N.N_ZONES):.1f}×</b> better than random (8%)"
         if top1 is not None else "", False),
    ]
    for col, (label, value, cmp, accent) in zip(st.columns(4), cards):
        col.markdown(
            f'<div class="kpi{" accent" if accent else ""}"><div class="label">{label}</div>'
            f'<div class="value">{value}</div><div class="cmp">{cmp}</div></div>',
            unsafe_allow_html=True,
        )

    html('<div class="section-title">How it works</div>'
         '<div class="section-sub">Four stages, from raw sensor readings to a '
         'district on the map</div>')
    steps = [
        ("Predict", "A regression learns what each of the 33 sensors <i>should</i> read "
                    "from inflow, tank level and time of day, refitted daily on the last "
                    "21 days."),
        ("Compare", "Residual = actual − predicted. Normally ~2 cm. A new leak drops "
                    "nearby pressure and the residual jumps (31 cm for leak p523)."),
        ("Detect", "CUSUM accumulates small daily deviations like a running balance, "
                   "so even slowly-growing cracks eventually cross the alarm line."),
        ("Locate", "The 33-sensor pressure fingerprint is matched against 902 leaks "
                   "simulated in EPANET, ranking the districts most likely to hold it."),
    ]
    for i, (col, (title, text)) in enumerate(zip(st.columns(4), steps), 1):
        col.markdown(f'<div class="step"><div class="num">{i}</div><h4>{title}</h4>'
                     f'<p>{text}</p></div>', unsafe_allow_html=True)

    html('<div class="section-title">The network</div>'
         f'<div class="section-sub">{len(ptable)} pipes grouped into {n_zones} search '
         'districts · ▲ pressure sensors · ✕ leaks that started this year</div>')

    fig = go.Figure()
    for z in sorted(zones_df["zone"].unique()):
        sub = zones_df[zones_df["zone"] == z]
        xs, ys = segments(sub, coords)
        fig.add_trace(go.Scatter(
            x=xs, y=ys, mode="lines", name=f"District {z}",
            line=dict(color=ZONE_COLORS[int(z) % len(ZONE_COLORS)], width=2.2),
            hoverinfo="skip",
        ))
        fig.add_annotation(x=sub["x"].mean(), y=sub["y"].mean(), text=f"<b>{z}</b>",
                           showarrow=False, font=dict(size=12, color="#fff"),
                           bgcolor="rgba(11,19,32,.75)",
                           bordercolor="rgba(255,255,255,.25)",
                           borderwidth=1, borderpad=3)
    fig.add_trace(go.Scatter(
        x=sensors["x"], y=sensors["y"], mode="markers", name="Pressure sensor",
        marker=dict(size=11, color="#ffffff", symbol="triangle-up",
                    line=dict(width=1.5, color="#0b1320")),
        hovertext=[f"Sensor {s}" for s in sensors.index], hoverinfo="text",
    ))
    detected = {m_["pipe"] for m_ in metrics["matches"]}
    for label, members, color in [
        ("Leak · detected", [x for x in leaks if x.pipe in detected], GREEN),
        ("Leak · missed", [x for x in leaks if x.pipe not in detected], RED),
    ]:
        if not members:
            continue
        pts = ptable.loc[[x.pipe for x in members]]
        fig.add_trace(go.Scatter(
            x=pts["x"], y=pts["y"], mode="markers", name=label,
            marker=dict(size=17, color=color, symbol="x",
                        line=dict(width=1.5, color="#0b1320")),
            hovertext=[f"<b>{x.pipe}</b> · {label.split('· ')[1]}<br>"
                       f"{x.start:%d %b %Y} · {x.kind}<br>"
                       f"{x.diameter_m * 1000:.1f} mm hole" for x in members],
            hoverinfo="text",
        ))
    style_fig(fig, 560)
    map_axes(fig)
    fig.update_layout(legend=dict(orientation="h", y=-0.02, x=0, font=dict(size=10)))
    st.plotly_chart(fig, width="stretch", config=PLOT_CFG)

# --------------------------------------------------------------------------
# Year replay
# --------------------------------------------------------------------------
with tab_replay:
    html('<div class="section-title">Replay the year as the control room saw it</div>'
         '<div class="section-sub">Drag the date. Leaks appear when they start, turn '
         'green once the detector has flagged them, and alarms accumulate on the '
         'timeline.</div>')

    days = list(trace.index)
    default_day = min(days, key=lambda d: abs(d - pd.Timestamp(f"{year}-07-01")))
    day = st.select_slider("Date", options=days, value=default_day,
                           format_func=lambda d: d.strftime("%d %b %Y"),
                           label_visibility="collapsed")

    alarm_day_by_pipe = {m_["pipe"]: pd.Timestamp(m_["alarm_day"])
                         for m_ in metrics["matches"]}
    active = [lk for lk in C.LEAKS
              if pd.Timestamp(lk.start) <= day <= pd.Timestamp(lk.end)]
    new_this_year = [lk for lk in active if lk.year == year]
    flagged = [lk for lk in new_this_year
               if lk.pipe in alarm_day_by_pipe and alarm_day_by_pipe[lk.pipe] <= day]
    fired = alarms[alarms["day"] <= day] if len(alarms) else alarms

    replay_cards = [
        ("Date", day.strftime("%d %b"), day.strftime("%A")),
        ("Leaks running", str(len(active)), f"{len(new_this_year)} began in {year}"),
        ("Alarms raised so far", str(len(fired)), "cumulative this year"),
        ("New leaks flagged", f"{len(flagged)}<small>/ {len(new_this_year)}</small>",
         "of those currently running"),
    ]
    for col, (label, value, cmp) in zip(st.columns(4), replay_cards):
        col.markdown(f'<div class="kpi"><div class="label">{label}</div>'
                     f'<div class="value">{value}</div><div class="cmp">{cmp}</div></div>',
                     unsafe_allow_html=True)

    left, right = st.columns([1.15, 1])
    with left:
        fig = go.Figure()
        xs, ys = segments(ptable, coords)
        fig.add_trace(go.Scatter(x=xs, y=ys, mode="lines", name="Pipes",
                                 line=dict(color="rgba(120,160,200,.35)", width=1.6),
                                 hoverinfo="skip", showlegend=False))
        fig.add_trace(go.Scatter(
            x=sensors["x"], y=sensors["y"], mode="markers", name="Sensor",
            marker=dict(size=8, color="rgba(255,255,255,.75)", symbol="triangle-up"),
            hoverinfo="skip",
        ))
        for name, items, color, size in [
            ("Older leak (already running)", [lk for lk in active if lk.year != year],
             "rgba(148,163,184,.9)", 12),
            ("New leak · not yet flagged",
             [lk for lk in new_this_year if lk not in flagged], RED, 20),
            ("New leak · flagged", flagged, GREEN, 20),
        ]:
            if not items:
                continue
            pts = ptable.loc[[lk.pipe for lk in items]]
            fig.add_trace(go.Scatter(
                x=pts["x"], y=pts["y"], mode="markers", name=name,
                marker=dict(size=size, color=color, symbol="circle",
                            line=dict(width=2, color="#0b1320"), opacity=.95),
                hovertext=[f"<b>{lk.pipe}</b><br>started {lk.start:%d %b %Y}<br>{lk.kind}"
                           for lk in items], hoverinfo="text",
            ))
        style_fig(fig, 470)
        map_axes(fig)
        fig.update_layout(legend=dict(orientation="h", y=-0.02, x=0))
        st.plotly_chart(fig, width="stretch", config=PLOT_CFG)

    with right:
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=trace.index, y=trace.to_numpy(), mode="lines",
                                 name="Evidence (CUSUM)", fill="tozeroy",
                                 line=dict(color=CYAN, width=2),
                                 fillcolor="rgba(34,184,207,.12)"))
        fig.add_hline(y=float(h), line_dash="dash", line_color=AMBER,
                      annotation_text="alarm line", annotation_font_color=AMBER)
        for lk in leaks:
            fig.add_vline(x=pd.Timestamp(lk.start.date()),
                          line_color="rgba(52,211,153,.35)", line_width=1)
        if len(alarms):
            fig.add_trace(go.Scatter(
                x=alarms["day"], y=[trace.loc[d] for d in alarms["day"]], mode="markers",
                name="Alarm", marker=dict(size=12, color=RED, symbol="triangle-down",
                                          line=dict(width=1, color="#0b1320")),
                hovertext=[f"Alarm {d:%d %b} · sensor {s}"
                           for d, s in zip(alarms["day"], alarms["sensor"])],
                hoverinfo="text",
            ))
        fig.add_vrect(x0=day, x1=trace.index[-1], fillcolor="rgba(11,19,32,.65)",
                      line_width=0, layer="above")
        fig.add_vline(x=day, line_color="#ffffff", line_width=2)
        style_fig(fig, 470)
        fig.update_layout(legend=dict(orientation="h", y=1.08, x=0),
                          yaxis_title="accumulated evidence")
        st.plotly_chart(fig, width="stretch", config=PLOT_CFG)
    st.caption("Green vertical lines mark true leak onsets. The shaded region is the "
               "future relative to the selected date.")

    with st.expander("Residual heatmap — every sensor, every day"):
        Z = res.normalised(R).clip(-6, 6)
        heat = go.Figure(go.Heatmap(z=Z.T.to_numpy(), x=Z.index, y=Z.columns,
                                    colorscale="RdBu_r", zmid=0,
                                    colorbar=dict(title=dict(text="z"))))
        style_fig(heat, 560, legend=False)
        st.plotly_chart(heat, width="stretch", config=PLOT_CFG)

# --------------------------------------------------------------------------
# Leak explorer
# --------------------------------------------------------------------------
with tab_explore:
    html('<div class="section-title">Where would the crew be sent?</div>'
         '<div class="section-sub">Choose a real leak. Its pressure fingerprint is '
         'matched against 902 EPANET-simulated leaks to rank the districts.</div>')
    default_idx = next((i for i, x in enumerate(leaks) if x.pipe == "p523"), 0)
    choice = st.selectbox(
        "Leak event", options=list(range(len(leaks))), index=default_idx,
        format_func=lambda i: (f"{leaks[i].pipe}  ·  {leaks[i].start:%d %b %Y}  ·  "
                               f"{leaks[i].kind}  ·  {leaks[i].diameter_m * 1000:.1f} mm"),
        label_visibility="collapsed",
    )
    lk = leaks[choice]
    obs = res.adaptive_signature(R, pd.Timestamp(lk.start.date()))

    if obs.isna().any():
        st.warning("Not enough residual history around this date to build a fingerprint.")
    else:
        loc = L.localise(obs, n_zones=int(n_zones))
        true_zone = int(zones_df.at[lk.pipe, "zone"])
        order = [int(z) for z in loc.zone_scores.index]
        rank = order.index(true_zone) + 1
        dist = float(np.hypot(ptable.at[loc.pipe_pred, "x"] - ptable.at[lk.pipe, "x"],
                              ptable.at[loc.pipe_pred, "y"] - ptable.at[lk.pipe, "y"]))
        total_km = float(zstats["length_km"].sum())
        search_km = float(zstats.loc[order[:rank], "length_km"].sum())

        if rank == 1:
            html(f'<div class="verdict ok"><span class="icon">🎯</span><div>'
                 f'<b>Correct on the first guess.</b> Crew sent to district {loc.zone_pred} '
                 f'— {search_km:.1f} km of pipe instead of {total_km:.0f} km. Best-matching '
                 f'pipe is <b>{dist:.0f} m</b> from the real leak.</div></div>')
        else:
            html(f'<div class="verdict bad"><span class="icon">🧭</span><div>'
                 f'<b>True district ranked {rank} of {n_zones}.</b> Searching the top '
                 f'{rank} districts covers {search_km:.1f} km of {total_km:.0f} km.'
                 f'</div></div>')

        explorer_cards = [
            ("Predicted district", str(loc.zone_pred), f"true district: {true_zone}"),
            ("Rank of true district", f"{rank}<small>/ {n_zones}</small>",
             "1 = first place searched"),
            ("Search area", f"{search_km:.1f}<small>km</small>",
             f"{100 * (1 - search_km / total_km):.0f}% less than the whole network"),
            ("Leak size", f"{lk.diameter_m * 1000:.1f}<small>mm</small>",
             f"{lk.kind} · started {lk.start:%d %b}"),
        ]
        for col, (label, value, cmp) in zip(st.columns(4), explorer_cards):
            col.markdown(f'<div class="kpi"><div class="label">{label}</div>'
                         f'<div class="value">{value}</div>'
                         f'<div class="cmp">{cmp}</div></div>',
                         unsafe_allow_html=True)

        left, right = st.columns([1.35, 1])
        with left:
            scores = loc.zone_scores
            lo, hi = float(scores.min()), float(scores.max())
            fig = go.Figure()
            for z in sorted(zones_df["zone"].unique()):
                sub = zones_df[zones_df["zone"] == z]
                t = (float(scores.get(z, lo)) - lo) / (hi - lo) if hi > lo else 0.0
                colr = sample_colorscale("Turbo", [0.08 + 0.87 * t])[0]
                xs, ys = segments(sub, coords)
                fig.add_trace(go.Scatter(
                    x=xs, y=ys, mode="lines", showlegend=False, hoverinfo="skip",
                    line=dict(color=colr, width=1.4 + 2.6 * t),
                ))
                fig.add_annotation(x=sub["x"].mean(), y=sub["y"].mean(),
                                   text=f"<b>{z}</b>", showarrow=False,
                                   font=dict(size=11, color="#fff"),
                                   bgcolor="rgba(11,19,32,.7)", borderpad=2)
            fig.add_trace(go.Scatter(
                x=sensors["x"], y=sensors["y"], mode="markers", name="Sensor",
                marker=dict(size=8, color="rgba(255,255,255,.8)", symbol="triangle-up"),
                hoverinfo="skip",
            ))
            fig.add_trace(go.Scatter(
                x=[ptable.at[lk.pipe, "x"]], y=[ptable.at[lk.pipe, "y"]],
                mode="markers", name=f"Real leak ({lk.pipe})",
                marker=dict(size=24, color=GREEN, symbol="star",
                            line=dict(width=2, color="#0b1320")),
            ))
            fig.add_trace(go.Scatter(
                x=[ptable.at[loc.pipe_pred, "x"]], y=[ptable.at[loc.pipe_pred, "y"]],
                mode="markers", name=f"Best match ({loc.pipe_pred})",
                marker=dict(size=26, color="rgba(0,0,0,0)", symbol="circle",
                            line=dict(width=3, color="#ffffff")),
            ))
            # Invisible trace that exists only to draw the colour bar.
            fig.add_trace(go.Scatter(
                x=[None], y=[None], mode="markers", showlegend=False,
                marker=dict(colorscale="Turbo", cmin=lo, cmax=hi, color=[lo],
                            showscale=True,
                            colorbar=dict(title=dict(text="match"), thickness=12,
                                          len=.7)),
            ))
            style_fig(fig, 500)
            map_axes(fig)
            fig.update_layout(legend=dict(orientation="h", y=-0.02, x=0))
            st.plotly_chart(fig, width="stretch", config=PLOT_CFG)

        with right:
            top = loc.zone_scores.head(6)[::-1]
            bar = go.Figure(go.Bar(
                x=top.to_numpy(), y=[f"District {i}" for i in top.index],
                orientation="h",
                marker=dict(color=[GREEN if int(i) == true_zone else "rgba(139,155,180,.45)"
                                   for i in top.index]),
                text=[f"{v:.2f}" for v in top.to_numpy()], textposition="outside",
            ))
            style_fig(bar, 290, legend=False)
            # Headroom so the labels outside the bars are not clipped.
            span = max(float(top.max()), 0.05)
            bar.update_traces(cliponaxis=False)
            bar.update_xaxes(range=[min(0.0, float(top.min())) * 1.1, span * 1.28])
            bar.update_layout(xaxis_title="similarity to simulated fingerprint",
                              title=dict(text="Ranked districts (green = correct)",
                                         font=dict(size=13), x=0),
                              margin=dict(t=40))
            st.plotly_chart(bar, width="stretch", config=PLOT_CFG)

            cand = loc.pipe_scores.head(6).round(3).rename("similarity").to_frame()
            cand["district"] = zones_df["zone"].reindex(cand.index).astype(int)
            cand["distance to leak (m)"] = [
                round(float(np.hypot(ptable.at[p, "x"] - ptable.at[lk.pipe, "x"],
                                     ptable.at[p, "y"] - ptable.at[lk.pipe, "y"])))
                for p in cand.index]
            st.dataframe(cand, width="stretch", height=250)

        import simulate as sim
        obs_u = obs / np.linalg.norm(obs.to_numpy())
        simd = sim.unit_signatures().loc[lk.pipe].reindex(obs.index)
        cos = float(np.dot(obs_u.to_numpy(), simd.to_numpy()))
        html(f'<div class="section-title">Pressure fingerprint</div>'
             f'<div class="section-sub">What the 33 real sensors did, against what physics '
             f'predicts for a leak on {lk.pipe} · cosine similarity '
             f'<b style="color:#22b8cf">{cos:.2f}</b></div>')
        cmp_fig = go.Figure()
        cmp_fig.add_trace(go.Bar(x=obs.index, y=obs_u.to_numpy(),
                                 name="Observed (real SCADA)", marker_color=CYAN))
        cmp_fig.add_trace(go.Bar(x=obs.index, y=simd.to_numpy(),
                                 name="Simulated (EPANET)", marker_color=AMBER))
        style_fig(cmp_fig, 330)
        cmp_fig.update_layout(barmode="group", bargap=.25,
                              yaxis_title="normalised response",
                              legend=dict(orientation="h", y=1.1, x=0))
        st.plotly_chart(cmp_fig, width="stretch", config=PLOT_CFG)

# --------------------------------------------------------------------------
# Evidence & limits
# --------------------------------------------------------------------------
with tab_evidence:
    html('<div class="section-title">How well does it actually work?</div>'
         '<div class="section-sub">All numbers below come from the held-out protocol: '
         'every setting chosen on 2018, then 2019 evaluated once.</div>')

    det = read_table("detection_summary.csv")
    sa = read_table("sensor_ablation.csv")
    c1, c2 = st.columns(2)
    if det is not None:
        fig = go.Figure()
        fig.add_trace(go.Bar(x=det["year"].astype(str), y=det["mnf_detection_rate"] * 100,
                             name="Night-flow method (industry)",
                             marker_color="rgba(139,155,180,.55)",
                             text=[f"{v:.0%}" for v in det["mnf_detection_rate"]],
                             textposition="outside"))
        fig.add_trace(go.Bar(x=det["year"].astype(str), y=det["detection_rate"] * 100,
                             name="AquaSentinel", marker_color=CYAN,
                             text=[f"{v:.0%}" for v in det["detection_rate"]],
                             textposition="outside"))
        style_fig(fig, 330)
        fig.update_layout(barmode="group",
                          yaxis=dict(title="leaks detected (%)", range=[0, 115]),
                          legend=dict(orientation="h", y=1.12, x=0))
        with c1:
            html('<div class="section-sub"><b style="color:#fff">Detection rate vs the '
                 'method in use today</b></div>')
            st.plotly_chart(fig, width="stretch", config=PLOT_CFG)
    if sa is not None:
        f2 = go.Figure()
        f2.add_trace(go.Bar(x=sa["n_sensors"], y=sa["false_alarms"],
                            name="False alarms / year", yaxis="y2",
                            marker_color="rgba(248,113,113,.35)"))
        f2.add_trace(go.Scatter(x=sa["n_sensors"], y=sa["detection_rate"] * 100,
                                name="Detection %", mode="lines+markers",
                                line=dict(color=CYAN, width=3)))
        f2.add_trace(go.Scatter(x=sa["n_sensors"], y=sa["localisation_top1"] * 100,
                                name="Right district %", mode="lines+markers",
                                line=dict(color=GREEN, width=3)))
        style_fig(f2, 330)
        f2.update_layout(
            xaxis_title="number of pressure sensors",
            yaxis=dict(title="%", range=[0, 110]),
            # Explicit ticks: otherwise Plotly syncs this axis to the left one
            # and labels it with values like 5.45 and 4.36.
            yaxis2=dict(title="false alarms / year", overlaying="y", side="right",
                        showgrid=False, range=[0, 6], tickmode="linear", tick0=0,
                        dtick=1),
            legend=dict(orientation="h", y=1.16, x=0),
            margin=dict(t=40),
        )
        with c2:
            html('<div class="section-sub"><b style="color:#fff">More sensors buy '
                 'precision and location, not raw recall</b></div>')
            st.plotly_chart(f2, width="stretch", config=PLOT_CFG)

    for title, name, caption in [
        ("Detection summary", "detection_summary.csv",
         "AquaSentinel against minimum night flow, the method utilities use today."),
        ("Localisation summary", "localisation_summary.csv",
         "District accuracy, split by leak type. Random guessing scores 1/12 = 8%."),
        ("Sensor count ablation", "sensor_ablation.csv",
         "Detection rate alone is gameable by lowering the threshold, so false "
         "alarms are shown alongside it."),
        ("District granularity ablation", "zone_ablation.csv",
         "Fewer districts are easier to hit but send a crew over more pipe."),
    ]:
        df = read_table(name)
        if df is not None:
            with st.expander(title):
                st.dataframe(df, width="stretch", hide_index=True)
                st.caption(caption)

    html('<div class="section-title">Known limitations</div>')
    lims = [
        ("🐢", "Slow cracks are the hard case",
         "Incipient leaks ramp up over weeks, so detection is slower and localisation "
         "less accurate than for sudden bursts."),
        ("🗺️", "District, not pipe",
         "33 sensors over 905 pipes cannot distinguish neighbouring pipes, a "
         "mathematical limit rather than a coding one."),
        ("📐", "Model error in simulation",
         "The published network model differs from the true one by up to 10%, "
         "capping how well fingerprints can match."),
        ("🔢", "Small test set",
         "19 held-out leaks: one leak ≈ 5 percentage points. Two were missed, both "
         "late in the year amid many concurrent leaks."),
    ]
    for col, (icon, title, text) in zip(st.columns(4), lims):
        col.markdown(f'<div class="step"><div style="font-size:1.4rem">{icon}</div>'
                     f'<h4>{title}</h4><p>{text}</p></div>', unsafe_allow_html=True)
