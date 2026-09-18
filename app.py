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
import streamlit as st

import config as C
import data_loader as dl
import detect as D
import localize as L
import network as N
import residuals as res

st.set_page_config(page_title="AquaSentinel", page_icon="💧", layout="wide")


# --------------------------------------------------------------------------
# Cached heavy computations
# --------------------------------------------------------------------------
@st.cache_data(show_spinner="Computing pressure residuals...")
def get_residuals(year: int) -> pd.DataFrame:
    return res.daily_residuals(year)


@st.cache_data(show_spinner="Tuning detector on 2018...")
def get_params() -> tuple[float, float, int]:
    p = D.tune(C.TRAIN_YEAR)
    return p.k, p.h, p.refractory_days


@st.cache_data(show_spinner="Running detector...")
def get_detection(year: int, k: float, h: float, refractory: int):
    Z = res.normalised(get_residuals(year))
    params = D.DetectorParams(k=k, h=h, refractory_days=refractory)
    alarms, trace = D.run_detector(Z, params)
    metrics = D.evaluate(alarms, year)
    return (
        pd.DataFrame([{"day": a.day, "sensor": a.sensor, "score": a.score}
                      for a in alarms]),
        trace,
        metrics,
    )


@st.cache_data(show_spinner="Loading network...")
def get_network(n_zones: int):
    return N.zone_assignment(n_zones), N.sensor_coords(), N.zone_stats(n_zones)


# --------------------------------------------------------------------------
# Sidebar
# --------------------------------------------------------------------------
st.sidebar.title("💧 AquaSentinel")
st.sidebar.caption("Leak detection & localisation on the L-Town network "
                   "(BattLeDIM 2020 benchmark)")

year = st.sidebar.selectbox(
    "Year", [C.TEST_YEAR, C.TRAIN_YEAR],
    format_func=lambda y: f"{y} — {'held-out test' if y == C.TEST_YEAR else 'development'}",
)
n_zones = st.sidebar.slider("Number of zones", 6, 20, N.N_ZONES, step=2)

k_def, h_def, r_def = get_params()
st.sidebar.markdown("**Detector settings**")
st.sidebar.caption(f"Tuned on {C.TRAIN_YEAR}: k={k_def}, h={h_def}")
k = st.sidebar.slider("CUSUM slack k", 0.25, 2.0, float(k_def), 0.25)
h = st.sidebar.slider("Alarm threshold h", 2.0, 20.0, float(h_def), 1.0)
refractory = st.sidebar.slider("Refractory period (days)", 5, 30, int(r_def))

alarms, trace, metrics = get_detection(year, k, h, refractory)
zones_df, sensors, zstats = get_network(n_zones)
R = get_residuals(year)
leaks = sorted(C.leaks_for(year), key=lambda x: x.start)

tab_overview, tab_monitor, tab_explore, tab_evidence = st.tabs(
    ["Overview", "Live monitor", "Leak explorer", "Evidence & limits"]
)


# --------------------------------------------------------------------------
# Overview
# --------------------------------------------------------------------------
with tab_overview:
    st.header(f"{year} results")
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Leaks detected",
              f"{metrics['detected']}/{metrics['n_leaks']}",
              f"{metrics['detection_rate']:.0%}")
    c2.metric("False alarms", metrics["false_alarms"])
    c3.metric("Median time to detection", f"{metrics['median_delay_days']:.0f} d")
    c4.metric("Water lost before alarm (est.)",
              f"{metrics['median_delay_days'] * 24 * 14.2 / 1000:.0f} ML",
              help="Median delay x a typical 14.2 m3/h leak. ML = megalitres.")

    st.subheader("Network")
    fig = go.Figure()
    for z in sorted(zones_df["zone"].unique()):
        sub = zones_df[zones_df["zone"] == z]
        fig.add_trace(go.Scatter(
            x=sub["x"], y=sub["y"], mode="markers", name=f"Zone {z}",
            marker=dict(size=5), hovertext=sub.index, hoverinfo="text+name",
        ))
    fig.add_trace(go.Scatter(
        x=sensors["x"], y=sensors["y"], mode="markers", name="Pressure sensors",
        marker=dict(size=11, color="black", symbol="triangle-up"),
        hovertext=sensors.index, hoverinfo="text",
    ))
    pt = N.pipe_table()
    lk = pt.loc[[x.pipe for x in leaks]]
    fig.add_trace(go.Scatter(
        x=lk["x"], y=lk["y"], mode="markers", name="True leaks",
        marker=dict(size=13, color="red", symbol="x"),
        hovertext=[f"{x.pipe} — {x.start:%d %b} ({x.kind})" for x in leaks],
        hoverinfo="text",
    ))
    fig.update_layout(height=520, yaxis=dict(scaleanchor="x"),
                      margin=dict(l=0, r=0, t=10, b=0))
    st.plotly_chart(fig, width="stretch")

    st.subheader("Zone sizes")
    st.dataframe(zstats.round(2), width="stretch")


# --------------------------------------------------------------------------
# Live monitor
# --------------------------------------------------------------------------
with tab_monitor:
    st.header("Detector output over the year")

    fig = go.Figure()
    fig.add_trace(go.Scatter(x=trace.index, y=trace.to_numpy(), name="CUSUM statistic",
                             line=dict(color="#1f4e79", width=1.5)))
    fig.add_hline(y=h, line_dash="dash", line_color="crimson",
                  annotation_text=f"threshold h={h}")
    for i, x in enumerate(leaks):
        fig.add_vline(x=pd.Timestamp(x.start.date()), line_color="green",
                      line_width=1, opacity=0.45)
    if len(alarms):
        fig.add_trace(go.Scatter(
            x=alarms["day"], y=[trace.loc[d] for d in alarms["day"]],
            mode="markers", name="Alarm",
            marker=dict(size=13, color="crimson", symbol="triangle-down"),
            hovertext=[f"{d:%d %b} — {s}" for d, s in
                       zip(alarms["day"], alarms["sensor"])],
            hoverinfo="text",
        ))
    fig.update_layout(height=380, margin=dict(l=0, r=0, t=10, b=0),
                      yaxis_title="CUSUM statistic")
    st.plotly_chart(fig, width="stretch")
    st.caption("Green lines are true leak onsets; red triangles are alarms our "
               "detector raised. Nothing about 2019 was used to set the thresholds.")

    st.subheader("Residual z-scores by sensor")
    Z = res.normalised(R).clip(-6, 6)
    heat = go.Figure(go.Heatmap(
        z=Z.T.to_numpy(), x=Z.index, y=Z.columns, colorscale="RdBu", zmid=0,
        colorbar=dict(title="z"),
    ))
    heat.update_layout(height=520, margin=dict(l=0, r=0, t=10, b=0))
    st.plotly_chart(heat, width="stretch")


# --------------------------------------------------------------------------
# Leak explorer
# --------------------------------------------------------------------------
with tab_explore:
    st.header("Localise an individual leak")
    choice = st.selectbox(
        "Leak event",
        options=list(range(len(leaks))),
        format_func=lambda i: (f"{leaks[i].pipe} — {leaks[i].start:%d %b %Y} — "
                               f"{leaks[i].kind}, {leaks[i].diameter_m * 1000:.1f} mm"),
    )
    lk = leaks[choice]
    obs = res.adaptive_signature(R, pd.Timestamp(lk.start.date()))

    if obs.isna().any():
        st.warning("Not enough residual history around this date to build a fingerprint.")
    else:
        loc = L.localise(obs, n_zones=n_zones)
        true_zone = int(zones_df.at[lk.pipe, "zone"])
        rank = list(loc.zone_scores.index).index(true_zone) + 1

        c1, c2, c3 = st.columns(3)
        c1.metric("Predicted zone", loc.zone_pred,
                  "correct" if loc.zone_pred == true_zone else f"true = {true_zone}")
        c2.metric("True zone rank", f"{rank} of {n_zones}")
        c3.metric("Search area",
                  f"{zstats.at[loc.zone_pred, 'length_km']:.1f} km of pipe")

        left, right = st.columns([3, 2])
        with left:
            fig = go.Figure()
            fig.add_trace(go.Scatter(
                x=zones_df["x"], y=zones_df["y"], mode="markers",
                marker=dict(size=6,
                            color=loc.zone_scores.reindex(zones_df["zone"]).to_numpy(),
                            colorscale="YlOrRd", showscale=True,
                            colorbar=dict(title="zone<br>score")),
                hovertext=zones_df.index, hoverinfo="text", name="pipes",
            ))
            fig.add_trace(go.Scatter(
                x=sensors["x"], y=sensors["y"], mode="markers", name="sensors",
                marker=dict(size=9, color="black", symbol="triangle-up"),
            ))
            fig.add_trace(go.Scatter(
                x=[pt.at[lk.pipe, "x"]], y=[pt.at[lk.pipe, "y"]], mode="markers",
                name=f"true leak {lk.pipe}",
                marker=dict(size=18, color="lime", symbol="x",
                            line=dict(width=2, color="black")),
            ))
            fig.update_layout(height=460, yaxis=dict(scaleanchor="x"),
                              margin=dict(l=0, r=0, t=10, b=0))
            st.plotly_chart(fig, width="stretch")

        with right:
            top = loc.zone_scores.head(6)
            bar = go.Figure(go.Bar(
                x=top.to_numpy()[::-1], y=[f"Zone {i}" for i in top.index][::-1],
                orientation="h",
                marker_color=["#2e7d32" if i == true_zone else "#9e9e9e"
                              for i in top.index][::-1],
            ))
            bar.update_layout(height=300, margin=dict(l=0, r=0, t=10, b=0),
                              xaxis_title="similarity to simulated fingerprint")
            st.plotly_chart(bar, width="stretch")

            st.markdown("**Top candidate pipes**")
            st.dataframe(
                loc.pipe_scores.head(8).round(3).rename("similarity").to_frame(),
                width="stretch",
            )

        st.subheader("Pressure fingerprint")
        import simulate as sim
        obs_u = obs / np.linalg.norm(obs.to_numpy())
        simd = sim.unit_signatures().loc[lk.pipe].reindex(obs.index)
        cmp = go.Figure()
        cmp.add_trace(go.Bar(x=obs.index, y=obs_u.to_numpy(),
                             name="observed (real SCADA)", marker_color="#1f4e79"))
        cmp.add_trace(go.Bar(x=obs.index, y=simd.to_numpy(),
                             name="simulated (EPANET)", marker_color="#e08214"))
        cmp.update_layout(height=320, barmode="group",
                          margin=dict(l=0, r=0, t=10, b=0),
                          yaxis_title="normalised response")
        st.plotly_chart(cmp, width="stretch")


# --------------------------------------------------------------------------
# Evidence & limits
# --------------------------------------------------------------------------
with tab_evidence:
    st.header("How well does this actually work?")

    tables = C.TABLES
    for title, name, caption in [
        ("Detection summary", "detection_summary.csv",
         "Our detector vs minimum night flow, the method utilities use today."),
        ("Localisation summary", "localisation_summary.csv",
         "Zone accuracy, split by leak type. Random guessing scores 1/12 = 8%."),
        ("Sensor count ablation", "sensor_ablation.csv",
         "Detection rate alone is gameable by lowering the threshold, so false "
         "alarms are shown alongside it."),
        ("Zone granularity ablation", "zone_ablation.csv",
         "Fewer zones are easier to hit but send a crew over more pipe."),
    ]:
        path = tables / name
        if path.exists():
            st.subheader(title)
            st.dataframe(pd.read_csv(path), width="stretch")
            st.caption(caption)

    st.subheader("Known limitations")
    st.markdown(
        """
- **Incipient leaks are the hard case.** They ramp up over weeks, so both
  detection latency and localisation accuracy are markedly worse than for
  abrupt bursts.
- **Localisation is zone-level, not pipe-level.** With 33 sensors covering 902
  pipes, pinpointing an individual pipe is not identifiable from pressure alone.
- **Simulated fingerprints carry model error.** The published L-Town model has
  up to 10% parameter error against the network that generated the data, which
  puts a ceiling on match quality.
- **One network, two years.** Results are not proof the method transfers to a
  different topology without re-tuning.
"""
    )
