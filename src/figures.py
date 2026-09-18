"""Generate every figure used in the report, into reports/figures/."""
from __future__ import annotations

import warnings

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

import config as C
import data_loader as dl
import detect as D
import localize as L
import network as N
import residuals as res

warnings.filterwarnings("ignore")

plt.rcParams.update({
    "figure.dpi": 140,
    "savefig.dpi": 160,
    "savefig.bbox": "tight",
    "font.size": 9,
    "axes.grid": True,
    "grid.alpha": 0.25,
    "axes.spines.top": False,
    "axes.spines.right": False,
})

CMAP_ZONE = "tab20"
BLUE, ORANGE, GREEN, GREY = "#1f4e79", "#e08214", "#2e7d32", "#9e9e9e"


def _save(fig, name: str) -> None:
    path = C.FIGURES / f"{name}.png"
    fig.savefig(path)
    plt.close(fig)
    print(f"  {path.name}")


def fig_network_map() -> None:
    """The L-Town network: zones, pressure sensors, and the 33 real leaks."""
    z = N.zone_assignment()
    sc = N.sensor_coords()
    pt = N.pipe_table()

    fig, ax = plt.subplots(figsize=(9, 5.5))
    ax.scatter(z["x"], z["y"], c=z["zone"], cmap=CMAP_ZONE, s=7, alpha=0.75,
               linewidths=0)
    for zone, row in N.zone_centroids().iterrows():
        ax.text(row["x"], row["y"], str(zone), fontsize=11, fontweight="bold",
                ha="center", va="center",
                bbox=dict(boxstyle="circle,pad=0.22", fc="white", ec="0.3", lw=0.8))

    ax.scatter(sc["x"], sc["y"], marker="^", s=70, c="black",
               label=f"pressure sensors (n={len(sc)})", zorder=5)
    lk = pt.loc[[l.pipe for l in C.LEAKS]]
    ax.scatter(lk["x"], lk["y"], marker="X", s=90, c="red", edgecolors="white",
               linewidths=0.8, label=f"leaks 2018-19 (n={len(lk)})", zorder=6)

    ax.set_title("L-Town: 902 pipes in 12 zones, 33 pressure sensors, 33 leaks")
    ax.set_xlabel("x (m)")
    ax.set_ylabel("y (m)")
    ax.set_aspect("equal")
    ax.legend(loc="upper left", framealpha=0.9)
    _save(fig, "01_network_map")


def fig_residual_heatmap(year: int = C.TEST_YEAR) -> None:
    """Residual z-scores for all sensors across the test year."""
    Z = res.normalised(res.daily_residuals(year)).clip(-6, 6)
    fig, ax = plt.subplots(figsize=(11, 5))
    im = ax.imshow(Z.T.to_numpy(), aspect="auto", cmap="RdBu_r", vmin=-6, vmax=6,
                   extent=[0, len(Z), len(Z.columns), 0], interpolation="nearest")
    ax.set_yticks(np.arange(len(Z.columns)) + 0.5)
    ax.set_yticklabels(Z.columns, fontsize=5)
    for lk in C.leaks_for(year):
        pos = Z.index.get_indexer([pd.Timestamp(lk.start.date())], method="nearest")[0]
        ax.axvline(pos, color="black", lw=0.9, ls="--", alpha=0.75)
    ticks = np.linspace(0, len(Z) - 1, 12).astype(int)
    ax.set_xticks(ticks)
    ax.set_xticklabels([Z.index[t].strftime("%b") for t in ticks])
    ax.set_title(f"{year} pressure residual z-scores (dashed = true leak onset)")
    ax.set_ylabel("pressure sensor")
    ax.grid(False)
    fig.colorbar(im, ax=ax, label="residual z-score", pad=0.01)
    _save(fig, "02_residual_heatmap")


def fig_cusum_trace(year: int = C.TEST_YEAR) -> None:
    """Detector statistic against the ground truth."""
    params = D.tune(C.TRAIN_YEAR)
    Z = res.normalised(res.daily_residuals(year))
    alarms, trace = D.run_detector(Z, params)
    m = D.evaluate(alarms, year)

    fig, ax = plt.subplots(figsize=(11, 4))
    ax.plot(trace.index, trace.to_numpy(), lw=1.0, color=BLUE,
            label="CUSUM statistic")
    ax.axhline(params.h, color="crimson", ls="--", lw=1.2,
               label=f"alarm threshold h={params.h}")
    for i, lk in enumerate(C.leaks_for(year)):
        ax.axvline(pd.Timestamp(lk.start.date()), color=GREEN, lw=0.9, alpha=0.55,
                   label="true leak onset" if i == 0 else None)
    for i, a in enumerate(alarms):
        ax.plot(a.day, trace.loc[a.day], marker="v", ms=8, color="crimson",
                label="alarm raised" if i == 0 else None)

    ax.set_title(f"{year} held-out: {m['detected']}/{m['n_leaks']} leaks detected, "
                 f"{m['false_alarms']} false alarms, "
                 f"median delay {m['median_delay_days']:.0f} d")
    ax.set_ylabel("CUSUM statistic")
    ax.legend(ncol=4, fontsize=8)
    _save(fig, "03_cusum_trace")


def fig_method_comparison() -> None:
    """Our detector against the minimum-night-flow method utilities use today."""
    det = pd.read_csv(C.TABLES / "detection_summary.csv")
    fig, axes = plt.subplots(1, 2, figsize=(10, 4))

    x = np.arange(len(det))
    w = 0.36
    axes[0].bar(x - w / 2, det["mnf_detection_rate"] * 100, w,
                label="Minimum night flow (industry baseline)", color=GREY)
    axes[0].bar(x + w / 2, det["detection_rate"] * 100, w,
                label="Residual CUSUM (this work)", color=BLUE)
    for xi, (a, b) in enumerate(zip(det["mnf_detection_rate"], det["detection_rate"])):
        axes[0].text(xi - w / 2, a * 100 + 2, f"{a:.0%}", ha="center", fontsize=8)
        axes[0].text(xi + w / 2, b * 100 + 2, f"{b:.0%}", ha="center", fontsize=8)
    axes[0].set_xticks(x)
    axes[0].set_xticklabels([f"{r.year}\n({r.role})" for r in det.itertuples()])
    axes[0].set_ylabel("leaks detected (%)")
    axes[0].set_ylim(0, 138)
    axes[0].set_title("Detection rate")
    axes[0].legend(fontsize=8, loc="upper center", framealpha=0.95)

    axes[1].bar(x - w / 2, det["mnf_median_delay_days"], w, color=GREY)
    axes[1].bar(x + w / 2, det["median_delay_days"], w, color=BLUE)
    axes[1].set_xticks(x)
    axes[1].set_xticklabels([str(y) for y in det["year"]])
    axes[1].set_ylabel("days")
    axes[1].set_title("Median time to detection (lower is better)")
    _save(fig, "04_method_comparison")


def fig_detection_delays(year: int = C.TEST_YEAR) -> None:
    """Per-leak detection latency, split by leak type."""
    df = pd.read_csv(C.TABLES / f"detections_{year}.csv").sort_values("delay_days")
    colors = [BLUE if k == "abrupt" else ORANGE for k in df["kind"]]

    fig, ax = plt.subplots(figsize=(9, 4))
    ax.barh(df["pipe"], df["delay_days"], color=colors)
    ax.set_xlabel("days from leak start to alarm")
    ax.set_title(f"{year}: time to detection per leak")
    handles = [plt.Rectangle((0, 0), 1, 1, color=c) for c in (BLUE, ORANGE)]
    ax.legend(handles, ["abrupt", "incipient"], fontsize=8)
    _save(fig, "05_detection_delays")


def fig_sensor_ablation() -> None:
    """The cost-benefit curve a utility actually needs."""
    df = pd.read_csv(C.TABLES / "sensor_ablation.csv")
    fig, ax1 = plt.subplots(figsize=(8, 4.2))
    ax1.plot(df["n_sensors"], df["detection_rate"] * 100, "o-", color=BLUE,
             label="detection rate")
    ax1.plot(df["n_sensors"], df["localisation_top1"] * 100, "s-", color=GREEN,
             label="localisation top-1")
    ax1.set_xlabel("number of pressure sensors")
    ax1.set_ylabel("%")
    ax1.set_ylim(0, 105)

    ax2 = ax1.twinx()
    ax2.bar(df["n_sensors"], df["false_alarms"], width=1.4, alpha=0.3,
            color="crimson", label="false alarms/year")
    ax2.set_ylabel("false alarms per year", color="crimson")
    ax2.grid(False)

    h1, l1 = ax1.get_legend_handles_labels()
    h2, l2 = ax2.get_legend_handles_labels()
    ax1.legend(h1 + h2, l1 + l2, loc="lower right", fontsize=8)
    ax1.set_title("More sensors buy precision and localisation, not raw recall")
    _save(fig, "06_sensor_ablation")


def fig_zone_ablation() -> None:
    """How precisely can we localise before accuracy collapses?"""
    df = pd.read_csv(C.TABLES / "zone_ablation.csv")
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.plot(df["n_zones"], df["top1"] * 100, "o-", label="top-1", color=BLUE)
    ax.plot(df["n_zones"], df["top3"] * 100, "s-", label="top-3", color=GREEN)
    ax.plot(df["n_zones"], df["random_top1"] * 100, "k--", label="random guess")
    for r in df.itertuples():
        ax.annotate(f"{r.mean_zone_length_km:.1f} km", (r.n_zones, r.top1 * 100),
                    textcoords="offset points", xytext=(0, -14), fontsize=7,
                    ha="center", color="0.35")
    ax.set_xlabel("number of zones the network is split into")
    ax.set_ylabel("zone accuracy (%)")
    ax.set_title("Localisation accuracy vs granularity "
                 "(label = mean pipe length per zone)")
    ax.legend(fontsize=8)
    _save(fig, "07_zone_ablation")


def fig_localisation_example(year: int = C.TEST_YEAR, pipe: str = "p523") -> None:
    """Show one leak being localised on the map."""
    lk = next(l for l in C.leaks_for(year) if l.pipe == pipe)
    obs = L.observed_signature(year, pd.Timestamp(lk.start.date()))
    loc = L.localise(obs)
    z = N.zone_assignment()
    pt = N.pipe_table()
    sc = N.sensor_coords()
    true_zone = int(z.at[pipe, "zone"])

    fig, axes = plt.subplots(1, 2, figsize=(12, 5),
                             gridspec_kw={"width_ratios": [1.5, 1]})
    ax = axes[0]
    ax.scatter(z["x"], z["y"],
               c=loc.zone_scores.reindex(z["zone"]).to_numpy(),
               cmap="YlOrRd", s=9, linewidths=0)
    ax.scatter(sc["x"], sc["y"], marker="^", s=45, c="black", alpha=0.6)
    ax.scatter(pt.at[pipe, "x"], pt.at[pipe, "y"], marker="X", s=200, c="lime",
               edgecolors="black", linewidths=1.2, zorder=6,
               label=f"true leak ({pipe}, zone {true_zone})")
    ax.scatter(pt.at[loc.pipe_pred, "x"], pt.at[loc.pipe_pred, "y"], marker="o",
               s=130, facecolors="none", edgecolors="blue", linewidths=2, zorder=6,
               label=f"best match ({loc.pipe_pred}, zone {loc.zone_pred})")
    ax.set_aspect("equal")
    ax.legend(fontsize=8, loc="upper left")
    ax.set_title(f"Zone likelihood for leak {pipe} "
                 f"({lk.start:%d %b %Y}, {lk.kind})")
    ax.set_xlabel("x (m)")
    ax.set_ylabel("y (m)")

    top = loc.zone_scores.head(6)[::-1]
    axes[1].barh([f"zone {i}" for i in top.index], top.to_numpy(),
                 color=[GREEN if i == true_zone else GREY for i in top.index])
    axes[1].set_xlabel("cosine similarity to simulated fingerprint")
    axes[1].set_title("Ranked zones (green = correct)")
    _save(fig, "08_localisation_example")


def fig_signature_match(year: int = C.TEST_YEAR, pipe: str = "p523") -> None:
    """Observed fingerprint against the EPANET-simulated one."""
    import simulate as sim

    lk = next(l for l in C.leaks_for(year) if l.pipe == pipe)
    obs = L.observed_signature(year, pd.Timestamp(lk.start.date()))
    obs = obs / np.linalg.norm(obs.to_numpy())
    simd = sim.unit_signatures().loc[pipe].reindex(obs.index)

    fig, ax = plt.subplots(figsize=(11, 4))
    x = np.arange(len(obs))
    w = 0.4
    ax.bar(x - w / 2, obs.to_numpy(), w, label="observed (real SCADA)", color=BLUE)
    ax.bar(x + w / 2, simd.to_numpy(), w, label="simulated (EPANET)", color=ORANGE)
    ax.set_xticks(x)
    ax.set_xticklabels(obs.index, rotation=90, fontsize=6)
    cos = float(np.dot(obs.to_numpy(), simd.to_numpy()))
    ax.set_title(f"Leak fingerprint for {pipe}: observed vs simulated "
                 f"(cosine similarity {cos:.2f})")
    ax.set_ylabel("normalised pressure response")
    ax.legend(fontsize=8)
    _save(fig, "09_signature_match")


def fig_mnf_noise() -> None:
    """Why the classical night-flow method fails on this network."""
    fig, ax = plt.subplots(figsize=(10, 4))
    inflow = dl.total_inflow(C.TEST_YEAR)
    night = inflow.between_time("02:00", "04:00").resample("D").mean()
    ax.plot(night.index, night.to_numpy(), lw=0.9, color="#555555",
            label="minimum night flow")
    for i, lk in enumerate(C.leaks_for(C.TEST_YEAR)):
        ax.axvline(pd.Timestamp(lk.start.date()), color="crimson", lw=0.9,
                   alpha=0.5, label="leak onset" if i == 0 else None)
    ax.set_ylabel("cubic metres per hour")
    ax.set_title("Minimum night flow in 2019: leak onsets are buried in demand noise")
    ax.legend(fontsize=8)
    _save(fig, "10_mnf_noise")


def build_all() -> None:
    print("writing figures:")
    fig_network_map()
    fig_residual_heatmap()
    fig_cusum_trace()
    fig_method_comparison()
    fig_detection_delays()
    fig_sensor_ablation()
    fig_zone_ablation()
    fig_localisation_example()
    fig_signature_match()
    fig_mnf_noise()


if __name__ == "__main__":
    build_all()
