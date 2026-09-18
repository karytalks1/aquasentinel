"""Export the L-Town network, zones, sensors and leak results as GIS layers.

Usage:  .venv/Scripts/python.exe src/export_gis.py   ->  gis/*.csv and gis/*.geojson

The CSVs are made for ArcGIS Pro:
  pipes.csv    -> "XY To Line"          (start_x, start_y, end_x, end_y)
  sensors.csv  -> "XY Table To Point"  (x, y)
  leaks.csv    -> "XY Table To Point"  (x, y), one row per leak with its detection result
  zones.csv    -> "XY Table To Point"  (zone centres, for labels)

The GeoJSON files hold the same layers for QGIS or any web map.

Coordinates are the network model's own local grid in metres (the L-Town file
has no real-world location), so any projected metric coordinate system displays
them at the right scale.
"""
import json
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).parent))
import config as C          # noqa: E402
import network as N         # noqa: E402

OUT = Path(__file__).resolve().parent.parent / "gis"


def _feature(geom_type, coords, props):
    return {"type": "Feature", "geometry": {"type": geom_type, "coordinates": coords},
            "properties": props}


def _write_geojson(name, features):
    (OUT / f"{name}.geojson").write_text(
        json.dumps({"type": "FeatureCollection", "features": features}), encoding="utf-8")


def main() -> None:
    OUT.mkdir(exist_ok=True)
    coords = N.node_coords()
    zones = N.zone_assignment()

    # pipes: one line per pipe with its zone
    pipes = zones.reset_index()[["pipe", "start", "end", "length_m", "diameter_m", "zone"]].copy()
    pipes["start_x"] = pipes["start"].map(coords["x"])
    pipes["start_y"] = pipes["start"].map(coords["y"])
    pipes["end_x"] = pipes["end"].map(coords["x"])
    pipes["end_y"] = pipes["end"].map(coords["y"])
    pipes["diameter_mm"] = (pipes["diameter_m"] * 1000).round(1)
    pipes["zone"] = pipes["zone"] + 1                      # zones 1..12 for people, not 0..11
    pipes = pipes.drop(columns="diameter_m")
    pipes[["start_x", "start_y", "end_x", "end_y"]] = pipes[["start_x", "start_y", "end_x", "end_y"]].round(2)
    pipes.to_csv(OUT / "pipes.csv", index=False)
    _write_geojson("pipes", [
        _feature("LineString", [[r.start_x, r.start_y], [r.end_x, r.end_y]],
                 {"pipe": r.pipe, "zone": int(r.zone), "length_m": r.length_m,
                  "diameter_mm": r.diameter_mm})
        for r in pipes.itertuples()])

    # pressure sensors
    sensors = N.sensor_coords().reset_index().rename(columns={"index": "sensor"})
    sensors.columns = ["sensor", "x", "y"]
    sensors.to_csv(OUT / "sensors.csv", index=False)
    _write_geojson("sensors", [_feature("Point", [r.x, r.y], {"sensor": r.sensor})
                               for r in sensors.itertuples()])

    # zone centres, for labelling
    cent = N.zone_centroids().reset_index()
    cent["zone"] = cent["zone"] + 1
    stats = N.zone_stats().reset_index()
    stats["zone"] = stats["zone"] + 1
    cent = cent.merge(stats, on="zone")
    cent.to_csv(OUT / "zones.csv", index=False)
    _write_geojson("zones", [_feature("Point", [r.x, r.y],
                                      {"zone": int(r.zone), "n_pipes": int(r.n_pipes),
                                       "length_km": round(r.length_km, 2)})
                             for r in cent.itertuples()])

    # leaks with the study's results, both years
    rows = []
    for year in (C.TRAIN_YEAR, C.TEST_YEAR):
        det = pd.read_csv(C.TABLES / f"detections_{year}.csv").set_index("pipe")
        loc = pd.read_csv(C.TABLES / f"localisation_{year}.csv").set_index("pipe")
        for lk in C.leaks_for(year):
            if lk.pipe not in zones.index:
                continue
            z = zones.loc[lk.pipe]
            d = det.loc[lk.pipe] if lk.pipe in det.index else None
            lo = loc.loc[lk.pipe] if lk.pipe in loc.index else None
            detected = d is not None and pd.notna(d.get("alarm_day"))
            rows.append({
                "pipe": lk.pipe, "year": year, "kind": lk.kind,
                "diameter_mm": round(lk.diameter_m * 1000, 1),
                "start": lk.start.strftime("%Y-%m-%d"),
                "x": z["x"], "y": z["y"], "true_zone": int(z["zone"]) + 1,
                "detected": "yes" if detected else "no",
                "delay_days": int(d["delay_days"]) if detected else None,
                "zone_rank": int(lo["zone_rank"]) if lo is not None else None,
            })
    leaks = pd.DataFrame(rows)
    leaks[["delay_days", "zone_rank"]] = leaks[["delay_days", "zone_rank"]].astype("Int64")
    leaks[["x", "y"]] = leaks[["x", "y"]].round(2)
    leaks.to_csv(OUT / "leaks.csv", index=False)
    _write_geojson("leaks", [
        _feature("Point", [r["x"], r["y"]], {k: v for k, v in r.items() if k not in ("x", "y")})
        for r in leaks.astype(object).where(leaks.notna(), None).to_dict("records")])

    print(f"wrote {OUT}")
    print(f"  pipes   {len(pipes)}   sensors {len(sensors)}   zones {len(cent)}   leaks {len(leaks)}")


if __name__ == "__main__":
    main()
