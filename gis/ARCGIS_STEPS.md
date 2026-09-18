# Making the AquaSentinel map in ArcGIS Pro

About 30 minutes. The result is `reports/figures/11_arcgis_zone_map.png`: the L-Town
network coloured by zone, the 33 pressure sensors, and the 2019 leaks marked by
whether they were detected and how well they were localised.

The layers in this folder come from `src/export_gis.py`. Re-run it if the study changes.

## 1. Project and coordinate system

1. ArcGIS Pro > New > Map. Name the project `aquasentinel`.
2. Map Properties > Coordinate Systems > choose any **projected** system in metres,
   for example *WGS 1984 World Mercator*. L-Town is a model network with local
   coordinates in metres and no real location, so this only sets the units and scale.
   Say exactly that if asked: "local model coordinates, metres".

## 2. Pipes as lines

1. Geoprocessing > **XY To Line**.
2. Input table `gis/pipes.csv`. Start X `start_x`, Start Y `start_y`, End X `end_x`,
   End Y `end_y`. Line type: Geodesic off / Planar. Tick **Preserve attributes**.
3. Symbology > **Unique Values** on `zone` (12 colours), line width 1.5.

## 3. Sensors, zone labels and leaks as points

1. Geoprocessing > **XY Table To Point**, run three times:
   - `gis/sensors.csv` (X `x`, Y `y`): black triangles, size 8.
   - `gis/zones.csv` (X `x`, Y `y`): no symbol, **label** with `zone`.
   - `gis/leaks.csv` (X `x`, Y `y`).
2. On the leaks layer, add a **Definition Query**: `year = 2019`.
3. Symbology > Unique Values on `detected`: yes = green circle, no = red cross.
4. Label leaks with `zone_rank` (1 means the right zone was ranked first).

## 4. Layout and export

1. Insert > New Layout > A4 landscape. Insert > Map Frame.
2. Add a **legend**, a **scale bar** (metres) and a **north arrow**.
3. Title: *L-Town water network: 12 search zones and 2019 leak results*.
4. Share > Export Layout > PNG, 200 dpi, save as
   `reports/figures/11_arcgis_zone_map.png`.
5. Save the ArcGIS project too, then push:

```bash
git add reports/figures/11_arcgis_zone_map.png
git commit -m "ArcGIS zone and leak map"
git push
```

## What to say in an interview

"I exported the network, sensors and leak results from Python as tables, built the
lines and points in ArcGIS with XY To Line and XY Table To Point, and symbolised the
12 search zones and whether each 2019 leak was found. It shows at a glance that the
missed leaks sit in zones that already had other leaks running."
Check that last sentence against your own map before you say it.
