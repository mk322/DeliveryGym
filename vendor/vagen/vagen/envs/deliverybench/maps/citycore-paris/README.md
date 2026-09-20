# CityCore Paris DeliveryBench export

Generated from `/Game/CityCore_Paris/Scenes/ParisCity_FinalBlueprints` with
Unreal Engine 5.8. The export preserves the complete authored actor/building
catalogue while producing a connected semantic subset suitable for
DeliveryBench routing.

## DeliveryBench inputs

- `roads.json` — connected navigation road graph; XY is in metres.
- `progen_world_enriched.json` — navigable buildings/POIs and traffic lights;
  locations and bounding boxes are in centimetres.

## Full-fidelity/provenance files

- `citycore_scene_raw.json` — complete UE extraction of 3,290 scene actors.
- `elements.json` — transforms/bounds for every scene actor.
- `buildings.json` — all 573 authored procedural buildings, including skyline
  shells that do not have a walkable road entrance.
- `roads_detailed.json` — authored centreline polylines, widths, sources, and
  explicitly marked intersection/component connectors.
- `map_transform.json` — identity UE-to-DeliveryBench coordinate contract.
- `export_report.json` — counts, bounds, errors, and road-connectivity audit.

`progen_world_enriched.json` intentionally omits background building shells
whose estimated entrance remains more than 25 m from a road. They remain in
`buildings.json` and on the full 2D map.

## 2D maps

- `citycore_paris_2d.png` — dependency-free raster preview.
- `citycore_paris_2d.svg` — scalable map with hover titles for POIs/lights.

## Regenerate

From the CityCore project root:

```bash
Scripts/export_deliverybench_map.sh
```

The UE pass extracts the scene, the standard-Python pass converts it, and the
final pass renders both map formats. No third-party Python packages are needed.
