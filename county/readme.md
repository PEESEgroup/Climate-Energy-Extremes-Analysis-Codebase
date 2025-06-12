# `county/` — County-Level Geospatial and Administrative Metadata

This folder contains supporting metadata used to map U.S. counties to larger regions (such as balancing regions or climate zones). It is critical for visualizing spatial aggregations and aligning simulation results to administrative boundaries.

---

## 📌 Purpose

- Define **county-to-region** and **county-to-zone** mappings
- Support **spatial aggregation** of energy and climate data
- Provide **shapefiles** for rendering maps of the continental U.S. at county-level resolution

---

## 🗂️ File Descriptions

| File Name                     | Description |
|------------------------------|-------------|
| `county2zone_new.csv`        | Mapping of county FIPS codes to custom-defined balancing regions or climate zones |
| `disagg_geosize.csv`         | Geographic area breakdowns used for disaggregation or normalization |
| `fips_population_2018.npz`   | Population data by county (FIPS-based) from 2018, used for weighting and normalization |
| `hierarchy_original.csv`     | Defines hierarchical relationships between counties, zones, and higher-level regions |
| `tl_2020_us_county.*`        | U.S. Census shapefile components for 2020 U.S. county boundaries |
| — `.shp`                     | Main geometry (polygons of counties) |
| — `.dbf`, `.shx`, `.prj`     | Attribute table, shape index, and projection metadata |
| — `.shp.ea`, `.shp.iso.xml`  | Optional metadata (may not be used directly) |
| — `.cpg`                     | Character encoding for DBF table |

> ✅ These shapefiles are used for rendering maps using libraries such as `geopandas`, `cartopy`, or `matplotlib`.

---

## 📂 Data Integration

To use this folder in combination with the full dataset stored on **Google Drive** (~1TB), ensure the following:

- You’ve mounted or synced your Google Drive and can access county-level hourly data or results
- All `FIPS` codes in the simulation results can be mapped using files in this folder
- The shapefile (`tl_2020_us_county.*`) is accessible for spatial plotting

> The geospatial structure defined here supports the aggregation and visualization of generation/load/extreme indicators by **county**, **zone**, or **balancing region**.

