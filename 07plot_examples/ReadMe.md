# `plot_examples/` — Visualization and Figure Generation

This folder contains Jupyter notebooks for generating the key figures and plots used in the main paper and supplementary materials.  
It includes analyses of solar/wind/hydro generation patterns, load-weather relationships, and future scenario comparisons.

---

## 📌 Purpose

- Visualize patterns in historical and projected energy data
- Support diagnostics of load fitting, generation consistency, and weather effects
- Generate summary plots by time (daily, seasonal), region, or variable type
- Reproduce figures used in the main text and supplement

---

## 🗂️ Notebook Descriptions

| Notebook Name                     | Function |
|----------------------------------|----------|
| `01_load_fitting.ipynb`          | Visualizes segmented load-temperature fitting for heating/cooling demand |
| `02_load_r2.ipynb`               | Plots R2 statistics of load-weather models for different regions/sectors |
| `10_map_gen.ipynb`               | Generates spatial example maps of renewable generation |
| `11_compare_generation.ipynb`    | Compares generation across true and predicted samples |
| `12_compare_output.ipynb`        | Compares weather across true and predicted samples |
| `13_future_weather_test.ipynb`   | Validates future weather series generated via downscaling or GANs |
| `21_dist_box.ipynb`              | Creates box plots showing inter-scenario distributions (extreme net load) |
| `22_solar_daily_trend.ipynb`     | Analyzes and plots daily solar generation trends across the year |
| `23_wind_daily_trend.ipynb`      | Same as above, but for wind generation |
| `24_solar_season_trend.ipynb`    | Aggregates solar generation by season and plots regional trends |
| `25_wind_season_trend.ipynb`     | Same as above, but for wind generation |
| `26_total_season_heatmap.ipynb`  | Heatmap of total seasonal generation by region |
| `31_hydro_summary.ipynb`         | Summary and visualization of regional hydropower trends |

---

## 📂 Data Integration

These notebooks assume that the following data are available (typically from Google Drive):

- Processed `.npz` or `.csv` files for generation, load, and weather
- Region/county mappings from the `county/` folder
- Scenario outputs from `future/` and `hydro/` folders

---


