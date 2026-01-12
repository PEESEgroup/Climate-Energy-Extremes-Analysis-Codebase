# `future/` — Future Scenario Projection Scripts

This folder contains scripts used to project future energy and climate conditions, leveraging historical models, scenario-specific drivers, and downscaled temperature and generation data.

These scripts support the core analysis of how energy systems might evolve under climate change and planning policy variations (e.g., SSP245, SSP585).

---

## 🔍 Purpose

- Use trained models and historical baselines to **extrapolate future conditions** (2030–2050)
- Generate county-level estimates of:
  - Downscaled temperatures
  - Renewable generation
  - Electricity loads

---

## 🗂️ Script Descriptions

| Script Name                      | Function |
|----------------------------------|----------|
| `01_future_downscale_complete.py` | Script to generate future high-resoltion weather projections |
| `02_future_county_temp.py`        | Downscales and generates county-level future temperature time series based on climate models and scenario inputs |
| `03_future_gen.py`               | Computes projected renewable generation (wind/solar) at high spatial and temporal resolution |
| `04_future_loads.py`            | Estimates future electricity loads (residential, commercial, industrial, transportation) per county |
| `05_future_gen_sub.py`          | Sub-component for `03_future_gen.py`: applies capacity adjustments and zonal factors for subregional analysis |
| `06_gen_unit_process.py`        | Processes generator-specific configuration and scaling assumptions; maps plant capacities to grid cells |

---

## 📂 Data Requirements

These scripts depend on inputs stored in Google Drive (~1 TB), including:

- Climate model projections (e.g., CMIP6, SSP245/585)
- Capacity expansion assumptions and generator deployment profiles

---


## 📌 Notes

- All outputs are saved in structured directories for later aggregation and plotting
- Please replace the directory with your own one

---

