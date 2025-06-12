# `load/` — Electricity Load Processing and Weather Sensitivity Analysis

This folder contains scripts and notebooks for preprocessing historical electricity load data, and analyzing its correlation with weather drivers such as temperature.  
It supports both residential and commercial/industrial sectors, and helps build the foundation for future demand projections under climate change.

---

## 📌 Purpose

- Normalize and aggregate electricity load data (hourly, county-level)
- Analyze correlation between temperature and load
- Separate treatment for residential and commercial sectors
- Prepare weather-dependent statistical models for downstream use

---

## 🗂️ Notebook Descriptions

| Notebook Name                     | Function |
|----------------------------------|----------|
| `01_normalize_loads.ipynb`       | Cleans and normalizes historical load profiles across different counties and sectors based on different datasets |
| `02_weather_mean.ipynb`          | Computes average weather variables (e.g., temperature) at county-level, used for correlation and fitting |
| `03_corre_commercial_save.ipynb` | Analyzes the temperature-load correlation for commercial & industrial sectors; saves fitting coefficients |
| `04_corre_residential_save.ipynb`| Performs similar correlation analysis for residential loads; includes nonlinear or segmented temperature response |

---

## 🧮 Analysis Methodology

- Loads and weather data are matched by hour and location (county FIPS)
- Normalized outputs are scaled using population, floor area, or other factors as available

---
