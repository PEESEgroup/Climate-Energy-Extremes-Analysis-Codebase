# Climate-Energy Extremes Analysis Codebase

Code for *Weather and climate extremes stress the US power system through two channels with
different hazard rankings*. The repository holds every program that produces a number, a table or a
figure in the manuscript and its Supplementary Information, organized in the order the pipeline
runs.

The study spans two channels. The adequacy channel simulates hourly county demand and plant-level
wind and solar output for 1980 to 2019 and for 32 future realizations of 2030 to 2050, aggregated to
18 planning subregions. The damage channel estimates hazard-conditioned outage on 6,418,440 observed
county-days. A generative downscaling network supplies the 4 km hub-height wind that both channels
depend on.

This is a code-only release, reduced to the programs that produce the main-text figures and the
numbers in them. Every input is public, and `01_acquire_data` holds an acquisition script for each
one, so the archive is rebuilt rather than transferred. The reason is size: a full rebuild writes
about 18 TB. See [Data sources](#data-sources).

---

## 1. System requirements

### Software

| Component | Version used | Notes |
|---|---|---|
| Operating system | Ubuntu 24.04.4 LTS, kernel 6.17 | Any Linux with the packages below; not tested on macOS or Windows |
| Analysis interpreter | Python 3.12.3 | numpy 2.5.1, pandas 3.0.3, scipy 1.18.0, matplotlib 3.11.1, pyarrow 25.0.0, xarray 2026.7.0, zarr 3.3.0, adlfs 2026.8.0, fsspec 2026.7.0, pystac-client 0.9.0, planetary-computer 1.0.0, nrel-pysam 7.1.1 |
| Demand interpreter | Python 3.10.20 | tell 1.3.0, numpy 1.26.4, pandas 2.3.3, scipy 1.15.3, scikit-learn 1.0.2, statsmodels 0.14.6, netCDF4 1.7.4, geopandas 1.1.4, shapely 2.1.2, tables 3.10.1 |
| Siting interpreter | Python 3.12.3 | cerf 2.4.0, rasterio 1.5.0, geopandas 1.1.4, shapely 2.1.2, openpyxl 3.1.5 |
| Deep-learning interpreter | Python 3.13.14 | torch 2.12.1+cu130, h5py 3.16.0, netCDF4 1.7.4 |

Four interpreters are used because the geospatial, TELL and torch stacks do not coexist cleanly in
one environment. Each script's header names the interpreter it expects. `04_demand_model` needs
`tell`; `05_capacity_siting` needs `cerf` and `rasterio`. All of `02_downscale_wind` needs torch,
and so do four programs that sit in other stages because that is where they belong in the pipeline:
`01_acquire_data/regrid_tgw.py`, `03_wind_solar_generation/02_cache_lr.py`,
`03_wind_solar_generation/03_tgw3d_srgan_gen.py` and `03_wind_solar_generation/17_fut_gen.py`. Run
those four under `cee-torch`. The four environments are pinned by `requirements.txt`, `requirements-load.txt`,
`requirements-siting.txt` and `requirements-gpu.txt`.

### Hardware

| Resource | Full pipeline | Figure and table regeneration |
|---|---|---|
| CPU | 16 cores | 2 cores |
| Memory | 128 GB. The outage panel peaks near 45 GB per fitting process, and two run concurrently | 8 GB |
| GPU | One NVIDIA L40S with 48 GB, required only to train or run the downscaling network | Not required |
| Disk | About 20 TB free | About 1 GB |

No other non-standard hardware is required. A Globus endpoint and a Microsoft Planetary Computer
account are needed for two of the acquisition scripts, and both are free.

### Run time on the hardware above

| Stage | Wall clock |
|---|---|
| Acquisition of the full archive | Days to weeks, bounded by the remote endpoints rather than by compute |
| Training the deployed downscaling network | 57.7 h on one L40S, of which the deployed checkpoint is reached at 40.4 h |
| Downscaling inference over the historical and future archives | About one week |
| Demand model, 1980 to 2019 plus 32 future realizations | About 5 h per arm |
| Outage attribution panel | 4 to 5 h |
| Figure 4 tercile interactions, six county traits | 4 h 46 min per trait; about 14 h 20 min for all six, sharded across two processes |
| Regenerating figures and table bodies from stored analysis artifacts | 1 to 5 min per figure |

---

## 2. Installation guide

```bash
git clone https://github.com/PEESEgroup/Climate-Energy-Extremes-Analysis-Codebase.git
cd Climate-Energy-Extremes-Analysis-Codebase

# analysis environment, used by most stages
conda create -n cee -c conda-forge --override-channels python=3.12 -y && conda activate cee
pip install -r requirements.txt
pip install pystac-client planetary-computer cdsapi gcamreader

# demand environment
conda create -n cee-tell -c conda-forge --override-channels python=3.10 -y && conda activate cee-tell
pip install -r requirements-load.txt
pip install tell==1.3.0 tables

# siting environment
conda create -n cee-cerf -c conda-forge --override-channels python=3.12 -y && conda activate cee-cerf
pip install -r requirements-siting.txt
pip install cerf==2.4.0 gcamreader

# downscaling environment, needed to train or run the network
conda create -n cee-torch -c conda-forge --override-channels python=3.13 -y && conda activate cee-torch
pip install -r requirements-gpu.txt
```

The four requirements files pin the exact versions the released results were produced with. `tell`,
`cerf`, `gcamreader`, `cdsapi`, `pystac-client` and `planetary-computer` are installed separately
because they pull their own solvers and are needed only by the stage named beside them.

Each environment is created from `conda-forge` alone. The default Anaconda channels require their
Terms of Service to be accepted before an environment can be created, and `--override-channels`
avoids that step.

Typical installation time on a normal desktop computer is **under 10 minutes** for the analysis
environment and **about 20 minutes** for all four, dominated by the torch download. No compilation
step is required.

Each script reads absolute paths from its header. Set them to your own archive root before running a
stage; `MANIFEST.csv` records the path each program was taken from.

---

## 3. Demo

No data are bundled, so the demo runs on the analysis artifacts that the pipeline writes. They are
small, so this stage is the fastest way to confirm an installation works end to end.

```bash
conda activate cee

# 1. Report which hazards survived the pre-event screen
cd 09_outage_attribution && python -c "import hazsets; print(hazsets.screened())" && cd ..

# 2. Regenerate the damage projection of Figure 6
python 12_figures/07_fig6_damage.py
```

**Expected output.** Step 1 prints

```
['tc', 'convective', 'heat']
```

which is the screened hazard set: hurricanes, severe convection and heat waves are carried into the
attribution, while cold outbreaks and fire weather are not. Step 2 writes `fig6_damage.svg` and
`fig6_damage.png` and prints

```
baseline 24.83%   headline rcp85hotter
dose +1.74 pp [+1.22, +2.44] vs binary +0.11 pp   waterfall sum +1.74
```

which is the headline of Figure 6c: the tropical-cyclone share of national outage rises 1.74
percentage points on a 24.83% baseline.

**Expected run time on a normal desktop computer.** Step 1 takes **under 2 seconds**. Step 2 takes
**about 40 seconds**. The interval it prints comes from 2,000 coefficient draws taken upstream in
`11_damage_projection/06_r5_dose2.py`.

Both steps read stored artifacts of a few hundred kilobytes. Reproducing those artifacts from raw
weather requires the full pipeline and the archive described below.

---

## 4. Instructions for use

### How to run the software on your own data

The stages run in numeric order. Each writes the inputs the next one reads.

| Stage | What it does | Interpreter | Needs |
|---|---|---|---|
| `01_acquire_data` (28) | Downloads TGW, CONUS404, ERA5, HURDAT2, EAGLE-I, EIA, FERC, FEMA, ACS | analysis | Network, Globus, Planetary Computer |
| `02_downscale_wind` (8) | Trains and runs the 3D wind generator, 12 km to 4 km at seven heights | gpu | GPU |
| `03_wind_solar_generation` (19) | Plant-level wind and solar generation through PySAM, hourly super-resolution | analysis | Stage 02 output |
| `04_demand_model` (11) | County and balancing-authority demand, calibration, sectoral allocation | load | Stage 01 weather |
| `05_capacity_siting` (9) | CERF siting with the local ordinance layer and the corrected price surface | siting | GCAM-USA, GRIDCERF |
| `06_netload_panel` (5) | Assembles subregion net load and the historical panel | analysis | Stages 03, 04 |
| `07_hazard_calendar` (22) | County and subregion hazard flags for all seven hazards | analysis | Stage 01 weather |
| `08_adequacy_analysis` (8) | Composites, the joint panel and the adopted atmospheric-river flag | analysis | Stages 06, 07 |
| `09_outage_attribution` (19) | The Poisson attribution panel, the screen, and the tercile interactions | analysis | Stages 01, 07 |
| `10_future_adequacy` (10) | Future net load, capacity credit, stress episodes, hazard frequencies | analysis | Stages 03, 04, 05 |
| `11_damage_projection` (8) | The wind dose response carried into the future arm | analysis | Stage 09 |
| `12_figures` (8) | The six main-text figures and the subregion map | analysis | All above |

Within a stage, programs are numbered in the order they run. Modules that other programs import
carry no number, because they are read rather than executed: `regrid_tgw.py`,
`srgan_dataset_3d.py`, `gen_physics.py`, `paths.py`, `loadcal.py`,
`cerf_lmp_zones.py`, `hazard_defs.py`, `attrib.py`, `attrib_artifacts.py`, `hazsets.py`,
`baseline.py` and `figstyle.py`.

The interpreter column names the requirements file: analysis is `requirements.txt`, load is
`requirements-load.txt`, siting is `requirements-siting.txt` and gpu is `requirements-gpu.txt`.

Three files decide what the rest of the tree does, and they are read rather than copied:
`hazard_defs.py` holds every hazard threshold and stamps each product it generates; `hazsets.py`
reports which hazards passed the pre-event screen; `paths.py` names the three county demand
products. A consumer that disagrees with any of them fails closed rather than silently using a
superseded definition.

### Reproduction instructions

With a complete archive, the main-text figures regenerate with

```bash
for f in 12_figures/0*_fig*.py; do python "$f"; done
```

Every quantitative claim in the main text traces to one of these programs. `MANIFEST.csv` gives the
stage, the original path and a one-line statement of purpose for all 161 programs.

---

## 5. Directory structure

```
.
├── 01_acquire_data/              acquisition scripts, one per public source
├── 02_downscale_wind/            3D wind generator: cache, training, bias correction
├── 03_wind_solar_generation/     PySAM wind and solar, temporal super-resolution
├── 04_demand_model/              TELL-based demand, calibration, county allocation
│   ├── paths.py                  the one name for each of the three demand products
│   └── loadcal.py                per-authority quantile-mapping calibration
├── 05_capacity_siting/           CERF siting, ordinance layer, price surface
├── 06_netload_panel/             subregion net load and the historical panel
├── 07_hazard_calendar/           county and subregion hazard flags
│   └── hazard_defs.py            every threshold, season and persistence rule, with stamps
├── 08_adequacy_analysis/         composites, joint panel, the adopted river flag
├── 09_outage_attribution/        Poisson attribution, the screen, tercile interactions
│   ├── attrib.py                 the panel that Figures 3, 4 and 6 all rest on
│   ├── 04_chk3.py                the pre-event screen that decides the carried hazards
│   ├── hazsets.py                reports that decision; fails closed on disagreement
│   └── 07_terc.py                Figure 4b, six county traits, sharded
├── 10_future_adequacy/           future net load, capacity credit, stress episodes
├── 11_damage_projection/         the wind dose response carried forward
├── 12_figures/                   the six main-text figures and the subregion map
├── MANIFEST.csv                  every program, its stage, its origin and its purpose
├── requirements.txt              analysis environment, Python 3.12
├── requirements-load.txt         demand environment, Python 3.10
├── requirements-siting.txt       siting environment, Python 3.12
├── requirements-gpu.txt          downscaling environment, Python 3.13 and CUDA 13.0
├── LICENSE
└── README.md
```

---

## Data sources

No data are included. Every input is public and is fetched by a script in `01_acquire_data`.

| Domain | Source |
|---|---|
| Historical and future weather | Thermodynamic Global Warming, WRF at 12 km, ERA5-forced 1980 to 2019 and CMIP6-forced futures under SSP2-4.5 and SSP5-8.5 (https://doi.org/10.57931/1885756) |
| Downscaling truth | CONUS404 at 4 km, `az://noaa/conus404.zarr` through the Microsoft Planetary Computer, and https://usgs.osn.mghpcc.org for surface fields |
| Reanalysis indices | ERA5 vapor transport and 500 hPa height through the NCAR Research Data Archive |
| Storms | HURDAT2 best track (https://www.nhc.noaa.gov/data/#hurdat) |
| Outages | EAGLE-I county 15-minute customers out (https://doi.org/10.1038/s41597-024-03095-5) |
| Demand | EIA-930 (https://www.eia.gov/electricity/gridmonitor/about), the county-hourly product (https://doi.org/10.25984/3366592), FERC Form 714 |
| Fleet | EIA-860, EIA-923, and GODEEEP as the published benchmark (https://doi.org/10.1038/s41597-024-03894-w) |
| Future capacity | GCAM-USA (https://doi.org/10.57931/2428940), CERF (https://doi.org/10.21105/joss.03601), GRIDCERF (https://doi.org/10.5281/zenodo.10041918), Cambium (https://doi.org/10.2172/1915250) |
| Grid quality and investment | EIA-861, FERC Form 1, FEMA award records, NOAA billion-dollar disasters |
| Demographics | American Community Survey five-year, 2022 |

A full rebuild writes about 18 TB, and we recommend at least 20 TB of free disk. CONUS404 is the
largest component at about 9.6 TB, with a further 1.9 TB of training cache; the TGW archive adds
about 5.4 TB. The derived products that the analysis reads are about 270 GB.

The trained weights of the downscaling network are distributed separately as one file.
`G_deploy.pth` is the generator only, 52 MB, a 21-channel to 14-channel network mapping 89 x 211 to
625 x 1475. Every inference program in this repository loads that file and nothing else. Training
the network from scratch is reproducible from `02_downscale_wind`, and does not need it.

## What is deliberately absent

Three components were built, tested and then replaced. They are named so that their absence is
deliberate rather than accidental: the vertical residual network that predicted hub wind from 10 m
wind, replaced by the seven-level 3D generator; six-hourly future wind sampling, replaced by
three-hourly with hourly super-resolution; and the atmospheric-river proxy `pwat x |V(836 m)|`,
which validated at r = 0.924 but was not used.

## License

Code is released under the **MIT License** (see `LICENSE`). Third-party datasets retain their
original licenses and are not redistributed here.

## Contact

Questions about the code are best raised as a GitHub issue on this repository. Correspondence about
the study should go to the corresponding author of the manuscript.
