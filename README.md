# Weather risks, investment alignment and power system resilience

Code accompanying **Divergent weather risks reveal misaligned investment and reshape
power system resilience priorities**.

The study distinguishes weather-related increases in **net load** (demand minus modeled
wind and solar generation) from **customer outages**. It compares these risks with
generation additions, distribution investment and federal mitigation awards, then examines
future priorities across 32 climate, demand and policy combinations. Historical net load
covers 1980–2019 and 18 US planning subregions; the outage estimation panel covers 2,268
counties. Future comparisons cover 2030–2050. The downscaling network supplies wind inputs
for generation modeling. Historical outage estimation uses separate weather and outage records.

## Release status

This is the current **code-only release candidate**, not a self-contained, one-command
reproduction package. The bounded Fig. 6 implementation is included. Older analyses and
figure versions remain for provenance and must not be mistaken for the current workflow.

Before public release, resolve these limitations:

- Data, trained weights and fitted artifacts are not bundled. Public availability of a
  source does not establish that all required inputs can be rebuilt by the included scripts.
- Many scripts retain `/data`, `/datasets/goes_repro` or author-workspace paths. Environment
  variables are supported only by the workflows listed below, not by every stage.
- `MANIFEST.csv` records historical origins, not a complete inventory of later additions
  or a complete execution order.
- Final assembly of Figs. 1–5 and several SI plotting programs are outside this directory.
  Native plotting output is not necessarily the submitted panel selection or layout.
- Requirements retain archived version pins. Installation and end-to-end execution have
  **not** been revalidated in a clean environment for this revision.

Do not execute every numbered file or every plotting program with a wildcard. Some are
alternative analyses, diagnostic branches or superseded calculations.

## Environments and paths

| Software stack | Requirements file | Main use |
|---|---|---|
| Analysis | `requirements.txt` | Arrays, analysis and plotting |
| Demand | `requirements-load.txt` | TELL, calibration and statistical dependencies |
| Siting | `requirements-siting.txt` | CERF and geospatial processing |
| GPU | `requirements-gpu.txt` | PyTorch downscaling training and inference |

Review script imports before installation: TELL, CERF and some acquisition clients are
installed separately, and some analysis scripts require packages listed in another stack.
The requirements files describe the archived setup, not a verified universal installation
recipe. Do not combine all four stacks without resolving version constraints.

Example, within an already selected compatible environment:

```bash
python -m pip install -r requirements.txt
```

Full reconstruction needs substantial weather storage and GPU resources for downscaling.
Earlier workspace estimates were approximately 18 TB and an L40S-class GPU; these are
planning estimates, not benchmarked requirements for this release. Plotting can use existing
analysis artifacts without retraining, but the artifacts and definition stamps must be
provided. No synthetic-data demo is included.

| Variable | Implemented workflow | Purpose |
|---|---|---|
| `GOES_PLOT_DATA_ROOT` | Shared native plotting for Figs. 1–5 | Analysis-artifact mirror |
| `GOES_PLOT_OUTPUT_DIR` | Most current native Figs. 1–5 plots | Output directory |
| `GOES_DATA_ROOT` | Bounded Fig. 6 | Original data archive |
| `GOES_FIG6_WORK` | Bounded Fig. 6 | Baseline, fits, projections and diagnostics |
| `GOES_FIGURE_DIR` | Fig. 6 export | Output directory |
| `GOES_ARIAL_FONT` | Fig. 6 export | Optional Arial TTF path |

Some map/grid paths remain hard-coded even with these variables set. Inspect headers
before running. Install Arial for the intended figure typography; do not redistribute
proprietary fonts without permission.

## Analysis stages

This table describes scientific roles, not a complete execution graph.

| Directory | Role |
|---|---|
| `01_acquire_data` | Weather, outage, demand, fleet, investment and demographic acquisition/preprocessing |
| `02_downscale_wind` | Wind downscaling: cache, normalization, training and inference |
| `03_wind_solar_generation` | Generation, temporal interpolation and loss treatment |
| `04_demand_model` | County/authority demand, calibration and future demand |
| `05_capacity_siting` | Future capacity and policy-dependent siting |
| `06_netload_panel` | Historical/future net load aggregation |
| `07_hazard_calendar` | Hazard definitions, event calendars and regression controls |
| `08_adequacy_analysis` | Net load comparisons, regressions and robustness; includes older branches |
| `09_outage_attribution` | Historical attribution, population comparisons and investment alignment |
| `10_future_adequacy` | Future peaks, duration, hazard frequency and variance decomposition |
| `11_damage_projection` | Bounded cyclone outage projection and legacy diagnostics |
| `12_figures` | Native panels, current exports and older layouts |

Keep shared modules with their consumers. In particular, hazard-definition stamps,
pre-event screens, path definitions and model utilities are not optional plotting files.
Circulation indices remain regression controls even though the submitted Fig. 1 no longer
displays the earlier circulation panels.

## Figure mapping

This follows the manuscript assembly workflow, not historical filename numbering.

| Submitted figure | Program(s) in `12_figures/` | Final selection |
|---|---|---|
| Fig. 1 | `02_fig1_full_oc.py` | Hazard maps, demand/generation estimates and lag panel |
| Fig. 2 | `03c_fig2_merged.py` | Demand–outage comparison, outage associations and county burden; also exports SI intensity/timing panels |
| Fig. 3 | `05c_si_adequacy_alignment.py` | Despite its `si` filename, supplies the main generation-investment figure |
| Fig. 4 | `05_fig4_v2.py` + `05b_fig4_alignment_panels.py` | Population, undergrounding and federal-funding panels |
| Fig. 5 | `06_fig5_adequacy_oc.py` | Five selected panels, including hazard frequency and variance decomposition |
| Fig. 6 | `07_fig6_damage.py` | Reads completed bounded-model results; does not fit the model |

The author workspace assembles Figs. 1–5 using
`submission_Revised/scripts_codex/assemble_native_figures_2026_10_01.py` and SVG helpers
in `close_remaining_figures_2026_09_28.py`. These files are **outside this release directory**.
They select existing SVG axes and change layout/labels, not scientific data. Their system
dependencies include `lxml`, PyGObject/librsvg and Cairo. A standalone release still needs
these files packaged with portable paths.

`03_fig2_v2_oc.py` and `04_fig3_v8.py` represent earlier figure organizations, not the
submitted Figs. 2 and 3. `06c_fig5_hazard_frequency.py` is a standalone export of a panel
already supplied by the current Fig. 5 program. Do not use these as replacement entry points.

The current SI also contains downscaling, demand, generation and siting diagnostics.
Not all their final plotting programs are present here; this directory alone does not
yet provide complete SI reproduction.

## Current Fig. 6

Follow the explicit fitting, projection, uncertainty and verification order in
[`11_damage_projection/README.md`](11_damage_projection/README.md). A separately fitted
fractional-logit auxiliary model bounds daily predictions between zero and 24 outage
hours per customer. Historical PPML attribution remains a separate analysis.

After all model artifacts and diagnostics are complete:

```bash
export GOES_DATA_ROOT=/path/to/data
export GOES_FIG6_WORK=/path/to/completed/fig6_work
export GOES_FIGURE_DIR=/path/to/figure_output
python 12_figures/07_fig6_damage.py
```

The hottest realization gives approximately **+9.56%** in modeled excess tropical cyclone
outage hours, with a combined 95% interval of **−1.23% to +20.17%**. This is a relative
change, not percentage points of observed outage share. The interval includes zero.
Storm occurrence is replayed and network vulnerability is fixed, so this is not an
unconditional forecast of future outages.

**Do not use `06_r5_dose2.py` or `07_dose6.py` for the submitted result.** Their former
1.74/1.37-percentage-point results are superseded. Keep `09_daily_projection.py` when
packaging the bounded model: although its standalone unbounded projection is not the
submitted method, `project_bounded_outage_2026_10_01.py` imports its design utilities.

## Documented data sources

Data are not redistributed here. Check access conditions and third-party licenses at
download time; this table is a source inventory, not a claim of verified endpoint availability.

| Domain | Source |
|---|---|
| Historical/future weather | [Thermodynamic Global Warming](https://doi.org/10.57931/1885756) |
| Downscaling reference | CONUS404; acquisition endpoints recorded in stage 01 |
| Reanalysis indices | ERA5 vapor transport and geopotential height |
| Tropical cyclones | [HURDAT2](https://www.nhc.noaa.gov/data/#hurdat) |
| Outages | [EAGLE-I](https://doi.org/10.1038/s41597-024-03095-5) |
| Demand | [EIA-930](https://www.eia.gov/electricity/gridmonitor/about), [county-hourly demand](https://doi.org/10.25984/3366592), FERC Form 714 |
| Fleet | EIA-860/EIA-923; [GODEEEP](https://doi.org/10.1038/s41597-024-03894-w) for comparison |
| Capacity/siting | [GCAM-USA](https://doi.org/10.57931/2428940), [CERF](https://doi.org/10.21105/joss.03601), [GRIDCERF](https://doi.org/10.5281/zenodo.10041918), [Cambium](https://doi.org/10.2172/1915250) |
| Investment/mitigation | EIA-861, FERC Form 1 and FEMA awards |
| Population | American Community Survey five-year estimates, 2022 |

Trained checkpoints, model metadata and fitted artifacts need documented distribution
locations before a complete release. Do not assume that every inference script loads
the same checkpoint solely because an earlier README named `G_deploy.pth`.

## License and contact

Code is released under the [MIT License](LICENSE). Third-party data and software retain
their own licenses. Scientific questions should go to the corresponding author listed
in the manuscript; code issues can be reported through the repository issue tracker.
