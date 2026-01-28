# Climate–Energy Extremes Analysis Codebase

This repository contains the code used in the manuscript:

> *Climate change drives imbalance spikes, generation droughts, and chronic seasonal synchronization in renewable power systems*.

The study quantifies **multi-timescale energy-system extremes** driven by physical climate variability and change, using data-driven methods and generative learning models.


---

## How to cite

If you use this repository, please cite the accompanying manuscript (Currently Under Review in Science Series Journals) and include a link to this repository and the commit hash (or a tagged release).

---

## Overview

A conceptual illustration of the study workflow:

![Figure 1](figs/figure1.png)

---

## System requirements

### Operating systems
Tested on:
- Linux (Ubuntu)
- macOS

Windows should also work for most analyses, but paths may require minor adjustments.

### Software
- **Python** 3.10+ recommended
- **JupyterLab** (recommended) or Jupyter Notebook

### Typical Python dependencies
Exact dependencies depend on which modules you run (plotting, ML training, geospatial, etc.). The code uses standard scientific Python tooling such as:
- numpy, pandas, scipy, xarray, netCDF4 / h5py
- matplotlib (and/or plotly) for figures
- scikit-learn
- torch (PyTorch) for `train_srgan/` and `train_unet/`
- tqdm

> **Recommended for reproducibility:** add a pinned environment file at the repo root (e.g., `requirements.txt` or `environment.yml`) and record the OS + Python version used for the final paper runs.

### Hardware
- **No GPU is required** to run most analysis + plotting code once inputs are prepared.
- **GPU recommended** only if you re-train SRGAN / U-Net from scratch.
- For full-scale analysis on the 1 TB dataset, we recommend ≥16–32 GB RAM and sufficient local storage.

---

## Data

### Google Drive bundle (required for full reproduction)
The full analysis depends on a **Google Drive dataset (~1 TB)**:  
https://drive.google.com/drive/folders/1nR9cPL55tvpurUy_4ExEhLzU9bbjwgeF?usp=sharing

It includes:
- 2018–2022 hourly data (renewable generation, load, meteorology, hydrology)
- County-level hourly loads (2018)
- Future weather projections (CMIP6-based inputs)
- AI model training checkpoints (where applicable)

### Recommended local layout
After download, place the Drive contents into a single directory and point the code to it via an environment variable:

```bash
export CLIMATE_EXTREMES_DATA="/path/to/drive_bundle"
```

If a module/script expects a specific subfolder layout, document it in that module’s header or in a short `CONFIG.md`.

### Data sources
Upstream datasets and preprocessing are described in the paper and Supplementary Information (SI). Some inputs were obtained using:
- Herbie (HRRR reanalysis): https://github.com/blaylockbk/Herbie
- AWS CLI (AWS archives): https://aws.amazon.com/cli/
- PyDrive (Google Drive utilities): https://pythonhosted.org/PyDrive/

> **Third-party terms:** Some raw inputs may have separate terms-of-use. Users are responsible for complying with upstream dataset licenses/requirements.

---

## Running the code

### Quick start: generate paper/SI figures
1. Install dependencies (see above).
2. Download the required subset (or full) data bundle and set `CLIMATE_EXTREMES_DATA`.
3. Launch Jupyter:
```bash
jupyter lab
```
4. Run the notebooks/scripts in `plot_examples/`.

**Expected outputs**
- Figures are typically written to `figs/` (or a script-specific output directory).
- Intermediate tables/arrays may be written alongside each module or under a dedicated outputs folder.

**Typical runtime (ballpark)**
- Figure generation using prepared inputs: minutes to ~1 hour per full figure set, depending on I/O and compute.
- Model training from scratch (SRGAN/U-Net): hours to days (GPU-dependent).

### Re-training models (optional)
- `train_srgan/`: weather super-resolution SRGAN training scripts/configs
- `train_unet/`: U-Net training for renewable prediction tasks
- `model/`: shared architectures and loss functions

If you only need to reproduce the paper’s plots/results, you typically do **not** need to re-train models if checkpoints are provided in the data bundle.

---

## Repository structure

| Folder | Description |
|---|---|
| `county/` | County-level metadata (region mappings, FIPS codes, etc.) |
| `figs/` | Figures used in the paper (and default output location for plots) |
| `future/` | Code for future energy/climate scenario projections (e.g., 2030–2050) |
| `hydro/` | Hydropower-related analysis (including SI components) |
| `load/` | Load data processing (commercial/residential/industrial/transport) |
| `model/` | Shared model architectures and loss functions |
| `plot_examples/` | Scripts/notebooks to generate main-text and SI figures |
| `train_srgan/` | SRGAN training scripts/configs |
| `train_unet/` | U-Net model training scripts/configs |
| `utils/` | Shared utilities (loading, normalization, evaluation, plotting helpers) |

---

## How to cite

Please cite the accompanying manuscript. For software citation, include:
- the GitHub repository URL, and
- the specific commit hash (or a tagged release)

---

## Contact

For questions, reproducibility issues, or missing-path documentation, please open a GitHub issue in this repository.
