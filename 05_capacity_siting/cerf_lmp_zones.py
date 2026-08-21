"""Swappable LMP source for CERF LMP-aware siting.

TWO surfaces are available through the SAME public API:

  * STATIC  (Option B)  : the original EIA Wholesale + ISO market-monitor
    (2022-2024) real annual-mean 57-zone surface (lmp_zone_lmp.json).

  * CAMBIUM (NREL Cambium-2024) : hourly `energy_cost_busbar` ($/MWh, 2023$;
    NREL "conceptually similar to a day-ahead LMP") per ReEDS balancing-area
    (p-region), annual-aggregated (flat, wind-gen-weighted, solar-gen-weighted),
    area-mapped to the 57 CERF pricing zones, and delivered as an
    ANCHORED ANOMALY:

        LMP(zone) = ISO_obs_mean[subregion(zone)]
                    + ( Cam(zone) - mean_over_zones_in_subregion(Cam) )

    Each subregion's mean stays pinned to the observed ISO anchor (so absolute
    price levels remain the trusted observed ones), while Cambium supplies the
    WITHIN-subregion spatial spread that actually moves economic siting inside a
    CERF competition region.  With anchor=False the raw Cambium level is returned
    (deflated 2023$ -> study$).

Public API (shape unchanged; new optional kwargs are all keyword-defaulted so
existing callers -- e.g. build_lmp2d() with no args -- behave EXACTLY as before):

  get_zone_lmp(scenario=None, year=None, mode='central', anchor=True, tech=None)
      -> {zone_id:int -> $/MWh}
  build_lmp2d(scenario=None, year=None, mode='central', anchor=True, tech=None)
      -> (H,W) float32 $/MWh on the CERF driver grid.

Argument resolution when an arg is None (lets a later re-site drive the surface
WITHOUT editing the driver -- it just exports env vars per run):
      explicit arg  >  env var  >  built-in default
  env vars: CERF_LMP_SCENARIO, CERF_LMP_YEAR, CERF_LMP_MODE, CERF_LMP_ANCHOR,
            CERF_LMP_TECH.
If the resolved scenario does not map to a Cambium scenario, or mode='static',
or any Cambium data file is missing, the STATIC Option-B surface is returned ->
current behaviour is preserved when nothing is set.

scenario -> Cambium mapping (cooler/hotter SHARE the mapping: ECS is weather,
already captured in the 4km CF; do NOT double-count it through price):
      strip 'cooler'/'hotter'  ->  rcp{45,85}_ssp{3,5}  key
"""
import os, re, json, functools
import numpy as np, rasterio

OUT = "/data/cerf_out"
CAM = f"{OUT}/lmp_cambium"
LMP_ZONES_TIF = f"{OUT}/lmp_zones_aligned.tif"
LMP_JSON = f"{OUT}/lmp_zone_lmp.json"
PARQUET = f"{CAM}/cambium_ecbusbar_annual.parquet"
WEIGHTS = f"{CAM}/zone_pregion_weights.csv"
ZSUB = f"{CAM}/zone_subregion.json"

# --- scenario -> (mode -> Cambium scenario) --------------------------------
SCEN_MAP = {
    "rcp45_ssp3": {"central": "LowRECost",
                   "low": "MidCase",              "high": "LowRECost_HighNGPrice"},
    "rcp45_ssp5": {"central": "LowRECost_HighNGPrice",
                   "low": "LowRECost",            "high": "HighDemandGrowth"},
    "rcp85_ssp3": {"central": "HighRECost_LowNGPrice",
                   "low": "LowNGPrice",           "high": "HighRECost"},
    "rcp85_ssp5": {"central": "HighRECost",
                   "low": "HighRECost_LowNGPrice", "high": "HighDemandGrowth"},
}
SOLVE_YEARS = [2030, 2035, 2040, 2045, 2050]
# Cambium is 2023$; study $ ~ 2023$ (GDP deflator ~1.0). Adjust here if the
# study year is re-based; anchored mode is deflator-insensitive by construction.
DEFLATOR_2023_TO_STUDY = 1.0
TECH_COL = {None: "ecb_mean", "": "ecb_mean", "flat": "ecb_mean", "mean": "ecb_mean",
            "wind": "ecb_wind_wtd", "wind_onshore": "ecb_wind_wtd",
            "solar": "ecb_solar_wtd", "solar_pv_non_dist": "ecb_solar_wtd", "pv": "ecb_solar_wtd"}


# --- key / year resolution --------------------------------------------------
def rcpxssp(scenario):
    """'rcp45hotter_ssp5' -> 'rcp45_ssp5'  (None if it doesn't parse)."""
    if not scenario:
        return None
    s = str(scenario).lower()
    m = re.search(r"rcp(45|85)", s)
    n = re.search(r"ssp([35])", s)
    if not (m and n):
        return None
    return f"rcp{m.group(1)}_ssp{n.group(1)}"


def year_map(year):
    """Nearest Cambium solve year; floor to 2030 for <2030, cap 2050."""
    if year is None:
        return 2050
    y = int(year)
    if y < 2030:
        return 2030
    if y > 2050:
        return 2050
    return min(SOLVE_YEARS, key=lambda s: (abs(s - y), s))


# --- cached loaders ---------------------------------------------------------
@functools.lru_cache(1)
def _static_json():
    return json.load(open(LMP_JSON))


def _static_zone_lmp():
    return {int(k): float(v) for k, v in _static_json()["zone_lmp"].items()}


def _static_national_median():
    return float(_static_json().get("national_median", 35.0))


@functools.lru_cache(1)
def _base_iso():
    return {k: float(v) for k, v in _static_json()["base_iso"].items()}


@functools.lru_cache(1)
def _weights():
    """{zone:int -> [(r, area_weight), ...]}"""
    import csv
    d = {}
    with open(WEIGHTS) as f:
        for row in csv.DictReader(f):
            z = int(row["cerf_zone"])
            d.setdefault(z, []).append((row["r"], float(row["area_weight"])))
    return d


@functools.lru_cache(1)
def _zsub():
    """{zone:int -> subregion(gea) str}"""
    j = json.load(open(ZSUB))
    return {int(k): v["subregion"] for k, v in j.items()}


@functools.lru_cache(1)
def _cam_df():
    import pandas as pd
    return pd.read_parquet(PARQUET)


@functools.lru_cache(None)
def _ecb_map(cam_scen, cam_yr, col):
    """{r -> $/MWh} for one (scenario, year); NaN gen-weighted -> flat ecb_mean."""
    df = _cam_df()
    sub = df[(df.scenario == cam_scen) & (df.year == cam_yr)]
    flat = {row.r: float(row.ecb_mean) for row in sub.itertuples()}
    if col == "ecb_mean":
        return flat
    out = {}
    for row in sub.itertuples():
        v = float(getattr(row, col))
        out[row.r] = v if v == v else flat[row.r]   # NaN -> flat
    return out


# --- env-var resolution -----------------------------------------------------
def _resolve(scenario, year, mode, anchor, tech):
    if scenario is None:
        scenario = os.environ.get("CERF_LMP_SCENARIO")
    if year is None:
        ye = os.environ.get("CERF_LMP_YEAR")
        year = int(ye) if ye not in (None, "") else None
    if mode is None:
        mode = os.environ.get("CERF_LMP_MODE") or "central"
    if anchor is None:
        ae = os.environ.get("CERF_LMP_ANCHOR")
        anchor = True if ae is None else (str(ae).lower() not in ("0", "false", "no", "off"))
    if tech is None:
        tech = os.environ.get("CERF_LMP_TECH")
    return scenario, year, mode, anchor, tech


# --- public API -------------------------------------------------------------
def get_zone_lmp(scenario=None, year=None, mode='central', anchor=True, tech=None):
    """Return {zone_id:int -> annual-mean LMP $/MWh}.

    mode : 'central' | 'low' | 'high' (Cambium bracket) | 'static' (Option-B).
    anchor : True -> ISO-anchored within-subregion anomaly; False -> raw Cambium.
    tech : None/'flat' | 'wind' | 'solar' -> flat / wind-wtd / solar-wtd column.
    Falls back to the static Option-B surface for any unmapped scenario or on
    any error, so callers never break.
    """
    scenario, year, mode, anchor, tech = _resolve(scenario, year, mode, anchor, tech)
    key = rcpxssp(scenario)
    if mode == "static" or key is None or key not in SCEN_MAP:
        return _static_zone_lmp()
    try:
        cam_scen = SCEN_MAP[key][mode]
        cam_yr = year_map(year)
        col = TECH_COL.get((tech or "").lower(), "ecb_mean")
        ecb = _ecb_map(cam_scen, cam_yr, col)
        weights = _weights()
        zsub = _zsub()

        cam = {}
        for z, lst in weights.items():
            num = 0.0; den = 0.0
            for r, w in lst:
                v = ecb.get(r)
                if v is not None and v == v:
                    num += w * v; den += w
            cam[z] = (num / den) if den > 0 else float("nan")

        if not anchor:
            return {int(z): float(v * DEFLATOR_2023_TO_STUDY)
                    for z, v in cam.items() if v == v}

        base = _base_iso()
        nat = _static_national_median()
        groups = {}
        for z in cam:
            groups.setdefault(zsub.get(z), []).append(z)
        out = {}
        for sub, zs in groups.items():
            vals = [cam[z] for z in zs if cam[z] == cam[z]]
            gmean = (sum(vals) / len(vals)) if vals else 0.0
            anchorval = base.get(sub, nat)
            for z in zs:
                c = cam[z]
                out[int(z)] = float(anchorval + (c - gmean)) if c == c else float(anchorval)
        return out
    except Exception as e:  # pragma: no cover - safety net for the drivers
        import sys
        print(f"cerf_lmp_zones: Cambium path failed ({e!r}) -> static Option-B", file=sys.stderr)
        return _static_zone_lmp()


def build_lmp2d(scenario=None, year=None, mode='central', anchor=True, tech=None):
    """(H,W) float32 $/MWh LMP surface on the CERF driver grid.
    Cells whose zone is unmapped or nodata (255) -> national median."""
    z = rasterio.open(LMP_ZONES_TIF).read(1)
    zl = get_zone_lmp(scenario, year, mode, anchor, tech)
    nat = float(np.median(list(zl.values()))) if zl else _static_national_median()
    a = np.full(z.shape, nat, "f4")
    for k, v in zl.items():
        a[z == k] = v
    return a
