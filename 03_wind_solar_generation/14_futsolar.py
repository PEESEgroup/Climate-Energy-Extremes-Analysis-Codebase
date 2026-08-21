"""Future solar at hourly cadence on the native 12 km grid.

The production path sampled irradiance from the 3-hourly 3D archive, so two thirds of the hourly
grid PVWatts is given were left empty. SWDOWN is one of the 25 hourly TGW variables and is already
on disk per scenario, so the fix is to read the hourly archive instead. Grid is unchanged at 12 km
native: solar was tested with and without downscaling and did not benefit.

Everything after the read is identical to the production solar phase: same plants, same PVWatts
configuration, same per-year 8760 embedding that drops 29 February.

ENV: CLIMATE, OUT, YEARS (default 2030-2050), NPROC
"""
import glob
import os
import sys
import time

import numpy as np
import pandas as pd

import os as _os_rp

for _rp in ("04_demand_model", "07_hazard_calendar", "09_outage_attribution",

            "02_downscale_wind", "12_figures"):

    _ap = _os_rp.path.abspath(_os_rp.path.join(

        _os_rp.path.dirname(_os_rp.path.abspath(__file__)), "..", _rp))

    if _os_rp.path.isdir(_ap) and _ap not in sys.path:

        sys.path.insert(0, _ap)
sys.path.insert(0, "/data/gen_targets/srgan3d_val")
from gen_physics import solar_cf
import multiprocessing as mp

CLIMATE = os.environ["CLIMATE"]
OUTF = os.environ["OUT"]
NPROC = int(os.environ.get("NPROC", "6"))
y0, y1 = [int(x) for x in os.environ.get("YEARS", "2030-2050").split("-")]
GRID = "/data/datasets/grid"
GC = "/data/datasets/gen/tgw-gen-historical"

# ---------------------------------------------------------------- sites, native 12 km cell
# one WRF grid serves every scenario and the historical run, so any scenario copy is equivalent
g = np.load(os.environ.get("GRIDNPZ", "/data/tgw_extract/rcp45cooler/tgw_grid.npz"))
XLAT = g["XLAT"].astype(float); XLONG = g["XLONG"].astype(float)
Xf = XLAT.ravel(); Yf = XLONG.ravel()
CERF = "/data/cerf_out"
VARIANTS = {"nopolicy": f"{CERF}/fleet_{{scenario}}.csv",
            "policy":   f"{CERF}/fleet_policy_{{scenario}}.csv",
            "obbba":    f"{CERF}/fleet_obbba_{{scenario}}.csv",
            "ordonly":  f"{CERF}/fleet_policy_{{scenario}}_ordonly.csv"}
SCENS = {c: [f"{c}_ssp3", f"{c}_ssp5"] for c in
         ["rcp45cooler", "rcp45hotter", "rcp85cooler", "rcp85hotter"]}
_co = np.load("/data/datasets/grid/coordinate.npz"); _LAT = _co["lat"]; _LON = _co["lon"]

def _load_fleet(scenario, tmpl):
    d = pd.read_csv(tmpl.format(scenario=scenario))
    d = d[(d.sited_year <= 2050) & (d.tech == "solar")]
    d = d[(d.retirement_year.isna()) | (d.retirement_year > 2050)]
    d = d[np.isfinite(d.lon) & np.isfinite(d.lat)]
    return d[(d.lat >= _LAT.min()) & (d.lat <= _LAT.max()) &
             (d.lon >= _LON.min()) & (d.lon <= _LON.max())]

scfg_all = pd.read_csv(f"{GC}/eia_solar_configs.csv", dtype={"plant_code_unique": str}).drop_duplicates("plant_code_unique")
scfg_all = scfg_all[np.isfinite(scfg_all.lat) & np.isfinite(scfg_all.lon)].reset_index(drop=True)

if CLIMATE == "historical":
    # fixed present fleet: the plants that exist today, so any change comes from weather alone
    scfg = scfg_all.copy()
    site_lat = scfg.lat.values.astype(float); site_lon = scfg.lon.values.astype(float)
    cfg_pcu = scfg.plant_code_unique.values.astype(str)
else:
    # THE FUTURE FLEET IS SITED BY CERF, NOT BY WHAT EXISTS TODAY. Wind already used this union;
    # taking today's plant list here would have dropped 98.8% of the future solar fleet at
    # aggregation, silently, because the aggregator matches sites by coordinate.
    keys = {}
    for v, tmpl in VARIANTS.items():
        if not all(os.path.exists(tmpl.format(scenario=sc)) for sc in SCENS[CLIMATE]):
            continue
        for sc in SCENS[CLIMATE]:
            d = _load_fleet(sc, tmpl)
            for lo, la in zip(d.lon.round(6), d.lat.round(6)):
                keys[(float(lo), float(la))] = None
    site_lon = np.array([k[0] for k in keys]); site_lat = np.array([k[1] for k in keys])
    assert len(site_lon) > 0, f"no CERF solar sites for {CLIMATE}"
    from scipy.spatial import cKDTree
    _tree = cKDTree(np.column_stack([scfg_all.lat.values, scfg_all.lon.values]))
    _, _idx = _tree.query(np.column_stack([site_lat, site_lon]))
    scfg = scfg_all.iloc[_idx].reset_index(drop=True)      # nearest existing plant's configuration
    cfg_pcu = scfg.plant_code_unique.values.astype(str)

nsite = len(site_lat)
natc = np.array([np.argmin((Xf - la) ** 2 + (Yf - lo) ** 2)
                 for la, lo in zip(site_lat, site_lon)])
print(f"[fs] {CLIMATE}: {nsite} solar sites on the native 12 km grid", flush=True)

# ---------------------------------------------------------------- hourly surface archive
# The historical hourly surface archive lives in a different directory from the per-scenario future
# extracts, but the file schema is identical, so only the directory changes.
SFCDIR = os.environ.get("SFCDIR", f"/data/tgw_extract/{CLIMATE}")
fs = [f for f in sorted(glob.glob(f"{SFCDIR}/*.npz")) if "grid" not in f]
assert fs, f"no surface files under {SFCDIR}"
VARS = ["U10", "V10", "PSFC", "T2", "SWDOWN"]
stamps = []; cols = {k: [] for k in VARS}
t0 = time.time()
for fi, f in enumerate(fs):
    z = np.load(f, allow_pickle=True)
    tm = [str(t) for t in z["times"]]
    keep = [i for i, s in enumerate(tm) if y0 <= int(s[:4]) <= y1]
    if not keep:
        z.close(); continue
    v = [str(x) for x in z["vars"]]; sc = z["scale"]; d = z["data"]
    for k in VARS:
        j = v.index(k)
        a = np.asarray(d[keep, j], np.float32)
        # 'scale' is the multiplier applied BEFORE the fp16 cast, so physical units are recovered by
        # DIVIDING, not multiplying. Only PSFC carries a non-unit entry: it is stored in hPa because
        # pascals overflow float16, and PVWatts wants millibar, so the stored value is already the
        # unit required and must be left alone. Multiplying it gave 9.6 mb and cost 10.6% of output.
        if k != "PSFC":
            a = a / float(sc[j])
        cols[k].append(a.reshape(len(keep), -1)[:, natc])
    stamps += [tm[i] for i in keep]
    z.close()
    if fi % 100 == 0:
        print(f"  [fs] read {fi}/{len(fs)} files {time.time()-t0:.0f}s", flush=True)
stamps = np.array(stamps)
o = np.argsort(stamps); stamps = stamps[o]
# pop each variable's chunk list as it is concatenated. Holding all five lists AND all four
# assembled arrays at once peaks near 96 GB at the 13k-site future fleets; popping keeps it near 50.
SW = np.concatenate(cols.pop("SWDOWN"))[o].T
T2 = np.concatenate(cols.pop("T2"))[o].T
PS = np.concatenate(cols.pop("PSFC"))[o].T
_u = np.concatenate(cols.pop("U10"))[o].T
_v = np.concatenate(cols.pop("V10"))[o].T
WS = np.hypot(_u, _v); del _u, _v
NH = len(stamps)
assert 500.0 < np.nanmean(PS) < 1100.0, f"PSFC is not in millibar: mean {np.nanmean(PS):.2f}"
assert 200.0 < np.nanmean(T2) < 330.0, f"T2 is not in kelvin: mean {np.nanmean(T2):.2f}"
print(f"[fs] {NH} hourly stamps {stamps[0]} -> {stamps[-1]}  SWDOWN mean {np.nanmean(SW):.1f} W/m2  "
      f"PSFC mean {np.nanmean(PS):.1f} mb  T2 mean {np.nanmean(T2):.1f} K", flush=True)
del cols

# ---------------------------------------------------------------- PySAM, unchanged from production
years = np.array([int(s[:4]) for s in stamps])
year_grids = {}
for yr in range(y0, y1 + 1):
    FULL = pd.date_range(f"{yr}-01-01", f"{yr}-12-31 23:00", freq="h")
    FULL = FULL[~((FULL.month == 2) & (FULL.day == 29))]
    year_grids[yr] = (FULL, {t.strftime("%Y%m%d%H"): k for k, t in enumerate(FULL)})
# Spill the driver arrays to disk BEFORE forking the pool. Held in anonymous memory they are copied
# on write page by page as each PySAM worker runs: with 14 workers this added ~35 GB on top of the
# parent's arrays and drove available memory from 121 GB to 3 GB. Disk-backed arrays are shared
# through the page cache instead, which the kernel can reclaim, so the pool costs almost nothing.
import gc
SPILL = os.environ.get("SPILL", "/data/tmp_spill")
os.makedirs(SPILL, exist_ok=True)
_paths = {}
def _spill(nm, arr):
    q = f"{SPILL}/{CLIMATE}_{nm}.npy"
    np.save(q, arr); _paths[nm] = q
    return q
_q = {nm: _spill(nm, a) for nm, a in (("sw", SW), ("t2", T2), ("ps", PS), ("ws", WS))}
SW = T2 = PS = WS = None
gc.collect()
SW = np.load(_q["sw"], mmap_mode="r"); T2 = np.load(_q["t2"], mmap_mode="r")
PS = np.load(_q["ps"], mmap_mode="r"); WS = np.load(_q["ws"], mmap_mode="r")
print(f"[fs] spilled drivers to {SPILL}; parent now holds only the output array", flush=True)

cfgpar = scfg_all.set_index("plant_code_unique")
_G = {"sw": SW, "t2": T2, "ws": WS, "ps": PS}

def one(pi):
    c = cfgpar.loc[cfg_pcu[pi]]
    gcfg = dict(system_capacity=float(c.system_capacity), losses=float(c.losses),
                array_type=int(c.array_type), module_type=int(c.module_type),
                azimuth=float(c.azimuth), tilt=float(c.tilt))
    out = np.full(NH, np.nan, np.float32)
    for yr in range(y0, y1 + 1):
        m = np.where(years == yr)[0]
        if len(m) == 0: continue
        FULL, hkey = year_grids[yr]; H = len(FULL)
        loc = np.array([hkey.get(stamps[k], -1) for k in m])
        keep = loc >= 0
        mk = m[keep]; hidx = loc[keep]
        if len(mk) == 0: continue
        ghi = np.zeros(H); ghi[hidx] = np.clip(np.nan_to_num(_G["sw"][pi, mk]), 0, 1400)
        t2c = np.full(H, 15.0); t2c[hidx] = np.nan_to_num(_G["t2"][pi, mk], nan=288.0) - 273.15
        wsp = np.full(H, 2.0); wsp[hidx] = np.nan_to_num(_G["ws"][pi, mk], nan=2.0)
        prs = np.full(H, 1013.0); prs[hidx] = np.nan_to_num(_G["ps"][pi, mk], nan=1013.0)
        try:
            cf = solar_cf(FULL, ghi, t2c, wsp, prs, float(site_lat[pi]), float(site_lon[pi]), gcfg)
            out[mk] = np.asarray(cf, np.float32)[hidx]
        except Exception as e:
            if pi == 0: print('  [fs] solar_cf failed:', type(e).__name__, str(e)[:200], flush=True)
    return pi, out

# Worker recycling interval is a MEMORY control here, not a tidiness setting. The photovoltaic
# model leaks a few megabytes per instance, and one task is one plant, which is 21 model runs. At 100
# tasks per child each worker accumulated about 8 GB of private memory, measured as Private_Dirty in
# smaps, and twelve workers took the host from 121 GB free to 19 GB. At 8 tasks per child the same
# workers hold well under 1 GB each. Forking again is cheap because the parent's arrays are shared.
CF = np.full((nsite, NH), np.nan, np.float32)
t0 = time.time()
with mp.get_context("fork").Pool(NPROC, maxtasksperchild=8) as pool:
    for k, (pi, cf) in enumerate(pool.imap_unordered(one, range(nsite), chunksize=16)):
        CF[pi] = cf
        if k % 500 == 0:
            print(f"  [fs] pysam {k}/{nsite} {time.time()-t0:.0f}s", flush=True)

np.savez(OUTF, cf=CF, stamps=stamps, plants=cfg_pcu,
         lat=site_lat.astype(np.float32), lon=site_lon.astype(np.float32),
         climate=CLIMATE, years=f"{y0}-{y1}",
         note="hourly plant solar CF; irradiance from the hourly 12 km TGW archive, native grid, "
              "no spatial downscaling (tested, no benefit); PVWatts identical to the production phase")
for _pp in _paths.values():
    try: os.remove(_pp)
    except OSError: pass
print(f"[fs] DONE  CF mean {np.nanmean(CF):.4f}  nan {100*np.isnan(CF).mean():.2f}%  -> {OUTF}", flush=True)
