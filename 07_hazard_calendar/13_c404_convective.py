"""
County-day SEVERE CONVECTIVE flag from CONUS404, 2015-01-01 .. 2022-09-30.

Severe convective storms are almost certainly the largest weather cause of US distribution
outages - NOAA's own attribution puts them at 8.8% of all customer-hours, second only to tropical
cyclones at 12.8% - and until now this project had no county-resolved flag for them at all. The
"only tropical cyclones matter" result is partly a statement about which hazards were built.

WHY NOT NOAA StormEvents AS THE TREATMENT. It has county FIPS and gust magnitudes and is on the
box, but it is a record of human reports: populous counties generate more of them. Population
density is correlated with grid form and with every demographic of interest, so using report
counts as the treatment would write population straight into the "hazard". StormEvents is used
here only to VALIDATE the physical flag.

DEFINITION, absolute and physical, so no climatology is needed and none of the threshold-freezing
or unit hazards of the temperature-based flags apply:

    severe convective county-day  =  a cell-hour with REFL_COM >= 50 dBZ
                                     AND daily max MUCAPE >= 1000 J/kg
                                     AND at least 20% of the county's cells above 50 dBZ in that
                                     same hour

All three conditions are applied HERE and written out as the boolean column `severe`. Consumers
read that column; they must not rebuild the flag from the intensity columns. The areal-fraction
condition used to live in every consumer separately rather than in the builder: nine files in
09_outage_attribution each carried their own `frac50 >= 0.20` line, so the hazard was defined in nine places
and any consumer that forgot the line would silently have read a wider hazard under the same name.
They agreed with each other on 2026-08-18; the defect was that nothing made them agree.

The CAPE condition excludes non-convective high reflectivity (bright-band melting snow, which
routinely exceeds 50 dBZ in winter). The area condition excludes a single cell clipping the county
corner. Intensity is stored three ways because the max saturates: areal fraction of the county
above 50 dBZ, the county maximum, and hours above threshold.

Aggregation is MAX over the county's cells, not mean: the question is whether a severe cell
crossed the county, and a mean over ~146 cells would erase it.

CONSTANTS. The three thresholds are read from 07_hazard_calendar/hazard_defs.py
(CONV_REFL_DBZ, CONV_CAPE_JKG, CONV_CELL_FRACTION). This file holds no private copy of them.

MERGE. Run this file with the single argument `merge` after every shard has finished. It checks
each shard's hazard_defs stamp, concatenates them and writes the merged
/data/enso/county_convective_daily.parquet, itself stamped. Before 2026-08-18 no merge step
existed anywhere in the repository, so the merged file that every consumer reads had no traceable
producer at all.

"""
import glob, os, sys, time
for _p in (os.path.dirname(os.path.abspath(__file__)),
           os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "07_hazard_calendar")):
    if os.path.isdir(_p) and _p not in sys.path:
        sys.path.insert(0, _p)
import hazard_defs as HD
import numpy as np, pandas as pd

OUT = "/data/enso/county_convective"; os.makedirs(OUT, exist_ok=True)
MERGED = "/data/enso/county_convective_daily.parquet"
T0, T1 = "2015-01-01", "2022-09-30"
REFL_THR, CAPE_THR, FRAC_THR = HD.CONV_REFL_DBZ, HD.CONV_CAPE_JKG, HD.CONV_CELL_FRACTION

# ------------------------------------------------------------------ merge mode
# `python 13_c404_convective.py merge` gathers the shards this same script wrote, refuses any shard
# whose stamp does not match the current convection constants, and writes the merged file the
# consumers in 09_outage_attribution read. The merged file carries its own stamp, so the refusal reaches them
# too.
if len(sys.argv) > 1 and sys.argv[1] == "merge":
    fs = sorted(glob.glob("%s/conv_shard*.parquet" % OUT))
    if not fs:
        raise SystemExit("no shards in %s; run the shards first" % OUT)
    parts = []
    for f in fs:
        HD.require_stamp(f, hazards=["convection"], script=__file__)
        parts.append(pd.read_parquet(f))
    M = pd.concat(parts, ignore_index=True)
    M["date"] = pd.to_datetime(M.date).dt.strftime("%Y-%m-%d")
    dup = int(M.duplicated(["fips", "date"]).sum())
    if dup:
        raise SystemExit("%d duplicated county-days across shards; the shards overlap" % dup)
    M = M.sort_values(["fips", "date"]).reset_index(drop=True)
    NC_, ND_ = int(M.fips.nunique()), int(M.date.nunique())
    HD.write_flags(M, MERGED, script=__file__, n_units=NC_, n_dates=ND_,
                   hazards=["convection"], flag_col="severe",
                   extra={"shards": len(fs), "window": [T0, T1]})
    rate = float(np.asarray(M.severe, bool).sum()) / float(NC_ * ND_)
    ok, e, t, b = HD.day_rate_ok("county", "convection", rate)
    print("merged %d shards -> %s   %s rows, %d counties x %d dates"
          % (len(fs), MERGED, format(len(M), ","), NC_, ND_))
    print("   severe day rate %.4f; recorded band %.4f +/- %.4f [%s] -> %s"
          % (rate, e, t, b.split(":")[0], "in band" if ok else "OUT OF BAND"))
    raise SystemExit(0)

from scipy.spatial import cKDTree
import warnings; warnings.filterwarnings("ignore")
import pystac_client, planetary_computer as pc, xarray as xr

si, sn = map(int, (sys.argv[1] if len(sys.argv) > 1 else "0/1").split("/"))

G = "/data/datasets/grid"
nn = np.load(f"{G}/c404_to_fine_nn.npz")
idx = nn["idx"].astype(np.int64)
SRC = tuple(nn["src_shape"]); DST = tuple(nn["dst_shape"])
zc = np.load(f"{G}/coordinate.npz")
flat_lat = np.repeat(zc["lat"].astype(float), DST[1])
flat_lon = np.tile(zc["lon"].astype(float), DST[0])
nsrc = SRC[0] * SRC[1]
cnt = np.bincount(idx, minlength=nsrc).astype(float)
ok = cnt > 0
c_lat = np.where(ok, np.bincount(idx, flat_lat, nsrc) / np.maximum(cnt, 1), np.nan)
c_lon = np.where(ok, np.bincount(idx, flat_lon, nsrc) / np.maximum(cnt, 1), np.nan)
tg = np.load("/data/tgw_hist/tgw_grid.npz")
cm = np.load("/data/loads_measured/county_mask_tgw.npz", allow_pickle=True)
fips = np.array([str(f).zfill(5) for f in cm["fips"]])
cell2cty = np.full(tg["XLAT"].size, -1, np.int32)
cell2cty[cm["pair_cell"].astype(np.int64)] = cm["pair_fips"].astype(np.int32)
sel = np.where(ok)[0]
_, j = cKDTree(np.c_[tg["XLAT"].ravel(), tg["XLONG"].ravel()]).query(np.c_[c_lat[sel], c_lon[sel]])
src_cty = np.full(nsrc, -1, np.int32); src_cty[sel] = cell2cty[j]
have = src_cty >= 0
order = np.argsort(src_cty[have], kind="stable")
src_idx = np.where(have)[0][order]
cty_of = src_cty[have][order]
lo = np.searchsorted(cty_of, np.arange(len(fips)), "left")
hi = np.searchsorted(cty_of, np.arange(len(fips)), "right")
keep = np.where(hi > lo)[0]
starts = lo[keep]; ncell = (hi - lo)[keep].astype(np.float32); fips_keep = fips[keep]
ends = hi[keep]
if si == 0:
    print("counties %d   cells/county mean %.1f" % (len(keep), ncell.mean()), flush=True)

cat = pystac_client.Client.open("https://planetarycomputer.microsoft.com/api/stac/v1",
                                modifier=pc.sign_inplace)
a = cat.get_collection("conus404").assets["zarr-abfs"]
ds = xr.open_zarr(a.href, storage_options=a.extra_fields["xarray:storage_options"],
                  **a.extra_fields["xarray:open_kwargs"])
VARS = ["REFL_COM", "MUCAPE"]

days = pd.date_range(T0, T1)
days = days[np.arange(len(days)) % sn == si]
print("shard %d/%d : %d days" % (si, sn, len(days)), flush=True)

# segment boundaries for per-county reductions over the sorted cell list
seg = np.repeat(np.arange(len(keep)), (ends - starts))
rows, bad = [], []
t0 = time.time()
for k, d in enumerate(days):
    try:
        sl = ds[VARS].sel(time=slice(d, d + pd.Timedelta("23h"))).load()
    except Exception as ex:
        bad.append((str(d.date()), repr(ex)[:100])); continue
    nh = sl.sizes["time"]
    R = sl["REFL_COM"].values.reshape(nh, -1)[:, src_idx]        # (nh, ncells_kept)
    C = sl["MUCAPE"].values.reshape(nh, -1)[:, src_idx]
    hit = (R >= REFL_THR)
    # per hour, per county: fraction of cells above threshold, and the max
    frac_h = np.zeros((nh, len(keep)), "f4")
    np.add.at(frac_h.T, seg, hit.T.astype("f4"))
    frac_h /= ncell[None, :]
    max_h = np.full((nh, len(keep)), -1e9, "f4")
    np.maximum.at(max_h.T, seg, R.T)
    cape_h = np.full((nh, len(keep)), -1e9, "f4")
    np.maximum.at(cape_h.T, seg, C.T)
    refl_max = max_h.max(0); frac50 = frac_h.max(0); cape_max = cape_h.max(0)
    rows.append(pd.DataFrame({
        "date": str(d.date()), "fips": fips_keep,
        "refl_max": refl_max, "frac50": frac50,
        "cape_max": cape_max, "hours50": (frac_h > 0).sum(0).astype("f4"),
        # the flag itself, all three conditions, so no consumer has to add one of its own
        "severe": (refl_max >= REFL_THR) & (cape_max >= CAPE_THR) & (frac50 >= FRAC_THR)}))
    del sl, R, C, hit, frac_h, max_h, cape_h
    if k % 25 == 0:
        el = time.time() - t0
        print("shard%d %d/%d  %.1f s/day  eta %.0f min"
              % (si, k, len(days), el / max(k, 1), (len(days) - k) * el / max(k, 1) / 60), flush=True)

res = pd.concat(rows, ignore_index=True)
# Stamped at the shard, so the merge step above can refuse a shard built from other constants.
HD.write_flags(res, f"{OUT}/conv_shard{si}.parquet", script=__file__,
               n_units=int(res.fips.nunique()), n_dates=int(res.date.nunique()),
               hazards=["convection"], flag_col="severe",
               extra={"shard": si, "of": sn, "window": [T0, T1]})
if bad:
    pd.DataFrame(bad, columns=["date", "error"]).to_csv(f"{OUT}/bad_shard{si}.csv", index=False)
print("shard %d DONE: %d days (%d failed) -> %s rows, severe %s (%.2f%%)"
      % (si, len(days), len(bad), format(len(res), ","), format(int(res.severe.sum()), ","),
         100 * float(res.severe.mean())), flush=True)
