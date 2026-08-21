"""
Task #63, step 1: rebuild county daily weather on CONUS404 instead of TGW.

WHY THE SOURCE HAS TO CHANGE, not just be extended. The hazard calendar is currently thresholded
on /data/enso/subregion_weather_daily.csv, which 02_subregion_weather2.py builds from
/data/tgw_hist/. TGW-hist is a historical *simulation* and stops at 2019, so it cannot be pushed
to 2023 at all. The project's own data-source rule already says historical analysis runs on
CONUS404 (1979-2024) and TGW-hist is for SRGAN bias work only, so the calendar was on the wrong
source to begin with. Rebuilding on CONUS404 fixes the policy violation and unlocks 2020-2023 in
one move, with thresholds still frozen on 1980-2019 so nothing downstream is recalibrated.

GRID CHAIN. CONUS404 native is 1015 x 1367; the real county mask lives on the TGW 299 x 424 grid;
the two are bridged by the 625 x 1475 fine grid, for which coordinate.npz gives lat/lon and
c404_to_fine_nn.npz gives fine -> c404 nearest-neighbour indices. So: invert that map to get the
fine cells behind each c404 cell, take their mean lat/lon, find the nearest TGW cell, and read the
county from county_mask_tgw. County polygons are therefore inherited from the real mask rather
than re-derived by Voronoi.

Channel order in c404_native is [u10, v10, t2, q2, psfc, snow, short, long, precip]. tmax = max
and tmin = min as in the TGW build; the rest are a plain mean over the hours the file holds, which
equals the TGW build's hour-count-weighted mean because c404_native stores one file per day and no
day is split across files. (The TGW arm needs the weighting because 01_county_weather.py carries a
per-chunk hour count n that 05_merge_county_weather.py divides by, its files not being day-aligned.)
So the two arms are directly comparable, which is what step 2 validates before anything is
re-thresholded.

NO HAZARD IS DEFINED HERE. This script writes weather, not flags, so it holds no percentile, no
persistence rule and no season gate, and it does not import hazard_defs. It is registered in
a weather file rather than a hazard file: it defines no threshold of its own. Every threshold applied to this
table is fitted in 04_c404_pipeline.py, which reads its constants from hazard_defs.py.

MINIMUM HOURS. A day is only kept if the file carries at least MIN_H = 20 of its 24 hours, the
same guard 04_r5_agg.py applies. A short file is a truncated day, and its tmax and tmin are biased
low and high because the missing hours are usually a contiguous block, which would then feed
straight into a day-of-year percentile. Short days are dropped, not NaN-filled, so they never
reach the parquet; they are listed in short_shard*.csv so the loss is countable rather than
silent.
"""
import glob, os, sys, zlib
import numpy as np, pandas as pd
from scipy.spatial import cKDTree

OUT = "/data/enso/county_weather_c404"; os.makedirs(OUT, exist_ok=True)
MIN_H = 20                                          # hours per day, matching 04_r5_agg.py
si, sn = map(int, (sys.argv[1] if len(sys.argv) > 1 else "0/1").split("/"))

# ---------------------------------------------------------------- c404 cell -> county
G = "/data/datasets/grid"
nn = np.load(f"{G}/c404_to_fine_nn.npz")
idx = nn["idx"].astype(np.int64)                    # fine cell -> c404 flat index
SRC = tuple(nn["src_shape"]); DST = tuple(nn["dst_shape"])
zc = np.load(f"{G}/coordinate.npz")
flat_lat = np.repeat(zc["lat"].astype(float), DST[1])
flat_lon = np.tile(zc["lon"].astype(float), DST[0])
nsrc = SRC[0] * SRC[1]
cnt = np.bincount(idx, minlength=nsrc).astype(float)
sum_la = np.bincount(idx, flat_lat, nsrc)
sum_lo = np.bincount(idx, flat_lon, nsrc)
ok = cnt > 0
c_lat = np.where(ok, sum_la / np.maximum(cnt, 1), np.nan)
c_lon = np.where(ok, sum_lo / np.maximum(cnt, 1), np.nan)

tg = np.load("/data/tgw_hist/tgw_grid.npz")
TL, TO = tg["XLAT"].ravel(), tg["XLONG"].ravel()
cm = np.load("/data/loads_measured/county_mask_tgw.npz", allow_pickle=True)
fips = np.array([str(f).zfill(5) for f in cm["fips"]])
cell2cty = np.full(len(TL), -1, np.int32)
cell2cty[cm["pair_cell"].astype(np.int64)] = cm["pair_fips"].astype(np.int32)

tree = cKDTree(np.c_[TL, TO])
sel = np.where(ok)[0]
_, j = tree.query(np.c_[c_lat[sel], c_lon[sel]])
src_cty = np.full(nsrc, -1, np.int32)
src_cty[sel] = cell2cty[j]
have = src_cty >= 0
order = np.argsort(src_cty[have], kind="stable")
src_idx = np.where(have)[0][order]
cty_of = src_cty[have][order]
bounds = np.searchsorted(cty_of, np.arange(len(fips)), side="left")
stops = np.searchsorted(cty_of, np.arange(len(fips)), side="right")
keep_c = np.where(stops > bounds)[0]
starts = bounds[keep_c]
ncell = (stops - bounds)[keep_c].astype(np.float32)
fips_keep = fips[keep_c]
if si == 0:
    print("c404 cells with a county: %s of %s   counties %d   cells/county mean %.1f"
          % (format(int(have.sum()), ","), format(nsrc, ","), len(keep_c), ncell.mean()),
          flush=True)

# ---------------------------------------------------------------- daily aggregation
files = sorted(glob.glob("/data/c404_native/*/c404_*.npz"))
files = [f for f in files if zlib.crc32(os.path.basename(f).encode()) % sn == si]
print("shard %d/%d : %d daily files" % (si, sn, len(files)), flush=True)

rows, bad, short = [], [], []
for k, f in enumerate(files):
    try:
        z = np.load(f, allow_pickle=True)
        d = z["data"].astype("f4")                   # (24, 9, 1015, 1367)
    except Exception as ex:                          # a few c404_native files are truncated
        bad.append((os.path.basename(f), repr(ex)[:80]))
        continue
    nh = d.shape[0]
    if nh < MIN_H:                                   # a part-day is not a day; see MINIMUM HOURS
        short.append((os.path.basename(f), nh))
        del d
        continue
    dd = d.reshape(nh, 9, -1)[:, :, src_idx]
    s = np.add.reduceat(dd, starts, axis=2)
    m = s / ncell[None, None, :]
    del d, dd, s
    t2 = m[:, 2, :]; q2 = m[:, 3, :]; ps = m[:, 4, :]; sw = m[:, 6, :]
    wspd = np.sqrt(m[:, 0, :] ** 2 + m[:, 1, :] ** 2)
    date = os.path.basename(f).split("_")[1][:8]
    date = "%s-%s-%s" % (date[:4], date[4:6], date[6:8])
    rows.append(pd.DataFrame({
        "date": date, "fips": fips_keep,
        "tmax": t2.max(0), "tmin": t2.min(0), "tmean": t2.mean(0),
        "q": q2.mean(0), "ps": ps.mean(0), "sw": sw.mean(0), "wspd": wspd.mean(0),
        "n": np.float32(nh)}))
    if k % 200 == 0:
        print("shard%d %d/%d" % (si, k, len(files)), flush=True)

res = pd.concat(rows, ignore_index=True)
res.to_parquet(f"{OUT}/c404_county_shard{si}.parquet", index=False)
if bad:
    pd.DataFrame(bad, columns=["file", "error"]).to_csv(f"{OUT}/bad_shard{si}.csv", index=False)
if short:
    pd.DataFrame(short, columns=["file", "hours"]).to_csv(f"{OUT}/short_shard{si}.csv", index=False)
print("shard %d DONE: %d files (%d unreadable, %d with < %d hours) -> %s rows"
      % (si, len(files), len(bad), len(short), MIN_H, format(len(res), ",")), flush=True)
