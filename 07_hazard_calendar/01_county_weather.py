"""
Task #62 — county-resolved DAILY weather 1980-2019, the same construction as the 18-subregion
table (/data/02_subregion_weather2.py) but aggregated to 3,108 counties instead.

Why: only 19.7% of the variance of log county outage burden is BETWEEN the 18 subregions. A
hazard calendar resolved at BA scale is structurally blind to the other 80%, and it attenuates
any hazard-day effect through binary misclassification. The county mask sits on exactly the TGW
grid (299 x 424 = 126,776 cells), so this is a pure re-aggregation of the same hours.

Identical conventions to the subregion build, so the thresholds transfer unchanged:
  real value = stored / scale ; var order [U10, V10, Q2, PSFC, T2, GLW, SWDOWN]
  tmax = max, tmin = min, everything else = hour-count-weighted mean
  n carried per (file, county, date) so days split across two file-chunks recombine EXACTLY
"""
import numpy as np, pandas as pd, glob, os, sys, zlib

OUT = "/data/enso/county_weather"; os.makedirs(OUT, exist_ok=True)
si, sn = map(int, (sys.argv[1] if len(sys.argv) > 1 else "0/1").split("/"))

cm = np.load("/data/loads_measured/county_mask_tgw.npz", allow_pickle=True)
fips = np.array([str(f).zfill(5) for f in cm["fips"]])
pc, pf = cm["pair_cell"].astype(np.int64), cm["pair_fips"].astype(np.int64)
o = np.argsort(pf, kind="stable")                      # group the (cell, county) pairs by county
pc, pf = pc[o], pf[o]
start = np.searchsorted(pf, np.arange(len(fips)), side="left")
stop = np.searchsorted(pf, np.arange(len(fips)), side="right")
have = np.where(stop > start)[0]                       # counties with at least one TGW cell
bounds = start[have]
cnt = (stop - start)[have].astype(np.float32)
fips_have = fips[have]
if si == 0:
    print("counties with cells: %d of %d   pairs %d   cells/county mean %.1f"
          % (len(have), len(fips), len(pc), cnt.mean()), flush=True)

files = sorted(glob.glob("/data/tgw_hist/tgw_historical_*hourly*.npz"))
files = [f for f in files if zlib.crc32(os.path.basename(f).encode()) % sn == si]
print("shard %d/%d : %d files" % (si, sn, len(files)), flush=True)

out = []
for k, f in enumerate(files):
    z = np.load(f, allow_pickle=True)
    d = z["data"].astype("f4")
    sc = np.asarray(z["scale"], "f4")
    d = d / sc[None, :, None, None]
    nh = d.shape[0]
    dd = d.reshape(nh, 7, -1)[:, :, pc]                # (nh, 7, npairs), pairs sorted by county
    s = np.add.reduceat(dd, bounds, axis=2)            # (nh, 7, ncounty) sums
    m = s / cnt[None, None, :]                         # county means
    del d, dd, s
    times = pd.to_datetime([str(t) for t in z["times"]], format="%Y%m%d%H")
    dates = np.asarray(times.strftime("%Y-%m-%d"))
    t2 = m[:, 4, :]; q2 = m[:, 2, :]; ps = m[:, 3, :]; sw = m[:, 6, :]
    wspd = np.sqrt(m[:, 0, :] ** 2 + m[:, 1, :] ** 2)
    ud, inv = np.unique(dates, return_inverse=True)
    nd, nc = len(ud), len(fips_have)
    acc = {}
    n_h = np.bincount(inv, minlength=nd).astype(np.float32)
    for nm, arr, how in [("tmax", t2, "max"), ("tmin", t2, "min"), ("tmean", t2, "sum"),
                         ("q", q2, "sum"), ("ps", ps, "sum"), ("sw", sw, "sum"),
                         ("wspd", wspd, "sum")]:
        r = np.full((nd, nc), -np.inf if how == "max" else (np.inf if how == "min" else 0.0),
                    dtype=np.float32)
        if how == "sum":
            np.add.at(r, inv, arr)
        elif how == "max":
            np.maximum.at(r, inv, arr)
        else:
            np.minimum.at(r, inv, arr)
        acc[nm] = r
    df = pd.DataFrame({
        "date": np.repeat(ud, nc), "fips": np.tile(fips_have, nd),
        "tmax": acc["tmax"].ravel(), "tmin": acc["tmin"].ravel(),
        "tmean": acc["tmean"].ravel(), "q": acc["q"].ravel(), "ps": acc["ps"].ravel(),
        "sw": acc["sw"].ravel(), "wspd": acc["wspd"].ravel(),
        "n": np.repeat(n_h, nc).astype(np.float32)})
    out.append(df)
    if k % 50 == 0:
        print("shard%d %d/%d" % (si, k, len(files)), flush=True)

res = pd.concat(out, ignore_index=True)
for c in ["tmean", "q", "ps", "sw", "wspd"]:
    res[c] = res[c].astype("f4")                       # still SUMS; merge divides by total n
res.to_parquet(f"{OUT}/county_weather_shard{si}.parquet", index=False)
print("shard %d DONE: %d files -> %d rows" % (si, len(files), len(res)), flush=True)
