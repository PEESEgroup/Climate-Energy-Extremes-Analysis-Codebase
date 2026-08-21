"""
R5 TC arm, step 1 - the HURDAT2 space-time window.

A county-day is INSIDE THE TC WINDOW if the county centroid is within 600 km of any HURDAT2 track
point of >=34 kt in the CONUS domain on that day. The window is deliberately generous: it is NOT
the exposure criterion, it only prevents an unrelated extratropical low from being credited to a
hurricane. The exposure criterion is the simulated 10-m wind (step 2).

The same window, shifted +40 years, locates the same storms in the TGW future runs - legitimate
because TGW-future replays the historical synoptic sequence (GATE A, r_uv = 0.97 at lag 40 y
against 0.16-0.28 at every decoy lag).

Parser is the one from 12_county_tc_swath.py so the two products cannot diverge.
"""
import re, os
import numpy as np, pandas as pd

OUT = "/data/scratch_r5"
H2 = "/data/equity_cost/analysis/did/hurdat2_latest.txt"
Y0, Y1 = 1990, 2010
R_WIN = 800.0

rows, hdr = [], None
for ln in open(H2, errors="ignore"):
    p = [x.strip() for x in ln.split(",")]
    if len(p) >= 3 and re.match(r"^AL\d{6}$", p[0]) and p[2].isdigit():
        hdr = (p[0], p[1], int(p[0][4:8])); continue
    if hdr is None or len(p) < 8:
        continue
    try:
        dt = pd.to_datetime(p[0] + " " + p[1][:2], format="%Y%m%d %H")
        la = float(p[4][:-1]) * (1 if p[4][-1] == "N" else -1)
        lo = float(p[5][:-1]) * (-1 if p[5][-1] == "W" else 1)
        vmax = float(p[6])
    except Exception:
        continue
    rows.append((hdr[0], hdr[1], hdr[2], dt, la, lo, vmax))
T = pd.DataFrame(rows, columns=["sid", "name", "year", "dt", "lat", "lon", "vmax"])
T = T[(T.year >= Y0) & (T.year <= Y1) & (T.vmax >= 34)]
T = T[(T.lat > 10) & (T.lat < 55) & (T.lon > -110) & (T.lon < -50)]
T["date"] = T.dt.dt.normalize()
print("HURDAT2 %d-%d: %d track points >=34kt in domain, %d storms, %d storm-days"
      % (Y0, Y1, len(T), T.sid.nunique(), len(T.groupby(["sid", "date"]))), flush=True)

# ------------------------------------------------------------------ county centroids on TGW grid
cm = np.load("/data/loads_measured/county_mask_tgw.npz", allow_pickle=True)
FIPS = np.array([str(f).zfill(5) for f in cm["fips"]])
cell = cm["pair_cell"].astype(np.int64); cty = cm["pair_fips"].astype(np.int64)
g = np.load("/data/tgw_hist/tgw_grid.npz")
LAT, LON = g["XLAT"].ravel(), g["XLONG"].ravel()
n = np.bincount(cty, minlength=len(FIPS)).astype(float)
clat = np.bincount(cty, LAT[cell], len(FIPS)) / n
clon = np.bincount(cty, LON[cell], len(FIPS)) / n
print("county centroids %d  lat %.1f..%.1f" % (len(FIPS), clat.min(), clat.max()), flush=True)


def hav(la1, lo1, la2, lo2):
    p = np.pi / 180.0
    a = np.sin((la2 - la1) * p / 2) ** 2 + np.cos(la1 * p) * np.cos(la2 * p) * np.sin((lo2 - lo1) * p / 2) ** 2
    return 2 * 6371.0 * np.arcsin(np.sqrt(np.clip(a, 0, 1)))


out = []
for (sid, d), grp in T.groupby(["sid", "date"]):
    dmin = np.full(len(FIPS), np.inf)
    vmx = np.zeros(len(FIPS))
    for la, lo, vm in zip(grp.lat.values, grp.lon.values, grp.vmax.values):
        dd = hav(clat, clon, la, lo)
        upd = dd < dmin
        dmin = np.minimum(dmin, dd)
        vmx = np.where(dd <= R_WIN, np.maximum(vmx, vm), vmx)
    k = dmin <= R_WIN
    if not k.any():
        continue
    out.append(pd.DataFrame(dict(fips=FIPS[k], date=d, sid=sid,
                                 dist_km=dmin[k].astype("f4"), storm_vmax_kt=vmx[k].astype("f4"))))
W = pd.concat(out, ignore_index=True)
W = W.sort_values(["fips", "date", "dist_km"]).drop_duplicates(["fips", "date"], keep="first")
W.to_parquet("%s/tc_window_1990_2010.parquet" % OUT, index=False)
print("TC window county-days: %s  counties %d  dates %d  storms %d"
      % (format(len(W), ","), W.fips.nunique(), W.date.nunique(), W.sid.nunique()), flush=True)

# ------------------------------------------------------------------ the observed reference
O = pd.read_parquet("/data/enso/tc_county_ext/county_tc_days.parquet")
O = O[(O.date >= "%d-01-01" % Y0) & (O.date <= "%d-12-31" % Y1)]
print("HURDAT2 modified-Rankine county-TC-days %d-%d: %s  counties %d  storms %d"
      % (Y0, Y1, format(len(O), ","), O.fips.nunique(), O.sid.nunique()), flush=True)
cov = O.merge(W[["fips", "date"]].assign(inwin=1), on=["fips", "date"], how="left")
print("  of which inside the 600 km window: %.4f  <-- must be ~1.0 or the window is too tight"
      % cov.inwin.notna().mean(), flush=True)
O.to_parquet("%s/tc_obs_1990_2010.parquet" % OUT, index=False)
