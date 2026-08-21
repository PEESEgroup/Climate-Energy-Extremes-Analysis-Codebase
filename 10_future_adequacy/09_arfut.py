"""The atmospheric river, projected. Figure 5g's sixth hazard, which used to be a blank row.

WHY IT WAS BLANK, AND WHY THAT WAS WRONG. Figure 1's atmospheric-river flag is `ivt_p95_cov25`,
built on ERA5 integrated vapour transport, and ERA5 does not run to 2050. The conclusion drawn was
that the hazard cannot be projected. But TGW-future is a warmed replay of the observed synoptic
sequence, so the storms are there; the only question is whether the download can see them. It can.

THE PROXY. IVT is a moisture FLUX, so column moisture alone does not carry it: `pwat` on its own
correlates 0.53 with the ERA5 IVT and catches 19% of its extreme days. Multiplying by the wind at
the highest level the download stores, 836 m above ground, gives `pwat * |V(836 m)|`, which
correlates 0.924 (median over 18 subregions, Spearman 0.882) and catches 74% of the ERA5 extreme
days, 85% in CAISO where the landfalls are. The download has no free-tropospheric wind, because it
was pulled for turbines, so this is a proxy and is labelled as one.

THE FLAG. Identical in structure to Figure 1's: a cell is active when its proxy exceeds its OWN
1990-2010 95th percentile, and a subregion-day is an atmospheric-river day when at least 25% of its
cells are active. Computed on TGW for BOTH periods, so the change is internally consistent even
though the level is not comparable with the ERA5 flag.

PAIRING. 1990-2010 against 2030-2050, the +40-year replay, matching 05_hazfreq.py.
"""
import glob, os, re, sys, json
import numpy as np, pandas as pd

TAG = sys.argv[1]                                    # "historical" or a climate name
G = "/data/datasets/grid"
OUT = "/data/ar_future"
os.makedirs(OUT, exist_ok=True)
zm = np.load(f"{G}/subregion_mask.npz", allow_pickle=True)
mask = zm["subregion_mask"]; id2 = {int(a): str(b) for a, b in zm["id_to_subregion"]}
zc = np.load(f"{G}/coordinate.npz"); lat, lon = zc["lat"].astype(float), zc["lon"].astype(float)
gt = np.load("/data/tgw_3d_future/tgw3d_grid.npz")
ila = np.clip(np.searchsorted(lat, gt["XLAT"].ravel()), 0, len(lat) - 1)
ilo = np.clip(np.searchsorted(lon, gt["XLONG"].ravel()), 0, len(lon) - 1)
SID = mask[ila, ilo].reshape(gt["XLAT"].shape)
LAND = SID > 0
NC = int(LAND.sum())
print("%s: %d cells inside the 18 subregions" % (TAG, NC), flush=True)

DATE = re.compile(r"_(\d{4})-\d{2}-\d{2}_")


def yr(f):
    m = DATE.search(os.path.basename(f))
    return int(m.group(1)) if m else -1


if TAG == "historical":
    fs = [f for f in sorted(glob.glob("/data/tgw_3d/tgw_wrf_historical_*.npz"))
          if 1990 <= yr(f) <= 2010]
else:
    fs = [f for f in sorted(glob.glob("/data/tgw_3d_future/tgw_wrf_%s_*.npz" % TAG))
          if 2030 <= yr(f) <= 2050]
print("%s: %d weekly files" % (TAG, len(fs)), flush=True)

dates, vals = [], []
for i, f in enumerate(fs):
    z = np.load(f, allow_pickle=True)
    pw = z["pwat"].astype(np.float32)
    uv = z["uv"][:, -1].astype(np.float32)           # 836 m AGL, the top stored level
    px = pw[:, LAND] * np.hypot(uv[:, 0], uv[:, 1])[:, LAND]
    t = pd.to_datetime(pd.Series(z["times"].astype(str)), format="%Y%m%d%H", errors="coerce")
    d = t.dt.floor("D").values
    df = pd.DataFrame(px).groupby(d).max()           # daily maximum, per cell
    dates.append(df.index.values); vals.append(df.values.astype(np.float32))
    del z, uv, pw, px, df
    if i % 50 == 0:
        print("  %s %d/%d" % (TAG, i, len(fs)), flush=True)
D = np.concatenate(dates); V = np.concatenate(vals)
o = np.argsort(D); D, V = D[o], V[o]
# The weekly files straddle the ends of the window, so the last one carries a few days of the year
# after it. Left in, they add a 22nd calendar year holding one day and divide the annual rate by 22
# instead of 21. Clip to the window itself, not to the files that cover it.
Y0, Y1 = (1990, 2010) if TAG == "historical" else (2030, 2050)
keep = (pd.DatetimeIndex(D).year >= Y0) & (pd.DatetimeIndex(D).year <= Y1)
D, V = D[keep], V[keep]
u, first = np.unique(D, return_index=True)
if len(u) != len(D):
    V = np.array([V[D == x].max(0) for x in u], dtype=np.float32); D = u
print("%s: %d days x %d cells, %.1f GB" % (TAG, V.shape[0], V.shape[1], V.nbytes / 1e9), flush=True)

TH = "%s/thresholds.npy" % OUT
if TAG == "historical":
    thr = np.percentile(V, 95, axis=0).astype(np.float32)
    np.save(TH, thr)
    print("wrote per-cell 95th percentiles", flush=True)
else:
    thr = np.load(TH)
act = V >= thr[None, :]
cov = np.zeros((V.shape[0], 18), dtype=np.float32)
sl = SID[LAND]
for s in range(1, 19):
    m = sl == s
    cov[:, s - 1] = act[:, m].mean(1) if m.any() else np.nan
F = pd.DataFrame(cov >= 0.25, index=pd.DatetimeIndex(D), columns=[id2[s] for s in range(1, 19)])
F.to_parquet("%s/ar_flag_%s.parquet" % (OUT, TAG))
ny = len(np.unique(F.index.year))
print("%s: atmospheric-river days per year, by subregion:" % TAG, flush=True)
print((F.sum() / ny).round(2).to_string(), flush=True)
print("%s: national mean %.2f d/yr over %d years" % (TAG, (F.sum() / ny).mean(), ny), flush=True)
