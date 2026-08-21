"""Stage 1 of the atmospheric-river rebuild: subregion-mean IVT, daily, 1980-2019.

WHY THIS EXISTS. The published AR flag `ar_pub` marks a subregion-day as AR-affected if any one of
its 1.5-degree cells lies inside an AR shape at any one of the day's four 6-hourly steps. With
subregions holding 6 to 55 such cells, that fires on 41.7% of all subregion-days, and its rate
correlates 0.744 with the NUMBER OF CELLS in the subregion. It is substantially a measure of how
big the box is. The physical check fails too: California ranks 12th of 18 and the Pacific Northwest
13th, behind the Southeast, the Southern Plains and the Mid-Atlantic. A flag that measured exposure
to atmospheric rivers would put the west coast first.

The fix is to stop asking whether an AR was present and start asking how strong the moisture
transport was, which is what the AR literature actually grades (Ralph et al. 2019 scale IVT ranges).
ERA5 IVT is already on disk at 0.5 degrees, 6-hourly, 1980-2019.

WHAT THIS PRODUCES. Per subregion and day: the mean and the maximum over the day's four steps of the
area-weighted mean IVT magnitude inside the subregion. Nothing is thresholded here; the flag
definitions come next, so that every variant is built from one aggregate.

IVT = hypot(viwve, viwvn), the vertically integrated horizontal water-vapour flux, kg m-1 s-1.
Cells are weighted by cos(latitude) because 0.5 degrees of longitude is not a constant area.
"""
import glob, os, time
import numpy as np
import xarray as xr

T0 = time.time()
def log(*a): print("[%6.1fs]" % (time.time() - T0), *a, flush=True)

OUT = "/data/enso/ivt_subregion_daily.npz"
zm = np.load("/data/datasets/grid/subregion_mask.npz", allow_pickle=True)
mask = zm["subregion_mask"]
id2name = {int(i): str(n) for i, n in zm["id_to_subregion"]}
NAMES = [id2name[i] for i in sorted(id2name)]
NS = len(NAMES)
zc = np.load("/data/datasets/grid/coordinate.npz")
glat, glon = zc["lat"].astype(float), zc["lon"].astype(float)
log("subregions %d, model grid %d x %d" % (NS, len(glat), len(glon)))

# ---- map each ERA5 cell to a subregion, once
d0 = xr.open_dataset(sorted(glob.glob("/data/era5/ivt_*.nc"))[0])
elat = d0.latitude.values.astype(float); elon = d0.longitude.values.astype(float)
d0.close()
ii = np.clip(np.searchsorted(glat, elat), 0, len(glat) - 1)
jj = np.clip(np.searchsorted(glon, elon), 0, len(glon) - 1)
SID = mask[np.ix_(ii, jj)]                       # (nlatE, nlonE) subregion id, 0 = outside
W = np.cos(np.deg2rad(elat))[:, None] * np.ones((1, len(elon)))
CELLS = {}
for k, nm in enumerate(NAMES):
    m = SID == (k + 1) if (SID == (k + 1)).any() else SID == k
    CELLS[nm] = m
log("ERA5 grid %d x %d ; cells per subregion: %s"
    % (len(elat), len(elon), {n: int(CELLS[n].sum()) for n in NAMES}))
bad = [n for n in NAMES if CELLS[n].sum() == 0]
assert not bad, "no ERA5 cell landed in %s" % bad

DATES, MEAN, MAXI = [], [], []
for f in sorted(glob.glob("/data/era5/ivt_*.nc")):
    yr = int(os.path.basename(f).split("_")[1][:4])
    d = xr.open_dataset(f)
    ivt = np.hypot(d.viwve.values, d.viwvn.values)          # (t, lat, lon)
    t = d.valid_time.values
    d.close()
    # area-weighted subregion mean at each 6-hourly step
    S = np.empty((len(t), NS), np.float32)
    for k, nm in enumerate(NAMES):
        m = CELLS[nm]; w = W[m]
        S[:, k] = (ivt[:, m] * w).sum(1) / w.sum()
    day = t.astype("datetime64[D]")
    ud, inv = np.unique(day, return_inverse=True)
    mn = np.zeros((len(ud), NS), np.float32); mx = np.zeros((len(ud), NS), np.float32)
    cnt = np.bincount(inv, minlength=len(ud)).astype(np.float32)
    for k in range(NS):
        mn[:, k] = np.bincount(inv, S[:, k], minlength=len(ud)) / cnt
        np.maximum.at(mx[:, k], inv, S[:, k])
    DATES.append(ud); MEAN.append(mn); MAXI.append(mx)
    log("%d  %d days  national mean IVT %.1f kg/m/s" % (yr, len(ud), mn.mean()))

dates = np.concatenate(DATES); mean = np.concatenate(MEAN); maxi = np.concatenate(MAXI)
o = np.argsort(dates)
np.savez_compressed(OUT, dates=dates[o].astype(str), subregions=np.array(NAMES),
                    ivt_mean=mean[o].T, ivt_max=maxi[o].T)
log("wrote %s  ivt_mean %s" % (OUT, mean.T.shape))
for k, nm in enumerate(NAMES):
    v = mean[:, k]
    print("  %-20s mean %6.1f  p90 %6.1f  p99 %6.1f  max %6.1f"
          % (nm, v.mean(), np.percentile(v, 90), np.percentile(v, 99), v.max()))
