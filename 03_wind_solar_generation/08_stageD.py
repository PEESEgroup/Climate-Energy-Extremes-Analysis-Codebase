"""STAGE D - carry hub wind from the 3-hourly 3D cadence to hourly, then apply the power curve.

ORDER MATTERS AND IS DELIBERATE. The temporal super-resolution acts on hub wind SPEED, and the
power curve is applied afterwards. Interpolating capacity factor instead would smooth across the
cut-in and cut-out discontinuities, and by Jensen's inequality would understate the contribution of
short high-wind episodes, because the power curve is strongly non-linear in speed.

    hub_1h(t) = interp3h(hub)(t) * clip( U_sfc,1h(t) / interp3h(U_sfc)(t), 0.3, 3.0 )

The multiplier is the hourly variability the 3-hourly series cannot carry, taken from the hourly
10 m wind, which is a genuinely hourly TGW field. Linear interpolation alone recovers a ramp ratio
of only about 0.75 to 0.85; this hybrid restores it to about 1.0 (validated on 2019).

Both arms use this one script. They differ only in where the hourly surface wind comes from:
historical from the packed hourly LR archive, future from the per-scenario hourly extracts.

ENV: WX (the wind_infer output), SRC (hist|future), CLIMATE (future only), OUT.
"""
import os
import sys
import time

import numpy as np
import pandas as pd

WX = os.environ["WX"]
SRC = os.environ.get("SRC", "hist")
CLIMATE = os.environ.get("CLIMATE", "")
OUTF = os.environ["OUT"]
NPROC = int(os.environ.get("NPROC", "6"))
GC = "/data/datasets/gen/tgw-gen-historical"

Y0 = int(os.environ.get("Y0", "1997")); Y1 = int(os.environ.get("Y1", "2019"))
Z = np.load(WX, allow_pickle=True)
done = np.asarray(Z["done"], bool)
_st = Z["stamps"].astype(str)
_yr = np.array([Y0 <= int(x[:4]) <= Y1 for x in _st])
done = done & _yr
# A window that selects nothing is the silent failure this stage is most exposed to: the default is
# the historical window, and running the future arm without Y0/Y1 would pass 0 of 61,360 stamps
# through and write a correctly shaped, entirely empty product.
assert done.sum() > 0, (f"year window {Y0}-{Y1} selected 0 of {len(_st)} stamps "
                        f"(data spans {_st.min()[:4]}-{_st.max()[:4]}); set Y0/Y1")
st3 = _st[done]
W3 = Z["wsphub"][:, done].astype(np.float32)          # (nsite, T3) hub speed, 3-hourly
T23 = Z["t2"][:, done].astype(np.float32)
PS3 = Z["psfc"][:, done].astype(np.float32)
slat = Z["lat"].astype(float); slon = Z["lon"].astype(float)
hub = Z["hub"].astype(float); cfg_pcu = Z["cfg_pcu"].astype(str)
nsite, T3 = W3.shape
print(f"[D] {WX}: {nsite} sites x {T3} 3-hourly stamps  hub mean {np.nanmean(W3):.2f} m/s", flush=True)

# ---------------------------------------------------------------- hourly surface wind at sites
# BOTH ARMS read the hourly 10 m wind from the SAME product: the native 12 km surface archive.
# The historical arm used to take it from the 89x211 coarsened archive that feeds the downscaler,
# which is about 28 km, so the two arms modulated hub wind at different effective resolutions.
# That is precisely the inconsistency this rebuild exists to remove.
SFCDIR = os.environ.get("SFCDIR", "/data/tgw_hist_sfc" if SRC == "hist" else f"/data/tgw_extract/{CLIMATE}")
GRIDNPZ = os.environ.get("GRIDNPZ", "/data/tgw_extract/rcp45cooler/tgw_grid.npz")   # one WRF grid serves all
_g = np.load(GRIDNPZ)
_XF = np.asarray(_g["XLAT"], np.float64).ravel(); _YF = np.asarray(_g["XLONG"], np.float64).ravel()
natcell = np.array([int(np.argmin((_XF - la) ** 2 + (_YF - lo) ** 2)) for la, lo in zip(slat, slon)])
_err = np.hypot(_XF[natcell] - slat, _YF[natcell] - slon)
print(f"[D] {SFCDIR}: site mapping median offset {np.median(_err):.3f} deg, max {_err.max():.3f} deg", flush=True)
# Indexing a flattened frame with another grid's index samples the wrong place and stays plausible,
# because the multiplier is clipped and the mean survives. Only this assertion catches it.
assert _err.max() < 0.5, f"site-to-cell mapping is wrong: max offset {_err.max():.2f} deg"

import glob
fs = [f for f in sorted(glob.glob(f"{SFCDIR}/*.npz")) if "grid" not in f]
assert fs, f"no surface files under {SFCDIR}"
hs = []; parts = []
for f in fs:
    z = np.load(f, allow_pickle=True)
    tm = [str(t) for t in z["times"]]
    keep = [k for k, t in enumerate(tm) if Y0 <= int(t[:4]) <= Y1]
    if not keep:
        z.close(); continue
    v = [str(x) for x in z["vars"]]; sc = z["scale"]
    iu, iv = v.index("U10"), v.index("V10")
    assert float(sc[iu]) == 1.0 and float(sc[iv]) == 1.0, "wind components are packed; divide by scale"
    d = z["data"]
    u = np.asarray(d[keep, iu], np.float32).reshape(len(keep), -1)[:, natcell]
    w = np.asarray(d[keep, iv], np.float32).reshape(len(keep), -1)[:, natcell]
    hs += [tm[k] for k in keep]; parts.append(np.hypot(u, w))
    z.close()
assert parts, f"no stamps inside {Y0}-{Y1} under {SFCDIR}"
hst = np.array(hs); S1 = np.concatenate(parts, 0).T
o = np.argsort(hst); hst = hst[o]; S1 = S1[:, o]
del parts

print(f"[D] hourly surface: {S1.shape[1]} stamps", flush=True)

# ---------------------------------------------------------------- time axes and the hybrid
def to_ns(a):
    return pd.to_datetime(pd.Series(a), format="%Y%m%d%H").values.astype("datetime64[ns]").astype(np.int64)

t3 = to_ns(st3); th = to_ns(hst)
keep = (th >= t3.min()) & (th <= t3.max())
th = th[keep]; hst = hst[keep]; S1 = S1[:, keep]
Th = len(th)
print(f"[D] hourly grid inside the 3-hourly span: {Th} stamps", flush=True)

W1 = np.empty((nsite, Th), np.float32)
T21 = np.empty((nsite, Th), np.float32)
PS1 = np.empty((nsite, Th), np.float32)
t0 = time.time()
for i in range(nsite):
    lin = np.interp(th, t3, W3[i])
    s3 = np.interp(t3, th, S1[i])                   # the surface wind seen at 3-hourly resolution
    s3l = np.interp(th, t3, s3)                     # and put back on the hourly grid
    mfac = np.clip(S1[i] / np.maximum(s3l, 0.1), 0.3, 3.0)
    W1[i] = lin * mfac
    T21[i] = np.interp(th, t3, T23[i])
    PS1[i] = np.interp(th, t3, PS3[i])
    if i % 200 == 0:
        print(f"  [D] SR {i}/{nsite} {time.time()-t0:.0f}s", flush=True)
print(f"[D] hourly hub mean {np.nanmean(W1):.2f} m/s (3-hourly was {np.nanmean(W3):.2f})", flush=True)

# ---------------------------------------------------------------- power curve on the hourly series
import os as _os_rp
for _rp in ("04_demand_model", "07_hazard_calendar", "09_outage_attribution",
            "02_downscale_wind", "12_figures"):
    _ap = _os_rp.path.abspath(_os_rp.path.join(
        _os_rp.path.dirname(_os_rp.path.abspath(__file__)), "..", _rp))
    if _os_rp.path.isdir(_ap) and _ap not in sys.path:
        sys.path.insert(0, _ap)
from gen_physics import wind_cf
import multiprocessing as mp

cfg = pd.read_csv(f"{GC}/eia_wind_configs.csv", dtype={"plant_code_unique": str}) \
        .drop_duplicates("plant_code_unique").set_index("plant_code_unique")
L = ((Th + 8759) // 8760) * 8760
FULLy = pd.date_range("2015-01-01", periods=L, freq="h")
# Free everything the power-curve stage does not need BEFORE forking the pool. At 6,233 sites x
# 184,077 hourly steps each float32 array is 4.6 GB, and holding the 3-hourly inputs, the hourly
# surface wind and a separate Celsius copy pushed peak resident memory to 40 GB, which the kernel
# OOM-killed while the solar job held 30 GB. Converting in place and dropping the spent arrays
# takes the peak to roughly 18 GB.
del S1, W3, T23, PS3
T21 -= 273.15                      # in place: a fresh Celsius array would cost another 4.6 GB
_G = {"w": W1, "t": T21, "p": PS1}

def one(pi):
    c = cfg.loc[cfg_pcu[pi]]
    ws = eval(c.wind_turbine_powercurve_windspeeds) if isinstance(c.wind_turbine_powercurve_windspeeds, str) else c.wind_turbine_powercurve_windspeeds
    pw = eval(c.wind_turbine_powercurve_powerout) if isinstance(c.wind_turbine_powercurve_powerout, str) else c.wind_turbine_powercurve_powerout
    tgl = float(c.turb_generic_loss) if "turb_generic_loss" in c.index and pd.notna(c.turb_generic_loss) else 15.0
    base = dict(wind_turbine_hub_ht=float(c.wind_turbine_hub_ht),
                wind_turbine_rotor_diameter=float(c.wind_turbine_rotor_diameter),
                system_capacity=float(max(pw)),
                wind_turbine_powercurve_windspeeds=list(ws),
                wind_turbine_powercurve_powerout=list(pw))
    w = np.zeros(L, np.float32); w[:Th] = np.nan_to_num(_G["w"][pi])
    t = np.full(L, 15.0, np.float32); t[:Th] = np.nan_to_num(_G["t"][pi], nan=10.0)
    p = np.full(L, 1013.0, np.float32); p[:Th] = np.nan_to_num(_G["p"][pi], nan=1013.0)
    cf = wind_cf(FULLy, w, t, p, float(slat[pi]), float(slon[pi]),
                 {**base, "shear": 0.0, "turb_generic_loss": tgl})[:Th]
    return pi, np.asarray(cf, np.float32)

CF = np.full((nsite, Th), np.nan, np.float32); t0 = time.time()
with mp.get_context("fork").Pool(NPROC, maxtasksperchild=100) as pool:
    for k, (pi, cf) in enumerate(pool.imap_unordered(one, range(nsite), chunksize=16)):
        CF[pi] = cf
        if k % 200 == 0:
            print(f"  [D] pysam {k}/{nsite} {time.time()-t0:.0f}s", flush=True)

np.savez(OUTF, cf=CF, stamps=hst, plants=cfg_pcu, lat=slat, lon=slon, hub=hub,
         hub_wind=W1.astype(np.float32),
         note="hourly plant CF; hub wind super-resolved 3h->1h on SPEED then power curve; "
              "hub(t)=interp3h(hub)*clip(Usfc_1h/interp3h(Usfc),0.3,3.0)")
print(f"[D] DONE  CF mean {np.nanmean(CF):.3f}  {Th} hourly stamps -> {OUTF}", flush=True)
