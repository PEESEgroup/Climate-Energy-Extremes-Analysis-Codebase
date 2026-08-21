"""CONUS404 3D hub-wind + surface-predictor extractor via Azure Open Data zarr (az://noaa/conus404.zarr).
Independent of NCAR THREDDS/dodsC. Anonymous via free Planetary-Computer SAS token (auto-refresh).
Produces the IDENTICAL c404_3d_YYYYMMDDHH.npz (bit-exact to the dodsC-built frames, verified).
Env: N (workers, default 16), ORDER (asc/desc), FLOOR_TB (default 1.5), TESTN (limit for smoke test)."""
import os, sys, time, json, glob, warnings, urllib.request, numpy as np
warnings.filterwarnings("ignore")
import xarray as xr

OUT   = "/data/c404_3d"
SPLIT = "/data/datasets/train/split3.npz"
STOP  = "/data/logs/c404_3d_azure.STOP"
FLOOR_TB = float(os.environ.get("FLOOR_TB", "1.5"))
NL = 15
HEIGHTS = np.array([10, 40, 80, 110, 140, 160, 200], dtype="float32")
PRED = ['u10', 'v10', 't2', 'rh', 'psfc', 'pwat', 'short', 'long']
ACCOUNT, CONTAINER = "azureopendatastorage", "noaa"
SAS_URL = f"https://planetarycomputer.microsoft.com/api/sas/v1/token/{ACCOUNT}/{CONTAINER}"
_W = {}  # per-worker state: ds, opened_at

def free_tb(p="/data"):
    s = os.statvfs(p); return s.f_bavail * s.f_frsize / 1e12

def _sas():
    for a in range(5):
        try: return json.load(urllib.request.urlopen(SAS_URL, timeout=30))["token"]
        except Exception:
            if a == 4: raise
            time.sleep(3)

def _open():
    ds = xr.open_zarr(f"az://{CONTAINER}/conus404.zarr",
                      storage_options={"account_name": ACCOUNT, "sas_token": _sas()},
                      consolidated=True, decode_times=True)
    _W['ds'] = ds; _W['t'] = time.time(); return ds

def _ds():
    # refresh token/handle every 40 min (SAS ~1h)
    if 'ds' not in _W or (time.time() - _W['t']) > 2400: _open()
    return _W['ds']

def _interp(arr, z, h):
    # vectorized, BIT-EXACT to the per-level loop (verified max|diff|=0.0)
    NLz = arr.shape[0]
    kk = np.clip((z <= h).sum(0) - 1, 0, NLz - 2)[None]
    z0 = np.take_along_axis(z, kk, 0)[0]; z1 = np.take_along_axis(z, kk + 1, 0)[0]
    a0 = np.take_along_axis(arr, kk, 0)[0]; a1 = np.take_along_axis(arr, kk + 1, 0)[0]
    w = (h - z0) / np.where(z1 > z0, z1 - z0, 1.0)
    out = a0 + w * (a1 - a0)
    out = np.where(h < z[0], arr[0], out)
    out = np.where(h >= z[-1], np.nan, out)
    return out.astype(np.float32)

def _rh(q2, t2, psfc):
    e = q2 * psfc / (0.622 + q2); es = 611.2 * np.exp(17.67 * (t2 - 273.15) / (t2 - 29.65))
    return np.clip(100.0 * e / es, 0, 100).astype(np.float32)

def one(ts):
    Y, M, D, H = int(ts[:4]), int(ts[4:6]), int(ts[6:8]), int(ts[8:10])
    tag = f"{Y:04d}{M:02d}{D:02d}{H:02d}"; fp = f"{OUT}/c404_3d_{tag}.npz"
    if os.path.exists(fp): return ('skip', tag)
    tstr = f"{Y:04d}-{M:02d}-{D:02d}T{H:02d}:00:00"
    for attempt in range(3):
        try:
            sel = _ds().sel(time=tstr, method="nearest")
            got = np.datetime64(sel["time"].values); want = np.datetime64(tstr)
            if abs((got - want) / np.timedelta64(1, "h")) > 0.5:
                return ("NOTAVAIL want=%s got=%s" % (tstr, str(got)[:16]), tag)
            U = np.asarray(sel['U'].isel(bottom_top=slice(0, NL)).values)
            V = np.asarray(sel['V'].isel(bottom_top=slice(0, NL)).values)
            Z = np.asarray(sel['Z'].isel(bottom_top_stag=slice(0, NL + 1)).values)
            g = lambda v: np.asarray(sel[v].values, np.float32)
            u10 = g('U10'); v10 = g('V10'); t2 = g('T2'); q2 = g('Q2'); psf = g('PSFC'); pw = g('PWAT')
            short = np.clip(g('SWDOWN'), 0, None); lng = np.clip(g('GLW'), 0, None)
            ust = g('UST'); pblh = g('PBLH')
            break
        except Exception as e:
            if attempt == 2: return (f'ERR {str(e)[:60]}', tag)
            _open(); time.sleep(2)
    HGT = Z[0].astype(np.float64)                       # surface geopotential height = terrain (bit-exact vs wrfconstants HGT)
    U = U.astype(np.float64); V = V.astype(np.float64); Z = Z.astype(np.float64)
    Um = 0.5 * (U[:, :, :-1] + U[:, :, 1:]); Vm = 0.5 * (V[:, :-1, :] + V[:, 1:, :])
    Zagl = 0.5 * (Z[:-1] + Z[1:]) - HGT[None]
    hub = np.stack([np.stack([_interp(Um, Zagl, h), _interp(Vm, Zagl, h)]) for h in HEIGHTS], 0).astype('float16')
    pred = np.stack([u10, v10, t2, _rh(q2, t2, psf), psf * 0.01, pw, short, lng], 0).astype('float16')
    abl = np.stack([ust, pblh], 0).astype('float16')
    if pred.shape[1:] != hub.shape[2:]: return (f'GRIDMISMATCH {pred.shape[1:]} vs {hub.shape[2:]}', tag)
    tmp = fp + '.tmp.npz'
    np.savez_compressed(tmp, hub=hub, pred=pred, abl=abl, heights=HEIGHTS,
                        pred_vars=np.array(PRED), abl_vars=np.array(['ust', 'pblh']), ts=tag)
    os.replace(tmp, fp); return ('ok', tag)

def _init(): _open()

def main():
    from multiprocessing import Pool
    stamps = [str(x) for x in np.load(SPLIT, allow_pickle=True)['stamps']]
    exist = set(os.path.basename(p)[8:18] for p in glob.glob(f"{OUT}/c404_3d_[0-9]*.npz"))
    ts = [s for s in stamps if s not in exist]
    ts.sort(reverse=(os.environ.get("ORDER", "asc") == "desc"))
    testn = int(os.environ.get("TESTN", "0"))
    if testn: ts = ts[:testn]
    N = int(os.environ.get("N", "16"))
    os.makedirs(OUT, exist_ok=True)
    print(f"=== azure START targets={len(ts)} workers={N} free={free_tb():.2f}TB {time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime())} ===", flush=True)
    done = sk = err = 0; t0 = time.time()
    with Pool(N, initializer=_init) as p:
        for i, (st, tag) in enumerate(p.imap_unordered(one, ts, chunksize=1)):
            if st == 'ok': done += 1
            elif st == 'skip': sk += 1
            else: err += 1; print(f"[{st}] {tag}", flush=True)
            if (i + 1) % 50 == 0:
                rate = done / max(time.time() - t0, 1) * 60
                rem = (len(ts) - i - 1) / max(rate, 0.1) / 60
                print(f"  {i+1}/{len(ts)} | ok {done} skip {sk} err {err} | {rate:.1f} f/min ETA {rem:.1f}h free={free_tb():.2f}TB", flush=True)
                if os.path.exists(STOP): print("STOP flag -> exit", flush=True); break
                if free_tb() < FLOOR_TB: print("DISK FLOOR -> exit", flush=True); break
    print(f"=== azure END ok {done} skip {sk} err {err} in {(time.time()-t0)/3600:.2f}h ===", flush=True)

if __name__ == "__main__":
    main()
