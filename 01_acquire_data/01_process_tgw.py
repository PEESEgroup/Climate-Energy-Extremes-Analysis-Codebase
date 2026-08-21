"""Stream-process TGW future weekly NetCDF files -> compact fp16 store of the 7 SRGAN LR inputs,
then DELETE the raw file (keeps disk small: ~1.3 TB total vs ~9 TB raw). Watches a Globus drop dir.

7 vars extracted (WRF name -> our LR channel order [u10,v10,q2,psfc,t2,long,short]):
  U10, V10, Q2, PSFC, T2, GLW(=long, downward LW W/m2), SWDOWN(=short, downward SW W/m2).
All on the unstaggered mass grid (Time, south_north=299, west_east=424); native SI units -> matches
CONUS404 (same z-score norm applies at inference). Grid XLAT/XLONG saved once (for the TGW->fine regrid).

Usage (run one per scenario, e.g. as a daemon while Globus transfers):
  python 01_process_tgw.py --scenario rcp85hotter \
      --indir /data/tgw_future/rcp85hotter --outdir /data/tgw_extract/rcp85hotter \
      --y0 2030 --y1 2050 --watch --delete
Idempotent/resumable: skips files whose output npz already exists; safe to restart."""
import os, sys, glob, time, argparse, traceback, zlib, numpy as np, xarray as xr

LR_VARS = ['U10','V10','Q2','PSFC','T2','GLW','SWDOWN']   # -> [u10,v10,q2,psfc,t2,long,short]
# PSFC in Pa (~1e5) OVERFLOWS float16 (max 65504) -> store PSFC in hPa (x0.01); restore x100 at inference (see 'scale' key).
SCALE = np.array([1., 1., 1., 0.01, 1., 1., 1.], dtype='float32')   # per-LR_VARS multiplier applied before the fp16 cast

def parse_times(ds):
    """Return YYYYMMDDHH strings, one per Time step, from WRF 'Times' char array (robust to layout)."""
    if 'Times' in ds.variables:
        raw = ds['Times'].values
        out = []
        for row in raw:
            if isinstance(row, bytes):            s = row.decode()
            elif getattr(row, 'ndim', 0) >= 1:    s = b''.join([c if isinstance(c, bytes) else c.tobytes() for c in row]).decode(errors='ignore')
            else:                                  s = str(row)
            d = ''.join(ch for ch in s if ch.isdigit())      # 2030-01-01_00:00:00 -> 20300101000000
            out.append(d[:10])                                # YYYYMMDDHH
        return np.array(out)
    # fallback: decoded datetime coord
    tc = 'Time' if 'Time' in ds.coords else ('time' if 'time' in ds.coords else None)
    tt = ds[tc].values
    return np.array([np.datetime_as_string(x, unit='h').translate(str.maketrans('', '', '-T:'))[:10] for x in tt])

def process(f, outdir, scen, y0, y1, gridpath, delete):
    out = f"{outdir}/tgw_{scen}_{os.path.basename(f).replace('.nc','')}.npz"
    if os.path.exists(out):
        try: _z = np.load(out); _old = ('scale' not in _z.files); _z.close()   # old format = PSFC-overflow corrupt
        except Exception: _old = True
        if not _old:
            if delete and os.path.exists(f): os.remove(f)     # already extracted (new/fixed format) -> drop raw
            return 'skip', 0
        # else fall through: re-extract the old corrupt (no-scale, PSFC=inf) output
    ds = xr.open_dataset(f, decode_times=False, mask_and_scale=True)
    have = [v for v in LR_VARS if v in ds.variables]
    if len(have) != len(LR_VARS):
        ds.close(); return f'MISSING {set(LR_VARS)-set(have)}', 0
    if not os.path.exists(gridpath) and 'XLAT' in ds.variables:
        xl = ds['XLAT']; xo = ds['XLONG']
        np.savez(gridpath, XLAT=np.asarray(xl.isel({xl.dims[0]:0}) if xl.ndim==3 else xl).astype('float32'),
                            XLONG=np.asarray(xo.isel({xo.dims[0]:0}) if xo.ndim==3 else xo).astype('float32'))
    times = parse_times(ds)
    yr = np.array([int(t[:4]) if t[:4].isdigit() else -1 for t in times])
    keep = (yr >= y0) & (yr <= y1)
    n = int(keep.sum())
    if n == 0:
        ds.close()
        if delete and os.path.exists(f): os.remove(f)          # outside 2030-2050 -> drop raw
        return 'nokeep', 0
    arr = np.stack([(np.asarray(ds[v].values)[keep]*SCALE[i]).astype('float16') for i,v in enumerate(LR_VARS)], axis=1)  # (n,7,299,424); PSFC in hPa
    ds.close()
    tmp = out + '.tmp.npz'
    np.savez(tmp, data=arr, times=times[keep], vars=np.array(LR_VARS), scale=SCALE)   # 'scale': divide-back multipliers (PSFC hPa->Pa = x100)
    chk = np.load(tmp); ok = (chk['data'].shape[0] == n and chk['data'].shape[1] == 7)   # verify before deleting raw
    del chk
    if not ok:
        os.remove(tmp); return 'VERIFY_FAIL', 0
    os.replace(tmp, out)
    if delete and os.path.exists(f): os.remove(f)
    return 'ok', n

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--scenario', required=True); ap.add_argument('--indir', required=True); ap.add_argument('--outdir', required=True)
    ap.add_argument('--y0', type=int, default=2030); ap.add_argument('--y1', type=int, default=2050)
    ap.add_argument('--watch', action='store_true', help='keep polling for new files (daemon)')
    ap.add_argument('--poll', type=int, default=60); ap.add_argument('--delete', action='store_true', help='delete raw .nc after successful extract')
    ap.add_argument('--shard', default='0/1', help='race-free parallel workers on the same dir: "i/N" -> this worker handles files where crc32(basename)%N==i')
    a = ap.parse_args()
    si, sn = (int(x) for x in a.shard.split('/'))           # shard index / count
    os.makedirs(a.outdir, exist_ok=True); gridpath = f"{a.outdir}/tgw_grid.npz"
    seen_empty = 0
    while True:
        files = sorted(glob.glob(f"{a.indir}/*.nc"))
        if sn > 1: files = [f for f in files if zlib.crc32(os.path.basename(f).encode()) % sn == si]  # deterministic partition, no locks
        did = 0; tot = 0
        for f in files:
            # skip files still being written by Globus (size changing / .tmp); process stable ones
            try:
                if time.time()-os.path.getmtime(f) < 25: continue  # Globus may still be writing
                s0 = os.path.getsize(f); time.sleep(0.2)
                if os.path.getsize(f) != s0: continue
            except OSError:
                continue
            try:
                st, n = process(f, a.outdir, a.scenario, a.y0, a.y1, gridpath, a.delete)
            except Exception:
                print(f"[ERR] {os.path.basename(f)}\n{traceback.format_exc()}", flush=True); continue
            if st == 'ok': did += 1; tot += n; print(f"[ok] {os.path.basename(f)} +{n} frames -> kept; raw {'deleted' if a.delete else 'kept'}", flush=True)
            elif st not in ('skip','nokeep'): print(f"[{st}] {os.path.basename(f)}", flush=True)
        done = len(glob.glob(f"{a.outdir}/tgw_{a.scenario}_*.npz"))
        print(f"=== pass: processed {did} new files (+{tot} frames); total outputs {done} ===", flush=True)
        if not a.watch: break
        if did == 0:
            seen_empty += 1
            if seen_empty >= 5: print("[watch] 5 idle passes — still watching (Ctrl-C to stop)", flush=True); seen_empty = 0
        else: seen_empty = 0
        time.sleep(a.poll)

if __name__ == '__main__':
    main()
