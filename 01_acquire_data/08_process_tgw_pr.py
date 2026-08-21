"""Stream-process TGW weekly NetCDF -> compact fp16 PRECIP/SNOW store, then DELETE raw.
Companion to 01_process_tgw.py (which does the 7 SRGAN vars). These channels are NEVER super-resolved
(no SRGAN channel) -> kept at TGW-native 12 km (299x424); aggregated to subregion/basin downstream.

3 channels stored (Time, south_north=299, west_east=424), all fp16:
  [0] precip  mm/hr = hourly d(RAINC+RAINNC+RAINSH)   -- run-total accumulators, de-accum in fp32 (they
                                                          OVERFLOW fp16 raw: RAINNC~9e4 > 65504)
  [1] swe     mm (kg/m2) = SNOW                        -- INSTANTANEOUS state (not accumulated), stored direct
  [2] snowh   m = SNOWH                                -- INSTANTANEOUS snow depth (bonus; cold-hazard/snow-load)

De-accum: within-file diff for t>=1; hour 0 = persistence(=hour1). This is robust to (a) parallel/out-of-order
shard processing (no cross-file state needed) and (b) any accumulator reset at WRF restarts (never diffed across a
file seam). Negatives clipped to 0. ~0.6% of hours (one per weekly file) use the persistence approx -> negligible
for daily/basin aggregation. All 3 channels fp16-safe after de-accum (precip<~200 mm/hr, swe<~2e4, snowh<~110).

Usage (one per scenario; historical is just scenario=historical y0=1980 y1=2019):
  python 08_process_tgw_pr.py --scenario historical \
      --indir /data/tgw_raw/historical --outdir /data/tgw_precip/historical \
      --y0 1980 --y1 2019 --watch --delete --shard 0/12
Idempotent/resumable: skips files whose output npz already exists; safe to restart / run many shards."""
import os, sys, glob, time, argparse, traceback, zlib, numpy as np, xarray as xr

PRECIP_ACC = ['RAINC', 'RAINNC', 'RAINSH']   # run-total accumulators (mm); summed then de-accumulated
CHANNELS   = ['precip', 'swe', 'snowh']       # output channel order

def parse_times(ds):
    """YYYYMMDDHH per Time step from WRF 'Times' char array (robust to layout)."""
    if 'Times' in ds.variables:
        raw = ds['Times'].values; out = []
        for row in raw:
            if isinstance(row, bytes):          s = row.decode()
            elif getattr(row, 'ndim', 0) >= 1:  s = b''.join([c if isinstance(c, bytes) else c.tobytes() for c in row]).decode(errors='ignore')
            else:                                s = str(row)
            d = ''.join(ch for ch in s if ch.isdigit()); out.append(d[:10])
        return np.array(out)
    tc = 'Time' if 'Time' in ds.coords else ('time' if 'time' in ds.coords else None)
    return np.array([np.datetime_as_string(x, unit='h').translate(str.maketrans('', '', '-T:'))[:10] for x in ds[tc].values])

def process(f, outdir, scen, y0, y1, gridpath, delete):
    out = f"{outdir}/tgw_pr_{scen}_{os.path.basename(f).replace('.nc','')}.npz"
    if os.path.exists(out):
        try: _z = np.load(out); _ok = ('channels' in _z.files); _z.close()
        except Exception: _ok = False
        if _ok:
            if delete and os.path.exists(f): os.remove(f)
            return 'skip', 0
    ds = xr.open_dataset(f, decode_times=False, mask_and_scale=True)
    # need at least one precip accumulator + SNOW + SNOWH
    acc = [v for v in PRECIP_ACC if v in ds.variables]
    if not acc or 'SNOW' not in ds.variables or 'SNOWH' not in ds.variables:
        ds.close(); return f'MISSING precip/snow (acc={acc}, SNOW={"SNOW" in ds.variables}, SNOWH={"SNOWH" in ds.variables})', 0
    if not os.path.exists(gridpath) and 'XLAT' in ds.variables:
        xl = ds['XLAT']; xo = ds['XLONG']
        np.savez(gridpath, XLAT=np.asarray(xl.isel({xl.dims[0]:0}) if xl.ndim==3 else xl).astype('float32'),
                            XLONG=np.asarray(xo.isel({xo.dims[0]:0}) if xo.ndim==3 else xo).astype('float32'))
    times = parse_times(ds)
    nt = len(times)
    # --- precip: sum accumulators (fp32), de-accumulate within-file, persistence hour-0, clip>=0 ---
    pacc = np.zeros((nt, 299, 424), dtype='float32')
    for v in acc: pacc += np.asarray(ds[v].values, dtype='float32')
    precip = np.empty_like(pacc)
    precip[1:] = pacc[1:] - pacc[:-1]
    precip[0]  = precip[1] if nt > 1 else 0.0     # persistence for the seam hour
    np.clip(precip, 0.0, None, out=precip)        # accumulators monotonic; guard fp noise / restart resets
    # --- swe, snowh: instantaneous state, stored direct ---
    swe   = np.asarray(ds['SNOW'].values,  dtype='float32')
    snowh = np.asarray(ds['SNOWH'].values, dtype='float32')
    ds.close()
    # year filter applied AFTER de-accum (so the diff never straddles a dropped boundary hour)
    yr = np.array([int(t[:4]) if t[:4].isdigit() else -1 for t in times])
    keep = (yr >= y0) & (yr <= y1); n = int(keep.sum())
    if n == 0:
        if delete and os.path.exists(f): os.remove(f)
        return 'nokeep', 0
    arr = np.stack([precip[keep], swe[keep], snowh[keep]], axis=1).astype('float16')  # (n,3,299,424)
    if not np.isfinite(arr).all():
        return f'NONFINITE (max|precip|={np.nanmax(np.abs(precip)):.3g} swe={np.nanmax(swe):.3g})', 0
    tmp = out + '.tmp.npz'
    np.savez(tmp, data=arr, times=times[keep], channels=np.array(CHANNELS),
             note=np.array('precip=mm/hr deaccum(RAINC+RAINNC+RAINSH); swe=SNOW mm inst; snowh=SNOWH m inst; NEVER super-resolved'))
    chk = np.load(tmp); ok = (chk['data'].shape[0] == n and chk['data'].shape[1] == 3); del chk
    if not ok:
        os.remove(tmp); return 'VERIFY_FAIL', 0
    os.replace(tmp, out)
    if delete and os.path.exists(f): os.remove(f)
    return 'ok', n

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--scenario', required=True); ap.add_argument('--indir', required=True); ap.add_argument('--outdir', required=True)
    ap.add_argument('--y0', type=int, required=True); ap.add_argument('--y1', type=int, required=True)
    ap.add_argument('--watch', action='store_true'); ap.add_argument('--poll', type=int, default=60)
    ap.add_argument('--delete', action='store_true', help='delete raw .nc after successful extract')
    ap.add_argument('--shard', default='0/1', help='"i/N": handle files where crc32(basename)%N==i (race-free parallel)')
    ap.add_argument('--once', action='store_true', help='single pass then exit (alias for no --watch)')
    a = ap.parse_args()
    si, sn = (int(x) for x in a.shard.split('/'))
    os.makedirs(a.outdir, exist_ok=True); gridpath = f"{a.outdir}/tgw_grid.npz"
    seen_empty = 0
    while True:
        files = sorted(glob.glob(f"{a.indir}/*.nc"))
        if sn > 1: files = [f for f in files if zlib.crc32(os.path.basename(f).encode()) % sn == si]
        did = 0; tot = 0
        for f in files:
            try:
                if time.time()-os.path.getmtime(f) < 25: continue    # Globus may still be writing
                s0 = os.path.getsize(f); time.sleep(0.2)
                if os.path.getsize(f) != s0: continue
            except OSError:
                continue
            try:
                st, n = process(f, a.outdir, a.scenario, a.y0, a.y1, gridpath, a.delete)
            except Exception:
                print(f"[ERR] {os.path.basename(f)}\n{traceback.format_exc()}", flush=True); continue
            if st == 'ok': did += 1; tot += n; print(f"[ok] {os.path.basename(f)} +{n} frames; raw {'deleted' if a.delete else 'kept'}", flush=True)
            elif st not in ('skip','nokeep'): print(f"[{st}] {os.path.basename(f)}", flush=True)
        done = len(glob.glob(f"{a.outdir}/tgw_pr_{a.scenario}_*.npz"))
        print(f"=== [{a.scenario} shard {si}/{sn}] pass: {did} new (+{tot} frames); total outputs {done} ===", flush=True)
        if not a.watch or a.once: break
        if did == 0:
            seen_empty += 1
            if seen_empty >= 5: print(f"[watch {si}/{sn}] idle — still watching", flush=True); seen_empty = 0
        else: seen_empty = 0
        time.sleep(a.poll)

if __name__ == '__main__':
    main()
