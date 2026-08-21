"""Regrid the 4 TGW FUTURE scenarios (2030-2050, 1092 files each) -> LR 89x211, per scenario.
Grid is IDENTICAL to historical (verified) so reuse the same conservative area-average index.
Output per scenario: /data/tgw_future_lr/<scen>.dat (Ntot,7,89,211 f32 physical, psfc Pa, empty->NaN)
                     /data/tgw_future_lr/<scen>_meta.npz (years,months,offs,N,shape,stamps)
Run: /opt/pytorch/bin/python 05_regrid_future.py [nworkers]
"""
import numpy as np, glob, sys, time, os
from scipy.sparse import csr_matrix
from multiprocessing import Pool
import regrid_tgw as R          # sits beside this file in 01_acquire_data
NLAT, NLON = 89, 211
GRID = "/data/tgw_hist/tgw_grid.npz"           # == future grid (XLAT/XLONG identical, verified)
SCENS = ['rcp45cooler', 'rcp45hotter', 'rcp85cooler', 'rcp85hotter']
OUTDIR = "/data/tgw_future_lr"; os.makedirs(OUTDIR, exist_ok=True)
_W = {}
def init():
    valid, cell, cc = R.build_index(GRID); nvalid = int(valid.sum())
    _W['valid'] = valid
    _W['M'] = csr_matrix((1.0 / cc[cell], (cell, np.arange(nvalid))), shape=(NLAT * NLON, nvalid))
    _W['empty'] = (cc == 0).reshape(NLAT, NLON); _W['dat'] = None; _W['Ntot'] = 0

def worker(a):
    f, off, n, dat, Ntot = a
    z = np.load(f); arr = z['data'].astype('float32') / z['scale'].astype('float32')[None, :, None, None]
    flat = arr.reshape(n * 7, 299 * 424)[:, _W['valid']]
    lr = np.asarray(flat @ _W['M'].T).reshape(n, 7, NLAT, NLON).astype('float32')
    lr[:, :, _W['empty']] = np.nan
    out = np.memmap(dat, dtype='float32', mode='r+', shape=(Ntot, 7, NLAT, NLON))
    out[off:off + n] = lr; out.flush(); del out; return n

if __name__ == '__main__':
    NW = int(sys.argv[1]) if len(sys.argv) > 1 else 16
    for scen in SCENS:
        t0 = time.time()
        HF = sorted(f for f in glob.glob(f"/data/tgw_extract/{scen}/*.npz") if "grid" not in f)
        cnts = np.zeros(len(HF), np.int64); stamps = []
        for i, f in enumerate(HF):
            t = np.load(f)['times'].astype(str); cnts[i] = len(t); stamps.append(t)
        offs = np.concatenate([[0], np.cumsum(cnts)]).astype(np.int64); Ntot = int(offs[-1])
        stamps = np.concatenate(stamps)
        years = np.array([int(s[:4]) for s in stamps]); months = np.array([int(s[4:6]) for s in stamps])
        dat = f"{OUTDIR}/{scen}.dat"
        np.memmap(dat, dtype='float32', mode='w+', shape=(Ntot, 7, NLAT, NLON)).flush()
        np.savez(f"{OUTDIR}/{scen}_meta.npz", years=years, months=months, offs=offs, N=Ntot,
                 shape=(Ntot, 7, NLAT, NLON), stamps=stamps)
        print(f"[{scen}] {len(HF)} files -> {Ntot} frames ({Ntot*7*NLAT*NLON*4/1e9:.1f} GB), {NW} workers", flush=True)
        args = [(HF[i], int(offs[i]), int(cnts[i]), dat, Ntot) for i in range(len(HF))]
        done = 0
        with Pool(NW, initializer=init) as p:
            for _ in p.imap_unordered(worker, args):
                done += 1
                if done % 200 == 0: print(f"  [{scen}] {done}/{len(HF)} ({time.time()-t0:.0f}s)", flush=True)
        m = np.memmap(dat, dtype='float32', mode='r', shape=(Ntot, 7, NLAT, NLON))
        t2 = m[Ntot // 2, 4]
        print(f"[{scen}] DONE {time.time()-t0:.0f}s | mid T2 {np.nanmin(t2):.1f}/{np.nanmean(t2):.1f}/{np.nanmax(t2):.1f}K", flush=True)
    print("ALL SCENARIOS DONE", flush=True)
