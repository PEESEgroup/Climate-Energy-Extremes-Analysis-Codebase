"""Stream-process TGW 3-hourly 3D weekly NetCDF -> compact fp16 store of near-surface WIND PROFILE
(destaggered U,V on the bottom NLEV mass levels + their AGL heights) plus surface predictors/stability,
then DELETE the 9.48 GB raw file. Mirrors 01_process_tgw.py (watch/delete/shard/idempotent).

Stored per file (npz, compressed):
  uv     (T, NLEV, 2, 299, 424) fp16  -- destaggered [u,v] on bottom NLEV mass levels (native SI m/s)
  zagl   (T, NLEV, 299, 424)   fp16  -- AGL height (m) of those mass levels  = (PH+PHB)/9.81 - HGT, mass-centred
  surf   (T, NSURF, 299, 424)  fp16  -- SURF vars; PSFC stored in hPa (x0.01) to avoid fp16 overflow
  pwat   (T, 299, 424)         fp16  -- column-integrated water vapour (mm) = (1/g) integral q dp; NaN if QVAPOR absent
  times  (T,) 'YYYYMMDDHH'  ;  surf_vars, scale, nlev
Grid saved once: tgw3d_grid.npz {XLAT, XLONG, LU_INDEX, HGT}.
"""
import os, sys, glob, time, argparse, traceback, zlib, numpy as np, xarray as xr

NLEV  = 8
SURF  = ['U10','V10','T2','PSFC','Q2','GLW','SWDOWN','UST','PBLH','HFX']   # predictors + stability(ablation)
SCALE = np.array([1.,1.,1.,0.01,1.,1.,1.,1.,1.,1.], dtype='float32')       # PSFC Pa->hPa so it fits fp16
G = 9.81

def parse_times(ds):
    if 'Times' in ds.variables:
        raw = ds['Times'].values; out=[]
        for row in raw:
            if isinstance(row, bytes): s=row.decode()
            elif getattr(row,'ndim',0)>=1: s=b''.join([c if isinstance(c,bytes) else c.tobytes() for c in row]).decode(errors='ignore')
            else: s=str(row)
            d=''.join(ch for ch in s if ch.isdigit()); out.append(d[:10])
        return np.array(out)
    tc='Time' if 'Time' in ds.coords else ('time' if 'time' in ds.coords else None)
    return np.array([np.datetime_as_string(x,unit='h').translate(str.maketrans('','','-T:'))[:10] for x in ds[tc].values])

def column_pwat(ds, T, ny, nx):
    """(1/g) integral of specific humidity over pressure -> precipitable water (kg/m2 == mm).
    Interface pressures from PSFC + midpoint averaging of full mass-level pressure (P+PB).
    Returns a NaN field if the 3-D moisture/pressure vars are absent."""
    if not all(v in ds.variables for v in ['QVAPOR','P','PB','PSFC']):
        return np.full((T,ny,nx), np.nan, dtype='float16')
    try:
        QV = np.asarray(ds['QVAPOR'].values, dtype='float32')          # (T,NZ,Y,X) mixing ratio kg/kg
        pm = np.asarray(ds['P'].values, dtype='float32') + np.asarray(ds['PB'].values, dtype='float32')  # full p (Pa)
        psfc = np.asarray(ds['PSFC'].values, dtype='float32')          # (T,Y,X) Pa
        NZ = pm.shape[1]
        p_int = np.empty((T, NZ+1, ny, nx), dtype='float32')
        p_int[:,0]   = psfc
        p_int[:,1:NZ]= 0.5*(pm[:,:-1] + pm[:,1:])
        p_int[:,NZ]  = np.maximum(pm[:,-1] - 0.5*(pm[:,-2]-pm[:,-1]), 1.0)
        dp = p_int[:,:-1] - p_int[:,1:]                                # (T,NZ,Y,X) positive downward
        q  = QV/(1.0+QV)                                               # specific humidity
        pwat = (np.sum(q*dp, axis=1)/G).astype('float16')
        del QV, pm, q, dp, p_int
        return pwat
    except Exception:
        return np.full((T,ny,nx), np.nan, dtype='float16')

def process(f, outdir, delete, gridpath):
    out = f"{outdir}/{os.path.basename(f).replace('.nc','')}.npz"
    if os.path.exists(out):
        try: _z=np.load(out); ok=('uv' in _z.files and _z['uv'].shape[1]==NLEV and 'pwat' in _z.files); _z.close()
        except Exception: ok=False
        if ok:
            if delete and os.path.exists(f): os.remove(f)
            return 'skip',0
    ds = xr.open_dataset(f, decode_times=False)
    need = ['U','V','PH','PHB','HGT'] + SURF
    miss = [v for v in need if v not in ds.variables]
    if miss: ds.close(); return f'MISSING {miss}',0
    U = np.asarray(ds['U'].values)
    V = np.asarray(ds['V'].values)
    Um = 0.5*(U[:,:NLEV,:,:-1]+U[:,:NLEV,:,1:])
    Vm = 0.5*(V[:,:NLEV,:-1,:]+V[:,:NLEV,1:,:])
    uv = np.stack([Um,Vm],axis=2).astype('float16')
    PH=np.asarray(ds['PH'].values); PHB=np.asarray(ds['PHB'].values)
    HGT=np.asarray(ds['HGT'].values)
    z_stag=(PH+PHB)/G
    z_mass=0.5*(z_stag[:,:-1]+z_stag[:,1:])
    zagl=(z_mass[:,:NLEV]-HGT[:,None,:,:]).astype('float16')
    surf=np.stack([np.asarray(ds[v].values)*SCALE[i] for i,v in enumerate(SURF)],axis=1).astype('float16')
    T_,ny,nx = uv.shape[0], uv.shape[3], uv.shape[4]
    pwat = column_pwat(ds, T_, ny, nx)                                 # from 3-D moisture already in this file
    times=parse_times(ds)
    if not os.path.exists(gridpath):
        np.savez(gridpath,
                 XLAT=np.asarray(ds['XLAT'].isel(Time=0)).astype('float32'),
                 XLONG=np.asarray(ds['XLONG'].isel(Time=0)).astype('float32'),
                 LU_INDEX=np.asarray(ds['LU_INDEX'].isel(Time=0)).astype('float32'),
                 HGT=HGT[0].astype('float32'))
    ds.close()
    tmp=out+'.tmp.npz'
    np.savez_compressed(tmp, uv=uv, zagl=zagl, surf=surf, pwat=pwat, times=times,
                        surf_vars=np.array(SURF), scale=SCALE, nlev=NLEV)
    chk=np.load(tmp); good=(chk['uv'].shape[0]==len(times) and chk['uv'].shape[1]==NLEV); del chk
    if not good: os.remove(tmp); return 'VERIFY_FAIL',0
    os.replace(tmp,out)
    if delete and os.path.exists(f): os.remove(f)
    return 'ok',len(times)

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument('--indir',required=True); ap.add_argument('--outdir',required=True)
    ap.add_argument('--watch',action='store_true'); ap.add_argument('--poll',type=int,default=60)
    ap.add_argument('--delete',action='store_true'); ap.add_argument('--shard',default='0/1')
    a=ap.parse_args()
    si,sn=(int(x) for x in a.shard.split('/'))
    os.makedirs(a.outdir,exist_ok=True); gridpath=f"{a.outdir}/tgw3d_grid.npz"
    idle=0
    while True:
        files=sorted(glob.glob(f"{a.indir}/*.nc"))
        if sn>1: files=[f for f in files if zlib.crc32(os.path.basename(f).encode())%sn==si]
        did=tot=0
        for f in files:
            try:
                if time.time()-os.path.getmtime(f)<25: continue
                s0=os.path.getsize(f); time.sleep(0.2)
                if os.path.getsize(f)!=s0: continue
            except OSError: continue
            try: st,n=process(f,a.outdir,a.delete,gridpath)
            except Exception: print(f"[ERR] {os.path.basename(f)}\n{traceback.format_exc()}",flush=True); continue
            if st=='ok': did+=1; tot+=n; print(f"[ok] {os.path.basename(f)} +{n} steps -> kept; raw {'deleted' if a.delete else 'kept'}",flush=True)
            elif st not in ('skip',): print(f"[{st}] {os.path.basename(f)}",flush=True)
        done=len(glob.glob(f"{a.outdir}/tgw_wrf_*.npz"))
        print(f"=== pass: {did} new (+{tot} steps); total outputs {done} ===",flush=True)
        if not a.watch: break
        idle = idle+1 if did==0 else 0
        if idle and idle%5==0: print("[watch] idle — still watching",flush=True)
        time.sleep(a.poll)

if __name__=='__main__':
    main()
