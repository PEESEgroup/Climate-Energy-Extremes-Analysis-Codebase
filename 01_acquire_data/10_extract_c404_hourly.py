"""CONUS404 CONTINUOUS-HOURLY 3D+2D extraction for the TEMPORAL-SR NN (3h->1h hub wind). Same THREDDS/vinterp as
extract_c404_3d.py but pulls CONTINUOUS HOURLY windows (all 24h over consecutive days) across 4 seasons/years so we can
learn how the intermediate-hour hub wind deviates from the 3h-linear-interp as a function of the hourly surface. -> /data/c404_hourly."""
import os, time, numpy as np, xarray as xr, warnings, multiprocessing as mp
warnings.filterwarnings("ignore")
BASE="https://thredds.rda.ucar.edu/thredds/dodsC/files/g/d559000"; INV=f"{BASE}/INVARIANT/wrfconstants_usgs404.nc"
NL=15; HEIGHTS=np.array([10,40,80,110,140,160,200],dtype='float32'); PRED=['u10','v10','t2','rh','psfc','pwat','short','long']
OUT="/data/c404_hourly"; _G={}
# 4 continuous windows (yr, month, day-range) x all 24 hours -> spans winter/spring/summer/fall + interannual
WINDOWS=[(2013,1,range(8,12)),(2014,4,range(8,12)),(2016,7,range(8,12)),(2018,10,range(8,12))]
def wy(y,m): return y+1 if m>=10 else y
def _init():
    di=xr.open_dataset(INV,decode_times=False); _G['HGT']=np.asarray(di['HGT'].isel({di['HGT'].dims[0]:0}).values).astype('float32'); di.close()
def _interp(arr,z,h):
    out=np.full(arr.shape[1:],np.nan,np.float32)
    for k in range(arr.shape[0]-1):
        z0=z[k]; z1=z[k+1]; m=(z0<=h)&(z1>h)&(z1>z0); w=(h-z0)/np.where(z1>z0,z1-z0,1.0); vk=arr[k]+w*(arr[k+1]-arr[k]); out[m]=vk[m]
    below=h<z[0]
    if below.any(): out[below]=arr[0][below]
    return out
def _rh(q2,t2,psfc):
    e=q2*psfc/(0.622+q2); es=611.2*np.exp(17.67*(t2-273.15)/(t2-29.65)); return np.clip(100.0*e/es,0,100).astype(np.float32)
def _open(url,tries=3):
    for a in range(tries):
        try: return xr.open_dataset(url,decode_times=False)
        except Exception:
            if a==tries-1: raise
            time.sleep(6)
def one(ts):
    Y,M,D,H=ts; tag=f"{Y:04d}{M:02d}{D:02d}{H:02d}"; fp=f"{OUT}/c404h_{tag}.npz"
    if os.path.exists(fp): return ('skip',tag)
    WYd=wy(Y,M); ym=f"{Y:04d}{M:02d}"; stamp=f"{Y:04d}-{M:02d}-{D:02d}_{H:02d}:00:00"
    u3=f"{BASE}/wy{WYd}/{ym}/wrf3d_d01_{stamp}.nc"; u2=f"{BASE}/wy{WYd}/{ym}/wrf2d_d01_{stamp}.nc"
    try:
        ds=_open(u3); U=np.asarray(ds['U'].isel(Time=0,bottom_top=slice(0,NL)).values); V=np.asarray(ds['V'].isel(Time=0,bottom_top=slice(0,NL)).values)
        Z=np.asarray(ds['Z'].isel(Time=0,bottom_top_stag=slice(0,NL+1)).values); ds.close()
    except Exception as e: return (f'ERR3d {str(e)[:50]}',tag)
    Um=0.5*(U[:,:,:-1]+U[:,:,1:]); Vm=0.5*(V[:,:-1,:]+V[:,1:,:]); Zagl=0.5*(Z[:-1]+Z[1:])-_G['HGT'][None]
    hub=np.stack([np.stack([_interp(Um,Zagl,h),_interp(Vm,Zagl,h)]) for h in HEIGHTS],0).astype('float16')
    try:
        d2=_open(u2); g=lambda v: np.asarray(d2[v].isel(Time=0).values,np.float32)
        u10=g('U10'); v10=g('V10'); t2=g('T2'); q2=g('Q2'); psf=g('PSFC'); pw=g('PWAT'); short=np.clip(g('SWDOWN'),0,None); long=np.clip(g('GLW'),0,None); ust=g('UST'); pblh=g('PBLH'); d2.close()
        pred=np.stack([u10,v10,t2,_rh(q2,t2,psf),psf*0.01,pw,short,long],0).astype('float16'); abl=np.stack([ust,pblh],0).astype('float16')
    except Exception as e: return (f'ERR2d {str(e)[:50]}',tag)
    if pred.shape[1:]!=hub.shape[2:]: return (f'GRIDMISMATCH',tag)
    tmp=fp+'.tmp.npz'; np.savez_compressed(tmp, hub=hub, pred=pred, abl=abl, heights=HEIGHTS, pred_vars=np.array(PRED), abl_vars=np.array(['ust','pblh']), ts=tag); os.replace(tmp,fp); return ('ok',tag)
def main():
    os.makedirs(OUT,exist_ok=True)
    ts=[(Y,M,D,H) for (Y,M,days) in WINDOWS for D in days for H in range(24)]
    print(f"{len(ts)} continuous-hourly frames ({len(WINDOWS)} windows) -> {OUT}",flush=True)
    if not os.path.exists(f"{OUT}/c404h_grid.npz"):
        di=xr.open_dataset(INV,decode_times=False); pick=lambda v: np.asarray(di[v].isel({di[v].dims[0]:0}).values).astype('float32')
        np.savez(f"{OUT}/c404h_grid.npz", XLAT=pick('XLAT'), XLONG=pick('XLONG'), HGT=pick('HGT')); di.close()
    t0=time.time(); done=err=sk=0
    with mp.Pool(12,initializer=_init) as p:
        for i,(st,tag) in enumerate(p.imap_unordered(one,ts,chunksize=1)):
            if st=='ok': done+=1
            elif st=='skip': sk+=1
            else: err+=1; print(f"[{st}] {tag}",flush=True)
            if (i+1)%24==0 or i==len(ts)-1: print(f"  {i+1}/{len(ts)} | ok {done} sk {sk} err {err} | {(time.time()-t0)/60:.1f}min",flush=True)
    print(f"HOURLY_DONE ok {done} sk {sk} err {err} in {(time.time()-t0)/60:.1f}min",flush=True)
if __name__=='__main__': main()
