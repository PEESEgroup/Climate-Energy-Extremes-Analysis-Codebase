"""Stage-1: preprocess the z-scored 21ch LR ONCE (identical across all 7 snapshots).
Byte-identical preprocessing to 03_tgw3d_srgan_gen.py PHASE=infer (build_lr/_interp/regrid/z-score),
except regrid is done with a scipy-sparse matmul that is SELF-CHECKED against the verbatim bincount
loop on the first frame (assert max|diff|<1e-4). Output: /data/gen_targets/srgan3d_val/cache/
  zc_tgw2019.npy  [NH,21,89,211] float32 (nan->0 z-scored, exactly what G eats)
  meta_tgw2019.npz  stamps, plants, ba, cap, lat, lon, hub(HUBp), cells(plant HR cell idx), psd_idx
Run: CUDA_VISIBLE_DEVICES='' /opt/pytorch/bin/python 02_cache_lr.py
"""
import os, glob, time, numpy as np, pandas as pd
import torch, torch.nn.functional as F
import scipy.sparse as sp

OUT   = "/data/gen_targets/srgan3d_val"
CACHE = f"{OUT}/cache"; os.makedirs(CACHE, exist_ok=True)
GC    = "/data/datasets/gen/tgw-gen-historical"
D3    = "/data/tgw_3d"
TRN   = "/data/datasets/train"
GRID  = "/data/datasets/grid"
YEAR  = os.environ.get("YEAR", "2019")
TSTRIDE = int(os.environ.get("TSTRIDE", "2"))
NHLIM = int(os.environ.get("NH_LIMIT", "0"))
TAG   = os.environ.get("TAG", "tgw2019")
NPSD  = int(os.environ.get("NPSD", "24"))

HEIGHTS = np.array([10,40,80,110,140,160,200], np.float32)
co  = np.load(f"{GRID}/coordinate.npz"); LAT = co["lat"].astype("float64"); LON = co["lon"].astype("float64")
assert LAT.shape==(625,) and LON.shape==(1475,)

def cell_hr(lat, lon):
    ih = np.abs(LAT[None,:]-np.asarray(lat,float)[:,None]).argmin(1)
    iw = np.abs(LON[None,:]-np.asarray(lon,float)[:,None]).argmin(1)
    return ih*1475+iw

def load_wind():
    c = pd.read_csv(f"{GC}/eia_wind_configs.csv", dtype={"plant_code_unique":str}).drop_duplicates("plant_code_unique")
    c = c[np.isfinite(c.lat)&np.isfinite(c.lon)]
    c = c[(c.lat>=LAT.min())&(c.lat<=LAT.max())&(c.lon>=LON.min())&(c.lon<=LON.max())]
    return c.reset_index(drop=True)

def _interp(arr, z, h):                               # arr (8,H,W), z (8,H,W) -- verbatim c404 _interp
    NLz=arr.shape[0]
    kk=np.clip((z<=h).sum(0)-1,0,NLz-2)[None]
    z0=np.take_along_axis(z,kk,0)[0]; z1=np.take_along_axis(z,kk+1,0)[0]
    a0=np.take_along_axis(arr,kk,0)[0]; a1=np.take_along_axis(arr,kk+1,0)[0]
    w=(h-z0)/np.where(z1>z0,z1-z0,1.0)
    out=a0+w*(a1-a0)
    out=np.where(h<z[0],arr[0],out)
    out=np.where(h>=z[-1],np.nan,out)
    return out.astype(np.float32)

def build_lr(uv_t, zagl_t, surf_t, pwat_t):           # per-timestep, all f32; returns (21,299,424)
    u=uv_t[:,0]; v=uv_t[:,1]
    wind=[]
    for h in HEIGHTS:
        wind.append(_interp(u,zagl_t,h)); wind.append(_interp(v,zagl_t,h))     # level-major u,v
    wind=np.stack(wind)
    ctx=np.stack([surf_t[0], surf_t[1], surf_t[2], surf_t[3],                  # u10,v10,t2,psfc(hPa)
                  pwat_t/1000.0, surf_t[6], surf_t[5]])                        # pwat(m), short=SWDOWN, long=GLW
    return np.concatenate([wind,ctx],0)

def file_list():
    return sorted(glob.glob(f"{D3}/tgw_wrf_historical_three_hourly_{YEAR}-*.npz"))

def select_stamps():
    seen=[]; sset=set()
    for fp in file_list():
        z=np.load(fp, allow_pickle=True); tms=[str(t) for t in z["times"]]; z.close()
        for s in tms:
            if s[:4]==YEAR and s not in sset: sset.add(s); seen.append(s)
    seen=sorted(seen); sel=seen[::TSTRIDE]
    if NHLIM: sel=sel[:NHLIM]
    return sel

# ---- regrid setup (verbatim from PHASE=infer) ----
def pool1d(a,n): return F.adaptive_avg_pool1d(torch.tensor(np.asarray(a,dtype='float64'))[None,None],n)[0,0].numpy()
def cell_edges(c): m=0.5*(c[1:]+c[:-1]); return np.concatenate([[2*c[0]-m[0]],m,[2*c[-1]-m[-1]]])
LATLR=pool1d(co['lat'],89); LONLR=pool1d(co['lon'],211); LATE=cell_edges(LATLR); LONE=cell_edges(LONLR)
NLAT,NLON=89,211
_g=np.load(f"{D3}/tgw3d_grid.npz"); _XLAT=_g['XLAT'].astype('float64').ravel(); _XLONG=_g['XLONG'].astype('float64').ravel()
_bil=np.digitize(_XLAT,LATE)-1; _bio=np.digitize(_XLONG,LONE)-1
VALID=(_bil>=0)&(_bil<NLAT)&(_bio>=0)&(_bio<NLON)
CELL=(_bil*NLON+_bio)[VALID]; COUNTS=np.bincount(CELL,minlength=NLAT*NLON).astype('float64'); NZ=COUNTS>0

def regrid_bincount(arr):    # verbatim
    C=arr.shape[0]; out=np.full((C,NLAT*NLON),np.nan); af=arr.reshape(C,-1)[:,VALID]
    for ch in range(C):
        s=np.bincount(CELL,weights=af[ch],minlength=NLAT*NLON); out[ch,NZ]=s[NZ]/COUNTS[NZ]
    return out.reshape(C,NLAT,NLON)

# sparse M [NCELL x NValid], M[cell,j]=1
_nval=int(VALID.sum())
M = sp.csr_matrix((np.ones(_nval), (CELL, np.arange(_nval))), shape=(NLAT*NLON, _nval))
def regrid_sparse(arr):
    C=arr.shape[0]; af=arr.reshape(C,-1)[:,VALID]        # [C,NValid]
    s = (M @ af.T)                                        # [NCELL, C]
    out=np.full((C,NLAT*NLON),np.nan)
    out[:,NZ]=(s[NZ,:]/COUNTS[NZ,None]).T
    return out.reshape(C,NLAT,NLON)

st=np.load(f"{TRN}/norm_stats_3d.npz")
lr_mean=st["lr_mean"].astype(np.float32).reshape(21,1,1); lr_std=st["lr_std"].astype(np.float32).reshape(21,1,1)

SEL=select_stamps(); NH=len(SEL); pos={s:i for i,s in enumerate(SEL)}
psd_idx=np.unique(np.linspace(0,NH-1,NPSD).astype(int))
print(f"[cache] TGW3D {YEAR} TSTRIDE={TSTRIDE} -> {NH} frames  npsd={len(psd_idx)}", flush=True)
P=load_wind(); cells=cell_hr(P.lat.values,P.lon.values); npl=len(P); HUBp=P.wind_turbine_hub_ht.values.astype(np.float32)

zc_all=np.lib.format.open_memmap(f"{CACHE}/zc_{TAG}.npy", mode="w+", dtype=np.float32, shape=(NH,21,NLAT,NLON))
done=np.zeros(NH,bool)
t0=time.time(); nproc=0; checked=False; use_sparse=True
for fp in file_list():
    z=np.load(fp, allow_pickle=True); tms=[str(t) for t in z["times"]]
    need=[(ti,s) for ti,s in enumerate(tms) if s in pos and not done[pos[s]]]
    if not need: z.close(); continue
    uv=z["uv"].astype(np.float32); zagl=z["zagl"].astype(np.float32); surf=z["surf"].astype(np.float32); pwat=z["pwat"].astype(np.float32); z.close()
    for ti,s in need:
        lr=build_lr(uv[ti],zagl[ti],surf[ti],pwat[ti])
        if not checked:
            rb=regrid_bincount(lr); rs=regrid_sparse(lr)
            d=np.nanmax(np.abs(rb-rs)); nanmis=int((np.isnan(rb)!=np.isnan(rs)).sum())
            print(f"[check] regrid sparse-vs-bincount max|diff|={d:.3e}  nan-mismatch={nanmis}", flush=True)
            assert d<1e-4 and nanmis==0, "sparse regrid mismatch -> abort"
            checked=True
            lrr=rb
            cm=np.nanmean(lrr.reshape(21,-1),1)
            print(f"[diag] ctx means u10 {cm[14]:.2f} v10 {cm[15]:.2f} t2 {cm[16]:.1f} psfc {cm[17]:.1f} pwat {cm[18]:.4f} short {cm[19]:.1f} long {cm[20]:.1f} | lr_mean pwat {lr_mean[18,0,0]:.4f} | nanfrac {np.isnan(lrr).mean():.3f}", flush=True)
        else:
            lrr=regrid_sparse(lr)
        zc=np.nan_to_num((lrr-lr_mean)/lr_std,nan=0.0).astype(np.float32)
        j=pos[s]; zc_all[j]=zc; done[j]=True; nproc+=1
        if nproc%100==0: print(f"  {nproc}/{NH}  {time.time()-t0:.0f}s  ({(time.time()-t0)/nproc:.2f}s/frame)", flush=True)
zc_all.flush()
assert done.all(), f"missing {int((~done).sum())} frames"
np.savez(f"{CACHE}/meta_{TAG}.npz", stamps=np.array(SEL), plants=P.plant_code_unique.values, ba=P.ba.values,
         cap=P.system_capacity.values, lat=P.lat.values, lon=P.lon.values, hub=HUBp, cells=cells, psd_idx=psd_idx)
print(f"[cache] DONE {NH} frames in {time.time()-t0:.0f}s -> {CACHE}/zc_{TAG}.npy + meta_{TAG}.npz", flush=True)
