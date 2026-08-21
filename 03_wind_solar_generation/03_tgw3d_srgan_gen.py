"""3D-WIND SRGAN downstream validation on TGW DEPLOYMENT domain, YEAR 2019 (3-hourly).

DEPLOYMENT pipeline (the point of the 3D-SRGAN -> TGW future scenarios):
  TGW-3D npz (uv 8 lvls + zagl + surf + pwat, 299x424)
   -> interp uv to 7 heights [10,40,80,110,140,160,200]m using zagl  (c404 _interp: clamp<z0, nan>=ztop)
   -> 21ch LR = [u@h,v@h x7 level-major] + [u10,v10,t2,psfc(hPa),pwat(m),short,long]
   -> area-average regrid TGW 299x424 -> SRGAN LR grid 89x211
   -> z-score with CONUS404 norm_stats_3d (lr_mean/lr_std)  == DEPLOYMENT (domain shift is REAL)
   -> 3D-SRGAN G_deploy (CPU-forced) -> 14ch hub u,v 4km (625x1475), denorm hr_mean/hr_std
   -> sample @ EIA wind plants -> speed profile -> interp to hub -> wsphub
  PySAM Windpower shear=0, system_capacity=max(pw) -> per-plant CF "s3d".

IMPORTANT: torch is imported ONLY inside PHASE=infer (run with /opt/pytorch, CUDA_VISIBLE_DEVICES='').
PHASE=pysam runs with /data/genenv (has PySAM, NO torch) -> must NOT import torch at module scope.

ENV: PHASE, CKPT, YEAR(2019), TSTRIDE(2=6-hourly), NH_LIMIT, TAG, THREADS, NPROC.
"""
import os, sys, glob, time, numpy as np, pandas as pd
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "02_downscale_wind"))

OUT   = "/data/gen_targets/srgan3d_val"; os.makedirs(OUT, exist_ok=True)
GC    = "/data/datasets/gen/tgw-gen-historical"
D3    = "/data/tgw_3d"
TRN   = "/data/datasets/train"
GRID  = "/data/datasets/grid"
PHASE = os.environ.get("PHASE", "infer")
TAG   = os.environ.get("TAG", "tgw2019")
CKPT  = os.environ.get("CKPT", "/data/ckpt/g3d_full/G_deploy.pth")
YEAR  = os.environ.get("YEAR", "2019")
TSTRIDE = int(os.environ.get("TSTRIDE", "2"))
NHLIM = int(os.environ.get("NH_LIMIT", "0"))

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

# ============================================================ INFER (CPU; /opt/pytorch)
if PHASE=="infer":
    import torch, torch.nn.functional as F
    from model.srgan import SRGAN_g
    torch.set_num_threads(int(os.environ.get("THREADS","6")))
    assert os.environ.get("CUDA_VISIBLE_DEVICES","")=="", "force CUDA_VISIBLE_DEVICES='' (GPU is TRAINING)"
    def pool1d(a,n): return F.adaptive_avg_pool1d(torch.tensor(np.asarray(a,dtype='float64'))[None,None],n)[0,0].numpy()
    def cell_edges(c): m=0.5*(c[1:]+c[:-1]); return np.concatenate([[2*c[0]-m[0]],m,[2*c[-1]-m[-1]]])
    LATLR=pool1d(co['lat'],89); LONLR=pool1d(co['lon'],211); LATE=cell_edges(LATLR); LONE=cell_edges(LONLR)
    NLAT,NLON=89,211
    _g=np.load(f"{D3}/tgw3d_grid.npz"); _XLAT=_g['XLAT'].astype('float64').ravel(); _XLONG=_g['XLONG'].astype('float64').ravel()
    _bil=np.digitize(_XLAT,LATE)-1; _bio=np.digitize(_XLONG,LONE)-1
    VALID=(_bil>=0)&(_bil<NLAT)&(_bio>=0)&(_bio<NLON)
    CELL=(_bil*NLON+_bio)[VALID]; COUNTS=np.bincount(CELL,minlength=NLAT*NLON).astype('float64'); NZ=COUNTS>0
    def regrid(arr):
        C=arr.shape[0]; out=np.full((C,NLAT*NLON),np.nan); af=arr.reshape(C,-1)[:,VALID]
        for ch in range(C):
            s=np.bincount(CELL,weights=af[ch],minlength=NLAT*NLON); out[ch,NZ]=s[NZ]/COUNTS[NZ]
        return out.reshape(C,NLAT,NLON)

    _S=np.load(f"{GRID}/static_terrain_landmask.npz")
    hgt=_S["hgt"].astype("float32"); lmask=_S["landmask"].astype("float32"); lake=_S["lakemask"].astype("float32")
    island=((lmask>0.5)&(lake<0.5)).astype("float32"); hgtz=(hgt-hgt.mean())/(hgt.std()+1e-6)
    static_hr=np.stack([hgtz,island*2-1],0)
    static_lr=F.adaptive_avg_pool2d(torch.from_numpy(static_hr)[None],(89,211))[0].numpy()
    G=SRGAN_g(in_channels=21,out_channels=14,crop_size=(625,1475),
              static_lr=static_lr,static_hr=static_hr,dyn_lr_ch=0,dyn_hr_ch=0)
    G.load_state_dict(torch.load(CKPT,map_location="cpu"),strict=True); G.eval().to("cpu")
    st=np.load(f"{TRN}/norm_stats_3d.npz")
    lr_mean=st["lr_mean"].astype(np.float32).reshape(21,1,1); lr_std=st["lr_std"].astype(np.float32).reshape(21,1,1)
    hr_mean=st["hr_mean"].astype(np.float32).reshape(14,1); hr_std=st["hr_std"].astype(np.float32).reshape(14,1)

    SEL=select_stamps(); NH=len(SEL); pos={s:i for i,s in enumerate(SEL)}
    print(f"[infer] TGW3D {YEAR} TSTRIDE={TSTRIDE} -> {NH} frames  ckpt={CKPT}", flush=True)
    P=load_wind(); cells=cell_hr(P.lat.values,P.lon.values); npl=len(P); HUBp=P.wind_turbine_hub_ht.values.astype(np.float32)
    wsphub=np.full((npl,NH),np.nan,np.float32); wsp80=np.full((npl,NH),np.nan,np.float32); done=np.zeros(NH,bool)
    t0=time.time(); nproc=0; diag=True
    for fp in file_list():
        z=np.load(fp, allow_pickle=True); tms=[str(t) for t in z["times"]]
        need=[(ti,s) for ti,s in enumerate(tms) if s in pos and not done[pos[s]]]
        if not need: z.close(); continue
        uv=z["uv"].astype(np.float32); zagl=z["zagl"].astype(np.float32); surf=z["surf"].astype(np.float32); pwat=z["pwat"].astype(np.float32); z.close()
        for ti,s in need:
            lrr=regrid(build_lr(uv[ti],zagl[ti],surf[ti],pwat[ti]))
            if diag:
                cm=np.nanmean(lrr.reshape(21,-1),1)
                print(f"[diag] ctx means u10 {cm[14]:.2f} v10 {cm[15]:.2f} t2 {cm[16]:.1f} psfc {cm[17]:.1f} pwat {cm[18]:.4f} short {cm[19]:.1f} long {cm[20]:.1f} | lr_mean pwat {lr_mean[18,0,0]:.4f} | nanfrac {np.isnan(lrr).mean():.3f}", flush=True); diag=False
            zc=np.nan_to_num((lrr-lr_mean)/lr_std,nan=0.0).astype(np.float32)
            with torch.no_grad(): y=G(torch.from_numpy(zc)[None])[0].numpy()
            raw=y.reshape(14,-1)[:,cells]*hr_std+hr_mean
            u=raw[0::2]; v=raw[1::2]; sp=np.sqrt(u**2+v**2)
            j=pos[s]; wsp80[:,j]=sp[2]
            wsphub[:,j]=np.array([np.interp(HUBp[k],HEIGHTS,sp[:,k]) for k in range(npl)],np.float32)
            done[j]=True; nproc+=1
            if nproc%50==0: print(f"  {nproc}/{NH}  {time.time()-t0:.0f}s  ({(time.time()-t0)/nproc:.2f}s/frame)", flush=True)
        np.savez(f"{OUT}/s3d_wx_{TAG}.npz", plants=P.plant_code_unique.values, ba=P.ba.values, cap=P.system_capacity.values,
                 lat=P.lat.values, lon=P.lon.values, hub=HUBp, stamps=np.array(SEL), wsphub=wsphub, wsp80=wsp80, done=done, ckpt=CKPT)
    print(f"[infer] wrote s3d_wx_{TAG}.npz  done {int(done.sum())}/{NH}  wsphub mean {np.nanmean(wsphub):.2f}  wsp80 mean {np.nanmean(wsp80):.2f}", flush=True)

# ============================================================ PYSAM (/data/genenv; NO torch)
elif PHASE=="pysam":
    from gen_physics import wind_cf
    import multiprocessing as mp
    Z=np.load(f"{OUT}/s3d_wx_{TAG}.npz", allow_pickle=True)
    done=np.asarray(Z["done"],bool) if "done" in Z.files else np.ones(len(Z["stamps"]),bool)
    cfg=pd.read_csv(f"{GC}/eia_wind_configs.csv", dtype={"plant_code_unique":str}).drop_duplicates("plant_code_unique").set_index("plant_code_unique")
    plants=Z["plants"].astype(str); stamps=Z["stamps"].astype(str)[done]; whub_all=Z["wsphub"][:,done]
    n=len(stamps); L=((n+8759)//8760)*8760
    FULLy=pd.date_range("2015-01-01", periods=L, freq="h")
    _G={}
    def one(pi):
        p=plants[pi]; c=cfg.loc[p]
        ws=eval(c.wind_turbine_powercurve_windspeeds) if isinstance(c.wind_turbine_powercurve_windspeeds,str) else c.wind_turbine_powercurve_windspeeds
        pw=eval(c.wind_turbine_powercurve_powerout) if isinstance(c.wind_turbine_powercurve_powerout,str) else c.wind_turbine_powercurve_powerout
        base=dict(wind_turbine_hub_ht=float(c.wind_turbine_hub_ht),wind_turbine_rotor_diameter=float(c.wind_turbine_rotor_diameter),
                  system_capacity=float(max(pw)),wind_turbine_powercurve_windspeeds=list(ws),wind_turbine_powercurve_powerout=list(pw))
        whub=np.zeros(L,np.float32); whub[:n]=np.nan_to_num(_G["whub"][pi])
        t2c=np.full(L,15.0); pmb=np.full(L,1013.0)
        cf=wind_cf(FULLy,whub,t2c,pmb,float(c.lat),float(c.lon),{**base,"shear":0.0})[:n]
        return pi,cf
    _G["whub"]=whub_all; npl=len(plants)
    CF=np.full((npl,n),np.nan,np.float32); t0=time.time()
    with mp.Pool(int(os.environ.get("NPROC","4"))) as pool:
        for pi,cf in pool.imap_unordered(one, range(npl), chunksize=8): CF[pi]=cf
    np.savez(f"{OUT}/s3d_cf_{TAG}.npz", plants=plants, ba=Z["ba"], cap=Z["cap"], lat=Z["lat"], lon=Z["lon"],
             stamps=stamps, cf=CF, s3d=CF)
    print(f"[pysam] {time.time()-t0:.0f}s  n={n}  CF mean {np.nanmean(CF):.3f}  -> s3d_cf_{TAG}.npz", flush=True)
else:
    raise SystemExit(f"unknown PHASE={PHASE}")
