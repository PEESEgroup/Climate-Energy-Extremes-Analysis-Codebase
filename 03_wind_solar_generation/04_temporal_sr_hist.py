"""TEMPORAL super-resolution of hub WIND, 3h->1h (user 2026-07-17: super-res the WEATHER, GODEEEP=validation only).
Stage-1 PHYSICS hybrid: hub_h(t) = interp3h(gold_hub_wind)(t) * [ Usfc_h(t) / interp3h(Usfc)(t) ].
gold_hub_wind = TGW-3D vinterp to plant hub ht (3-hourly, accurate). Usfc_h = hourly 10m wind (full_lr). GODEEEP hourly
= independent validation (ramp recovery). PHASE=extract(tellenv)->pysam(genenv)->validate(tellenv). Year via YEAR env."""
import os, sys, glob, math, numpy as np, pandas as pd
import os as _os_rp
for _rp in ("04_demand_model", "07_hazard_calendar", "09_outage_attribution",
            "02_downscale_wind", "12_figures"):
    _ap = _os_rp.path.abspath(_os_rp.path.join(
        _os_rp.path.dirname(_os_rp.path.abspath(__file__)), "..", _rp))
    if _os_rp.path.isdir(_ap) and _ap not in sys.path:
        sys.path.insert(0, _ap)
YEAR=int(os.environ.get("YEAR","2019")); OUT=os.environ.get("OUT","/data/gen_targets/srgan_val"); GC="/data/datasets/gen/tgw-gen-historical"; PHASE=os.environ.get("PHASE","extract")
co=np.load("/data/datasets/grid/coordinate.npz"); LATg=co["lat"].astype(float); LONg=co["lon"].astype(float)
def load_wind():
    c=pd.read_csv(f"{GC}/eia_wind_configs.csv",dtype={"plant_code_unique":str}).drop_duplicates("plant_code_unique")
    c=c[np.isfinite(c.lat)&np.isfinite(c.lon)]; return c[(c.lat>=LATg.min())&(c.lat<=LATg.max())&(c.lon>=LONg.min())&(c.lon<=LONg.max())].reset_index(drop=True)
def corr(a,b):
    ok=np.isfinite(a)&np.isfinite(b)
    if ok.sum()<100 or a[ok].std()<1e-9 or b[ok].std()<1e-9: return np.nan
    return float(np.corrcoef(a[ok],b[ok])[0,1])

if PHASE=="extract":
    from scipy.spatial import cKDTree
    P=load_wind(); npl=len(P); HUB=P.wind_turbine_hub_ht.values.astype(np.float32)
    # --- 3-hourly gold hub wind via 3D vinterp at plant 12km cells ---
    g=np.load("/data/tgw_3d/tgw3d_grid.npz"); XLAT=g["XLAT"].astype(float); XLONG=g["XLONG"].astype(float)
    tree=cKDTree(np.c_[XLAT.ravel(),XLONG.ravel()]); _,cell=tree.query(np.c_[P.lat.values,P.lon.values]); cell=cell.astype(int)
    fs=sorted(glob.glob(f"/data/tgw_3d/tgw_wrf_historical_three_hourly_{YEAR}-*.npz"))
    print(f"{npl} plants, {len(fs)} {YEAR} 3D files",flush=True)
    ts3=[]; HUBW=[]
    for fp in fs:
        z=np.load(fp); uv=z["uv"].astype(np.float32); zagl=z["zagl"].astype(np.float32); times=[str(t) for t in z["times"]]; z.close()
        T,L=uv.shape[0],uv.shape[1]
        uvc=uv.reshape(T,L,2,-1)[:,:,:,cell]     # (T,L,2,npl)
        zc=zagl.reshape(T,L,-1)[:,:,cell]        # (T,L,npl)
        for t,tstamp in enumerate(times):
            if not tstamp.startswith(str(YEAR)): continue
            hu=np.full(npl,np.nan,np.float32); hv=np.full(npl,np.nan,np.float32)
            for k in range(L-1):
                z0=zc[t,k]; z1=zc[t,k+1]; m=(z0<=HUB)&(z1>HUB)&(z1>z0); w=(HUB-z0)/np.where(z1>z0,z1-z0,1.0)
                hu[m]=(uvc[t,k,0]+w*(uvc[t,k+1,0]-uvc[t,k,0]))[m]; hv[m]=(uvc[t,k,1]+w*(uvc[t,k+1,1]-uvc[t,k,1]))[m]
            below=HUB<zc[t,0]; hu[below]=uvc[t,0,0][below]; hv[below]=uvc[t,0,1][below]
            ts3.append(tstamp); HUBW.append(np.sqrt(hu*hu+hv*hv))
    o=np.argsort(ts3); ts3=np.array(ts3)[o]; HUBW=np.stack(HUBW)[o].T.astype(np.float32)   # (npl,T3)
    # --- hourly 10m wind at plants from full_lr (89x211 LR grid) ---
    latlr=np.array([float(x) for x in np.interp(np.linspace(0,len(LATg)-1,89),np.arange(len(LATg)),LATg)])
    lonlr=np.array([float(x) for x in np.interp(np.linspace(0,len(LONg)-1,211),np.arange(len(LONg)),LONg)])
    ih=np.abs(latlr[None,:]-P.lat.values[:,None]).argmin(1); iw=np.abs(lonlr[None,:]-P.lon.values[:,None]).argmin(1); lrcell=ih*211+iw
    M=np.load("/data/tgw_hist/full_lr_meta.npz",allow_pickle=True); st=M["stamps"].astype(str); N=int(M["N"]); LR=np.memmap("/data/tgw_hist/full_lr.dat","float32","r",shape=(N,7,89,211))
    yrmask=np.array([s.startswith(str(YEAR)) for s in st]); yidx=np.where(yrmask)[0]; hstamps=st[yidx]
    ho=np.argsort(hstamps); hstamps=hstamps[ho]; yidx=yidx[ho]
    u10=LR[yidx][:,0].reshape(len(yidx),-1)[:,lrcell]; v10=LR[yidx][:,1].reshape(len(yidx),-1)[:,lrcell]   # (Th,npl)
    surf_h=np.sqrt(u10**2+v10**2).T.astype(np.float32)   # (npl,Th)
    # --- time axes + hybrid ---
    def hrs(s): return float((np.datetime64(f"{s[:4]}-{s[4:6]}-{s[6:8]}T{s[8:10]}")-np.datetime64(f"{YEAR}-01-01T00"))/np.timedelta64(1,"h"))
    t3=np.array([hrs(s) for s in ts3]); th=np.array([hrs(s) for s in hstamps]); Th=len(th)
    gold_lin=np.empty((npl,Th),np.float32); surf_lin=np.empty((npl,Th),np.float32)
    for i in range(npl):
        gold_lin[i]=np.interp(th,t3,HUBW[i]); surf_lin[i]=np.interp(th,t3,np.interp(t3,th,surf_h[i]))
    Mfac=np.clip(surf_h/np.maximum(surf_lin,0.1),0.3,3.0)
    hub_hybrid=(gold_lin*Mfac).astype(np.float32)   # temporal-SR hub wind
    np.savez(f"{OUT}/tempsr_wx_{YEAR}.npz",plants=P.plant_code_unique.values,ba=P.ba.values,cap=P.system_capacity.values,
             lat=P.lat.values,lon=P.lon.values,hub=HUB,stamps=hstamps,hub_linear=gold_lin,hub_hybrid=hub_hybrid,surf_h=surf_h)
    print(f"[extract] T3={len(t3)} Th={Th}; hub_hybrid mean {np.nanmean(hub_hybrid):.2f} linear {np.nanmean(gold_lin):.2f}",flush=True)

elif PHASE=="pysam":
    from gen_physics import wind_cf
    import multiprocessing as mp
    Z=np.load(f"{OUT}/tempsr_wx_{YEAR}.npz",allow_pickle=True); cfg=pd.read_csv(f"{GC}/eia_wind_configs.csv",dtype={"plant_code_unique":str}).drop_duplicates("plant_code_unique").set_index("plant_code_unique")
    plants=Z["plants"].astype(str); stamps=Z["stamps"].astype(str); FULLy=pd.to_datetime([f"{s[:4]}-{s[4:6]}-{s[6:8]} {s[8:10]}:00" for s in stamps]); _G={}
    def one(pi):
        p=plants[pi]; c=cfg.loc[p]
        ws=eval(c.wind_turbine_powercurve_windspeeds) if isinstance(c.wind_turbine_powercurve_windspeeds,str) else c.wind_turbine_powercurve_windspeeds
        pw=eval(c.wind_turbine_powercurve_powerout) if isinstance(c.wind_turbine_powercurve_powerout,str) else c.wind_turbine_powercurve_powerout
        base=dict(wind_turbine_hub_ht=float(c.wind_turbine_hub_ht),wind_turbine_rotor_diameter=float(c.wind_turbine_rotor_diameter),system_capacity=float(max(pw)),wind_turbine_powercurve_windspeeds=list(ws),wind_turbine_powercurve_powerout=list(pw))
        t2c=np.full(len(FULLy),15.0); pmb=np.full(len(FULLy),1013.0)
        return pi,wind_cf(FULLy,_G["lin"][pi],t2c,pmb,float(c.lat),float(c.lon),{**base,"shear":0.0}),wind_cf(FULLy,_G["hyb"][pi],t2c,pmb,float(c.lat),float(c.lon),{**base,"shear":0.0})
    _G["lin"]=Z["hub_linear"]; _G["hyb"]=Z["hub_hybrid"]; npl=len(plants); CFl=np.full((npl,len(stamps)),np.nan,np.float32); CFh=np.full((npl,len(stamps)),np.nan,np.float32)
    with mp.Pool(12) as pool:
        for pi,l,h in pool.imap_unordered(one,range(npl),chunksize=8): CFl[pi]=l; CFh[pi]=h
    np.savez(f"{OUT}/tempsr_cf_{YEAR}.npz",plants=plants,ba=Z["ba"],cap=Z["cap"],stamps=stamps,cf_linear=CFl,cf_hybrid=CFh)
    print(f"[pysam] CF mean linear={np.nanmean(CFl):.3f} hybrid={np.nanmean(CFh):.3f}",flush=True)

elif PHASE=="validate":
    Z=np.load(f"{OUT}/tempsr_cf_{YEAR}.npz",allow_pickle=True); plants=Z["plants"].astype(str); stamps=Z["stamps"].astype(str); LIN=Z["cf_linear"]; HYB=Z["cf_hybrid"]
    G=np.load("/data/gen_targets/wind_cf_1980_2019.npz",allow_pickle=True); gp={p:i for i,p in enumerate(G["plants"].astype(str))}; gt={t:i for i,t in enumerate(G["times"].astype(str))}
    col=np.array([gt.get(s,-1) for s in stamps]); ok=col>=0; god=np.full((len(plants),len(stamps)),np.nan,np.float32)
    ax=0 if G["cf"].shape[0]==len(G["times"]) else 1
    for i,p in enumerate(plants):
        if p in gp: god[i,ok]=(G["cf"][col[ok],gp[p]] if ax==0 else G["cf"][gp[p],col[ok]])
    def perplant_metric(A,fn):
        r=[fn(A[i],god[i]) for i in range(len(plants))]; return np.nanmedian(np.array(r))
    def r_(a,b):
        m=np.isfinite(a)&np.isfinite(b); return np.corrcoef(a[m],b[m])[0,1] if (m.sum()>100 and a[m].std()>1e-6 and b[m].std()>1e-6) else np.nan
    def rampr(a,b):
        m=np.isfinite(a)&np.isfinite(b); da=np.diff(a[m]); db=np.diff(b[m]); return db.std()/da.std() if da.std()>1e-9 else np.nan  # true/pred? -> pred/true below
    print(f"\n=== TEMPORAL-SR validation vs GODEEEP hourly ({YEAR}) ===")
    print(f"{'method':10s} {'perplant_r':>11s} {'ramp_ratio(pred/true)':>22s} {'CFmean':>7s}")
    for name,A in [("linear",LIN),("hybrid",HYB)]:
        rr=[]
        for i in range(len(plants)):
            m=np.isfinite(A[i])&np.isfinite(god[i])
            if m.sum()>200:
                da=np.diff(A[i][m]); dg=np.diff(god[i][m])
                if dg.std()>1e-9: rr.append(da.std()/dg.std())
        print(f"{name:10s} {perplant_metric(A,r_):>11.4f} {np.nanmedian(rr):>22.4f} {np.nanmean(A):>7.3f}")
    print("god CFmean %.3f (baseline linear ramp was ~0.75-0.85; target hybrid ramp -> ~1.0, r stays high)"%np.nanmean(god))
