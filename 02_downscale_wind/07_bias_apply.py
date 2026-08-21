"""Standalone TGW->CONUS404 bias apply (apply_hist extracted verbatim from 06_fit_bias.py; no /scratch dep).
Input aT_val (n,7,89,211) physical in LR_KEYS order [u10,v10,q2,psfc,t2,long,short]; stamps np.array of YYYYMMDDHH."""
import numpy as np, datetime as _dt, torch, torch.nn.functional as _Fn
NLAT, NLON = 89, 211; S0 = 1361.0
_co = np.load("/data/datasets/grid/coordinate.npz")
_latlr = _Fn.adaptive_avg_pool1d(torch.tensor(_co["lat"].astype("float64"))[None,None],89)[0,0].numpy()
_lonlr = _Fn.adaptive_avg_pool1d(torch.tensor(_co["lon"].astype("float64"))[None,None],211)[0,0].numpy()
LATg = np.deg2rad(_latlr)[:,None]*np.ones((1,NLON)); LONg = np.ones((NLAT,1))*_lonlr[None,:]
def cosz_stack(stamps):
    out = np.empty((len(stamps), NLAT, NLON), np.float32)
    for i, s in enumerate(stamps):
        s=str(s); yr,mo,dy,hh = int(s[:4]),int(s[4:6]),int(s[6:8]),int(s[8:10])
        doy = _dt.date(yr,mo,dy).timetuple().tm_yday; g = 2*np.pi/365.0*(doy-1+(hh-12)/24.0)
        eq = 229.18*(7.5e-5+1.868e-3*np.cos(g)-.032077*np.sin(g)-.014615*np.cos(2*g)-.040849*np.sin(2*g))
        dec = .006918-.399912*np.cos(g)+.070257*np.sin(g)-.006758*np.cos(2*g)+9.07e-4*np.sin(2*g)-.002697*np.cos(3*g)+.00148*np.sin(3*g)
        ha = np.deg2rad((hh*60+eq+4*LONg)/4.0-180.0)
        out[i] = np.clip(np.sin(LATg)*np.sin(dec)+np.cos(LATg)*np.cos(dec)*np.cos(ha),0,1)
    return out
def load_R(path="/data/tgw_hist/bias_fit.npz"):
    b=np.load(path,allow_pickle=True); return {k:b[k] for k in b.files}
def apply_hist(aT_val, R, stamps_val):
    stamps_val=np.asarray(stamps_val).astype(str)
    out = aT_val.copy(); mo = np.array([int(s[4:6]) for s in stamps_val])
    for m in range(1,13):
        idx = np.nonzero(mo==m)[0]
        if not len(idx): continue
        j=m-1
        out[idx,4]+=R["t2_off"][j]; out[idx,3]+=R["psfc_off"][j]; out[idx,5]+=R["long_off"][j]
        out[idx,2]=np.maximum(out[idx,2],0)*np.nan_to_num(R["q2_ratio"][j],nan=1.0)
        u=aT_val[idx,0]; v=aT_val[idx,1]; sp=np.sqrt(u**2+v**2); ang=np.arctan2(v,u)
        Tq=R["wind_Tq"][:,j]; Cq=R["wind_Cq"][:,j]; spc=np.empty_like(sp)
        for a in range(NLAT):
            for b in range(NLON):
                xp=Tq[:,a,b]
                if not np.isfinite(xp).all(): spc[:,a,b]=sp[:,a,b]; continue
                spc[:,a,b]=np.interp(sp[:,a,b],xp,Cq[:,a,b])
        ang2=ang+np.nan_to_num(R["dir_off"][j],nan=0.0)
        out[idx,0]=spc*np.cos(ang2); out[idx,1]=spc*np.sin(ang2)
        cz=cosz_stack(stamps_val[idx]); day=cz>0.15
        sw=aT_val[idx,6]; Kt=np.where(day,sw/(S0*np.where(day,cz,1)),0)
        Ktc=Kt*np.nan_to_num(R["sw_kt_ratio"][j],nan=1.0)
        out[idx,6]=np.where(day,np.clip(Ktc,0,1.2)*S0*cz,sw)
    return out
