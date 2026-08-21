"""Extract per-plant T2(K) and PSFC(hPa) from native TGW-3D 2019 at the SAME 1460 stamps as the deploy
inference, nearest native cell. -> s3d_pt_tgw2019.npz for the air-density-aware PySAM re-run."""
import numpy as np, glob, pandas as pd, re
OUT="/data/gen_targets/srgan3d_val"
# reference stamps + plant order (match the wx file exactly)
W=np.load(f"{OUT}/s3d_wx_advep10.npz", allow_pickle=True)
plants=W["plants"].astype(str); S=W["stamps"].astype(str); la=W["lat"].astype(float); lo=W["lon"].astype(float)
npl=len(plants); n=len(S); pos={s:i for i,s in enumerate(S)}
g=np.load("/data/tgw_3d/tgw3d_grid.npz"); XLAT=g["XLAT"].astype(float).ravel(); XLONG=g["XLONG"].astype(float).ravel()
idx=np.array([np.argmin((XLAT-la[k])**2+(XLONG-lo[k])**2) for k in range(npl)])   # plant -> native flat cell
T2=np.full((npl,n),np.nan,np.float32); PS=np.full((npl,n),np.nan,np.float32); Q2=np.full((npl,n),np.nan,np.float32)
got=0
for fp in sorted(glob.glob("/data/tgw_3d/tgw_wrf_historical_three_hourly_2019-*.npz")):
    z=np.load(fp, allow_pickle=True); tms=[str(t) for t in z["times"]]
    need=[(ti,s) for ti,s in enumerate(tms) if s in pos]
    if not need: z.close(); continue
    surf=z["surf"].astype(np.float32); z.close()   # [T,10,H,W] : U10,V10,T2,PSFC,Q2,...
    for ti,s in need:
        j=pos[s]; f=surf[ti].reshape(surf.shape[1],-1)
        T2[:,j]=f[2][idx]; PS[:,j]=f[3][idx]; Q2[:,j]=f[4][idx]; got+=1
np.savez(f"{OUT}/s3d_pt_tgw2019.npz", plants=plants, stamps=S, t2=T2, psfc=PS, q2=Q2)
ok=np.isfinite(T2)&np.isfinite(PS)
print("extracted %d/%d stamps ; T2 mean %.1fK PSFC mean %.1fhPa (finite %.3f) -> s3d_pt_tgw2019.npz"%(
    got, n, np.nanmean(T2[ok]), np.nanmean(PS[ok]), ok.mean()))
