"""Prereq (#15) v2: aggregate TGW cells -> 18 subregions, DAILY weather stats, 1980-2019.
FIX vs v1: carry hour-count n per (file,sub,date) partial so boundary days (24h split across two
file-chunks) can be recombined EXACTLY at merge (tmax=max, tmin=min, means weighted by n).
real value = stored/scale. vars order [U10,V10,Q2,PSFC,T2,GLW,SWDOWN]. Sharded by crc32."""
import numpy as np, pandas as pd, glob, os, sys, zlib
G="/data/datasets/grid"
lat=np.load(f"{G}/coordinate.npz")["lat"]; lon=np.load(f"{G}/coordinate.npz")["lon"]
mask=np.load(f"{G}/subregion_mask.npz",allow_pickle=True)["subregion_mask"]
g=np.load("/data/tgw_hist/tgw_grid.npz"); XLAT=g["XLAT"].ravel(); XLONG=g["XLONG"].ravel()
ila=np.clip(np.searchsorted(lat,XLAT),0,len(lat)-1); ilo=np.clip(np.searchsorted(lon,XLONG),0,len(lon)-1)
csub=mask[ila,ilo]; NS=18
subcells={s:np.where(csub==s)[0] for s in range(1,NS+1)}
si,sn=map(int,(sys.argv[1] if len(sys.argv)>1 else "0/1").split("/"))
files=sorted(glob.glob("/data/tgw_hist/tgw_historical_*hourly*.npz"))
files=[f for f in files if zlib.crc32(os.path.basename(f).encode())%sn==si]
out=[]
for k,f in enumerate(files):
    z=np.load(f,allow_pickle=True); d=z["data"].astype("f4"); sc=np.asarray(z["scale"],"f4")
    d=d/sc[None,:,None,None]                          # real units
    nh=d.shape[0]; dd=d.reshape(nh,7,-1)
    times=pd.to_datetime([str(t) for t in z["times"]],format="%Y%m%d%H"); dates=times.strftime("%Y-%m-%d")
    for s in range(1,NS+1):
        cc=subcells[s]
        if len(cc)==0: continue
        sub=dd[:,:,cc].mean(2)                        # (nh,7): U10,V10,Q2,PSFC,T2,GLW,SWDOWN
        df=pd.DataFrame({"date":dates,"t2":sub[:,4],"q2":sub[:,2],"ps":sub[:,3],
                         "sw":sub[:,6],"wspd":np.sqrt(sub[:,0]**2+sub[:,1]**2)})
        agg=df.groupby("date").agg(tmax=("t2","max"),tmin=("t2","min"),tmean=("t2","mean"),
             q=("q2","mean"),ps=("ps","mean"),sw=("sw","mean"),wspd=("wspd","mean"),
             n=("t2","size")).reset_index()          # <-- n = hours in this file for this date
        agg["sub"]=s; out.append(agg)
    if k%100==0: print(f"shard{si} {k}/{len(files)}",flush=True)
res=pd.concat(out); res.to_csv(f"/data/enso/subweather2_shard{si}.csv",index=False)
print(f"shard {si} DONE: {len(files)} files -> {len(res)} rows",flush=True)
