"""Build 18-subregion hourly generation (solar+wind) for 1997-2019 from GODEEEP tgw-gen CF.
Fixed present-fleet counterfactual: Gen[sub,t] = Σ_{plant∈sub} CF[t,plant] × capacity[plant].
Plant→subregion by nearest cell in the regular-lat/lon subregion_mask (nearest-nonzero fallback)."""
import numpy as np, pandas as pd
G="/data/datasets/grid"; GT="/data/gen_targets"
lat=np.load(f"{G}/coordinate.npz")["lat"]; lon=np.load(f"{G}/coordinate.npz")["lon"]
sm=np.load(f"{G}/subregion_mask.npz",allow_pickle=True)
mask=sm["subregion_mask"]; id2=dict((int(r[0]),str(r[1])) for r in sm["id_to_subregion"])
NS=18; names=[id2[i] for i in range(1,NS+1)]
print(f"grid lat {lat.min():.1f}..{lat.max():.1f} ({len(lat)}), lon {lon.min():.1f}..{lon.max():.1f} ({len(lon)}); mask{mask.shape}")
# nonzero cells for fallback
nz=np.argwhere(mask>0); nzll=np.column_stack([lat[nz[:,0]],lon[nz[:,1]]])
def plant_sub(plat,plon):
    ilat=int(np.argmin(np.abs(lat-plat))); ilon=int(np.argmin(np.abs(lon-plon)))
    s=int(mask[ilat,ilon])
    if s>0: return s
    d=(nzll[:,0]-plat)**2+(nzll[:,1]-plon)**2; j=int(np.argmin(d)); return int(mask[nz[j,0],nz[j,1]])

out={}
for tech in ["solar","wind"]:
    z=np.load(f"{GT}/{tech}_cf_1980_2019.npz",allow_pickle=True)
    cf=z["cf"]; times=pd.to_datetime([str(x) for x in z["times"]],format="%Y%m%d%H"); m=pd.read_csv(f"{GT}/{tech}_meta.csv")
    yr=times.year.values; keep=(yr>=1980)&(yr<=2019)
    cf=cf[keep]; tk=times[keep]; T=cf.shape[0]
    subs=np.array([plant_sub(float(r.lat),float(r.lon)) for r in m.itertuples()])
    cap=m["system_capacity"].values.astype(np.float64)  # kW
    W=np.zeros((len(m),NS))
    for p in range(len(m)):
        if 1<=subs[p]<=NS: W[p,subs[p]-1]=cap[p]
    gen=(cf.astype(np.float32)@W.astype(np.float32))/1000.0  # kW->MW ; (T,18)
    out[f"{tech}_gen"]=gen.astype(np.float32)
    capsub=W.sum(0)/1e3  # MW per subregion
    print(f"\n{tech}: T={T}h {tk.min().date()}..{tk.max().date()}  plants={len(m)} mapped={ (subs>=1).sum() }  fleet={cap.sum()/1e6:.1f} GW")
    print("  cap by subregion (MW):", {names[i]:int(capsub[i]) for i in range(NS) if capsub[i]>0})
    print(f"  fleet-mean CF={cf.mean():.3f}  gen-sum={gen.sum()/1e6:.0f} TWh over {T/8760:.0f}yr")
out["times"]=np.array([str(t) for t in tk],dtype=object); out["subregions"]=np.array(names,dtype=object)
np.savez(f"{GT}/subregion_gen_1980_2019.npz",**out)
print("\nsaved /data/gen_targets/subregion_gen_1980_2019.npz  solar_gen+wind_gen (T,18)")
