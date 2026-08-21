"""Build 18-subregion hourly LOAD, GEN, and NET-LOAD for 1997-2019.
 - county load (TELL) -> subregion by majority-vote of each county's TGW cells into the subregion_mask
 - gen from subregion_gen_1997_2019.npz (solar+wind, fixed present fleet)
 - align load(real calendar, has Feb29) and gen(365-day) on timestamp intersection
 net = load - (solar+wind).  All MW."""
import numpy as np, pandas as pd
from scipy import stats
G="/data/datasets/grid"; GT="/data/gen_targets"; LO="/data/tell_pred/future/hist_full"
lat=np.load(f"{G}/coordinate.npz")["lat"]; lon=np.load(f"{G}/coordinate.npz")["lon"]
sm=np.load(f"{G}/subregion_mask.npz",allow_pickle=True); mask=sm["subregion_mask"]
id2=dict((int(r[0]),str(r[1])) for r in sm["id_to_subregion"]); NS=18; names=[id2[i] for i in range(1,NS+1)]
def cell_sub(la,lo):  # nearest regular-grid cell -> subregion id (0 if outside)
    ila=np.clip(np.searchsorted(lat,la),0,len(lat)-1); ilo=np.clip(np.searchsorted(lon,lo),0,len(lon)-1)
    return mask[ila,ilo]
# ---- county -> subregion (majority vote over the county's TGW cells) ----
g=np.load("/data/tgw_hist/tgw_grid.npz"); XLAT=g["XLAT"].ravel(); XLONG=g["XLONG"].ravel()
cm=np.load("/data/loads_measured/county_mask_tgw.npz",allow_pickle=True)
cfips=np.array([str(x).zfill(5) for x in cm["fips"]]); pf=cm["pair_fips"]; pc=cm["pair_cell"]
csub_cell=cell_sub(XLAT[pc],XLONG[pc])  # subregion per (county,cell) membership
county_sub=np.zeros(len(cfips),int)
for i in range(len(cfips)):
    v=csub_cell[pf==i]; v=v[v>0]
    county_sub[i]=int(stats.mode(v,keepdims=False).mode) if len(v) else 0
# nearest-nonzero fallback for counties fully outside mask
nz=np.argwhere(mask>0); nzll=np.column_stack([lat[nz[:,0]],lon[nz[:,1]]])
for i in np.where(county_sub==0)[0]:
    cc=pc[pf==i]
    if len(cc)==0: continue
    la,lo=XLAT[cc].mean(),XLONG[cc].mean()
    j=np.argmin((nzll[:,0]-la)**2+(nzll[:,1]-lo)**2); county_sub[i]=int(mask[nz[j,0],nz[j,1]])
print("counties assigned:", {names[s-1]:int((county_sub==s).sum()) for s in range(1,NS+1)})
print("unassigned counties:", int((county_sub==0).sum()))
# ---- load: county_load_hourly (nC,NH) -> subregion ----
meta=np.load(f"{LO}/meta.npz",allow_pickle=True); lfips=np.array([str(x).zfill(5) for x in meta["fips"]])
county=np.load(f"{LO}/county_load_hourly.npy",mmap_mode="r")  # (nC,NH) MW
yrs=meta["years"]; lidx=pd.date_range(f"{yrs[0]}-01-01",f"{yrs[-1]}-12-31 23:00",freq="h")
assert county.shape[1]==len(lidx), (county.shape,len(lidx))
# map load fips order -> county_sub (align by fips)
f2s={f:county_sub[i] for i,f in enumerate(cfips)}
lsub_id=np.array([f2s.get(f,0) for f in lfips])
loadS=np.zeros((NS,len(lidx)),np.float32)
for s in range(1,NS+1):
    rows=np.where(lsub_id==s)[0]
    if len(rows): loadS[s-1]=np.asarray(county[rows]).sum(0)
# ---- gen ----
gz=np.load(f"{GT}/subregion_gen_1997_2019.npz",allow_pickle=True)
gtimes=pd.to_datetime([str(x) for x in gz["times"]],format="%Y-%m-%d %H:%M:%S")
solar=gz["solar_gen"].T; wind=gz["wind_gen"].T  # (18,Tg)
# ---- align on timestamp intersection (drop load Feb29) ----
lkey=lidx.strftime("%Y%m%d%H"); gkey=gtimes.strftime("%Y%m%d%H")
gpos={k:j for j,k in enumerate(gkey)}
common=[(i,gpos[k]) for i,k in enumerate(lkey) if k in gpos]
li=np.array([c[0] for c in common]); gi=np.array([c[1] for c in common])
tidx=lidx[li]
L=loadS[:,li]; S=solar[:,gi]; Wd=wind[:,gi]; NET=L-S-Wd
np.savez(f"{LO}/subregion_netload_1997_2019.npz",load=L,solar=S,wind=Wd,net=NET,
         times=np.array([str(t) for t in tidx],dtype=object),subregions=np.array(names,dtype=object),
         county_sub=county_sub,county_fips=cfips)
print(f"\naligned T={len(tidx)}h  {tidx.min().date()}..{tidx.max().date()} (dropped {len(lidx)-len(tidx)} leap-h)")
print(f"{'subregion':20s} {'load_GW':>8s} {'solar':>7s} {'wind':>7s} {'VRE%':>6s} {'netpk_GW':>8s}")
for s in range(NS):
    l=L[s].mean()/1e3; so=S[s].mean()/1e3; wi=Wd[s].mean()/1e3; vre=100*(so+wi)/l if l>0 else 0
    print(f"{names[s]:20s} {l:8.1f} {so:7.2f} {wi:7.2f} {vre:6.1f} {NET[s].max()/1e3:8.1f}")
print(f"\nUS total: load {L.sum(0).mean()/1e3:.0f} GW avg, VRE {(S+Wd).sum(0).mean()/1e3:.0f} GW avg, "
      f"penetration {100*(S+Wd).sum()/L.sum():.1f}%")
print("saved -> subregion_netload_1997_2019.npz")
