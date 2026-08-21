"""STEP B2: DAILY composite of subregion net-load & VRE on BLOCKED vs UNBLOCKED DJF days, by sector.
Sector-resolved WHERE. Significance = moving-block bootstrap (L=10d) preserving autocorrelation."""
import numpy as np, pandas as pd, matplotlib
import sys as _sys
import os as _os_rp
for _rp in ("04_demand_model", "07_hazard_calendar", "09_outage_attribution",
            "02_downscale_wind", "12_figures"):
    _ap = _os_rp.path.abspath(_os_rp.path.join(
        _os_rp.path.dirname(_os_rp.path.abspath(__file__)), "..", _rp))
    if _os_rp.path.isdir(_ap) and _ap not in _sys.path:
        _sys.path.insert(0, _ap)
import paths as _PATHS   # the one name for the anchored net-load product
matplotlib.use("Agg"); import matplotlib.pyplot as plt
LO="/data/tell_pred/future/hist_full40"; G="/data/datasets/grid"
z=_PATHS.netload()
names=[str(x) for x in z["subregions"]]; NS=len(names)
t=pd.to_datetime([str(x) for x in z["times"]]); net=z["net"]/1e3; vre=(z["solar"]+z["wind"])/1e3
daykey=t.strftime("%Y-%m-%d").values; days,inv=np.unique(daykey,return_inverse=True)
cnt=np.bincount(inv)
dnet=np.vstack([np.bincount(inv,net[s])/cnt for s in range(NS)])   # (18,ndays) daily-mean net
dvre=np.vstack([np.bincount(inv,vre[s])/cnt for s in range(NS)])
dd=pd.to_datetime(days); djf=np.isin(dd.month,[12,1,2])
bl=pd.read_csv("/data/enso/blocking_daily.csv"); bl["date"]=bl.date.str[:10]
bmap={r.date:(r.pac_block,r.atl_block) for r in bl.itertuples()}
pac=np.array([bmap.get(d,(0,0))[0] for d in days]); atl=np.array([bmap.get(d,(0,0))[1] for d in days])
rng=np.random.default_rng(3)
def mbb_diff(x,flag,mask,L=10,B=4000):
    """moving-block bootstrap of (mean[flag]-mean[~flag])/mean[~flag]*100 over masked days."""
    xi=x[mask]; fi=flag[mask].astype(bool); n=len(xi)
    base=xi[~fi].mean(); d0=100*(xi[fi].mean()-xi[~fi].mean())/base
    nb=int(np.ceil(n/L)); starts=np.arange(n-L+1)
    bs=[]
    for _ in range(B):
        idx=np.concatenate([np.arange(s,s+L) for s in rng.choice(starts,nb)])[:n]
        xb=xi[idx]; fb=fi[idx]
        if fb.sum()<3 or (~fb).sum()<3: continue
        bs.append(100*(xb[fb].mean()-xb[~fb].mean())/xb[~fb].mean())
    bs=np.array(bs); p=2*min((bs<=0).mean(),(bs>=0).mean())
    return d0,np.percentile(bs,5),np.percentile(bs,95),max(p,1/B)
rows=[]
for sec,flag in [("Pacific",pac),("Atlantic",atl)]:
    for metric,arr in [("net",dnet),("vre",dvre)]:
        for s in range(NS):
            d0,lo,hi,p=mbb_diff(arr[s],flag,djf)
            rows.append(dict(sector=sec,metric=metric,subregion=names[s],pct=d0,lo=lo,hi=hi,p=p))
df=pd.DataFrame(rows); df.to_csv(f"{LO}/blocking_composite_djf_ourchain.csv",index=False)
for sec in ["Pacific","Atlantic"]:
    for metric in ["net","vre"]:
        d=df[(df.sector==sec)&(df.metric==metric)]; nsig=(d.p<0.05).sum()
        top=d.reindex(d.pct.abs().sort_values(ascending=False).index).head(4)
        print(f"[{sec} block Δ{metric}] sig {nsig}/18 field-sig {'YES' if nsig>=4 else 'no'}: "+
              ", ".join(f"{r.subregion}{r.pct:+.0f}%{'*' if r.p<.05 else ''}" for r in top.itertuples()))
# figure: sector-resolved net-load response maps
sm=np.load(f"{G}/subregion_mask.npz",allow_pickle=True); mask=sm["subregion_mask"]
id2=dict((int(r[0]),str(r[1])) for r in sm["id_to_subregion"]); n2i={v:k for k,v in id2.items()}
lat=np.load(f"{G}/coordinate.npz")["lat"]; lon=np.load(f"{G}/coordinate.npz")["lon"]
cent={nm:(int(np.where(mask==i)[0].mean()),int(np.where(mask==i)[1].mean())) for nm,i in n2i.items() if (mask==i).any()}
fig,axs=plt.subplots(2,2,figsize=(15,9))
for r,sec in enumerate(["Pacific","Atlantic"]):
    for c,(metric,lab) in enumerate([("vre","Δ daily VRE gen"),("net","Δ daily net-load")]):
        ax=axs[r,c]; d=df[(df.sector==sec)&(df.metric==metric)].set_index("subregion")
        img=np.full(mask.shape,np.nan)
        for nm,i in n2i.items():
            if nm in d.index: img[mask==i]=d.loc[nm,"pct"]
        vmax=max(2,np.nanpercentile(np.abs(img),98))
        im=ax.imshow(img,origin="lower",cmap="RdBu_r",vmin=-vmax,vmax=vmax,extent=[lon.min(),lon.max(),lat.min(),lat.max()],aspect="auto")
        for nm,(cy,cx) in cent.items():
            if nm in d.index and d.loc[nm,"p"]<0.05: ax.plot(lon[cx],lat[cy],"k*",ms=11)
        ax.set_title(f"{sec} blocking: {lab} (blocked − unblocked DJF days)",fontsize=10); ax.set_xticks([]); ax.set_yticks([])
        plt.colorbar(im,ax=ax,shrink=0.75,label="%")
fig.suptitle("STEP B — Sector-resolved blocking composite (daily, NCEP Z500, ★ p<0.05 moving-block bootstrap)",fontsize=12)
fig.tight_layout(rect=[0,0,1,0.97]); fig.savefig(f"{LO}/blocking_composite_map.png",dpi=115)
print("\nsaved blocking_composite_djf_ourchain.csv + blocking_composite_map.png")
