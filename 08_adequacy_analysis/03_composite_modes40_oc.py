"""Multi-mode DJF composite: ENSO/PNA/NAO/AO on 18-subregion net-load & VRE gen, 1998-2019.
Each mode: DJF-mean index per winter -> HIGH(z>+.5)/LOW(z<-.5)/MID; merge consecutive same-phase -> episodes;
composite POSITIVE-phase minus NEGATIVE-phase; permutation p + bootstrap 90% CI + BH-FDR + field-sig."""
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
LO="/data/tell_pred/future/hist_full40"; G="/data/datasets/grid"; ENSO="/data/enso"
z=_PATHS.netload()
names=[str(x) for x in z["subregions"]]; NS=len(names)
t=pd.to_datetime([str(x) for x in z["times"]]); net=z["net"]; vre=z["solar"]+z["wind"]
yr=t.year.values; mo=t.month.values; wy=np.where(mo==12,yr+1,yr)
idx=pd.read_csv(f"{ENSO}/mode_tags_monthly_1980.csv")
MODES={"ENSO":"ONI","PNA":"PNA","NAO":"NAO","AO":"AO"}
winters=list(range(1981,2020))
def djf(col,w):
    v=idx[((idx.year==w-1)&(idx.month==12))|((idx.year==w)&(idx.month.isin([1,2])))][col]
    return v.mean() if v.notna().sum()==3 else np.nan
def winter_mask(w): return (wy==w)&np.isin(mo,[12,1,2])
def metrics_ep(ws,s):
    pk=[];mv=[]
    for w in ws:
        m=winter_mask(w)
        if m.sum()<2000: continue
        pk.append((net[s,m]/1e3).max()); mv.append((vre[s,m]/1e3).mean())
    return np.mean(pk),np.mean(mv)
rng=np.random.default_rng(7)
def compose(a,b,B=20000):
    base=b.mean(); d=a.mean()-b.mean(); pct=100*d/base if base else np.nan
    pool=np.concatenate([a,b]); nA=len(a); c=0
    for _ in range(B):
        pr=rng.permutation(pool)
        if abs(pr[:nA].mean()-pr[nA:].mean())>=abs(d): c+=1
    bs=[100*(rng.choice(a,len(a)).mean()-rng.choice(b,len(b)).mean())/base for _ in range(B//4)]
    return pct,(c+1)/(B+1),np.percentile(bs,5),np.percentile(bs,95)
rows=[]
for mode,col in MODES.items():
    di=np.array([djf(col,w) for w in winters]); zz=(di-np.nanmean(di))/np.nanstd(di)
    ph=np.where(zz>0.5,"POS",np.where(zz<-0.5,"NEG","MID"))
    eps=[]; cur=None
    for k,w in enumerate(winters):
        p=ph[k]
        if cur and cur["p"]==p and w==cur["ws"][-1]+1: cur["ws"].append(w)
        else:
            if cur: eps.append(cur)
            cur={"p":p,"ws":[w]}
    eps.append(cur)
    pe=np.array([e["p"] for e in eps]); iP=np.where(pe=="POS")[0]; iN=np.where(pe=="NEG")[0]
    PK=np.zeros((len(eps),NS)); MV=np.zeros((len(eps),NS))
    for ei,e in enumerate(eps):
        for s in range(NS): PK[ei,s],MV[ei,s]=metrics_ep(e["ws"],s)
    print(f"\n### {mode}: {len(eps)} episodes POS {sum(pe=='POS')} / NEG {sum(pe=='NEG')} / MID {sum(pe=='MID')}")
    for metric,M in [("peak_net_GW",PK),("mean_vre_GW",MV)]:
        ps=[]
        for s in range(NS):
            pct,p,lo,hi=compose(M[iP,s],M[iN,s]); ps.append(p)
            rows.append(dict(mode=mode,metric=metric,subregion=names[s],pct=pct,p=p,lo=lo,hi=hi))
        ps=np.array(ps); nraw=int((ps<0.05).sum())
        order=np.argsort(ps); passed=ps[order]<=0.05*(np.arange(NS)+1)/NS
        nfdr=(np.where(passed)[0].max()+1) if passed.any() else 0
        top=sorted(zip(names,[r['pct'] for r in rows if r['mode']==mode and r['metric']==metric],ps),key=lambda x:-abs(x[1]))[:3]
        print(f"  {metric:12s} POS-NEG: raw p<.05 {nraw}/18 | FDR {nfdr} | field-sig {'YES' if nraw>=4 else 'no'} | "
              f"top: "+", ".join(f"{n}{v:+.0f}%{'*' if p<.05 else ''}" for n,v,p in top))
df=pd.DataFrame(rows); df.to_csv(f"{LO}/composite_allmodes_djf_ourchain.csv",index=False)
# figure: 4 modes x 2 metrics choropleth (POS - NEG)
sm=np.load(f"{G}/subregion_mask.npz",allow_pickle=True); mask=sm["subregion_mask"]
id2=dict((int(r[0]),str(r[1])) for r in sm["id_to_subregion"]); n2i={v:k for k,v in id2.items()}
lat=np.load(f"{G}/coordinate.npz")["lat"]; lon=np.load(f"{G}/coordinate.npz")["lon"]
cent={nm:(int(np.where(mask==i)[0].mean()),int(np.where(mask==i)[1].mean())) for nm,i in n2i.items() if (mask==i).any()}
fig,axs=plt.subplots(4,2,figsize=(13,18))
for r,mode in enumerate(MODES):
    for c,(metric,lab) in enumerate([("mean_vre_GW","ΔVRE gen"),("peak_net_GW","Δpeak net-load")]):
        ax=axs[r,c]; d=df[(df["mode"]==mode)&(df.metric==metric)].set_index("subregion")
        img=np.full(mask.shape,np.nan)
        for nm,i in n2i.items():
            if nm in d.index: img[mask==i]=d.loc[nm,"pct"]
        vmax=max(2,np.nanpercentile(np.abs(img),98))
        im=ax.imshow(img,origin="lower",cmap="RdBu_r",vmin=-vmax,vmax=vmax,extent=[lon.min(),lon.max(),lat.min(),lat.max()],aspect="auto")
        for nm,(cy,cx) in cent.items():
            if nm in d.index and d.loc[nm,"p"]<0.05: ax.plot(lon[cx],lat[cy],"k*",ms=11)
        ax.set_title(f"{mode} (+phase − −phase): {lab}",fontsize=10); ax.set_xticks([]); ax.set_yticks([])
        plt.colorbar(im,ax=ax,shrink=0.75,label="%")
fig.suptitle("Historical circulation-mode composite, DJF 1998-2019 (★ p<0.05)  |  ENSO·PNA·NAO·AO",fontsize=13)
fig.tight_layout(rect=[0,0,1,0.98]); fig.savefig(f"{LO}/composite_allmodes_map.png",dpi=115)
print("\nsaved composite_allmodes_djf_ourchain.csv + composite_allmodes_map.png")
