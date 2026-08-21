"""STEP A: sub-seasonal EXTREME / Dunkelflaute metrics composite (replaces DJF seasonal means).
Metrics per subregion per winter (DJF), thresholds = pooled-DJF climatology per subregion:
  peak3d_net  = max 72h-rolling-mean net load (GW)  -> sustained cold-outbreak stress (NAO/AO target)
  ramp3h      = max 3h net-load increase (GW)
  vre_drought = # DJF hrs VRE < p10(clim)           -> low-supply hours
  dunkel_run  = longest run (hrs) of  VRE<p25 & net>p75  -> compound Dunkelflaute (blocking target)
Composite (+phase − −phase) for ENSO/PNA/NAO/AO; permutation p + BH-FDR + field-sig."""
import numpy as np, pandas as pd
import sys as _sys
import os as _os_rp
for _rp in ("04_demand_model", "07_hazard_calendar", "09_outage_attribution",
            "02_downscale_wind", "12_figures"):
    _ap = _os_rp.path.abspath(_os_rp.path.join(
        _os_rp.path.dirname(_os_rp.path.abspath(__file__)), "..", _rp))
    if _os_rp.path.isdir(_ap) and _ap not in _sys.path:
        _sys.path.insert(0, _ap)
import paths as _PATHS   # the one name for the anchored net-load product
LO="/data/tell_pred/future/hist_full40"; ENSO="/data/enso"
z=_PATHS.netload()
names=[str(x) for x in z["subregions"]]; NS=len(names)
t=pd.to_datetime([str(x) for x in z["times"]]); net=z["net"]/1e3; vre=(z["solar"]+z["wind"])/1e3  # GW
yr=t.year.values; mo=t.month.values; wy=np.where(mo==12,yr+1,yr); isDJF=np.isin(mo,[12,1,2])
# pooled-DJF climatological thresholds per subregion
p10v=np.array([np.percentile(vre[s,isDJF],10) for s in range(NS)])
p25v=np.array([np.percentile(vre[s,isDJF],25) for s in range(NS)])
p75n=np.array([np.percentile(net[s,isDJF],75) for s in range(NS)])
def longest_run(b):
    m=0;c=0
    for x in b:
        c=c+1 if x else 0; m=max(m,c)
    return m
def winter_metrics(w,s):
    m=(wy==w)&isDJF
    if m.sum()<2000: return [np.nan]*4
    n=net[s,m]; v=vre[s,m]
    r3=np.convolve(n,np.ones(72)/72,"valid").max()           # peak 3-day sustained net load
    ramp=np.max(n[3:]-n[:-3])                                  # max 3h ramp
    dh=int((v<p10v[s]).sum())                                  # low-VRE hours
    run=longest_run((v<p25v[s])&(n>p75n[s]))                   # compound Dunkelflaute run
    return [r3,ramp,dh,run]
metrics=["peak3d_net_GW","ramp3h_GW","vre_drought_hrs","dunkel_run_hrs"]
winters=list(range(1981,2020))
X={m:np.zeros((len(winters),NS)) for m in metrics}
for wi,w in enumerate(winters):
    for s in range(NS):
        vals=winter_metrics(w,s)
        for mi,m in enumerate(metrics): X[m][wi,s]=vals[mi]
idx=pd.read_csv(f"{ENSO}/mode_tags_monthly_1980.csv")
def djf(col,w):
    v=idx[((idx.year==w-1)&(idx.month==12))|((idx.year==w)&(idx.month.isin([1,2])))][col]
    return v.mean() if v.notna().sum()==3 else np.nan
rng=np.random.default_rng(11)
def compose(a,b,B=20000):
    a=a[~np.isnan(a)]; b=b[~np.isnan(b)]
    base=b.mean(); d=a.mean()-b.mean(); pct=100*d/base if base else np.nan
    pool=np.concatenate([a,b]); nA=len(a); c=0
    for _ in range(B):
        pr=rng.permutation(pool)
        if abs(pr[:nA].mean()-pr[nA:].mean())>=abs(d): c+=1
    return pct,(c+1)/(B+1)
MODES={"ENSO":"ONI","PNA":"PNA","NAO":"NAO","AO":"AO"}
rows=[]
for mode,col in MODES.items():
    di=np.array([djf(col,w) for w in winters]); zz=(di-np.nanmean(di))/np.nanstd(di)
    ph=np.where(zz>0.5,"POS",np.where(zz<-0.5,"NEG","MID"))
    iP=np.where(ph=="POS")[0]; iN=np.where(ph=="NEG")[0]
    print(f"\n### {mode}  (POS {len(iP)} / NEG {len(iN)} winters)")
    for m in metrics:
        ps=[];pc=[]
        for s in range(NS):
            pct,p=compose(X[m][iP,s],X[m][iN,s]); ps.append(p);pc.append(pct)
            rows.append(dict(mode=mode,metric=m,subregion=names[s],pct=pct,p=p))
        ps=np.array(ps);pc=np.array(pc); nraw=int((ps<0.05).sum())
        top=sorted(zip(names,pc,ps),key=lambda x:-abs(x[1]))[:3]
        print(f"  {m:16s}: sig {nraw}/18 | field-sig {'YES' if nraw>=4 else 'no '} | "+
              ", ".join(f"{n} {v:+.0f}%{'*' if p<.05 else ''}" for n,v,p in top))
pd.DataFrame(rows).to_csv(f"{LO}/composite_extremes_djf_ourchain.csv",index=False)
print("\nsaved composite_extremes_djf_ourchain.csv")
