"""Item 5: statistical hardening of Hazard #2 (compound VRE-drought), to match the other 5 hazards.
Same circular year-shift permutation null + BH-FDR + Livezey-Chen field significance as 16_harden_hazards.py.
VRE-drought tag is REBUILT here with hazard_defs.VRE_FRACTION and hazard_defs.doy_window,
so it is the same rule as 14_vre_drought.py by construction rather than by coincidence. Appends 'vre' rows to
hazard_significance.csv. NOTE (honest): net=load-solar-wind, so a low-VRE day is mechanically higher net-load;
this test quantifies the MAGNITUDE + spatial coherence + that it exceeds a season-preserving null, not sign."""
import numpy as np, pandas as pd
import os as _os, sys as _sys
_HD = _os.path.join(_os.path.dirname(_os.path.abspath(__file__)), ".")
if _HD not in _sys.path: _sys.path.insert(0, _HD)
import hazard_defs as HD
LO="/data/tell_pred/future/hist_full40"; G="/data/datasets/grid"
z=np.load(f"{LO}/subregion_netload_1980_2019.npz",allow_pickle=True)
names=[str(x) for x in z["subregions"]]; NS=len(names)
th=pd.to_datetime([str(x) for x in z["times"]])
dk=th.strftime("%Y-%m-%d").values; days,inv=np.unique(dk,return_inverse=True); cnt=np.bincount(inv)
def daily(x): return np.vstack([np.bincount(inv,x[s])/cnt for s in range(NS)])
dnet=daily(z["net"]); dvre=daily(z["solar"]+z["wind"])
dd=pd.to_datetime(days); doy=np.minimum(dd.dayofyear.values,365); ND=len(days)
def climf(x):
    cl=np.zeros((NS,367))
    for d in range(1,366):
        sel=HD.doy_window(doy, d); cl[:,d]=x[:,sel].mean(1)
    return cl
clN=climf(dnet); aN=dnet-clN[:,doy]
# VRE-drought tag (per subregion)
clV=climf(dvre); tag=dvre<HD.VRE_FRACTION*clV[:,doy]
# ---- identical hardening machinery ----
SH=[int(round(365.25*k))+j for k in range(1,40) for j in (-14,-7,0,7,14)]
def bh_fdr(p,q=0.05):
    v=~np.isnan(p); pv=p[v]; o=np.argsort(pv); m=len(pv); thr=(np.arange(1,m+1)/m)*q
    passed=pv[o]<=thr; sig=np.zeros(m,bool)
    if passed.any(): sig[o[:np.max(np.where(passed))+1]]=True
    out=np.zeros(len(p),bool); out[np.where(v)[0]]=sig; return out
def harden(A,tag):
    obs=np.array([A[s,tag[s]].mean() if tag[s].sum()>0 else np.nan for s in range(NS)])
    null=np.full((len(SH),NS),np.nan)
    for i,sh in enumerate(SH):
        As=np.roll(A,sh,axis=1)
        for s in range(NS):
            if tag[s].sum()>0: null[i,s]=As[s,tag[s]].mean()
    mu=np.nanmean(null,0)
    p=np.array([(1+np.sum(np.abs(null[:,s]-mu[s])>=np.abs(obs[s]-mu[s])))/(1+len(SH)) if not np.isnan(obs[s]) else np.nan for s in range(NS)])
    fdr=bh_fdr(p); q95=np.nanpercentile(np.abs(null-mu),95,axis=0)
    Robs=int(np.nansum(np.abs(obs-mu)>q95)); Rnull=np.array([np.nansum(np.abs(null[i]-mu)>q95) for i in range(len(SH))])
    return obs,p,fdr,Robs,float(np.percentile(Rnull,95))
obs,p,fdr,Robs,R95=harden(aN,tag)
fs=Robs>R95
rows=[]
for s in range(NS):
    rows.append(dict(hazard="vre",sub=names[s],net_MW=round(obs[s],0),
        pct=round(100*obs[s]/np.nanmean(clN[s,doy[tag[s]]]),1),p=round(p[s],3),
        fdr_sig=bool(fdr[s]),tag_days=int(tag[s].sum())))
new=pd.DataFrame(rows)
csv=f"{LO}/hazard_significance.csv"; old=pd.read_csv(csv); old=old[old.hazard!="vre"]
pd.concat([old,new],ignore_index=True).to_csv(csv,index=False)
print(f"VRE-drought hardening: #FDR-sig {int(fdr.sum())}/18   R_obs {Robs}  R95(null) {R95:.1f}   field-sig? {'YES' if fs else 'no'}")
d=new[new.fdr_sig].sort_values("pct",key=lambda c:c.abs(),ascending=False)
print("FDR-surviving: "+", ".join(f"{r['sub']} {r['pct']:+.0f}%" for _,r in d.head(12).iterrows()))
print(f"mean tag_days/sub {int(new.tag_days.mean())}  appended 'vre' rows to hazard_significance.csv  [HARDEN_VRE_DONE]")
