"""Historical ENSO composite of 18-subregion winter (DJF) net-load extremes, 1998-2019.
Per ENSO_ANALYSIS Part C/E: merge consecutive same-phase winters into ONE episode; composite each phase
(EN, LN) vs neutral; permutation p-value (small-K honest test) + bootstrap-by-episode CI; BH-FDR + field sig."""
import numpy as np, pandas as pd, json
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
t=pd.to_datetime([str(x) for x in z["times"]]); net=z["net"]; solar=z["solar"]; wind=z["wind"]; vre=solar+wind
yr=t.year.values; mo=t.month.values
# winter-year: Dec belongs to next year's DJF
wy=np.where(mo==12,yr+1,yr)
# ---- ONI DJF phase per winter-year ----
oni=pd.read_csv(f"{ENSO}/mode_tags_monthly_1980.csv")
def djf_oni(w):
    v=oni[((oni.year==w-1)&(oni.month==12))|((oni.year==w)&(oni.month.isin([1,2])))].ONI
    return v.mean() if len(v)==3 else np.nan
winters=[w for w in range(1981,2020) if np.isin([f"{w-1}12",f"{w}01",f"{w}02"],
         (oni.year*100+oni.month).astype(str).values).all()]
phase={}
for w in winters:
    o=djf_oni(w); phase[w]="EN" if o>=0.5 else ("LN" if o<=-0.5 else "NEU")
# merge consecutive same-phase winters -> episodes
eps=[]; cur=None
for w in winters:
    p=phase[w]
    if cur and cur["p"]==p and w==cur["ws"][-1]+1: cur["ws"].append(w)
    else:
        if cur: eps.append(cur)
        cur={"p":p,"ws":[w]}
eps.append(cur)
print("winters:",winters)
print("phase seq:", [phase[w] for w in winters])
print(f"episodes: {len(eps)} = EN {sum(e['p']=='EN' for e in eps)}, LN {sum(e['p']=='LN' for e in eps)}, NEU {sum(e['p']=='NEU' for e in eps)}")
for e in eps: print(f"   {e['p']}: {e['ws']}")
# ---- per-episode, per-subregion winter metrics (avg over the episode's winters) ----
def winter_mask(w): return (wy==w)&np.isin(mo,[12,1,2])
metrics=["peak_net_GW","mean_net_GW","mean_vre_GW","vre_drought_GW"]  # drought = min daily-mean VRE in DJF
def episode_metric(ws,s):
    pk=[];mn=[];mv=[];dr=[]
    for w in ws:
        m=winter_mask(w)
        if m.sum()<2000: continue
        n=net[s,m]/1e3; v=vre[s,m]/1e3
        pk.append(n.max()); mn.append(n.mean()); mv.append(v.mean())
        # daily mean VRE, take season min (worst renewable day)
        nd=len(v)//24*24; dr.append(v[:nd].reshape(-1,24).mean(1).min())
    return [np.mean(pk),np.mean(mn),np.mean(mv),np.mean(dr)]
X={m:np.zeros((len(eps),NS)) for m in metrics}
for ei,e in enumerate(eps):
    for s in range(NS):
        vals=episode_metric(e["ws"],s)
        for mi,m in enumerate(metrics): X[m][ei,s]=vals[mi]
pe=np.array([e["p"] for e in eps])
iN=np.where(pe=="NEU")[0]; iE=np.where(pe=="EN")[0]; iL=np.where(pe=="LN")[0]
rng=np.random.default_rng(42)
def compose(M,idxP,idxN,B=20000):
    """return pct diff (P vs neutral), permutation p (two-sided), bootstrap 90% CI."""
    a=M[idxP]; b=M[idxN]; base=b.mean(); d=a.mean()-b.mean(); pct=100*d/base if base else np.nan
    pool=np.concatenate([a,b]); nA=len(a); cnt=0
    for _ in range(B):
        pr=rng.permutation(pool); dd=pr[:nA].mean()-pr[nA:].mean()
        if abs(dd)>=abs(d): cnt+=1
    pperm=(cnt+1)/(B+1)
    bs=[]
    for _ in range(B//4):
        da=rng.choice(a,len(a)); db=rng.choice(b,len(b)); bs.append(100*(da.mean()-db.mean())/base)
    lo,hi=np.percentile(bs,[5,95])
    return pct,pperm,lo,hi
rows=[]
for m in metrics:
    for lab,idxP in [("EN",iE),("LN",iL)]:
        ps=[]
        for s in range(NS):
            pct,pp,lo,hi=compose(X[m][:,s],idxP,iN)
            rows.append(dict(metric=m,phase=lab,subregion=names[s],pct=pct,p=pp,lo=lo,hi=hi)); ps.append(pp)
        ps=np.array(ps)
        # BH-FDR
        order=np.argsort(ps); thr=0.05*(np.arange(NS)+1)/NS; passed=ps[order]<=thr
        nfdr=(np.where(passed)[0].max()+1) if passed.any() else 0
        nraw=int((ps<0.05).sum())
        print(f"[{m} {lab}] raw p<.05: {nraw}/{NS} subregions | BH-FDR sig: {nfdr} | field-sig(binom>3): {'YES' if nraw>=4 else 'no'}")
df=pd.DataFrame(rows); df.to_csv(f"{LO}/enso_composite_djf_ourchain.csv",index=False)
# headline: winter peak net load, EN and LN, sorted
print("\n=== DJF PEAK NET LOAD: El Nino vs neutral (top movers) ===")
pk=df[(df.metric=='peak_net_GW')].copy()
for lab in ["EN","LN"]:
    sub=pk[pk.phase==lab].reindex(pk[pk.phase==lab].pct.abs().sort_values(ascending=False).index)
    print(f" {lab}:")
    for _,r in sub.head(6).iterrows():
        star="*" if r.p<0.05 else " "
        print(f"   {r.subregion:20s} {r.pct:+6.1f}%  (90%CI {r.lo:+5.1f},{r.hi:+5.1f})  p={r.p:.3f}{star}")
print("\nsaved -> enso_composite_djf_ourchain.csv")
