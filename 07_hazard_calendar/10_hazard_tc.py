"""Hazard #6: the subregion tropical-cyclone flag and its net-load composite, 1980-2019, HURDAT2.

THE RULE, and it is now the agreed one. A subregion is tropical-cyclone flagged on a day when a
retained HURDAT2 track point of that day lies within `hazard_defs.TC_RADIUS_KM` of the subregion
centroid. A track point is retained when its year is in range and its maximum wind is not the
`hazard_defs.TC_MISSING_WIND` code. There is no wind threshold at subregion scale: the 34 kt test
belongs to the county build, where it is the wind-radii rule, not a filter on track points.

WHAT THIS REPLACED, stated plainly because it changes every number below. Until 2026-08-18 this
file built a different object: the FIRST track point of each storm carrying the HURDAT2 landfall
marker `L`, inside a fixed 24 to 48 N by -98 to -66 W box, at 34 kt or more, with the 500 km radius
measured from that single landfall point. That is a landfall construction, not the subregion rule
hazard_defs states, so the repository carried two parallel subregion TC hazards under one name. The
box is gone, the `L` marker is gone and the 34 kt test is gone. The composite anchor moved with
them: an event is now a storm's FIRST flagged day, which is its approach, not its landfall. The
event count, the affected-subregion lists, the composite curves and the per-subregion percentages
all move. Nothing printed or written by this file reproduces the published landfall composite, and
no attempt has been made to check how far it moves, because the track file and the net-load store
are not on this machine.

TRACKS. The source is hurdat2_latest.txt, the same file the county build (12_county_tc_swath.py)
reads. The older /data/enso/hurdat2_atlantic.txt stopped on 2017-10-29, so every day of 2018 and
2019 was structurally free of tropical cyclones, two of the forty years. Records whose maximum wind
is the -999 missing code are DROPPED, exactly as the county build drops them; they used to be
admitted here as if they were tropical-storm strength, which made the two builds disagree in the
early record.

CONSTANTS. Every number that defines the hazard is read from 07_hazard_calendar/hazard_defs.py.
This file holds no private copy of the radius, the missing-wind code or the day-of-year window."""
import os as _os, sys as _sys
for _p in (_os.path.dirname(_os.path.abspath(__file__)),
           _os.path.join(_os.path.dirname(_os.path.abspath(__file__)), "..", "07_hazard_calendar")):
    if _os.path.isdir(_p) and _p not in _sys.path:
        _sys.path.insert(0, _p)
import hazard_defs as HD
import numpy as np, pandas as pd, matplotlib
matplotlib.use("Agg"); import matplotlib.pyplot as plt
LO="/data/tell_pred/future/hist_full40"; G="/data/datasets/grid"
H2="/data/equity_cost/analysis/did/hurdat2_latest.txt"   # same track file as the county build
Y0,Y1=HD.CLIM_Y0,HD.CLIM_Y1        # 1980 to 2019, the frozen period this record already spans
rng=np.random.default_rng(0)

# ---------- net-load -> daily anomalies ----------
# The pre-ourchain product this used to read no longer exists. The flag itself depends only on the
# track file and the subregion centroids, so the net load here supplies the calendar, the subregion
# names and the composite anomalies that accompany the flag. Those come from the anchored product,
# the canonical historical net load, named once in paths.py.
import sys as _sys
import os as _os_rp
for _rp in ("04_demand_model", "07_hazard_calendar", "09_outage_attribution",
            "02_downscale_wind", "12_figures"):
    _ap = _os_rp.path.abspath(_os_rp.path.join(
        _os_rp.path.dirname(_os_rp.path.abspath(__file__)), "..", _rp))
    if _os_rp.path.isdir(_ap) and _ap not in _sys.path:
        _sys.path.insert(0, _ap)
import paths as _PATHS
z=_PATHS.netload()
names=[str(x) for x in z["subregions"]]; NS=len(names); n2ix={nm:i for i,nm in enumerate(names)}
t=pd.to_datetime([str(x) for x in z["times"]])
dk=t.strftime("%Y-%m-%d").values; days,inv=np.unique(dk,return_inverse=True); cnt=np.bincount(inv)
def daily(x): return np.vstack([np.bincount(inv,x[s])/cnt for s in range(NS)])
dnet=daily(z["net"]); dload=daily(z["load"]); dsol=daily(z["solar"]); dwind=daily(z["wind"])
dd=pd.to_datetime(days); doy=HD.clip_doy(dd)
di={d:i for i,d in enumerate(days)}; ND=len(days)
def anom(x):
    # the day-of-year window is hazard_defs.doy_window, the same one the percentiles use; the
    # climatology is a mean, and the record is exactly Y0..Y1, so no separate freeze is needed
    cl=np.zeros((NS,367))
    for d in range(1,366):
        cl[:,d]=x[:,HD.doy_window(doy,d)].mean(1)
    return x-cl[:,doy], cl
anet,clnet=anom(dnet); aload,_=anom(dload); asol,_=anom(dsol); awind,_=anom(dwind)

# ---------- subregion centroids ----------
sm=np.load(f"{G}/subregion_mask.npz",allow_pickle=True); mask=sm["subregion_mask"]
id2=dict((int(r[0]),str(r[1])) for r in sm["id_to_subregion"]); n2i={v:k for k,v in id2.items()}
lat=np.load(f"{G}/coordinate.npz")["lat"]; lon=np.load(f"{G}/coordinate.npz")["lon"]
LON,LAT=np.meshgrid(lon,lat)
cent={}
for nm,i in n2i.items():
    m=mask==i
    if m.sum(): cent[nm]=(LAT[m].mean(),LON[m].mean())
subs=[nm for nm in names if nm in cent]
CLA=np.array([cent[nm][0] for nm in subs]); CLO=np.array([cent[nm][1] for nm in subs])
SIX=np.array([n2ix[nm] for nm in subs])
def hav(la1,lo1,la2,lo2):
    r=6371.0;p=np.pi/180
    a=np.sin((la2-la1)*p/2)**2+np.cos(la1*p)*np.cos(la2*p)*np.sin((lo2-lo1)*p/2)**2
    return 2*r*np.arcsin(np.sqrt(a))

# ---------- parse HURDAT2 ----------
def parse(fn):
    storms=[]; cur=None
    for line in open(fn):
        p=[x.strip() for x in line.split(",")]
        if len(p)>=3 and p[0][:2] in("AL","EP","CP") and p[0][2:4].isdigit() and len(p[0])==8:
            cur={"id":p[0],"name":p[1],"pts":[]}; storms.append(cur)
        elif cur is not None and len(p)>=7 and p[0].isdigit() and len(p[0])==8:
            la=float(p[4][:-1])*(1 if p[4][-1]=="N" else -1)
            lo=float(p[5][:-1])*(1 if p[5][-1]=="E" else -1)
            wd=int(p[6]) if p[6] not in("","-999") else HD.TC_MISSING_WIND
            cur["pts"].append((p[0],p[1],p[2],p[3],la,lo,wd))
    return storms
storms=parse(H2)
print("basins parsed: %s" % sorted({s["id"][:2] for s in storms}))

# ---------- the flag: any retained track point within TC_RADIUS_KM of the centroid ----------
# HURDAT2 writes -999 where the maximum wind is missing, which is common before ~1990. Those points
# are dropped here and by the county build, so the two builds see the same track points. The count
# of dropped points that would otherwise have flagged a subregion is printed, because it is the
# only part of the missing-wind rule that changes an answer.
tcflag=np.zeros((NS,ND),bool)
first={}                 # storm id -> (day index, lat, lon, wind, name) of its first flagging point
affset={}                # storm id -> set of subregion indices it ever flags
n999=0; n999_near=0
for s in storms:
    for p in s["pts"]:
        y=int(p[0][:4])
        la,lo,wd=p[4],p[5],p[6]
        if wd==HD.TC_MISSING_WIND:
            n999+=1
            if Y0<=y<=Y1 and (hav(la,lo,CLA,CLO)<=HD.TC_RADIUS_KM).any(): n999_near+=1
            continue
        if y<Y0 or y>Y1: continue
        d=f"{p[0][:4]}-{p[0][4:6]}-{p[0][6:8]}"
        j=di.get(d)
        if j is None: continue
        near=hav(la,lo,CLA,CLO)<=HD.TC_RADIUS_KM
        if not near.any(): continue
        idxs=SIX[near]
        tcflag[idxs,j]=True
        affset.setdefault(s["id"],set()).update(int(k) for k in idxs)
        if s["id"] not in first or j<first[s["id"]][0]:
            first[s["id"]]=(j,la,lo,wd,s["name"])
print(f"HURDAT2 points with missing wind ({HD.TC_MISSING_WIND}), dropped: {n999}  "
      f"(of which within {HD.TC_RADIUS_KM:.0f} km of a subregion centroid in {Y0}-{Y1}: {n999_near})")

rate=float(tcflag.mean())
_ok,_e,_t,_b=HD.day_rate_ok("subregion","tc",rate)
print("subregion TC day rate %.4f (%d of %d subregion-days); recorded band %.4f +/- %.4f [%s] -> %s"
      % (rate,int(tcflag.sum()),NS*ND,_e,_t,_b.split(":")[0],"in band" if _ok else "OUT OF BAND"))

_fs,_fj=np.where(tcflag)
FL=pd.DataFrame({"subregion":[names[s] for s in _fs],"date":[days[j] for j in _fj]})
HD.write_flags(FL,f"{LO}/subregion_tc_days.parquet",script=__file__,n_units=NS,n_dates=ND,
               hazards=["tc"],hazard_col=None,
               extra={"radius_km":HD.TC_RADIUS_KM,"geometry":"track point to subregion centroid",
                      "years":[Y0,Y1]})
print(f"wrote {LO}/subregion_tc_days.parquet with a hazard_defs stamp")

# ---------- events for the composite: one per storm, anchored on its first flagged day ----------
events=[]
for sid,(j,la,lo,wd,nm) in sorted(first.items(),key=lambda kv:kv[1][0]):
    events.append(dict(id=sid,name=nm,date=days[j],lat=la,lon=lo,wind=wd,i0=j,
                       affected=[names[k] for k in sorted(affset[sid])]))
print(f"storm events {Y0}-{Y1}: {len(events)}  (~{len(events)/(Y1-Y0+1):.1f}/yr), "
      "anchored on the first day the storm brings a track point within "
      f"{HD.TC_RADIUS_KM:.0f} km of a subregion centroid")
yrs=pd.Series([int(e['date'][:4]) for e in events]); print("by-decade:",dict(yrs.groupby(yrs//10*10).count()))

# ---------- lagged composite over (event, affected-subregion) ----------
LAGS=np.arange(-3,8)
def gather(A):  # returns list per lag of anomaly values across event x affected-sub
    vals={L:[] for L in LAGS}
    for e in events:
        for nm in e["affected"]:
            s=n2ix[nm]
            for L in LAGS:
                j=e["i0"]+L
                if 0<=j<ND: vals[L].append(A[s,j])
    return {L:np.array(v) for L,v in vals.items()}
def curve(A):
    g=gather(A); m=np.array([g[L].mean() for L in LAGS])
    # bootstrap over EVENTS (resample whole events to respect within-event corr)
    ev_idx=np.arange(len(events)); B=2000; bs=np.zeros((B,len(LAGS)))
    for b in range(B):
        pick=rng.choice(ev_idx,len(ev_idx),replace=True)
        acc={L:[] for L in LAGS}
        for ei in pick:
            e=events[ei]
            for nm in e["affected"]:
                s=n2ix[nm]
                for L in LAGS:
                    j=e["i0"]+L
                    if 0<=j<ND: acc[L].append(A[s,j])
        bs[b]=[np.mean(acc[L]) if acc[L] else np.nan for L in LAGS]
    lo=np.nanpercentile(bs,2.5,0); hi=np.nanpercentile(bs,97.5,0)
    return m,lo,hi
cN,loN,hiN=curve(anet); cL,_,_=curve(aload); cS,_,_=curve(asol); cW,_,_=curve(awind)

# net-load %-of-clim in the arrival window (lag 0..+2) per affected subregion
sub_ct={}; sub_pk={}
for e in events:
    for nm in e["affected"]:
        s=n2ix[nm]; j=e["i0"]
        w=[anet[s,j+L]/clnet[s,doy[j+L]] for L in (0,1,2) if 0<=j+L<ND]
        sub_ct[nm]=sub_ct.get(nm,0)+1; sub_pk.setdefault(nm,[]).extend(w)
rows=[]
for nm in names:
    if nm in sub_ct:
        rows.append(dict(sub=nm,n_events=sub_ct[nm],netload_pct_arrival=round(100*np.mean(sub_pk[nm]),1)))
summ=pd.DataFrame(rows).sort_values("netload_pct_arrival",ascending=False)
summ.to_csv(f"{LO}/tc_proximity_summary.csv",index=False)
print(summ.to_string(index=False))
print(f"\nArrival-window (lag0..2) net-load anomaly: {cN[3:6].mean():+.0f} MW  "
      f"(solar {cS[3:6].mean():+.0f}, wind {cW[3:6].mean():+.0f}, load {cL[3:6].mean():+.0f} MW)")

# ---------- figure ----------
fig,ax=plt.subplots(1,2,figsize=(15,5.5))
ax[0].axhline(0,c="k",lw=.6); ax[0].axvline(0,c="grey",ls="--",lw=.8)
ax[0].fill_between(LAGS,loN,hiN,alpha=.2,color="firebrick")
ax[0].plot(LAGS,cN,"-o",c="firebrick",label="net-load")
ax[0].plot(LAGS,cL,"-s",c="black",label="load (demand)")
ax[0].plot(LAGS,cS,"-^",c="orange",label="solar gen")
ax[0].plot(LAGS,cW,"-v",c="steelblue",label="wind gen")
ax[0].set_xlabel("lag (days from first flagged day)"); ax[0].set_ylabel("anomaly (MW, vs day-of-yr clim)")
ax[0].set_title(f"TC weather-envelope composite (n={len(events)} storms)\n"
                "supply loss (solar+wind down) vs demand; NOT outage collapse"); ax[0].legend(fontsize=8); ax[0].grid(alpha=.3)
img=np.full(mask.shape,np.nan)
pv=dict(zip(summ["sub"],summ["netload_pct_arrival"]))
for nm,i in n2i.items():
    if nm in pv: img[mask==i]=pv[nm]
im=ax[1].imshow(img,origin="lower",cmap="RdBu_r",vmin=-15,vmax=15,
                extent=[lon.min(),lon.max(),lat.min(),lat.max()],aspect="auto")
for e in events: ax[1].plot(e["lon"],e["lat"],"k.",ms=2,alpha=.4)
ax[1].set_title("net-load anomaly at arrival (% of clim, lag0..+2)\nblack dots = first flagging track point")
ax[1].set_xticks([]); ax[1].set_yticks([]); plt.colorbar(im,ax=ax[1],shrink=.8)
fig.suptitle(f"Hazard #6 - TC composite (HURDAT2 + fixed-fleet net-load), {Y0}-{Y1}",fontsize=12)
fig.tight_layout(rect=[0,0,1,.96]); fig.savefig(f"{LO}/tc_proximity_composite.png",dpi=120)
print("\nsaved tc_proximity_summary.csv + tc_proximity_composite.png")
