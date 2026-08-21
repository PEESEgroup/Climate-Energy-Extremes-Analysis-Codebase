"""Hazard #2: compound VRE-drought (Dunkelflaute), 1980-2019. Rinaldi-style daily index (<50% of day-of-year
climatology of combined wind+solar), return-period, demand-coincidence, and blocking/ENSO/AO association."""
import numpy as np, pandas as pd, matplotlib
matplotlib.use("Agg"); import matplotlib.pyplot as plt
import os as _os, sys as _sys
_HD = _os.path.join(_os.path.dirname(_os.path.abspath(__file__)), ".")
if _HD not in _sys.path: _sys.path.insert(0, _HD)
import hazard_defs as HD
LO="/data/tell_pred/future/hist_full40"; G="/data/datasets/grid"; EN="/data/enso"
z=np.load(f"{LO}/subregion_netload_1980_2019.npz",allow_pickle=True)
names=[str(x) for x in z["subregions"]]; NS=len(names)
t=pd.to_datetime([str(x) for x in z["times"]]); vre=z["solar"]+z["wind"]; net=z["net"]  # (18,T) MW
dk=t.strftime("%Y-%m-%d").values; days,inv=np.unique(dk,return_inverse=True); cnt=np.bincount(inv)
dvre=np.vstack([np.bincount(inv,vre[s])/cnt for s in range(NS)])
dnet=np.vstack([np.bincount(inv,net[s])/cnt for s in range(NS)])
dd=pd.to_datetime(days); doy=np.minimum(dd.dayofyear.values,365); yr=dd.year.values
years=np.arange(yr.min(),yr.max()+1); nyr=len(years)
def doyclim(x):
    cl=np.zeros(366)
    for d in range(1,366):
        sel=HD.doy_window(doy, d); cl[d]=np.nanmean(x[sel])
    return cl
drought=np.zeros((NS,len(days)),bool)
for s in range(NS):
    cl=doyclim(dvre[s]); drought[s]=dvre[s]<HD.VRE_FRACTION*cl[doy]
def runs(b):
    r=[];c=0
    for x in b:
        if x:c+=1
        elif c:r.append(c);c=0
    if c:r.append(c)
    return r if r else [0]
p90=np.array([np.percentile(dnet[s],90) for s in range(NS)])
bl=pd.read_csv(f"{EN}/blocking_daily.csv"); bl["d"]=pd.to_datetime(bl.date).dt.strftime("%Y-%m-%d")
blk=dict(zip(bl.d,bl.pac_block)); pac=np.array([blk.get(d,0) for d in days]).astype(bool)
rows=[]
for s in range(NS):
    dr=drought[s]; ddays=dr.sum()/nyr; mr=max(runs(dr))
    amax=np.array([max(runs(dr[yr==y])) for y in years])
    rp2,rp10=np.percentile(amax,50),np.percentile(amax,90)
    comp=(dr&(dnet[s]>p90[s])).sum()/nyr
    # blocking relative risk: P(drought|pac-block)/P(drought)
    rr=(dr[pac].mean()/dr.mean()) if dr.mean()>0 else np.nan
    rows.append(dict(sub=names[s],drought_d_yr=round(ddays,1),max_run=mr,rp2=round(rp2,1),
                     rp10=round(rp10,1),compound_d_yr=round(comp,1),blk_RR=round(rr,2)))
df=pd.DataFrame(rows); df.to_csv(f"{LO}/vre_drought_summary.csv",index=False)
print(df.to_string(index=False))
# ENSO/AO winter association (drought days in DJF by phase)
idx=pd.read_csv(f"{EN}/mode_tags_monthly_1980.csv")
def djf(col,w):
    v=idx[((idx.year==w-1)&(idx.month==12))|((idx.year==w)&(idx.month.isin([1,2])))][col]
    return v.mean() if v.notna().sum()==3 else np.nan
wy=np.where(dd.month==12,yr+1,yr); isDJF=np.isin(dd.month,[12,1,2])
winters=np.arange(1981,2020)
oni=np.array([djf("ONI",w) for w in winters]); ao=np.array([djf("AO",w) for w in winters])
enW=winters[oni>=0.5]; lnW=winters[oni<=-0.5]; aoNeg=winters[ao<=-0.5]
def djf_drought_rate(ws):  # mean DJF drought days per winter, fleet-mean over subregions
    m=isDJF&np.isin(wy,ws); return drought[:,m].mean()*90  # ~per 90-day winter
print(f"\nDJF fleet VRE-drought day-rate: ElNino {djf_drought_rate(enW):.1f} vs LaNina {djf_drought_rate(lnW):.1f} vs AO- {djf_drought_rate(aoNeg):.1f} (per subregion-winter)")
print(f"Pacific-blocking relative-risk of drought (fleet mean): {np.nanmean([r['blk_RR'] for r in rows]):.2f}x")
# figure: maps of drought days/yr, compound days/yr, blocking RR + SDF curves
sm=np.load(f"{G}/subregion_mask.npz",allow_pickle=True); mask=sm["subregion_mask"]
id2=dict((int(r[0]),str(r[1])) for r in sm["id_to_subregion"]); n2i={v:k for k,v in id2.items()}
lat=np.load(f"{G}/coordinate.npz")["lat"]; lon=np.load(f"{G}/coordinate.npz")["lon"]
def choro(ax,vals,ttl,cmap="viridis"):
    img=np.full(mask.shape,np.nan)
    for nm,i in n2i.items():
        if nm in vals: img[mask==i]=vals[nm]
    im=ax.imshow(img,origin="lower",cmap=cmap,extent=[lon.min(),lon.max(),lat.min(),lat.max()],aspect="auto")
    ax.set_title(ttl,fontsize=10); ax.set_xticks([]); ax.set_yticks([]); plt.colorbar(im,ax=ax,shrink=0.8)
fig,ax=plt.subplots(2,2,figsize=(14,9))
choro(ax[0,0],dict(zip(df["sub"],df["drought_d_yr"])),"VRE-drought days / yr (<50% clim)","YlOrRd")
choro(ax[0,1],dict(zip(df["sub"],df["compound_d_yr"])),"COMPOUND days/yr (drought & net-load>p90)","YlOrRd")
choro(ax[1,0],dict(zip(df["sub"],df["blk_RR"])),"Pacific-blocking relative risk of drought","RdBu_r")
# SDF: annual-max run distribution for 4 high-VRE subregions
for s in [names.index(n) for n in ["CAISO","ERCOT","SPP_South","MISO_North"] if n in names]:
    amax=np.sort([max(runs(drought[s][yr==y])) for y in years])[::-1]
    rp=(nyr+1)/np.arange(1,nyr+1)
    ax[1,1].plot(rp,amax,marker="o",ms=3,label=names[s])
ax[1,1].set_xscale("log"); ax[1,1].set_xlabel("return period (yr)"); ax[1,1].set_ylabel("annual-max drought duration (days)")
ax[1,1].set_title("Severity-Duration-Frequency"); ax[1,1].legend(fontsize=7); ax[1,1].grid(alpha=0.3)
fig.suptitle("Hazard #2 — Compound VRE-drought (Dunkelflaute), 1980-2019 (18 subregions, fixed present fleet)",fontsize=12)
fig.tight_layout(rect=[0,0,1,0.97]); fig.savefig(f"{LO}/vre_drought_map.png",dpi=120)
print("\nsaved vre_drought_summary.csv + vre_drought_map.png")
