"""Hazard #3: atmospheric-river landfall net-load composite, 1980-2019 — SAME methodology as ENSO
(tag days -> composite net-load response -> partial regression), tag from NCEP-derived IVT.
AR landfall = coastal (lon 232.5-240E) IVT>=250 kg/m/s with onshore (IVTu>0) transport at a subregion's
coastal latitude. Composite solar/wind/load/net-load anomaly (NDJFM); net-load ~ IVT | AO, ENSO.
NOTE: flood/wind outages + reservoir/hydro refill benefit NOT modeled.

SUPERSEDED FOR THE PUBLISHED FLAG. The absolute 250 kg/m/s coastal rule below is the ORIGINAL
NCEP construction. The study's atmospheric-river flag is the subregion day-of-year percentile
rule in hazard_defs (AR_PCTL with AR_COVERAGE_FRACTION), built by 08_adequacy_analysis/06_ar_variants.py
and adopted by 07_ar_adopt_oc.py. Nothing published reads the flag this script builds; it is kept
because its composite and partial regression are the descriptive figures for hazard #3. The
absolute threshold is named here rather than typed inline so it cannot be mistaken for a
hazard_defs constant."""
import numpy as np, pandas as pd, matplotlib
matplotlib.use("Agg"); import matplotlib.pyplot as plt
import os as _os, sys as _sys
_HD = _os.path.join(_os.path.dirname(_os.path.abspath(__file__)), ".")
if _HD not in _sys.path: _sys.path.insert(0, _HD)
import hazard_defs as HD
AR_ABS_IVT = 250.0   # kg/m/s, the superseded absolute rule; NOT a hazard_defs constant
LO="/data/tell_pred/future/hist_full40"; G="/data/datasets/grid"; EN="/data/enso"; NCEP="/data/ncep"

# ---------- IVT ----------
iv=np.load(f"{NCEP}/ivt_daily_1980_2019.npz",allow_pickle=True)
IVT=iv["ivt"]; IVTU=iv["ivtu"]; ilat=iv["lat"]; ilon=iv["lon"]; idates=pd.to_datetime([str(x) for x in iv["dates"]])
coast=(ilon>=232.5)&(ilon<=240.0)              # ~ -127.5 .. -120 E, US West Coast band
idi={d.strftime("%Y-%m-%d"):k for k,d in enumerate(idates)}

# ---------- netload daily anomalies ----------
z=np.load(f"{LO}/subregion_netload_1980_2019.npz",allow_pickle=True)
names=[str(x) for x in z["subregions"]]; NS=len(names)
th=pd.to_datetime([str(x) for x in z["times"]])
dk=th.strftime("%Y-%m-%d").values; days,inv=np.unique(dk,return_inverse=True); cnt=np.bincount(inv)
def daily(x): return np.vstack([np.bincount(inv,x[s])/cnt for s in range(NS)])
dnet=daily(z["net"]); dload=daily(z["load"]); dsol=daily(z["solar"]); dwind=daily(z["wind"])
dd=pd.to_datetime(days); doy=np.minimum(dd.dayofyear.values,365); mon=dd.month.values; yr=dd.year.values; ND=len(days)
def clim(x):
    cl=np.zeros((NS,367))
    for d in range(1,366):
        sel=np.abs(((doy-d+182)%365)-182)<=15; cl[:,d]=x[:,sel].mean(1)
    return cl
clN=clim(dnet); aN=dnet-clN[:,doy]; aL=dload-clim(dload)[:,doy]; aS=dsol-clim(dsol)[:,doy]; aW=dwind-clim(dwind)[:,doy]

# ---------- WECC subregion coastal latitudes ----------
sm=np.load(f"{G}/subregion_mask.npz",allow_pickle=True); mask=sm["subregion_mask"]
id2=dict((int(r[0]),str(r[1])) for r in sm["id_to_subregion"]); n2i={v:k for k,v in id2.items()}
lat=np.load(f"{G}/coordinate.npz")["lat"]; lon=np.load(f"{G}/coordinate.npz")["lon"]; LON,LAT=np.meshgrid(lon,lat)
cent={nm:(LAT[mask==n2i[nm]].mean(),LON[mask==n2i[nm]].mean()) for nm in names if (mask==n2i[nm]).any()}
WECC=["CAISO","NorthernGrid_West","NorthernGrid_South","NorthernGrid_East","WestConnect_North","WestConnect_South"]

# ---------- per-subregion AR-landfall day (coastal IVT at its latitude band) ----------
def ar_days_for(latc):
    mlat=(ilat>=latc-3.0)&(ilat<=latc+3.0)
    if not mlat.any(): mlat=np.argmin(np.abs(ilat-latc))[None]
    sub=IVT[:,mlat][:,:,coast]; subu=IVTU[:,mlat][:,:,coast]
    onshore=subu>0
    strong=(sub>=AR_ABS_IVT)&onshore
    return strong.reshape(strong.shape[0],-1).any(1)   # per IVT-day boolean
NDJFM=np.isin(mon,[11,12,1,2,3])
rows=[]; ar_index=np.full(ND,np.nan)
# continuous AR index = max coastal IVT at 37.5N (central CA) per day, aligned to netload days
mCA=(ilat>=34)&(ilat<=41)
caIVT=IVT[:,mCA][:,:,coast].max((1,2))
for k,d in enumerate(days):
    if d in idi: ar_index[k]=caIVT[idi[d]]
for nm in WECC:
    latc=cent[nm][0]
    ard=ar_days_for(min(max(latc,33),49))     # clamp to coastal range
    isar=np.array([ard[idi[d]] if d in idi else False for d in days])
    m=isar&NDJFM; base=(~isar)&NDJFM
    if m.sum()<20: rows.append(dict(sub=nm,ar_d_yr=round(m.sum()/40,1))); continue
    rows.append(dict(sub=nm,ar_d_yr=round(m.sum()/40,1),
        net_pct=round(100*aN[names.index(nm)][m].mean()/np.nanmean(clN[names.index(nm),doy[m]]),1),
        solar_MW=round(aS[names.index(nm)][m].mean(),0),
        wind_MW=round(aW[names.index(nm)][m].mean(),0),
        load_MW=round(aL[names.index(nm)][m].mean(),0)))
df=pd.DataFrame(rows); df.to_csv(f"{LO}/ar_landfall_summary.csv",index=False)
print(df.to_string(index=False))

# ---------- partial regression net-load ~ IVT | AO, ENSO (daily, NDJFM) ----------
idx=pd.read_csv(f"{EN}/mode_tags_monthly_1980.csv")
ym=(dd.year.values*100+dd.month.values)
aoM=dict(zip(idx.year*100+idx.month,idx.AO)); oniM=dict(zip(idx.year*100+idx.month,idx.ONI))
AOd=np.array([aoM.get(v,np.nan) for v in ym]); ONId=np.array([oniM.get(v,np.nan) for v in ym])
def zsc(x): return (x-np.nanmean(x))/np.nanstd(x)
print("\nPartial std-beta of net-load anomaly on AR-IVT | AO,ENSO (NDJFM):")
for nm in ["CAISO","NorthernGrid_West","NorthernGrid_South"]:
    s=names.index(nm); sel=NDJFM&~np.isnan(ar_index)&~np.isnan(AOd)&~np.isnan(ONId)
    X=np.column_stack([zsc(ar_index[sel]),zsc(AOd[sel]),zsc(ONId[sel]),np.ones(sel.sum())])
    y=zsc(aN[s,sel]); beta,*_=np.linalg.lstsq(X,y,rcond=None)
    print(f"  {nm:18s} beta_IVT={beta[0]:+.3f}  (beta_AO={beta[1]:+.3f}, beta_ENSO={beta[2]:+.3f})")

# ---------- figure ----------
fig,ax=plt.subplots(1,3,figsize=(16,5))
# IVT climatology (mean) map over NA window
mIVT=IVT.mean(0)
im=ax[0].pcolormesh(ilon-360,ilat,mIVT,cmap="viridis",shading="auto")
ax[0].plot([-124,-124],[33,49],"r-",lw=2); ax[0].set_title("Mean IVT (kg/m/s) + West-Coast detection line")
ax[0].set_xlabel("lon"); ax[0].set_ylabel("lat"); plt.colorbar(im,ax=ax[0],shrink=.8)
# net_pct bar for WECC
d2=df.dropna(subset=["net_pct"]) if "net_pct" in df else df
x=np.arange(len(d2))
ax[1].bar(x,d2["net_pct"],color=["firebrick" if v>0 else "steelblue" for v in d2["net_pct"]])
ax[1].set_xticks(x); ax[1].set_xticklabels(d2["sub"],rotation=60,ha="right",fontsize=7); ax[1].axhline(0,c="k",lw=.6)
ax[1].set_ylabel("net-load anomaly on AR days (% clim)"); ax[1].set_title("AR-landfall net-load response (NDJFM)")
# solar vs wind for WECC
ax[2].bar(x-0.2,d2["solar_MW"]/1000,0.4,label="solar Δ",color="orange")
ax[2].bar(x+0.2,d2["wind_MW"]/1000,0.4,label="wind Δ",color="steelblue")
ax[2].set_xticks(x); ax[2].set_xticklabels(d2["sub"],rotation=60,ha="right",fontsize=7); ax[2].axhline(0,c="k",lw=.6)
ax[2].set_ylabel("gen anomaly (GW)"); ax[2].legend(fontsize=8); ax[2].set_title("AR: solar deficit (clouds) + wind ramp")
fig.suptitle("Hazard #3 — Atmospheric-river landfall composite (NCEP IVT + fixed-fleet net-load), 1980-2019",fontsize=12)
fig.tight_layout(rect=[0,0,1,.96]); fig.savefig(f"{LO}/ar_landfall_composite.png",dpi=120)
print("\nsaved ar_landfall_summary.csv + ar_landfall_composite.png")
