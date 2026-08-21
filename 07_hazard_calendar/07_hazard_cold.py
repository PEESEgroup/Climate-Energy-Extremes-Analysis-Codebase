"""Hazard #4: cold-outbreak (CSDI) net-load composite, 1980-2019, the EXTREME metric that sharpens the
seasonal-mean AO- demand result (which washed out in Part I). CSDI = tmin below its own day-of-year
COLD_PCTL (10) on the shared +/-15 day window, COLD_PERSIST_DAYS consecutive days, then the
COLD_MONTHS gate of December to February (ETCCDI/Zhang 2011). Composite net-load/load anomaly on
cold-spell days; relate CSDI frequency to AO phase.

DEFINITIONS. Every constant comes from 07_hazard_calendar/hazard_defs.py. This file carries no
percentile literal, no private day-of-year percentile and no private persistence rule.

COLD PERSISTENCE, the one scale exception in the agreed specification. At SUBREGION scale, which is
what this file builds, the persistence is COLD_PERSIST_DAYS = 6 days. At COUNTY scale it is
COLD_PERSIST_DAYS_COUNTY = 3 days, in every county builder, because a county cold spell averages
1.77 days and the six-day rule leaves 0.96 county cold days a year, which is no usable signal.
Everything else about cold is identical at the two scales. Never sum or compare a six-day subregion
count with a three-day county count.

Substituting the shared helpers leaves the numbers unchanged: the arithmetic they replace was
identical, so this run reproduces the previous one.

NOTE: no gas freeze-off / thermal / turbine-icing (the ~75% of Uri's lost MW)."""
import os as _os, sys as _sys

import numpy as np, pandas as pd, matplotlib
matplotlib.use("Agg"); import matplotlib.pyplot as plt

_HD_DIR = _os.path.dirname(_os.path.abspath(__file__))
if _HD_DIR not in _sys.path:
    _sys.path.insert(0, _HD_DIR)
import hazard_defs as HD

LO="/data/tell_pred/future/hist_full40"; G="/data/datasets/grid"; EN="/data/enso"

# ---------- netload daily ----------
z=np.load(f"{LO}/subregion_netload_1980_2019.npz",allow_pickle=True)
names=[str(x) for x in z["subregions"]]; NS=len(names)
th=pd.to_datetime([str(x) for x in z["times"]])
dk=th.strftime("%Y-%m-%d").values; days,inv=np.unique(dk,return_inverse=True); cnt=np.bincount(inv)
def daily(x): return np.vstack([np.bincount(inv,x[s])/cnt for s in range(NS)])
dnet=daily(z["net"]); dload=daily(z["load"])
dd=pd.to_datetime(days); doy=HD.clip_doy(dd); mon=dd.month.values
yr=dd.year.values; ND=len(days)
# The threshold is fitted on the frozen climatology period of hazard_defs, 1980 to 2019, and never
# on days outside it. The record on file is exactly those years, so this mask is all True today and
# the threshold is unchanged by the freeze; it is passed anyway so that a longer record cannot
# silently refit the threshold on the days it then scores.
CLIM=(yr>=HD.CLIM_Y0)&(yr<=HD.CLIM_Y1)
assert CLIM.any(), "no day of the record falls inside the frozen climatology period"
def clim(x):
    cl=np.zeros((NS,367))
    for d in range(1,366):
        sel=HD.doy_window(doy,d); cl[:,d]=x[:,sel].mean(1)
    return cl
clN=clim(dnet); aN=dnet-clN[:,doy]; aL=dload-clim(dload)[:,doy]

# ---------- weather -> tmin arrays ----------
sm=np.load(f"{G}/subregion_mask.npz",allow_pickle=True); mask=sm["subregion_mask"]
id2=dict((int(r[0]),str(r[1])) for r in sm["id_to_subregion"]); n2i={v:k for k,v in id2.items()}
w=pd.read_csv(f"{EN}/subregion_weather_daily.csv")
tmin=np.full((NS,ND),np.nan)
for s,nm in enumerate(names):
    sub=w[w["sub"]==n2i[nm]][["date","tmin"]].set_index("date")["tmin"]
    tmin[s]=sub.reindex(days).values

# ---------- CSDI, from hazard_defs ----------
# HD.spell is the one sanctioned composition of the persistence rule and the season gate, in that
# order: HD.SPELL_ORDER is "persist_then_season" and it is part of the hashed definition. Gating
# first would drop every cold spell that straddles 30 November and 1 December, or 28 February and
# 1 March.
# COLD_PERSIST_DAYS is the SUBREGION length, 6 days. The county builders use
# COLD_PERSIST_DAYS_COUNTY, 3 days; see the scale exception in the docstring.
cold=tmin<HD.doy_pctl(tmin,HD.COLD_PCTL,doy,clim=CLIM)[:,doy]
csdi=HD.spell(cold,HD.COLD_PERSIST_DAYS,HD.COLD_MONTHS,mon)
djf=HD.season(HD.COLD_MONTHS,mon)   # kept for the winter-year masks below; csdi is already gated
_rate=float(csdi.mean()); _ok,_e,_t,_b=HD.day_rate_ok("subregion","cold",_rate)
print("cold flag: %.4f of subregion-days at %d-day persistence (%s %.4f +/- %.4f, %s), definition %s"
      % (_rate,HD.COLD_PERSIST_DAYS,"in band" if _ok else "OUT OF BAND",_e,_t,_b.split(":")[0],
         HD.definition_hash("cold")))

# ---------- composite net-load/load anomaly on CSDI days (DJF) ----------
rows=[]
for s in range(NS):
    m=csdi[s]&djf
    if m.sum()<15:
        rows.append(dict(sub=names[s],csdi_d_yr=round((csdi[s]&djf).sum()/40,1),net_pct=np.nan,load_MW=np.nan)); continue
    rows.append(dict(sub=names[s],csdi_d_yr=round(m.sum()/40,1),
        net_pct=round(100*aN[s,m].mean()/np.nanmean(clN[s,doy[m]]),1),
        load_MW=round(aL[s,m].mean(),0)))
df=pd.DataFrame(rows); df.to_csv(f"{LO}/cold_csdi_summary.csv",index=False)
print(df.to_string(index=False))

# ---------- CSDI frequency vs AO phase + SHARPENING vs seasonal-mean AO ----------
idx=pd.read_csv(f"{EN}/mode_tags_monthly_1980.csv")
def djf_ao(wtr):
    v=idx[((idx.year==wtr-1)&(idx.month==HD.COLD_MONTHS[0]))|((idx.year==wtr)&(idx.month.isin([1,2])))]["AO"]
    return v.mean() if v.notna().sum()==3 else np.nan
wy=np.where(dd.month==12,yr+1,yr); winters=np.arange(1981,2020)
ao=np.array([djf_ao(w) for w in winters])
aoNeg=winters[ao<=-0.5]; aoPos=winters[ao>=0.5]
# fleet-mean CSDI days per winter by AO phase
# 90 is the number of December-to-February days in a winter. The netload calendar carries no 29
# February, so December-February is exactly 31 + 31 + 28 days and the figure is not an approximation.
DJF_DAYS=90
def csdi_rate(ws): m=djf&np.isin(wy,ws); return csdi[:,m].mean()*DJF_DAYS
print(f"\nCSDI fleet day-rate per winter:  AO- {csdi_rate(aoNeg):.1f}   AO+ {csdi_rate(aoPos):.1f}   (per subregion-winter)")
# sharpening: demand anomaly on CSDI days vs on all AO- DJF days (seasonal-mean signal)
print("\nDemand anomaly (MW): CSDI-day composite vs AO- DJF seasonal mean (sharpening factor):")
for nm in ["MISO_North","SPP_North","ERCOT","MISO_Central","PJM_West","NYISO"]:
    s=names.index(nm)
    c=csdi[s]&djf; aom=djf&np.isin(wy,aoNeg)
    lc=aL[s,c].mean(); la=aL[s,aom].mean()
    print(f"  {nm:12s} CSDI {lc:+7.0f}   AO-mean {la:+7.0f}   x{(lc/la if la else np.nan):.1f}")

# ---------- figure ----------
lat=np.load(f"{G}/coordinate.npz")["lat"]; lon=np.load(f"{G}/coordinate.npz")["lon"]
fig,ax=plt.subplots(1,3,figsize=(16,5))
img=np.full(mask.shape,np.nan); pv=dict(zip(df["sub"],df["net_pct"]))
for nm,i in n2i.items():
    if nm in pv and not np.isnan(pv[nm]): img[mask==i]=pv[nm]
im=ax[0].imshow(img,origin="lower",cmap="Blues",extent=[lon.min(),lon.max(),lat.min(),lat.max()],aspect="auto")
ax[0].set_title("net-load anomaly on CSDI cold-spell days (% clim, DJF)"); ax[0].set_xticks([]); ax[0].set_yticks([]); plt.colorbar(im,ax=ax[0],shrink=.8)
img2=np.full(mask.shape,np.nan); lv=dict(zip(df["sub"],df["load_MW"]))
for nm,i in n2i.items():
    if nm in lv and not np.isnan(lv[nm]): img2[mask==i]=lv[nm]/1000
im2=ax[1].imshow(img2,origin="lower",cmap="Purples",extent=[lon.min(),lon.max(),lat.min(),lat.max()],aspect="auto")
ax[1].set_title("demand anomaly on CSDI days (GW)"); ax[1].set_xticks([]); ax[1].set_yticks([]); plt.colorbar(im2,ax=ax[1],shrink=.8)
# bar: sharpening
subs=["MISO_North","SPP_North","ERCOT","MISO_Central","PJM_West","NYISO"]; x=np.arange(len(subs))
lc=[aL[names.index(nm)][csdi[names.index(nm)]&djf].mean()/1000 for nm in subs]
la=[aL[names.index(nm)][djf&np.isin(wy,aoNeg)].mean()/1000 for nm in subs]
ax[2].bar(x-0.2,lc,0.4,label="CSDI-day (extreme)",color="navy")
ax[2].bar(x+0.2,la,0.4,label="AO- DJF mean",color="lightsteelblue")
ax[2].set_xticks(x); ax[2].set_xticklabels(subs,rotation=60,ha="right",fontsize=7); ax[2].axhline(0,c="k",lw=.6)
ax[2].set_ylabel("demand anomaly (GW)"); ax[2].legend(fontsize=8); ax[2].set_title("CSDI extreme SHARPENS the AO- demand signal")
fig.suptitle("Hazard #4 — Cold-outbreak (CSDI) net-load composite, 1980-2019 (fixed present fleet)",fontsize=12)
fig.tight_layout(rect=[0,0,1,.96]); fig.savefig(f"{LO}/cold_csdi_composite.png",dpi=120)
print("\nsaved cold_csdi_summary.csv + cold_csdi_composite.png")
