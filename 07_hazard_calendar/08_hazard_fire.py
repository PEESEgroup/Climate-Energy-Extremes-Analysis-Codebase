"""Hazard #5: fire-WEATHER (HDW) climatology + power-system composite, 1980-2019.
HDW = VPD(hPa) x 10m wind (Srock 2018). VPD from tmax & RH-at-tmax. Descriptive climatology, the
1980-2019 trend, and a demand/wind/net-load composite on FIRE-WEATHER DAYS as hazard_defs defines
them: HDW above the subregion's own day-of-year 99th percentile on the shared +/-15 day window
(Abatzoglou et al. 2019), no persistence rule and no season gate.

DEFINITIONS. The percentile, the window, the persistence length and the season gate all come from
07_hazard_calendar/hazard_defs.py. This file holds no percentile literal of its own for the flag.

BEHAVIOR CHANGE, stated plainly: the composite used to run on a flat whole-record top decile of HDW,
`h >= np.nanpercentile(h, 90)`, about 1,461 days per subregion. It now runs on the shared fire flag,
about 1% of days. The composite columns of fire_hdw_summary.csv therefore no longer reproduce the
published run, and they are renamed so that no consumer can read the new quantity under the old
name. This has NOT been checked against the published figures; the published Figure 1 fire panel is
built from hazard_significance_ourchain.csv, not from this file.

HONEST SCOPE: fire's dominant impact is outage/PSPS + smoke->solar (NOT modeled); this is the
fire-weather ENVELOPE + its demand/wind signature.
CAVEAT: daily-MEAN wind (not gust) understates HDW peaks -> lower bound."""
import os as _os, sys as _sys

import numpy as np, pandas as pd, matplotlib
matplotlib.use("Agg"); import matplotlib.pyplot as plt

_HD_DIR = _os.path.dirname(_os.path.abspath(__file__))
if _HD_DIR not in _sys.path:
    _sys.path.insert(0, _HD_DIR)
import hazard_defs as HD

LO="/data/tell_pred/future/hist_full40"; G="/data/datasets/grid"; EN="/data/enso"

z=np.load(f"{LO}/subregion_netload_1980_2019.npz",allow_pickle=True)
names=[str(x) for x in z["subregions"]]; NS=len(names)
th=pd.to_datetime([str(x) for x in z["times"]])
dk=th.strftime("%Y-%m-%d").values; days,inv=np.unique(dk,return_inverse=True); cnt=np.bincount(inv)
def daily(x): return np.vstack([np.bincount(inv,x[s])/cnt for s in range(NS)])
dnet=daily(z["net"]); dload=daily(z["load"]); dwind=daily(z["wind"])
dd=pd.to_datetime(days); doy=HD.clip_doy(dd); mon=dd.month.values; yr=dd.year.values; ND=len(days)
# The fire threshold is fitted on the frozen climatology period of hazard_defs, 1980 to 2019, and
# never on days outside it. The record on file is exactly those years, so this mask is all True
# today and the threshold is unchanged by the freeze; it is passed anyway so that a longer record
# cannot silently refit the threshold on the days it then scores.
CLIM=(yr>=HD.CLIM_Y0)&(yr<=HD.CLIM_Y1)
assert CLIM.any(), "no day of the record falls inside the frozen climatology period"
def clim(x):
    cl=np.zeros((NS,367))
    for d in range(1,366):
        sel=HD.doy_window(doy,d); cl[:,d]=x[:,sel].mean(1)
    return cl
clN=clim(dnet); aN=dnet-clN[:,doy]; aL=dload-clim(dload)[:,doy]; aW=dwind-clim(dwind)[:,doy]

sm=np.load(f"{G}/subregion_mask.npz",allow_pickle=True); mask=sm["subregion_mask"]
id2=dict((int(r[0]),str(r[1])) for r in sm["id_to_subregion"]); n2i={v:k for k,v in id2.items()}
w=pd.read_csv(f"{EN}/subregion_weather_daily.csv")
def wvar(col):
    a=np.full((NS,ND),np.nan)
    for s,nm in enumerate(names):
        sub=w[w["sub"]==n2i[nm]][["date",col]].set_index("date")[col]
        a[s]=sub.reindex(days).values
    return a
tmax=wvar("tmax"); q=wvar("q"); ps=wvar("ps"); wspd=wvar("wspd")

# ---------- HDW ----------
def es(TK): return 611.2*np.exp(17.67*(TK-273.15)/(TK-29.65))
e=q*ps/(0.622+0.378*q); RH=np.clip(100*e/es(tmax),1,100)
VPD_hPa=np.clip(es(tmax)*(1-RH/100),0,None)/100.0
HDW=VPD_hPa*wspd                                     # hPa * m/s

# ---------- the fire-weather flag, from hazard_defs ----------
# HD.spell is the one sanctioned composition of the persistence rule and the season gate.
# FIRE_PERSIST_DAYS is 1 and FIRE_MONTHS is None, so it is an identity here. It is called anyway so
# that a change to either shared constant moves this file with no edit.
FIRE=HD.spell(HDW>HD.doy_pctl(HDW,HD.FIRE_PCTL,doy,clim=CLIM)[:,doy],
              HD.FIRE_PERSIST_DAYS,HD.FIRE_MONTHS,mon)
_rate=float(FIRE.mean()); _ok,_e,_t,_b=HD.day_rate_ok("subregion","fire",_rate)
print("fire flag: %.4f of subregion-days, %.1f days per subregion-year (%s %.4f +/- %.4f, %s), "
      "definition %s" % (_rate,365*_rate,"in band" if _ok else "OUT OF BAND",_e,_t,
                         _b.split(":")[0],HD.definition_hash("fire")))

# climatology, seasonality, trend
rows=[]
for s in range(NS):
    # HDW_p95 is a DESCRIPTIVE whole-record climatology statistic for the map in panel a. It is not
    # a threshold, nothing is flagged with it, and it is not a second construction of the fire
    # hazard: the flag is FIRE above, built from hazard_defs.
    h=HDW[s]; p95=np.nanpercentile(h,95)
    # season of peak HDW
    mm=np.array([np.nanmean(h[mon==k]) for k in range(1,13)]); pk=int(np.argmax(mm)+1)
    # linear trend of annual-mean HDW
    ann=np.array([np.nanmean(h[yr==y]) for y in range(1980,2020)])
    sl=np.polyfit(np.arange(40),ann,1)[0]*10          # per decade
    # power-system composite on the shared fire-weather days
    hi=FIRE[s]
    rows.append(dict(sub=names[s],HDW_mean=round(np.nanmean(h),1),HDW_p95=round(p95,1),
        peak_month=pk,trend_per_decade=round(sl,3),fire_d_yr=round(hi.sum()/40,1),
        net_pct_fire=round(100*np.nanmean(aN[s,hi])/np.nanmean(clN[s,doy[hi]]),1),
        load_MW_fire=round(np.nanmean(aL[s,hi]),0),wind_MW_fire=round(np.nanmean(aW[s,hi]),0)))
df=pd.DataFrame(rows); df.to_csv(f"{LO}/fire_hdw_summary.csv",index=False)
print(df.to_string(index=False))
mo={1:"Jan",2:"Feb",3:"Mar",4:"Apr",5:"May",6:"Jun",7:"Jul",8:"Aug",9:"Sep",10:"Oct",11:"Nov",12:"Dec"}
print("\nWestern fire-weather regions (peak month):",
      {nm:mo[df[df["sub"]==nm]["peak_month"].values[0]] for nm in
       ["CAISO","NorthernGrid_West","WestConnect_North","WestConnect_South","NorthernGrid_South"]})

# ---------- figure ----------
lat=np.load(f"{G}/coordinate.npz")["lat"]; lon=np.load(f"{G}/coordinate.npz")["lon"]
fig,ax=plt.subplots(1,3,figsize=(16,5))
def choro(a,vals,ttl,cmap,vmin=None,vmax=None):
    img=np.full(mask.shape,np.nan)
    for nm,i in n2i.items():
        if nm in vals and not np.isnan(vals[nm]): img[mask==i]=vals[nm]
    im=a.imshow(img,origin="lower",cmap=cmap,vmin=vmin,vmax=vmax,extent=[lon.min(),lon.max(),lat.min(),lat.max()],aspect="auto")
    a.set_title(ttl); a.set_xticks([]); a.set_yticks([]); plt.colorbar(im,ax=a,shrink=.8)
choro(ax[0],dict(zip(df["sub"],df["HDW_p95"])),"HDW p95 climatology (hPa·m/s)","YlOrRd")
choro(ax[1],dict(zip(df["sub"],df["trend_per_decade"])),"HDW trend (per decade, 1980-2019)","RdBu_r",-0.6,0.6)
# seasonal cycle for western regions
for nm in ["CAISO","NorthernGrid_West","WestConnect_South","SERTP"]:
    s=names.index(nm); mm=np.array([np.nanmean(HDW[s][mon==k]) for k in range(1,13)])
    ax[2].plot(range(1,13),mm,marker="o",ms=3,label=nm)
ax[2].set_xlabel("month"); ax[2].set_ylabel("mean HDW (hPa·m/s)"); ax[2].legend(fontsize=8); ax[2].grid(alpha=.3)
ax[2].set_title("HDW seasonal cycle (West = late-summer/autumn ridge)")
fig.suptitle("Hazard #5 — Fire-weather (HDW) climatology + trend, 1980-2019  [fire-weather ENVELOPE; outage/PSPS/smoke NOT modeled]",fontsize=11)
fig.tight_layout(rect=[0,0,1,.95]); fig.savefig(f"{LO}/fire_hdw_map.png",dpi=120)
print("\nsaved fire_hdw_summary.csv + fire_hdw_map.png")
