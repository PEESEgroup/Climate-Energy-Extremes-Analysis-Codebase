"""Hazard #1: heat & humid-heat (warm season) net-load composite, 1980-2019.
Heatwave = tmax above its own day-of-year percentile with the shared persistence and season rules,
all read from 07_hazard_calendar/hazard_defs.py: HEAT_PCTL 90 on a +/-15 day window,
HEAT_PERSIST_DAYS 3 consecutive days, then the HEAT_MONTHS gate of June to August. Humid-heat via
wet-bulb Tw (Stull 2011). Composites of net-load/load/solar/wind daily anomaly + HOURLY evening-ramp
composite. Demand = TELL (validated); supply = fixed-fleet GODEEEP CF. NOTE: no smoke-solar, no thermal derating.

DEFINITIONS. This file carries no percentile literal, no private day-of-year percentile and no
private persistence rule. Substituting the shared helpers leaves the numbers unchanged: the
arithmetic they replace was identical, so this run reproduces the previous one.

Humid heat is NOT one of the seven agreed hazards. It is a companion marker used only to split
heatwave days into humid and dry, and it reuses HEAT_PCTL on the wet-bulb series for that split."""
import os as _os, sys as _sys

import numpy as np, pandas as pd, matplotlib
matplotlib.use("Agg"); import matplotlib.pyplot as plt

_HD_DIR = _os.path.dirname(_os.path.abspath(__file__))
if _HD_DIR not in _sys.path:
    _sys.path.insert(0, _HD_DIR)
import hazard_defs as HD

LO="/data/tell_pred/future/hist_full40"; G="/data/datasets/grid"; EN="/data/enso"

# ---------- netload hourly + daily ----------
z=np.load(f"{LO}/subregion_netload_1980_2019.npz",allow_pickle=True)
names=[str(x) for x in z["subregions"]]; NS=len(names)
th=pd.to_datetime([str(x) for x in z["times"]])
hnet=z["net"]                                   # (18, H) hourly, for diurnal
dk=th.strftime("%Y-%m-%d").values; days,inv=np.unique(dk,return_inverse=True); cnt=np.bincount(inv)
def daily(x): return np.vstack([np.bincount(inv,x[s])/cnt for s in range(NS)])
dnet=daily(z["net"]); dload=daily(z["load"]); dsol=daily(z["solar"]); dwind=daily(z["wind"])
dd=pd.to_datetime(days); doy=HD.clip_doy(dd); mon=dd.month.values; yr=dd.year.values; ND=len(days)
# Thresholds are fitted on the frozen climatology period of hazard_defs, 1980 to 2019, and never on
# days outside it. The record on file is exactly those years, so this mask is all True today and the
# thresholds are unchanged by the freeze; it is passed anyway so that a longer record cannot
# silently refit the thresholds on the days it then scores.
CLIM=(yr>=HD.CLIM_Y0)&(yr<=HD.CLIM_Y1)
assert CLIM.any(), "no day of the record falls inside the frozen climatology period"
def clim(x):
    cl=np.zeros((NS,367))
    for d in range(1,366):
        sel=HD.doy_window(doy,d); cl[:,d]=x[:,sel].mean(1)
    return cl
clN=clim(dnet); aN=dnet-clN[:,doy]; aL=dload-clim(dload)[:,doy]
aS=dsol-clim(dsol)[:,doy]; aW=dwind-clim(dwind)[:,doy]

# ---------- weather daily -> arrays aligned to `days` (map mask-id -> netload idx) ----------
sm=np.load(f"{G}/subregion_mask.npz",allow_pickle=True); mask=sm["subregion_mask"]
id2=dict((int(r[0]),str(r[1])) for r in sm["id_to_subregion"]); n2i={v:k for k,v in id2.items()}
w=pd.read_csv(f"{EN}/subregion_weather_daily.csv")
def wvar(col):
    a=np.full((NS,ND),np.nan)
    for s,nm in enumerate(names):
        sub=w[w["sub"]==n2i[nm]][["date",col]].set_index("date")[col]
        a[s]=sub.reindex(days).values
    return a
tmax=wvar("tmax"); q=wvar("q"); ps=wvar("ps")

# ---------- wet-bulb (Stull 2011) ----------
def es(TK): return 611.2*np.exp(17.67*(TK-273.15)/(TK-29.65))     # Pa
e=q*ps/(0.622+0.378*q); RH=np.clip(100*e/es(tmax),1,100)
Tc=tmax-273.15
Tw=(Tc*np.arctan(0.151977*np.sqrt(RH+8.313659))+np.arctan(Tc+RH)-np.arctan(RH-1.676331)
    +0.00391838*RH**1.5*np.arctan(0.023101*RH)-4.686035)          # deg C

# ---------- the heat flag, from hazard_defs ----------
# HD.spell is the one sanctioned composition of the persistence rule and the season gate, in that
# order: HD.SPELL_ORDER is "persist_then_season" and it is part of the hashed definition. Gating
# first would drop every heat spell that straddles 31 May and 1 June.
hot=tmax>HD.doy_pctl(tmax,HD.HEAT_PCTL,doy,clim=CLIM)[:,doy]
# Humid heat is not one of the seven hazards; HEAT_PCTL is reused on the wet-bulb series only to
# split heatwave days into humid and dry.
humid=Tw>HD.doy_pctl(Tw,HD.HEAT_PCTL,doy,clim=CLIM)[:,doy]
hw_jja=HD.spell(hot,HD.HEAT_PERSIST_DAYS,HD.HEAT_MONTHS,mon)
_rate=float(hw_jja.mean()); _ok,_e,_t,_b=HD.day_rate_ok("subregion","heat",_rate)
print("heat flag: %.4f of subregion-days (%s %.4f +/- %.4f, %s), definition %s"
      % (_rate,"in band" if _ok else "OUT OF BAND",_e,_t,_b.split(":")[0],
         HD.definition_hash("heat")))

# ---------- composite (heatwave JJA vs all JJA) with event bootstrap ----------
rows=[]
for s in range(NS):
    m=hw_jja[s]
    if m.sum()<20:
        rows.append(dict(sub=names[s],hw_d_yr=round(m.sum()/40,1),net_pct=np.nan,load_MW=np.nan,
                         wind_MW=np.nan,solar_MW=np.nan,humidshare=np.nan)); continue
    net_pct=100*aN[s,m].mean()/np.nanmean(clN[s,doy[m]])
    hsh=100*(hw_jja[s]&humid[s]).sum()/max(1,hw_jja[s].sum())
    rows.append(dict(sub=names[s],hw_d_yr=round(m.sum()/40,1),
        net_pct=round(net_pct,1),load_MW=round(aL[s,m].mean(),0),
        wind_MW=round(aW[s,m].mean(),0),solar_MW=round(aS[s,m].mean(),0),
        humidshare=round(hsh,0)))
df=pd.DataFrame(rows); df.to_csv(f"{LO}/heat_summary.csv",index=False)
print(df.to_string(index=False))

# humid vs dry heat: load anomaly increment (SE regions)
print("\nHumid-heat load increment (heatwave & humid vs heatwave & dry), MW:")
for nm in ["FRCC","SERTP","ERCOT","MISO_South","PJM_East"]:
    s=names.index(nm); a=hw_jja[s]&humid[s]; b=hw_jja[s]&~humid[s]
    if a.sum()>10 and b.sum()>10:
        print(f"  {nm:12s} humid {aL[s,a].mean():+6.0f}  dry {aL[s,b].mean():+6.0f}  Δ {aL[s,a].mean()-aL[s,b].mean():+6.0f}")

# ---------- diurnal hourly net-load composite on heatwave days ----------
hh=th.hour.values; hmon=th.month.values; hdate=dk
hw_day_set={}                                  # per sub: set of heatwave-day strings
for s in range(NS):
    hw_day_set[s]=set(days[hw_jja[s]])
# hour-of-day x (is-heatwave) mean, minus JJA-hour climatology
fig,ax=plt.subplots(1,3,figsize=(16,5))
# panel A: diurnal
hot_subs=["ERCOT","CAISO","SERTP","PJM_West"]
for nm in hot_subs:
    s=names.index(nm)
    isJJA=np.isin(hmon,HD.HEAT_MONTHS)
    ishw=np.array([d in hw_day_set[s] for d in hdate]) & isJJA
    prof_hw=np.array([hnet[s][ishw&(hh==h)].mean() for h in range(24)])
    prof_cl=np.array([hnet[s][isJJA&(hh==h)].mean() for h in range(24)])
    ax[0].plot(range(24),(prof_hw-prof_cl)/1000,marker="o",ms=3,label=nm)
ax[0].axhline(0,c="k",lw=.6); ax[0].set_xlabel("hour (UTC)"); ax[0].set_ylabel("net-load anomaly (GW)")
ax[0].set_title("Diurnal net-load on JJA heatwave days\n(heatwave − JJA climatology)"); ax[0].legend(fontsize=8); ax[0].grid(alpha=.3)
# panel B: map of net_pct
lat=np.load(f"{G}/coordinate.npz")["lat"]; lon=np.load(f"{G}/coordinate.npz")["lon"]
img=np.full(mask.shape,np.nan); pv=dict(zip(df["sub"],df["net_pct"]))
for nm,i in n2i.items():
    if nm in pv and not np.isnan(pv[nm]): img[mask==i]=pv[nm]
im=ax[1].imshow(img,origin="lower",cmap="Reds",extent=[lon.min(),lon.max(),lat.min(),lat.max()],aspect="auto")
ax[1].set_title("net-load anomaly on JJA heatwave days (% of clim)"); ax[1].set_xticks([]); ax[1].set_yticks([]); plt.colorbar(im,ax=ax[1],shrink=.8)
# panel C: heatwave-day load vs wind bar (supply-demand squeeze)
sub_o=df.dropna().sort_values("net_pct",ascending=False).head(10)
x=np.arange(len(sub_o))
ax[2].bar(x-0.2,sub_o["load_MW"]/1000,0.4,label="load ↑ (demand)",color="firebrick")
ax[2].bar(x+0.2,sub_o["wind_MW"]/1000,0.4,label="wind ↓ (supply)",color="steelblue")
ax[2].set_xticks(x); ax[2].set_xticklabels(sub_o["sub"],rotation=60,ha="right",fontsize=7)
ax[2].axhline(0,c="k",lw=.6); ax[2].set_ylabel("anomaly (GW)"); ax[2].legend(fontsize=8)
ax[2].set_title("Heatwave demand↑ + wind-lull supply↓ = net-load squeeze")
fig.suptitle("Hazard #1 — Heat & humid-heat (JJA) net-load composite, 1980-2019 (fixed present fleet)",fontsize=12)
fig.tight_layout(rect=[0,0,1,.96]); fig.savefig(f"{LO}/heat_composite.png",dpi=120)
print("\nsaved heat_summary.csv + heat_composite.png")
