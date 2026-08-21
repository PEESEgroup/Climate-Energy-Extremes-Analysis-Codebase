"""Item 1: statistical hardening of the 5 event-composite hazards (heat, cold, fire, AR, TC) to match the
ENSO work's rigor. Per-subregion significance via a CIRCULAR YEAR-SHIFT permutation null (preserves
day-of-year/season + spatial covariance), Benjamini-Hochberg FDR applied within each hazard across the
subregions that carry at least one tagged day, and Livezey-Chen field significance (Monte-Carlo count
of local rejections vs the coherently-shifted null).

DEFINITIONS. Every percentile, window, persistence length and season gate is read from
07_hazard_calendar/hazard_defs.py. This file carries no percentile literal and no private
day-of-year percentile or persistence rule of its own, so one edit to the shared module moves this
script and every other builder together.

TAG PROVENANCE, since an earlier docstring claimed the wrong thing: heat, cold and fire are recomputed
here from the subregion daily weather through the shared helpers; AR and TC are NOT rebuilt, they are
READ from the deployed products (the adopted array of ar_flag_variants.npz, whose key is assembled
from HD.AR_PCTL and HD.AR_COVERAGE_FRACTION and is 'ivt_p95_cov25' at the current constants, and the
panel's tc_local column) so this script cannot carry a second, parallel definition of either.

Output: hazard_significance.csv + console table. That CSV is a statistics table, not a flag table, and
07_hazard_calendar/17_harden_vre.py appends rows to it, so it is not written through
hazard_defs.write_flags; the definition hash of every rebuilt hazard is printed to the console
instead."""
import os as _os, sys as _sys

import numpy as np, pandas as pd

_HD_DIR = _os.path.dirname(_os.path.abspath(__file__))
if _HD_DIR not in _sys.path:
    _sys.path.insert(0, _HD_DIR)
import hazard_defs as HD

LO="/data/tell_pred/future/hist_full40"; G="/data/datasets/grid"; EN="/data/enso"; R1="/data/enso/r1_causal"

# ---------- netload daily anomalies ----------
z=np.load(f"{LO}/subregion_netload_1980_2019.npz",allow_pickle=True)
names=[str(x) for x in z["subregions"]]; NS=len(names)
th=pd.to_datetime([str(x) for x in z["times"]])
dk=th.strftime("%Y-%m-%d").values; days,inv=np.unique(dk,return_inverse=True); cnt=np.bincount(inv)
def daily(x): return np.vstack([np.bincount(inv,x[s])/cnt for s in range(NS)])
dnet=daily(z["net"])
dd=pd.to_datetime(days); doy=HD.clip_doy(dd); mon=dd.month.values; yr=dd.year.values; ND=len(days)
# Every percentile below is fitted on the frozen climatology period of hazard_defs, 1980 to 2019,
# and never on days outside it. The historical record on file is exactly those years, so the mask
# is all True today and the thresholds are unchanged by the freeze; the mask is passed anyway so
# that a longer record cannot silently refit the thresholds on the days it then scores.
CLIM=(yr>=HD.CLIM_Y0)&(yr<=HD.CLIM_Y1)
assert CLIM.any(), "no day of the record falls inside the frozen climatology period"
if not CLIM.all():
    print(f"NOTE: {int((~CLIM).sum())} of {ND} days fall outside {HD.CLIM_Y0}-{HD.CLIM_Y1} and are "
          f"used for scoring but not for fitting the thresholds")
def climf(x):
    cl=np.zeros((NS,367))
    for d in range(1,366):
        sel=HD.doy_window(doy,d); cl[:,d]=x[:,sel].mean(1)
    return cl
clN=climf(dnet); aN=dnet-clN[:,doy]

sm=np.load(f"{G}/subregion_mask.npz",allow_pickle=True)
id2=dict((int(r[0]),str(r[1])) for r in sm["id_to_subregion"]); n2i={v:k for k,v in id2.items()}

# ---------- weather -> arrays ----------
w=pd.read_csv(f"{EN}/subregion_weather_daily.csv")
def wvar(col):
    a=np.full((NS,ND),np.nan)
    for s,nm in enumerate(names):
        sub=w[w["sub"]==n2i[nm]][["date",col]].set_index("date")[col]; a[s]=sub.reindex(days).values
    return a
tmax=wvar("tmax"); tmin=wvar("tmin"); q=wvar("q"); ps=wvar("ps"); wspd=wvar("wspd")

# ---------- build per-subregion tags (hazard_defs, nothing local) ----------
# HD.spell is the one sanctioned composition of the persistence rule and the season gate, in that
# order: HD.SPELL_ORDER is "persist_then_season" and it is part of the hashed definition. Calling
# persist() and season() here instead would put the order back into a comment, where it drifted
# before. Gating first would drop every heat spell straddling 31 May and every cold spell
# straddling 30 November.
TAGS={}
# heat: HEAT_VAR above its own day-of-year HEAT_PCTL, HEAT_PERSIST_DAYS consecutive days, HEAT_MONTHS
TAGS["heat"]=HD.spell(tmax>HD.doy_pctl(tmax,HD.HEAT_PCTL,doy,clim=CLIM)[:,doy],
                      HD.HEAT_PERSIST_DAYS,HD.HEAT_MONTHS,mon)
# cold: COLD_VAR below its own day-of-year COLD_PCTL, COLD_PERSIST_DAYS consecutive days, COLD_MONTHS.
# COLD_PERSIST_DAYS is the SUBREGION length, 6 days. The county builders use
# COLD_PERSIST_DAYS_COUNTY, 3 days, because a county cold spell averages 1.77 days and the six-day
# rule leaves 0.96 county cold days a year. This file is a subregion build, so 6 applies here.
TAGS["cold"]=HD.spell(tmin<HD.doy_pctl(tmin,HD.COLD_PCTL,doy,clim=CLIM)[:,doy],
                      HD.COLD_PERSIST_DAYS,HD.COLD_MONTHS,mon)
def es(TK): return 611.2*np.exp(17.67*(TK-273.15)/(TK-29.65))
e=q*ps/(0.622+0.378*q); RH=np.clip(100*e/es(tmax),1,100); VPD=np.clip(es(tmax)*(1-RH/100),0,None)/100; HDW=VPD*wspd
# fire: HDW = VPD x 10m wind (Srock 2018) above the subregion's OWN day-of-year FIRE_PCTL
# (Abatzoglou 2019). FIRE_PERSIST_DAYS is 1 and FIRE_MONTHS is None, so HD.spell is an identity
# here; it is called anyway so a change to either shared constant moves this file with no edit.
# This replaces a flat whole-record p90.
TAGS["fire"]=HD.spell(HDW>HD.doy_pctl(HDW,HD.FIRE_PCTL,doy,clim=CLIM)[:,doy],
                      HD.FIRE_PERSIST_DAYS,HD.FIRE_MONTHS,mon)
for hz in ("heat","cold","fire"):
    rate=float(TAGS[hz].mean()); ok,exp,tol,basis=HD.day_rate_ok("subregion",hz,rate)
    print(f"built {hz:4s} tag: {rate:.4f} of subregion-days ({'in band' if ok else 'OUT OF BAND'} "
          f"{exp:.4f} +/- {tol:.4f}, {basis.split(':')[0]}), definition {HD.definition_hash(hz)}")
# AR: the adopted flag, read from disk. ivt_p95_cov25 = subregion-mean IVT above its own +-15d
# day-of-year AR_PCTL AND a catalog AR covering >= AR_COVERAGE_FRACTION of the subregion's cells.
# The NCEP absolute-250 kg/m/s coastal-band NDJFM WECC-only rule that used to be rebuilt here is
# superseded.
AR_KEY="ivt_p%d_cov%d"%(HD.AR_PCTL,round(100*HD.AR_COVERAGE_FRACTION))
az=np.load(f"{EN}/ar_flag_variants.npz",allow_pickle=True)
asub=[str(x) for x in az["subregions"]]; ARF=az[AR_KEY].astype(bool)
adi={str(x):k for k,x in enumerate(az["dates"])}
arT=np.zeros((NS,ND),bool)
for s,nm in enumerate(names):
    if nm not in asub: continue
    r=ARF[asub.index(nm)]; arT[s]=np.array([r[adi[d]] if d in adi else False for d in days])
assert arT.any(), "AR flag did not align onto the net-load calendar"
TAGS["AR"]=arT
# TC: the deployed tc_local column, read from the analysis panel and consumed AS IS.
# hazard_defs defines the subregion tropical-cyclone hazard as a HURDAT2 track point within
# HD.TC_RADIUS_KM of the subregion CENTROID, with HD.TC_MISSING_WIND records dropped at both scales.
# Nothing in this repository builds tc_local, so neither that centroid-to-track-point geometry nor
# the -999 handling of the deployed column is verified here; the sentence above states the agreed
# definition, not a fact about how the column on disk was produced. The
# landfall-within-500km-of-centroid rule that used to be rebuilt in this file is superseded; the
# internal notes show it gave the OPPOSITE SIGN to tc_local.
pan=pd.read_parquet(f"{R1}/panel_v3.parquet"); pan["date"]=pd.to_datetime(pan.date)
_pst=HD.read_stamp(f"{R1}/panel_v3.parquet")
print("panel_v3.parquet stamp: %s"
      % (("written by %s, hazard_defs %s"%(_pst.get("script"),_pst.get("hazard_defs_version")))
         if _pst else "NONE. tc_local is consumed unverified; see the comment above."))
tcM=pan.pivot_table(index="subregion",columns="date",values="tc_local")
tcM.columns=tcM.columns.strftime("%Y-%m-%d")
tcT=np.zeros((NS,ND),bool)
for s,nm in enumerate(names):
    if nm in tcM.index: tcT[s]=tcM.loc[nm].reindex(days).fillna(0).values>0.5
assert tcT.any(), "tc_local did not align onto the net-load calendar"
TAGS["TC"]=tcT
for hz in ("AR","TC"):
    print(f"read {hz} flag: {int(TAGS[hz].sum())} subregion-days on "
          f"{sum(TAGS[hz][s].any() for s in range(NS))} subregions")

# ---------- hardening machinery ----------
SH=[int(round(365.25*k))+j for k in range(1,40) for j in (-14,-7,0,7,14)]   # 195 season-preserving circular shifts
def bh_fdr(p,q=0.05):
    """Benjamini-Hochberg WITHIN one hazard. The denominator m is the number of subregions that
    actually carry a test, that is those with at least one tagged day. A subregion with no tagged
    day has p = NaN, and it is excluded from m rather than counted as a non-rejection. For TC, which
    fires only where a track passes, m is well below 18, so the correction is weaker than an
    18-subregion correction would be. The realized m is printed as #tests(BH) in the table below."""
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
    p=np.array([ (1+np.sum(np.abs(null[:,s]-mu[s])>=np.abs(obs[s]-mu[s])))/(1+len(SH)) if not np.isnan(obs[s]) else np.nan for s in range(NS)])
    fdr=bh_fdr(p)
    q95=np.nanpercentile(np.abs(null-mu),95,axis=0)
    Robs=int(np.nansum(np.abs(obs-mu)>q95))
    Rnull=np.array([np.nansum(np.abs(null[i]-mu)>q95) for i in range(len(SH))])
    R95=float(np.percentile(Rnull,95))
    return obs,p,fdr,Robs,R95

rows=[]
print(f"{'hazard':6s} {'#tagged-subs':12s} {'#tests(BH)':10s} {'#FDR-sig':9s} {'R_obs':6s} {'R95(null)':10s} field-sig?")
for hz,tag in TAGS.items():
    obs,p,fdr,Robs,R95=harden(aN,tag)
    fs = Robs>R95
    ntest=int(np.isfinite(p).sum())
    for s in range(NS):
        if tag[s].sum()>0:
            rows.append(dict(hazard=hz,sub=names[s],net_MW=round(obs[s],0),
                pct=round(100*obs[s]/np.nanmean(clN[s,doy[tag[s]]]),1),p=round(p[s],3),
                fdr_sig=bool(fdr[s]),tag_days=int(tag[s].sum())))
    print(f"{hz:6s} {sum(tag[s].sum()>0 for s in range(NS)):12d} {ntest:10d} {int(fdr.sum()):9d} {Robs:6d} {R95:10.1f} {'YES' if fs else 'no'}")
# 17_harden_vre.py appends its own rows to this file. Writing fresh used to delete them, so the
# published table depended on which of the two ran last. Rows for hazards this script did not
# compute are preserved.
df=pd.DataFrame(rows); _sig=f"{LO}/hazard_significance.csv"
if _os.path.exists(_sig):
    _old=pd.read_csv(_sig); _keep=_old[~_old.hazard.isin(df.hazard.unique())]
    if len(_keep): print("  keeping %d rows from other hazards" % len(_keep), flush=True)
    df=pd.concat([_keep, df], ignore_index=True)
df.to_csv(_sig, index=False)
print("\nFDR-surviving subregions per hazard:")
for hz in TAGS:
    d=df[(df.hazard==hz)&(df.fdr_sig)].sort_values("pct",key=lambda c:c.abs(),ascending=False)
    print(f"  {hz:5s} ({int(df[df.hazard==hz].fdr_sig.sum())}/18): "+", ".join(f"{r['sub']} {r['pct']:+.0f}%" for _,r in d.head(8).iterrows()))
print("\nsaved hazard_significance.csv")
