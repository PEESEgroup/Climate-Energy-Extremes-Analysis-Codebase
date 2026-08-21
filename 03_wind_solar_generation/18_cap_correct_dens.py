"""CAPACITY correction: reweight each plant's capacity by its 2019 online fraction (EIA-860 2020,
op_year<=2019 with mid-year commissioning time-weighting; post-2019 plants -> 0). Recompute national +
per-BA wind gen vs EIA-930 2019 for the loss-corrected NET CF. Prints the bias ladder gross->net->net+cap.
Runs under /opt/pytorch (pandas+numpy)."""
import numpy as np, pandas as pd, re
OUT="/data/gen_targets/srgan3d_val"
# ---- EIA-860 2019 online fraction per plant code (pre-extracted csv) ----
pc=pd.read_csv("/tmp/e860/wind2019_onlinefrac.csv")
onlinefrac={int(k): float(f) for k,f in zip(pc["Plant Code"], pc["online_frac"])}
# ---- load NET CF + build corrected capacity ----
Z=np.load(f"{OUT}/s3d_cf_advep10_netdens.npz",allow_pickle=True)
plants=Z["plants"].astype(str); ba=Z["ba"].astype(str); cap=Z["cap"].astype(float); S=Z["stamps"].astype(str); cf=Z["cf"]
npl=len(plants); n=len(S)
def code(s):
    m=re.match(r"^(\d+)",str(s)); return int(m.group(1)) if m else -1
frac=np.array([onlinefrac.get(code(p),0.0) for p in plants])   # 0 for post-2019 (not in 2019 set)
cap_corr=cap*frac
print("cap: assumed %.1f GW  ->  2019-corrected %.1f GW  (dropped %.1f GW post-2019/unmatched)"%(
    cap.sum()/1e6, cap_corr.sum()/1e6, (cap.sum()-cap_corr.sum())/1e6))
# ---- EIA-930 observed per BA on S (verbatim validate_fig_tgw logic) ----
def corr(a,b):
    ok=np.isfinite(a)&np.isfinite(b)
    if ok.sum()<100 or a[ok].std()<1e-9 or b[ok].std()<1e-9: return np.nan
    return float(np.corrcoef(a[ok],b[ok])[0,1])
FULLy=pd.to_datetime([f"{s[:4]}-{s[4:6]}-{s[6:8]} {s[8:10]}:00" for s in S])
g=pd.read_parquet("/data/hydro/real/eia930_hourly_netgen_by_source.parquet",
    columns=["balancing_authority_code_eia","datetime_utc","generation_energy_source","net_generation_adjusted_mwh","net_generation_reported_mwh"])
g["v"]=g["net_generation_adjusted_mwh"].fillna(g["net_generation_reported_mwh"]); g["dt"]=pd.to_datetime(g["datetime_utc"]).dt.tz_localize(None)
g=g[(g.dt>=FULLy.min())&(g.dt<=FULLy.max())]
sw=g[g.generation_energy_source.astype(str).str.contains("wind",case=False,na=False)].groupby(["dt","balancing_authority_code_eia"],as_index=False)["v"].sum()
obs=sw.pivot(index="dt",columns="balancing_authority_code_eia",values="v").reindex(FULLy)
COV_MIN=min(2000,int(0.6*n))
def national(capvec):
    natl=np.zeros(n); nobs=np.zeros(n); used=[]; perba=[]
    for b in sorted(set(ba)):
        if b not in obs.columns: continue
        o=obs[b].values.astype(float); o=np.where(np.abs(o)>3e5,np.nan,o)
        if np.isfinite(o).sum()<COV_MIN or np.nanmean(o)<5: continue
        used.append(b); m=ba==b
        gen=np.nansum(cf[m]*capvec[m,None]/1000.0,axis=0); natl+=gen; nobs+=np.nan_to_num(o); perba.append(corr(gen,o))
    return natl,nobs,used,np.array(perba)
nA,nobs,used,pbA=national(cap)        # net, uncorrected cap
nB,_,_,pbB=national(cap_corr)         # net + capacity corrected
ob=np.nanmean(nobs)/1000
print("\nNATIONAL MEAN (GW), n=%d, %d BAs:"%(n,len(used)))
print("  EIA-930 observed         %.1f   (x1.00)"%ob)
print("  s3d NET (cap uncorr)     %.1f   (x%.2f)  r=%.3f perBAmed=%.3f"%(np.nanmean(nA)/1000, np.nanmean(nA)/1000/ob, corr(nA,nobs), np.nanmedian(pbA)))
print("  s3d NET + CAPACITY-2019  %.1f   (x%.2f)  r=%.3f perBAmed=%.3f"%(np.nanmean(nB)/1000, np.nanmean(nB)/1000/ob, corr(nB,nobs), np.nanmedian(pbB)))
np.savez(f"{OUT}/s3d_cf_advep10_netdenscap.npz", plants=plants, ba=ba, cap=cap_corr, stamps=S, cf=cf, s3d=cf, cap_orig=cap, online_frac=frac)
np.savez(f"{OUT}/s3d_val_summary_advep10_netdenscap.npz", stamps=S, used=np.array(used), natl_s3d=nB, nobs=nobs)
print("\nsaved s3d_cf_advep10_netdenscap.npz (corrected capacity) + summary")
