#!/usr/bin/env python
"""4-factor variance decomposition behind r5_adequacy_decomp4.json (the file
fig4_adequacy.py panel e reads).  Factors: ssp (demand growth), vlab (VRE policy),
rcp (deployment axis), warm (cooler/hotter = the only fixed-fleet weather contrast).
Same one-way sum-of-squares share as r5_adequacy.decomp(), just four factors.
Reads R4_NETLOAD_HYDRO_SUMMARY.csv + r5_adequacy.csv; writes r5_adequacy_decomp4.json."""
import json, sys, numpy as np, pandas as pd

RN = "/data/cerf_out/r4_netload"
OUTF = sys.argv[1] if len(sys.argv) > 1 else f"{RN}/r5_adequacy_decomp4.json"

def prep(d):
    d = d.copy()
    d["ssp"] = d.scenario.str.split("_").str[-1]
    d["rcp"] = d.climate.str[:5]
    d["warm"] = np.where(d.climate.str.contains("hotter"), "hotter", "cooler")
    return d

S = prep(pd.read_csv(f"{RN}/R4_NETLOAD_HYDRO_SUMMARY.csv"))
D = prep(pd.read_csv(f"{RN}/r5_adequacy.csv"))

def decomp(d, col):
    y = d[col].values.astype(float); tot = ((y - y.mean()) ** 2).sum()
    out = {}
    for f in ["ssp", "vlab", "rcp", "warm"]:
        gm = d.groupby(f)[col].transform("mean").values
        out[f] = ((gm - y.mean()) ** 2).sum() / tot
    out["resid"] = max(0.0, 1 - sum(out.values()))
    return out

REP = {}
for nm, d_, c_ in [("d_peak_robust_pct (published)", S, "d_peak_robust_pct"),
                   ("exceed_histNHp999_pct", S, "exceed_histNHp999_pct"),
                   ("vre_ratio_top1", S, "vre_ratio_top1"),
                   ("firm_share", D, "firm_share"),
                   ("cap_credit", D, "cap_credit"),
                   ("peak_util", D, "peak_util")]:
    REP[nm] = decomp(d_, c_)
json.dump(REP, open(OUTF, "w"), indent=1)
print("rows: summary %d  adequacy %d  variants %s" % (len(S), len(D), sorted(S.vlab.unique())))
print("%-30s %7s %7s %7s %7s %7s" % ("metric", "ssp", "policy", "rcp", "warm", "resid"))
for k, v in REP.items():
    print("%-30s %6.1f%% %6.1f%% %6.1f%% %6.1f%% %6.1f%%"
          % (k, 100*v["ssp"], 100*v["vlab"], 100*v["rcp"], 100*v["warm"], 100*v["resid"]))
print("wrote", OUTF)
