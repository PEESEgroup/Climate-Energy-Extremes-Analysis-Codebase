"""Adopt the ivt_p95_cov25 atmospheric-river flag and regenerate everything Figure 1 reads.

Verified first by `ar_verify.py` in this folder: the ERA5-to-subregion geometry correlates 0.9973
with model-grid area; five documented landfalls are all detected, each at the 98.9th to 100th percentile of its
subregion's own IVT record; and the two properties that follow from the construction are recorded
rather than hidden (the day-of-year percentile removes the seasonal concentration, and a threshold
on a continuous field marks the peak day of a passage rather than its whole duration).

DEFINITIONS. The flag name is assembled from hazard_defs.AR_PCTL and AR_COVERAGE_FRACTION rather
than typed out, so this file cannot drift from `06_ar_variants.py`, which assembles the same name from
the same two constants. No percentile, window or threshold is written here.

REFUSAL. `ar_flag_variants.npz` must carry the hazard_defs stamp that `06_ar_variants.py` writes, and
its `ar` definition hash must equal the current one. A missing or stale stamp raises here rather
than being read, which is the point of the stamp: it stops a superseded flag reaching the
regressions. The deployed file predates the stamp, so `06_ar_variants.py` must be rerun before this
script will run.

Recomputed here, with the new flag in the AR slot and nothing else changed:
  main_joint            daily mean and daily peak, six hazards, two-way clustered SE
  channel decomposition demand and generation terms for all six, same clustering
  distributed lag       six hazards at t-7 .. t+7, betas only, which is all panel e uses
Written to r1_final_main_ourchain.json and r1_decomposition_ourchain.json, so the published files
stay untouched.
"""
import json
import os as _os, sys as _sys

import numpy as np, pandas as pd

for _c in (_os.path.join(_os.path.dirname(_os.path.dirname(_os.path.abspath(__file__))),
                         "07_hazard_calendar"),
           _os.path.dirname(_os.path.abspath(__file__))):
    if _os.path.isdir(_c) and _c not in _sys.path:
        _sys.path.insert(0, _c)
import hazard_defs as HD

R1 = "/data/enso/r1_causal"
HAZ = ["heat", "cold", "fire", "vre_drought", "ar", "tc_local"]
AR_KEY = "ivt_p%d_cov%d" % (HD.AR_PCTL, round(100 * HD.AR_COVERAGE_FRACTION))
P = pd.read_parquet(f"{R1}/panel_v3.parquet")
P["date"] = pd.to_datetime(P.date)
# The five panel hazards are consumed as they are. panel_v3.parquet is built outside this file, so
# its stamp is reported and not enforced here; a missing stamp means the heat, cold, fire,
# vre_drought and tc_local columns cannot be traced to a definition hash from this script.
_pst = HD.read_stamp(f"{R1}/panel_v3.parquet")
print("panel_v3.parquet stamp: %s"
      % (("written by %s, hazard_defs %s" % (_pst.get("script"), _pst.get("hazard_defs_version")))
         if _pst else "NONE. The five panel hazard columns are unverified here."), flush=True)

AR_NPZ = "/data/enso/ar_flag_variants.npz"
Z = np.load(AR_NPZ, allow_pickle=True)
# Refuse a flag file that no builder stamped, or one built from superseded constants.
if "hazard_defs_stamp" not in Z.files:
    raise ValueError("%s carries no hazard_defs stamp: it predates the shared definitions. Rerun "
                     "08_adequacy_analysis/06_ar_variants.py before reading it." % AR_NPZ)
_st = json.loads(str(Z["hazard_defs_stamp"]))
_want, _got = HD.definition_hash("ar"), (_st.get("definition_hash") or {}).get("ar")
if _got != _want:
    raise ValueError("%s holds the atmospheric-river flag at definition %s, but the current "
                     "definition is %s: it was written by a superseded builder. Rerun "
                     "08_adequacy_analysis/06_ar_variants.py." % (AR_NPZ, _got, _want))
if _st.get("extra", {}).get("adopted_key") not in (None, AR_KEY):
    raise ValueError("%s was written with adopted key %s, but this file reads %s"
                     % (AR_NPZ, _st["extra"]["adopted_key"], AR_KEY))
print("AR flag file: written by %s, hazard_defs %s, definition %s"
      % (_st.get("script"), _st.get("hazard_defs_version"), _got), flush=True)
sub = [str(x) for x in Z["subregions"]]
dts = pd.to_datetime([str(x) for x in Z["dates"]])
A = pd.DataFrame(Z[AR_KEY].T, index=dts, columns=sub).stack()
A.index = A.index.set_names(["date", "subregion"])
ridx = pd.MultiIndex.from_arrays([P.subregion.values, P.date.values])
P["ar"] = A.reorder_levels(["subregion", "date"]).reindex(ridx).fillna(False).values.astype(float)
print("new AR flag on panel: %s days, %.2f%% of rows"
      % (format(int(P.ar.sum()), ","), 100 * P.ar.mean()), flush=True)

def fit(y, X, cl_a, cl_b):
    XtXi = np.linalg.pinv(X.T @ X)
    b = XtXi @ (X.T @ y)
    e = y - X @ b
    def meat(g):
        M = np.zeros((X.shape[1], X.shape[1]))
        for _, idx in pd.Series(np.arange(len(g))).groupby(g):
            Xi = X[idx.values]; s = Xi.T @ e[idx.values]
            M += np.outer(s, s)
        return M
    V = XtXi @ (meat(cl_a) + meat(cl_b) - meat(pd.Series(list(zip(cl_a, cl_b))))) @ XtXi
    return b, np.sqrt(np.maximum(np.diag(V), 0))

def ols(y, X):
    return np.linalg.pinv(X.T @ X) @ (X.T @ y)

FE_sub = pd.get_dummies(P.subregion, drop_first=True).values.astype(float)
FE_mon = pd.get_dummies(P.month, drop_first=True).values.astype(float)
CTRL = P[["ONI", "PNA", "NAO", "AO", "GMST"]].values.astype(float)
H = P[HAZ].values.astype(float)
m = P.netload_anom_mean.notna().values
X = np.column_stack([np.ones(len(P)), H, CTRL, FE_sub, FE_mon])[m]
d = P[m]

OUT = {"main_joint": {}, "flag": AR_KEY, "ar_definition_hash": HD.definition_hash("ar"),
       "n_ar_days": int(P.ar.sum()), "ar_day_pct": float(100 * P.ar.mean())}
for ycol, key in (("netload_anom_mean", "netload_anom_mean"), ("netload_anom_peak", "netload_anom_peak")):
    b, se = fit(d[ycol].values.astype(float), X, d.subregion.values, d.date.values)
    OUT["main_joint"][key] = {h: dict(beta=float(b[1 + i]), se=float(se[1 + i]),
                                      t=float(b[1 + i] / se[1 + i])) for i, h in enumerate(HAZ)}
    print("%s: %s" % (key, {h: round(b[1 + i], 1) for i, h in enumerate(HAZ)}), flush=True)

DEC = {}
bl, sl = fit(d.load_anom_mean.values.astype(float), X, d.subregion.values, d.date.values)
bv, sv = fit(d.vre_anom_mean.values.astype(float), X, d.subregion.values, d.date.values)
bn = OUT["main_joint"]["netload_anom_mean"]
for i, h in enumerate(HAZ):
    DEC[h] = dict(beta_load=float(bl[1 + i]), se_load=float(sl[1 + i]),
                  t_load=float(bl[1 + i] / sl[1 + i]),
                  beta_vre=float(bv[1 + i]), se_vre=float(sv[1 + i]),
                  t_vre=float(bv[1 + i] / sv[1 + i]), beta_net=bn[h]["beta"])
    print("  %-12s load %+8.1f (t %+5.2f)  vre %+8.1f (t %+5.2f)  net %+8.1f  identity gap %.3f"
          % (h, DEC[h]["beta_load"], DEC[h]["t_load"], DEC[h]["beta_vre"], DEC[h]["t_vre"],
             DEC[h]["beta_net"], abs(DEC[h]["beta_load"] - DEC[h]["beta_vre"] - DEC[h]["beta_net"])),
          flush=True)

# distributed lag: betas only, which is all the cumulative panel uses
print("distributed lag ...", flush=True)
cols = []
for h in HAZ:
    s = P.set_index(["subregion", "date"])[h].unstack(0)
    for k in range(-7, 8):
        # shift(k), not shift(-k). pandas shift(-k) puts the value from t+k at t, which is a LEAD,
        # and the label below calls positive k a lag. The two were swapped, so the whole lag axis
        # of Figure 1e ran backwards: the shaded pre-event region was built from post-event days.
        # Check: pd.Series([0,0,1,0,0]).shift(1) puts the impulse at t=3, one day AFTER it happened,
        # which is what "lag 1" means.
        cols.append(("%s|%s" % (h, k), s.shift(k).stack().reorder_levels([1, 0]).reindex(ridx).values))
XL = np.column_stack([np.ones(len(P))] + [c[1] for c in cols] + [CTRL, FE_sub, FE_mon])
ok = m & np.isfinite(XL).all(1)
bL = ols(P.netload_anom_mean.values.astype(float)[ok], XL[ok])
print("  lag model n = %s" % format(int(ok.sum()), ","), flush=True)
G = {}
for j, (nm, _) in enumerate(cols):
    h, k = nm.split("|"); k = int(k)
    lab = "day_of" if k == 0 else ("lead%d" % (-k) if k < 0 else "lag%d" % k)
    G.setdefault(h, {})[lab] = dict(beta=float(bL[1 + j]))
OUT["gate2_distributed_lag"] = {"netload_anom_mean": G}
for h in HAZ:
    order = ["lead%d" % i for i in range(7, 0, -1)] + ["day_of"] + ["lag%d" % i for i in range(1, 8)]
    cum = np.cumsum([G[h][k]["beta"] for k in order]) / 1e3
    print("  %-12s cum15 %+7.2f GW, share before day 0 %+4.0f%% (of the 15-day total)"
          % (h, cum[-1], 100 * cum[6] / cum[-1]), flush=True)

json.dump(OUT, open(f"{R1}/r1_final_main_ourchain.json", "w"), indent=1)
json.dump({"channel_decomposition_mean": DEC}, open(f"{R1}/r1_decomposition_ourchain.json", "w"), indent=1)
print("\nwrote r1_final_main_ourchain.json and r1_decomposition_ourchain.json")
