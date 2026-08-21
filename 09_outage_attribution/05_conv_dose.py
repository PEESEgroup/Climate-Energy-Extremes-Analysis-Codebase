"""Severe convection, identified inside the treated counties rather than against a control pool.

Clean controls do not exist for a hazard that flags five county-days a year almost everywhere, so
the question is changed to one the data can answer: given that a storm occurred, does a STRONGER
storm cause more outage? Identification comes from variation in storm intensity within a county and
calendar month, which is a property of the atmosphere on the day rather than of the county.

  y_it = sum_k beta_k z_k,it + alpha_{i,month} + lambda_year + e_it     on convective county-days
  y    = customer-hours out per customer accumulated over days 0..+2
  z    = standardised peak reflectivity, peak CAPE, and the fraction of the county above 50 dBZ

Inference is two-way clustered on county and date. A placebo repeats the regression on the outage
accumulated over days -5..-3, which the storm cannot have caused.
"""
import json
import numpy as np, pandas as pd
import os as _os, sys as _sys
_HD = _os.path.abspath(_os.path.join(_os.path.dirname(_os.path.abspath(__file__)), "..", "07_hazard_calendar"))
if _HD not in _sys.path: _sys.path.insert(0, _HD)
import hazard_defs as HD
from scipy import stats as st
T0, T1 = "2015-01-01", "2022-09-30"
D = pd.read_parquet("/data/equity_cost/analysis/eaglei_county_daily.parquet",
                    columns=["fips", "date", "customer_hours_out"])
D["date"] = pd.to_datetime(D.date); D["fips"] = D.fips.astype(str).str.zfill(5)
sp = D.groupby("fips").date.agg(["min", "max"])
keep = sp[(sp["min"] <= "2015-07-01") & (sp["max"] >= "2022-06-30")].index
D = D[(D.date >= T0) & (D.date <= T1) & D.fips.isin(keep)]
E = pd.read_csv("/data/equity_cost/analysis/equity_joined_v2.csv", dtype={"fips": str})
E = E[E.denom > 0][["fips", "denom"]]
cf = sorted(set(D.fips) & set(E.fips)); cidx = {f: i for i, f in enumerate(cf)}
days = pd.date_range(T0, T1); ND = len(days); didx = {d: i for i, d in enumerate(days)}
den = E.set_index("fips").reindex(cf).denom.values.astype(float)
Y = np.zeros((len(cf), ND), np.float32)
D = D[D.fips.isin(cidx)]
Y[D.fips.map(cidx).values.astype(int), D.date.map(didx).values.astype(int)] = \
    (D.customer_hours_out.values / den[D.fips.map(cidx).values.astype(int)]).astype(np.float32)
C = pd.read_parquet("/data/enso/county_convective_daily.parquet",
                    columns=["fips", "date", "refl_max", "cape_max", "frac50"])
C["date"] = pd.to_datetime(C.date)
C = C[C.severe]
C["ci"] = C.fips.map(cidx); C["di"] = C.date.map(didx)
C = C.dropna(subset=["ci", "di"]); C["ci"] = C.ci.astype(int); C["di"] = C.di.astype(int)
C = C[(C.di >= 5) & (C.di < ND - 3)]
print("convective county-days with a usable window: %d over %d counties" % (len(C), C.ci.nunique()), flush=True)
def acc(ci, di, a, b):
    return np.array([Y[c, d + a:d + b + 1].sum() for c, d in zip(ci, di)])
C["y"] = acc(C.ci.values, C.di.values, 0, 2)
C["y_pl"] = acc(C.ci.values, C.di.values, -5, -3)
for z in ["refl_max", "cape_max", "frac50"]:
    C["z_" + z] = (C[z] - C[z].mean()) / C[z].std()
C["month"] = C.date.dt.month; C["year"] = C.date.dt.year
def fe2(d, ycol, xs):
    g1 = pd.factorize(d.fips.astype(str) + "_" + d.month.astype(str))[0]
    g2 = pd.factorize(d.year)[0]
    n1, n2 = g1.max() + 1, g2.max() + 1
    def absorb(v):
        v = v.astype(float).copy()
        for _ in range(60):
            v0 = v.copy()
            v -= np.bincount(g1, v, n1)[g1] / np.bincount(g1, minlength=n1)[g1]
            v -= np.bincount(g2, v, n2)[g2] / np.bincount(g2, minlength=n2)[g2]
            if np.max(np.abs(v - v0)) < 1e-11: break
        return v
    yy = absorb(d[ycol].values); XX = np.column_stack([absorb(d[c].values) for c in xs])
    XtX = XX.T @ XX; b = np.linalg.solve(XtX, XX.T @ yy); e = yy - XX @ b; Xi = np.linalg.inv(XtX)
    gc = pd.factorize(d.fips)[0]; gd = pd.factorize(d.date)[0]
    def meat(g):
        o = np.argsort(g); gs = g[o]
        bd = np.r_[0, np.where(np.diff(gs) != 0)[0] + 1, len(gs)]
        Xe = XX[o] * e[o][:, None]; m = np.zeros((len(xs), len(xs)))
        for x, y_ in zip(bd[:-1], bd[1:]):
            s = Xe[x:y_].sum(0); m += np.outer(s, s)
        return m
    g12 = pd.factorize(d.fips.astype(str) + "_" + d.date.astype(str))[0]
    V = Xi @ (meat(gc) + meat(gd) - meat(g12)) @ Xi
    se = np.sqrt(np.diag(V)); G = min(len(np.unique(gc)), len(np.unique(gd)))
    return b, se, 2 * st.t.sf(np.abs(b / se), G - 1), G
XS = ["z_refl_max", "z_cape_max", "z_frac50"]
res = {}
for tag, ycol in [("effect, days 0..+2", "y"), ("placebo, days -5..-3", "y_pl")]:
    b, se, p, G = fe2(C, ycol, XS)
    print("  %-22s  G=%d" % (tag, G), flush=True)
    for k, x in enumerate(XS):
        print("     %-12s %+8.4f (se %.4f, p %.3g)" % (x.replace("z_", ""), b[k], se[k], p[k]), flush=True)
    res[tag] = {x: dict(beta=float(b[k]), se=float(se[k]), p=float(p[k])) for k, x in enumerate(XS)}
mb = C.y.mean()
print("\nmean outage over days 0..+2 on a convective county-day: %.4f customer-hours per customer" % mb)
print("a one standard deviation stronger storm, by peak reflectivity, changes that by %+.1f%%"
      % (100 * res["effect, days 0..+2"]["z_refl_max"]["beta"] / mb))
json.dump(dict(n=len(C), mean_burden=float(mb), results=res),
          open("/data/equity_cost/analysis/stage1_uni/conv_dose.json", "w"), indent=1)
print("wrote conv_dose.json")
