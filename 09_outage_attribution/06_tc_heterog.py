"""Does the hurricane effect differ by county trait?

Figure 3's causal column rested on the severe-convection event study, whose placebo fails once the
control pool is built explicitly. The question is therefore re-asked on the one design that does
survive: the matched hurricane contrast, estimated separately inside terciles of each trait, with
the same estimator, the same pool rule and a placebo per tercile.
"""
import json, os
import numpy as np, pandas as pd
import os as _os, sys as _sys
_HD = _os.path.abspath(_os.path.join(_os.path.dirname(_os.path.abspath(__file__)), "..", "07_hazard_calendar"))
if _HD not in _sys.path: _sys.path.insert(0, _HD)
import hazard_defs as HD
T0, T1 = "2015-01-01", "2022-09-30"
KMIN, KMAX, PRE0, PRE1, M, B = -7, 14, -14, -8, 20, 999
rng = np.random.default_rng(20260813)
D = pd.read_parquet("/data/equity_cost/analysis/eaglei_county_daily.parquet",
                    columns=["fips", "date", "customer_hours_out"])
D["date"] = pd.to_datetime(D.date); D["fips"] = D.fips.astype(str).str.zfill(5)
sp = D.groupby("fips").date.agg(["min", "max"])
keep = sp[(sp["min"] <= "2015-07-01") & (sp["max"] >= "2022-06-30")].index
D = D[(D.date >= T0) & (D.date <= T1) & D.fips.isin(keep)]
E = pd.read_csv("/data/equity_cost/analysis/equity_joined_v2.csv", dtype={"fips": str})
E = E[E.denom > 0][["fips", "state", "denom"]]
cf = sorted(set(D.fips) & set(E.fips)); cidx = {f: i for i, f in enumerate(cf)}
days = pd.date_range(T0, T1); ND = len(days); didx = {d: i for i, d in enumerate(days)}
E = E.set_index("fips").reindex(cf)
state = pd.factorize(E.state.values)[0]; denom = E.denom.values.astype(float)
D = D[D.fips.isin(cidx)]
Y = np.zeros((len(cf), ND), np.float32)
ii = D.fips.map(cidx).values.astype(int); jj = D.date.map(didx).values.astype(int)
Y[ii, jj] = (D.customer_hours_out.values / denom[ii]).astype(np.float32)
TR = pd.read_parquet("/data/equity_cost/analysis/step4/county_attributable.parquet")
TR = TR.set_index("fips").reindex(cf)
TRAITS = {"poverty_rate": "poverty", "minority_pct": "minority share", "median_income": "income",
          "median_age": "median age", "ug_share_dom": "undergrounding"}
TC = pd.read_parquet("/data/enso/tc_county_ext/county_tc_days.parquet"); TC["date"] = pd.to_datetime(TC.date)
ci = TC.fips.map(cidx).values; di = TC.date.map(didx).values
m = (~pd.isna(ci)) & (~pd.isna(di)); ci, di = ci[m].astype(int), di[m].astype(int)
o = np.lexsort((di, ci)); ci, di = ci[o], di[o]
G = np.zeros((len(cf), ND), bool); G[ci, di] = True; CS = np.cumsum(G, 1)
new = np.r_[True, (ci[1:] != ci[:-1]) | (di[1:] - di[:-1] > 2)]
ev_c, ev_d = ci[new], di[new]
ok = (ev_d + PRE0 >= 0) & (ev_d + KMAX < ND); ev_c, ev_d = ev_c[ok], ev_d[ok]
ks = np.arange(KMIN, KMAX + 1)
ANY = np.zeros_like(G)
for src, flt in [("/data/enso/county_hazard_flags_c404.parquet", None),
                 ("/data/enso/county_convective_daily.parquet", "conv")]:
    f = pd.read_parquet(src); f["date"] = pd.to_datetime(f.date)
    if flt == "conv":
        f = f[f.severe]
    a = f.fips.map(cidx).values; b = f.date.map(didx).values
    mm = (~pd.isna(a)) & (~pd.isna(b)); ANY[a[mm].astype(int), b[mm].astype(int)] = True
ANY |= G
eff = np.full((len(ev_c), len(ks)), np.nan); w = np.zeros(len(ev_c))
for e, (i, t0) in enumerate(zip(ev_c, ev_d)):
    lo, hi = max(t0 + PRE0, 0), min(t0 + KMAX, ND - 1)
    clean = (CS[:, hi] - CS[:, max(lo - 1, 0)]) == 0
    elig = np.where(clean & (state == state[i]))[0]; elig = elig[elig != i]
    if len(elig) < M: continue
    a, b = t0 + PRE0, t0 + PRE1
    pi = Y[i, a:b].mean(); pj = Y[elig, a:b].mean(1)
    sel = elig[np.argsort(np.abs(pj - pi))[:M]]
    dev = Y[sel][:, t0 + ks] - Y[sel, a:b].mean(1)[:, None]
    dev = np.where(ANY[sel][:, t0 + ks], np.nan, dev)
    nok = np.isfinite(dev).sum(0)
    if nok.mean() < 10 or nok.min() < 5: continue
    eff[e] = (Y[i, t0 + ks] - pi) - np.nanmean(dev, axis=0); w[e] = denom[i]
use = w > 0
print("usable hurricane events: %d of %d" % (use.sum(), len(ev_c)), flush=True)
wpost = (ks >= -3).astype(float); wpre = (ks <= -4).astype(float)
out = {}
for col, nm in TRAITS.items():
    v = TR[col].values.astype(float)[ev_c]
    good = use & np.isfinite(v)
    q = np.nanquantile(v[good], [1 / 3, 2 / 3])
    terc = np.digitize(v, q)
    row = {}
    for t, lab in enumerate(["low", "middle", "high"]):
        s = good & (terc == t)
        if s.sum() < 30:
            row[lab] = dict(n=int(s.sum()), status="too few events"); continue
        ee, ww = eff[s], w[s]
        cu = float(wpost @ ((ee * ww[:, None]).sum(0) / ww.sum()))
        pr = float(wpre @ ((ee * ww[:, None]).sum(0) / ww.sum()))
        bs = np.empty(B)
        for bb in range(B):
            sg = rng.choice([-1.0, 1.0], len(ee))
            bs[bb] = wpost @ (((ee * sg[:, None]) * ww[:, None]).sum(0) / ww.sum())
        row[lab] = dict(n=int(s.sum()), cumulative=cu, se=float(bs.std()),
                        p=float((np.abs(bs) >= abs(cu)).mean()), placebo=pr)
    ok3 = [row[l] for l in ["low", "high"] if "cumulative" in row[l]]
    if len(ok3) == 2:
        d = ok3[1]["cumulative"] - ok3[0]["cumulative"]
        sd = np.hypot(ok3[0]["se"], ok3[1]["se"])
        row["high_minus_low"] = dict(diff=float(d), se=float(sd), z=float(d / sd))
    out[nm] = row
    print("  %-16s low %+7.2f (n %3d)  mid %+7.2f  high %+7.2f (n %3d)   high-low %+6.2f (z %+.2f)"
          % (nm, row["low"].get("cumulative", np.nan), row["low"]["n"],
             row["middle"].get("cumulative", np.nan), row["high"].get("cumulative", np.nan),
             row["high"]["n"], row.get("high_minus_low", {}).get("diff", np.nan),
             row.get("high_minus_low", {}).get("z", np.nan)), flush=True)
json.dump(out, open("/data/equity_cost/analysis/stage1_stack/tc_heterogeneity.json", "w"), indent=1)
print("wrote tc_heterogeneity.json")
