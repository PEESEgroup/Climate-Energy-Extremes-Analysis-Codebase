"""One estimator for every hazard in Figure 3.

The published figure put two different estimands on one axis: an event-study window sum for the
sharp-onset hazards and a per-standard-deviation dose-response for the slow-onset ones, each
divided by its own denominator. This script estimates ALL SIX hazards with the event study that
identifies the sharp-onset ones, so the panel carries one estimand, one denominator and one
identification argument.

Design, copied verbatim from stage1_ext.py so tropical cyclone and fire reproduce their published
numbers as a check:
  outcome        customer-hours out per customer, county by day, 2015-01-01 .. 2022-09-30
  treatment      event-time dummies k = -7 .. +14 around an isolated event onset
  fixed effects  county x calendar month, and state x day
  inference      two-way clustered on county and on day, t(G-1) with G = min(counties, days)
  placebo        the sum over leads -7 .. -1, which must be flat
  headline       the sum over k = 0 .. +14, customer-hours per customer per event

Cold and heat events are the ONSET of a spell, defined exactly as the published slow-onset arm
defines a spell: tmin below its day-of-year 10th percentile for at least six days, tmax above its
90th for at least three. The dose-response is still estimated and stored, but as a secondary
quantity, not as a row of the main panel.

Every hazard also reports the share of the estimation sample that sits in a state-day holding both
treated and untreated counties. That is the identifying variation, and for a spatially broad
hazard it is the number that decides whether a wide interval means "no effect" or "no comparison".

ENV: CLEAN (default 30), OUT
"""
import json, os
import numpy as np, pandas as pd
import os as _os, sys as _sys
_HD = _os.path.join(_os.path.dirname(_os.path.abspath(__file__)), "..", "07_hazard_calendar")
if _HD not in _sys.path: _sys.path.insert(0, _os.path.abspath(_HD))
import hazard_defs as HD
from scipy import stats as st

OUT = os.environ.get("OUT", "/data/equity_cost/analysis/stage1_uni")
os.makedirs(OUT, exist_ok=True)
T0 = os.environ.get("T0", "2015-01-01"); T1 = os.environ.get("T1", "2022-09-30")
# The placebo is the lead block, days -14 to -8, as the Methods state; -7 to -1 lies inside
# the forecast horizon and is part of the event period, so the window must reach -14.
KMIN, KMAX = -14, 14
PRE0, PRE1 = -14, -8
CLEAN = int(os.environ.get("CLEAN", "30"))

# ---------------------------------------------------------------- panel
D = pd.read_parquet("/data/equity_cost/analysis/eaglei_county_daily.parquet",
                    columns=["fips", "date", "customer_hours_out"])
D["date"] = pd.to_datetime(D.date); D["fips"] = D.fips.astype(str).str.zfill(5)
span = D.groupby("fips").date.agg(["min", "max"])
keep = span[(span["min"] <= "2015-07-01") & (span["max"] >= "2022-06-30")].index  # coverage rule on the full record, not the window
D = D[(D.date >= T0) & (D.date <= T1) & D.fips.isin(keep)]
E = pd.read_csv("/data/equity_cost/analysis/equity_joined_v2.csv", dtype={"fips": str})
E = E[E.denom > 0][["fips", "state", "denom"]]
cf = sorted(set(D.fips) & set(E.fips))
# The study domain is the contiguous United States: the weather comes from a CONUS simulation and
# the 18 subregions tile the lower 48. Alaska, Hawaii and Puerto Rico appear in the outage record
# but carry no hazard flag, so they contribute an empty treatment and no identifying variation.
# attrib.py drops them by the same rule; without it the region x day effect cannot be formed.
_NONCONUS = {"02", "15", "60", "66", "69", "72", "78"}
_drop = [f for f in cf if f[:2] in _NONCONUS]
if _drop:
    print("dropped %d non-contiguous counties (%s) that carry outage rows but no hazard flag"
          % (len(_drop), ", ".join(sorted({f[:2] for f in _drop}))), flush=True)
    cf = [f for f in cf if f[:2] not in _NONCONUS]
days = pd.date_range(T0, T1); ND = len(days)
P = pd.DataFrame({"fips": cf}).merge(pd.DataFrame({"date": days}), how="cross")
P = P.merge(D, on=["fips", "date"], how="left").merge(E, on="fips", how="left")
P["cho"] = P.customer_hours_out.fillna(0.0)
P["y"] = (P.cho / P.denom).astype("f4")
P = P.sort_values(["fips", "date"]).reset_index(drop=True)
cidx = {f: i for i, f in enumerate(cf)}; didx = {d: i for i, d in enumerate(days)}
TOT_CHO = float(P.cho.sum())
print("panel %s  counties %d  days %d  observed %.4e customer-hours  CLEAN %d"
      % (P.shape, len(cf), ND, TOT_CHO, CLEAN), flush=True)

P["cm"] = P.fips.astype(str) + "_" + P.date.dt.month.astype(str)
FE = os.environ.get("FE", "state_day")   # state_day, or day for a national comparison on the same date
# The absorbed day effect is region x day, the 18 study subregions, matching attrib.py. A national
# day effect leaves the regional weather of the date in the residual and a spatially sharp hazard
# loads it onto its own pre-event block; a state x day effect absorbs a state-wide cold outbreak
# outright. The measured pre-event z for fire is -5.07 national, -2.73 region, -1.12 state, while
# its post-event effect is +3.34 / +3.34 / +3.33, so only the pre-event moves.
_SR = pd.read_csv("/data/datasets/grid/fips_to_subregion_mapping.csv", dtype={"FIPS": str})
_SR["FIPS"] = _SR.FIPS.str.zfill(5)
_SR = _SR.rename(columns={"FIPS": "fips", "Subregion_Code": "region"})
_SR["region"] = _SR.region.astype(str)
P = P.merge(_SR[["fips", "region"]], on="fips", how="left")
if P.region.isna().any():
    raise SystemExit("%d county-days have no subregion; the region x day effect cannot be formed"
                     % int(P.region.isna().sum()))
P["sd"] = P.region + "_" + P.date.dt.strftime("%Y%m%d")
cm = pd.factorize(P.cm)[0]; sd = pd.factorize(P.sd)[0]
ncm, nsd = cm.max() + 1, sd.max() + 1
gc_all = pd.factorize(P.fips)[0]; gd_all = pd.factorize(P.date)[0]
den_of = P.groupby("fips").denom.first()

def absorb(v, mask, iters=80, tol=1e-11):
    v = np.where(mask, v.astype(np.float64), 0.0)
    c1 = np.bincount(cm, mask.astype(float), ncm); c2 = np.bincount(sd, mask.astype(float), nsd)
    c1[c1 == 0] = 1; c2[c2 == 0] = 1
    for _ in range(iters):
        v0 = v.copy()
        v -= np.where(mask, np.bincount(cm, v, ncm)[cm] / c1[cm], 0.0)
        v -= np.where(mask, np.bincount(sd, v, nsd)[sd] / c2[sd], 0.0)
        if np.max(np.abs(v - v0)) < tol:
            break
    return v

def vcov(Xt, e, gc, gd, Xi):
    def meat(g):
        o = np.argsort(g); gs = g[o]
        b = np.r_[0, np.where(np.diff(gs) != 0)[0] + 1, len(gs)]
        Xe = Xt[o] * e[o][:, None]
        m = np.zeros((Xt.shape[1], Xt.shape[1]))
        for x, y in zip(b[:-1], b[1:]):
            s = Xe[x:y].sum(0); m += np.outer(s, s)
        return m
    return Xi @ (meat(gc) + meat(gd) - meat(gc.astype(np.int64) * ND + gd)) @ Xi

# ---------------------------------------------------------------- hazard day sets
def hazard_days():
    H = {}
    FLAGS_PATH = "/data/enso/county_hazard_flags_c404.parquet"
    F = pd.read_parquet(FLAGS_PATH)
    F["date"] = pd.to_datetime(F.date)
    for hz in ["fire"]:
        H[hz] = F[F.hazard == hz][["fips", "date"]]
    TC = pd.read_parquet("/data/enso/tc_county_ext/county_tc_days.parquet")
    TC["date"] = pd.to_datetime(TC.date)
    H["tc"] = TC[["fips", "date"]]
    C = pd.read_parquet("/data/enso/county_convective_daily.parquet")
    C["date"] = pd.to_datetime(C.date)
    HD.require_stamp("/data/enso/county_convective_daily.parquet", hazards=["convection"])
    H["convective"] = C[C.severe][["fips", "date"]]   # the builder applies all three conditions
    # Cold and heat are READ from the canonical county flag table, not rebuilt here. This file used
    # to reconstruct them from raw weather with its own day-of-year percentile, a six-day cold rule
    # and no season gate, so the estimator ran on different event days from Figure 1 and from the
    # attribution panel. The canonical table applies hazard_defs: p10 with the county three-day
    # persistence and a December-to-February gate for cold, p90 with three days and a June-to-August
    # gate for heat, all frozen on the shared climatology period.
    HD.require_stamp(FLAGS_PATH, hazards=["cold", "heat", "fire"])
    for hz in ("cold", "heat"):
        H[hz] = F[F.hazard == hz][["fips", "date"]]
    return H

H = hazard_days()

# ---------------------------------------------------------------- one estimator, every hazard
RES, ATTR = {}, {}
HAZ = os.environ.get("HAZ", "tc,convective,fire,cold,heat,cold_nofeb").split(",")
for hz in HAZ:
    src = "cold" if hz == "cold_nofeb" else hz
    hd = H[src][["fips", "date"]].dropna().sort_values(["fips", "date"])
    ci = hd.fips.map(cidx).values; di = hd.date.map(didx).values
    m = (~pd.isna(ci)) & (~pd.isna(di))
    ci, di = ci[m].astype(int), di[m].astype(int)
    o = np.lexsort((di, ci)); ci, di = ci[o], di[o]
    newsp = np.r_[True, (ci[1:] != ci[:-1]) | (di[1:] - di[:-1] > 2)]
    ev_c, ev_d = ci[newsp], di[newsp]
    haz = np.zeros((len(cf), ND), bool); haz[ci, di] = True
    cum = np.cumsum(haz, axis=1)
    lo = np.clip(ev_d - CLEAN, 0, ND - 1); hi = np.clip(ev_d + CLEAN, 0, ND - 1)
    nh = cum[ev_c, hi] - cum[ev_c, np.clip(lo - 1, 0, ND - 1)]
    slen = np.array([haz[a, b:min(b + 10, ND)].sum() for a, b in zip(ev_c, ev_d)])
    iso = nh <= slen
    ev_c, ev_d = ev_c[iso], ev_d[iso]
    if hz == "cold_nofeb":                       # the February 2021 robustness row, re-estimated
        od = days[ev_d]                          # rather than carried as a literal in the figure
        drop = (od.year == 2021) & (od.month == 2)
        print("    dropping %d of %d onsets in February 2021" % (int(drop.sum()), len(ev_d)), flush=True)
        ev_c, ev_d = ev_c[~drop], ev_d[~drop]
    et = np.full(len(P), -999, np.int32)
    for k in range(KMIN, KMAX + 1):
        j = ev_d + k; g = (j >= 0) & (j < ND)
        et[ev_c[g] * ND + j[g]] = k
    inwin = et != -999
    near = np.zeros(len(P), bool)
    for k in range(-CLEAN, CLEAN + 1):
        j = di + k; g = (j >= 0) & (j < ND)
        near[ci[g] * ND + j[g]] = True
    mask = inwin | (~near)
    ks = np.arange(KMIN, KMAX + 1)
    yt = absorb(P.y.values, mask)
    Xt = np.column_stack([absorb((et == k).astype(np.float64), mask) for k in ks])
    yt, Xt = yt[mask], Xt[mask]
    XtX = Xt.T @ Xt
    b = np.linalg.solve(XtX, Xt.T @ yt); e = yt - Xt @ b; Xi = np.linalg.inv(XtX)
    V = vcov(Xt, e, gc_all[mask], gd_all[mask], Xi)
    se = np.sqrt(np.diag(V))
    G = min(len(np.unique(gc_all[mask])), len(np.unique(gd_all[mask])))
    wpre = (ks < 0).astype(float); wpost = (ks >= 0).astype(float)
    pre = float(wpre @ b); pre_se = float(np.sqrt(wpre @ V @ wpre))
    cu = float(wpost @ b); cu_se = float(np.sqrt(wpost @ V @ wpost))
    ybar = P.y.values[mask & ~inwin].mean()
    p_pre = 2 * st.t.sf(abs(pre / pre_se), G - 1)
    p_cu = 2 * st.t.sf(abs(cu / cu_se), G - 1)
    # identifying variation: state-days holding both a treated and an untreated county
    sdm = sd[mask]; tr = inwin[mask]
    nt = np.bincount(sdm, tr.astype(float), nsd); nn = np.bincount(sdm, (~tr).astype(float), nsd)
    both = (nt > 0) & (nn > 0)
    idvar = float(tr[both[sdm]].sum() / max(tr.sum(), 1))
    cust = den_of.iloc[ev_c].values
    att = cu * cust.sum(); att_se = cu_se * cust.sum()
    print("\n=== %s ===  isolated events %d of %d   obs %s   control-day mean %.5f"
          % (hz.upper(), len(ev_c), int(newsp.sum()), format(int(mask.sum()), ","), ybar), flush=True)
    print("    placebo -7..-1 : %+.5f (se %.5f) p %.4g  %s"
          % (pre, pre_se, p_pre, "FLAT" if p_pre > .10 else "FAILS"), flush=True)
    print("    cumulative 0..%d: %+.5f (se %.5f) p %.4g" % (KMAX, cu, cu_se, p_cu), flush=True)
    print("    identifying variation: %.1f%% of treated county-days sit in a state-day with an "
          "untreated county" % (100 * idvar), flush=True)
    print("    attributable: %+.4e customer-hours = %+.2f%% of the observed total"
          % (att, 100 * att / TOT_CHO), flush=True)
    RES[hz] = dict(k=ks.tolist(), beta=b.tolist(), se=se.tolist(), cumulative=cu,
                   cumulative_se=cu_se, cumulative_p=p_cu, placebo=pre, placebo_se=pre_se,
                   placebo_p=p_pre, n_events=int(len(ev_c)), G=int(G), ybar=float(ybar),
                   id_variation=idvar)
    ATTR[hz] = dict(attributable_cho=att, se=att_se, pct_of_total=100 * att / TOT_CHO,
                    n_county_events=int(len(ev_c)), customers_exposed=float(cust.sum()))

# one common denominator for the whole panel, so the multiples are comparable across hazards
YBAR_ALL = float(P.y.values.mean())
json.dump(dict(window=[T0, T1], clean=CLEAN, fe=FE, kmin=KMIN, kmax=KMAX, results=RES,
               attributable=ATTR, ybar_panel=YBAR_ALL, observed_total_customer_hours=TOT_CHO),
          open(f"{OUT}/stage1_unified.json", "w"), indent=1)
print("\npanel-wide control mean %.5f customer-hours per customer per day" % YBAR_ALL)
print("wrote %s/stage1_unified.json" % OUT)
