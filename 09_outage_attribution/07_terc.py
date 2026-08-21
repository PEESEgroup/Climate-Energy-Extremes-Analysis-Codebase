"""Figure 4b and 4c: does the same hazard cost more where the county is disadvantaged?

The question is asked inside the panel that produced Figure 3, so the comparison, the fixed
effects and the pre-event test are the same. For one county trait at a time, counties are split
into terciles, and each lag block of every hazard that passed the screen is interacted
with the worst and the best tercile, the middle being the reference. The interaction on the lead
block is the placebo: if the worst tercile already differs 14 to 8 days before the event, the
difference on the event day is not the storm.

One trait at a time, and not all six together, because the question is whether poor counties lose
more, not whether they lose more holding minority share fixed.
"""
import json, sys, time
import numpy as np, pandas as pd
from scipy import sparse as sps
T_START = time.time()
def log(*a): print("[%7.1fs]" % (time.time() - T_START), *a, flush=True)
exec(open("/data/attrib.py").read().split("# ================================================================= design matrix")[0])

# THE ESTIMATION SAMPLE. attrib.py drops county-days EAGLE-I never reported, because an absent row
# is a county that did not report and not a county with no outage; only 70% of the county-day grid
# is present. That mask is built as SAMP below attrib.py's design-matrix marker, so the exec above
# stops short of it and this script used to fit and cluster over all 6,418,440 cells, silently
# reading the missing 30% as zero outage. The docstring's claim that the comparison here is the
# same as Figure 3's was false on exactly this point. REP does come through the exec.
# The control buffer attrib.py applies is rebuilt here, because 07_terc.py assembles its own design
# from the same HAZ dict and would otherwise compare against the same contaminated shadow days.
CLEAN = 30
_near = np.zeros(N, bool)
_inwin = np.zeros(N, bool)
for _h in ORDER:
    _c, _d = HAZ[_h]["ci"].astype(np.int64), HAZ[_h]["di"].astype(np.int64)
    for _k in range(-CLEAN, CLEAN + 1):
        _j = _d + _k
        _g = (_j >= 0) & (_j < ND)
        _near[_c[_g] * ND + _j[_g]] = True
    for _k in range(-14, 15):
        _j = _d + _k
        _g = (_j >= 0) & (_j < ND)
        _inwin[_c[_g] * ND + _j[_g]] = True
SAMP = REP.reshape(-1) & ~(_near & ~_inwin)
log("estimation sample %s of %s county-days (%.1f%%); the rest are unreported and are dropped"
    % (format(int(SAMP.sum()), ","), format(N, ","), 100 * SAMP.mean()))

ORDER = ["tc", "convective", "cold", "heat", "fire"]
BLOCKS = [("lead", -14, -8), ("antic", -7, -1), ("impact", 0, 1), ("restore", 2, 6), ("tail", 7, 14)]
BINNED = {"impact", "restore"}
import os as _os_rp
for _rp in ("04_demand_model", "07_hazard_calendar", "09_outage_attribution",
            "02_downscale_wind", "12_figures"):
    _ap = _os_rp.path.abspath(_os_rp.path.join(
        _os_rp.path.dirname(_os_rp.path.abspath(__file__)), "..", _rp))
    if _os_rp.path.isdir(_ap) and _ap not in sys.path:
        sys.path.insert(0, _ap)
import hazsets                                            # noqa: E402
# WAS: IDH = ["tc", "convective"]. That was a private copy of the screen 04_chk3.py applies, and it
# stayed at two hazards after the screen moved, so this script kept interacting a hazard set the
# attribution no longer used. The set is now read from the screen and fails closed if it is absent.
IDH = hazsets.screened()
# Every block a hazard has must be interacted, not a subset. The anticipation block is
# large for hurricanes, and its interaction column is strongly correlated with the lead
# interaction column, because both scale with a county's number of storms. Leaving it out
# biases the lead coefficient, which is the placebo, and it does so only for the hazard
# whose anticipation is large. That is exactly the pattern the first run produced.
IBLK = ["lead", "antic", "impact", "restore", "tail"]

E = pd.read_csv("/data/equity_cost/analysis/equity_joined_v2.csv", dtype={"fips": str})
GF = pd.read_parquet("/data/equity_cost/gridphys/county_gridform.parquet")[["fips", "ug_share_dom"]]
E = E.merge(GF, on="fips", how="left")
E["dens"] = E.population / E.land_km2.replace(0, np.nan)
TR = [("median_age", "median age", "high"), ("poverty_rate", "poverty", "high"),
      ("minority_pct", "minority share", "high"), ("median_income", "income", "low"),
      ("dens", "rurality", "low"), ("ug_share_dom", "undergrounding", "low")]

# TRAIT SHARDING. The six traits are independent fits on one shared panel, so they parallelize
# perfectly in principle. What limits it is memory: one fit peaks near 44 GB on a 123 GB box, so
# two processes fit and six do not. Passing "k/n" runs traits k, k+n, k+2n ... and writes a shard
# file; 08_merge_terc.py combines them. The panel build ahead of this costs about 30 s and is repeated
# per shard, which is the whole overhead.
import sys as _sys
SHARD, NSHARD = 0, 1
for _a in _sys.argv[1:]:
    if "/" in _a and _a.replace("/", "").isdigit():
        SHARD, NSHARD = (int(x) for x in _a.split("/"))
if NSHARD > 1:
    TR = TR[SHARD::NSHARD]
    log("shard %d of %d: %d traits, %s" % (SHARD, NSHARD, len(TR), ", ".join(t[1] for t in TR)))

# ---- the base design, built once
base_cols, base_names = [], []
for h in ORDER:
    Hh = HAZ[h]
    for blk, a_, z_ in BLOCKS:
        for bi in range(4 if blk in BINNED else 1):
            sel = (Hh["b"] == bi) if blk in BINNED else np.ones(len(Hh["ci"]), bool)
            c0, d0 = Hh["ci"][sel], Hh["di"][sel]
            acc = np.zeros(N, np.float32)
            for lag in range(a_, z_ + 1):
                d = d0 + lag
                ok = (d >= 0) & (d < ND)
                acc += np.bincount(c0[ok].astype(np.int64) * ND + d[ok], minlength=N).astype(np.float32)
            base_cols.append(acc)
            base_names.append("%s|%s%s" % (h, blk, ("|%s" % Hh["edges"][bi]) if blk in BINNED else ""))
BASE = np.column_stack(base_cols).astype(np.float32); del base_cols
log("base design %s x %d" % (format(N, ","), BASE.shape[1]))

# ---- block totals per hazard, for the interactions
BT = {}
for h in IDH:
    Hh = HAZ[h]
    for blk, a_, z_ in BLOCKS:
        if blk not in IBLK:
            continue
        acc = np.zeros(N, np.float32)
        for lag in range(a_, z_ + 1):
            d = Hh["di"] + lag
            ok = (d >= 0) & (d < ND)
            acc += np.bincount(Hh["ci"][ok].astype(np.int64) * ND + d[ok], minlength=N).astype(np.float32)
        BT[(h, blk)] = acc
log("block totals built")

mo = days.month.values.astype(np.int64)
g_cm = (np.repeat(np.arange(NC, dtype=np.int64), ND) * 12 + np.tile(mo, NC) - 1).astype(np.int32)
g_d = np.tile(np.arange(ND, dtype=np.int32), NC)

# THE SAME DAY EFFECT FIGURE 3 ABSORBS. attrib.py absorbs county-month and REGION x day, the 18
# study subregions, and its own comment records why: a national day effect leaves the east-west
# weather contrast of the date in the residual, and a spatially sharp hazard then loads it onto its
# own pre-event block, giving fire a lead z of -5.07 and heat -3.39 where region x day gives clean
# pre-events. This script absorbed a national day effect while its docstring claimed the fixed
# effects matched Figure 3's, and it is the direction that manufactures the failing lead placebos
# this figure reports. g_rd is built here because attrib.py defines it below the marker this
# script's exec cuts at.
_SR = pd.read_csv("/data/datasets/grid/fips_to_subregion_mapping.csv", dtype={"FIPS": str})
_SR["FIPS"] = _SR.FIPS.str.zfill(5)
_r = _SR.set_index("FIPS").Subregion_Code.reindex(cf)
if _r.isna().any():
    raise SystemExit("%d of %d counties have no subregion; the region x day effect cannot be formed"
                     % (int(_r.isna().sum()), len(cf)))
_rid = pd.factorize(_r.values)[0].astype(np.int64)
g_rd = pd.factorize((np.repeat(_rid, ND) * ND
                     + np.tile(np.arange(ND, dtype=np.int64), NC)))[0].astype(np.int32)
log("fixed effects: %d county-month, %d region-day (%d subregions x %d days); clustering stays "
    "county and day" % (g_cm.max() + 1, g_rd.max() + 1, len(set(_rid)), ND))
g_c = np.repeat(np.arange(NC, dtype=np.int32), ND)
y = Y.reshape(-1)

def make_S(g, w):
    n = int(g.max()) + 1
    return sps.csr_matrix((w, (g, np.arange(len(g), dtype=np.int64))), shape=(n, len(g))), None

def demean(V, w, groups, sweeps=400, tol=1e-10):
    SS = []
    for g in groups:
        S, _ = make_S(g, w)
        SS.append((S, np.asarray(S.sum(1)).ravel()))
    def sweep(Aa):
        for g, (S, d_) in zip(groups, SS):
            Aa -= ((S @ Aa) / np.maximum(d_, 1e-300)[:, None])[g]
        return Aa
    sc = np.maximum(np.abs(V).max(0), 1e-12)
    for it in range(0, sweeps, 3):
        v0 = V.copy(); sweep(V); v1 = V.copy(); sweep(V); v2 = V
        d1, d2 = v1 - v0, v2 - v1
        dd = d2 - d1
        num = (dd * d2).sum(0); den = (dd * dd).sum(0)
        ok = den > 1e-300
        st = np.zeros_like(num); st[ok] = num[ok] / den[ok]
        V -= st * d2
        if float((np.abs(d2).max(0) / sc).max()) < tol:
            return it + 2
    return sweeps

def deviance(y_, mu_):
    """Summed over the reported county-days only, to match the weights the fit actually uses."""
    t = np.zeros_like(mu_); m = (y_ > 0) & SAMP
    t[m] = y_[m] * np.log(y_[m] / np.maximum(mu_[m], 1e-300))
    return 2.0 * float(((t - (y_ - mu_)) * SAMP).sum())

OUT = {}
for col, nm, bad_end in TR:
    v = E.set_index("fips")[col].reindex(cf).values.astype(float)
    ok = np.isfinite(v)
    q1, q2 = np.nanpercentile(v[ok], [100/3, 200/3])
    terc = np.full(NC, 1, np.int8)                       # 1 = middle = reference
    terc[ok & (v <= q1)] = 0
    terc[ok & (v > q2)] = 2
    worst = 0 if bad_end == "low" else 2
    best = 2 if bad_end == "low" else 0
    tw = (terc == worst).astype(np.float32)
    tb = (terc == best).astype(np.float32)
    cols, names = [], []
    for h in IDH:
        for blk in IBLK:
            for tag, ind in (("worst", tw), ("best", tb)):
                cols.append(BT[(h, blk)] * np.repeat(ind, ND))
                names.append("%s|%s|%s" % (h, blk, tag))
    X = np.hstack([BASE, np.column_stack(cols).astype(np.float32)])
    nm_all = base_names + names
    log("%s: terciles %s, design %d columns" % (nm, np.bincount(terc + 0, minlength=3).tolist(), X.shape[1]))
    beta = np.zeros(X.shape[1]); eta = np.log(y + 0.1); dev_prev = deviance(y, np.exp(eta))
    for it in range(20):
        mu = np.exp(eta); w = np.maximum(mu, 1e-10) * SAMP   # zero weight drops the unreported
        z = eta + (y - mu) / np.maximum(mu, 1e-10)
        V = np.empty((N, X.shape[1] + 1)); V[:, :-1] = X; V[:, -1] = z
        ns = demean(V, w, [g_cm, g_rd])
        Xd, zd = V[:, :-1], V[:, -1]
        A = Xd.T @ (w[:, None] * Xd); rhs = Xd.T @ (w * zd)
        A[np.diag_indices_from(A)] += 1e-10 * np.trace(A) / A.shape[0]
        nb = np.linalg.solve(A, rhs)
        stp = nb - beta; hh = 1.0
        for _ in range(14):
            bt = beta + hh * stp
            et = z - (zd - Xd @ bt)
            dv = deviance(y, np.exp(np.clip(et, -50, 25)))
            if dv <= dev_prev * (1 + 1e-12):
                break
            hh *= .5
        d = float(np.abs(bt - beta).max()); beta = bt; eta = np.clip(et, -50, 25); dev_prev = dv
        del V, Xd
        if d / max(1.0, float(np.abs(beta).max())) < 1e-4 and it >= 4:
            break
    mu = np.exp(eta); w = np.maximum(mu, 1e-10) * SAMP
    z = eta + (y - mu) / np.maximum(mu, 1e-10)
    V = np.empty((N, X.shape[1] + 1)); V[:, :-1] = X; V[:, -1] = z
    demean(V, w, [g_cm, g_rd]); Xd = V[:, :-1]
    A = Xd.T @ (w[:, None] * Xd); Ai = np.linalg.pinv(A)
    u = ((y - mu) * SAMP)[:, None] * Xd
    assert not np.any(u[~SAMP]), "an unreported county-day contributed to the covariance score"
    def meat(g):
        n = int(g.max()) + 1
        S = sps.csr_matrix((np.ones(len(g)), (g, np.arange(len(g), dtype=np.int64))), shape=(n, len(g)))
        G = S @ u
        return G.T @ G
    Vc = Ai @ (meat(g_c) + meat(g_d) - (u.T @ u)) @ Ai
    se = np.sqrt(np.maximum(np.diag(Vc), 0))
    R = {n_: dict(beta=float(b), se=float(s)) for n_, b, s in zip(nm_all, beta, se)}
    ix = {n_: k for k, n_ in enumerate(nm_all)}
    res = {}
    for h in IDH:
        for blk in IBLK:
            iw, ib = ix["%s|%s|worst" % (h, blk)], ix["%s|%s|best" % (h, blk)]
            g = beta[iw] - beta[ib]
            sg = float(np.sqrt(max(Vc[iw, iw] + Vc[ib, ib] - 2 * Vc[iw, ib], 0)))
            res["%s|%s" % (h, blk)] = dict(worst=float(beta[iw]), best=float(beta[ib]),
                                           gap=float(g), gap_se=sg,
                                           z=float(g / sg) if sg > 0 else np.nan)
    OUT[nm] = dict(col=col, results=res, n_worst=int(tw.sum()), n_best=int(tb.sum()))
    # Printed over IDH rather than over a fixed pair, so the log follows the screen.
    log("  %-15s impact gap: %s" % (nm, "   ".join(
        "%s %+.3f (z %+.2f)" % (h, res["%s|impact" % h]["gap"], res["%s|impact" % h]["z"])
        for h in IDH)))
    log("  %-15s lead placebo: %s" % ("", "   ".join(
        "%s %+.3f (z %+.2f)" % (h, res["%s|lead" % h]["gap"], res["%s|lead" % h]["z"])
        for h in IDH)))
    # a shard writes its own file so two processes never race on one path; 08_merge_terc.py joins them
    _OUTP = "/data/equity_cost/analysis/attrib/tercile_gaps_v2.json"
    if NSHARD > 1:
        _OUTP = _OUTP.replace(".json", "_s%d.json" % SHARD)
    json.dump(OUT, open(_OUTP, "w"), indent=1)
    log("wrote %s" % _OUTP)
log("wrote tercile_gaps.json")
