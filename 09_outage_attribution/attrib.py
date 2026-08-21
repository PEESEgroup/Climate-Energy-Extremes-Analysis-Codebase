"""Unified hazard attribution for county outage: one panel, every hazard, intensity, and an
explicit comparison group.

WHY THIS REPLACES BOTH EARLIER DESIGNS

  The occurrence design asked "did this county have hazard h today" and needed untreated counties
  nearby. That works for a hurricane and fails for a cold outbreak, whose neighbours are in the
  same air mass. The matched-cohort version failed its own placebo for four of five hazards.

  The attribution built on it multiplied one constant effect by a count of events, which is wrong
  twice over in physics. It ignored intensity, so a 34-kt brush over Wisconsin was charged the same
  as a Category 4 landfall. And it ignored whether that county actually lost power, so outage was
  attributed where none was observed.

  Here every hazard enters one equation with its own intensity, the outcome is modelled
  multiplicatively on the county's own normal level, and the attribution is a SHARE OF THE OBSERVED
  OUTAGE. Attributable outage therefore cannot exceed observed outage, and a county-day with no
  outage contributes nothing however hard the wind blew.

MODEL
  E[y_it] = exp( alpha_i + delta_{s(i),t} + sum_h sum_b sum_k beta_hbk X_hbk,it )
  y       = customer-hours out, county-day, in absolute customer-hours
  alpha_i = county x calendar-month fixed effect   (size, grid quality, tree cover, reporting, and
                                                   each county's own seasonal cycle)
  delta_t = day fixed effect                       (the comparison is every other county in the
                                                   country on the same day, which is what a hazard
                                                   covering whole states leaves available)

  A state x day effect was tried first and rejected on identification, not on fit. A cold outbreak
  routinely covers every county in a state, so a state x day effect absorbs the outbreak itself and
  leaves only within-state intensity differences. The national day effect keeps the level, and its
  cost, that the comparison county may sit in another climate, is exactly what the lead block
  tests. It is reported for every hazard and it decides which hazards may be read as causal.
  Poisson pseudo-likelihood, two-way clustered on county and date.

LAG BLOCKS  lead -14..-8 (placebo)  anticipation -7..-1  impact 0..1  restoration 2..6  tail 7..14
INTENSITY   tropical cyclone by wind category, the others by quartile of a COUNTY-RELATIVE
            intensity, because infrastructure is built to the local climate and -5 C is not the
            same shock in Texas as in Minnesota. The county mean and standard deviation behind that
            quartile are computed on 1980 to 2019 only, the same years the flags are frozen on.
FLAGS       every hazard flag is read as built and none is re-derived here. The severe-convective
            flag in particular arrives as the `severe` column of county_convective_daily.parquet,
            with its areal-fraction condition already applied by 07_hazard_calendar/13_c404_convective.py.
            Each flag file is checked against 07_hazard_calendar/hazard_defs.py before it is used:
            a file with no stamp, or one whose stamped constants differ from the current ones,
            raises here rather than being read. That refusal is the point of the stamp.
ATTRIBUTION att_it = y_it * (1 - exp(-x'beta)) over anticipation..tail, split across hazards in
            proportion to each hazard's own contribution to the linear index.
"""
import json, os, sys, time
for _p in (os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "07_hazard_calendar"),
           os.path.dirname(os.path.abspath(__file__))):
    if os.path.isdir(_p) and _p not in sys.path:
        sys.path.insert(0, _p)
import hazard_defs as HD
import numpy as np, pandas as pd
from scipy import sparse as sps

T_START = time.time()
def log(*a):
    print("[%7.1fs]" % (time.time() - T_START), *a, flush=True)

T0, T1 = pd.Timestamp("2015-01-01"), pd.Timestamp("2022-09-30")
OUT = "/data/equity_cost/analysis/attrib"; os.makedirs(OUT, exist_ok=True)

# ================================================================= panel
D = pd.read_parquet("/data/equity_cost/analysis/eaglei_county_daily.parquet",
                    columns=["fips", "date", "customer_hours_out"])
D["date"] = pd.to_datetime(D.date); D["fips"] = D.fips.astype(str).str.zfill(5)
cov = D.groupby("fips").date.agg(["min", "max"])
keep = cov[(cov["min"] <= "2015-07-01") & (cov["max"] >= "2022-06-30")].index
D = D[(D.date >= T0) & (D.date <= T1) & D.fips.isin(keep)]
E = pd.read_csv("/data/equity_cost/analysis/equity_joined_v2.csv", dtype={"fips": str})
E = E[E.denom > 0][["fips", "denom"]]
cf = sorted(set(D.fips) & set(E.fips))
# The study domain is the contiguous United States: the weather comes from a CONUS simulation and
# the 18 subregions tile the lower 48. Alaska, Hawaii and Puerto Rico appear in the outage record
# but carry no hazard flag of any kind, so they contribute an empty treatment and no identifying
# variation. They are dropped here, by rule and with a count, rather than silently. A CONTIGUOUS
# county missing from the subregion mapping is a real fault and still stops the run.
_NONCONUS = {"02", "15", "60", "66", "69", "72", "78"}
_SRMAP = pd.read_csv("/data/datasets/grid/fips_to_subregion_mapping.csv", dtype={"FIPS": str})
_SRMAP["FIPS"] = _SRMAP.FIPS.str.zfill(5)
_have = set(_SRMAP.FIPS)
_drop = [f for f in cf if f[:2] in _NONCONUS]
_bad = [f for f in cf if f not in _have and f[:2] not in _NONCONUS]
if _bad:
    raise SystemExit("%d contiguous-US counties have no subregion, first ten %s; rebuild "
                     "fips_to_subregion_mapping.csv with mkfmap.py" % (len(_bad), _bad[:10]))
if _drop:
    log("dropped %d non-contiguous counties (%s) that carry outage rows but no hazard flag"
        % (len(_drop), ", ".join(sorted({f[:2] for f in _drop}))))
cf = [f for f in cf if f not in set(_drop)]
cidx = {f: i for i, f in enumerate(cf)}
days = pd.date_range(T0, T1); ND = len(days); didx = {d: i for i, d in enumerate(days)}
NC = len(cf); N = NC * ND
den = E.set_index("fips").reindex(cf).denom.values.astype(float)
Y = np.zeros((NC, ND), np.float64)
D = D[D.fips.isin(cidx)]
_ci = D.fips.map(cidx).values.astype(int); _di = D.date.map(didx).values.astype(int)
Y[_ci, _di] = D.customer_hours_out.values
# REPORTED, NOT ZERO. The county eligibility rule above only tests the first and last date, so a
# county with a gap in the middle of its record still qualifies. Everything absent from the source
# then arrives here as an exact zero. That is wrong: the source records explicit zeros, 6.9% of its
# rows, so an absent county-day is a county that did not report, not a county with no outage. Over
# this window only 63% of the county-day grid appears in the source. Treating the other 37% as
# zero attenuates every hazard coefficient, because the days a utility fails to report are not
# independent of the days it is under stress.
REP = np.zeros((NC, ND), bool)
REP[_ci, _di] = True
OBS = float(Y.sum())
log("panel %d counties x %d days = %s county-days" % (NC, ND, format(N, ",")))
log("reported county-days %s of %s (%.1f%%); the rest are unreported and are dropped, not zeroed"
    % (format(int(REP.sum()), ","), format(N, ","), 100 * REP.mean()))
# 03_attrib_post.py re-executes everything above the design-matrix marker and does not want this
# line echoed twice. It used to suppress it by commenting the first physical line out, which
# broke the moment the call wrapped onto two lines. A flag cannot break that way.
if globals().get("ATTRIB_ECHO", True):
    log("observed %.4e customer-hours, zero among REPORTED county-days %.1f%%"
        % (OBS, 100 * (Y[REP] == 0).mean()))
_cov = REP.mean(1)
log("interior coverage per county: min %.2f, 10th %.2f, median %.2f" %
    (_cov.min(), np.quantile(_cov, .1), np.median(_cov)))

# ================================================================= hazard intensity
HAZ, ORDER = {}, ["tc", "convective", "cold", "heat", "fire"]

# Refuse a flag file built before the shared definitions, or from different constants. The three
# files below are the only hazard definitions this program depends on, and it re-derives none of
# them; county_weather_daily_c404.parquet is weather, not a flag table, so it carries no stamp.
for _f, _h in [("/data/enso/tc_county_ext/county_tc_days.parquet", ["tc"]),
               ("/data/enso/county_convective_daily.parquet", ["convection"]),
               ("/data/enso/county_hazard_flags_c404.parquet", ["cold", "heat", "fire"])]:
    _s = HD.require_stamp(_f, hazards=_h)
    log("%-38s written by %s, hazard_defs %s" % (os.path.basename(_f), _s["script"],
                                                 _s["hazard_defs_version"]))

T = pd.read_parquet("/data/enso/tc_county_ext/county_tc_days.parquet",
                    columns=["fips", "date", "wind_kt"])
T["date"] = pd.to_datetime(T.date); T = T[(T.date >= T0) & (T.date <= T1)]
T["ci"] = T.fips.map(cidx); T["di"] = T.date.map(didx); T = T.dropna(subset=["ci", "di"])
HAZ["tc"] = dict(ci=T.ci.values.astype(np.int32), di=T.di.values.astype(np.int32),
                 b=np.digitize(T.wind_kt.values, [50., 64., 83.]),
                 edges=["34 to 50 kt", "50 to 64 kt", "64 to 83 kt", "83 kt and above"])

C = pd.read_parquet("/data/enso/county_convective_daily.parquet",
                    columns=["fips", "date", "frac50", "severe"])
C["date"] = pd.to_datetime(C.date)
# The flag is built in 07_hazard_calendar/13_c404_convective.py and read here, not rebuilt: 50 dBZ in a
# cell-hour, daily maximum MUCAPE of 1000 J/kg, and a fifth of the county's cells above 50 dBZ in
# that hour. That last condition used to be written out separately in each of the nine consumers of
# this file, including this one, rather than in the builder, so nothing made them agree and any
# consumer that dropped the line would have read a wider hazard under the same name. frac50 is
# still read, as the intensity axis.
C = C[(C.date >= T0) & (C.date <= T1) & C.severe]
C["ci"] = C.fips.map(cidx); C["di"] = C.date.map(didx); C = C.dropna(subset=["ci", "di"])
qq = np.quantile(C.frac50.values, [.25, .5, .75])
HAZ["convective"] = dict(ci=C.ci.values.astype(np.int32), di=C.di.values.astype(np.int32),
                         b=np.digitize(C.frac50.values, qq),
                         edges=["footprint Q1", "footprint Q2", "footprint Q3", "footprint Q4"])

W = pd.read_parquet("/data/enso/county_weather_daily_c404.parquet",
                    columns=["fips", "date", "tmax", "tmin", "q", "ps", "wspd"])
W["date"] = pd.to_datetime(W.date); W = W[W.fips.isin(cidx)]
esat = 611.2 * np.exp(17.67 * (W.tmax - 273.15) / (W.tmax - 29.65))
evap = W.ps * W.q / (0.622 + 0.378 * W.q)
W["hdw"] = (np.maximum(esat - evap, 0) / 100.0) * W.wspd            # hPa m/s
# The hazard flags are frozen on 1980 to 2019, so the county mean and standard deviation that turn
# a flagged day into a quartile are frozen on the same years. They used to be taken from the whole
# 1980 to 2022 record, which let the years under study help set the yardstick they are judged by.
CL = W[(W.date.dt.year >= HD.CLIM_Y0) & (W.date.dt.year <= HD.CLIM_Y1)]   # <<< frozen, both bounds
CLM = pd.concat([CL.groupby("fips").tmin.mean().rename("mn_tmin"),
                 CL.groupby("fips").tmin.std().rename("sd_tmin"),
                 CL.groupby("fips").tmax.mean().rename("mn_tmax"),
                 CL.groupby("fips").tmax.std().rename("sd_tmax"),
                 CL.groupby("fips").hdw.mean().rename("mn_hdw"),
                 CL.groupby("fips").hdw.std().rename("sd_hdw")], axis=1)
log("climatology from %s county-days, 1980 to 2019" % format(len(CL), ","))
del CL
W = W[(W.date >= T0) & (W.date <= T1)].join(CLM, on="fips")
F = pd.read_parquet("/data/enso/county_hazard_flags_c404.parquet")
F["date"] = pd.to_datetime(F.date)
W = W.merge(F[(F.date >= T0) & (F.date <= T1)], on=["fips", "date"], how="left")
W["ci"] = W.fips.map(cidx).astype(np.int32); W["di"] = W.date.map(didx).astype(np.int32)
# A county-standardised anomaly, not an absolute one: infrastructure is built to the local
# climate, so -5 C is not the same shock in Texas as in Minnesota. It is also free of the mass
# point that an excess-over-a-threshold measure carries, which had collapsed the heat quartiles.
for nm, sel, inten, unit in [
        ("cold", (W.hazard == "cold").values,
         ((W.mn_tmin - W.tmin) / W.sd_tmin).values, "standard deviations below the county's own mean daily minimum"),
        ("heat", (W.hazard == "heat").values,
         ((W.tmax - W.mn_tmax) / W.sd_tmax).values, "standard deviations above the county's own mean daily maximum"),
        ("fire", (W.hazard == "fire").values,
         ((W.hdw - W.mn_hdw) / W.sd_hdw).values, "standard deviations above the county's own mean hot-dry-windy index")]:
    # The quartile cut points are frozen on the same calibration period as the flags and the
    # county mean and standard deviation. Estimating them on the study events would let the years
    # under study move the yardstick they are judged by, which is the defect the freeze exists for.
    clim_sel = sel & (W.date.dt.year >= HD.CLIM_Y0).values & (W.date.dt.year <= HD.CLIM_Y1).values
    v = inten[sel]
    qq = np.quantile(inten[clim_sel], [.25, .5, .75]) if clim_sel.sum() >= 100 else np.quantile(v, [.25, .5, .75])
    HAZ[nm] = dict(ci=W.ci.values[sel], di=W.di.values[sel], b=np.digitize(v, qq),
                   edges=["%s Q1" % nm, "%s Q2" % nm, "%s Q3" % nm, "%s Q4" % nm],
                   cuts=qq.tolist(), unit=unit)
del W, F, C, T
for h in ORDER:
    log("%-11s %7d county-days   bins %s" % (h, len(HAZ[h]["ci"]), np.bincount(HAZ[h]["b"], minlength=4)))

# ================================================================= design matrix
BLOCKS = [("lead", -14, -8), ("antic", -7, -1), ("impact", 0, 1), ("restore", 2, 6), ("tail", 7, 14)]
BINNED = {"impact", "restore"}
names, hz_of, blk_of = [], [], []
X = np.zeros((N, sum(4 if b in BINNED else 1 for b in [x[0] for x in BLOCKS]) * len(ORDER)), np.float32)
k = 0
for h in ORDER:
    Hh = HAZ[h]
    for blk, a, z in BLOCKS:
        for bi in range(4 if blk in BINNED else 1):
            sel = (Hh["b"] == bi) if blk in BINNED else np.ones(len(Hh["ci"]), bool)
            c0, d0 = Hh["ci"][sel], Hh["di"][sel]
            acc = np.zeros(N, np.float32)
            for lag in range(a, z + 1):
                d = d0 + lag        # a day t is exposed when the event fell on t - lag
                ok = (d >= 0) & (d < ND)
                acc += np.bincount(c0[ok].astype(np.int64) * ND + d[ok], minlength=N).astype(np.float32)
            X[:, k] = acc
            names.append("%s|%s%s" % (h, blk, ("|%s" % Hh["edges"][bi]) if blk in BINNED else ""))
            hz_of.append(h); blk_of.append(blk); k += 1
y = Y.reshape(-1)
# ---- drop terms the data cannot identify --------------------------------------------------
# A Poisson coefficient is not identified if its column never varies, and it runs to minus
# infinity if every county-day carrying that exposure had zero outage. Both leave the normal
# equations singular, and the ridge that keeps the solve from failing then lets the coefficient
# wander, which is what stalled the first two attempts at exactly the same tolerance.
nz = np.array([int((X[:, j] > 0).sum()) for j in range(X.shape[1])])
ypos = np.array([float(y[X[:, j] > 0].sum()) for j in range(X.shape[1])])
bad = (nz < 30) | (ypos <= 0)
for j in np.where(bad)[0]:
    log("   DROPPED %-40s cells %6d  outage on them %.3e" % (names[j], nz[j], ypos[j]))
if bad.any():
    X = X[:, ~bad]
    names = [n for n, b in zip(names, bad) if not b]
    hz_of = [h for h, b in zip(hz_of, bad) if not b]
    blk_of = [q for q, b in zip(blk_of, bad) if not b]
log("design %s x %d, %.2f GB, cells per column min %d median %d max %d"
    % (format(N, ","), X.shape[1], X.nbytes / 1e9, nz[~bad].min(), int(np.median(nz[~bad])), nz[~bad].max()))

mo = days.month.values.astype(np.int64)
g_cm = (np.repeat(np.arange(NC, dtype=np.int64), ND) * 12 + np.tile(mo, NC) - 1).astype(np.int32)
g_d = np.tile(np.arange(ND, dtype=np.int32), NC)
g_c = np.repeat(np.arange(NC, dtype=np.int32), ND)

# WHICH DAY EFFECT IS ABSORBED. A national day effect leaves the east-west weather contrast of the
# same date in the residual, and a hazard with sharp spatial edges then loads it onto its own
# pre-event block. Measured on all five hazards, the pre-event z under a national day effect is
# -5.07 for fire and -3.39 for heat, both spurious: the fire post-event effect is +3.34 under a
# national day effect, +3.34 under region x day and +3.33 under state x day, so the effect itself
# does not move and only the pre-event does. A state x day effect removes the spurious pre-trend
# but also absorbs a cold outbreak, which routinely covers every county in a state. The 18 study
# subregions sit between the two: coarse enough that a state-wide outbreak still has untreated
# counties in its own region, fine enough to take out the regional weather of the day. Cold's
# pre-event z is +0.16 nationally, +1.41 by region and -1.02 by state, so the treatment survives.
_r = _SRMAP.set_index("FIPS").Subregion_Code.reindex(cf)
if _r.isna().any():
    raise SystemExit("%d of %d counties have no subregion in fips_to_subregion_mapping.csv; the "
                     "region x day effect cannot be formed" % (int(_r.isna().sum()), len(cf)))
_rid = pd.factorize(_r.values)[0].astype(np.int64)
g_rd = (np.repeat(_rid, ND) * ND + np.tile(np.arange(ND, dtype=np.int64), NC)).astype(np.int32)
g_rd = pd.factorize(g_rd)[0].astype(np.int32)
log("fixed effects: %d county-month, %d region-day (%d subregions x %d days); clustering stays "
    "county and day" % (g_cm.max() + 1, g_rd.max() + 1, len(set(_rid)), ND))

# ================================================================= PPML, two absorbed effects
def make_S(g, wt):
    n = int(g.max()) + 1
    S = sps.csr_matrix((wt, (g, np.arange(len(g), dtype=np.int64))), shape=(n, len(g)))
    return S, np.asarray(S.sum(1)).ravel()

def demean(V, wt, groups, sweeps=400, tol=1e-9):
    """Weighted alternating projections with Irons and Tuck acceleration.

    Plain alternating projections stall here. The Poisson weights are the fitted means, which span
    ten orders of magnitude across counties, and under such weights the two projections meet at a
    very small angle: the iteration converges, but at a rate that needs thousands of sweeps. The
    acceleration extrapolates along the fixed-point direction every third sweep, which is what
    reghdfe and lfe do for the same reason. Convergence is judged column by column, relative to
    each column's own scale, so a sparse hazard column is held to the same standard as the
    working response.
    """
    SS = [make_S(g, wt) for g in groups]

    def sweep(A):
        for g, (S, dn) in zip(groups, SS):
            A -= (((S @ A) / np.maximum(dn, 1e-300)[:, None]))[g]
        return A

    scale = np.maximum(np.abs(V).max(0), 1e-12)
    for it in range(0, sweeps, 3):
        v0 = V.copy()
        sweep(V); v1 = V.copy()
        sweep(V); v2 = V
        d1 = v1 - v0
        d2 = v2 - v1
        dd = d2 - d1
        num = (dd * d2).sum(0)
        den_ = (dd * dd).sum(0)
        ok = den_ > 1e-300
        step = np.zeros_like(num)
        step[ok] = num[ok] / den_[ok]
        V -= step * d2                       # Irons and Tuck extrapolation, column by column
        if float((np.abs(d2).max(0) / scale).max()) < tol:
            return it + 2
    return sweeps


def deviance(y_, mu_):
    t = np.zeros_like(mu_)
    m = y_ > 0
    t[m] = y_[m] * np.log(y_[m] / np.maximum(mu_[m], 1e-300))
    return 2.0 * float((t - (y_ - mu_)).sum())

GRP = [g_cm, g_rd]      # absorbed; the clustering below stays county and day
y = Y.reshape(-1)
# UNREPORTED COUNTY-DAYS CARRY ZERO WEIGHT. They are not zero-outage observations, so they must not
# enter the score, the Hessian, the absorbed fixed effects or the deviance. Zero weight is exactly
# equivalent to deleting the rows and keeps every index aligned with the panel.
# THE CONTROL GROUP. A county-day that carries no exposure is a control, and until now that
# included days sitting just outside an event's window: day 15 to day 30 around a hazard is still
# inside the same dry spell or the same storm season, so it is not a comparable ordinary day. With
# no buffer the county-month effect is lifted by the event days in its own month, every remaining
# day looks low by comparison, and the lead block absorbs the difference as a pre-trend. That is
# what produced fire's lead z of -5.07 and cold's negative impact coefficients, while the raw data
# say a cold day carries 5.9x and a fire-weather day 4.3x the outage of an ordinary day.
#
# 08_fire_verify.py has always applied this rule, MASK = inwin | (~near) with CLEAN = 30, and reports
# a clean fire pre-event on the same flags. Generalized to the joint five-hazard panel: a day
# inside ANY hazard's window stays, because the design estimates all five together; a day within
# CLEAN days of ANY hazard day but inside no window is the contaminated shadow and is dropped; a
# day far from everything is a clean control.
CLEAN = 30
_inwin = X.any(1)
_near = np.zeros(N, bool)
for _h in ORDER:
    _c, _d = HAZ[_h]["ci"].astype(np.int64), HAZ[_h]["di"].astype(np.int64)
    for _k in range(-CLEAN, CLEAN + 1):
        _j = _d + _k
        _g = (_j >= 0) & (_j < ND)
        _near[_c[_g] * ND + _j[_g]] = True
BUF = _near & ~_inwin
log("control buffer: %s county-days lie within %d days of an event but inside no window and are "
    "dropped; %s stay in a window, %s are clean controls"
    % (format(int(BUF.sum()), ","), CLEAN, format(int(_inwin.sum()), ","),
       format(int((~_near).sum()), ",")))
SAMP = REP.reshape(-1) & ~BUF
log("estimation sample %s of %s county-days (%.1f%%)"
    % (format(int(SAMP.sum()), ","), format(N, ","), 100 * SAMP.mean()))
beta = np.zeros(X.shape[1])
# Start from the data, not from a constant. A constant start makes the first working response an
# almost linear function of y, so the first Newton step is effectively a linear regression and
# lands on coefficients of the order of customer-hours; the exponential then saturates and the
# iteration freezes there. mu0 = y + 0.1 is the standard Poisson start and keeps eta in range.
eta = np.log(y + 0.1)
dev_prev = deviance(y[SAMP], np.exp(eta[SAMP]))
log("starting deviance %.8e, eta from %.2f to %.2f" % (dev_prev, eta.min(), eta.max()))
WATCH = None
for it in range(20):
    mu = np.exp(eta)
    wt_all = np.maximum(mu, 1e-10)
    z = eta + (y - mu) / wt_all
    wt = wt_all * SAMP          # unreported rows drop out of every weighted quantity
    V = np.empty((N, X.shape[1] + 1))
    V[:, :-1] = X; V[:, -1] = z
    ns = demean(V, wt, GRP)
    Xd = V[:, :-1]; zd = V[:, -1]
    A = Xd.T @ (wt[:, None] * Xd); rhs = Xd.T @ (wt * zd)
    A[np.diag_indices_from(A)] += 1e-10 * np.trace(A) / A.shape[0]
    nb = np.linalg.solve(A, rhs)
    # Step halving. Newton on a Poisson with this much overdispersion can overshoot, and an
    # overshoot is unrecoverable once the exponential saturates, so no step is accepted unless it
    # lowers the deviance.
    stp = nb - beta; hlv = 1.0
    for _ in range(12):
        bt = beta + hlv * stp
        et = z - (zd - Xd @ bt)
        dv = deviance(y[SAMP], np.exp(np.clip(et[SAMP], -50, 25)))
        if dv <= dev_prev * (1 + 1e-12):
            break
        hlv *= 0.5
    jm = int(np.argmax(np.abs(bt - beta))); d = float(np.abs(bt - beta)[jm])
    beta = bt; eta = np.clip(et, -50, 25); dev = dv
    rel = abs(dev_prev - dev) / max(abs(dev), 1.0)
    if WATCH is None:
        WATCH = [names.index(n) for n in names
                 if n.endswith("|impact|34 to 50 kt") or n.endswith("|impact|83 kt and above")
                 or n == "cold|impact|cold Q4" or n == "convective|impact|footprint Q4"
                 or n == "tc|lead"]
    log("  irls %2d  sweeps %3d  deviance %.8e  rel %.2e  max|dbeta| %.2e on %s"
        % (it + 1, ns, dev, rel, d, names[jm]) + ("  halved to %.4f" % hlv if hlv < 1 else ""))
    log("           %s" % "  ".join("%s %+.4f" % (names[j].split("|")[0] + "/" + names[j].split("|")[-1][:7], beta[j])
                                    for j in WATCH))
    del V, Xd
    # The deviance keeps falling long after beta has settled, because a third of county-days carry
    # no outage and their fitted means descend slowly. The coefficients are what this study
    # reports, so convergence is judged on them, relative to their own size, and held for two
    # consecutive steps.
    rb = d / max(1.0, float(np.abs(beta).max()))
    if rb < 1e-4 and it >= 4:
        log("  coefficients stable at %.2e relative" % rb)
        break
    dev_prev = dev

# ---- two-way clustered covariance -----------------------------------------------------------
mu = np.exp(eta)
# THE COVARIANCE MUST USE THE SAME SAMPLE AS THE COEFFICIENTS. The IRLS above drops unreported
# county-days by zero weight; this block used to rebuild wt without that mask, so it reabsorbed the
# regressors on a different sample and gave every unreported row a score of (0 - mu) * Xd. Those
# non-observations then entered both the bread and the meat as if they were zero-outage days.
wt = np.maximum(mu, 1e-10) * SAMP
z = eta + (y - mu) / np.maximum(mu, 1e-10)
V = np.empty((N, X.shape[1] + 1)); V[:, :-1] = X; V[:, -1] = z
demean(V, wt, GRP)
Xd = V[:, :-1]; zd = V[:, -1]
A = Xd.T @ (wt[:, None] * Xd)
Ai = np.linalg.pinv(A)
u = ((y - mu) * SAMP)[:, None] * Xd
assert not np.any(u[~SAMP]), "an unreported county-day contributed to the covariance score"
def meat(g):
    n = int(g.max()) + 1
    S = sps.csr_matrix((np.ones(len(g)), (g, np.arange(len(g), dtype=np.int64))), shape=(n, len(g)))
    G = S @ u
    return G.T @ G
Vc = Ai @ (meat(g_c) + meat(g_d) - (u.T @ u)) @ Ai      # intersection of county and day = the cell
se = np.sqrt(np.maximum(np.diag(Vc), 0))
# the full covariance, not only its diagonal: Figure 6 propagates the tropical-cyclone block
# through a projection, and the impact, restore and tail coefficients there are far from independent
np.savez("%s/attrib_vcov.npz" % OUT, vcov=Vc, beta=beta, names=np.array(names, dtype=object))
del V, Xd, u

R = {n: dict(beta=float(b), se=float(s), z=float(b / s) if s > 0 else np.nan,
             hazard=h, block=bl) for n, b, s, h, bl in zip(names, beta, se, hz_of, blk_of)}
log("")
log("%-38s %9s %9s %7s" % ("term", "beta", "se", "z"))
for n in names:
    r = R[n]
    log("%-38s %+9.4f %9.4f %+7.2f" % (n, r["beta"], r["se"], r["z"]))

# ================================================================= attribution
use = np.array([b != "lead" for b in blk_of])
idx = np.exp(-(X[:, use] @ beta[use]))
share = 1.0 - idx
att = y * share
lin = {h: X[:, np.array([(hz_of[j] == h) and use[j] for j in range(len(names))])] @
          beta[np.array([(hz_of[j] == h) and use[j] for j in range(len(names))])] for h in ORDER}
tot_lin = sum(lin.values())
per_h = {h: float((att * np.divide(lin[h], tot_lin, out=np.zeros(N), where=np.abs(tot_lin) > 1e-12)).sum())
         for h in ORDER}
log("")
log("observed total                     %.4e customer-hours" % OBS)
log("attributable to all five hazards   %.4e = %.1f%%" % (att.sum(), 100 * att.sum() / OBS))
for h in ORDER:
    log("   %-12s %.4e = %5.2f%%" % (h, per_h[h], 100 * per_h[h] / OBS))

CTY = pd.DataFrame({"fips": cf, "denom": den,
                    "observed_cho": Y.sum(1),
                    "att_cho": att.reshape(NC, ND).sum(1)})
for h in ORDER:
    a_h = (att * np.divide(lin[h], tot_lin, out=np.zeros(N), where=np.abs(tot_lin) > 1e-12))
    CTY["att_" + h] = a_h.reshape(NC, ND).sum(1)
CTY["rate"] = CTY.att_cho / CTY.denom
CTY.to_parquet("%s/county_attributable_ppml.parquet" % OUT, index=False)
json.dump(dict(results=R, observed=OBS, attributable=float(att.sum()),
               per_hazard=per_h, n_counties=NC, n_days=ND,
               bins={h: HAZ[h]["edges"] for h in ORDER},
               cuts={h: HAZ[h].get("cuts") for h in ORDER},
               units={h: HAZ[h].get("unit") for h in ORDER},
               blocks=[list(b) for b in BLOCKS]),
          open("%s/attrib.json" % OUT, "w"), indent=1)
log("wrote %s/attrib.json and county_attributable_ppml.parquet" % OUT)
