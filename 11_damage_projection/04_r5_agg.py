"""
R5 damage arm - county-day aggregation of TGW 2D + precipitation.

One row per county-day per run. Identical code on TGW-historical and on all four future runs, so
that the model bias divides out when the future/historical ratio is taken (R5_DAMAGE_PLAN 3.2).

County reduction is MAX over the county's ~18 TGW cells, matching the convective-flag convention
in R3_METHODS 1.3 - a mean over cells would erase a passing storm swath. Wind speed is formed
PER CELL and only then reduced, so the "speed of the mean vector" bias R3_METHODS 1.3 warns about
for the subregion builds does not arise here.

usage: 04_r5_agg.py <scenario> <shard> <nshards> <y0> <y1>
"""
import os, sys, glob, bisect, time
import numpy as np, pandas as pd

SCEN, SHARD, NSH, Y0, Y1 = sys.argv[1], int(sys.argv[2]), int(sys.argv[3]), int(sys.argv[4]), int(sys.argv[5])
OUT = "/data/scratch_r5/county_agg"
os.makedirs(OUT, exist_ok=True)

KT = 0.514444
THR_W = [34 * KT, 50 * KT, 64 * KT]          # 17.49, 25.72, 32.92 m/s
THR_P = [10.0, 20.0]                          # mm/hr

# ------------------------------------------------------------------ county mask (TGW native grid)
cm = np.load("/data/loads_measured/county_mask_tgw.npz", allow_pickle=True)
FIPS = np.array([str(f).zfill(5) for f in cm["fips"]])
NC = len(FIPS)
cell = cm["pair_cell"].astype(np.int64)
cty = cm["pair_fips"].astype(np.int64)
o = np.argsort(cty, kind="stable")
cell, cty = cell[o], cty[o]
assert (np.diff(cty) >= 0).all()
cnt = np.bincount(cty, minlength=NC).astype("f4")
assert (cnt > 0).all(), "a county has no cells - reduceat would misalign"
STARTS = np.searchsorted(cty, np.arange(NC), "left")


# ------------------------------------------------------------------ weekly-file indices
def index_dir(d, pat, split):
    fs = sorted(glob.glob(os.path.join(d, pat)))
    ks = [pd.Timestamp(os.path.basename(f).split(split)[-1].split("_")[0]) for f in fs]
    return ks, fs


if SCEN == "historical":
    K2, F2 = index_dir("/data/tgw_hist", "tgw_historical_*hourly_*.npz", "_hourly_")
    KP, FP = index_dir("/data/tgw_precip/historical", "tgw_pr_*hourly_*.npz", "_hourly_")
else:
    K2, F2 = index_dir("/data/tgw_extract/%s" % SCEN, "tgw_%s_*hourly_*.npz" % SCEN, "_hourly_")
    KP, FP = index_dir("/data/tgw_precip/%s" % SCEN, "tgw_pr_*hourly_*.npz", "_hourly_")

_c = {}


def get(kind, keys, files, ts):
    i = bisect.bisect_right(keys, ts) - 1
    if i < 0:
        return None
    f = files[i]
    if _c.get(kind, (None,))[0] != f:
        z = np.load(f)
        tix = {str(t): j for j, t in enumerate(z["times"])}
        _c[kind] = (f, z["data"], tix, z["scale"].astype("f4") if "scale" in z.files else None)
    return _c[kind]


def thetae(T, q, p):
    """Bolton (1980) equivalent potential temperature. T[K], q[kg/kg], p[Pa]."""
    q = np.clip(q, 1e-8, 0.05)
    r = q / (1.0 - q)
    e = np.maximum(q * p / (0.622 + 0.378 * q), 1.0)          # Pa
    Tl = 2840.0 / (3.5 * np.log(T) - np.log(e / 100.0) - 4.805) + 55.0
    return T * (100000.0 / p) ** (0.2854 * (1 - 0.28 * r)) * \
        np.exp((3376.0 / Tl - 2.54) * r * (1 + 0.81 * r))


def cmax(A):    # (nh, ncells_sorted) -> (ncty,)
    return np.maximum.reduceat(A, STARTS, axis=1).max(0)


def cmin(A):
    return np.minimum.reduceat(A, STARTS, axis=1).min(0)


def cfracmax(hit):
    return (np.add.reduceat(hit.astype("f4"), STARTS, axis=1) / cnt).max(0)


days = pd.date_range("%d-01-01" % Y0, "%d-12-31" % Y1)
days = days[(days >= K2[0]) & (days <= K2[-1] + pd.Timedelta("6D"))]
blk = np.array_split(np.arange(len(days)), NSH)[SHARD]
days = days[blk]
print("%s shard %d/%d : %d days %s..%s" % (SCEN, SHARD, NSH, len(days), days[0].date(), days[-1].date()), flush=True)

rows, miss = [], 0
t0 = time.time()
for n, d in enumerate(days):
    g2 = get("2d", K2, F2, d)
    gp = get("pr", KP, FP, d)
    if g2 is None:
        miss += 1
        continue
    _, D2, T2i, SC = g2
    hh = ["%s%02d" % (d.strftime("%Y%m%d"), h) for h in range(1, 24)] + \
         [(d + pd.Timedelta("1D")).strftime("%Y%m%d") + "00"]
    j2 = [T2i[k] for k in hh if k in T2i]
    if len(j2) < 20:
        miss += 1
        continue
    A = D2[np.array(j2)].astype("f4")                     # (nh,7,299,424)
    A = A.reshape(A.shape[0], A.shape[1], -1)[:, :, cell]  # (nh,7,ncells)
    U, V, Q, P, T = A[:, 0], A[:, 1], A[:, 2], A[:, 3] / SC[3], A[:, 4]
    W = np.sqrt(U * U + V * V)
    r = dict(fips=FIPS, date=d,
             wmax=cmax(W), psfc_min=cmin(P), t2max=cmax(T), t2min=cmin(T), q2max=cmax(Q),
             thetae_max=cmax(thetae(T, Q, P)))
    for thr, nm in zip(THR_W, ("wfrac34", "wfrac50", "wfrac64")):
        r[nm] = cfracmax(W >= thr)
    if gp is not None:
        _, DP, TPi, _ = gp
        jp = [TPi[k] for k in hh if k in TPi]
        if len(jp) >= 20:
            PR = DP[np.array(jp), 0].astype("f4").reshape(len(jp), -1)[:, cell]
            r["pr_max"] = cmax(PR)
            r["pr_sum"] = np.add.reduceat(PR.sum(0), STARTS) / cnt
            for thr, nm in zip(THR_P, ("pr_frac10", "pr_frac20")):
                r[nm] = cfracmax(PR >= thr)
    for k in ("pr_max", "pr_sum", "pr_frac10", "pr_frac20"):
        r.setdefault(k, np.full(NC, np.nan, "f4"))
    rows.append(pd.DataFrame(r))
    if n % 400 == 0:
        el = time.time() - t0
        print("  %s s%d %d/%d  %.2f s/day  eta %.0f min"
              % (SCEN, SHARD, n, len(days), el / max(n, 1), (len(days) - n) * el / max(n, 1) / 60), flush=True)

R = pd.concat(rows, ignore_index=True)
for c in R.columns:
    if c not in ("fips", "date"):
        R[c] = R[c].astype("f4")
p = "%s/agg_%s_s%02d.parquet" % (OUT, SCEN, SHARD)
R.to_parquet(p, index=False)
print("DONE %s shard %d -> %s rows %s  missing days %d  %.1f min"
      % (SCEN, SHARD, p, format(len(R), ","), miss, (time.time() - t0) / 60), flush=True)
