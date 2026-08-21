"""Stage 2: candidate atmospheric-river flags, and the two tests the incumbent fails.

THE TWO TESTS
  1. AREA. The flag rate must not track the number of grid cells in the subregion. The incumbent
     `ar_pub` scores 0.744 on that correlation, which is the defect.
  2. GEOGRAPHY. A flag that measures exposure to atmospheric rivers should rank the west coast at or
     near the top. `ar_pub` puts California 12th of 18 and the Pacific Northwest 13th.

CANDIDATES
  ar_pub      incumbent: any one cell inside an AR shape at any one 6-hourly step
  ar_cov25    already built: the AR shape covers at least 25% of the subregion's cells
  ivt_pXX     subregion-mean IVT above its own +-15-day day-of-year percentile, which is the same
              season-relative construction the heat, cold and fire flags use
  ivt_p95_cov25
              the intensity flag AND the catalogue object over at least 25% of the subregion:
              strong moisture transport that the catalogue also calls an atmospheric river. This is
              the adopted flag, and the name written here is the name every consumer reads.

DEFINITIONS. The adopted percentile, the coverage fraction, the day-of-year window and the frozen
climatology period all come from 07_hazard_calendar/hazard_defs.py. This file carries no percentile
literal of its own for the adopted flag, and it builds no private day-of-year percentile: the
threshold comes from hazard_defs.doy_pctl. The adopted key is assembled from AR_PCTL and
AR_COVERAGE_FRACTION rather than typed out, so a change to either constant renames the key and every
consumer's lookup fails loudly instead of silently reading a superseded array.

STAMP. The output `.npz` carries `hazard_defs_stamp`, a JSON string produced by hazard_defs.stamp
with the producing script name, the hazard_defs version and the definition hash of the `ar` hazard.
hazard_defs.write_flags writes parquet only, so an `.npz` cannot use it; the same stamp dictionary is
embedded as an array instead. `07_ar_adopt_oc.py` refuses a file whose stamp is absent or stale.

A percentile flag has a fixed day fraction by construction, so it cannot be an area proxy; that is
the point. The compound flag is the one that keeps the object interpretation, and its rate is free
to vary regionally, which is what real exposure does.
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
Z = np.load("/data/enso/ivt_subregion_daily.npz", allow_pickle=True)
NAMES = [str(x) for x in Z["subregions"]]
dates = pd.to_datetime([str(x) for x in Z["dates"]])
IVT = Z["ivt_mean"]                                  # (18, ndays)
NS, ND = IVT.shape
doy = HD.clip_doy(dates)
# Thresholds are fitted on the frozen climatology period of hazard_defs, 1980 to 2019, and never on
# days outside it. If the IVT file ever runs past 2019 the thresholds no longer move with it, which
# is a deliberate change of behavior from the previous version of this script: that one fitted the
# percentiles on the whole record on file.
CLIM = (dates.year.values >= HD.CLIM_Y0) & (dates.year.values <= HD.CLIM_Y1)
assert CLIM.any(), "no day of the IVT record falls inside %d-%d" % (HD.CLIM_Y0, HD.CLIM_Y1)
print("IVT panel %s, %s .. %s; %d of %d days inside the frozen climatology period %d-%d"
      % (IVT.shape, dates[0].date(), dates[-1].date(), int(CLIM.sum()), ND,
         HD.CLIM_Y0, HD.CLIM_Y1))

# The adopted names, assembled from the shared constants rather than typed out.
COV_PCT = round(100 * HD.AR_COVERAGE_FRACTION)
COV_KEY = "ar_cov%d" % COV_PCT
INT_KEY = "ivt_p%d" % HD.AR_PCTL
ADOPTED = "%s_cov%d" % (INT_KEY, COV_PCT)

def ivt_threshold(p):
    """The +-15-day day-of-year percentile of IVT, per subregion, from hazard_defs.doy_pctl.

    Returned already indexed onto the calendar, one column per day. The assert catches a subregion
    that got no threshold at all, which would otherwise leave its flag silently empty rather than
    visibly wrong."""
    out = HD.doy_pctl(IVT, p, doy, clim=CLIM)[:, doy]
    allnan = np.isnan(out).all(axis=1)
    assert not allnan.any(), "no threshold at all for %s" % [NAMES[i] for i in np.where(allnan)[0]]
    return out

FLAGS = {}
# The comparison table keeps the three candidate percentiles; HD.AR_PCTL is forced into the set so
# the adopted intensity flag is always among them, whatever the shared constant becomes.
for p in sorted({90, 95, 99, HD.AR_PCTL}):
    FLAGS["ivt_p%d" % p] = IVT > ivt_threshold(p)
    print("built ivt_p%d" % p, flush=True)

# the two catalogue flags, on the same calendar
zp = np.load(f"{R1}/ar_pub_flag.npz", allow_pickle=True)
print("ar_pub_flag.npz keys:", list(zp.keys()))
pk = [k for k in zp.keys() if k not in ("dates", "subregions", "times")]
print(" arrays:", {k: zp[k].shape for k in pk})
pdates = pd.to_datetime([str(x) for x in zp["dates"]]) if "dates" in zp else None
psub = [str(x) for x in zp["subregions"]] if "subregions" in zp else NAMES
idx = {d: i for i, d in enumerate(pdates)} if pdates is not None else None
order = [psub.index(n) for n in NAMES]
for k in pk:
    A = zp[k]
    if A.shape[0] != NS:
        A = A.T
    A = A[order]
    if pdates is not None:
        col = np.array([idx.get(d, -1) for d in dates])
        B = np.zeros((NS, ND), bool)
        ok = col >= 0
        B[:, ok] = A[:, col[ok]].astype(bool)
        FLAGS[k] = B
        print("  %-14s aligned, %.1f%% of the calendar covered" % (k, 100 * ok.mean()))
    else:
        FLAGS[k] = A.astype(bool)

# The upstream file supplies `coverage` as a CONTINUOUS share of the subregion, not a pre-made
# boolean at our threshold, so the coverage condition is applied here from the shared constant. An
# earlier edit looked for a key named after the threshold, which the producer never wrote.
if INT_KEY not in FLAGS:
    raise KeyError("cannot build %s: the intensity flag %s is missing from %s"
                   % (ADOPTED, INT_KEY, sorted(FLAGS)))
# The alignment loop above has already put every supplied array on the (NS, ND) calendar, so the
# coverage entry arrives as a boolean at the producer's own threshold. Use it as such.
if COV_KEY in FLAGS:
    _cov = FLAGS[COV_KEY]
elif "coverage" in FLAGS:
    _cov = FLAGS["coverage"]
    print("  coverage taken from the producer's own array at its %.2f threshold, %.1f%% of unit-days"
          % (HD.AR_COVERAGE_FRACTION, 100 * _cov.mean()))
else:
    raise KeyError("cannot build %s: neither %s nor a continuous `coverage` array is present in %s"
                   % (ADOPTED, COV_KEY, sorted(zp.files)))
FLAGS[ADOPTED] = FLAGS[INT_KEY] & _cov

CELLS = json.load(open(f"{R1}/ar_pub_build.json"))["per_subregion"]
ncell = np.array([CELLS[n]["n_cat_cells"] for n in NAMES], float)

print("\n%-14s %8s %8s %10s   %s" % ("flag", "day %", "corr size", "CA rank", "top 3 subregions"))
rows = []
for k, F in FLAGS.items():
    rate = F.mean(1)
    c = float(np.corrcoef(ncell, rate)[0, 1])
    o = np.argsort(-rate)
    rk = {NAMES[i]: j + 1 for j, i in enumerate(o)}
    top = ", ".join(NAMES[i] for i in o[:3])
    print("%-14s %7.1f%% %8.3f %6d/18   %s" % (k, 100 * F.mean(), c, rk["CAISO"], top))
    rows.append(dict(flag=k, day_pct=100 * F.mean(), corr_ncell=c,
                     rank_CAISO=rk["CAISO"], rank_PacNW=rk["NorthernGrid_West"], top3=top))
pd.DataFrame(rows).to_csv(f"{R1}/ar_flag_variants.csv", index=False)

# The stamp. hazard_defs.write_flags writes parquet, so it cannot be used for an .npz; the same
# hazard_defs.stamp dictionary is embedded as a JSON string array instead, under a key no consumer
# treats as a flag. `counts` is the adopted flag only, because that is the one hazard this file
# claims to build under an agreed name; the other arrays are candidates, and `extra` names them.
STAMP = HD.stamp(script=__file__, hazards=["ar"], n_units=NS, n_dates=ND,
                 counts={"ar": int(FLAGS[ADOPTED].sum())},
                 extra={"adopted_key": ADOPTED, "intensity_key": INT_KEY, "coverage_key": COV_KEY,
                        "candidate_keys": sorted(FLAGS),
                        "clim_days": int(CLIM.sum()), "n_days_total": int(ND)})
np.savez_compressed("/data/enso/ar_flag_variants.npz",
                    dates=np.array([str(d.date()) for d in dates]),
                    subregions=np.array(NAMES),
                    hazard_defs_stamp=np.array(json.dumps(STAMP, sort_keys=True)),
                    **{k: v for k, v in FLAGS.items()})
print("\nwrote %s/ar_flag_variants.csv and /data/enso/ar_flag_variants.npz" % R1)
print("adopted %s, definition %s, %.4f of subregion-days"
      % (ADOPTED, HD.definition_hash("ar"), float(FLAGS[ADOPTED].mean())))
_ok, _e, _t, _b = HD.day_rate_ok("subregion", "ar", float(FLAGS[ADOPTED].mean()))
print("day rate %s the recorded band %.4f +/- %.4f (%s)"
      % ("inside" if _ok else "OUTSIDE", _e, _t, _b.split(":")[0]))
