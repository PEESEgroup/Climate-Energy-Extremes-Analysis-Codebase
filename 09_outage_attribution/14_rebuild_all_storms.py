"""
Rebuild the DAILY observed-vs-expected demand series for all nine measurable landfalls.

`irma_daily_series_v2.csv` holds curves for only three storms (Irma, Emily, Gordon); the other
six exist in `tc_storm_table_v2.csv` as summary triples but have no day-by-day series, so a 3x3
small-multiple panel cannot be drawn from what is on disk. The cached inputs
(`ba_slab.npy`, `obs_mean.pkl`, the subregion net-load archive) are full time series and
storm-agnostic, and the storm table carries each storm's landfall date and BA list, so the same
construction generalises.

Helpers are copied verbatim from rebuild_irma_daily.py so the two agree by construction:
baseline is day -14..-5 before landfall, the headline window is day 0..+3.

SAME DISCIPLINE AS THE ORIGINAL: nothing is written unless every storm reproduces its published
obs_pct / mod_pct / div_pts from tc_storm_table_v2.csv to within 0.05 points. A storm that fails
is reported and dropped rather than quietly plotted.
"""
import json, sys
import os as _os_rp
for _rp in ("04_demand_model", "07_hazard_calendar", "09_outage_attribution",
            "02_downscale_wind", "12_figures"):
    _ap = _os_rp.path.abspath(_os_rp.path.join(
        _os_rp.path.dirname(_os_rp.path.abspath(__file__)), "..", _rp))
    if _os_rp.path.isdir(_ap) and _ap not in sys.path:
        sys.path.insert(0, _ap)
import numpy as np, pandas as pd
import paths

F = "/data/audit_orphans/fix"
# 2026-08-19. This used to name /data/tell_pred/future/hist_full40 directly, the directory paths.py
# marks superseded. Only the balancing-authority name list is taken from meta.npz, but naming the
# superseded product here is how a consumer drifts back onto it, so the name comes from paths.py.
# The load itself arrives through ba_slab.npy, which s3_build.py now writes SEDS-anchored.
LO = paths.LOAD_DIR
TOL = 0.05

meta = np.load(f"{LO}/meta.npz", allow_pickle=True)
BAS = [str(x) for x in meta["bas"]]
tt = pd.to_datetime(np.load(f"{F}/slab_times.npy"))
BASL = np.load(f"{F}/ba_slab.npy", mmap_mode="r")
OBSM = pd.read_pickle(f"{F}/obs_mean.pkl")


def d2d(a, idx):
    s = pd.Series(np.asarray(a, dtype=float), index=idx)
    d = s.groupby(s.index.date).mean(); d.index = pd.to_datetime(d.index); return d


def ba_series(bas):
    acc = np.zeros(len(tt))
    for b in bas:
        if b in BAS:
            acc += np.asarray(BASL[BAS.index(b)], dtype=float)
    return d2d(acc, tt)


def obs_series(bas):
    bas = [b for b in bas if b in OBSM.columns]
    return OBSM.loc[:, bas].sum(1, min_count=len(bas)), bas


def an(s, land, b0=14, b1=5):
    base = s[(s.index >= land - pd.Timedelta(days=b0)) & (s.index <= land - pd.Timedelta(days=b1))].mean()
    return (s / base - 1) * 100, base


def ac(a, land, d0=0, d1=3):
    return a[(a.index >= land + pd.Timedelta(days=d0)) & (a.index <= land + pd.Timedelta(days=d1))].mean()


T = pd.read_csv(f"{F}/tc_storm_table_v2.csv")
T = T[T.status == "ok"].copy()
print("storms to rebuild: %d" % len(T))

rows, report, bad = [], [], []
for _, r in T.iterrows():
    land = pd.Timestamp(r.landfall)
    bas = [b for b in str(r.bas).split(";") if b]
    o, used = obs_series(bas)
    m = ba_series(bas)
    oa, ob = an(o, land)
    ma, mb = an(m, land)
    got_o, got_m = float(ac(oa, land)), float(ac(ma, land))
    d_o, d_m = abs(got_o - r.obs_pct), abs(got_m - r.mod_pct)
    ok = (d_o <= TOL) and (d_m <= TOL)
    report.append(dict(storm=r.storm, n_bas_used=len(used), published_obs=float(r.obs_pct),
                       rebuilt_obs=round(got_o, 2), published_mod=float(r.mod_pct),
                       rebuilt_mod=round(got_m, 2), reproduced=bool(ok)))
    print("   %-9s bas %d/%d   obs %+7.2f vs %+7.2f (%+.2f)   mod %+7.2f vs %+7.2f (%+.2f)   %s"
          % (r.storm, len(used), len(bas), r.obs_pct, got_o, got_o - r.obs_pct,
             r.mod_pct, got_m, got_m - r.mod_pct, "OK" if ok else "MISMATCH"))
    if not ok:
        bad.append(r.storm); continue
    idx = pd.date_range(land - pd.Timedelta(days=14), land + pd.Timedelta(days=10))
    rows.append(pd.DataFrame({
        "storm": r.storm, "landfall": land, "date": idx,
        "day_rel": (idx - land).days,
        "obs_anom_pct": oa.reindex(idx).values,
        "mod_anom_pct": ma.reindex(idx).values,
        "wind_kt": r.wind_kt, "n_ba": len(used),
        "obs_pct": r.obs_pct, "div_pts": r.div_pts}))

if bad:
    print("\nNOT REPRODUCED, dropped: %s" % ", ".join(bad))
D = pd.concat(rows, ignore_index=True)
D.to_csv(f"{F}/tc_daily_series_all.csv", index=False)
json.dump({"tolerance_pts": TOL, "storms": report, "dropped": bad,
           "n_storms_written": int(D.storm.nunique())},
          open(f"{F}/tc_daily_series_all.json", "w"), indent=1)
print("\nWROTE %s/tc_daily_series_all.csv   storms %d   rows %d"
      % (F, D.storm.nunique(), len(D)))
