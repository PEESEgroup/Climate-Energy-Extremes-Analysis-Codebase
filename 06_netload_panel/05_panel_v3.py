"""Rebuild the analysis panel on our own net load.

The outcome columns are re-derived rather than assumed: the same construction is first applied to
the OLD net-load product, and its agreement with the published panel is printed. That comparison no
longer validates the port to the new net-load product, because the deseasonalization changed at the
same time; read it as a measurement of how far the new climatology moves the published columns.

Every outcome column is deseasonalized by subtracting the subregion's own day-of-year climatology,
pooled over a +-15 day circular window. That window is fixed here and cannot be changed from the
environment. It is the same +-15 day day-of-year window the hazard thresholds use, taken from
hazard_defs, and the window the Methods describe.

It is NOT the construction panel_v2 was built with. panel_v2 subtracted a raw per-calendar-day
mean: the smoothing used to be reached only by setting SMOOTH in the environment, and its default
was 0. So this file no longer reproduces the published anomaly columns, the reproduction check
below will no longer return r = 1.00000, and every anomaly column in panel_v3.parquet moves.
Everything fitted on the panel has to be refitted. That has not been measured here: this repository
has no /data, so no run of either construction was made while this file was edited."""
import json as _json
import os, sys

import numpy as np, pandas as pd

_HD = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "07_hazard_calendar")
if os.path.isdir(_HD) and _HD not in sys.path:
    sys.path.insert(0, _HD)
import hazard_defs as HD                      # the single source of truth for the shared window
import sys as _sys
import os as _os_rp
for _rp in ("04_demand_model", "07_hazard_calendar", "09_outage_attribution",
            "02_downscale_wind", "12_figures"):
    _ap = _os_rp.path.abspath(_os_rp.path.join(
        _os_rp.path.dirname(_os_rp.path.abspath(__file__)), "..", _rp))
    if _os_rp.path.isdir(_ap) and _ap not in _sys.path:
        _sys.path.insert(0, _ap)
import paths as _PATHS   # the one name for each of the three net-load products
LO = "/data/tell_pred/future/hist_full40"
P = pd.read_parquet("/data/enso/r1_causal/panel_v2.parquet"); P["date"] = pd.to_datetime(P.date)
subs = list(pd.unique(P.subregion))

def daily(npz):
    z = np.load(npz, allow_pickle=True)
    t = pd.to_datetime([str(x) for x in z["times"]])
    names = [str(x) for x in z["subregions"]]
    out = {}
    for key in ["net", "load", "wind", "solar"]:
        if key not in z.files: continue
        A = np.asarray(z[key], float)
        df = pd.DataFrame(A.T, index=t, columns=names)
        out[key + "_mean"] = df.resample("D").mean()
        out[key + "_peak"] = df.resample("D").max()
    return out

HALF = HD.DOY_WINDOW      # +/-15 days, from the shared module; not a literal, not an env var
DOY = np.arange(1, HD.DOY_CLIP + 1)
WIN = np.array([HD.doy_window(DOY, d, HALF) for d in DOY], dtype=float)   # row d = the days near d


def anom(df):
    """subtract each subregion's own day-of-year climatology, pooled over a +-15 day window

    The climatology for day of year d is the mean of EVERY observation within 15 days of d,
    wrapping across the year end: the window sums and the window counts are formed first, and
    divided once. Days of year with unequal record lengths are therefore weighted by the
    observations they actually contribute.

    This is the pooled estimator 08_adequacy_analysis/06_cond_oc.py uses for the covariates that condition
    these anomalies, so the panel and its controls are deseasonalized on one rule. It is not a mean
    of per-calendar-day means: the two differ on every day of year whose window contains a calendar
    day with a different number of observations, which the 366-to-365 clip guarantees at day 365.

    The window mask is hazard_defs.doy_window and the clip is hazard_defs.clip_doy, so this file
    carries no private copy of the day-of-year arithmetic."""
    doy = HD.clip_doy(df.index)
    v = df.values.astype(float)
    ok = np.isfinite(v)
    tot = pd.DataFrame(np.where(ok, v, 0.0), index=doy).groupby(level=0).sum() \
            .reindex(DOY).fillna(0.0).values
    cnt = pd.DataFrame(ok.astype(float), index=doy).groupby(level=0).sum() \
            .reindex(DOY).fillna(0.0).values
    num, den = WIN @ tot, WIN @ cnt
    clim = np.where(den > 0, num / np.where(den > 0, den, 1.0), np.nan)
    return pd.DataFrame(v - clim[doy - 1], index=df.index, columns=df.columns)

def build(npz):
    d = daily(npz)
    rows = []
    for s in subs:
        if s not in d["net_mean"].columns: continue
        idx = d["net_mean"].index
        r = pd.DataFrame({"subregion": s, "date": idx})
        r["netload_anom_mean"] = anom(d["net_mean"])[s].values
        r["netload_anom_peak"] = anom(d["net_peak"])[s].values
        r["load_anom_mean"] = anom(d["load_mean"])[s].values
        r["load_anom_peak"] = anom(d["load_peak"])[s].values
        vre_m = d["wind_mean"] + d["solar_mean"] if "wind_mean" in d else None
        vre_p = d["wind_peak"] + d["solar_peak"] if "wind_peak" in d else None
        r["vre_anom_mean"] = anom(vre_m)[s].values
        r["vre_anom_peak"] = anom(vre_p)[s].values
        rows.append(r)
    return pd.concat(rows, ignore_index=True)

# ------------------------------------------------------------------- the stamp on the output
# A consumer refuses a panel that carries no hazard_defs stamp. The panel fits none of
# the three table shapes hazard_defs.write_flags accepts: it is one dense row per subregion-day
# carrying SIX hazard columns at once, not a long table with a hazard column and not a single
# boolean flag. The stamp is therefore built with hazard_defs.stamp and written under
# hazard_defs.FLAG_META_KEY, which is what write_flags does internally, so hazard_defs.read_stamp
# and require_stamp read it back unchanged.
#
# The stamp names NO hazard, on purpose. This script builds no hazard flag: heat, cold, fire,
# vre_drought, ar_pub and tc_local are carried through from panel_v2.parquet untouched, and nothing
# here re-derives them or checks them against hazard_defs.HAZARDS. A definition hash on them would
# assert a provenance this script cannot verify. The day counts of the carried columns go in
# `extra` instead, so a consumer sees what is in the file without being told the constants were
# honored.
FLAG_COLS = {"heat": "heat", "cold": "cold", "fire": "fire", "vre_drought": "vre_drought",
             "ar_pub": "ar", "tc_local": "tc"}      # panel column -> hazard_defs.HAZARDS name

# TC IS REBUILT HERE, NOT CARRIED THROUGH. panel_v2.parquet carries an unstamped tc_local whose
# track file stopped on 2017-10-29, so 2018 and 2019 held zero tropical-cyclone subregion-days
# while the current stamped county product carries 481 and 801 county-days in those years. It was
# also the wrong quantity: 3,926 subregion-days against the 1,043 that HURDAT2 anchoring gives,
# because it counted any day reaching 34 kt whatever the cause. 10_hazard_tc.py writes the stamped
# flag, a track point within HD.TC_RADIUS_KM of the subregion centroid, and it is read here.
_TCF = "/data/tell_pred/future/hist_full40/subregion_tc_days.parquet"
HD.require_stamp(_TCF, ["tc"])
_TC = pd.read_parquet(_TCF)
_TC["date"] = pd.to_datetime(_TC["date"])
# The stamped product is a long list of flagged (subregion, date) pairs, not a wide indicator, so
# membership is the flag. Nothing else in the table needs reading.
_hit = set(zip(_TC.subregion.astype(str).values, pd.to_datetime(_TC.date).values))
P["tc_local"] = np.fromiter(
    ((sr, dt) in _hit for sr, dt in zip(P.subregion.astype(str).values, P.date.values)),
    dtype=float, count=len(P))
print("tc_local rebuilt from the stamped product: %d subregion-days (was %d carried through)"
      % (int((P.tc_local > 0).sum()), 3926), flush=True)


def write_panel(df, path):
    import pyarrow as pa
    import pyarrow.parquet as pq
    carried = {c: int((df[c] > 0.5).sum()) for c in FLAG_COLS if c in df.columns}
    st = HD.stamp(script=__file__, hazards=[], n_units=df.subregion.nunique(),
                  n_dates=df.date.nunique(), counts={},
                  extra={"builds_no_hazard_flag": True,
                         "hazard_columns_inherited_from": "panel_v2.parquet",
                         "hazard_columns_not_verified_here": dict(FLAG_COLS),
                         "hazard_column_days": carried,
                         "deseasonalization_window_days": HALF})
    t = pa.Table.from_pandas(df, preserve_index=False)
    meta = dict(t.schema.metadata or {})
    meta[HD.FLAG_META_KEY] = _json.dumps(st, sort_keys=True).encode()
    pq.write_table(t.replace_schema_metadata(meta), path)
    print("stamped %s: %s, hazard_defs %s, %d x %d" % (os.path.basename(path), st["script"],
                                                       st["hazard_defs_version"],
                                                       st["n_units"], st["n_dates"]))
    return path


OLD = f"{LO}/subregion_netload_1980_2019.npz"
# LO still names the analysis directory, but the net-load product moved to the SEDS-anchored
# tree when 03_stageE.py was repointed. Taking it from paths keeps one name for one product.
#
# FIXED ECONOMY. This panel feeds Figure 1, which is a fixed-fleet counterfactual: it asks what
# the weather does to a system held still, so the demand path is frozen at 2019 and only the
# weather varies. The anchored product grows 1.82x from 1980 to 2019, and on it the day-of-year
# climatology below would carry that trend into every anomaly column. Figure 2, Figure 3 and the
# future arm take the anchored product instead; see paths.py for which is which.
NEW = _PATHS.NETLOAD_FIXEDECON
if os.path.exists(OLD):
    print("comparing the rebuilt columns against the published panel (they will NOT agree: "
          "panel_v2 was built without the +-15 day window, so this measures the window, not the "
          "port to the new net-load product)")
    o = build(OLD)
else:
    print("superseded panel not present, skipping the one-time construction check "
          "(the r = 1.00000 agreement recorded on 2026-08-13 was measured with the unsmoothed "
          "climatology and no longer applies)")
    o = None
if o is not None:
  chk = P[["subregion", "date", "netload_anom_mean", "netload_anom_peak", "load_anom_mean", "vre_anom_mean"]].merge(
    o, on=["subregion", "date"], suffixes=("_pub", "_reb"))
  for c in ["netload_anom_mean", "netload_anom_peak", "load_anom_mean", "vre_anom_mean"]:
    a = chk[c + "_pub"].values; b = chk[c + "_reb"].values
    m = np.isfinite(a) & np.isfinite(b)
    print("  %-20s r=%.5f  bias %+8.1f MW  rmse %8.1f  n=%d"
          % (c, np.corrcoef(a[m], b[m])[0, 1], (b[m] - a[m]).mean(), np.sqrt(((b[m] - a[m]) ** 2).mean()), m.sum()))
print("\nrebuilding on our chain")
n = build(NEW)
keep = [c for c in P.columns if c not in n.columns or c in ("subregion", "date")]
V3 = n.merge(P[keep], on=["subregion", "date"], how="inner")
write_panel(V3, "/data/enso/r1_causal/panel_v3.parquet")
print("panel_v3 %s  %s -> %s" % (V3.shape, V3.date.min().date(), V3.date.max().date()))
for c in ["netload_anom_mean", "load_anom_mean", "vre_anom_mean"]:
    print("  %-20s sd published %8.1f   ours %8.1f MW" % (c, P[c].std(), V3[c].std()))
