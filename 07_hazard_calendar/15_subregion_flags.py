"""Build the 18-subregion hazard flags and write them into the analysis panel.

This file did not exist before 2026-08-18. Until then the subregion heat, cold, fire and
renewable-drought columns were inherited from panel_v2, which had been built with a different and
undocumented rule set: heat was a day-of-year p95 single day gated to June-August with no
persistence, cold a p5 single day gated to December-February with no persistence, fire a compound
percentile triple on humidity, wind and temperature, and renewable drought a wind-and-sunshine
percentile compound rather than an output rule. None of those four matched what the paper and the
supplementary information describe, and none matched the county builds.

Every rule here comes from hazard_defs, so this build and the county builds are the same
definitions evaluated on different units.

INPUTS
  /data/enso/subregion_weather_daily.csv                      daily tmax, tmin, q, ps, wspd by subregion
  /data/tell_pred/future/hist_full40_seds/subregion_netload_ourchain_1980_2019_fixedecon.npz  wind and solar

OUTPUT
  /data/enso/r1_causal/panel_v3.parquet                       the four flag columns, overwritten in place

RUN ORDER. 06_netload_panel/05_panel_v3.py must run FIRST: it rebuilds the outcome columns and writes the file.
This script then overwrites the four flag columns and replaces the stamp, so the stamp on the finished
panel names this file. Running them the other way round leaves the old panel_v2 flags in place.
"""
import os as _os, sys as _sys
_HD = _os.path.join(_os.path.dirname(_os.path.abspath(__file__)), ".")
if _HD not in _sys.path: _sys.path.insert(0, _HD)
import hazard_defs as HD
import numpy as np, pandas as pd
import sys as _sys
import os as _os_rp
for _rp in ("04_demand_model", "07_hazard_calendar", "09_outage_attribution",
            "02_downscale_wind", "12_figures"):
    _ap = _os_rp.path.abspath(_os_rp.path.join(
        _os_rp.path.dirname(_os_rp.path.abspath(__file__)), "..", _rp))
    if _os_rp.path.isdir(_ap) and _ap not in _sys.path:
        _sys.path.insert(0, _ap)
import paths as _PATHS   # the one name for each of the three net-load products

SUBW = "/data/enso/subregion_weather_daily.csv"
# The fixed-economy product, so this build reads the same file the Figure 1 panel is built on.
# Only wind and solar are taken from it, and those two arrays are bit-identical between the
# fixed-economy and the anchored product, so the renewable-drought flag is unchanged by this.
GEN = _PATHS.NETLOAD_FIXEDECON
PANEL = "/data/enso/r1_causal/panel_v3.parquet"

W = pd.read_csv(SUBW); W["date"] = pd.to_datetime(W.date)
subs = sorted(W["sub"].unique()); dates = np.array(sorted(W.date.unique()))
NS, ND = len(subs), len(dates)
si = {s: i for i, s in enumerate(subs)}; di = {d: i for i, d in enumerate(dates)}
r = W["sub"].map(si).values; c = W.date.map(di).values
def grid(col):
    a = np.full((NS, ND), np.nan, "f8"); a[r, c] = W[col].values; return a
idx = pd.DatetimeIndex(dates); doy = HD.clip_doy(idx); mon = idx.month.values
print("subregions %d  days %d  %s .. %s" % (NS, ND, dates[0], dates[-1]), flush=True)

tmax, tmin, wspd, q, ps = (grid(k) for k in ("tmax", "tmin", "wspd", "q", "ps"))
H = {}
HEAT, COLD, FIRE = HD.HAZARDS["heat"], HD.HAZARDS["cold"], HD.HAZARDS["fire"]
H["heat"] = HD.spell(tmax > HD.doy_pctl(tmax, HEAT["pctl"], doy)[:, doy],
                     HEAT["persist_days"], HEAT["months"], mon)
H["cold"] = HD.spell(tmin < HD.doy_pctl(tmin, COLD["pctl"], doy)[:, doy],
                     COLD["persist_days"], COLD["months"], mon)   # subregion keeps the six-day rule
# HDW = VPD * 10 m wind, VPD in hPa, saturation from Tetens at the daily maximum temperature
e = q * ps / (0.622 + 0.378 * q) / 100.0
es = 6.112 * np.exp(17.67 * (tmax - 273.15) / (tmax - 29.65))
hdw = np.clip(es - e, 0, None) * wspd
H["fire"] = HD.spell(hdw > HD.doy_pctl(hdw, FIRE["pctl"], doy)[:, doy],
                     FIRE["persist_days"], FIRE["months"], mon)
# renewable drought is an OUTPUT rule, Rinaldi: combined wind and solar below a fraction of its own
# day-of-year climatology. It is not a wind-and-sunshine percentile compound.
z = np.load(GEN, allow_pickle=True)
gt = pd.to_datetime([str(x) for x in z["times"]]); gn = [str(x) for x in z["subregions"]]
V = pd.DataFrame((np.asarray(z["wind"], float) + np.asarray(z["solar"], float)).T,
                 index=gt, columns=gn).resample("D").mean()
name_of = {i + 1: n for i, n in enumerate(gn)}
V = V.reindex(columns=[name_of[s] for s in subs]).reindex(idx)
vre = V.values.T
cl = np.full((NS, HD.DOY_CLIP + 1), np.nan)
for d in range(1, HD.DOY_CLIP + 1):
    cl[:, d] = np.nanmean(vre[:, HD.doy_window(doy, d)], axis=1)
H["vre_drought"] = vre < HD.VRE_FRACTION * cl[:, doy]

for k, v in H.items():
    v = np.where(np.isnan(tmax), False, v); H[k] = v
    print("  %-13s %7d subregion-days  %6.3f%%   hash %s"
          % (k, v.sum(), 100 * v.mean(), HD.definition_hash(k)), flush=True)

P = pd.read_parquet(PANEL); P["date"] = pd.to_datetime(P.date)
key = pd.MultiIndex.from_arrays([P.subregion.values, P.date.values])
full = pd.MultiIndex.from_product([[name_of[s] for s in subs], idx])
for k, v in H.items():
    P[k] = pd.Series(v.ravel(), index=full).reindex(key).values.astype(float)
import pyarrow as pa, pyarrow.parquet as pq, json as _json
st = HD.stamp(script=__file__, hazards=list(H), n_units=NS, n_dates=ND,
              counts={k: int(v.sum()) for k, v in H.items()},
              extra={"overwrites_flag_columns_of": "05_panel_v3.py",
                     "cold_persist_days": int(COLD["persist_days"])})
t = pa.Table.from_pandas(P, preserve_index=False)
meta = dict(t.schema.metadata or {}); meta[HD.FLAG_META_KEY] = _json.dumps(st, sort_keys=True).encode()
pq.write_table(t.replace_schema_metadata(meta), PANEL)
print("WROTE %s %s, hazard_defs %s" % (PANEL, P.shape, st["hazard_defs_version"]), flush=True)
