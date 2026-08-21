"""
Task #63, step 3: merge both CONUS404 arms, rebuild the hazard flags with the climatology frozen
on 1980-2019, and check against the TGW-built flags before anything downstream is re-estimated.

Arms:
  1980-2019  /data/enso/county_weather_c404/        from local /data/c404_native
  2020-2022  /data/enso/county_weather_c404_tail/   streamed from the Planetary Computer zarr

Both are the same 1015 x 1367 grid and the same county aggregation, so the join is exact rather
than a splice. CONUS404 on PC ends 2022-09-30, so the record is 1980-01-01 .. 2022-09-30 and
Elliott (Dec 2022) is out of reach: declared, not silently dropped.

THRESHOLDS ARE FROZEN ON 1980-2019, the period hazard_defs.CLIM_Y0 to CLIM_Y1 names. Day-of-year
percentiles are computed on that period only and then applied to the whole record, so 2020-2022 is
scored against the same climatology as everything already estimated and the extension cannot
manufacture a trend.

NOTHING IS HARD-CODED HERE. The percentiles, the persistence lengths, the season months, the
+/-15 day window and the climatology years are read from hazard_defs.HAZARDS and from
hazard_defs.SHARED, and the day-of-year percentile and the persistence rule are hazard_defs
functions. The private copies this file used to carry are deleted. The flag parquet is written
through hazard_defs.write_flags, so it carries this script's name and a hash of the constants.

Hazard definitions, identical to the TGW county build in 11_county_hazard_flags.py:
  cold  tmin < day-of-year p10 (+/-15 d window), >= 3 consecutive days, December to February
  heat  tmax > day-of-year p90 (+/-15 d window), >= 3 consecutive days, June to August
  fire  HDW = VPD(hPa) x 10 m wind above the county's own day-of-year p99 (+/-15 d window),
        no persistence and no season gate

COLD PERSISTENCE IS THREE DAYS AT COUNTY SCALE, hazard_defs.COLD_PERSIST_DAYS_COUNTY. This is a
county build, so the documented scale exception applies here exactly as it does in
11_county_hazard_flags.py: a county cold spell averages 1.77 days, and the six-day subregion rule
leaves 0.96 county cold days a year. The rule changed on 2026-08-18. Every county cold count this
script produced before that date used six days, is superseded, and is not reproduced by the code
below; the change has not been checked against any earlier run.

The agreement check is the decision point: if the CONUS404 flags and the TGW flags disagree
badly, the Stage 1 numbers already reported are not comparable to anything built on this record
and must be re-estimated. This script does NOT re-run Stage 1. It writes the kappa and Jaccard
per hazard to c404_vs_tgw_flag_agreement.json and stops; acting on them is a separate step.
"""
import glob, json
import os as _os, sys as _sys

# hazard_defs.py is the single source of truth. It sits next to this file in the repository
# (07_hazard_calendar/) and must be deployed next to it again on the flat tree
# (/data/hazard_defs.py, alongside /data/04_c404_pipeline.py). Both candidates go on sys.path.
for _d in (_os.path.dirname(_os.path.abspath(__file__)),
           _os.path.join(_os.path.dirname(_os.path.abspath(__file__)), "..", "07_hazard_calendar")):
    if _os.path.isdir(_d) and _d not in _sys.path:
        _sys.path.insert(0, _d)
import hazard_defs as HD

import numpy as np, pandas as pd

E = "/data/enso"
OUTP = f"{E}/county_weather_daily_c404.parquet"
FLAGP = f"{E}/county_hazard_flags_c404.parquet"

a = sorted(glob.glob(f"{E}/county_weather_c404/c404_county_shard*.parquet"))
b = sorted(glob.glob(f"{E}/county_weather_c404_tail/tail_shard*.parquet"))
print("shards: %d local + %d tail" % (len(a), len(b)), flush=True)
d = pd.concat([pd.read_parquet(f) for f in a + b], ignore_index=True)
nbad = sum(len(pd.read_csv(f)) for f in glob.glob(f"{E}/county_weather_c404*/bad_shard*.csv"))
d["date"] = pd.to_datetime(d.date)
d = d.drop_duplicates(["fips", "date"]).sort_values(["fips", "date"])

# UNIT MISMATCH BETWEEN THE TWO ARMS, caught by an implausible -85% fall in fire-weather days.
# /data/c404_native stores PSFC in hPa - surface pressure is ~100,000 Pa and float16 tops out at
# 65,504, so the download must have scaled it - while the Planetary Computer zarr serves Pa. The
# ratio is exactly 100.004 and every other variable matches across the join (tmax 1.003, q 1.007,
# wspd 1.005), which is what identified pressure as the culprit rather than a real climate signal.
# Left uncorrected, e_ = q*ps/(0.622+0.378q)/100 comes out 100x too small on the local arm, VPD
# collapses to es, HDW runs ~2x high, and the day-of-year p99 threshold fitted on 1980-2019 is
# then unreachable for 2020-2022. TGW is unaffected (its ps is already Pa), so the fire result
# already reported from the TGW record stands.
lo = d.ps < 2000
if lo.any():
    print("   PSFC unit fix: %s rows in hPa -> Pa (%.1f%% of the record)"
          % (format(int(lo.sum()), ","), 100 * lo.mean()), flush=True)
    d.loc[lo, "ps"] = d.loc[lo, "ps"] * 100.0
print("   ps after fix: mean %.1f Pa  (1980-2019 %.1f, 2020-2022 %.1f)"
      % (d.ps.mean(), d.loc[d.date.dt.year <= 2019, "ps"].mean(),
         d.loc[d.date.dt.year >= 2020, "ps"].mean()), flush=True)
print("rows %s   counties %d   dates %d   %s .. %s   unreadable skipped %d"
      % (format(len(d), ","), d.fips.nunique(), d.date.nunique(),
         d.date.min().date(), d.date.max().date(), nbad), flush=True)
d.drop(columns=["n"]).to_parquet(OUTP, index=False)
print("WROTE", OUTP, flush=True)

fips = np.array(sorted(d.fips.unique()))
dates = np.array(sorted(d.date.unique()), dtype="datetime64[ns]")
NC, ND = len(fips), len(dates)
fi = {f: i for i, f in enumerate(fips)}; di = {x: i for i, x in enumerate(dates)}
r = d.fips.map(fi).values; c = d.date.map(di).values


def grid(col):
    x = np.full((NC, ND), np.nan, "f4"); x[r, c] = d[col].values
    return x


tmin, tmax = grid("tmin"), grid("tmax")
q, ps, wspd = grid("q"), grid("ps"), grid("wspd")
del d
idx = pd.DatetimeIndex(dates)
doy = HD.clip_doy(dates)                            # np.minimum(dayofyear, 365), the shared rule
mon = idx.month.values
CLIM = (idx.year.values >= HD.CLIM_Y0) & (idx.year.values <= HD.CLIM_Y1)   # frozen climatology

COLD, HEAT, FIRE = HD.HAZARDS["cold"], HD.HAZARDS["heat"], HD.HAZARDS["fire"]
K_COLD = COLD["persist_days_county"]                # the county scale exception, 3 days

print("flags, thresholds frozen on %d-%d ..." % (HD.CLIM_Y0, HD.CLIM_Y1), flush=True)
# persistence first, then the season gate, so a spell is scored on its own length and only
# afterwards restricted to winter or summer. Same order as 11_county_hazard_flags.py, and the same
# county cold run length of 3 days.
cold = (HD.persist(tmin < HD.doy_pctl(tmin, COLD["pctl"], doy, clim=CLIM)[:, doy], K_COLD)
        & HD.season(COLD["months"], mon)[None, :])
heat = (HD.persist(tmax > HD.doy_pctl(tmax, HEAT["pctl"], doy, clim=CLIM)[:, doy],
                   HEAT["persist_days"])
        & HD.season(HEAT["months"], mon)[None, :])
# Fire weather is HDW above the county's OWN day-of-year p99 on the same +/-15 day window, with
# no persistence rule and no season gate (Srock 2018 for the index, Abatzoglou 2019 for the
# local-percentile fire-weather day). The percentile goes through doy_pctl, so it is fitted on
# 1980-2019 like heat and cold and the 2020-2022 tail is scored against the frozen climatology.
# VPD is in hPa: es is the 6.112 hPa form and e_ divides by 100 because ps is Pa here, which is
# true only AFTER the hPa -> Pa repair applied to the local arm above.
es = 6.112 * np.exp(17.67 * (tmax - 273.15) / (tmax - 29.65))
e_ = q * ps / (0.622 + 0.378 * q) / 100.0
hdw = np.clip(es - e_, 0, None) * wspd                     # hPa m/s
del es, e_
fire = (HD.persist(hdw > HD.doy_pctl(hdw, FIRE["pctl"], doy, clim=CLIM)[:, doy],
                   FIRE["persist_days"])
        & HD.season(FIRE["months"], mon)[None, :])   # persist_days 1 and months None: both no-ops
FL = {"cold": cold, "heat": heat, "fire": fire}
for k, v in FL.items():
    pre = v[:, CLIM].mean(); post = v[:, ~CLIM].mean()
    print("   %-5s %s county-days   1980-2019 rate %.4f   2020-2022 rate %.4f  (%+.0f%%)"
          % (k, format(int(v.sum()), ","), pre, post, 100 * (post / max(pre, 1e-9) - 1)), flush=True)

long = []
for k, v in FL.items():
    ii, jj = np.where(v)
    long.append(pd.DataFrame({"fips": fips[ii], "date": dates[jj], "hazard": k}))
L = pd.concat(long, ignore_index=True)
# Stamped: the producing script, the hazard_defs version, a hash of the constants, and
# extra["cold_persist_days"] so a consumer can tell this three-day county build from the
# superseded six-day one. A consumer refuses an unstamped flag file.
HD.write_flags(L, FLAGP, script=__file__, n_units=NC, n_dates=ND,
               hazards=["cold", "heat", "fire"], extra={"cold_persist_days": int(K_COLD)})
print("WROTE %s  rows %s  stamped hazard_defs %s, cold_persist_days %d"
      % (FLAGP, format(len(L), ","), HD.VERSION, K_COLD), flush=True)

# ---------------------------------------------------------------- agreement with TGW, 1980-2019
# Refuse a TGW flag table built by a superseded builder: comparing two hazard taxonomies would
# produce a kappa that means nothing. require_stamp raises on a missing stamp or a changed hash.
_ST = HD.require_stamp(f"{E}/county_hazard_flags.parquet", hazards=["cold", "heat", "fire"])
_KT = _ST.get("extra", {}).get("cold_persist_days")
if int(_KT if _KT is not None else -1) != int(K_COLD):
    raise ValueError("the TGW county flags were built with cold_persist_days=%s, this build uses "
                     "%d; the cold rows are not comparable" % (_KT, K_COLD))
print("TGW flags written by %s, hazard_defs %s, cold_persist_days %s"
      % (_ST.get("script"), _ST.get("hazard_defs_version"), _KT), flush=True)
T = pd.read_parquet(f"{E}/county_hazard_flags.parquet")
T["date"] = pd.to_datetime(T.date)
rep = {}
print("\nagreement with the TGW-built flags over 1980-2019:", flush=True)
print("   %-5s %11s %11s %11s %8s %8s" % ("haz", "TGW", "C404", "both", "Jaccard", "kappa"))
nCLIM = NC * int(CLIM.sum())
for k in FL:
    A = set(map(tuple, T[T.hazard == k][["fips", "date"]].values))
    bi = np.where(FL[k][:, CLIM])
    dsub = dates[CLIM]
    B = set(zip(fips[bi[0]], pd.DatetimeIndex(dsub[bi[1]])))
    inter = len(A & B); uni = len(A | B)
    pa, pb = len(A) / nCLIM, len(B) / nCLIM
    po = (inter + (nCLIM - uni)) / nCLIM
    pe = pa * pb + (1 - pa) * (1 - pb)
    kap = (po - pe) / (1 - pe)
    print("   %-5s %11s %11s %11s %8.3f %8.3f"
          % (k, format(len(A), ","), format(len(B), ","), format(inter, ","),
             inter / max(uni, 1), kap), flush=True)
    rep[k] = dict(tgw=len(A), c404=len(B), both=inter, jaccard=inter / max(uni, 1), kappa=kap)

json.dump({"agreement_1980_2019": rep, "n_counties": int(NC), "n_dates": int(ND),
           "record": [str(dates[0])[:10], str(dates[-1])[:10]],
           "unreadable_files": int(nbad),
           "note": "kappa >= 0.6 -> the two arms are comparable; below that the TGW-based Stage 1 "
                   "numbers are not transferable and only the CONUS404 re-estimate should be used."},
          open(f"{E}/c404_vs_tgw_flag_agreement.json", "w"), indent=1)
print("\ndone", flush=True)
