"""
Task #62 (second half): county-resolved hazard flags 1980-2019, evaluated on each COUNTY's own
climatology. Every constant below is read from hazard_defs.py, the single source of truth.

  cold  CSDI      tmin < day-of-year p10 (smoothed +/-15 d), >= 3 consecutive days, December to
                  February only  [ETCCDI cold-spell duration index, Zhang 2011]
  heat            tmax > day-of-year p90 (smoothed +/-15 d), >= 3 consecutive days, June to
                  August only  [Perkins & Alexander 2013]
  fire  HDW       VPD(hPa) x 10 m wind (Srock 2018) above the county's own day-of-year p99
                  (smoothed +/-15 d), no persistence and no season gate  [Abatzoglou 2019 for the
                  local-percentile fire-weather-day construction]

NOTHING IS HARD-CODED HERE. The percentiles, the persistence lengths, the season months, the
+/-15 day window and the frozen climatology period all come from hazard_defs.HAZARDS and
hazard_defs.SHARED, and the day-of-year percentile and the persistence rule are hazard_defs
functions. This file used to carry private copies of both helpers and its own literals. They are
deleted: two copies of one definition are how the fire flag came to mean two different things.
The output parquet is written through hazard_defs.write_flags, so it carries the name of this
script and a hash of the constants used, and a consumer can refuse a file from a superseded build.

The record read here is exactly 1980-2019, and the climatology mask passed to the percentile is
hazard_defs.CLIM_Y0 to CLIM_Y1, the same 1980-2019. So no threshold is fitted on a period the
flags are not scored on, and the freeze survives if this table is ever extended past 2019.

COLD PERSISTENCE IS THREE DAYS AT COUNTY SCALE. That is the documented scale exception,
hazard_defs.COLD_PERSIST_DAYS_COUNTY; the 18-subregion build keeps the six-day ETCCDI rule,
hazard_defs.COLD_PERSIST_DAYS. The reason is that a county cold spell averages 1.77 days and only
1.5% of them reach six, so the six-day rule leaves 0.96 cold days a year and no usable county
signal. The future-adequacy arm (10_future_adequacy/05_hazfreq.py) already used three days, so the
persistence rule no longer differs between the two county arms. That arm still scores a different
record against its own frozen window, so the two sets of counts are not interchangeable.

THIS CHANGES WHAT THIS FILE PUBLISHED. Every county cold count produced by this script before
2026-08-18 was built with the six-day rule and is superseded. A three-day rule admits every spell
a six-day rule admits and more, so the new counts are strictly higher; by how much has not been
measured here, and no claim is made that any earlier county cold number is reproduced. County
cold and subregion cold remain two different rules and must never be pooled; the stamp records
which one was used.

WHY PERCENTILES TRANSFER AND ABSOLUTE CUTS DO NOT. Both builds average U and V over cells first
and only then take sqrt(U^2+V^2), the speed of the mean vector. A subregion has thousands of
cells so the vectors cancel; a county has 17.9 so they barely do, and county wind runs +0.84 m/s
higher (validated: r = 0.936 against the subregion table, temperature r = 0.99999). A percentile
computed on the county's own 1980-2019 distribution is therefore self-consistent; an absolute
threshold copied from the subregion build would not be. HDW magnitudes must never be compared
across the two scales.
"""
import os as _os, sys as _sys

# hazard_defs.py is the single source of truth for every constant and helper used below. It sits
# next to this file in the repository (07_hazard_calendar/) and must be deployed next to it again
# on the flat tree (/data/hazard_defs.py, alongside /data/11_county_hazard_flags.py). Both candidate
# directories are offered to sys.path so the same file runs unchanged in either layout.
for _d in (_os.path.dirname(_os.path.abspath(__file__)),
           _os.path.join(_os.path.dirname(_os.path.abspath(__file__)), "..", "07_hazard_calendar")):
    if _os.path.isdir(_d) and _d not in _sys.path:
        _sys.path.insert(0, _d)
import hazard_defs as HD

import numpy as np, pandas as pd

W = pd.read_parquet("/data/enso/county_weather_daily.parquet")
W["date"] = pd.to_datetime(W.date)
fips = np.array(sorted(W.fips.unique()))
dates = np.array(sorted(W.date.unique()))
NC, ND = len(fips), len(dates)
print("counties %d  days %d" % (NC, ND), flush=True)

fi = {f: i for i, f in enumerate(fips)}
di = {d: i for i, d in enumerate(dates)}
r = W.fips.map(fi).values
c = W.date.map(di).values


def grid(col):
    a = np.full((NC, ND), np.nan, dtype=np.float32)
    a[r, c] = W[col].values
    return a


tmin, tmax = grid("tmin"), grid("tmax")
q, ps, wspd = grid("q"), grid("ps"), grid("wspd")
del W

dd = pd.DatetimeIndex(dates)
doy = HD.clip_doy(dates)                            # np.minimum(dayofyear, 365), the shared rule
mon = dd.month.values
CLIM = (dd.year.values >= HD.CLIM_Y0) & (dd.year.values <= HD.CLIM_Y1)
if not CLIM.all():
    print("   %d of %d days lie outside the frozen %d-%d climatology: they are SCORED against it, "
          "never fitted into it" % (int((~CLIM).sum()), ND, HD.CLIM_Y0, HD.CLIM_Y1), flush=True)

COLD, HEAT, FIRE = HD.HAZARDS["cold"], HD.HAZARDS["heat"], HD.HAZARDS["fire"]
K_COLD = COLD["persist_days_county"]                # the county scale exception, 3 days

print("cold ...", flush=True)
# persistence first, then the December-to-February gate, so a spell is scored on its own length
# and only afterwards restricted to winter. Same order as the 18-subregion build. The run length
# is the COUNTY constant: 3 days here, 6 at subregion scale. See the docstring for why.
p10 = HD.doy_pctl(tmin, COLD["pctl"], doy, clim=CLIM)
cold = HD.persist(tmin < p10[:, doy], K_COLD) & HD.season(COLD["months"], mon)[None, :]

print("heat ...", flush=True)
p90 = HD.doy_pctl(tmax, HEAT["pctl"], doy, clim=CLIM)
heat = (HD.persist(tmax > p90[:, doy], HEAT["persist_days"])
        & HD.season(HEAT["months"], mon)[None, :])

print("fire ...", flush=True)
# HDW = VPD(hPa) x 10 m wind (Srock 2018), flagged above the county's OWN day-of-year p99 on the
# same +/-15 day window used by heat and cold. There is no persistence rule and no season gate:
# fire weather is a single-day hazard, and the western fire season is autumn, so a summer gate
# would delete exactly the days that matter. VPD is in hPa; es is the 6.112 hPa form and e_
# divides by 100 because ps is Pa in the TGW county table.
es = 6.112 * np.exp(17.67 * (tmax - 273.15) / (tmax - 29.65))
e_ = q * ps / (0.622 + 0.378 * q) / 100.0
hdw = np.clip(es - e_, 0, None) * wspd                     # hPa m/s
del es, e_
p99h = HD.doy_pctl(hdw, FIRE["pctl"], doy, clim=CLIM)
fire = (HD.persist(hdw > p99h[:, doy], FIRE["persist_days"])
        & HD.season(FIRE["months"], mon)[None, :])   # persist_days 1 and months None: both no-ops

flags = {"cold": cold, "heat": heat, "fire": fire}
for k, v in flags.items():
    n = v.sum()
    print("   %-5s %s county-days  (%.2f%% of all)   counties with >=10 d: %d"
          % (k, format(int(n), ","), 100 * n / (NC * ND), int((v.sum(1) >= 10).sum())), flush=True)

long = []
for k, v in flags.items():
    ii, jj = np.where(v)
    long.append(pd.DataFrame({"fips": fips[ii], "date": dates[jj], "hazard": k}))
L = pd.concat(long, ignore_index=True).sort_values(["hazard", "fips", "date"])
# The stamp travels inside the parquet: the producing script, the hazard_defs version, a hash of
# the constants used, and extra["cold_persist_days"] so a consumer can tell this three-day county
# build from the superseded six-day one. A consumer refuses an unstamped file.
HD.write_flags(L, "/data/enso/county_hazard_flags.parquet",
               script=__file__, n_units=NC, n_dates=ND,
               hazards=["cold", "heat", "fire"], extra={"cold_persist_days": int(K_COLD)})
print("\nWROTE /data/enso/county_hazard_flags.parquet   rows %s   stamped hazard_defs %s, "
      "cold_persist_days %d" % (format(len(L), ","), HD.VERSION, K_COLD))

# ---------------- sanity: do county flags aggregate up to the subregion flags? ----------------
P = pd.read_parquet("/data/enso/r1_causal/panel_v2.parquet",
                    columns=["subregion", "date", "cold", "heat", "fire"])
P["date"] = pd.to_datetime(P.date)
sm = np.load("/data/datasets/grid/subregion_mask.npz", allow_pickle=True)
smask = sm["subregion_mask"]; id2sub = {int(a): str(b) for a, b in sm["id_to_subregion"]}
zc = np.load("/data/datasets/grid/coordinate.npz")
glat, glon = zc["lat"].astype(float), zc["lon"].astype(float)
gz = pd.read_csv("/data/equity_cost/analysis/did/2023_Gaz_counties_national.txt",
                 sep="\t", dtype={"GEOID": str})
gz.columns = [x.strip() for x in gz.columns]; gz["fips"] = gz.GEOID.str.zfill(5)
ri = np.clip(np.searchsorted(glat, gz.INTPTLAT.values), 0, len(glat) - 1)
ci = np.clip(np.searchsorted(glon, gz.INTPTLONG.values), 0, len(glon) - 1)
sid = smask[ri, ci]
c2s = {f: id2sub[int(s)] for f, s in zip(gz.fips, sid) if int(s) in id2sub}
sub_of = np.array([c2s.get(f, "") for f in fips])
# The cold row of this diagnostic compares two persistence rules: three days here against the
# six-day subregion column of panel_v2. It is a direction check, not an agreement measure.
print("\ncounty flag share on subregion-flagged days vs off (a valid county flag should be much")
print("higher inside its subregion's flagged days):")
for k, v in flags.items():
    pv = P.pivot_table(index="subregion", columns="date", values=k, aggfunc="max")
    pv = pv.reindex(columns=pd.DatetimeIndex(dates))
    rows = []
    for s in np.unique(sub_of):
        if s == "" or s not in pv.index:
            continue
        m = sub_of == s
        sf = pv.loc[s].values.astype(bool)
        rows.append((v[m][:, sf].mean(), v[m][:, ~sf].mean()))
    a = np.array(rows)
    print("   %-5s inside %.3f   outside %.3f   ratio %.1fx"
          % (k, a[:, 0].mean(), a[:, 1].mean(), a[:, 0].mean() / max(a[:, 1].mean(), 1e-9)))
