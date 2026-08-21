"""Figure 1's hazard taxonomy, carried into the futures, by county.

Figure 1 says where each hazard bites historically. Nothing in the futures figure answers the
obvious next question, which is how often each of those same hazards happens under warming, and
where. This computes it on the county daily aggregates the damage analysis already built and writes
one row per county. Rolling those counties up to the 18 planning subregions happens downstream;
nothing in this file does it.

WHERE THE DEFINITIONS COME FROM. Heat, cold and fire weather are read from
07_hazard_calendar/hazard_defs.py, which is the single source of truth for the seven agreed
hazards. This file holds no percentile, no persistence length and no season month of its own for
those three, and it uses the shared day-of-year percentile, the shared persistence rule and the
shared season gate rather than private copies. Two rows of the output table are NOT among the seven
agreed hazards and therefore have no shared entry: humid heat and heavy rain. Their constants are
named below, in one place, with the reason.

PAIRING. TGW-future is a +40-year replay of the observed synoptic sequence, so 2030-2050 pairs with
1990-2010 and nothing else. The historical baseline is that window, not the full record: using
1990-2019 would compare 21 future years against 30 historical ones drawn from a different set of
storms. Both sides of the pairing are now enforced in code: the historical load is filtered to
1990-2010 and each future load is filtered to 2030-2050.

CLIMATOLOGY PERIOD, a declared exception. hazard_defs freezes every threshold in this repository on
1980-2019 (hazard_defs.CLIM_Y0, CLIM_Y1). The future arm cannot use that period, because the replay
pairs 2030-2050 with 1990-2010 and the county aggregates read here do not run back to 1980. The
exception is the two constants CLIM_Y0_FUTURE_ARM and CLIM_Y1_FUTURE_ARM below, it is printed at
run time, and it is recorded in the stamp written with the output. It means county day-counts from
this file are never comparable in level with county day-counts built on the 1980-2019 freeze, only
as changes within this file.

DEFINITIONS, as the code runs them. Every day-of-year percentile below uses the shared +/-15 day
circular window, frozen on the 1990-2010 baseline:
  heat         t2max above the county's own day-of-year 90th percentile, 3 consecutive days,
               June to August                                   [hazard_defs.HAZARDS["heat"]]
  cold         t2min below the county's own day-of-year 10th percentile, 3 consecutive days,
               December to February. Three days, not six, is the documented COUNTY scale
               exception: hazard_defs.COLD_PERSIST_DAYS_COUNTY.
  humid heat   theta-e max above the county's own day-of-year 90th percentile, 3 consecutive
               days, June to August. Not one of the seven agreed hazards; see HUMID_* below.
  fire weather HDW = 10 m wind x VPD above the county's own day-of-year 99th percentile, single
               day, no persistence rule and no season gate    [hazard_defs.HAZARDS["fire"]]
  heavy rain   daily total above the county's own 99th percentile of wet days. This threshold is
               ONE number per county taken on the whole 1990-2010 record, not a day-of-year
               curve, so heavy rain is an annual-percentile flag and it fires mostly in the
               county's own wet season. Not one of the seven agreed hazards; see RAIN_* below.
  34 kt wind   any county-day where some part of the county reached 34 kt, whatever caused it.
               It is written out under the column name "tropical cyclone"; 06_fixtc.py then renames
               it "high wind" and writes the HURDAT2-anchored TC county-days into that name.
Every threshold here except the 34 kt one is the county's OWN historical value, so a heat day
means the same thing in Maine and in Arizona, and the future number is a frequency change rather
than a level. The 34 kt row is an absolute wind speed, identical in every county. So is the
tropical cyclone row that 06_fixtc.py substitutes afterwards, which comes from the HURDAT2 catalog
and not from any county percentile.

OUTPUT AND ITS STAMP. The table is written as a CSV, because that is what 06_fixtc.py and
12_figures/06_fig5_adequacy_oc.py read, and a CSV has nowhere to carry provenance. The same table is
therefore also written as a parquet twin whose schema metadata carries the hazard_defs stamp: the
script name, the hazard_defs version, the definition hash of heat, cold and fire, the flagged
county-day counts behind the baseline row, and the climatology exception. A consumer that needs to
refuse a file built by a superseded builder reads the twin with hazard_defs.read_stamp or
hazard_defs.require_stamp. The CSV itself remains unstamped and cannot be checked.
"""
import glob
import json
import os
import sys

import numpy as np, pandas as pd

# hazard_defs.py is the single source of truth for the hazard constants, the day-of-year
# percentile, the persistence rule and the season gate. It sits in 07_hazard_calendar/ in the
# repository and beside this script on the deployment box, where every script lives flat in /data.
# Both places go on the path, the repository copy first, so a flat deployment cannot shadow it.
_HERE = os.path.dirname(os.path.abspath(__file__))
for _p in (_HERE, os.path.join(_HERE, os.pardir, "07_hazard_calendar")):
    if os.path.isdir(_p) and _p not in sys.path:
        sys.path.insert(0, _p)
import hazard_defs as HD

CA = "/data/scratch_r5/county_agg"
OUT_CSV = "/data/cerf_out/r4_netload/county_hazard_freq.csv"
OUT_STAMPED = "/data/cerf_out/r4_netload/county_hazard_freq.parquet"
COLS = ["fips", "date", "t2max", "t2min", "thetae_max", "wmax", "q2max", "psfc_min",
        "wmean", "q2mean", "psfc_mean", "pr_sum",
        "wfrac34"]
SC = ["rcp45cooler", "rcp85cooler", "rcp45hotter", "rcp85hotter"]

# The declared climatology exception, see the module docstring. Everything else in this file takes
# its numbers from hazard_defs.
CLIM_Y0_FUTURE_ARM, CLIM_Y1_FUTURE_ARM = 1990, 2010
REPLAY_OFFSET_YEARS = 40
FUT_Y0 = CLIM_Y0_FUTURE_ARM + REPLAY_OFFSET_YEARS
FUT_Y1 = CLIM_Y1_FUTURE_ARM + REPLAY_OFFSET_YEARS
NY_BASELINE = CLIM_Y1_FUTURE_ARM - CLIM_Y0_FUTURE_ARM + 1

# The three agreed hazards this file builds, read from the shared table and never restated here.
HEAT, COLD, FIRE = HD.HAZARDS["heat"], HD.HAZARDS["cold"], HD.HAZARDS["fire"]
# THE COUNTY SCALE EXCEPTION for cold, hazard_defs.COLD_PERSIST_DAYS_COUNTY. A county cold spell
# averages 1.77 days and only 1.5% of them reach six, so the six-day subregion rule leaves 0.96
# cold days a year and no usable county signal. This file is a county build, so it takes three.
COLD_PERSIST_DAYS = HD.COLD_PERSIST_DAYS_COUNTY

# Humid heat and heavy rain are NOT among the seven agreed hazards, so hazard_defs carries no entry
# for them and inventing one here would put a second source of truth in the repository. Their
# constants are named instead, in this one place.
#   Humid heat is the theta-e analogue of the heat rule and deliberately borrows heat's percentile
#   and persistence, so an edit to the heat definition moves it too. It keeps heat's June-to-August
#   gate on evidence rather than on symmetry: the published table, provenance/pre_seedfix/
#   county_hazard_freq.csv outside this release, carries hist "humid heat" at 7.32 days a year,
#   while the ungated build of the same rule beside it carries 14.24. Dropping the gate would
#   therefore change a published column.
HUMID_PCTL = HEAT["pctl"]
HUMID_PERSIST_DAYS = HEAT["persist_days"]
HUMID_MONTHS = HEAT["months"]
#   Heavy rain is a flat annual percentile of wet days, by decision, so it has no day-of-year
#   curve, no persistence rule and no season gate.
RAIN_PCTL = HD.RAIN_PCTL
RAIN_WET_MM = 1.0


def load(tag, lo=None, hi=None):
    d = pd.concat([pd.read_parquet(f, columns=COLS)
                   for f in sorted(glob.glob(f"{CA}/agg_{tag}_*.parquet"))], ignore_index=True)
    d["date"] = pd.to_datetime(d.date)
    if lo is not None:
        d = d[(d.date.dt.year >= lo) & (d.date.dt.year <= hi)]
    return d.sort_values(["fips", "date"]).reset_index(drop=True)


def vpd(t2, q, p):
    """kPa. es from Tetens over water; e from specific humidity and surface pressure."""
    es = 0.6108 * np.exp(17.27 * (t2 - 273.15) / (t2 - 273.15 + 237.3))
    e = q * (p / 1000.0) / (0.622 + 0.378 * q)
    return np.maximum(es - e, 0.0)


def grid(d, col, F, D):
    """(county, day) array"""
    out = np.full((len(F), len(D)), np.nan, dtype=np.float32)
    out[d.fi.values, d.di.values] = d[col].values
    return out


def write_stamped(df, path, hazards, n_units, n_dates, counts, extra):
    """Write `df` to parquet with the hazard_defs stamp in the schema metadata, and return `path`.

    hazard_defs.write_flags is not used here and cannot be: its three table shapes are all flag
    tables, one row per flagged unit-day or a boolean flag column, and this table holds days per
    year per county. The stamp itself is the shared one, hazard_defs.stamp, under the shared
    metadata key, so the definition hashes are the shared hashes and hazard_defs.read_stamp and
    hazard_defs.require_stamp read this file unchanged."""
    import pyarrow as pa
    import pyarrow.parquet as pq
    t = pa.Table.from_pandas(df, preserve_index=False)
    meta = dict(t.schema.metadata or {})
    meta[HD.FLAG_META_KEY] = json.dumps(
        HD.stamp(__file__, hazards, n_units, n_dates, counts, extra), sort_keys=True).encode()
    pq.write_table(t.replace_schema_metadata(meta), path)
    return path


print("climatology frozen on %d-%d, a declared exception to the repository freeze of %d-%d in "
      "hazard_defs; futures restricted to %d-%d"
      % (CLIM_Y0_FUTURE_ARM, CLIM_Y1_FUTURE_ARM, HD.CLIM_Y0, HD.CLIM_Y1, FUT_Y0, FUT_Y1),
      flush=True)

H = load("historical", CLIM_Y0_FUTURE_ARM, CLIM_Y1_FUTURE_ARM)
FIPS = np.array(sorted(H.fips.unique()))
fi = {f: i for i, f in enumerate(FIPS)}
DH = np.array(sorted(H.date.unique()))
di = {d: i for i, d in enumerate(DH)}
H["fi"] = H.fips.map(fi); H["di"] = H.date.map(di)
print("historical %d-%d: %d counties x %d days"
      % (CLIM_Y0_FUTURE_ARM, CLIM_Y1_FUTURE_ARM, len(FIPS), len(DH)), flush=True)

HV = {c: grid(H, c, FIPS, DH) for c in ["t2max", "t2min", "thetae_max"]}
# Same construction as the historical arm: daily-mean wind, daily-mean humidity and daily-mean
# pressure, with the daily maximum temperature. It used to use wmax, q2max and psfc_min, three
# extrema from three different hours.
HV["hdw"] = grid(H, "wmean", FIPS, DH) * vpd(grid(H, "t2max", FIPS, DH),
                                             grid(H, "q2mean", FIPS, DH),
                                             grid(H, "psfc_mean", FIPS, DH))
HV["pr"] = grid(H, "pr_sum", FIPS, DH)
HW34 = grid(H, "wfrac34", FIPS, DH)
del H

# HD.clip_doy is the shared day-of-year rule, np.minimum(dayofyear, 365), and HD.doy_pctl is the
# shared +/-15 day circular-window percentile. No `clim` mask is passed because the frame loaded
# above already IS the frozen window: every day in it lies in 1990-2010.
doy = HD.clip_doy(DH)
THR = {}
for nm, arr, q in [("heat", HV["t2max"], HEAT["pctl"]), ("cold", HV["t2min"], COLD["pctl"]),
                   ("humid", HV["thetae_max"], HUMID_PCTL), ("fire", HV["hdw"], FIRE["pctl"])]:
    THR[nm] = HD.doy_pctl(arr, q, doy)
    print("  threshold %s at p%g done" % (nm, q), flush=True)
# rain is the one flat threshold left: p99 of the county's wet days over the whole 1990-2010
# window, with no day-of-year curve, so it is an annual percentile and not a seasonal anomaly
wet = np.where(HV["pr"] > RAIN_WET_MM, HV["pr"], np.nan)
THR["rain"] = np.nanpercentile(wet, RAIN_PCTL, axis=1)
print("thresholds built", flush=True)


def flags(V, W34, dates, thr):
    """(county, day) boolean flag per hazard. Persistence first, then the season gate, which is the
    order every other builder in this repository uses: a spell that starts in late May may reach
    three days on its May days, and contributes only its June days to the count."""
    dy = HD.clip_doy(dates)
    mon = pd.DatetimeIndex(dates).month.values
    m = {}
    m["heat"] = (HD.persist(V["t2max"] > thr["heat"][:, dy], HEAT["persist_days"])
                 & HD.season(HEAT["months"], mon)[None, :])
    m["cold"] = (HD.persist(V["t2min"] < thr["cold"][:, dy], COLD_PERSIST_DAYS)
                 & HD.season(COLD["months"], mon)[None, :])
    m["humid heat"] = (HD.persist(V["thetae_max"] > thr["humid"][:, dy], HUMID_PERSIST_DAYS)
                       & HD.season(HUMID_MONTHS, mon)[None, :])
    # fire weather takes no persistence and no season gate, and both facts come from the shared
    # table: FIRE["persist_days"] is 1 and FIRE["months"] is None
    m["fire weather"] = (HD.persist(V["hdw"] > thr["fire"][:, dy], FIRE["persist_days"])
                         & HD.season(FIRE["months"], mon)[None, :])
    m["heavy rain"] = V["pr"] > thr["rain"][:, None]
    m["tropical cyclone"] = W34 > 0
    return m


def count(V, W34, dates, thr):
    """days per year of each hazard per county, and the flagged county-day total per hazard"""
    ny = len(np.unique(pd.DatetimeIndex(dates).year))
    m = flags(V, W34, dates, thr)
    return ({k: v.sum(1) / ny for k, v in m.items()},
            {k: int(v.sum()) for k, v in m.items()})


HC, HN = count(HV, HW34, DH, THR)
del HV, HW34
print("historical days/yr, county mean: %s"
      % {k: round(float(np.nanmean(v)), 2) for k, v in HC.items()}, flush=True)

FC = {}
for s in SC:
    F = load(s, FUT_Y0, FUT_Y1)
    DF = np.array(sorted(F.date.unique()))
    yrs = sorted(pd.DatetimeIndex(DF).year.unique())
    if len(yrs) != NY_BASELINE:
        print("  WARNING %s: %d years loaded (%d-%d); the replay pairs %d baseline years with "
              "%d-%d" % (s, len(yrs), yrs[0], yrs[-1], NY_BASELINE, FUT_Y0, FUT_Y1), flush=True)
    dfi = {d: i for i, d in enumerate(DF)}
    F["fi"] = F.fips.map(fi); F["di"] = F.date.map(dfi)
    F = F[F.fi.notna()]
    F["fi"] = F.fi.astype(int)
    V = {c: grid(F, c, FIPS, DF) for c in ["t2max", "t2min", "thetae_max"]}
    # SAME construction as the threshold block above and as the historical arm: daily-mean wind,
    # humidity and pressure with the daily maximum temperature. This loop used to score future days
    # with wmax/q2max/psfc_min while the thresholds were fitted on the mean-based index, so the two
    # were not even the same quantity inside this one script.
    V["hdw"] = grid(F, "wmean", FIPS, DF) * vpd(grid(F, "t2max", FIPS, DF),
                                                grid(F, "q2mean", FIPS, DF),
                                                grid(F, "psfc_mean", FIPS, DF))
    V["pr"] = grid(F, "pr_sum", FIPS, DF)
    W = grid(F, "wfrac34", FIPS, DF)
    FC[s], _ = count(V, W, DF, THR)
    del F, V, W
    print("%-12s %s" % (s, {k: round(float(np.nanmean(v)), 2) for k, v in FC[s].items()}),
          flush=True)

rows = []
for i, f in enumerate(FIPS):
    r = {"fips": f}
    for k in HC:
        r["hist_" + k] = HC[k][i]
        for s in SC:
            r["%s_%s" % (s, k)] = FC[s][k][i]
    rows.append(r)
D = pd.DataFrame(rows)
D.to_csv(OUT_CSV, index=False)
# the refusable twin: same table, hazard_defs stamp in the parquet schema metadata. The counts are
# the baseline ones, because the definition hash they certify is the one the thresholds were built
# with, and the thresholds are built on the baseline.
write_stamped(D, OUT_STAMPED,
              hazards=["heat", "cold", "fire"],
              n_units=len(FIPS), n_dates=len(DH),
              counts={"heat": HN["heat"], "cold": HN["cold"], "fire": HN["fire weather"]},
              extra={HD.PENDING_KEY: ["06_fixtc.py"],
                     "cold_persist_days": COLD_PERSIST_DAYS,
                     "clim_y0": CLIM_Y0_FUTURE_ARM, "clim_y1": CLIM_Y1_FUTURE_ARM,
                     "clim_exception": "the future arm freezes on %d-%d, not the repository's "
                                       "%d-%d, because the TGW replay pairs %d-%d with the "
                                       "baseline" % (CLIM_Y0_FUTURE_ARM, CLIM_Y1_FUTURE_ARM,
                                                     HD.CLIM_Y0, HD.CLIM_Y1, FUT_Y0, FUT_Y1),
                     "future_years": [FUT_Y0, FUT_Y1],
                     "counts_are": "flagged county-days in the baseline window, not in the futures",
                     "table": "days per year per county, hist_<hazard> and <scenario>_<hazard>",
                     "columns_outside_the_shared_table": [
                         "humid heat", "heavy rain",
                         "tropical cyclone (34 kt wfrac34 here; 06_fixtc.py replaces it)"]})
print("\nnational county-mean change, days per year:")
for k in HC:
    h = float(np.nanmean(HC[k]))
    fu = [float(np.nanmean(FC[s][k])) for s in SC]
    print("  %-17s hist %6.2f   fut %s   change %+.2f to %+.2f"
          % (k, h, " ".join("%6.2f" % v for v in fu), min(fu) - h, max(fu) - h))
print("wrote %s and the stamped twin %s"
      % (os.path.basename(OUT_CSV), os.path.basename(OUT_STAMPED)))
