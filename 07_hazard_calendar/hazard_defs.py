"""
The single source of truth for every hazard definition in this repository.

WHY THIS FILE EXISTS. Until today the same hazard was built five times: once for the 18
subregions, once for counties on TGW, once for counties on CONUS404, once again in the merge
script, and once more in the future arm. Most of those copies carried a private helper and a typed
percentile literal, and they drifted. The inventory, checked against the 2026-08-18 baseline:
11_county_hazard_flags.py, 04_c404_pipeline.py and 07_merge_c404_county.py each carried a private `doy_pctl`
and a private `persist`; 06_hazard_heat.py, 07_hazard_cold.py, 16_harden_hazards.py and
11_item6_obs_validation.py each carried a private `doy_pct` and a private `persist`; 06_ar_variants.py
carried a private `doy_pct`; 05_hazfreq.py carried a private `runsK` and took whole-record percentiles
with no day-of-year helper at all. The fire-weather flag drifted furthest: a compound percentile
triple (q < p25 and wspd > p75 and tmax > p75) was running in the county builders under the name
"fire" while 08_hazard_fire.py and 05_hazfreq.py used the Hot-Dry-Windy index on a whole-record top
decile. Two different objects shared one name, and nothing in the chain could tell them apart.

So: the constants, the day-of-year percentile, the persistence rule and the TGW descaling all live
here, once. Every builder is to import them, and every flag file written through write_flags
carries a hash of the constants it used, so a consumer can refuse a file written by a superseded
builder. The conversion of the builders was under way when this module was written, so do not read
the paragraph above as a claim that it is complete: a builder that still carries a private copy of
a constant will not be caught by this module, only by reading it.

THE SEVEN HAZARDS, exactly as agreed.

  heat wave        tmax above its own day-of-year 90th percentile, +/-15 day window,
                   at least 3 consecutive days, June to August.
                   Perkins & Alexander 2013.
  cold outbreak    tmin below its own day-of-year 10th percentile, +/-15 day window, December to
                   February, and a run of consecutive days whose required length depends on scale.
                   ETCCDI cold-spell duration index, Zhang et al. 2011.
                   SCALE EXCEPTION, binding on every county build: the persistence is
                   COLD_PERSIST_DAYS = 6 days at SUBREGION scale and COLD_PERSIST_DAYS_COUNTY = 3
                   days at COUNTY scale. There is no six-day county build in the specification,
                   historical or future, so 11_county_hazard_flags.py, 04_c404_pipeline.py and 05_hazfreq.py
                   all take 3. The reason is that a county cold spell averages 1.77 days and only
                   1.5% of them reach six, so the six-day rule leaves 0.96 cold days a year and no
                   usable county signal. Everything else about cold is identical at both scales.
                   The definition hash cannot tell the two scales apart, so a county builder MUST
                   record extra={"cold_persist_days": 3} through write_flags.
  fire weather     HDW = VPD x 10 m wind speed, above the unit's own day-of-year 99th percentile,
                   +/-15 day window, no persistence rule and no season gate.
                   Srock et al. 2018 for the index, Abatzoglou et al. 2019 for the local-percentile
                   fire-weather-day construction.
  renewable        combined wind and solar output below 50% of its own day-of-year climatology.
    drought        Rinaldi et al. 2021. This is an OUTPUT rule. It is not a compound percentile on
                   wind speed and irradiance.
  atmospheric      subregion daily mean IVT above its own +/-15 day day-of-year 95th percentile,
    river          AND a catalog river covering at least 25% of the subregion. This is the adopted
                   ivt_p95_cov25 flag. The NCEP absolute-250 NDJFM WECC-only construction is
                   superseded and must not be rebuilt.
  tropical         subregion: a HURDAT2 track point within 500 km of the subregion centroid.
    cyclone        county: the county centroid inside the four-quadrant 34 kt wind radii.
                   HURDAT2 -999 wind records are dropped at both scales.
  severe           a cell-hour at or above 50 dBZ, a daily maximum MUCAPE at or above 1000 J/kg,
    convection     AND at least 20% of the county's cells above the reflectivity threshold in that
                   hour. All three conditions belong in the BUILDER, never in the consumer.

SHARED RULES, honored by every helper below.
  - Day-of-year percentiles use np.nanpercentile, a +/-15 day circular window, day-of-year clipped
    with np.minimum(doy, 365), and the window arithmetic |((doy - d + 182) % 365) - 182| <= 15.
  - Thresholds are frozen on the stated climatology period, 1980 to 2019, and are never estimated
    on the full record when the record runs past it.
  - For heat and cold the persistence rule runs FIRST and the season gate SECOND. The order is
    part of the definition, not a convention in a comment: it is recorded as
    "order": SPELL_ORDER in both hazard entries, so it is inside the hashed payload, and spell()
    implements it once. Gating first would delete every heat spell straddling 31 May and every
    cold spell straddling 30 November.
  - No behavior in this module, or in anything that imports it, may be controlled by an
    environment variable.
"""
import datetime as _dt
import hashlib as _hashlib
import json as _json
import os as _os
import re as _re

import numpy as np

VERSION = "2026-08-18"

# ---------------------------------------------------------------- shared construction constants
DOY_WINDOW = 15                 # +/- days in the circular day-of-year window
DOY_CLIP = 365                  # np.minimum(dayofyear, DOY_CLIP); only DOY 366 is folded onto 365
DOY_MODULUS = 365               # the modulus in the window arithmetic
DOY_HALF = 182                  # the offset in the window arithmetic
PCTL_FUNC = "np.nanpercentile"  # named so the hash changes if the estimator ever changes
SPELL_ORDER = "persist_then_season"   # the run-length rule first, the season gate second; spell()
CLIM_Y0, CLIM_Y1 = 1980, 2019   # the frozen climatology period, inclusive

# ---------------------------------------------------------------- per-hazard constants
# Heat, Perkins & Alexander 2013
HEAT_VAR = "tmax"
HEAT_PCTL = 90
HEAT_SIDE = "above"
HEAT_PERSIST_DAYS = 3
HEAT_MONTHS = (6, 7, 8)

# Cold, ETCCDI cold-spell duration index, Zhang et al. 2011
COLD_VAR = "tmin"
COLD_PCTL = 10
COLD_SIDE = "below"
COLD_PERSIST_DAYS = 6           # SUBREGION scale only
COLD_PERSIST_DAYS_COUNTY = 3    # COUNTY scale, every county build; see the module docstring
COLD_MONTHS = (12, 1, 2)

# Fire weather, HDW of Srock et al. 2018 on the local percentile of Abatzoglou et al. 2019
FIRE_VAR = "hdw"
FIRE_PCTL = 99
FIRE_SIDE = "above"
FIRE_PERSIST_DAYS = 1           # 1 means no persistence rule
FIRE_MONTHS = None              # None means no season gate

# Renewable drought, Rinaldi et al. 2021
VRE_VAR = "wind_plus_solar"
VRE_FRACTION = 0.50             # flagged below this fraction of the day-of-year climatology
VRE_PERSIST_DAYS = 1
VRE_MONTHS = None

# Atmospheric river, the adopted ivt_p95_cov25 flag
AR_VAR = "ivt_mean"
RAIN_PCTL = 99                  # heavy rain, the county's own percentile of WET days (> 1 mm)
RAIN_WET_MM = 1.0
AR_PCTL = 95
AR_SIDE = "above"
AR_COVERAGE_FRACTION = 0.25     # catalog river over at least this share of the subregion
AR_PERSIST_DAYS = 1
AR_MONTHS = None

# Tropical cyclone, HURDAT2
TC_RADIUS_KM = 500.0            # subregion scale, centroid to track point
TC_WIND_KT = 34.0               # county scale, the four-quadrant 34 kt wind radii
TC_MISSING_WIND = -999          # dropped at both scales, never treated as a valid intensity

# Severe convection, CONUS404
CONV_REFL_DBZ = 50.0
CONV_CAPE_JKG = 1000.0
CONV_CELL_FRACTION = 0.20       # share of the county's cells above the reflectivity threshold

# ---------------------------------------------------------------- the machine-readable table
# Everything the definition hash is taken over. Add a key here and every stamp changes, which is
# the intended behavior: a constant that moves must invalidate the files built before it moved.
SHARED = {
    "doy_window": DOY_WINDOW,
    "doy_clip": DOY_CLIP,
    "doy_modulus": DOY_MODULUS,
    "doy_half": DOY_HALF,
    "pctl_func": PCTL_FUNC,
    "clim_y0": CLIM_Y0,
    "clim_y1": CLIM_Y1,
}

HAZARDS = {
    "heat": {
        "var": HEAT_VAR, "pctl": HEAT_PCTL, "side": HEAT_SIDE,
        "persist_days": HEAT_PERSIST_DAYS, "months": list(HEAT_MONTHS),
        "order": SPELL_ORDER,
        "reference": "Perkins & Alexander 2013",
    },
    "cold": {
        "var": COLD_VAR, "pctl": COLD_PCTL, "side": COLD_SIDE,
        "persist_days": COLD_PERSIST_DAYS, "persist_days_county": COLD_PERSIST_DAYS_COUNTY,
        "months": list(COLD_MONTHS),
        "order": SPELL_ORDER,
        "reference": "Zhang et al. 2011, ETCCDI cold-spell duration index",
    },
    "fire": {
        "var": FIRE_VAR, "pctl": FIRE_PCTL, "side": FIRE_SIDE,
        "persist_days": FIRE_PERSIST_DAYS, "months": FIRE_MONTHS,
        "reference": "Srock et al. 2018; Abatzoglou et al. 2019",
    },
    "vre_drought": {
        "var": VRE_VAR, "fraction_of_climatology": VRE_FRACTION,
        "persist_days": VRE_PERSIST_DAYS, "months": VRE_MONTHS,
        "reference": "Rinaldi et al. 2021",
    },
    "ar": {
        "var": AR_VAR, "pctl": AR_PCTL, "side": AR_SIDE,
        "coverage_fraction": AR_COVERAGE_FRACTION,
        "persist_days": AR_PERSIST_DAYS, "months": AR_MONTHS,
        "reference": "adopted ivt_p95_cov25; Ralph et al. 2019 for the IVT scale",
    },
    "tc": {
        "radius_km": TC_RADIUS_KM, "wind_kt": TC_WIND_KT,
        "missing_wind": TC_MISSING_WIND,
        "reference": "HURDAT2",
    },
    "convection": {
        "refl_dbz": CONV_REFL_DBZ, "cape_jkg": CONV_CAPE_JKG,
        "cell_fraction": CONV_CELL_FRACTION,
        "reference": "CONUS404 REFL_COM and MUCAPE",
    },
}

# ---------------------------------------------------------------- expected day rates
# (expected fraction of unit-days, tolerance, basis). Each builder self-checks that
# |measured - expected| <= tolerance. Read the basis before quoting any of these: some are
# measurements this project has recorded, and some are bounds that follow from the definition and
# are deliberately loose. A bound is not a result.
MEASURED, BOUND, EXACT = "measured", "bound", "exact-by-construction"

EXPECTED_DAY_RATE = {
    # ---- 18 subregions, 14,610 days, 1980 to 2019
    "subregion.heat": (0.0126, 0.0126, BOUND + ": at most 10% of the 92 June-to-August days, so "
                       "at most 0.0252 of all days; the superseded p95 June-to-August flag "
                       "measured 0.01236"),
    "subregion.cold": (0.0123, 0.0123, BOUND + ": at most 10% of the 90 December-to-February days, "
                       "so at most 0.0247 of all days; the superseded p5 December-to-February flag "
                       "measured 0.01241"),
    "subregion.fire": (0.0100, 0.0030, EXACT + ": a day-of-year 99th percentile with no persistence "
                       "rule and no season gate fires on close to 1% of days when the climatology "
                       "period and the scored record are the same years; a seasonal cycle inside "
                       "the +/-15 day window pulls the realized rate down to about 0.87%, which the "
                       "tolerance covers"),
    "subregion.vre_drought": (0.0981, 0.0200, MEASURED + ": 35.8 subregion drought days a year, "
                              "RECOMPUTE_2026-08-13.md, divided by 365"),
    "subregion.ar": (0.0453, 0.0100, MEASURED + ": ivt_p95_cov25 fires on 4.53% of subregion-days, "
                     "R1_CLIMATE_POWER_CAUSAL.md"),
    "subregion.tc": (0.0038, 0.0015, MEASURED + ": 996 of 262,980 subregion-days have a track point "
                     "within 500 km of the subregion centroid, R1_CLIMATE_POWER_CAUSAL.md"),
    # ---- counties
    "county.heat": (0.0126, 0.0126, BOUND + ": the same June-to-August bound as the subregion flag"),
    # The agreed county rule is three days, so county.cold IS the three-day band. The six-day
    # measurement is kept only under county.cold_6day, to name what a superseded file looks like.
    "county.cold": (0.0123, 0.0123, BOUND + ": the agreed three-day county rule, at most 10% of the "
                    "90 December-to-February days; never measured, so the band is the bound"),
    "county.cold_6day": (0.0026, 0.0015, MEASURED + ": the SUPERSEDED six-day rule at county scale "
                         "left 0.96 county cold days a year, recorded in the docstring of "
                         "11_county_hazard_flags.py, divided by 365. A file scored against this key "
                         "was built against a rule the specification does not contain."),
    "county.fire": (0.0100, 0.0030, EXACT + ": as the subregion fire flag"),
    "county.tc": (0.0050, 0.0050, BOUND + ": never measured; a county centroid cannot sit inside a "
                  "34 kt wind field on more than about 1% of days over 1980 to 2019"),
    "county.convection": (0.0122, 0.0030, MEASURED + ": the three-condition flag fires on 1.22% of "
                          "county-days, 09_outage_attribution/10_conv_eventstudy.py"),
}


def expected_day_rate(scale, hazard, variant=None):
    """Look up (expected, tolerance, basis). `variant` selects a documented scale exception,
    for example expected_day_rate('county', 'cold', '3day')."""
    key = "%s.%s" % (scale, hazard) + ("_%s" % variant if variant else "")
    if key not in EXPECTED_DAY_RATE:
        raise KeyError("no recorded expectation for %s; add one to EXPECTED_DAY_RATE" % key)
    return EXPECTED_DAY_RATE[key]


def day_rate_ok(scale, hazard, rate, variant=None):
    """True when `rate` sits inside the recorded band. Returns (ok, expected, tol, basis)."""
    exp, tol, basis = expected_day_rate(scale, hazard, variant)
    return (abs(float(rate) - exp) <= tol), exp, tol, basis


# ---------------------------------------------------------------- shared helpers
def clip_doy(dates):
    """Day-of-year clipped at 365, the project convention.

    Only day 366 is folded onto 365. In a leap year 29 February is day 60 and 1 March is day 61,
    so the day-of-year index shifts by one relative to a common year from 1 March onward. The
    +/-15 day window absorbs that shift, and no threshold is ever fitted on ten days of data
    because day 366 is never its own bin. An earlier version of this docstring said 29 February
    shares 28 February's threshold, which is not what the code does."""
    import pandas as pd
    return np.minimum(pd.DatetimeIndex(dates).dayofyear.values, DOY_CLIP)


def doy_window(doy, d, window=DOY_WINDOW):
    """The circular +/-`window` day mask around day-of-year `d`, |((doy - d + 182) % 365) - 182|."""
    return np.abs(((np.asarray(doy) - d + DOY_HALF) % DOY_MODULUS) - DOY_HALF) <= window


def doy_pctl(x, p, doy, window=DOY_WINDOW, clim=None):
    """Day-of-year percentile of x, one row per unit, one column per day-of-year 0 to 365.

    x     (n_units, n_days) float, NaN allowed
    p     the percentile, for example 90
    doy   (n_days,) day-of-year already clipped at 365, see clip_doy
    clim  optional (n_days,) boolean selecting the frozen climatology period. When the record runs
          past CLIM_Y1 this MUST be passed, otherwise the threshold is fitted on days it is then
          used to score and a trend can be manufactured.

    Column 0 repeats column 1 so that `out[:, doy]` is safe for any clipped day-of-year.
    """
    x = np.asarray(x)
    doy = np.asarray(doy)
    out = np.empty((x.shape[0], 366), dtype=np.float32)
    for d in range(1, 366):
        sel = doy_window(doy, d, window)
        if clim is not None:
            sel = sel & np.asarray(clim, bool)
        out[:, d] = np.nanpercentile(x[:, sel], p, axis=1)
    out[:, 0] = out[:, 1]
    return out


def persist(mask, k):
    """Keep only runs of at least k consecutive True along the time axis, and keep the whole run.

    k = 1 returns the mask unchanged, which is how the hazards with no persistence rule are
    expressed without a special case at the call site."""
    mask = np.asarray(mask, bool)
    if mask.ndim == 1:                      # a single unit; iterate rows below expects 2-D
        return persist(mask[None, :], k)[0]
    if mask.ndim != 2:
        raise ValueError("persist expects a 1-D or 2-D mask, got %d dimensions" % mask.ndim)
    if k <= 1:
        return mask.copy()
    out = np.zeros_like(mask)
    for i in range(mask.shape[0]):
        m = mask[i]
        if not m.any():
            continue
        d = np.diff(np.r_[0, m.view(np.int8), 0])
        s, e = np.where(d == 1)[0], np.where(d == -1)[0]
        for a, b in zip(s, e):
            if b - a >= k:
                out[i, a:b] = True
    return out


def season(months, month_of_day):
    """(n_days,) boolean season gate. `months` None means no gate, so every day passes."""
    month_of_day = np.asarray(month_of_day)
    if months is None:
        return np.ones(month_of_day.shape, bool)
    return np.isin(month_of_day, list(months))


def spell(mask, k, months, month_of_day):
    """The one sanctioned composition of the persistence rule and the season gate, in that order.

    mask          (n_units, n_days) or (n_days,) boolean, the threshold exceedance
    k             the run length, HEAT_PERSIST_DAYS or COLD_PERSIST_DAYS / COLD_PERSIST_DAYS_COUNTY
    months        the season gate, None for no gate
    month_of_day  (n_days,) the calendar month of each day

    SPELL_ORDER is "persist_then_season" and it is part of the definition, hashed with the
    constants: a spell is scored on its own length first and only then restricted to the season.
    Gating first instead would delete every heat spell straddling 31 May and every cold spell
    straddling 30 November, and would produce a materially different flag under the same name. Use
    this helper rather than calling persist() and season() at the call site, so the order cannot
    drift back into a comment."""
    mask = np.asarray(mask, bool)
    one_d = mask.ndim == 1
    m = persist(mask[None, :] if one_d else mask, k)
    g = season(months, month_of_day)
    out = m & g[None, :]
    return out[0] if one_d else out


def descale(data, scale, axis=1):
    """Restore physical units from a packed TGW or CONUS404 store: DIVIDE by the 'scale' array.

    01_acquire_data/01_process_tgw.py multiplies by SCALE before the float16 cast, because PSFC in Pa is
    about 1e5 and float16 tops out at 65,504. SCALE is therefore [1, 1, 1, 0.01, 1, 1, 1] and the
    stored PSFC is hPa. Every reader in this repository divides. Multiplying instead leaves PSFC a
    further factor of 100 too small, which silently breaks VPD, HDW and anything built on them."""
    sc = np.asarray(scale, dtype="float32")
    shp = [1] * np.ndim(data)
    shp[axis] = sc.size
    return np.asarray(data, dtype="float32") / sc.reshape(shp)


# ---------------------------------------------------------------- provenance stamps
FLAG_META_KEY = b"hazard_defs"


def _canonical(hazard):
    return {"hazard": hazard, "definition": HAZARDS[hazard], "shared": SHARED, "version": VERSION}


def definition_hash(hazard):
    """First 12 hex digits of the SHA-256 of this hazard's constants plus the shared rules. Two
    builders that produce the same hash used the same numbers."""
    blob = _json.dumps(_canonical(hazard), sort_keys=True, separators=(",", ":")).encode()
    return _hashlib.sha256(blob).hexdigest()[:12]


PENDING_KEY = "pending_post_steps"


def require_complete(path, reader=None):
    """Refuse a flag file whose producer declared a post-step that has not run.

    Two scripts in this study write the same table in sequence: the producer leaves a row that a
    second script must replace. Before this guard, running them in the natural order left the
    wrong row in place and nothing said so. The producer now records what still has to run, and
    the second script clears it."""
    st = read_stamp(path) if reader is None else reader(path)
    if st is None:
        raise RuntimeError("%s carries no stamp, so its completeness cannot be checked" % path)
    pend = (st.get("extra") or {}).get(PENDING_KEY) or []
    if pend:
        raise RuntimeError("%s is incomplete: %s must run first" % (path, ", ".join(pend)))
    return st


def stamp(script, hazards, n_units, n_dates, counts, extra=None):
    """The dictionary written into a flag file. `counts` maps hazard -> flagged unit-days."""
    n_units, n_dates = int(n_units), int(n_dates)
    denom = float(n_units * n_dates)
    return {
        "hazard_defs_version": VERSION,
        "script": _os.path.basename(str(script)),
        "written_utc": _dt.datetime.utcnow().strftime("%Y-%m-%dT%H:%M:%SZ"),
        "n_units": n_units,
        "n_dates": n_dates,
        "hazards": sorted(hazards),
        "definition_hash": {h: definition_hash(h) for h in sorted(hazards)},
        "constants": {h: HAZARDS[h] for h in sorted(hazards)},
        "shared": SHARED,
        "counts": {h: int(counts[h]) for h in sorted(hazards)},
        "day_rate": {h: (int(counts[h]) / denom if denom else float("nan"))
                     for h in sorted(hazards)},
        "extra": extra or {},
    }


def write_flags(df, path, script, n_units, n_dates, hazards=None, hazard_col="hazard",
                flag_col=None, extra=None):
    """Write a flag table with the stamp inside the parquet, and return the path.

    Three table shapes are supported, and exactly one applies per file.
      hazard_col   a long table, one row per flagged unit-day, with a column naming the hazard.
                   This is the county_hazard_flags shape.
      flag_col     a dense table, one row per unit-day, with a boolean column that is the flag.
                   This is the county_convective_daily shape.
      neither      a long table for a single hazard, named in `hazards`.

    `extra` records anything a consumer needs that is not a constant, and county builders MUST
    record extra={"cold_persist_days": N}, because COLD_PERSIST_DAYS and COLD_PERSIST_DAYS_COUNTY
    sit in the same hazard entry and produce the same hash. The agreed county value is 3; the gate
    fails a county flag file whose stamp omits the key or records anything else. The stamp travels in the parquet schema metadata, not in a
    sidecar, so it cannot be separated from the data. Read it back with read_stamp, and refuse a
    stale file with require_stamp."""
    import pandas as pd  # noqa: F401  (kept explicit so the dependency is visible)
    import pyarrow as pa
    import pyarrow.parquet as pq

    if flag_col is not None:
        if not hazards or len(hazards) != 1:
            raise ValueError("a table flagged by a boolean column must name exactly one hazard")
        hz = list(hazards)
        counts = {hz[0]: int(np.asarray(df[flag_col], bool).sum())}
    elif hazard_col is not None and hazard_col in df.columns:
        counts = df[hazard_col].value_counts().to_dict()
        hz = sorted(hazards) if hazards else sorted(counts)
        counts = {h: int(counts.get(h, 0)) for h in hz}
    else:
        if not hazards or len(hazards) != 1:
            raise ValueError("a table without a hazard column must name exactly one hazard")
        hz = list(hazards)
        counts = {hz[0]: int(len(df))}
    unknown = [h for h in hz if h not in HAZARDS]
    if unknown:
        raise ValueError("unknown hazard name %s; the seven names are %s"
                         % (unknown, sorted(HAZARDS)))
    t = pa.Table.from_pandas(df, preserve_index=False)
    meta = dict(t.schema.metadata or {})
    meta[FLAG_META_KEY] = _json.dumps(
        stamp(script, hz, n_units, n_dates, counts, extra), sort_keys=True).encode()
    pq.write_table(t.replace_schema_metadata(meta), path)
    return path


def rewrite_stamp(path, st):
    """Replace the stamp on an existing parquet without touching its rows."""
    import pyarrow.parquet as _pq, json as _js
    tb = _pq.read_table(path)
    meta = dict(tb.schema.metadata or {})
    meta[FLAG_META_KEY] = _js.dumps(st, sort_keys=True).encode()
    _pq.write_table(tb.replace_schema_metadata(meta), path)
    return path


def read_stamp(path):
    """The stamp dictionary, or None when the file carries none."""
    import pyarrow.parquet as pq
    md = pq.read_schema(path).metadata or {}
    raw = md.get(FLAG_META_KEY)
    return _json.loads(raw.decode()) if raw else None


def script_name(path):
    """Basename of a script with any leading pipeline number removed.

    The released tree numbers each script by its position in the stage, while the working tree is
    flat and unnumbered. A stamp written under one layout must still be recognized under the
    other, so every comparison of a stamped script name goes through this."""
    return _re.sub(r"^\d+_", "", _os.path.basename(str(path)))


def require_stamp(path, hazards=None, script=None):
    """Raise unless `path` was written by a builder using exactly these constants.

    This is the refusal a consumer needs. A file with no stamp predates the shared module. A file
    whose hash differs was built from different numbers, whatever its hazard column says."""
    s = read_stamp(path)
    if s is None:
        raise ValueError("%s carries no hazard_defs stamp: it predates the shared definitions and "
                         "must be rebuilt before it is read" % path)
    if script is not None and script_name(s.get("script")) != script_name(script):
        raise ValueError("%s was written by %s, not by %s"
                         % (path, s.get("script"), _os.path.basename(script)))
    for h in (hazards or s.get("hazards", [])):
        want = definition_hash(h)
        got = (s.get("definition_hash") or {}).get(h)
        if got != want:
            raise ValueError("%s holds hazard '%s' at definition %s, but the current definition is "
                             "%s: the file was written by a superseded builder"
                             % (path, h, got, want))
    return s


if __name__ == "__main__":
    print("hazard_defs %s" % VERSION)
    for h in sorted(HAZARDS):
        print("  %-12s %s  %s" % (h, definition_hash(h),
                                  _json.dumps(HAZARDS[h], sort_keys=True)))
    print("shared: %s" % _json.dumps(SHARED, sort_keys=True))
    print("\nexpected day rates (expected, tolerance, basis):")
    for k in sorted(EXPECTED_DAY_RATE):
        e, t, b = EXPECTED_DAY_RATE[k]
        print("  %-24s %.4f +/- %.4f   %s" % (k, e, t, b.split(":")[0]))
