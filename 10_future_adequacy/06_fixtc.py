"""The 34 kt row was counting every windy day, not tropical cyclones.

`wfrac34 > 0` in the county aggregates is "some part of this county saw 34 kt on this day", whatever
caused it, which gives 1.51 days a year against the 0.13 that the HURDAT2-anchored detection gives
for the same window. That row is replaced by the vetted TC county-days that Figure 6 is built on, so
the two figures now count the same storms. The honest version of what the aggregate measured, any
day reaching 34 kt and mostly extratropical, is kept as its own row, because it is a real
grid-relevant hazard and nothing else in the figure carries it.

RUNNING IT TWICE IS SAFE NOW. The rename is guarded on the column names: when `hist_high wind` is
already present the rename is skipped, and the tropical cyclone columns are simply rebuilt from
`tc_flags.parquet`, which gives the same numbers. Before the guard a second run renamed the
freshly written tropical cyclone columns into `high wind` as well, leaving two columns literally
named `hist_high wind`, wrote that duplicate-named CSV to disk, and only then raised TypeError on
the summary print.

WHAT IT REFUSES. `05_hazfreq.py` writes a stamped parquet twin of the CSV. This script reads that
stamp, and it stops if the twin is missing or if its heat, cold and fire definition hashes are not
the current ones in hazard_defs. A CSV built before the shared definitions carries the superseded
fire rule, and no rename can repair that.
"""
import json
import os
import sys

import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

# hazard_defs.py sits in 07_hazard_calendar/ in the repository and beside this script on the
# deployment box, where every script lives flat in /data. Both go on the path, repository first.
_HERE = os.path.dirname(os.path.abspath(__file__))
for _p in (_HERE, os.path.join(_HERE, os.pardir, "07_hazard_calendar")):
    if os.path.isdir(_p) and _p not in sys.path:
        sys.path.insert(0, _p)
import hazard_defs as HD

P = "/data/cerf_out/r4_netload/county_hazard_freq.csv"
TWIN = "/data/cerf_out/r4_netload/county_hazard_freq.parquet"
TCF = "/data/scratch_r5/tc_flags.parquet"
SC = ["rcp45cooler", "rcp85cooler", "rcp45hotter", "rcp85hotter"]

# the refusal: the table this script edits must have been built by the current definitions
if not os.path.exists(TWIN):
    raise ValueError("%s carries no stamped twin at %s: it was written before the shared hazard "
                     "definitions and must be rebuilt with 05_hazfreq.py before this script edits it"
                     % (P, TWIN))
ST = HD.require_stamp(TWIN, ["heat", "cold", "fire"])
if HD.script_name(ST.get("script")) not in ("hazfreq.py", "fixtc.py"):
    raise ValueError("%s was written by %s; this script edits the 05_hazfreq.py table only"
                     % (TWIN, ST.get("script")))
print("editing the table stamped by %s, hazard_defs %s"
      % (ST.get("script"), ST.get("hazard_defs_version")), flush=True)

D = pd.read_csv(P, dtype={"fips": str})
D["fips"] = D.fips.str.zfill(5)
if "hist_high wind" in D.columns:
    # already renamed by an earlier run; renaming again would move the HURDAT2 numbers into the
    # high wind columns and leave two columns with the same name
    print("high wind columns already present: skipping the rename, rebuilding tropical cyclone",
          flush=True)
else:
    D = D.rename(columns={"hist_tropical cyclone": "hist_high wind",
                          **{"%s_tropical cyclone" % s: "%s_high wind" % s for s in SC}})

F = pd.read_parquet(TCF)
F["date"] = pd.to_datetime(F.date)
NY = {"historical": F[F.scen == "historical"].date.dt.year.nunique()}
for s in SC:
    NY[s] = F[F.scen == s].date.dt.year.nunique()
print("years per scenario: %s" % NY)
for tag, col in [("historical", "hist_tropical cyclone")] + [(s, "%s_tropical cyclone" % s)
                                                             for s in SC]:
    c = F[F.scen == tag].groupby("fips").size() / NY[tag]
    D[col] = D.fips.map(c).fillna(0.0)
D.to_csv(P, index=False)
# The producer declares this script as a pending post-step; clear it on the stamped twin so a
# consumer that calls hazard_defs.require_complete can tell a finished table from a raw one.
try:
    _tw = P.replace(".csv", ".parquet")
    _st = HD.read_stamp(_tw)
    if _st is not None:
        _st.setdefault("extra", {})[HD.PENDING_KEY] = []
        HD.rewrite_stamp(_tw, _st)
        print("  cleared the pending post-step on %s" % os.path.basename(_tw), flush=True)
except Exception as _e:
    print("  WARNING could not clear the pending post-step: %s" % _e, flush=True)

# rewrite the stamped twin so the twin and the CSV never disagree. The heat, cold and fire columns
# are untouched here, so their counts, hashes and unit-day totals are carried through from the
# 05_hazfreq.py stamp unchanged. The tropical cyclone columns come from tc_flags.parquet, which this
# script does not build and cannot verify, so they are recorded in `extra` and NOT as a stamped
# hazard: nothing here certifies that tc_flags.parquet used hazard_defs.TC_WIND_KT or dropped
# hazard_defs.TC_MISSING_WIND.
_t = pa.Table.from_pandas(D, preserve_index=False)
_meta = dict(_t.schema.metadata or {})
_extra = dict(ST.get("extra") or {})
_extra.update({
    "tc_source": TCF,
    "tc_provenance_verified_here": False,
    "tc_note": "the tropical cyclone columns are county-day rates from tc_flags.parquet; the 34 kt "
               "wfrac34 counts 05_hazfreq.py wrote under that name are kept as high wind",
    "tc_years_per_scenario": {k: int(v) for k, v in NY.items()},
    "columns_outside_the_shared_table": ["humid heat", "heavy rain", "high wind (34 kt wfrac34)",
                                         "tropical cyclone (tc_flags.parquet)"],
})
_meta[HD.FLAG_META_KEY] = json.dumps(
    HD.stamp(__file__, ["heat", "cold", "fire"], ST["n_units"], ST["n_dates"],
             {h: int(ST["counts"][h]) for h in ("heat", "cold", "fire")}, _extra),
    sort_keys=True).encode()
pq.write_table(_t.replace_schema_metadata(_meta), TWIN)

H = ["heat", "humid heat", "cold", "fire weather", "heavy rain", "high wind", "tropical cyclone"]
print("\n%-18s %8s  %s" % ("hazard", "hist", "  ".join("%12s" % s for s in SC)))
for k in H:
    print("%-18s %8.2f  %s" % (k, D["hist_" + k].mean(),
                               "  ".join("%12.2f" % D["%s_%s" % (s, k)].mean() for s in SC)))
print("\nTC county-days per year, national total: hist %.0f  hottest %.0f"
      % (D["hist_tropical cyclone"].sum(), D["rcp85hotter_tropical cyclone"].sum()))
