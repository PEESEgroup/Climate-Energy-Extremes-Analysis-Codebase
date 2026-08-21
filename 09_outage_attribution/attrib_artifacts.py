"""One place that names the hazard screen's artifacts, so no consumer spells them out.

WHY THE NAMES CHANGED. 04_chk3.py applies a pre-event (placebo) test and a precision test. Its own
log lines, and the label_note it writes, say that passing that screen is NOT identification: a
pre-event null is a failure to reject, not a proof. The artifact was nevertheless called
county_attributable_identified.parquet, with a JSON key "identified" and a column "att_id", which
is the one word the screen refuses to claim. The names are now "screened" throughout.

WHY THERE IS STILL A FALLBACK. The producer, 04_chk3.py, is being edited by someone else, so the
rename lands in the producer separately. Until it does, the readers below accept the
old names and say so once. When 04_chk3.py writes the new names, delete OLD_PARQUET, OLD_ATT_COL,
OLD_JSON_KEY and the three branches that use them; nothing else needs to change.
"""
import os
import pandas as pd

ATTRIB_DIR = "/data/equity_cost/analysis/attrib"

PARQUET = "county_attributable_screened.parquet"
JSON = "attrib_identified.json"          # the file name is 04_chk3.py's to change, not ours
ATT_COL = "att_screened"
JSON_KEY = "screened"

OLD_PARQUET = "county_attributable_identified.parquet"
OLD_ATT_COL = "att_id"
OLD_JSON_KEY = "identified"

_WARNED = set()


def _warn(what, old, new):
    if what in _WARNED:
        return
    _WARNED.add(what)
    print("[attrib_artifacts] reading the pre-rename %s %r; the producer still writes the old name "
          "rather than %r, and both are accepted." % (what, old, new), flush=True)


def screened_parquet_path(base=ATTRIB_DIR):
    """Absolute path to the screen's county artifact, new name preferred."""
    new = os.path.join(base, PARQUET)
    if os.path.exists(new):
        return new
    old = os.path.join(base, OLD_PARQUET)
    if os.path.exists(old):
        _warn("artifact", OLD_PARQUET, PARQUET)
        return old
    raise SystemExit("neither %s nor %s exists under %s; run 04_chk3.py" % (PARQUET, OLD_PARQUET, base))


def read_screened(base=ATTRIB_DIR):
    """The screen's county table, with the attributable column always called att_screened."""
    M = pd.read_parquet(screened_parquet_path(base))
    if ATT_COL not in M.columns:
        if OLD_ATT_COL not in M.columns:
            raise SystemExit("%s has neither %r nor %r; columns are %s"
                             % (screened_parquet_path(base), ATT_COL, OLD_ATT_COL, list(M.columns)))
        _warn("column", OLD_ATT_COL, ATT_COL)
        M = M.rename(columns={OLD_ATT_COL: ATT_COL})
    return M


def screened_json(base=ATTRIB_DIR):
    """The screen's summary, with the total always under the key "screened"."""
    import json
    J = json.load(open(os.path.join(base, JSON)))
    if JSON_KEY not in J:
        if OLD_JSON_KEY not in J:
            raise SystemExit("%s has neither %r nor %r; keys are %s"
                             % (JSON, JSON_KEY, OLD_JSON_KEY, sorted(J)))
        _warn("json key", OLD_JSON_KEY, JSON_KEY)
        J[JSON_KEY] = J[OLD_JSON_KEY]
    if "screened_hazards" not in J:
        raise SystemExit("%s carries no screened_hazards list; rerun 04_chk3.py" % JSON)
    return J
