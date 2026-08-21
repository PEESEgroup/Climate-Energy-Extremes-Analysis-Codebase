"""The screened hazard SET, read from one place. attrib_artifacts.py owns the artifact NAMES.

WHY THIS FILE EXISTS. 04_chk3.py decides which hazards survive the pre-event placebo screen and
writes that decision to the screen JSON under "screened_hazards". Before this file, five
consumers of that decision each carried a private copy of it: 07_terc.py and its two generated
descendants held IDH = ["tc", "convective"], tercsum.py held the same pair and DERIVED the
figure's Bonferroni bar from its length, 16_fig4data.py typed that bar as the literal 2.865, conc.py
enumerated the remaining hazards by name, and 02_verify4.py read one hazard key by literal. When the
screen moved off two hazards, every copy kept the old answer and nothing failed, so Figure 4
would have been published against a hazard set the estimator no longer used, under a multiplicity
bar counted for a family that no longer existed.

Nothing here decides anything. 04_chk3.py decides; this reports what it decided and fails closed
when a consumer and the screen disagree. Loading goes through attrib_artifacts so the path to the
screen is written down once, not twice.
"""
import attrib_artifacts as _AA

# Display names only. This maps a hazard to English; it never decides which hazards exist, so it
# is not another copy of the set. An unknown key falls through to itself rather than failing.
LABEL = {"tc": "hurricane", "convective": "severe convection", "cold": "cold spell",
         "heat": "heat wave", "fire": "fire weather", "vre_drought": "VRE drought"}


def label(h):
    return LABEL.get(h, h)


def screened(base=_AA.ATTRIB_DIR):
    """Hazards whose pre-event blocks passed 04_chk3.py, in the order 04_chk3.py wrote them."""
    h = list(_AA.screened_json(base).get("screened_hazards") or [])
    if not h:
        raise SystemExit("hazsets: screened_hazards is empty, so nothing is attributable")
    return h


def excluded(base=_AA.ATTRIB_DIR):
    """{hazard: the reason 04_chk3.py excluded it}."""
    return dict(_AA.screened_json(base).get("excluded_hazards") or {})


def all_carried(base=_AA.ATTRIB_DIR):
    """Every hazard the regression carries: screened ones first, then excluded ones.

    Recovered from the same JSON rather than restated, so this file holds no hazard list of its own.
    """
    s = screened(base)
    return s + [h for h in excluded(base) if h not in s]


def att_col(h):
    """Column holding one hazard's attributable total in the screened county table."""
    return "att_%s" % h


def check_columns(df, base=_AA.ATTRIB_DIR, where=""):
    """Fail unless the table's per-hazard columns are exactly the screened set.

    A consumer that reads the attributable total without this check will publish a number built
    from one hazard set while describing another, which is the failure this file exists to stop.
    The totals column itself is not a hazard, under either its old or its new name.
    """
    totals = {_AA.ATT_COL, _AA.OLD_ATT_COL}
    want = set(screened(base))
    have = {c[4:] for c in df.columns if c.startswith("att_") and c not in totals}
    if want != have:
        raise SystemExit(
            "hazsets: %s sees per-hazard columns %s but the screen says %s. The table and the "
            "screen disagree; rerun 04_chk3.py and then this script's producer."
            % (where or "this consumer", sorted(have), sorted(want)))
    return sorted(want)


def bonferroni_z(n_tests, alpha=0.05):
    """Two-sided Bonferroni critical value for a family of n_tests."""
    from scipy import stats as _st
    n = int(n_tests)
    if n < 1:
        raise SystemExit("hazsets: a Bonferroni family of %d tests is not a family" % n)
    return float(_st.norm.ppf(1.0 - alpha / (2.0 * n)))
