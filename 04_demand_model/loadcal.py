"""Apply the validated quantile-mapping calibration to a balancing-authority load series.

WHY THIS FILE EXISTS. 05_calibrate_final.py selects, per balancing authority, among no transform, a
seasonal transform and a monthly transform, fitting on 2016 to 2018 and testing on the held-out
year 2019, and saves the chosen maps to qm_transfer_prod.npz. Production did not read that file.
It read /data/tell_pred/bias_factors.npz, a per-authority-per-month scalar with no producer in
this repository, and multiplied by it. So the calibration the supplementary information describes
and the calibration the results used were two different procedures, and re-running the selection
changed nothing. This module is the missing link.

WHAT THE ARTIFACT HOLDS. Keys are "BA|scheme|nq|stratum|pred" and "...|obs", where scheme is
"se" for seasonal or "mo" for monthly, nq is the number of quantiles, and stratum is the season or
month index. An authority absent from the file takes no transform, which is the selected outcome
for most of them.
"""
import numpy as np

QM_PATH = "/data/tell_pred/qm_transfer_prod.npz"


def load_maps(path=QM_PATH):
    """Return {ba: (scheme, nq, {stratum: (pred_quantiles, obs_quantiles)})}."""
    z = np.load(path)
    out = {}
    for k in z.files:
        ba, sc, nq, st, kind = k.split("|")
        e = out.setdefault(ba, (sc, int(nq), {}))
        e[2].setdefault(int(st), {})[kind] = z[k]
    return {b: (sc, nq, {s: (d["pred"], d["obs"]) for s, d in st.items()})
            for b, (sc, nq, st) in out.items()}


def stratum_of(scheme, month_of_hour):
    """Season index 1 to 4 for 'se', month 1 to 12 for 'mo'."""
    if scheme == "mo":
        return month_of_hour
    return ((month_of_hour % 12) // 3) + 1          # DJF=1, MAM=2, JJA=3, SON=4


def apply_ba(series, month_of_hour, ba, maps):
    """Map one authority's hourly series through its selected transform.

    An authority with no entry is returned unchanged, which is the 'no transform' decision. Values
    outside the fitted quantile range are held at the end points rather than extrapolated, so the
    correction can never invent a tail beyond what 2016 to 2018 supports."""
    if ba not in maps:
        return series
    scheme, _nq, strata = maps[ba]
    out = np.asarray(series, np.float32).copy()
    st = stratum_of(scheme, np.asarray(month_of_hour))
    for s, (pred, obs) in strata.items():
        m = st == s
        if not m.any():
            continue
        o = np.argsort(pred)
        out[m] = np.interp(out[m], np.asarray(pred)[o], np.asarray(obs)[o]).astype(np.float32)
    return out
