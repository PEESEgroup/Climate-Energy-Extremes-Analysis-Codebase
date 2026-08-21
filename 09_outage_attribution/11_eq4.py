"""Figure 4a and 4e on the attribution of Figure 3b.

The old build valued an event count at a per-hazard effect, with a tercile-specific convective
effect, and that construction is what allowed a split into exposure and vulnerability. The new
attribution carries one coefficient per hazard and intensity bin, applied to each county-day's own
observed outage, so there is no county-specific vulnerability parameter to separate out. Panel (a)
is therefore a concentration ratio alone, and the vulnerability question is answered in panel (b).
"""
import json
import numpy as np, pandas as pd
import sys
import os as _os_rp
for _rp in ("04_demand_model", "07_hazard_calendar", "09_outage_attribution",
            "02_downscale_wind", "12_figures"):
    _ap = _os_rp.path.abspath(_os_rp.path.join(
        _os_rp.path.dirname(_os_rp.path.abspath(__file__)), "..", _rp))
    if _os_rp.path.isdir(_ap) and _ap not in sys.path:
        sys.path.insert(0, _ap)
import attrib_artifacts as AA          # F3.9: the screen's artifact names live in one place
import hazsets                        # F4.2: the screen's hazard SET lives in one place
A = "/data/equity_cost/analysis"
M = AA.read_screened("%s/attrib" % A)
hazsets.check_columns(M, where="11_eq4.py")
E = pd.read_csv("%s/equity_joined_v2.csv" % A, dtype={"fips": str})
GF = pd.read_parquet("/data/equity_cost/gridphys/county_gridform.parquet")[["fips", "ug_share_dom"]]
D = M.merge(E[["fips", "state", "population", "land_km2", "median_income", "poverty_rate",
               "median_age", "minority_pct"]], on="fips", how="left").merge(GF, on="fips", how="left")
D["dens"] = D.population / D.land_km2.replace(0, np.nan)
print("counties %d, attributable %.3e, observed %.3e (%.2f%%)"
      % (len(D), D.att_screened.sum(), D.observed_cho.sum(), 100 * D.att_screened.sum() / D.observed_cho.sum()))

TR = [("median_age", "median age", False), ("poverty_rate", "poverty", False),
      ("minority_pct", "minority share", False), ("median_income", "income", True),
      ("dens", "rurality", True), ("ug_share_dom", "undergrounding", True)]
rows = []
print()
print("%-16s %8s %10s %10s %8s" % ("trait", "counties", "pop share", "burden sh", "ratio"))
for col, nm, low_is_bad in TR:
    v = D[col]
    ok = v.notna() & D.population.notna()
    cut = v[ok].quantile(0.2 if low_is_bad else 0.8)
    sel = ok & ((v <= cut) if low_is_bad else (v >= cut))
    ps = D.population[sel].sum() / D.population[ok].sum()
    bs = D.att_screened[sel].sum() / D.att_screened[ok].sum()
    rows.append(dict(trait=nm, col=col, n=int(sel.sum()), pop_share=float(ps),
                     burden_share=float(bs), ratio=float(bs / ps)))
    print("%-16s %8d %9.1f%% %9.1f%% %8.2f" % (nm, sel.sum(), 100 * ps, 100 * bs, bs / ps))
pd.DataFrame(rows).to_csv("%s/attrib/fig4a_exposure.csv" % A, index=False)

print()
print("state totals for the funding panel")
S = D.groupby("state").agg(att=("att_screened", "sum"), cust=("denom", "sum"),
                           obs=("observed_cho", "sum")).reset_index()
S["att_rate"] = S.att / S.cust
B = pd.read_csv("%s/r4_panelB_states.csv" % A)
print("r4_panelB_states columns:", list(B.columns))
key = "state" if "state" in B.columns else B.columns[0]
J = B.merge(S[["state", "att_rate", "att", "cust"]], left_on=key, right_on="state",
            how="left", suffixes=("_old", ""))
print(J.head(4).to_string())
# 12_eq4b.py is the single writer of fig4e_states.csv, the file Figure 4e reads. This script wrote
# that same name with a narrower set of columns and no observed-outage total, so whichever of the
# two ran last decided whether panel (e) could be drawn. This one writes its own name now.
J.to_csv("%s/attrib/fig4e_states_eq4_diag.csv" % A, index=False)
print("wrote fig4a_exposure.csv and fig4e_states_eq4_diag.csv")
print("NOTE: Figure 4e reads fig4e_states.csv, which 12_eq4b.py writes.")
